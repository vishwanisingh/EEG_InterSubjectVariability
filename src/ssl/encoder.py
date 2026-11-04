"""
src/ssl/encoders.py
-------------------
Three encoder architectures for SSL pretraining.

All encoders share the same interface:
    input  : (batch, n_chans, n_times)  float32
    output : (batch, embed_dim)         float32

Encoders
--------
EEGTCNetEncoder   — EEGNet spatial stem + dilated TCN. 0.006M params.
                    Fast, validated in supervised experiments.

EEGConformerEncoder — Conv stem + Transformer. 0.4M params.
                    Best supervised result, global attention.

CustomEncoder     — Spatial conv + temporal conv + TCN. ~50K params.
                    Designed for SSL: between EEGTCNet and Conformer in
                    capacity, fast enough for 1M+ window pretraining.

Usage
-----
    from src.ssl.encoders import create_encoder

    encoder = create_encoder("tcnet",    n_chans=129, n_times=200, embed_dim=256)
    encoder = create_encoder("conformer", n_chans=129, n_times=200, embed_dim=256)
    encoder = create_encoder("custom",   n_chans=129, n_times=200, embed_dim=256)

    x = torch.randn(32, 129, 200)
    z = encoder(x)   # (32, 256)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Encoder 1: EEGTCNet
# ─────────────────────────────────────────────────────────────────────────────

class _DepthwiseConv2d(nn.Module):
    def __init__(self, in_ch, depth_mul, kernel):
        super().__init__()
        out_ch = in_ch * depth_mul
        self.conv = nn.Conv2d(in_ch, out_ch, kernel,
                              groups=in_ch, bias=False, padding=(0, kernel[1]//2))
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)))

        # Temporal stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

class _SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        pad = (0, kernel[1]//2)
        self.dw = nn.Conv2d(in_ch, in_ch,  kernel, groups=in_ch, bias=False, padding=pad)
        self.pw = nn.Conv2d(in_ch, out_ch, (1,1),  bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.elu(self.bn(self.pw(self.dw(x))))


class _TCNBlock(nn.Module):
    """One dilated causal TCN block with residual connection."""
    def __init__(self, n_ch, kernel=4, dilation=1, dropout=0.3):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(n_ch, n_ch, kernel, dilation=dilation,
                               padding=pad, bias=False)
        self.conv2 = nn.Conv1d(n_ch, n_ch, kernel, dilation=dilation,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(n_ch)
        self.bn2   = nn.BatchNorm1d(n_ch)
        self.drop  = nn.Dropout(dropout)
        self._pad  = pad

    def forward(self, x):
        # Causal: remove future leakage by cropping right side
        h = F.elu(self.bn1(self.conv1(x)[..., :-self._pad]))
        h = self.drop(h)
        h = F.elu(self.bn2(self.conv2(h)[..., :-self._pad]))
        h = self.drop(h)
        return F.elu(h + x)


class EEGTCNetEncoder(nn.Module):
    """
    EEGNet spatial stem → dilated TCN temporal backbone → embedding.

    Architecture matches the EEGTCNet baseline used in supervised experiments.
    Validated at R²=0.208 on CCD. 0.006M params.

    Parameters
    ----------
    n_chans   : number of EEG channels (129 for HBN)
    n_times   : samples per window (200 at 100 Hz)
    embed_dim : output embedding dimension
    f1        : number of temporal filters in EEGNet stem
    d         : depth multiplier for depthwise conv
    tcn_filters : filters in each TCN block
    tcn_layers  : number of TCN dilation layers
    dropout   : dropout rate
    """
    def __init__(
        self,
        n_chans:     int = 129,
        n_times:     int = 200,
        embed_dim:   int = 256,
        f1:          int = 8,
        d:           int = 2,
        tcn_filters: int = 32,
        tcn_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        f2 = f1 * d

        # ── EEGNet stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            # Temporal conv — learns frequency filters
            nn.Conv2d(1, f1, (1, 64), bias=False, padding=(0, 32)),
            nn.BatchNorm2d(f1),
            # Spatial (depthwise) conv — learns spatial filters per freq
            nn.Conv2d(f1, f2, (n_chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            # Separable conv — mixes filters
            _SeparableConv2d(f2, f2, (1, 16)),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute TCN input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            tcn_in_t = self.stem(dummy).shape[-1]

        # ── TCN backbone ──────────────────────────────────────────────────────
        self.proj_in = nn.Conv1d(f2, tcn_filters, 1)
        self.tcn     = nn.Sequential(*[
            _TCNBlock(tcn_filters, kernel=4, dilation=2**i, dropout=dropout)
            for i in range(tcn_layers)
        ])

        # ── Embedding head ────────────────────────────────────────────────────
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_filters, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.stem(x.unsqueeze(1))          # (B, F2, 1, T')
        h = h.squeeze(2)                        # (B, F2, T')
        h = self.proj_in(h)                     # (B, tcn_filters, T')
        h = self.tcn(h)                         # (B, tcn_filters, T')
        return self.embed(h)                    # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder 2: EEGConformer
# ─────────────────────────────────────────────────────────────────────────────

class _ConvStem(nn.Module):
    """Local temporal feature extractor before the Transformer."""
    def __init__(self, n_chans, n_filters=40, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            # Temporal conv
            nn.Conv2d(1, n_filters, (1, 25), bias=False, padding=(0, 12)),
            nn.BatchNorm2d(n_filters),
            # Spatial (depthwise)
            nn.Conv2d(n_filters, n_filters, (n_chans, 1),
                      groups=n_filters, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)   # (B, n_filters, 1, T')


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(h))


class _FFN(nn.Module):
    def __init__(self, d_model, dim_ff=128, dropout=0.5):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.net(x)))


class _ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.5):
        super().__init__()
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn  = _FFN(d_model, dim_ff, dropout)

    def forward(self, x):
        return self.ffn(self.attn(x))


class EEGConformerEncoder(nn.Module):
    """
    Conv stem + Transformer encoder → embedding.

    Captures both local temporal structure (conv) and global
    long-range dependencies (self-attention). 0.4M params.

    Parameters
    ----------
    n_chans     : EEG channels
    n_times     : samples per window
    embed_dim   : output embedding dimension
    n_filters   : conv stem filters
    n_heads     : transformer attention heads
    n_layers    : transformer encoder layers
    dim_ff      : transformer FFN hidden dim
    dropout     : dropout rate
    """
    def __init__(
        self,
        n_chans:   int = 129,
        n_times:   int = 200,
        embed_dim: int = 256,
        n_filters: int = 40,
        n_heads:   int = 8,
        n_layers:  int = 6,
        dim_ff:    int = 128,
        dropout:   float = 0.5,
    ):
        super().__init__()

        self.stem = _ConvStem(n_chans, n_filters, dropout)

        # Compute sequence length after stem
        with torch.no_grad():
            dummy   = torch.zeros(1, 1, n_chans, n_times)
            stem_out = self.stem(dummy)           # (1, n_filters, 1, T')
            seq_len  = stem_out.shape[-1]

        d_model = n_filters
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.Sequential(*[
            _ConformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.embed = nn.Sequential(
            nn.Flatten(1),                       # mean pool over time
            nn.Linear(d_model * seq_len, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Recompute with actual seq_len for the linear layer
        self._seq_len = seq_len
        self._d_model = d_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.stem(x.unsqueeze(1))            # (B, n_filters, 1, T')
        h = h.squeeze(2).permute(0, 2, 1)        # (B, T', n_filters)
        h = h + self.pos_embed                   # add positional embedding
        h = self.transformer(h)                  # (B, T', d_model)
        h = h.reshape(h.size(0), -1)             # (B, T'*d_model)
        return self.embed(h)                     # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder 3: Custom (designed for SSL at scale)
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock1d(nn.Module):
    """1D residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, stride=stride,
                               padding=pad, bias=False)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class CustomEncoder(nn.Module):
    """
    Lightweight encoder designed for SSL pretraining at scale.

    Architecture:
        Spatial filter    — cross-channel linear combination (129 → 32)
        Temporal blocks   — 3 residual 1D conv blocks with downsampling
        TCN               — 2 dilated blocks for long-range context
        Projection        — global avg pool → linear → LayerNorm

    ~50K parameters. Faster than Conformer (no attention), more expressive
    than EEGTCNet (residual blocks + more filters).

    Good for SSL because:
    - Residual blocks prevent vanishing gradients over many pretraining epochs
    - GELU activations (smoother than ELU for contrastive losses)
    - No causal constraint — sees full 2s window bidirectionally
    - LayerNorm output stabilizes VICReg variance term

    Parameters
    ----------
    n_chans   : EEG channels
    n_times   : samples per window
    embed_dim : output embedding dimension
    dropout   : dropout rate
    """
    def __init__(
        self,
        n_chans:   int = 129,
        n_times:   int = 200,
        embed_dim: int = 256,
        dropout:   float = 0.2,
    ):
        super().__init__()

        # ── Spatial filter: (B, C, T) → (B, 32, T) ───────────────────────────
        # Learns cross-channel weighting — equivalent to spatial filtering
        # in BCI (like CSP or spatial ICA)
        self.spatial = nn.Sequential(
            nn.Conv1d(n_chans, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # ── Temporal backbone: residual blocks with downsampling ──────────────
        # Stride=2 at each block: 200 → 100 → 50 → 25
        self.temporal = nn.Sequential(
            _ResBlock1d(32,  64,  kernel=7, stride=2, dropout=dropout),  # T/2
            _ResBlock1d(64,  128, kernel=5, stride=2, dropout=dropout),  # T/4
            _ResBlock1d(128, 128, kernel=3, stride=2, dropout=dropout),  # T/8
        )

        # ── TCN: long-range context without downsampling ──────────────────────
        self.tcn = nn.Sequential(
            _TCNBlock(128, kernel=4, dilation=1, dropout=dropout),
            _TCNBlock(128, kernel=4, dilation=2, dropout=dropout),
        )

        # ── Embedding head ────────────────────────────────────────────────────
        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.spatial(x)    # (B, 32, T)
        h = self.temporal(h)   # (B, 128, T/8)
        h = self.tcn(h)        # (B, 128, T/8)
        return self.embed(h)   # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "tcnet":     EEGTCNetEncoder,
    "conformer": EEGConformerEncoder,
    "custom":    CustomEncoder,
}


def create_encoder(
    name:      str,
    n_chans:   int = 129,
    n_times:   int = 200,
    embed_dim: int = 256,
    **kwargs,
) -> nn.Module:
    """
    Instantiate an encoder by name.

    Parameters
    ----------
    name      : "tcnet" | "conformer" | "custom"
    n_chans   : EEG channels (129 for HBN)
    n_times   : samples per window (200 = 2s @ 100Hz)
    embed_dim : output embedding size
    **kwargs  : passed to encoder constructor

    Returns
    -------
    nn.Module with .embed_dim attribute and forward(x) → (B, embed_dim)
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available: {list(ENCODER_REGISTRY)}"
        )
    encoder = ENCODER_REGISTRY[name](
        n_chans=n_chans, n_times=n_times, embed_dim=embed_dim, **kwargs
    )

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder '{name}': {n_params:,} params  "
          f"({n_params/1e6:.3f}M)  embed_dim={embed_dim}")

    # Sanity check forward pass
    with torch.no_grad():
        dummy = torch.zeros(2, n_chans, n_times)
        out   = encoder(dummy)
        assert out.shape == (2, embed_dim), \
            f"Encoder output shape mismatch: expected (2, {embed_dim}), got {out.shape}"
    print(f"  Forward check passed: (2, {n_chans}, {n_times}) → (2, {embed_dim})")

    return encoder
