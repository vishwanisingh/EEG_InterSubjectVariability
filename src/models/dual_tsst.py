"""
Dual-TSST: Dual-branch Temporal-Spatial-Spectral Transformer
-------------------------------------------------------------
Two parallel branches process EEG from different viewpoints and are fused:
  - Temporal branch : captures local & global temporal dynamics via a TCN
                      followed by a Transformer.
  - Spectral branch : computes a learnable frequency-band decomposition via
                      multi-scale 1-D convolutions, then applies a Transformer.

Both branch outputs are concatenated and passed through a fusion MLP.

Input  : (B, n_chans, n_times)
Output : (B, n_outputs)

Reference: inspired by "Dual-branch Spatial-Temporal Graph Network for
           EEG-based Emotion Recognition" and similar dual-path EEG works.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable 1-D convolution block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch,  kernel, padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act= nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.pw(self.dw(x)))))


class _TransformerBlock(nn.Module):
    """Single pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Temporal branch
# ─────────────────────────────────────────────────────────────────────────────

class _TemporalBranch(nn.Module):
    """TCN encoder → mean pool → Transformer."""

    def __init__(self, n_chans: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        # Spatial compression: (B, C, T) → (B, d_model, T)
        self.spatial = nn.Sequential(
            nn.Conv1d(n_chans, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
        )
        # Temporal convolutions at three scales
        self.tcn = nn.ModuleList([
            _DepthwiseSeparableConv(d_model, d_model, k, dropout)
            for k in [3, 7, 15]
        ])
        self.proj = nn.Conv1d(d_model * 3, d_model, 1, bias=False)

        # Transformer over time tokens
        self.transformer = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.spatial(x)                         # (B, D, T)
        tcn_outs = [conv(x) for conv in self.tcn]
        x = self.proj(torch.cat(tcn_outs, dim=1))   # (B, D, T)
        x = x.permute(0, 2, 1)                      # (B, T, D)
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)                         # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral branch
# ─────────────────────────────────────────────────────────────────────────────

class _SpectralBranch(nn.Module):
    """Multi-scale frequency decomposition → Transformer over band tokens."""

    # Kernel sizes approximate EEG frequency bands at 100 Hz
    BAND_KERNELS = [36, 13, 8, 4, 2]   # delta, theta, alpha, beta, gamma

    def __init__(self, n_chans: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        n_bands = len(self.BAND_KERNELS)

        # One conv per frequency band: (B, C, T) → (B, d_model, T) × n_bands
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_chans, d_model, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(d_model), nn.ELU(), nn.Dropout(dropout),
            )
            for k in self.BAND_KERNELS
        ])

        # Pool each band to a single token, then project
        self.band_proj = nn.Linear(d_model, d_model)

        # Transformer over the n_bands tokens
        self.pos_embed  = nn.Parameter(torch.zeros(1, n_bands, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.ModuleList([
            _TransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        band_tokens = []
        for conv in self.band_convs:
            feat = conv(x)                    # (B, D, T)
            # Truncate to minimum length (padding may add 1 sample)
            band_tokens.append(feat.mean(dim=-1))   # (B, D) — global avg pool

        x = torch.stack(band_tokens, dim=1)  # (B, n_bands, D)
        x = self.band_proj(x) + self.pos_embed

        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)                 # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Full Dual-TSST model
# ─────────────────────────────────────────────────────────────────────────────

class DualTSST(nn.Module):
    """
    Dual-branch Temporal-Spectral-Spatial Transformer for EEG regression.

    Parameters
    ----------
    n_chans : int
        Number of EEG input channels.
    n_outputs : int
        Regression output dimension.
    n_times : int
        Time samples per window (unused directly, kept for registry compat.).
    sfreq : int
        Sampling frequency (unused, kept for registry compatibility).
    d_model : int
        Hidden dimension for both branches.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Transformer layers in each branch.
    dropout : float
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.temporal_branch = _TemporalBranch(n_chans, d_model, n_heads, n_layers, dropout)
        self.spectral_branch = _SpectralBranch(n_chans, d_model, n_heads, n_layers, dropout)

        # Fusion: concatenate both branch outputs
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)
        """
        t_feat = self.temporal_branch(x)          # (B, d_model)
        s_feat = self.spectral_branch(x)           # (B, d_model)
        fused  = torch.cat([t_feat, s_feat], dim=-1)  # (B, d_model * 2)
        return self.fusion(fused)                  # (B, n_outputs)