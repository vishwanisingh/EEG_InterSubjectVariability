"""
src/ssl/encoder.py
------------------
Shared EEG encoder backbone used by all SSL methods (VICReg, BYOL, MAE)
and the supervised fine-tuning wrapper.

Architecture:
  1. Spatial filter  — depthwise Conv2d across channels (like EEGNet layer 1)
  2. Temporal CNN    — 4 residual blocks with BatchNorm + ELU + downsampling
  3. Global avg pool — collapses time dimension → fixed-size embedding

The same encoder is used in both pretraining and fine-tuning, making all
comparisons (supervised vs SSL) directly controlled experiments.

Input  : (B, n_chans, n_times)
Output : (B, embed_dim)
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """1-D residual block: two convolutions with BatchNorm, ELU, and skip."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act  = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class EEGEncoder(nn.Module):
    """
    EEG encoder backbone shared across all SSL methods.

    Parameters
    ----------
    n_chans : int
        Number of input EEG channels.
    n_times : int
        Time samples per window.
    n_spatial : int
        Number of spatial filters (output channels of the spatial stage).
    base_filters : int
        Base number of filters in stage 1 of the temporal CNN.
        Stages double: base → base*2 → base*4 → base*4.
    embed_dim : int
        Final embedding dimension after global average pooling.
    dropout : float
        Dropout probability in residual blocks.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        n_spatial: int = 32,
        base_filters: int = 64,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Stage 0: Spatial filtering ────────────────────────────────────────
        # Learns a linear combination of EEG channels → n_spatial feature maps.
        # This is the same depthwise Conv2d used in EEGNet / STTCNN.
        self.spatial = nn.Sequential(
            nn.Conv2d(1, n_spatial, kernel_size=(n_chans, 1), bias=False),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
            nn.Dropout2d(dropout),
        )

        # ── Stages 1-4: Temporal residual CNN ────────────────────────────────
        f = base_filters
        self.stage1 = nn.Sequential(
            _ResBlock(n_spatial, f, kernel=7, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            nn.AvgPool1d(2),
            _ResBlock(f, f * 2, kernel=5, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            nn.AvgPool1d(2),
            _ResBlock(f * 2, f * 4, kernel=3, dropout=dropout),
        )
        self.stage4 = nn.Sequential(
            nn.AvgPool1d(2),
            _ResBlock(f * 4, f * 4, kernel=3, dropout=dropout),
        )

        cnn_out_dim = f * 4

        # ── Projection to embed_dim ───────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global average pool over time
            nn.Flatten(),
            nn.Linear(cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        z : (B, embed_dim)
        """
        # Spatial stage: (B, C, T) → (B, 1, C, T) → (B, n_spatial, 1, T) → (B, n_spatial, T)
        x = self.spatial(x.unsqueeze(1)).squeeze(2)

        # Temporal stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Pool + project
        return self.proj(x)   # (B, embed_dim)
