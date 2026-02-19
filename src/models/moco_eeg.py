"""
MoCo-EEG
---------
Supervised adaptation of Momentum Contrast (MoCo) for EEG.

In its original form, MoCo is a self-supervised contrastive learning method.
Here we expose the encoder network (a CNN + projection head) as a standalone
supervised regression model. The architecture preserves the key design choices:
  - A ResNet-style temporal CNN backbone for feature extraction
  - A projection MLP head mapping features → output

For actual self-supervised pretraining + fine-tuning, see the ssl_pretrain
utility (out of scope here). This module plugs directly into the supervised
training pipeline.

Input  : (B, n_chans, n_times)
Output : (B, n_outputs)

Reference: "MoCo-EEG: Self-Supervised Representation Learning for EEG-based
            Motor Imagery Classification" and related works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock1D(nn.Module):
    """1-D residual block with two depthwise-separable convolutions."""

    def __init__(self, channels: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class _CNNEncoder(nn.Module):
    """
    Four-stage CNN backbone for EEG feature extraction.

    Stage 1: Spatial compression across channels.
    Stages 2-4: Residual temporal processing with progressive downsampling.
    """

    def __init__(self, n_chans: int, base_filters: int, dropout: float):
        super().__init__()

        # Stage 1: channel → base_filters (spatial compression)
        self.stage1 = nn.Sequential(
            nn.Conv1d(n_chans, base_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ELU(),
        )

        # Stages 2-4: residual blocks + downsampling
        f = base_filters
        self.stage2 = nn.Sequential(_ResBlock1D(f, 7, dropout), nn.AvgPool1d(2))
        self.stage3 = nn.Sequential(
            nn.Conv1d(f, f * 2, 1, bias=False), nn.BatchNorm1d(f * 2), nn.ELU(),
            _ResBlock1D(f * 2, 5, dropout), nn.AvgPool1d(2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(f * 2, f * 4, 1, bias=False), nn.BatchNorm1d(f * 4), nn.ELU(),
            _ResBlock1D(f * 4, 3, dropout), nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = f * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x.squeeze(-1)   # (B, out_dim)


class MoCoEEG(nn.Module):
    """
    MoCo-EEG encoder adapted for supervised EEG regression.

    The CNN backbone extracts rich temporal-spatial features.
    A two-layer MLP projection head maps these to the output space.

    Parameters
    ----------
    n_chans : int
        Number of EEG input channels.
    n_outputs : int
        Output dimension (1 for regression).
    n_times : int
        Time samples per window (unused directly, kept for registry compat.).
    sfreq : int
        Sampling frequency (unused, kept for registry compatibility).
    base_filters : int
        Number of filters in the first CNN stage. Later stages are 2× and 4×.
    proj_dim : int
        Hidden dimension of the projection MLP.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        base_filters: int = 64,
        proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = _CNNEncoder(n_chans, base_filters, dropout)
        feat_dim     = self.encoder.out_dim

        # Two-layer projection MLP (matches MoCo v2 / SimCLR design)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the backbone features (before the projection head)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        (B, n_outputs)
        """
        feat = self.encoder(x)       # (B, feat_dim)
        return self.projector(feat)  # (B, n_outputs)