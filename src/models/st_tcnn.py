"""
ST-TCNN: Spatial-Temporal Temporal Convolutional Neural Network
---------------------------------------------------------------
A two-stage architecture that explicitly separates spatial and temporal
processing:

  Stage 1 — Spatial filtering:
    Depthwise Conv2d across the channel dimension (like EEGNet's depthwise
    layer) to learn spatial filters over electrode space.

  Stage 2 — Temporal TCN:
    A dilated causal TCN (Temporal Convolutional Network) that captures
    multi-scale temporal dependencies with exponentially growing receptive
    fields. Each TCN block uses:
      - Dilated causal conv → BatchNorm → activation → Dropout
      - Residual connection with 1×1 projection if channels differ

  Head: Global average pooling → Linear regression output.

Input  : (B, n_chans, n_times)
Output : (B, n_outputs)

Reference: Inspired by "Spatial-Temporal Convolutional Neural Network for
           EEG-based Motor Imagery Classification" and dilated TCN designs
           from "An Empirical Evaluation of Generic Convolutional and
           Recurrent Networks for Sequence Modeling" (Bai et al., 2018).
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ─────────────────────────────────────────────────────────────────────────────
# Spatial filtering stage
# ─────────────────────────────────────────────────────────────────────────────

class _SpatialBlock(nn.Module):
    """
    EEG spatial filter: learns a linear combination of channels.

    Depthwise Conv2d with kernel (n_chans, 1) projects C channels → n_spatial
    feature maps while preserving the time axis.
    """

    def __init__(self, n_chans: int, n_spatial: int, dropout: float):
        super().__init__()
        self.spatial_conv = nn.Conv2d(
            1, n_spatial, kernel_size=(n_chans, 1), bias=False
        )
        self.bn  = nn.BatchNorm2d(n_spatial)
        self.act = nn.ELU()
        self.drop= nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.drop(self.act(self.bn(self.spatial_conv(x))))
        # (B, n_spatial, 1, T) → (B, n_spatial, T)
        return x.squeeze(2)


# ─────────────────────────────────────────────────────────────────────────────
# Dilated TCN stage
# ─────────────────────────────────────────────────────────────────────────────

class _CausalConv1d(nn.Module):
    """Dilated causal convolution: future information is not leaked."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=self.padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding to maintain causal property
        return out[:, :, : -self.padding] if self.padding else out


class _TCNBlock(nn.Module):
    """Single dilated TCN residual block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        # Residual projection if channel dims differ
        self.res_proj = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.res_proj(x))


class _DilatedTCN(nn.Module):
    """
    Stack of TCN blocks with exponentially growing dilation:
      dilation = 2^0, 2^1, ..., 2^(n_blocks-1)
    Gives receptive field = kernel × (2^n_blocks - 1).
    """

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        n_blocks: int,
        kernel: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            inc = in_ch if i == 0 else hidden_ch
            blocks.append(_TCNBlock(inc, hidden_ch, kernel, dilation, dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full ST-TCNN model
# ─────────────────────────────────────────────────────────────────────────────

class STTCNN(nn.Module):
    """
    Spatial-Temporal TCN for EEG regression.

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
    n_spatial : int
        Number of spatial filters (output channels of the spatial stage).
    tcn_hidden : int
        Number of filters in each TCN block.
    tcn_blocks : int
        Number of dilated TCN blocks (receptive field ≈ kernel × 2^n_blocks).
    tcn_kernel : int
        Kernel size for each TCN block.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        n_spatial: int = 32,
        tcn_hidden: int = 64,
        tcn_blocks: int = 5,       # receptive field = 3 × (2^5 - 1) = 93 samples ≈ ~1 s
        tcn_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Stage 1: spatial filtering
        self.spatial = _SpatialBlock(n_chans, n_spatial, dropout)

        # Stage 2: dilated TCN
        self.tcn = _DilatedTCN(n_spatial, tcn_hidden, tcn_blocks, tcn_kernel, dropout)

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global average pool over time
            nn.Flatten(),
            nn.Linear(tcn_hidden, n_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)
        """
        x = self.spatial(x)   # (B, n_spatial, T)
        x = self.tcn(x)       # (B, tcn_hidden, T)
        return self.head(x)   # (B, n_outputs)