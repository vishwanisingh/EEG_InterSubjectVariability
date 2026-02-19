"""Multi-scale temporal convolution for EEG frequency bands.

Kernel sizes are chosen to approximate the standard EEG frequency bands
at 100 Hz sampling rate:

    Band   | Centre freq | Approx kernel
    -------|-------------|---------------
    Delta  | 2.25 Hz     | 36 samples
    Theta  | 6 Hz        | 13 samples
    Alpha  | 10.5 Hz     | 8 samples
    Beta   | 21.5 Hz     | 4 samples
    Gamma  | 55 Hz       | 2 samples
"""

import torch
from torch import nn

temporal_kernels = [2, 4, 8, 13, 36]


class MultiKernelConv1d(nn.Module):
    """Parallel 1-D convolutions with multiple kernel sizes, concatenated along channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list = temporal_kernels,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_size
        ])
        self.bn         = nn.BatchNorm1d(out_channels * len(kernel_size))
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs    = [conv(x) for conv in self.convs]
        min_len = min(o.shape[2] for o in outs)
        outs    = [o[:, :, :min_len] for o in outs]
        x_cat   = torch.cat(outs, dim=1)
        return self.activation(self.bn(x_cat))