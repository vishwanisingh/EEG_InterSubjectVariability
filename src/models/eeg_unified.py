"""Custom EEG backbone with optional spatial filtering, MoE head, and multi-scale kernels."""

import torch
from torch import nn
from torchaudio.transforms import Resample
from braindecode.models.base import EEGModuleMixin

from .mlp import MLP
from .moe import SimpleMoE, MoE, MoEConfig
from .multi_temporal_kernels import MultiKernelConv1d, temporal_kernels


class EEGUnifiedModel(EEGModuleMixin, nn.Module):
    """
    Flexible EEG regression backbone.

    Supports:
    - Optional spatial filtering (Conv2d across channels)
    - Optional multi-scale temporal kernels (one per EEG frequency band)
    - Configurable final head: Linear | MLP | SimpleMoE | sparse MoE
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        sfreq: int = 100,
        n_times: int = 200,
        # Architecture
        n_convs: int = 2,
        resampling: int = 80,
        kernel_size: int = 8,
        feature_maps: int = 128,
        num_spatial_filters: int = 32,   # 0 → disable spatial filtering
        use_multi_temporal_kernels: bool = False,
        # Final head
        use_mlp: bool = False,
        use_simple_moe: bool = False,
        use_aux_moe: bool = False,
        hidden_dim: int = 540,
        num_experts: int = 8,
        dropout: float = 0.1,
        activation: type = nn.ReLU,
        # Misc
        chs_info=None,
        input_window_seconds=None,
        debug: bool = False,
    ):
        self.num_spatial_filters = int(num_spatial_filters)

        if self.num_spatial_filters > 0:
            super().__init__(
                n_outputs=n_outputs,
                n_chans=n_chans,
                chs_info=chs_info,
                n_times=n_times,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            )
            del chs_info, n_times, input_window_seconds

            self.resample = (
                Resample(orig_freq=int(self.sfreq), new_freq=int(resampling))
                if self.sfreq != resampling
                else nn.Identity()
            )

            assert self.n_chans is not None, "n_chans must be provided"
            self.spatial_filter = nn.Conv2d(
                in_channels=1,
                out_channels=self.num_spatial_filters,
                kernel_size=(self.n_chans, 1),
                bias=False,
            )
            conv_in_channels = self.num_spatial_filters
        else:
            super(EEGUnifiedModel, self).__init__()
            self.rs = Resample(orig_freq=sfreq, new_freq=resampling)
            conv_in_channels = n_chans

        if debug:
            print(
                f"[EEGUnifiedModel] spatial({self.num_spatial_filters}), "
                f"mlp({use_mlp}), simple_moe({use_simple_moe}), aux_moe({use_aux_moe}), "
                f"multi_temporal({use_multi_temporal_kernels}), "
                f"hidden({hidden_dim}), experts({num_experts}), fmaps({feature_maps})"
            )

        # ── Temporal convolution stack ────────────────────────────────────────
        if use_multi_temporal_kernels:
            self.conv = MultiKernelConv1d(conv_in_channels, feature_maps, kernel_size=temporal_kernels)
            self.blocks = nn.ModuleList()
            old_channels = feature_maps * len(temporal_kernels)
            for i in range(n_convs):
                new_channels = int(feature_maps * (1.414 ** i))
                self.blocks.append(nn.Sequential(
                    MultiKernelConv1d(old_channels, new_channels, kernel_size=temporal_kernels),
                    nn.MaxPool1d(2) if i > 0 else nn.MaxPool1d(1),
                    activation(),
                    MultiKernelConv1d(new_channels * len(temporal_kernels), new_channels, kernel_size=temporal_kernels),
                    activation(),
                ))
                old_channels = new_channels * len(temporal_kernels)
        else:
            self.conv = nn.Conv1d(conv_in_channels, feature_maps, kernel_size=kernel_size,
                                  padding=kernel_size // 2, bias=False)
            self.bn   = nn.BatchNorm1d(feature_maps)
            self.blocks = nn.ModuleList()
            old_channels = feature_maps
            new_channels = feature_maps
            for i in range(n_convs):
                if i > 0:
                    new_channels = int(1.414 * new_channels)
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(old_channels, new_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(new_channels),
                    nn.MaxPool1d(2),
                    activation(),
                    nn.Conv1d(new_channels, new_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False),
                    nn.BatchNorm1d(new_channels),
                    activation(),
                ))
                old_channels = new_channels

        # ── Final head ────────────────────────────────────────────────────────
        if use_mlp:
            self.final_layer = MLP(old_channels, hidden_dim, n_outputs, dropout=dropout)
        elif use_simple_moe:
            self.final_layer = SimpleMoE(old_channels, num_experts, hidden_dim, n_outputs, dropout=dropout)
        elif use_aux_moe:
            cfg = MoEConfig(
                d_in=old_channels, d_hidden=hidden_dim, d_out=n_outputs,
                num_experts=num_experts, top_k=2, router_temp=1.5,
                router_noise_std=0.05, capacity_factor=1.25,
                expert_dropout=0.05, lb_loss_weight=1e-2,
                entropy_bonus=1e-3, z_loss_weight=1e-4,
            )
            self.final_layer = MoE(config=cfg)
        else:
            self.final_layer = nn.Linear(old_channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, n_chans, n_times).
        """
        if self.num_spatial_filters > 0:
            x = self.resample(x.contiguous())           # (B, C, T')
            x = self.spatial_filter(x.unsqueeze(1))     # (B, F, 1, T')
            x = x.squeeze(2)                            # (B, F, T')
            y = torch.relu(self.bn(self.conv(x))) if hasattr(self, "bn") else self.conv(x)
        else:
            x = self.rs(x.contiguous())
            y = torch.relu(self.bn(self.conv(x)))

        for block in self.blocks:
            y = block(y)

        y = y.mean(dim=2)   # global average pool over time
        return self.final_layer(y)