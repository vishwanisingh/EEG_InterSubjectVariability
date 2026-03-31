"""
src/ssl/encoder.py
------------------
EEG encoder backbones for SSL pretraining (VICReg, BYOL, MAE).

Two encoders are provided, both replacing the braindecode model's final
classification/regression head with a linear projection to embed_dim:

  EEGConformerEncoder — wraps braindecode EEGConformer
  EEGTCNetEncoder     — wraps braindecode EEGTCNet

Both expose:
  - forward(x) → (B, embed_dim)
  - embed_dim  attribute (required by VICReg, BYOL, MAE, FinetunedEEGModel)

Strategy: instantiate the braindecode model with n_outputs=1, then replace
its final Linear layer with nn.Linear(in_features → embed_dim) + LayerNorm.
This preserves all pretrained-friendly architecture choices (spatial filters,
temporal convolutions, attention) while giving us a fixed-size embedding we
can plug into any SSL framework.

Input  : (B, n_chans, n_times)
Output : (B, embed_dim)
"""

import torch
import torch.nn as nn
from braindecode.models import EEGConformer, EEGTCNet


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _find_and_replace_final_linear(
    model: nn.Module,
    embed_dim: int,
) -> int:
    """
    Walk the model, find the last nn.Linear (or LinearWithConstraint) layer,
    replace it with nn.Linear(in_features, embed_dim).

    Returns the in_features of the replaced layer so callers can verify.
    """
    # Collect all Linear layers with their (parent_module, attr_name)
    linear_layers = []
    for name, module in model.named_modules():
        # braindecode uses LinearWithConstraint in some models;
        # it is a subclass of nn.Linear so isinstance catches it
        if isinstance(module, nn.Linear):
            # name is like "model.fc" or "final_layer.0" — split off the leaf
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
                attr = name
            linear_layers.append((parent, attr, module))

    if not linear_layers:
        raise RuntimeError("No nn.Linear found in model — cannot replace head.")

    # Replace the LAST linear layer (the output head)
    parent, attr, last_linear = linear_layers[-1]
    in_features = last_linear.in_features
    setattr(parent, attr, nn.Linear(in_features, embed_dim))
    return in_features


# ─────────────────────────────────────────────────────────────────────────────
# EEGConformer encoder
# ─────────────────────────────────────────────────────────────────────────────

class EEGConformerEncoder(nn.Module):
    """
    EEGConformer backbone with its classification head replaced by a
    linear projection to embed_dim.

    Architecture (from Song et al., 2022):
      1. _PatchEmbedding  — ShallowFBCSPNet conv stem + temporal pooling
      2. _TransformerEncoder — multi-head self-attention over patch tokens
      3. Projection head — Linear(final_fc_length → embed_dim) + LayerNorm

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_times : int
        Time samples per window.
    embed_dim : int
        Output embedding dimension.
    n_filters_time : int
        Temporal filters in the conv stem (default 40).
    pool_time_length : int
        Pooling window in the conv stem (default 75).
    pool_time_stride : int
        Pooling stride in the conv stem (default 15).
    num_layers : int
        Transformer encoder depth (default 6).
    num_heads : int
        Attention heads (default 10).
    drop_prob : float
        Dropout in the conv stem (default 0.5).
    att_drop_prob : float
        Dropout in attention layers (default 0.5).
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        embed_dim: int = 256,
        n_filters_time: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        num_layers: int = 6,
        num_heads: int = 10,
        drop_prob: float = 0.5,
        att_drop_prob: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Build the full braindecode model (n_outputs=1 is a placeholder)
        self.backbone = EEGConformer(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
            n_filters_time=n_filters_time,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            num_layers=num_layers,
            num_heads=num_heads,
            drop_prob=drop_prob,
            att_drop_prob=att_drop_prob,
            final_fc_length="auto",
        )

        # Replace the final linear head with our projection
        in_features = _find_and_replace_final_linear(self.backbone, embed_dim)
        print(f"[EEGConformerEncoder] Replaced head: "
              f"Linear({in_features} → {embed_dim})")

        # Add LayerNorm after the projection for stable SSL training
        self.norm = nn.LayerNorm(embed_dim)

        self._init_projection()

    def _init_projection(self):
        """Xavier-initialise the replacement projection layer."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == self.embed_dim:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        z : (B, embed_dim)
        """
        return self.norm(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# EEGTCNet encoder
# ─────────────────────────────────────────────────────────────────────────────

class EEGTCNetEncoder(nn.Module):
    """
    EEGTCNet backbone with its classification head replaced by a
    linear projection to embed_dim.

    Architecture (from Ingolfsson et al., 2020):
      1. EEGNet stem     — temporal conv + depthwise spatial conv + separable conv
      2. TCN blocks      — dilated causal residual convolutions (depth × 2 blocks)
      3. Global avg pool — collapses time → flat feature vector
      4. Projection head — Linear(filters × depth_multiplier → embed_dim) + LayerNorm

    Default feature size before the head:
      filters (12) × depth_multiplier (2) = 24 channels, after global avg pool → 24-d vector.
    Use larger filters / depth_multiplier if embed_dim >> 24.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_times : int
        Time samples per window.
    embed_dim : int
        Output embedding dimension.
    filter_1 : int
        Temporal filters in the EEGNet stem (default 8).
    kern_length : int
        Temporal kernel length in the stem (default 64).
    depth_multiplier : int
        Depthwise multiplier in the stem (default 2).
    depth : int
        Number of TCN residual blocks (default 2).
    kernel_size : int
        TCN kernel size (default 4).
    filters : int
        TCN filters per block (default 12).
    drop_prob : float
        Dropout rate (default 0.5).
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        embed_dim: int = 256,
        filter_1: int = 8,
        kern_length: int = 64,
        depth_multiplier: int = 2,
        depth: int = 2,
        kernel_size: int = 4,
        filters: int = 12,
        drop_prob: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Build the full braindecode model (n_outputs=1 is a placeholder)
        self.backbone = EEGTCNet(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
            filter_1=filter_1,
            kern_length=kern_length,
            depth_multiplier=depth_multiplier,
            depth=depth,
            kernel_size=kernel_size,
            filters=filters,
            drop_prob=drop_prob,
        )

        # Replace the final linear head with our projection
        in_features = _find_and_replace_final_linear(self.backbone, embed_dim)
        print(f"[EEGTCNetEncoder] Replaced head: "
              f"Linear({in_features} → {embed_dim})")

        # Add LayerNorm after the projection for stable SSL training
        self.norm = nn.LayerNorm(embed_dim)

        self._init_projection()

    def _init_projection(self):
        """Xavier-initialise the replacement projection layer."""
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == self.embed_dim:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        z : (B, embed_dim)
        """
        return self.norm(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

ENCODER_REGISTRY = {
    "conformer": EEGConformerEncoder,
    "tcnet":     EEGTCNetEncoder,
}


def create_encoder(
    name: str,
    n_chans: int = 129,
    n_times: int = 200,
    embed_dim: int = 256,
    **kwargs,
) -> nn.Module:
    """
    Instantiate an SSL encoder by name.

    Parameters
    ----------
    name : str
        One of 'conformer' or 'tcnet'.
    n_chans : int
    n_times : int
    embed_dim : int
    **kwargs
        Extra constructor arguments forwarded to the encoder class.

    Returns
    -------
    nn.Module with .embed_dim attribute and forward(x) → (B, embed_dim)
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available: {list(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[name](
        n_chans=n_chans,
        n_times=n_times,
        embed_dim=embed_dim,
        **kwargs,
    )