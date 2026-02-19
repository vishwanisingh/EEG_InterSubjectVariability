"""
src/ssl/finetune.py
-------------------
Fine-tuning wrapper that attaches a regression head to a pretrained encoder.

Supports three fine-tuning modes:
  - 'full'   : unfreeze the entire encoder + train head (best performance)
  - 'linear' : freeze encoder, train only the linear head (fastest, tests
               representation quality in isolation)
  - 'partial': freeze early layers, unfreeze later layers + head (compromise)

Also includes a supervised-from-scratch baseline that uses the same
EEGEncoder architecture but trains end-to-end with labels from random init.
This is the most important ablation — it isolates the SSL pretraining benefit.

Usage:
    # Load a pretrained VICReg / BYOL / MAE model
    ssl_model = VICReg(encoder, ...)
    # ... run pretraining ...
    encoder = ssl_model.get_encoder()

    # Wrap for fine-tuning
    model = FinetunedEEGModel(encoder, finetune_mode='full')
    # ... train with RT labels ...
"""

import torch
import torch.nn as nn
from typing import Literal

from .encoder import EEGEncoder
from .mae import EEGMaskedEncoder


class FinetunedEEGModel(nn.Module):
    """
    Fine-tuning wrapper: pretrained encoder + regression head.

    Parameters
    ----------
    encoder : EEGEncoder or EEGMaskedEncoder
        Pretrained encoder from VICReg / BYOL / MAE.
    n_outputs : int
        Output dimension (1 for RT regression).
    finetune_mode : str
        One of 'full', 'linear', 'partial'.
        - 'full'   : entire model trained end-to-end
        - 'linear' : encoder frozen, only head trained
        - 'partial': first half of encoder frozen, second half + head trained
    head_hidden : int, optional
        If > 0, use a 2-layer MLP head instead of a linear head.
    dropout : float
        Dropout before the final regression layer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_outputs: int = 1,
        finetune_mode: Literal["full", "linear", "partial"] = "full",
        head_hidden: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder      = encoder
        self.finetune_mode = finetune_mode

        # Determine embed dimension
        embed_dim = getattr(encoder, "embed_dim", None)
        if embed_dim is None:
            raise AttributeError("Encoder must have an embed_dim attribute.")

        # ── Regression head ───────────────────────────────────────────────────
        self.drop = nn.Dropout(dropout)
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, n_outputs),
            )
        else:
            self.head = nn.Linear(embed_dim, n_outputs)

        # ── Apply freezing strategy ───────────────────────────────────────────
        self._apply_finetune_mode()

        self._init_head()

    def _apply_finetune_mode(self):
        if self.finetune_mode == "linear":
            # Freeze entire encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[FinetunedEEGModel] Mode: linear — encoder frozen")

        elif self.finetune_mode == "partial":
            # Freeze first half, unfreeze second half
            params = list(self.encoder.parameters())
            n_freeze = len(params) // 2
            for i, param in enumerate(params):
                param.requires_grad = i >= n_freeze
            n_trainable = sum(p.requires_grad for p in self.encoder.parameters())
            print(f"[FinetunedEEGModel] Mode: partial — "
                  f"{n_trainable}/{len(params)} encoder params trainable")

        else:  # 'full'
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("[FinetunedEEGModel] Mode: full — entire model trainable")

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        (B, n_outputs)
        """
        z = self.encoder(x)        # (B, embed_dim)
        z = self.drop(z)
        return self.head(z)        # (B, n_outputs)

    def trainable_parameters(self):
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def n_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SupervisedEEGModel(nn.Module):
    """
    Supervised from-scratch baseline using the same EEGEncoder architecture.

    This is the CRITICAL ablation for your SSL experiments:
      SupervisedEEGModel  = same architecture, random init, trained with labels
      FinetunedEEGModel   = same architecture, SSL pretrained, fine-tuned

    If FinetunedEEGModel doesn't beat SupervisedEEGModel, your SSL pretraining
    adds no value (the architecture itself is what matters, not the pretraining).

    Parameters
    ----------
    n_chans : int
    n_outputs : int
    n_times : int
    sfreq : int
        Unused, kept for registry compatibility.
    embed_dim : int
        Encoder embedding dimension.
    head_hidden : int
        If > 0, use MLP head. If 0, use linear head.
    dropout : float
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        embed_dim: int = 256,
        head_hidden: int = 128,
        dropout: float = 0.1,
        n_spatial: int = 32,
        base_filters: int = 64,
    ):
        super().__init__()
        self.encoder = EEGEncoder(
            n_chans=n_chans,
            n_times=n_times,
            n_spatial=n_spatial,
            base_filters=base_filters,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, n_outputs),
            )
        else:
            self.head = nn.Linear(embed_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.encoder(x)))
