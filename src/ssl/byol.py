"""
src/ssl/byol.py
---------------
BYOL: Bootstrap Your Own Latent

Two-network setup:
  - Online network  : encoder + projector + predictor (trained with gradients)
  - Target network  : encoder + projector (EMA of online, no gradients)

Online network predicts the target network's representation of a different
augmented view. The EMA prevents representational collapse without negatives.

Paper: Grill et al., 2020 — https://arxiv.org/abs/2006.07733

Key design choices:
  - Target network updated via Exponential Moving Average (EMA), NOT gradients
  - Predictor is a small extra MLP on top of the online projector
  - Stop-gradient on target network is critical — without it the network collapses
  - Symmetric loss: both views predict each other

Input  : (B, C, T) EEG windows
Output : scalar loss during pretraining
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .encoder import EEGEncoder
from .augmentations import EEGAugmentation, get_default_augmentation


class _MLP(nn.Module):
    """Two-layer MLP with BatchNorm and ReLU (used for projector and predictor)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL pretraining model.

    Parameters
    ----------
    encoder : EEGEncoder
        The shared backbone encoder (will be cloned for the target network).
    proj_dim : int
        Output dimension of the projector MLP.
    proj_hidden : int
        Hidden dimension of the projector and predictor MLPs.
    augmentation : EEGAugmentation, optional
        Augmentation pipeline. Defaults to get_default_augmentation().
    ema_decay : float
        EMA decay rate for the target network update.
        Typical values: 0.996 (small datasets) to 0.999 (large datasets).
    """

    def __init__(
        self,
        encoder: EEGEncoder,
        proj_dim: int = 256,
        proj_hidden: int = 1024,
        augmentation: EEGAugmentation = None,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.augment   = augmentation or get_default_augmentation()

        # ── Online network (trained with gradients) ───────────────────────────
        self.online_encoder   = encoder
        self.online_projector = _MLP(encoder.embed_dim, proj_hidden, proj_dim)
        self.predictor        = _MLP(proj_dim, proj_hidden, proj_dim)

        # ── Target network (EMA copy, no gradients) ───────────────────────────
        self.target_encoder   = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target network — it's updated via EMA only
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """
        Update target network via EMA.
        Call this once per training step AFTER optimizer.step().
        """
        tau = self.ema_decay
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data = tau * target.data + (1 - tau) * online.data

        for online, target in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target.data = tau * target.data + (1 - tau) * online.data

    def _online_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Online: encode → project → predict."""
        z = self.online_projector(self.online_encoder(x))
        return self.predictor(z)

    @torch.no_grad()
    def _target_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Target: encode → project (stop gradient)."""
        return self.target_projector(self.target_encoder(x))

    @staticmethod
    def _regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Normalized cosine similarity loss (as in the BYOL paper)."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (B, C, T) — original EEG windows

        Returns
        -------
        loss : scalar tensor
        log  : dict with loss components
        """
        self.augment.train()
        x1 = self.augment(x)
        x2 = self.augment(x)

        # Online predicts target, symmetrically
        p1 = self._online_forward(x1)
        p2 = self._online_forward(x2)

        z1 = self._target_forward(x1)
        z2 = self._target_forward(x2)

        # Symmetric loss: each view predicts the other
        loss = (self._regression_loss(p1, z2) + self._regression_loss(p2, z1)) * 0.5

        return loss, {"total": loss.item()}

    def get_encoder(self) -> EEGEncoder:
        """Return the pretrained online encoder for fine-tuning."""
        return self.online_encoder
