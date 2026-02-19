"""
src/ssl/vicreg.py
-----------------
VICReg: Variance-Invariance-Covariance Regularization

Three loss terms that together prevent representation collapse without
needing negative pairs or a momentum encoder:

  Invariance  : embeddings from two views of the same signal should be similar
  Variance    : keep the std of each embedding dimension above a threshold
                (prevents all embeddings collapsing to the same point)
  Covariance  : decorrelate different dimensions of the embedding
                (prevents redundant features)

Paper: Bardes et al., 2022 — https://arxiv.org/abs/2105.04906

Input  : (B, C, T) EEG windows
Output : scalar loss during pretraining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .encoder import EEGEncoder
from .augmentations import EEGAugmentation, get_default_augmentation


class _Projector(nn.Module):
    """Two-layer MLP expander. Projects embed_dim → proj_dim."""

    def __init__(self, embed_dim: int, proj_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_inv: float = 25.0,
    lambda_var: float = 25.0,
    lambda_cov: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute VICReg loss between two batches of projected embeddings.

    Parameters
    ----------
    z1, z2 : (B, proj_dim)
        Projected embeddings from two augmented views.
    lambda_inv : float
        Weight for the invariance term.
    lambda_var : float
        Weight for the variance term.
    lambda_cov : float
        Weight for the covariance term.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    loss : scalar tensor
    components : dict with 'inv', 'var', 'cov' for logging
    """
    B, D = z1.shape

    # ── Invariance: MSE between the two views ────────────────────────────────
    inv_loss = F.mse_loss(z1, z2)

    # ── Variance: keep std of each dimension above 1 ─────────────────────────
    # Gamma = 1 in the paper — we want std ≥ 1 to prevent collapse
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = (torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))) / 2

    # ── Covariance: off-diagonal elements of the covariance matrix ───────────
    cov1 = (z1.T @ z1) / (B - 1)
    cov2 = (z2.T @ z2) / (B - 1)

    # Zero out the diagonal, keep only off-diagonal terms
    off_diag1 = cov1 - torch.diag(cov1.diag())
    off_diag2 = cov2 - torch.diag(cov2.diag())
    cov_loss = (off_diag1.pow_(2).sum() + off_diag2.pow_(2).sum()) / (2 * D)

    loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

    return loss, {
        "inv":  inv_loss.item(),
        "var":  var_loss.item(),
        "cov":  cov_loss.item(),
        "total": loss.item(),
    }


class VICReg(nn.Module):
    """
    VICReg pretraining model.

    Wraps an EEGEncoder + projector MLP. During pretraining,
    call forward(x) which internally creates two augmented views
    and returns the VICReg loss.

    Parameters
    ----------
    encoder : EEGEncoder
        The shared backbone encoder.
    proj_dim : int
        Output dimension of the projector MLP.
    proj_hidden : int
        Hidden dimension of the projector MLP.
    augmentation : EEGAugmentation, optional
        Augmentation pipeline. Defaults to get_default_augmentation().
    lambda_inv, lambda_var, lambda_cov : float
        VICReg loss weights.
    """

    def __init__(
        self,
        encoder: EEGEncoder,
        proj_dim: int = 512,
        proj_hidden: int = 2048,
        augmentation: EEGAugmentation = None,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
    ):
        super().__init__()
        self.encoder    = encoder
        self.projector  = _Projector(encoder.embed_dim, proj_dim, proj_hidden)
        self.augment    = augmentation or get_default_augmentation()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (B, C, T) — original (non-augmented) EEG windows

        Returns
        -------
        loss : scalar tensor
        log  : dict with loss components for logging
        """
        # Create two independently augmented views
        self.augment.train()
        x1 = self.augment(x)
        x2 = self.augment(x)

        # Encode both views
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        return vicreg_loss(z1, z2, self.lambda_inv, self.lambda_var, self.lambda_cov)

    def get_encoder(self) -> EEGEncoder:
        """Return the pretrained encoder for fine-tuning."""
        return self.encoder
