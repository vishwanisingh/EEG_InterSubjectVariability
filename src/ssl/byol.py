"""
ssl_byol.py
-----------
BYOL (Bootstrap Your Own Latent) for EEG SSL.

Reference: Grill et al. 2020 — https://arxiv.org/abs/2006.07733

BYOL vs VICReg for EEG:
    VICReg: explicit variance/covariance terms prevent collapse
            → more stable, less sensitive to batch size
            → recommended for smaller batches (<256)

    BYOL:   online/target network asymmetry prevents collapse
            → needs careful momentum tuning (default 0.996)
            → slightly better empirically on large datasets
            → recommended when batch size >= 256

Use BYOL when you have large batch sizes on HPC.
Use VICReg when memory-constrained or batch size < 256.

Architecture:
    Online network  : encoder + projector + predictor
    Target network  : encoder + projector (EMA of online, no predictor)

    Loss: MSE between online predictor output and target projector output
    (stop-gradient on target — key to preventing collapse)

Usage
-----
    from ssl_encoders import create_encoder
    from ssl_byol import BYOL, EEGAugmentation

    encoder = create_encoder("custom", n_chans=129, n_times=200, embed_dim=256)
    aug     = EEGAugmentation(n_chans=129, n_times=200)
    model   = BYOL(encoder, aug, proj_dim=256, momentum=0.996)

    x = torch.randn(32, 129, 200)
    loss, log = model(x)
    loss.backward()
    model.update_target()   # call after each optimizer step
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_vicreg import EEGAugmentation


# ─────────────────────────────────────────────────────────────────────────────
# MLP heads
# ─────────────────────────────────────────────────────────────────────────────

class BYOLProjector(nn.Module):
    """
    BYOL projection head: embed_dim → proj_dim.
    2-layer MLP with BatchNorm (no activation on last layer).
    """
    def __init__(self, embed_dim: int, proj_dim: int, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),  # no affine on last BN
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLPredictor(nn.Module):
    """
    BYOL prediction head: proj_dim → proj_dim.
    Only in online network — asymmetry is key to BYOL's collapse prevention.
    """
    def __init__(self, proj_dim: int, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BYOL model
# ─────────────────────────────────────────────────────────────────────────────

class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent.

    Online network:  encoder + projector + predictor  (trained by gradient)
    Target network:  encoder + projector               (EMA of online)

    Parameters
    ----------
    encoder    : any encoder from ssl_encoders.create_encoder()
    aug        : EEGAugmentation instance
    proj_dim   : projector output dimension (default 256)
    hidden_dim : MLP hidden dimension (default 4096)
    momentum   : EMA momentum for target network update (default 0.996)
                 Higher = slower target update = more stable but slower
    """
    def __init__(
        self,
        encoder,
        aug:        EEGAugmentation,
        proj_dim:   int   = 256,
        hidden_dim: int   = 4096,
        momentum:   float = 0.996,
    ):
        super().__init__()
        self.aug      = aug
        self.momentum = momentum
        embed_dim     = encoder.embed_dim

        # ── Online network ────────────────────────────────────────────────────
        self.online_encoder   = encoder
        self.online_projector = BYOLProjector(embed_dim, proj_dim, hidden_dim)
        self.online_predictor = BYOLPredictor(proj_dim,  hidden_dim)

        # ── Target network (EMA copy — no gradients) ──────────────────────────
        self.target_encoder   = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        """
        EMA update of target network.
        Call after every optimizer.step().
        """
        for online, target in [
            (self.online_encoder,   self.target_encoder),
            (self.online_projector, self.target_projector),
        ]:
            for p_online, p_target in zip(online.parameters(),
                                          target.parameters()):
                p_target.data = (
                    self.momentum * p_target.data
                    + (1 - self.momentum) * p_online.data
                )

    def _online_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Online: encode → project → predict."""
        return self.online_predictor(
            self.online_projector(self.online_encoder(x))
        )

    @torch.no_grad()
    def _target_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Target: encode → project (no predict, stop gradient)."""
        return self.target_projector(self.target_encoder(x))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (B, C, T) EEG batch

        Returns
        -------
        loss : scalar — symmetric BYOL loss
        log  : dict with loss value
        """
        v1, v2 = self.aug(x)

        # Online predictions
        p1 = self._online_forward(v1)   # (B, proj_dim)
        p2 = self._online_forward(v2)

        # Target projections (stop gradient)
        z1 = self._target_forward(v1)   # (B, proj_dim)
        z2 = self._target_forward(v2)

        # Symmetric loss: each online prediction vs the other's target
        loss = (
            byol_loss(p1, z2.detach()) +
            byol_loss(p2, z1.detach())
        ) / 2

        log = {"loss": loss.item()}
        return loss, log

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without augmentation — used at fine-tune/eval time."""
        return self.online_encoder(x)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    BYOL regression loss: negative cosine similarity.
    Normalized L2 distance in embedding space.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()