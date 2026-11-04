"""
ssl_vicreg.py
-------------
VICReg (Variance-Invariance-Covariance Regularization) for EEG SSL.

Reference: Bardes et al. 2022 — https://arxiv.org/abs/2105.04906

Why VICReg for EEG:
  - No negative pairs needed → memory efficient
  - Variance term prevents collapse without large batches
  - Works well with small embed_dim (256)
  - More stable than BYOL for EEG (no momentum tuning)
  - Covariance term decorrelates embedding dimensions →
    each dim captures different EEG feature

EEG-specific augmentations (confirmed safe for EEG):
  GaussianNoise       — electrode measurement noise
  ChannelDropout      — bad channel simulation
  TemporalCrop        — random temporal sub-window
  AmplitudeScale      — subject amplitude variation
  BandstopNoise       — powerline interference simulation

Augmentations NOT used (would destroy signal):
  Time reversal       — EEG is not time-symmetric (P300 is directional)
  Channel shuffle     — spatial topology matters
  Heavy freq masking  — destroys the features we want to learn

Usage
-----
    from ssl_encoders import create_encoder
    from ssl_vicreg import VICReg, EEGAugmentation

    encoder = create_encoder("tcnet", n_chans=129, n_times=200, embed_dim=256)
    aug     = EEGAugmentation(n_chans=129, n_times=200)
    model   = VICReg(encoder, aug, proj_dim=512)

    x    = torch.randn(32, 129, 200)
    loss, log = model(x)
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# EEG Augmentations
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNoise(nn.Module):
    """Add Gaussian noise to simulate electrode measurement noise."""
    def __init__(self, std: float = 0.1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class ChannelDropout(nn.Module):
    """
    Randomly zero out entire channels — simulates bad electrodes.
    Applied per-sample, not per-batch.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if not self.training:
            return x
        B, C, T = x.shape
        mask = (torch.rand(B, C, 1, device=x.device) > self.p).float()
        return x * mask


class AmplitudeScale(nn.Module):
    """
    Randomly scale amplitude per sample — simulates subject-level
    amplitude variation and impedance differences.
    """
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        super().__init__()
        self.lo, self.hi = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B = x.shape[0]
        scale = (torch.rand(B, 1, 1, device=x.device)
                 * (self.hi - self.lo) + self.lo)
        return x * scale


class TemporalShift(nn.Module):
    """
    Random circular shift along the time axis.
    Breaks strict temporal alignment while preserving signal statistics.
    Max shift = max_shift samples (default 10 = 100ms at 100Hz).
    """
    def __init__(self, max_shift: int = 10):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class BandstopNoise(nn.Module):
    """
    Add narrow-band noise at a random frequency — simulates powerline
    interference (50/60 Hz) or electrode artifacts.
    Operates in time domain via sinusoidal addition.
    """
    def __init__(self, sfreq: float = 100.0, amplitude: float = 0.05):
        super().__init__()
        self.sfreq     = sfreq
        self.amplitude = amplitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, T = x.shape
        # Random frequency between 5-45 Hz (within signal band)
        freq  = torch.rand(1).item() * 40 + 5
        phase = torch.rand(1).item() * 2 * np.pi
        t     = torch.arange(T, device=x.device).float() / self.sfreq
        noise = (self.amplitude * torch.sin(2 * np.pi * freq * t + phase))
        return x + noise.unsqueeze(0).unsqueeze(0)


class EEGAugmentation(nn.Module):
    """
    Compose EEG-specific augmentations for VICReg/BYOL.

    Creates two views of the same window by applying random augmentations
    independently. Both views should be semantically similar (same neural
    activity) but perceptually different (different noise/scale).

    Parameters
    ----------
    n_chans    : EEG channels
    n_times    : samples per window
    noise_std  : Gaussian noise std (default 0.1)
    ch_dropout : channel dropout probability (default 0.1)
    max_shift  : max temporal shift in samples (default 10 = 100ms)
    scale_range: amplitude scaling range (default 0.8-1.2)
    """
    def __init__(
        self,
        n_chans:     int   = 129,
        n_times:     int   = 200,
        noise_std:   float = 0.1,
        ch_dropout:  float = 0.1,
        max_shift:   int   = 10,
        scale_range: tuple = (0.8, 1.2),
        sfreq:       float = 100.0,
    ):
        super().__init__()
        self.augment = nn.Sequential(
            AmplitudeScale(scale_range),
            GaussianNoise(noise_std),
            ChannelDropout(ch_dropout),
            TemporalShift(max_shift),
            BandstopNoise(sfreq, amplitude=0.05),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of x."""
        self.train()
        v1 = self.augment(x)
        v2 = self.augment(x)
        return v1, v2


# ─────────────────────────────────────────────────────────────────────────────
# Projector head
# ─────────────────────────────────────────────────────────────────────────────

class Projector(nn.Module):
    """
    MLP projection head: embed_dim → proj_dim.
    Used during pretraining only — discarded at fine-tuning.
    3 layers with BatchNorm (no activation on last layer per VICReg paper).
    """
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# VICReg loss
# ─────────────────────────────────────────────────────────────────────────────

def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coef:  float = 25.0,
    var_coef:  float = 25.0,
    cov_coef:  float = 1.0,
    gamma:     float = 1.0,
    eps:       float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    VICReg loss: Invariance + Variance + Covariance.

    Parameters
    ----------
    z1, z2    : projected embeddings (B, D)
    sim_coef  : invariance loss weight (default 25)
    var_coef  : variance loss weight (default 25)
    cov_coef  : covariance loss weight (default 1)
    gamma     : target std for variance term (default 1.0)
    eps       : numerical stability

    Returns
    -------
    loss : scalar tensor
    log  : dict with individual loss components
    """
    B, D = z1.shape

    # ── Invariance: MSE between two views ────────────────────────────────────
    loss_inv = F.mse_loss(z1, z2)

    # ── Variance: push std of each dim above gamma ────────────────────────────
    def variance_loss(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(gamma - std))

    loss_var = variance_loss(z1) + variance_loss(z2)

    # ── Covariance: decorrelate embedding dimensions ──────────────────────────
    def covariance_loss(z):
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1)
        # Off-diagonal elements should be zero
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    loss_cov = covariance_loss(z1) + covariance_loss(z2)

    loss = sim_coef * loss_inv + var_coef * loss_var + cov_coef * loss_cov

    log = {
        "loss":     loss.item(),
        "inv":      loss_inv.item(),
        "var":      loss_var.item(),
        "cov":      loss_cov.item(),
    }

    return loss, log


# ─────────────────────────────────────────────────────────────────────────────
# VICReg model
# ─────────────────────────────────────────────────────────────────────────────

class VICReg(nn.Module):
    """
    VICReg model: encoder + projector + augmentation.

    During pretraining:
        1. Two views created by EEGAugmentation
        2. Both views encoded by shared encoder
        3. Both embeddings projected by shared Projector
        4. VICReg loss computed on projected embeddings

    During fine-tuning:
        Use encoder only (discard projector).

    Parameters
    ----------
    encoder   : any encoder from ssl_encoders.create_encoder()
    aug       : EEGAugmentation instance
    proj_dim  : projector output dimension (default 512)
    sim_coef  : invariance loss weight
    var_coef  : variance loss weight
    cov_coef  : covariance loss weight
    """
    def __init__(
        self,
        encoder,
        aug:      EEGAugmentation,
        proj_dim: int   = 512,
        sim_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
    ):
        super().__init__()
        self.encoder  = encoder
        self.aug      = aug
        self.projector = Projector(encoder.embed_dim, proj_dim)
        self.sim_coef  = sim_coef
        self.var_coef  = var_coef
        self.cov_coef  = cov_coef

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        x : (B, C, T) EEG batch

        Returns
        -------
        loss : scalar
        log  : dict with loss components
        """
        # Two augmented views
        v1, v2 = self.aug(x)

        # Encode
        e1 = self.encoder(v1)   # (B, embed_dim)
        e2 = self.encoder(v2)

        # Project
        z1 = self.projector(e1)   # (B, proj_dim)
        z2 = self.projector(e2)

        return vicreg_loss(
            z1, z2,
            sim_coef=self.sim_coef,
            var_coef=self.var_coef,
            cov_coef=self.cov_coef,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without augmentation — used at fine-tune/eval time."""
        return self.encoder(x)