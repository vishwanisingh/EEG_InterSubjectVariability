"""
src/ssl/augmentations.py
------------------------
EEG-specific augmentations for self-supervised learning.

Key design principle: augmentations must be LABEL-PRESERVING — they should
change the signal in ways that don't affect the underlying cognitive state
(RT, attention, etc.) so the encoder learns to be invariant to them.

For RT prediction:
  SAFE  — amplitude scaling, Gaussian noise, channel dropout, time shift
  RISKY — frequency masking (may remove RT-relevant beta/gamma), strong crop

Each augmentation is a callable that takes (B, C, T) tensor and returns
(B, C, T) tensor of the same shape. All are differentiable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class GaussianNoise(nn.Module):
    """Add zero-mean Gaussian noise. Standard denoising augmentation."""

    def __init__(self, std: float = 0.1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        return x + torch.randn_like(x) * self.std


class AmplitudeScale(nn.Module):
    """
    Randomly scale signal amplitude per sample.
    Simulates electrode impedance variation and gain differences.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.75, 1.25)):
        super().__init__()
        self.lo, self.hi = scale_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B = x.shape[0]
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(self.lo, self.hi)
        return x * scale


class ChannelDropout(nn.Module):
    """
    Zero out random channels (electrodes) per sample.
    Simulates bad electrode contact — very common in real EEG.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, T = x.shape
        mask = torch.bernoulli(
            torch.full((B, C, 1), 1.0 - self.p, device=x.device)
        )
        return x * mask


class TemporalShift(nn.Module):
    """
    Randomly shift the signal along the time axis by circular rolling.
    Teaches the encoder to be invariant to small temporal offsets.
    Max shift is bounded so the stimulus-locked structure is mostly preserved.
    """

    def __init__(self, max_shift_frac: float = 0.1):
        super().__init__()
        self.max_shift_frac = max_shift_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        T = x.shape[-1]
        max_shift = int(T * self.max_shift_frac)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(x, shifts=shift, dims=-1)


class FrequencyMask(nn.Module):
    """
    Zero out a random band of frequencies in the Fourier domain.
    Applied independently per sample. Teaches frequency robustness.

    Use cautiously for RT — beta/gamma power is RT-predictive, so
    masking these may degrade downstream performance if used too aggressively.
    """

    def __init__(self, max_mask_frac: float = 0.15):
        super().__init__()
        self.max_mask_frac = max_mask_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, T = x.shape
        x_fft = torch.fft.rfft(x, dim=-1)
        n_freqs = x_fft.shape[-1]
        mask_len = int(n_freqs * self.max_mask_frac)

        for b in range(B):
            start = torch.randint(0, n_freqs - mask_len + 1, (1,)).item()
            x_fft[b, :, start: start + mask_len] = 0.0

        return torch.fft.irfft(x_fft, n=T, dim=-1)


class TemporalCrop(nn.Module):
    """
    Randomly crop a contiguous sub-window and resize back to original length.
    Forces the encoder to work from partial temporal context.
    """

    def __init__(self, min_crop_frac: float = 0.7):
        super().__init__()
        self.min_crop_frac = min_crop_frac

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        T = x.shape[-1]
        min_len = int(T * self.min_crop_frac)
        crop_len = torch.randint(min_len, T + 1, (1,)).item()
        start = torch.randint(0, T - crop_len + 1, (1,)).item()
        cropped = x[:, :, start: start + crop_len]
        return F.interpolate(cropped.float(), size=T, mode="linear", align_corners=False)


class SignFlip(nn.Module):
    """
    Randomly flip the sign of the signal (multiply by -1).
    EEG signals have arbitrary polarity conventions.
    Applied per sample with probability p.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B = x.shape[0]
        flip = (torch.rand(B, 1, 1, device=x.device) < self.p).float() * 2 - 1
        return x * flip


class EEGAugmentation(nn.Module):
    """
    Compose multiple EEG augmentations into a single callable.

    Used in SSL to produce two different augmented views of the same
    EEG window.

    Parameters
    ----------
    augmentations : list of nn.Module
        List of augmentation modules to apply sequentially.
    """

    def __init__(self, augmentations: List[nn.Module]):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x)
        return x


# ── Pre-built augmentation pipelines ─────────────────────────────────────────

def get_default_augmentation(
    noise_std: float = 0.05,
    amplitude_range: Tuple[float, float] = (0.8, 1.2),
    channel_dropout_p: float = 0.1,
    temporal_shift_frac: float = 0.1,
) -> EEGAugmentation:
    """
    Conservative augmentation pipeline suitable for RT prediction.
    Avoids aggressive frequency masking to preserve RT-relevant bands.
    """
    return EEGAugmentation([
        AmplitudeScale(scale_range=amplitude_range),
        GaussianNoise(std=noise_std),
        ChannelDropout(p=channel_dropout_p),
        TemporalShift(max_shift_frac=temporal_shift_frac),
        SignFlip(p=0.3),
    ])


def get_strong_augmentation() -> EEGAugmentation:
    """
    Stronger augmentation pipeline — use for contrastive methods
    where large view difference is beneficial.
    """
    return EEGAugmentation([
        AmplitudeScale(scale_range=(0.6, 1.4)),
        GaussianNoise(std=0.1),
        ChannelDropout(p=0.2),
        TemporalShift(max_shift_frac=0.15),
        FrequencyMask(max_mask_frac=0.1),
        TemporalCrop(min_crop_frac=0.75),
        SignFlip(p=0.5),
    ])


def get_mae_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.75,
    patch_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random patch mask for MAE pretraining.

    Parameters
    ----------
    x : (B, C, T)
    mask_ratio : float
        Fraction of patches to mask.
    patch_size : int
        Time samples per patch.

    Returns
    -------
    mask : (B, n_patches) bool tensor — True where masked
    patch_indices : (B, n_patches) long tensor — shuffled patch order
    """
    B, C, T = x.shape
    n_patches = T // patch_size
    n_mask = int(n_patches * mask_ratio)

    # Random patch order per sample
    noise = torch.rand(B, n_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)

    mask = torch.zeros(B, n_patches, dtype=torch.bool, device=x.device)
    mask.scatter_(1, ids_shuffle[:, :n_mask], True)

    return mask, ids_shuffle
