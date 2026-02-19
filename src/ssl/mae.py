"""
src/ssl/mae.py
--------------
MAE: Masked Autoencoder for EEG

Mask random time patches of the EEG signal, then train an encoder-decoder
to reconstruct the masked patches from the visible ones.

After pretraining, only the ENCODER is kept for fine-tuning — the decoder
is discarded. The encoder learns to extract meaningful representations from
partial EEG context, which generalizes well to downstream tasks.

Architecture:
  Encoder : ViT-style patch embedding + Transformer (processes visible patches only)
  Decoder : Smaller Transformer + linear reconstruction head
            (processes all patches with masked tokens for missing ones)

This is particularly well-motivated for your setup:
  - Pretrain on ALL HBN tasks (resting, movies, CCD) — no labels needed
  - Fine-tune encoder on CCD RT prediction
  - The resting/movie EEG provides free unsupervised signal from same subjects

Paper: He et al., 2022 (original MAE) — https://arxiv.org/abs/2111.06377
       Applied to EEG: EEG-MAE — https://arxiv.org/abs/2211.02625

Input  : (B, C, T) EEG windows
Output : scalar reconstruction loss during pretraining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .augmentations import get_mae_mask


class _PatchEmbed(nn.Module):
    """Embed EEG patches: (B, C, T) → (B, n_patches, d_model)."""

    def __init__(self, n_chans: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = n_chans * patch_size
        self.proj = nn.Linear(patch_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        P = self.patch_size
        n = T // P
        x = x.reshape(B, C, n, P).permute(0, 2, 1, 3)  # (B, n, C, P)
        x = x.reshape(B, n, C * P)                       # (B, n, patch_dim)
        return self.norm(self.proj(x))                    # (B, n, d_model)


class _TransformerEncoder(nn.Module):
    """Standard Transformer encoder with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.transformer(x))


class EEGMaskedEncoder(nn.Module):
    """
    MAE encoder: processes only the VISIBLE (unmasked) patches.

    This is more efficient than processing all patches — the encoder
    never sees masked tokens during pretraining, forcing it to learn
    from partial context.

    Parameters
    ----------
    n_chans : int
        EEG input channels.
    n_times : int
        Time samples per window.
    patch_size : int
        Time samples per patch. n_times must be divisible by patch_size.
    d_model : int
        Encoder hidden dimension.
    n_heads : int
        Attention heads.
    n_layers : int
        Encoder Transformer layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        patch_size: int = 10,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_times % patch_size == 0, \
            f"n_times ({n_times}) must be divisible by patch_size ({patch_size})"

        self.patch_size = patch_size
        self.n_patches  = n_times // patch_size
        self.d_model    = d_model
        self.embed_dim  = d_model   # exposed for finetune.py compatibility

        self.patch_embed = _PatchEmbed(n_chans, patch_size, d_model)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = _TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)

    def forward_with_mask(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode only the visible patches.

        Parameters
        ----------
        x    : (B, C, T)
        mask : (B, n_patches) bool — True where MASKED (hidden from encoder)

        Returns
        -------
        visible_tokens : (B, n_visible, d_model)
        ids_restore    : (B, n_patches) — indices to restore original patch order
        """
        tokens = self.patch_embed(x) + self.pos_embed  # (B, n, d)

        # Split into visible and masked
        B, N, D = tokens.shape
        ids_shuffle = torch.argsort(mask.float(), dim=1, stable=True)   # visible first
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        n_visible = (~mask[0]).sum().item()
        visible_ids = ids_shuffle[:, :n_visible]
        visible_tokens = torch.gather(
            tokens, 1,
            visible_ids.unsqueeze(-1).expand(-1, -1, D)
        )

        visible_tokens = self.transformer(visible_tokens)
        return visible_tokens, ids_restore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode all patches (used during fine-tuning — no masking).

        Returns
        -------
        (B, d_model) — mean-pooled representation
        """
        tokens = self.patch_embed(x) + self.pos_embed
        tokens = self.transformer(tokens)
        return tokens.mean(dim=1)   # global average pool over patches


class EEGMaskedAutoencoder(nn.Module):
    """
    Full MAE model for pretraining: encoder + decoder + reconstruction head.

    During pretraining:
      1. Mask mask_ratio fraction of patches randomly
      2. Encoder processes only visible patches
      3. Decoder receives visible tokens + learnable mask tokens,
         reconstructs all patches
      4. Loss = MSE on masked patches only (not visible ones)

    During fine-tuning:
      - Use only self.encoder.forward(x) — discard decoder
      - The encoder already knows how to encode from this module

    Parameters
    ----------
    n_chans : int
    n_times : int
    patch_size : int
    mask_ratio : float
        Fraction of patches to mask during pretraining (e.g. 0.75).
    encoder_d_model : int
        Encoder hidden dimension.
    encoder_heads : int
    encoder_layers : int
    decoder_d_model : int
        Decoder hidden dimension (can be smaller than encoder).
    decoder_heads : int
    decoder_layers : int
    dropout : float
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        patch_size: int = 10,
        mask_ratio: float = 0.75,
        encoder_d_model: int = 256,
        encoder_heads: int = 8,
        encoder_layers: int = 6,
        decoder_d_model: int = 128,
        decoder_heads: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_times % patch_size == 0
        self.n_chans    = n_chans
        self.patch_size = patch_size
        self.n_patches  = n_times // patch_size
        self.mask_ratio = mask_ratio
        patch_dim       = n_chans * patch_size

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = EEGMaskedEncoder(
            n_chans=n_chans, n_times=n_times, patch_size=patch_size,
            d_model=encoder_d_model, n_heads=encoder_heads,
            n_layers=encoder_layers, d_ff=encoder_d_model * 2, dropout=dropout,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Project encoder tokens into decoder space
        self.enc_to_dec   = nn.Linear(encoder_d_model, decoder_d_model, bias=False)
        # Learnable mask token (same vector for all masked positions)
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Positional embedding for decoder (covers all n_patches positions)
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, decoder_d_model))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        self.decoder = _TransformerEncoder(
            decoder_d_model, decoder_heads, decoder_layers,
            decoder_d_model * 2, dropout,
        )
        # Reconstruction head: project back to raw patch values
        self.recon_head = nn.Linear(decoder_d_model, patch_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Full MAE forward pass for pretraining.

        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        loss : scalar — MSE on masked patches only
        log  : dict with 'recon_loss' for logging
        """
        B, C, T = x.shape
        N = self.n_patches

        # Generate mask
        mask, _ = get_mae_mask(x, mask_ratio=self.mask_ratio, patch_size=self.patch_size)

        # ── Encode visible patches ─────────────────────────────────────────────
        visible_tokens, ids_restore = self.encoder.forward_with_mask(x, mask)

        # ── Decode: reconstruct all patches ───────────────────────────────────
        # Project visible tokens to decoder dim
        dec_tokens = self.enc_to_dec(visible_tokens)  # (B, n_visible, dec_d)

        # Build full token sequence: visible + mask tokens
        n_visible = dec_tokens.shape[1]
        n_masked  = N - n_visible
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        all_tokens  = torch.cat([dec_tokens, mask_tokens], dim=1)  # (B, N, dec_d)

        # Unshuffle back to original patch order
        all_tokens = torch.gather(
            all_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
        )

        # Add decoder positional embedding and decode
        all_tokens = all_tokens + self.dec_pos_embed
        decoded    = self.decoder(all_tokens)               # (B, N, dec_d)
        pred       = self.recon_head(decoded)               # (B, N, patch_dim)

        # ── Reconstruction target: actual patch values ─────────────────────────
        patch_dim = C * self.patch_size
        target = x.reshape(B, C, N, self.patch_size).permute(0, 2, 1, 3)
        target = target.reshape(B, N, patch_dim)

        # ── Loss: MSE on masked patches only ──────────────────────────────────
        loss = F.mse_loss(pred[mask], target[mask])

        return loss, {"recon_loss": loss.item()}

    def get_encoder(self) -> EEGMaskedEncoder:
        """Return the pretrained encoder for fine-tuning."""
        return self.encoder
