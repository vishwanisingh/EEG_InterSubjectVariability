"""
EEG-Transformer
---------------
A straightforward Transformer encoder applied to EEG time series.

Input  : (B, n_chans, n_times)
Output : (B, n_outputs)

Architecture:
  1. Linear patch embedding — splits the time axis into non-overlapping patches
     and projects each patch to d_model dimensions.
  2. Learnable [CLS] token prepended to the patch sequence.
  3. Learnable positional embedding added to the full sequence.
  4. N stacked Transformer encoder layers (multi-head self-attention + FFN).
  5. Linear head applied to the [CLS] token representation.

Reference: inspired by "Transformers are Effective for EEG Decoding" style
           architectures used across multiple EEG BCI papers.
"""

import torch
import torch.nn as nn


class EEGTransformer(nn.Module):
    """
    Patch-based Transformer encoder for EEG regression / classification.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels (used as input feature dim per time-patch).
    n_outputs : int
        Number of output values (1 for regression).
    n_times : int
        Number of time samples per window.
    sfreq : int
        Sampling frequency (unused directly, kept for registry compatibility).
    patch_size : int
        Number of time samples per patch. n_times must be divisible by patch_size.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads. d_model must be divisible by n_heads.
    n_layers : int
        Number of Transformer encoder layers.
    d_ff : int
        Feed-forward inner dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        patch_size: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert n_times % patch_size == 0, (
            f"n_times ({n_times}) must be divisible by patch_size ({patch_size})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.patch_size = patch_size
        n_patches       = n_times // patch_size
        patch_dim       = n_chans * patch_size   # raw values per patch

        # ── Patch embedding ───────────────────────────────────────────────────
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # ── [CLS] token + positional embedding ───────────────────────────────
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout_emb = nn.Dropout(dropout)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── Regression / classification head ─────────────────────────────────
        self.head = nn.Linear(d_model, n_outputs)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)

        Returns
        -------
        (B, n_outputs)
        """
        B, C, T = x.shape
        P = self.patch_size
        n_patches = T // P

        # (B, n_chans, n_times) → (B, n_patches, n_chans * patch_size)
        x = x.reshape(B, C, n_patches, P)          # (B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3)                  # (B, n_patches, C, P)
        x = x.reshape(B, n_patches, C * P)         # (B, n_patches, patch_dim)

        # Patch embedding
        x = self.patch_embed(x)                    # (B, n_patches, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)           # (B, n_patches+1, d_model)

        # Positional embedding
        x = self.dropout_emb(x + self.pos_embed)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # Use [CLS] token for prediction
        return self.head(x[:, 0])                  # (B, n_outputs)