"""
EEGformer
---------
Re-implementation of the EEGformer architecture.

Key idea: three sequential attention modules that capture different aspects
of EEG signals:
  1. Temporal Self-Attention  — captures temporal dynamics within each channel
  2. Regional Self-Attention  — captures spatial relationships across channels
  3. Synchronous Attention    — captures cross-region synchrony via cross-attention

Input  : (B, n_chans, n_times)
Output : (B, n_outputs)

Reference: "EEGformer: Towards Transferable and Interpretable Large-Scale EEG
            Foundation Model" (Yang et al., 2024) — simplified version adapted
            for supervised regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalAttention(nn.Module):
    """Self-attention along the time axis, applied independently per channel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B * n_chans, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class _RegionalAttention(nn.Module):
    """Self-attention along the channel axis, applied independently per time step."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, T, d_model) → reshape for channel attention
        B, C, T, D = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(B * T, C, D)  # (B*T, C, D)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.norm(x_t + self.drop(attn_out))
        x_t = self.norm2(x_t + self.drop(self.ff(x_t)))
        x = x_t.reshape(B, T, C, D).permute(0, 2, 1, 3)   # (B, C, T, D)
        return x


class _SynchronousAttention(nn.Module):
    """Cross-attention between channel summary tokens and global context."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(d_model)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, d_model)  — channel summary tokens
        # Query: each channel attends to all other channels (synchrony)
        attn_out, _ = self.cross_attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class EEGformer(nn.Module):
    """
    EEGformer: temporal → regional → synchronous attention pipeline.

    Parameters
    ----------
    n_chans : int
        Number of EEG channels.
    n_outputs : int
        Output dimension (1 for regression).
    n_times : int
        Time samples per window.
    sfreq : int
        Sampling frequency (unused, kept for registry compatibility).
    d_model : int
        Internal embedding dimension.
    n_heads : int
        Attention heads (must divide d_model).
    n_layers : int
        Number of EEGformer blocks stacked.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        n_chans: int = 129,
        n_outputs: int = 1,
        n_times: int = 200,
        sfreq: int = 100,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project raw time samples → d_model per time step per channel
        self.input_proj = nn.Linear(1, d_model)

        # Positional embedding over time
        self.pos_embed  = nn.Parameter(torch.zeros(1, 1, n_times, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stacked EEGformer blocks
        self.temporal_attn    = nn.ModuleList([_TemporalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.regional_attn    = nn.ModuleList([_RegionalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.synchronous_attn = nn.ModuleList([_SynchronousAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Head: project pooled representation to output
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_chans, n_times)
        """
        B, C, T = x.shape

        # Project each scalar sample to d_model: (B, C, T, 1) → (B, C, T, D)
        x = x.unsqueeze(-1)                       # (B, C, T, 1)
        x = self.input_proj(x)                    # (B, C, T, D)
        x = x + self.pos_embed                    # broadcast over B, C

        for t_attn, r_attn, s_attn in zip(self.temporal_attn, self.regional_attn, self.synchronous_attn):
            # ── Temporal attention (per channel) ─────────────────────────────
            B, C, T, D = x.shape
            x_t = x.reshape(B * C, T, D)
            x_t = t_attn(x_t)
            x   = x_t.reshape(B, C, T, D)

            # ── Regional attention (per time step, across channels) ───────────
            x   = r_attn(x)

            # ── Synchronous attention (channel summary tokens) ────────────────
            x_summary = x.mean(dim=2)             # (B, C, D) — avg over time
            x_summary = s_attn(x_summary)
            # Broadcast synchrony context back
            x = x + x_summary.unsqueeze(2)

        # Pool over channels and time → single vector per sample
        out = x.mean(dim=(1, 2))                  # (B, D)
        out = self.norm(out)
        return self.head(out)                      # (B, n_outputs)