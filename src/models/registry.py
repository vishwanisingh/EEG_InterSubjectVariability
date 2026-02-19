"""
Model registry and factory.

Braindecode models (available directly):
    EEGNet, Deep4Net, ShallowFBCSPNet, EEGTCNet, EEGConformer,
    BDTCN, ContraWR, EEGSimpleConv, EEGITNet, EEGNeX,
    EEGInceptionERP, FBCNet, TSception, SyncNet, SPARCNet,
    SCCNet, IFNet, SleepStagerChambon2018, SleepStagerBlanco2020

Custom models (implemented in src/models/):
    EEGTransformer  — patch-based ViT-style transformer
    EEGformer       — temporal + regional + synchronous attention
    DualTSST        — dual temporal-spectral-spatial transformer branches
    MoCoEEG         — contrastive CNN encoder (supervised adaptation)
    STTCNN          — spatial filter + dilated causal TCN
    EEGUnifiedModel — custom CNN with optional MoE / multi-scale kernels
"""

from typing import List
from torch.nn import Module

# ── braindecode models ────────────────────────────────────────────────────────
from braindecode.models import (
    BDTCN,
    ContraWR,
    Deep4Net,
    EEGConformer,
    EEGITNet,
    EEGInceptionERP,
    EEGNeX,
    EEGNet,
    EEGSimpleConv,
    EEGTCNet,
    FBCNet,
    IFNet,
    SCCNet,
    SPARCNet,
    ShallowFBCSPNet,
    SleepStagerBlanco2020,
    SleepStagerChambon2018,
    SyncNet,
    TSception,
)

# ── custom models ─────────────────────────────────────────────────────────────
from .eeg_transformer import EEGTransformer
from .eegformer import EEGformer
from .dual_tsst import DualTSST
from .moco_eeg import MoCoEEG
from .st_tcnn import STTCNN
from .eeg_unified import EEGUnifiedModel


# ── Registry ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    # ── braindecode baselines ─────────────────────────────────────────────────
    "BDTCN":                  BDTCN,
    "ContraWR":               ContraWR,
    "Deep4Net":               Deep4Net,
    "EEGConformer":           EEGConformer,    # ← EEG Conformer
    "EEGITNet":               EEGITNet,
    "EEGInceptionERP":        EEGInceptionERP,
    "EEGNeX":                 EEGNeX,
    "EEGNet":                 EEGNet,
    "EEGSimpleConv":          EEGSimpleConv,
    "EEGTCNet":               EEGTCNet,        # ← EEG-TCNet
    "FBCNet":                 FBCNet,
    "IFNet":                  IFNet,
    "SCCNet":                 SCCNet,
    "SPARCNet":               SPARCNet,
    "ShallowFBCSPNet":        ShallowFBCSPNet,
    "SleepStagerBlanco2020":  SleepStagerBlanco2020,
    "SleepStagerChambon2018": SleepStagerChambon2018,
    "SyncNet":                SyncNet,
    "TSception":              TSception,
    # ── custom models ─────────────────────────────────────────────────────────
    "EEGTransformer":         EEGTransformer,  # ← EEG-Transformer
    "EEGformer":              EEGformer,       # ← EEGformer
    "DualTSST":               DualTSST,        # ← Dual-TSST
    "MoCoEEG":                MoCoEEG,         # ← MoCo-EEG
    "STTCNN":                 STTCNN,          # ← ST-TCNN
    "EEGUnifiedModel":        EEGUnifiedModel,
}

# Models from your requested list — maps your name → registry key
YOUR_MODEL_MAP = {
    "EEGNet":           "EEGNet",
    "Deep4Net":         "Deep4Net",
    "ShallowFBCSPNet":  "ShallowFBCSPNet",
    "EEG-TCNet":        "EEGTCNet",
    "EEG-Transformer":  "EEGTransformer",
    "EEG Conformer":    "EEGConformer",
    "Dual-TSST":        "DualTSST",
    "EEGformer":        "EEGformer",
    "MoCo-EEG":         "MoCoEEG",
    "ST-TCNN":          "STTCNN",
}

BASELINE_MODELS = [
    "EEGNet",
    "Deep4Net",
    "ShallowFBCSPNet",
    "EEGTCNet",
    "EEGConformer",
    "EEGTransformer",
    "EEGformer",
    "DualTSST",
    "MoCoEEG",
    "STTCNN",
]


# ── Factory ───────────────────────────────────────────────────────────────────

def create_model(
    model_name: str,
    n_chans: int = 129,
    n_outputs: int = 1,
    n_times: int = 200,
    sfreq: int = 100,
    **kwargs,
) -> Module:
    """
    Instantiate a model by name.

    Accepts both registry keys (e.g. "EEGTCNet") and your shorthand names
    (e.g. "EEG-TCNet") via YOUR_MODEL_MAP.

    Parameters
    ----------
    model_name : str
        Registry key or shorthand name.
    n_chans : int
        EEG input channels.
    n_outputs : int
        Output values (1 for regression).
    n_times : int
        Time samples per window.
    sfreq : int
        Sampling frequency in Hz.
    **kwargs
        Extra model-specific constructor arguments.

    Returns
    -------
    Module

    Raises
    ------
    ValueError
        If model_name is not found in the registry or shorthand map.
    """
    # Resolve shorthand names
    resolved = YOUR_MODEL_MAP.get(model_name, model_name)

    if resolved not in MODEL_REGISTRY:
        available = "\n  ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'.\nAvailable models:\n  {available}"
        )

    model_class = MODEL_REGISTRY[resolved]
    params = dict(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, sfreq=sfreq)
    params.update(kwargs)

    try:
        return model_class(**params)
    except TypeError as exc:
        # Some braindecode models don't accept all four standard params
        print(f"  [create_model] Warning: {exc}. Retrying with minimal params.")
        minimal = {k: v for k, v in params.items() if k in ("n_chans", "n_outputs", "n_times")}
        return model_class(**minimal)


def get_available_models() -> List[str]:
    """Return sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY)


def get_baseline_models() -> List[str]:
    """Return the full list of baseline models (braindecode + custom)."""
    return list(BASELINE_MODELS)