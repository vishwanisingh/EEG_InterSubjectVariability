"""
ssl_finetune.py
---------------
Fine-tune a pretrained SSL encoder on CCD RT prediction,
then evaluate transfer to SeqLearning and SymbolSearch.

Three fine-tuning modes
-----------------------
linear   — encoder frozen, only regression head trained (linear probe)
           Tests: are SSL embeddings linearly separable for RT?
partial  — last 2 encoder layers + head trained
           Balance between transfer and task adaptation
full     — all encoder layers + head trained
           Maximum performance

Three evaluation scenarios
--------------------------
ISV      — CCD held-out subjects (inter-subject variability)
TI_SP    — CCD → SymbolSearch (within-paradigm transfer)
TI_AP    — CCD → SeqLearning  (across-paradigm transfer)

Critical ablation
-----------------
method=supervised → same encoder architecture, random init, trained from scratch
Isolates: is the gain from SSL pretraining or just the architecture?

Weight file naming convention (matches ssl_pretrain.py):
    vicreg_{encoder}_{data_regime}_encoder_best.pt
    byol_{encoder}_{data_regime}_encoder_best.pt

Usage
-----
    # Full fine-tune, VICReg custom encoder pretrained on passive data
    python ssl_finetune.py --method vicreg --encoder custom --data passive --finetune_mode full

    # Linear probe (frozen encoder)
    python ssl_finetune.py --method vicreg --encoder custom --data passive --finetune_mode linear

    # BYOL encoder
    python ssl_finetune.py --method byol --encoder tcnet --data passive --finetune_mode full

    # Supervised from scratch (ablation)
    python ssl_finetune.py --method supervised --encoder custom

    # Run all modes for one encoder/data combo
    python ssl_finetune.py --method vicreg --encoder custom --data passive --all_modes
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from configs.config import (
    TASK, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    N_CHANS, N_TIMES, SFREQ, N_OUTPUTS,
    BATCH_SIZE, NUM_WORKERS,
    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, MIN_DELTA,
    VALID_FRAC, TEST_FRAC, SEED, EXCLUDED_SUBJECTS,
)
from disk_dataset import load_windowed_dataset, EEGWindowDataset
from ssl_encoders import create_encoder
from src.utils.device import setup_device


# ── Config ────────────────────────────────────────────────────────────────────
CCD_RELEASE      = "R5"         # fine-tune and ISV test release
TRANSFER_RELEASE = "R5"         # transfer evaluation release

TRANSFER_TASKS = {
    "TI_SP": "symbolSearch",         # within-paradigm
    "TI_AP": "seqLearning8target",   # across-paradigm
}

FINETUNE_MODES = ["linear", "partial", "full"]


# ─────────────────────────────────────────────────────────────────────────────
# Normalization (global z-score — transfer compatible)
# ─────────────────────────────────────────────────────────────────────────────

class NormStats:
    """Global z-score stats from training split."""
    def __init__(self, train_dataset, max_windows: int = 5000, eps: float = 1e-6):
        n   = min(max_windows, len(train_dataset))
        idx = np.random.default_rng(SEED).choice(len(train_dataset), n, replace=False)

        eeg_sum = np.zeros(N_CHANS, dtype=np.float64)
        eeg_sq  = np.zeros(N_CHANS, dtype=np.float64)
        rt_vals = []

        for i in idx:
            x, y = train_dataset[int(i)][:2]
            xn    = x.numpy()
            eeg_sum += xn.mean(axis=1)
            eeg_sq  += (xn ** 2).mean(axis=1)
            v = y.item()
            if not np.isnan(v):
                rt_vals.append(v)

        self.eeg_mean = (eeg_sum / n).astype(np.float32)
        self.eeg_std  = (np.sqrt(
            np.maximum(eeg_sq / n - (eeg_sum / n) ** 2, 0)
        ) + eps).astype(np.float32)

        rt_arr       = np.array(rt_vals, dtype=np.float64)
        self.rt_mean = float(rt_arr.mean())
        self.rt_std  = float(rt_arr.std()) + eps

        print(f"  [NormStats] EEG std [{self.eeg_std.min():.4f}, {self.eeg_std.max():.4f}]")
        print(f"  [NormStats] RT  mean={self.rt_mean:.4f}  std={self.rt_std:.4f}")


class NormalizedDataset(Dataset):
    """Wraps EEGWindowDataset with on-the-fly global z-score normalization."""
    def __init__(self, dataset, stats: NormStats):
        self.dataset   = dataset
        self._eeg_mean = torch.from_numpy(stats.eeg_mean).unsqueeze(1)
        self._eeg_std  = torch.from_numpy(stats.eeg_std).unsqueeze(1)
        self._rt_mean  = stats.rt_mean
        self._rt_std   = stats.rt_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item   = self.dataset[idx]
        x, y   = item[0], item[1]
        x_norm = (x - self._eeg_mean) / self._eeg_std
        y_norm = (y - self._rt_mean)  / self._rt_std
        return x_norm, y_norm

    def denormalize_rt(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self._rt_std + self._rt_mean


# ─────────────────────────────────────────────────────────────────────────────
# Regression head + finetuned model
# ─────────────────────────────────────────────────────────────────────────────

class RegressionHead(nn.Module):
    """
    2-layer MLP regression head attached to encoder output.
    embed_dim → hidden → 1
    """
    def __init__(self, embed_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_OUTPUTS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinetunedEEGModel(nn.Module):
    """
    Pretrained encoder + regression head.

    Finetune modes
    --------------
    linear  : encoder frozen, head trained
    partial : last 2 encoder children + head trained
    full    : everything trained (lower LR for encoder recommended)
    """
    def __init__(
        self,
        encoder,
        finetune_mode: str = "full",
        head_hidden:   int = 128,
    ):
        super().__init__()
        self.encoder       = encoder
        self.head          = RegressionHead(encoder.embed_dim, head_hidden)
        self.finetune_mode = finetune_mode
        self._apply_freeze()

    def _apply_freeze(self) -> None:
        if self.finetune_mode == "linear":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            print(f"  [freeze] Encoder fully frozen")

        elif self.finetune_mode == "partial":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # Unfreeze last 2 named children of encoder
            children = list(self.encoder.named_children())
            for name, module in children[-2:]:
                for p in module.parameters():
                    p.requires_grad_(True)
                print(f"  [partial] Unfrozen: {name}")

        else:  # full
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            print(f"  [freeze] Encoder fully trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def get_param_groups(self, lr: float) -> list:
        """
        Return param groups with lower LR for encoder (full mode).
        Helps prevent catastrophic forgetting of pretrained representations.
        """
        if self.finetune_mode == "full":
            return [
                {"params": self.encoder.parameters(), "lr": lr * 0.1},
                {"params": self.head.parameters(),    "lr": lr},
            ]
        return [{"params": filter(lambda p: p.requires_grad, self.parameters()),
                 "lr": lr}]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model:  nn.Module,
    device: torch.device,
    stats:  NormStats,
    label:  str,
) -> dict:
    model.eval()
    preds, targets = [], []

    for batch in loader:
        x, y = batch[0].to(device).float(), batch[1].to(device).float()
        preds.append(model(x).cpu())
        targets.append(y.cpu())

    p = torch.cat(preds).view(-1)
    t = torch.cat(targets).view(-1)

    rmse_norm = float(torch.sqrt(((p - t) ** 2).mean()))
    mae_norm  = float(torch.abs(p - t).mean())
    ss_res    = float(((t - p) ** 2).sum())
    ss_tot    = float(((t - t.mean()) ** 2).sum())
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Denormalize to seconds
    p_sec    = p * stats.rt_std + stats.rt_mean
    t_sec    = t * stats.rt_std + stats.rt_mean
    rmse_sec = float(torch.sqrt(((p_sec - t_sec) ** 2).mean()))
    mae_sec  = float(torch.abs(p_sec - t_sec).mean())

    print(f"  [{label}]")
    print(f"    Norm   RMSE={rmse_norm:.4f}  MAE={mae_norm:.4f}  R²={r2:.4f}")
    print(f"    Denorm RMSE={rmse_sec:.4f}s  MAE={mae_sec:.4f}s")

    return dict(
        rmse_norm=rmse_norm, mae_norm=mae_norm, r2=r2,
        rmse_sec=rmse_sec,   mae_sec=mae_sec,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_finetune(
    model:        nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device:       torch.device,
    n_epochs:     int,
    lr:           float,
    weight_decay: float,
) -> dict:
    """Simple training loop with early stopping."""
    optimizer = torch.optim.AdamW(
        model.get_param_groups(lr),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state    = None
    patience_ctr  = 0
    train_hist, val_hist = [], []

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            x, y = batch[0].to(device).float(), batch[1].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred.view(-1), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        nv = 0
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch[0].to(device).float(), batch[1].to(device).float()
                pred = model(x)
                val_loss += criterion(pred.view(-1), y.view(-1)).item()
                nv += 1

        train_rmse = (train_loss / max(n,  1)) ** 0.5
        val_rmse   = (val_loss   / max(nv, 1)) ** 0.5
        train_hist.append(train_rmse)
        val_hist.append(val_rmse)

        print(f"  Epoch {epoch:03d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse - MIN_DELTA:
            best_val_rmse = val_rmse
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return {
        "best_val_rmse": best_val_rmse,
        "best_state":    best_state,
        "train_history": train_hist,
        "val_history":   val_hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main finetune function
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    method:        str   = "vicreg",
    encoder_name:  str   = "custom",
    data_regime:   str   = "passive",
    finetune_mode: str   = "full",
    embed_dim:     int   = 256,
    head_hidden:   int   = 128,
    n_epochs:      int   = N_EPOCHS,
    lr:            float = 1e-3,
    weight_decay:  float = WEIGHT_DECAY,
) -> dict:
    """
    Fine-tune pretrained encoder on CCD, evaluate ISV + transfer.

    Parameters
    ----------
    method        : "vicreg" | "byol" | "supervised"
    encoder_name  : "tcnet" | "conformer" | "custom"
    data_regime   : "passive" | "active" | "all" (matches pretrain)
    finetune_mode : "linear" | "partial" | "full"
    embed_dim     : must match pretrained encoder
    head_hidden   : regression head hidden dim
    n_epochs      : max fine-tuning epochs
    lr            : learning rate (encoder gets lr*0.1 in full mode)
    weight_decay  : L2 regularization
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = setup_device()
    torch.manual_seed(SEED)

    # Run name matches ssl_pretrain.py naming
    if method == "supervised":
        run_name   = f"supervised_{encoder_name}"
        model_name = f"supervised_{encoder_name}_{finetune_mode}"
    else:
        run_name   = f"{method}_{encoder_name}_{data_regime}"
        model_name = f"{method}_{encoder_name}_{data_regime}_{finetune_mode}"

    print(f"\n{'='*60}")
    print(f"SSL FINE-TUNE")
    print(f"{'='*60}")
    print(f"  Method        : {method}")
    print(f"  Encoder       : {encoder_name}")
    print(f"  Data regime   : {data_regime}")
    print(f"  Finetune mode : {finetune_mode}")
    print(f"  Model name    : {model_name}")
    print('='*60)

    # ── [1/6] Load CCD event-locked data ─────────────────────────────────────
    print(f"\n[1/6] Loading CCD R5 (event-locked) ...")
    root = WINDOWS_DIR / f"contrastChangeDetection_{CCD_RELEASE}"
    ccd  = load_windowed_dataset(
        root_dir=root, target_key="target",
        eager=True, excluded_subjects=EXCLUDED_SUBJECTS,
    )
    print(f"  {len(ccd)} windows, {len(ccd.get_subject_ids())} subjects")

    # ── [2/6] Subject split ───────────────────────────────────────────────────
    print(f"\n[2/6] Splitting by subjects (80:10:10) ...")
    subjects = ccd.get_subject_ids()
    tv, test_s = train_test_split(
        subjects, test_size=TEST_FRAC,
        random_state=check_random_state(SEED), shuffle=True,
    )
    train_s, val_s = train_test_split(
        tv, test_size=VALID_FRAC / (1 - TEST_FRAC),
        random_state=check_random_state(SEED + 1), shuffle=True,
    )
    train_set, val_set, test_set = ccd.split_by_subjects(train_s, val_s, test_s)
    print(f"  Train: {len(train_s)} subj ({len(train_set)} windows)")
    print(f"  Val  : {len(val_s)}   subj ({len(val_set)} windows)")
    print(f"  Test : {len(test_s)}  subj ({len(test_set)} windows)")

    # ── [3/6] Normalization ───────────────────────────────────────────────────
    print(f"\n[3/6] Computing normalization stats from training split ...")
    stats         = NormStats(train_set)
    norm_train    = NormalizedDataset(train_set, stats)
    norm_val      = NormalizedDataset(val_set,   stats)
    norm_test     = NormalizedDataset(test_set,  stats)

    train_loader = DataLoader(norm_train, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(norm_val,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(norm_test,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # ── [4/6] Build model ─────────────────────────────────────────────────────
    print(f"\n[4/6] Building model ...")
    encoder = create_encoder(encoder_name, N_CHANS, N_TIMES, embed_dim)

    if method != "supervised":
        # Load pretrained weights
        ckpt = WEIGHTS_DIR / f"{run_name}_encoder_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found: {ckpt}\n"
                f"Run: python ssl_pretrain.py "
                f"--method {method} --encoder {encoder_name} --data {data_regime}"
            )
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"  Loaded: {ckpt.name}")
    else:
        print(f"  Random init (supervised ablation)")

    model = FinetunedEEGModel(encoder, finetune_mode, head_hidden)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total")

    # ── [5/6] Fine-tune ───────────────────────────────────────────────────────
    print(f"\n[5/6] Fine-tuning ...")
    train_results = train_finetune(
        model, train_loader, val_loader, device,
        n_epochs, lr, weight_decay,
    )

    # ── [6/6] Evaluate ────────────────────────────────────────────────────────
    print(f"\n[6/6] Evaluating ...")

    # ISV — CCD held-out subjects
    isv_metrics = evaluate(test_loader, model, device, stats, "ISV (CCD test)")

    # Transfer — SymbolSearch and SeqLearning
    transfer_metrics = {}
    for scenario, task in TRANSFER_TASKS.items():
        transfer_root = WINDOWS_DIR / f"{task}_{TRANSFER_RELEASE}"
        if not transfer_root.exists():
            print(f"  [skip] {scenario} ({task}) — not found at {transfer_root}")
            continue
        try:
            transfer_ds = load_windowed_dataset(
                root_dir=transfer_root, target_key="target",
                eager=True, excluded_subjects=EXCLUDED_SUBJECTS,
            )
            # Apply CCD training stats for normalization
            norm_transfer   = NormalizedDataset(transfer_ds, stats)
            transfer_loader = DataLoader(norm_transfer, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)
            m = evaluate(transfer_loader, model, device, stats,
                         f"{scenario} ({task})")
            transfer_metrics[scenario] = m
        except Exception as e:
            print(f"  [skip] {scenario}: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    row = {
        "model_name":    model_name,
        "method":        method,
        "encoder":       encoder_name,
        "data_regime":   data_regime,
        "finetune_mode": finetune_mode,
        "best_val_rmse": train_results["best_val_rmse"],
        # ISV
        "ISV_rmse_norm": isv_metrics["rmse_norm"],
        "ISV_r2":        isv_metrics["r2"],
        "ISV_rmse_sec":  isv_metrics["rmse_sec"],
        # Transfer
        **{f"{s}_rmse_norm": transfer_metrics.get(s, {}).get("rmse_norm", float("nan"))
           for s in TRANSFER_TASKS},
        **{f"{s}_r2":        transfer_metrics.get(s, {}).get("r2",        float("nan"))
           for s in TRANSFER_TASKS},
        **{f"{s}_rmse_sec":  transfer_metrics.get(s, {}).get("rmse_sec",  float("nan"))
           for s in TRANSFER_TASKS},
        # Normalization stats (for reproducibility)
        "rt_mean_train": stats.rt_mean,
        "rt_std_train":  stats.rt_std,
    }

    results_csv = RESULTS_DIR / "ssl_results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_csv, index=False)

    # Save weights
    ckpt_out = WEIGHTS_DIR / f"{model_name}_finetuned.pt"
    torch.save(train_results["best_state"], ckpt_out)

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"  Val  RMSE (norm) : {train_results['best_val_rmse']:.4f}")
    print(f"  ISV  R²          : {isv_metrics['r2']:.4f}  "
          f"RMSE={isv_metrics['rmse_sec']:.4f}s")
    for s, m in transfer_metrics.items():
        print(f"  {s:<8} R²     : {m['r2']:.4f}  RMSE={m['rmse_sec']:.4f}s")
    print(f"  Saved → {results_csv}")
    print('='*60)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Fine-tuning")
    parser.add_argument("--method",        type=str, default="vicreg",
                        choices=["vicreg", "byol", "supervised"])
    parser.add_argument("--encoder",       type=str, default="custom",
                        choices=["tcnet", "conformer", "custom"])
    parser.add_argument("--data",          type=str, default="passive",
                        choices=["passive", "active", "all"],
                        help="Must match the data regime used in pretraining.")
    parser.add_argument("--finetune_mode", type=str, default="full",
                        choices=["linear", "partial", "full"])
    parser.add_argument("--all_modes",     action="store_true",
                        help="Run all three finetune modes sequentially.")
    parser.add_argument("--embed_dim",     type=int,   default=256)
    parser.add_argument("--head_hidden",   type=int,   default=128)
    parser.add_argument("--epochs",        type=int,   default=N_EPOCHS)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    args = parser.parse_args()

    modes = FINETUNE_MODES if args.all_modes else [args.finetune_mode]

    for mode in modes:
        finetune(
            method=args.method,
            encoder_name=args.encoder,
            data_regime=args.data,
            finetune_mode=mode,
            embed_dim=args.embed_dim,
            head_hidden=args.head_hidden,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )