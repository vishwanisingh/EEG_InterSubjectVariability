"""
ssl_pretrain.py
---------------
SSL pretraining using VICReg on sliding-window EEG data.

Loads from {task}_sliding_{release}/ folders created by prepare_ccd_data.py
and prepare_sl_data.py. No labels used — purely unsupervised.

Three encoder options:
    tcnet     — EEGTCNet stem + dilated TCN (0.006M, fast)
    conformer — Conv stem + Transformer (0.4M, best supervised result)
    custom    — Spatial + ResBlocks + TCN (~50K, designed for SSL)

Pretrain data options:
    passive   — RestingState + Movies + SurroundSupp (no task demand)
    active    — CCD + SeqLearning + SymbolSearch (excl. CCD test release)
    all       — passive + active

Usage
-----
    python ssl_pretrain.py --encoder custom --data passive --all_releases
    python ssl_pretrain.py --encoder tcnet  --data all    --all_releases
    python ssl_pretrain.py --encoder custom --data passive --releases R1 --epochs 5
"""

import gc
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from configs.config import (
    WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    N_CHANS, N_TIMES, SFREQ,
    BATCH_SIZE, NUM_WORKERS, SEED,
)
from disk_dataset import load_windowed_dataset
from ssl_encoders import create_encoder
from ssl_vicreg import VICReg, EEGAugmentation
from ssl_byol import BYOL
from src.utils.device import setup_device


RELEASE_MAP = {
    "R1":  "ds005505", "R2":  "ds005506", "R3":  "ds005507",
    "R4":  "ds005508", "R5":  "ds005509", "R6":  "ds005510",
    "R7":  "ds005511", "R8":  "ds005512", "R9":  "ds005514",
    "R10": "ds005515", "R11": "ds005516",
}
SSL_EXCLUDED = {"R5", "R7"}
ALL_RELEASES = [r for r in RELEASE_MAP if r not in SSL_EXCLUDED]

PASSIVE_TASKS = [
    "RestingState",
    "DiaryOfAWimpyKid",
    "DespicableMe",
    "ThePresent",
    "surroundSupp",
]
ACTIVE_TASKS = [
    "contrastChangeDetection",
    "seqLearning8target",
    "seqLearning6target",
    "symbolSearch",
]
ALL_TASKS = PASSIVE_TASKS + ACTIVE_TASKS


def load_ssl_datasets(tasks: list, releases: list) -> ConcatDataset:
    datasets = []
    total    = 0
    for task in tasks:
        for release in releases:
            root = WINDOWS_DIR / f"{task}_sliding_{release}"
            if not root.exists():
                continue
            try:
                ds = load_windowed_dataset(
                    root_dir=root, target_key="target", eager=False
                )
                datasets.append(ds)
                total += len(ds)
                print(f"  + {task}_sliding_{release}: {len(ds):>7,} windows")
            except Exception as e:
                print(f"  [skip] {task}_sliding_{release}: {e}")

    if not datasets:
        raise RuntimeError(
            "No sliding window data found.\n"
            "Run prepare_ccd_data.py --mode sliding and "
            "prepare_sl_data.py --mode sliding first."
        )
    print(f"\n  Total: {total:,} windows across {len(datasets)} datasets")
    return ConcatDataset(datasets)


def pretrain(
    method:       str   = "vicreg",
    encoder_name: str   = "custom",
    data_regime:  str   = "passive",
    releases:     list  = None,
    n_epochs:     int   = 100,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    embed_dim:    int   = 256,
    proj_dim:     int   = 512,
    batch_size:   int   = None,
    save_every:   int   = 10,
    sim_coef:     float = 25.0,
    var_coef:     float = 25.0,
    cov_coef:     float = 1.0,
) -> Path:

    if releases is None:
        releases = ALL_RELEASES
    if batch_size is None:
        batch_size = BATCH_SIZE

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_name   = f"{method}_{encoder_name}_{data_regime}"
    best_ckpt  = WEIGHTS_DIR / f"{run_name}_encoder_best.pt"
    final_ckpt = WEIGHTS_DIR / f"{run_name}_encoder_final.pt"
    log_file   = RESULTS_DIR / f"{run_name}_pretrain_log.csv"

    device = setup_device()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Select tasks
    tasks = {"passive": PASSIVE_TASKS,
             "active":  ACTIVE_TASKS,
             "all":     ALL_TASKS}.get(data_regime)
    if tasks is None:
        raise ValueError(f"Unknown data_regime '{data_regime}'. Choose: passive | active | all")

    print(f"\n{'='*60}")
    print(f"SSL PRETRAINING — VICReg")
    print(f"{'='*60}")
    print(f"  Encoder    : {encoder_name}")
    print(f"  Method     : {method}")
    print(f"  Data       : {data_regime}  ({len(tasks)} tasks)")
    print(f"  Releases   : {releases}")
    print(f"  Epochs     : {n_epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  embed_dim  : {embed_dim}  proj_dim: {proj_dim}")
    print(f"  VICReg     : sim={sim_coef} var={var_coef} cov={cov_coef}")
    print(f"  Run name   : {run_name}")
    print('='*60)

    # ── [1/4] Data ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading data ...")
    combined = load_ssl_datasets(tasks, releases)
    loader   = DataLoader(
        combined, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True, pin_memory=True,
    )
    print(f"  Batches/epoch: {len(loader):,}")

    # ── [2/4] Model ───────────────────────────────────────────────────────────
    print(f"\n[2/4] Building model ...")
    encoder = create_encoder(encoder_name, N_CHANS, N_TIMES, embed_dim)
    aug     = EEGAugmentation(
        n_chans=N_CHANS, n_times=N_TIMES,
        noise_std=0.1, ch_dropout=0.1,
        max_shift=10, scale_range=(0.8, 1.2),
        sfreq=float(SFREQ),
    )

    if method == "vicreg":
        model = VICReg(encoder, aug, proj_dim, sim_coef, var_coef, cov_coef)
    elif method == "byol":
        model = BYOL(encoder, aug, proj_dim=proj_dim)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: vicreg | byol")
    model.to(device)

    enc_p   = sum(p.numel() for p in encoder.parameters())
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Encoder: {enc_p:,} params ({enc_p/1e6:.3f}M)")
    print(f"  Total  : {total_p:,} params ({total_p/1e6:.3f}M)")

    # ── [3/4] Train ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Training ...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(n_epochs - warmup, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_loss = float("inf")

    with open(log_file, "w") as f:
        f.write("epoch,loss,inv,var,cov,lr\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        sums = {"loss": 0.0, "inv": 0.0, "var": 0.0, "cov": 0.0}
        n = 0

        bar = tqdm(loader,
                   desc=f"Epoch {epoch:03d}/{n_epochs} [{encoder_name}|{data_regime}]",
                   leave=False, dynamic_ncols=True)

        for batch in bar:
            x = batch[0].to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            loss, log = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if method == "byol":
                model.update_target()
            for k in sums:
                sums[k] += log[k]
            n += 1
            bar.set_postfix(loss=f"{log['loss']:.3f}",
                            inv=f"{log['inv']:.3f}",
                            var=f"{log['var']:.3f}")

        scheduler.step()
        avgs   = {k: v / max(n, 1) for k, v in sums.items()}
        cur_lr = scheduler.get_last_lr()[0]

        print(f"  Epoch {epoch:03d}  "
              f"loss={avgs['loss']:.4f}  "
              f"inv={avgs['inv']:.4f}  "
              f"var={avgs['var']:.4f}  "
              f"cov={avgs['cov']:.4f}  "
              f"lr={cur_lr:.2e}")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{avgs['loss']:.6f},{avgs['inv']:.6f},"
                    f"{avgs['var']:.6f},{avgs['cov']:.6f},{cur_lr:.2e}\n")

        if avgs["loss"] < best_loss:
            best_loss = avgs["loss"]
            torch.save(encoder.state_dict(), best_ckpt)

        if epoch % save_every == 0:
            ckpt = WEIGHTS_DIR / f"{run_name}_encoder_epoch{epoch:03d}.pt"
            torch.save(encoder.state_dict(), ckpt)
            print(f"  [ckpt] {ckpt.name}")

        gc.collect()

    # ── [4/4] Save ────────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving ...")
    torch.save(encoder.state_dict(), final_ckpt)
    print(f"  Best   -> {best_ckpt}  (loss={best_loss:.4f})")
    print(f"  Final  -> {final_ckpt}")
    print(f"  Log    -> {log_file}")

    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VICReg SSL Pretraining")
    parser.add_argument("--method",      type=str,   default="vicreg",
                        choices=["vicreg", "byol"])
    parser.add_argument("--encoder",     type=str,   default="custom",
                        choices=["tcnet", "conformer", "custom"])
    parser.add_argument("--data",        type=str,   default="passive",
                        choices=["passive", "active", "all"])
    parser.add_argument("--releases",    type=str,   default=None,
                        help="Comma-separated e.g. R1,R2. Default: all non-test.")
    parser.add_argument("--all_releases",action="store_true")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--embed_dim",   type=int,   default=256)
    parser.add_argument("--proj_dim",    type=int,   default=512)
    parser.add_argument("--save_every",  type=int,   default=10)
    parser.add_argument("--sim_coef",    type=float, default=25.0)
    parser.add_argument("--var_coef",    type=float, default=25.0)
    parser.add_argument("--cov_coef",    type=float, default=1.0)
    args = parser.parse_args()

    releases = (ALL_RELEASES if args.all_releases
                else (args.releases.split(",") if args.releases else ALL_RELEASES))

    pretrain(
        method=args.method,
        encoder_name=args.encoder,
        data_regime=args.data,
        releases=releases,
        n_epochs=args.epochs,
        lr=args.lr,
        embed_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        batch_size=args.batch_size,
        save_every=args.save_every,
        sim_coef=args.sim_coef,
        var_coef=args.var_coef,
        cov_coef=args.cov_coef,
    )