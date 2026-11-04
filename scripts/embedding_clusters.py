"""
scripts/embedding_clusters.py
-----------------------------
Do the baseline models' learned embeddings cluster by subject or by RT?

For each baseline model trained by scripts/train.py (weights/{norm}__{model}_weights.pt),
takes the penultimate representation (the input to `final_layer`) for every window
of a split, under the exact normalisation used in training (train.py's fit_norm on
the training releases), and compares it with
  - the same architecture at random init  (what the input alone gives you)
  - log band power of the raw input       (no network at all)

Metrics per representation
--------------------------
  subject_eta2     share of embedding variance explained by subject identity
                   (null ≈ (n_subjects-1)/(n_windows-1))
  subject_knn_acc  10-NN subject identification accuracy, 5-fold, vs chance
  silhouette       silhouette score with subject labels
  rt_r2_cross      ridge R² for RT, GroupKFold by subject — RT info that
                   transfers to new subjects
  rt_r2_within     ridge R² for within-subject RT deviations — trial-level info
Plus, for trained and random-init models:
  pred_r2          the model's own R² on this split (sanity check vs training logs)
  pred_std_ratio   std(pred) / std(target) — near 0 means predictions collapsed to the mean
  pred_subject_eta2 share of prediction variance that is between-subject

Embeddings wider than 512 dims (Deep4Net: 36,400) are reduced with a fixed Gaussian
random projection to 512 dims, then all are standardised and PCA'd to ≤50 dims.

Splits (train.py defaults: train R1-R6,R8-R10 | val R11 | test R5)
------------------------------------------------------------------
  val   R11 — subjects never used for gradient updates          [default]
  test  R5  — NOTE: R5 is also in the training releases used by
              run_ablation*.sh, so these subjects were seen in training

Outputs (--out_dir, default results/embedding_clusters/{norm}_{split}/)
----------------------------------------------------
  tsne_subjects.png   t-SNE per model, trained vs random init, 3 subjects highlighted
  tsne_rt.png         t-SNE per trained model coloured by RT
  metrics.png         the metrics above, trained vs random init vs input
  metrics.csv

Memory: the split's normalised windows are held in RAM
(~100 KB per window; R11 ≈ 2 GB). Use --max_subjects to cap.

Usage
-----
    python scripts/embedding_clusters.py                          # global_zscore, val (R11)
    python scripts/embedding_clusters.py --split val test         # unseen vs seen subjects
    python scripts/embedding_clusters.py --norm robust --models EEGNet EEGConformer
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.config import (
    TASK, WINDOWS_DIR, RESULTS_DIR, WEIGHTS_DIR,
    BATCH_SIZE, NUM_WORKERS, N_CHANS, N_OUTPUTS, N_TIMES, SFREQ,
    SEED, EXCLUDED_SUBJECTS,
)
from src.data.disk_dataset import load_windowed_dataset
from src.models.registry import create_model
from src.utils.device import setup_device
from scripts.train import (
    BASELINE_MODELS, TRAIN_RELEASES, VAL_RELEASES, TEST_RELEASES, FREQ_BANDS,
    NormalisedDataset, fit_norm, fit_subject_references, _sid_list,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RELEASES = {"val": VAL_RELEASES, "test": TEST_RELEASES, "train": TRAIN_RELEASES}

MAX_DIM     = 512     # random-project wider embeddings down to this
PCA_DIMS    = 50
KNN_K       = 10
MIN_WINDOWS = 5       # subjects with fewer windows are dropped from kNN / silhouette
TSNE_MAX    = 6000    # windows shown in t-SNE (whole subjects are kept together)
N_HIGHLIGHT = 3       # subjects coloured in the t-SNE (categorical slots 1-3)

INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_MUTED = "#8f8e88"
GRID      = "#e6e5e0"
BACKGROUND_PTS = "#c9c8c2"
SLOTS     = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3
C_TRAINED = SLOTS[0]
SEQ_BLUE  = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.labelcolor": INK_2, "axes.edgecolor": INK_MUTED, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_releases(releases: List[str], windows_dir: Path):
    datasets = []
    for rel in releases:
        root = windows_dir / f"{TASK}_{rel}"
        if not root.exists():
            print(f"  [skip] {root} not found")
            continue
        ds = load_windowed_dataset(root_dir=root, target_key="target", eager=False,
                                   excluded_subjects=EXCLUDED_SUBJECTS)
        print(f"  {rel}: {len(ds)} windows, {len(ds.get_subject_ids())} subjects")
        datasets.append(ds)
    if not datasets:
        raise RuntimeError(f"No windows found for releases {releases} under {windows_dir}")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def log_band_power(x: np.ndarray) -> np.ndarray:
    """(B, C, T) raw windows -> (B, C * n_bands) log band power."""
    f, p = welch(x, fs=SFREQ, nperseg=min(N_TIMES, int(SFREQ)), axis=-1)
    bp = [p[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in FREQ_BANDS.values()]
    return np.log(np.stack(bp, -1) + 1e-30).reshape(len(x), -1)


def materialise(split_set, norm_params, max_subjects: int, rng: np.random.Generator) -> Dict:
    """Normalised windows, RT and subject IDs for a split, plus input band power."""
    sids = np.array(_sid_list(split_set))
    idx = np.arange(len(sids))
    if max_subjects and len(set(sids)) > max_subjects:
        keep = rng.choice(sorted(set(sids)), max_subjects, replace=False)
        idx = np.where(np.isin(sids, keep))[0]
    norm_ds = Subset(NormalisedDataset(split_set, norm_params, list(sids)), idx)
    raw_ds = Subset(split_set, idx)

    X, Y, BP = [], [], []
    for (xn, yn), (xr, *_) in zip(
        DataLoader(norm_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS),
        DataLoader(raw_ds,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS),
    ):
        X.append(xn)
        Y.append(yn.view(-1))
        BP.append(log_band_power(xr.numpy()))
    y_norm = torch.cat(Y).numpy()
    return {
        "X": torch.cat(X), "y_norm": y_norm,
        "rt": y_norm * (norm_params.rt_std + 1e-8) + norm_params.rt_mean,
        "sids": sids[idx], "bandpower": np.concatenate(BP),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed(model: torch.nn.Module, X: torch.Tensor, device: torch.device) -> tuple:
    """Penultimate features (input to final_layer) and predictions."""
    captured = {}
    hook = model.final_layer.register_forward_pre_hook(
        lambda mod, inp: captured.__setitem__("z", inp[0])
    )
    model.eval()
    proj, feats, preds = None, [], []
    for i in range(0, len(X), BATCH_SIZE):
        out = model(X[i : i + BATCH_SIZE].to(device).float())
        z = captured["z"].flatten(1)
        if z.shape[1] > MAX_DIM:
            if proj is None:
                g = torch.Generator().manual_seed(SEED)
                proj = (torch.randn(z.shape[1], MAX_DIM, generator=g) / MAX_DIM ** 0.5).to(device)
            z = z @ proj
        feats.append(z.cpu())
        preds.append(out.view(-1).cpu())
    hook.remove()
    return torch.cat(feats).numpy(), torch.cat(preds).numpy()


def build_model(name: str, weights: Path | None, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(SEED)
    model = create_model(name, n_chans=N_CHANS, n_outputs=N_OUTPUTS, n_times=N_TIMES, sfreq=SFREQ)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    return model.to(device)


def reduce(F: np.ndarray) -> np.ndarray:
    F = F[:, F.std(0) > 1e-8]                      # drop dead units
    F = StandardScaler().fit_transform(F)
    k = min(PCA_DIMS, F.shape[1], F.shape[0] - 1)
    return PCA(k, random_state=SEED).fit_transform(F)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def eta2(Z: np.ndarray, groups: np.ndarray) -> float:
    """Between-group share of total variance (summed over dims)."""
    Z = Z.reshape(len(Z), -1)
    Zc = Z - Z.mean(0)
    total = (Zc ** 2).sum()
    between = sum((groups == g).sum() * (Zc[groups == g].mean(0) ** 2).sum()
                  for g in np.unique(groups))
    return float(between / total) if total > 0 else float("nan")


def _oof_ridge_r2(Z: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pred = np.empty_like(y)
    for tr, te in GroupKFold(min(5, len(set(groups)))).split(Z, y, groups):
        pred[te] = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Z[tr], y[tr]).predict(Z[te])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _demean_by(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = a.astype(np.float64).copy()
    for g in np.unique(groups):
        out[groups == g] -= out[groups == g].mean(0)
    return out


def cluster_metrics(Z: np.ndarray, rt: np.ndarray, sids: np.ndarray) -> Dict:
    counts = pd.Series(sids).value_counts()
    ok = np.isin(sids, counts[counts >= MIN_WINDOWS].index)
    Zk, sk = Z[ok], sids[ok]

    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Zk, sk):
        acc.append(KNeighborsClassifier(KNN_K).fit(Zk[tr], sk[tr]).score(Zk[te], sk[te]))

    return {
        "subject_eta2":    eta2(Z, sids),
        "subject_knn_acc": float(np.mean(acc)),
        "silhouette":      float(silhouette_score(Zk, sk, sample_size=min(5000, len(Zk)),
                                                  random_state=SEED)),
        "rt_r2_cross":     _oof_ridge_r2(Z, rt, sids),
        "rt_r2_within":    _oof_ridge_r2(_demean_by(Z, sids), _demean_by(rt, sids), sids),
    }


def prediction_metrics(pred: np.ndarray, y: np.ndarray, sids: np.ndarray) -> Dict:
    return {
        "pred_r2":           float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "pred_std_ratio":    float(pred.std() / y.std()),
        "pred_subject_eta2": eta2(pred, sids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def tsne_subset(sids: np.ndarray, rng: np.random.Generator) -> tuple:
    """Whole subjects, in random order, up to TSNE_MAX windows; plus highlighted subjects."""
    counts = pd.Series(sids).value_counts()
    chosen, total = [], 0
    for s in rng.permutation(counts.index.to_numpy()):
        if total >= TSNE_MAX:
            break
        chosen.append(s)
        total += counts[s]
    idx = np.where(np.isin(sids, chosen))[0]
    highlight = counts[chosen].sort_values(ascending=False).index[:N_HIGHLIGHT].tolist()
    return idx, highlight


def run_tsne(Z: np.ndarray) -> np.ndarray:
    return TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(Z)


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def fig_tsne_subjects(emb2d: Dict, sids: np.ndarray, highlight: List[str],
                      rows: Dict, out: Path, title: str) -> None:
    models = list(emb2d["trained"])
    fig, axes = plt.subplots(2, len(models), figsize=(3.1 * len(models), 6.6), squeeze=False)
    is_bg = ~np.isin(sids, highlight)
    for r, kind in enumerate(["trained", "random_init"]):
        for c, m in enumerate(models):
            ax, P = axes[r, c], emb2d[kind][m]
            ax.scatter(P[is_bg, 0], P[is_bg, 1], s=3, color=BACKGROUND_PTS, alpha=0.5, lw=0)
            for s, color in zip(highlight, SLOTS):
                k = sids == s
                ax.scatter(P[k, 0], P[k, 1], s=16, color=color, edgecolor="white", lw=0.6)
            met = rows[(m, kind)]
            ax.set_title(f"{m}\nsubject η² {met['subject_eta2']:.0%} · "
                         f"kNN {met['subject_knn_acc']:.0%}", fontsize=8.5)
            _bare(ax)
        axes[r, 0].set_ylabel("Trained" if kind == "trained" else "Random init",
                              fontsize=10, fontweight="bold", color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=c, mec="white", ms=6, label=s)
               for s, c in zip(highlight, SLOTS)]
    handles.append(Line2D([], [], marker="o", ls="", color=BACKGROUND_PTS, ms=4,
                          label="other subjects"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def fig_tsne_rt(emb2d: Dict, rt: np.ndarray, rows: Dict, out: Path, title: str) -> None:
    models = list(emb2d)
    fig, axes = plt.subplots(1, len(models), figsize=(3.1 * len(models), 3.6), squeeze=False)
    lo, hi = np.percentile(rt, [2, 98])
    for ax, m in zip(axes[0], models):
        P = emb2d[m]
        order = np.argsort(rt)
        sc = ax.scatter(P[order, 0], P[order, 1], c=rt[order], cmap=SEQ_BLUE,
                        vmin=lo, vmax=hi, s=4, lw=0)
        met = rows[(m, "trained")]
        ax.set_title(f"{m}\nRT R² across subjects {met['rt_r2_cross']:.2f}", fontsize=8.5)
        _bare(ax)
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cb.set_label("Response time (s)")
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    fig.savefig(out)
    plt.close(fig)


METRIC_PANELS = [
    ("subject_eta2",    "Embedding variance explained\nby subject", True),
    ("subject_knn_acc", "Subject identified by\n10 nearest neighbours", True),
    ("rt_r2_cross",     "RT decodable across subjects\n(ridge R², GroupKFold)", False),
    ("rt_r2_within",    "Within-subject RT decodable\n(ridge R² on deviations)", False),
]


def fig_metrics(df: pd.DataFrame, chance: Dict, out: Path, title: str) -> None:
    models = [m for m in df["model"].unique() if m != "input"]
    ypos = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 0.55 * len(models) + 1.9), sharey=True)
    for ax, (key, label, pct) in zip(axes, METRIC_PANELS):
        k = 100 if pct else 1
        for y, m in zip(ypos, models):
            t = df[(df.model == m) & (df.representation == "trained")][key].item() * k
            r = df[(df.model == m) & (df.representation == "random_init")][key].item() * k
            ax.plot([r, t], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.plot(r, y, "o", ms=8, mfc="white", mec=INK_2, mew=1.5, zorder=2)
            ax.plot(t, y, "o", ms=8, color=C_TRAINED, mec="white", mew=1.5, zorder=3)
        ref = df[df.representation == "input_bandpower"][key].item() * k
        ax.axvline(ref, color=INK_MUTED, ls="--", lw=1)
        ax.text(ref, 1.0, " input band power", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=INK_2, va="bottom", rotation=0)
        if key in chance:
            ax.axvline(chance[key] * k, color=INK_MUTED, ls=":", lw=1)
            ax.text(chance[key] * k, -0.08, "chance", transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color=INK_2, ha="center", va="top")
        if not pct:
            ax.axvline(0, color=INK_2, lw=0.8)
        ax.set_title(label, loc="left", pad=16)
        ax.set_xlabel("%" if pct else "R²")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, models)
    handles = [
        Line2D([], [], marker="o", ls="", color=C_TRAINED, mec="white", ms=8, label="Trained"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK_2, ms=8, label="Random init"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.06)
    fig.savefig(out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_split(split: str, norm: str, models: List[str], norm_params, windows_dir: Path,
              device: torch.device, max_subjects: int, out_root: Path) -> pd.DataFrame:
    releases = SPLIT_RELEASES[split]
    print(f"\n{'='*70}\n{split} split ({','.join(releases)})  |  norm={norm}\n{'='*70}")
    rng = np.random.default_rng(SEED)
    split_set = load_releases(releases, windows_dir)
    fit_subject_references(norm_params, split_set)   # per-subject refs (riemannian_recentering)
    data = materialise(split_set, norm_params, max_subjects, rng)
    sids, rt, y = data["sids"], data["rt"], data["y_norm"]
    n_subj = len(set(sids))
    print(f"  {len(sids)} windows, {n_subj} subjects")

    counts = pd.Series(sids).value_counts()
    kept = counts[counts >= MIN_WINDOWS]
    chance = {"subject_eta2": (n_subj - 1) / (len(sids) - 1),
              "subject_knn_acc": float(kept.max() / kept.sum())}
    rt_eta2 = eta2(rt, sids)
    print(f"  RT variance between subjects: {rt_eta2:.0%}")

    t_idx, highlight = tsne_subset(sids, rng)
    rows, emb2d = {}, {"trained": {}, "random_init": {}}

    Z = reduce(data["bandpower"])
    rows[("input", "input_bandpower")] = cluster_metrics(Z, rt, sids)

    for m in models:
        weights = WEIGHTS_DIR / f"{norm}__{m}_weights.pt"
        if not weights.exists():
            print(f"  [skip] {weights} not found")
            continue
        for kind, w in [("trained", weights), ("random_init", None)]:
            F, pred = embed(build_model(m, w, device), data["X"], device)
            Z = reduce(F)
            row = {**cluster_metrics(Z, rt, sids), **prediction_metrics(pred, y, sids),
                   "embed_dim": F.shape[1]}
            rows[(m, kind)] = row
            emb2d[kind][m] = run_tsne(Z[t_idx])
            print(f"  {m:<16} {kind:<12} "
                  f"subject η² {row['subject_eta2']:.2f}  kNN {row['subject_knn_acc']:.2f}  "
                  f"RT R² cross {row['rt_r2_cross']:+.3f} within {row['rt_r2_within']:+.3f}  "
                  f"pred R² {row['pred_r2']:+.3f}")

    df = pd.DataFrame([{"model": m, "representation": k, **v} for (m, k), v in rows.items()])
    df.insert(2, "split", split)
    df.insert(3, "n_windows", len(sids))
    df.insert(4, "n_subjects", n_subj)
    df["chance_subject_eta2"] = chance["subject_eta2"]
    df["chance_subject_knn_acc"] = chance["subject_knn_acc"]
    df["rt_subject_eta2"] = rt_eta2

    out_dir = out_root / f"{norm}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    head = (f"{norm} · {split} split ({','.join(releases)}) · {len(sids):,} windows, "
            f"{n_subj} subjects")
    if emb2d["trained"]:
        fig_tsne_subjects(emb2d, sids[t_idx], highlight, rows, out_dir / "tsne_subjects.png",
                          f"Do embeddings cluster by subject? — {head}")
        fig_tsne_rt(emb2d["trained"], rt[t_idx], rows, out_dir / "tsne_rt.png",
                    f"Do embeddings order by RT? — {head}")
        fig_metrics(df, chance, out_dir / "metrics.png",
                    f"Subject vs RT information in penultimate embeddings\n{head} · "
                    f"RT variance between subjects: {rt_eta2:.0%}")
    print(f"  -> {out_dir}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--norm", default="global_zscore",
                   help="Normalisation the weights were trained with (weights/{norm}__{model}_weights.pt)")
    p.add_argument("--models", nargs="+", default=BASELINE_MODELS)
    p.add_argument("--split", nargs="+", default=["val"], choices=list(SPLIT_RELEASES))
    p.add_argument("--max_subjects", type=int, default=0, help="Cap subjects per split (0 = all)")
    p.add_argument("--windows_dir", type=Path, default=WINDOWS_DIR)
    p.add_argument("--out_dir", type=Path, default=RESULTS_DIR / "embedding_clusters")
    args = p.parse_args()

    device = setup_device()
    print(f"[norm] fitting '{args.norm}' on training releases {TRAIN_RELEASES} (as in train.py)")
    norm_params = fit_norm(args.norm, load_releases(TRAIN_RELEASES, args.windows_dir))

    for split in args.split:
        run_split(split, args.norm, args.models, norm_params, args.windows_dir,
                  device, args.max_subjects, args.out_dir)


if __name__ == "__main__":
    main()
