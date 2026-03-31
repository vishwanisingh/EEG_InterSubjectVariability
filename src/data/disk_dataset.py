"""
src/data/disk_dataset.py
------------------------
Dataset that reads the per-subject .npy / .npz files written by
scripts/prepare_ccd_data.py, WITHOUT loading everything into RAM at once.

Two modes:
  - lazy  (default) : each window is read from disk on __getitem__
  - eager            : all data for the requested subjects is loaded upfront

Normalization options (both default to False — opt-in):
  - normalize_targets : z-score RT per subject using that subject's own
                        mean and std. Forces models to learn trial-level
                        deviations rather than between-subject mean RT.
  - normalize_eeg     : z-score each EEG channel per subject using that
                        subject's mean and std computed across all their
                        windows. Removes amplitude scale differences across
                        subjects and electrodes.

Both normalizations are computed from each subject's OWN data only —
no information leaks across subjects.

On-disk layout (written by prepare_ccd_data.py):
    data/windows/{task}_{release}/
        subjects.txt
        sub-{subject_id}/
            windows.npy     float32  (n_windows, n_chans, n_times)
            metadata.npz    target, rt_from_stimulus, subject, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SubjectWindowDataset(Dataset):
    """
    Lazy per-subject window dataset.

    Reads from a single subject's sub-{id}/ folder.
    Each __getitem__ returns (X, y) where:
        X : float32 tensor  (n_chans, n_times)
        y : float32 tensor  (1,)  — the regression target

    Parameters
    ----------
    subject_dir : Path
        Path to sub-{id}/ directory.
    target_key : str
        Which metadata field to use as regression target.
    normalize_targets : bool
        If True, z-score the target values using this subject's mean and std.
        Skipped if the subject has only one window or zero std.
    normalize_eeg : bool
        If True, z-score each EEG channel using this subject's per-channel
        mean and std computed across all their windows.
    """

    def __init__(
        self,
        subject_dir: Path,
        target_key: str = "target",
        normalize_targets: bool = False,
        normalize_eeg: bool = False,
    ):
        self.subject_dir      = subject_dir
        self.target_key       = target_key
        self.normalize_targets = normalize_targets
        self.normalize_eeg    = normalize_eeg

        windows_path  = subject_dir / "windows.npy"
        metadata_path = subject_dir / "metadata.npz"

        if not windows_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing data in {subject_dir}")

        # Memory-map the windows array — reads from disk on access, no RAM spike
        try:
            self._windows = np.load(windows_path, mmap_mode="r")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load windows.npy in {subject_dir}: {e}"
            ) from e

        if self._windows.ndim != 3 or self._windows.shape[0] == 0:
            raise ValueError(
                f"Invalid windows array shape {self._windows.shape} in {subject_dir}"
            )

        try:
            meta_raw   = np.load(metadata_path, allow_pickle=True)
            self._meta = {k: meta_raw[k].tolist() for k in meta_raw.files}
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load metadata.npz in {subject_dir}: {e}"
            ) from e

        if target_key not in self._meta:
            raise KeyError(
                f"target_key '{target_key}' not found in metadata. "
                f"Available: {list(self._meta.keys())}"
            )

        raw_targets = self._meta[target_key]
        self.n      = len(raw_targets)

        # ── Per-subject target z-scoring ──────────────────────────────────────
        vals = np.array(
            [float(v) for v in raw_targets if v is not None],
            dtype=np.float32,
        )
        if normalize_targets and len(vals) > 1 and vals.std() > 1e-6:
            self._target_mean = float(vals.mean())
            self._target_std  = float(vals.std())
        else:
            self._target_mean = 0.0
            self._target_std  = 1.0   # identity — no normalization

        self._targets = raw_targets

        # ── Per-subject per-channel EEG z-scoring ─────────────────────────────
        # Initialize to None first so object is always in valid state
        # even if the stats computation fails.
        self._eeg_mean = None
        self._eeg_std  = None

        if normalize_eeg:
            # Load full array once for stats, then release.
            # Memory cost: one subject (~30-50 MB) momentarily.
            all_windows = self._windows.copy()        # (N, C, T) — forces load from mmap
            if all_windows.ndim != 3:
                raise ValueError(
                    f"Expected 3D windows array (N, C, T), got shape {all_windows.shape}"
                )
            self._eeg_mean = all_windows.mean(axis=(0, 2))   # (C,)
            self._eeg_std  = all_windows.std(axis=(0, 2))    # (C,)
            # Avoid division by zero for flat/dead channels
            self._eeg_std  = np.where(self._eeg_std < 1e-6, 1.0, self._eeg_std)
            del all_windows

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── EEG ───────────────────────────────────────────────────────────────
        x = self._windows[idx].copy().astype(np.float32)   # (C, T)

        if self.normalize_eeg and self._eeg_mean is not None:
            x = (x - self._eeg_mean[:, None]) / self._eeg_std[:, None]

        x = torch.from_numpy(x)

        # ── Target ────────────────────────────────────────────────────────────
        raw = self._targets[idx]
        y_val = float(raw) if raw is not None else float("nan")

        if self.normalize_targets:
            y_val = (y_val - self._target_mean) / self._target_std

        y = torch.tensor([y_val], dtype=torch.float32)

        # ── Normalization stats (for inverse-transform at eval time) ──────────
        # Always returned so DataLoader collation is consistent.
        # If normalize_targets=False, mean=0 and std=1 -> inverse is identity.
        norm = torch.tensor(
            [self._target_mean, self._target_std], dtype=torch.float32
        )

        return x, y, norm

    def get_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(self._meta)

    @property
    def subject_id(self) -> str:
        return self.subject_dir.name.replace("sub-", "")

    @property
    def target_mean(self) -> float:
        """Subject's raw RT mean (useful for inverse-transforming predictions)."""
        return self._target_mean

    @property
    def target_std(self) -> float:
        """Subject's raw RT std (useful for inverse-transforming predictions)."""
        return self._target_std


class EEGWindowDataset(Dataset):
    """
    Multi-subject dataset built from per-subject disk folders.

    Parameters
    ----------
    root_dir : Path
        The {task}_{release} directory created by prepare_ccd_data.py.
    subjects : list of str, optional
        Subset of subject IDs to include. If None, loads all from subjects.txt.
    target_key : str
        Which metadata column to use as the regression target.
    eager : bool
        If True, load all windows into RAM on construction (faster training,
        higher memory). If False, read from disk on each __getitem__ (lower
        memory).
    excluded_subjects : list of str, optional
        Subject IDs to exclude.
    normalize_targets : bool
        Z-score RT per subject using that subject's own mean/std.
    normalize_eeg : bool
        Z-score each EEG channel per subject using that subject's own stats.
    """

    def __init__(
        self,
        root_dir: Path,
        subjects: Optional[List[str]] = None,
        target_key: str = "target",
        eager: bool = False,
        excluded_subjects: Optional[List[str]] = None,
        normalize_targets: bool = False,
        normalize_eeg: bool = False,
    ):
        self.root_dir         = Path(root_dir)
        self.target_key       = target_key
        self.eager            = eager
        self.normalize_targets = normalize_targets
        self.normalize_eeg    = normalize_eeg
        excluded              = set(excluded_subjects or [])

        # Resolve subject list
        if subjects is None:
            index_path = self.root_dir / "subjects.txt"
            if not index_path.exists():
                raise FileNotFoundError(
                    f"subjects.txt not found in {self.root_dir}. "
                    "Run scripts/prepare_ccd_data.py first."
                )
            subjects = index_path.read_text().strip().splitlines()

        subjects = [s for s in subjects if s not in excluded]

        if not subjects:
            raise ValueError("No subjects remaining after exclusions.")

        # Build per-subject datasets
        self._sub_datasets: List[SubjectWindowDataset] = []
        for sid in subjects:
            sub_dir = self.root_dir / f"sub-{sid}"
            if not sub_dir.exists():
                print(f"  [warn] {sid}: folder not found, skipping")
                continue
            try:
                ds = SubjectWindowDataset(
                    sub_dir,
                    target_key=target_key,
                    normalize_targets=normalize_targets,
                    normalize_eeg=normalize_eeg,
                )
                self._sub_datasets.append(ds)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"  [warn] {sid}: {exc}, skipping")
            except Exception as exc:
                # Unexpected error — print full type for diagnosis
                print(f"  [warn] {sid}: {type(exc).__name__}: {exc}, skipping")

        if not self._sub_datasets:
            raise RuntimeError("No valid subject datasets could be loaded.")

        if normalize_targets:
            n_normalized = sum(
                1 for ds in self._sub_datasets if ds.target_std > 1e-6
            )
            print(f"  [norm] RT z-scored for {n_normalized}/{len(self._sub_datasets)} subjects")
        if normalize_eeg:
            print(f"  [norm] EEG z-scored per channel for {len(self._sub_datasets)} subjects")

        # Build a flat index: (dataset_idx, local_window_idx)
        self._index: List[Tuple[int, int]] = []
        for ds_idx, ds in enumerate(self._sub_datasets):
            for w_idx in range(len(ds)):
                self._index.append((ds_idx, w_idx))

        # Eager mode: load everything into a single tensor
        if eager:
            print(f"  [eager] Loading {len(self._index)} windows into RAM...")
            all_x, all_y, all_norm = [], [], []
            for ds_idx, w_idx in self._index:
                x, y, norm = self._sub_datasets[ds_idx][w_idx]
                all_x.append(x)
                all_y.append(y)
                all_norm.append(norm)
            self._X    = torch.stack(all_x)     # (N, C, T)
            self._Y    = torch.stack(all_y)     # (N, 1)
            self._NORM = torch.stack(all_norm)  # (N, 2)
            print(f"  [eager] Loaded. X={self._X.shape}  Y={self._Y.shape}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.eager:
            return self._X[idx], self._Y[idx], self._NORM[idx]
        ds_idx, w_idx = self._index[idx]
        return self._sub_datasets[ds_idx][w_idx]

    def get_metadata(self) -> pd.DataFrame:
        """Return a combined metadata DataFrame for all subjects."""
        frames = [ds.get_metadata() for ds in self._sub_datasets]
        return pd.concat(frames, ignore_index=True)

    def get_subject_ids(self) -> List[str]:
        return [ds.subject_id for ds in self._sub_datasets]

    def get_normalization_stats(self) -> pd.DataFrame:
        """
        Return a DataFrame of per-subject normalization stats.
        Useful for inverse-transforming predictions back to seconds.
        """
        rows = []
        for ds in self._sub_datasets:
            rows.append({
                "subject":     ds.subject_id,
                "target_mean": ds.target_mean,
                "target_std":  ds.target_std,
            })
        return pd.DataFrame(rows)

    def split_by_subjects(
        self,
        train_subj: List[str],
        valid_subj: List[str],
        test_subj: List[str],
    ) -> Tuple["EEGWindowDataset", "EEGWindowDataset", "EEGWindowDataset"]:
        """
        Return three EEGWindowDataset views for train / valid / test splits.
        No data is copied — each view points to the same on-disk files.
        Normalization settings are preserved in each split.
        """
        def _make(subj_list):
            return EEGWindowDataset(
                root_dir=self.root_dir,
                subjects=subj_list,
                target_key=self.target_key,
                eager=self.eager,
                normalize_targets=self.normalize_targets,
                normalize_eeg=self.normalize_eeg,
            )

        return _make(train_subj), _make(valid_subj), _make(test_subj)


def load_windowed_dataset(
    root_dir: Path,
    subjects: Optional[List[str]] = None,
    target_key: str = "target",
    eager: bool = False,
    excluded_subjects: Optional[List[str]] = None,
    normalize_targets: bool = False,
    normalize_eeg: bool = False,
) -> EEGWindowDataset:
    """
    Convenience function to load the windowed dataset from disk.

    Parameters
    ----------
    root_dir : Path
        The {task}_{release} directory written by prepare_ccd_data.py.
    subjects : list of str, optional
        Load only these subjects. None = all available.
    target_key : str
        Metadata column to use as regression target.
    eager : bool
        Load all data into RAM now (faster training, more memory).
    excluded_subjects : list of str, optional
        Subject IDs to skip.
    normalize_targets : bool
        Z-score RT per subject using that subject's own mean/std.
    normalize_eeg : bool
        Z-score each EEG channel per subject using that subject's own stats.

    Returns
    -------
    EEGWindowDataset
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Windowed dataset not found at {root_dir}.\n"
            "Run  python scripts/prepare_ccd_data.py  first."
        )
    dataset = EEGWindowDataset(
        root_dir=root_dir,
        subjects=subjects,
        target_key=target_key,
        eager=eager,
        excluded_subjects=excluded_subjects,
        normalize_targets=normalize_targets,
        normalize_eeg=normalize_eeg,
    )
    print(
        f"✓ Loaded {len(dataset)} windows from "
        f"{len(dataset.get_subject_ids())} subjects"
        + (" [RT z-scored]" if normalize_targets else "")
        + (" [EEG z-scored]" if normalize_eeg else "")
    )
    return dataset