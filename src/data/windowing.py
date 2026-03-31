"""Window creation utilities."""

from typing import Tuple

from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import create_windows_from_events
from eegdash.hbn.windows import add_extras_columns, keep_only_recordings_with


def create_windows(
    dataset: BaseConcatDataset,
    anchor: str = "stimulus_anchor",
    shift_after_stim: float = 0.5,
    window_len: float = 2.0,
    epoch_len_s: float = 2.0,
    sfreq: int = 100,
    preload: bool = True,
    metadata_keys: Tuple[str, ...] = (
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type",
    ),
) -> BaseConcatDataset:
    """
    Cut stimulus-locked windows from an annotated dataset.

    Parameters
    ----------
    dataset : BaseConcatDataset
        Dataset that has been annotated by :func:`preprocess_dataset`.
    anchor : str
        MNE annotation marker to align windows to.
    shift_after_stim : float
        Seconds after the anchor to start each window (avoids artifact).
    window_len : float
        Window length in seconds.
    epoch_len_s : float
        Trial epoch length in seconds (usually == window_len).
    sfreq : int
        Sampling frequency in Hz.
    preload : bool
        If True, load all windows into RAM immediately.
    metadata_keys : tuple of str
        Annotation fields to attach as metadata columns.

    Returns
    -------
    BaseConcatDataset
        Windowed dataset where each item is (channels × time_samples).
    """
    # Drop recordings that don't contain the anchor event
    dataset_filtered = keep_only_recordings_with(anchor, dataset)

    start_samples = int(shift_after_stim * sfreq)
    stop_samples  = int((shift_after_stim + window_len) * sfreq)

    windows = create_windows_from_events(
        dataset_filtered,
        mapping={anchor: 0},
        trial_start_offset_samples=start_samples,
        trial_stop_offset_samples=stop_samples,
        window_size_samples=int(epoch_len_s * sfreq),
        window_stride_samples=sfreq,
        preload=preload,
    )

    windows = add_extras_columns(
        windows,
        dataset_filtered,
        desc=anchor,
        keys=metadata_keys,
    )

    return windows