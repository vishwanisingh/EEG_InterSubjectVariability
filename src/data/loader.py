"""Dataset loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional

from braindecode.preprocessing import preprocess, Preprocessor
from eegdash.dataset import EEGChallengeDataset, EEGDashDataset
from eegdash.hbn.windows import annotate_trials_with_target, add_aux_anchors


def load_eegdash_dataset(
    dataset_id: str,
    cache_dir: Path,
    task: Optional[str] = None,
    subject: Optional[str | list[str]] = None,
    session: Optional[str | list[str]] = None,
    run: Optional[str | list[str]] = None,
    download: bool = True,
    **kwargs,
) -> EEGDashDataset:
    """
    Load a general EEGDash dataset from OpenNeuro.

    Parameters
    ----------
    dataset_id : str
        OpenNeuro dataset identifier (e.g. "ds005509").
    cache_dir : Path
        Local directory for cached data.
    task : str, optional
        Task name filter (e.g. "contrastChangeDetection").
    subject : str or list of str, optional
        Subject identifier(s) to filter by.
    session : str or list of str, optional
        Session identifier(s) to filter by.
    run : str or list of str, optional
        Run identifier(s) to filter by.
    download : bool
        If False, load from local BIDS files only (offline mode).
    **kwargs
        Extra keyword arguments forwarded to EEGDashDataset.

    Returns
    -------
    EEGDashDataset
    """
    dataset_kwargs: dict = {
        "dataset": dataset_id,
        "cache_dir": cache_dir,
        "download": download,
    }

    if task is not None:
        dataset_kwargs["task"] = task
    if subject is not None:
        dataset_kwargs["subject"] = subject
    if session is not None:
        dataset_kwargs["session"] = session
    if run is not None:
        dataset_kwargs["run"] = run

    dataset_kwargs.update(kwargs)
    return EEGDashDataset(**dataset_kwargs)


def load_challenge_dataset(
    release: str,
    cache_dir: Path,
    mini: bool = False,
    task: Optional[str] = None,
) -> EEGChallengeDataset:
    """
    Load the EEG 2025 Challenge (preprocessed competition) dataset.

    Parameters
    ----------
    release : str
        Release identifier, e.g. "R5".
    cache_dir : Path
        Local directory for cached data.
    mini : bool
        If True, load the official mini-subset of subjects.
    task : str, optional
        Task name filter.

    Returns
    -------
    EEGChallengeDataset
    """
    kwargs: dict = {"release": release, "cache_dir": cache_dir, "mini": mini}
    if task is not None:
        kwargs["task"] = task
    return EEGChallengeDataset(**kwargs)


def preprocess_dataset(
    dataset: EEGDashDataset,
    epoch_len_s: float = 2.0,
    target_field: Optional[str] = "rt_from_stimulus",
    require_stimulus: bool = True,
    require_response: bool = True,
    n_jobs: int = 1,
) -> EEGDashDataset:
    """
    Annotate trials with RT targets and add windowing anchors.

    This must be called before :func:`create_windows`.

    Parameters
    ----------
    dataset : EEGDashDataset
        Raw loaded dataset.
    epoch_len_s : float
        Trial epoch length in seconds.
    target_field : str, optional
        Field to use as regression target (e.g. "rt_from_stimulus").
        Pass None for passive tasks.
    require_stimulus : bool
        Drop trials without a stimulus event.
    require_response : bool
        Drop trials without a response event.
    n_jobs : int
        Number of parallel preprocessing workers.

    Returns
    -------
    EEGDashDataset
        Dataset with stimulus anchors and target annotations added in-place.
    """
    transforms = [
        Preprocessor(
            annotate_trials_with_target,
            target_field=target_field,
            epoch_length=epoch_len_s,
            require_stimulus=require_stimulus,
            require_response=require_response,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    preprocess(dataset, transforms, n_jobs=n_jobs)
    return dataset