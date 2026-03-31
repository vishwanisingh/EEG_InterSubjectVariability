"""Subject-level train / validation / test splitting."""

from typing import List, Tuple

from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state


def split_by_subjects(
    windowed_dataset: BaseConcatDataset,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 2025,
    excluded_subjects: List[str] = None,
) -> Tuple[BaseConcatDataset, BaseConcatDataset, BaseConcatDataset]:
    """
    Split a windowed dataset into train / validation / test by subject.

    Splitting at the subject level prevents data leakage — all windows
    from a given subject are confined to a single split.

    Parameters
    ----------
    windowed_dataset : BaseConcatDataset
        Output of :func:`create_windows`.
    valid_frac : float
        Fraction of subjects reserved for validation.
    test_frac : float
        Fraction of subjects reserved for test.
    seed : int
        Random seed for reproducibility.
    excluded_subjects : list of str, optional
        Subject IDs to exclude entirely (e.g. known bad data).

    Returns
    -------
    train_set, valid_set, test_set : BaseConcatDataset
    """
    if excluded_subjects is None:
        excluded_subjects = []

    metadata = windowed_dataset.get_metadata()
    subjects  = metadata["subject"].unique().tolist()
    subjects  = [s for s in subjects if s not in excluded_subjects]

    # First split off the test subjects
    train_valid_subj, test_subj = train_test_split(
        subjects,
        test_size=test_frac,
        random_state=check_random_state(seed),
        shuffle=True,
    )

    # Then split the remainder into train / valid
    # Adjust valid_frac relative to the remaining pool
    adjusted_valid = valid_frac / (1.0 - test_frac)
    train_subj, valid_subj = train_test_split(
        train_valid_subj,
        test_size=adjusted_valid,
        random_state=check_random_state(seed + 1),
        shuffle=True,
    )

    # Sanity check — every subject ends up in exactly one split
    assert set(train_subj) | set(valid_subj) | set(test_subj) == set(subjects)

    subject_split = windowed_dataset.split("subject")

    train_set = BaseConcatDataset([subject_split[s] for s in train_subj if s in subject_split])
    valid_set = BaseConcatDataset([subject_split[s] for s in valid_subj if s in subject_split])
    test_set  = BaseConcatDataset([subject_split[s] for s in test_subj  if s in subject_split])

    print("\nDataset split summary:")
    print(f"  Train : {len(train_set):>6} windows  ({len(train_subj)} subjects)")
    print(f"  Valid : {len(valid_set):>6} windows  ({len(valid_subj)} subjects)")
    print(f"  Test  : {len(test_set):>6} windows  ({len(test_subj)} subjects)")

    return train_set, valid_set, test_set