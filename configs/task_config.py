"""Task-specific preprocessing configurations for all HBN tasks.

Each config specifies:
    - whether stimulus / response events are required
    - the regression target field name
    - the MNE annotation anchor used for windowing
    - shift and window length in seconds
    - task type: active | semi_passive | passive
"""

from typing import Dict, Any


# ── Task configuration registry ───────────────────────────────────────────────
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {

    # ── Active tasks (button-press, RT prediction) ────────────────────────────
    "contrastChangeDetection": dict(
        require_stimulus=True,
        require_response=True,
        target_field="rt_from_stimulus",
        anchor="stimulus_anchor",
        shift_after_stim=0.5,
        window_len=2.0,
        task_type="active",
        description="Visual contrast change detection — subject presses button on detection.",
    ),
    "surroundSupp": dict(
        require_stimulus=True,
        require_response=True,
        target_field="rt_from_stimulus",
        anchor="stimulus_anchor",
        shift_after_stim=0.5,
        window_len=2.0,
        task_type="active",
        description="Surround suppression — subject detects grating changes.",
    ),
    "symbolSearch": dict(
        require_stimulus=True,
        require_response=True,
        target_field="rt_from_stimulus",
        anchor="stimulus_anchor",
        shift_after_stim=0.5,
        window_len=2.0,
        task_type="active",
        description="Digitised symbol-search cognitive test with timed responses.",
    ),
    "seqLearning8target": dict(
        require_stimulus=True,
        require_response=True,
        target_field="rt_from_stimulus",
        anchor="stimulus_anchor",
        shift_after_stim=0.5,
        window_len=2.0,
        task_type="active",
        description="Sequential learning (8 targets) — subject responds to target sequences.",
    ),
    "seqLearning6target": dict(
        require_stimulus=True,
        require_response=True,
        target_field="rt_from_stimulus",
        anchor="stimulus_anchor",
        shift_after_stim=0.5,
        window_len=2.0,
        task_type="active",
        description="Sequential learning (6 targets) — subject responds to target sequences.",
    ),

    # ── Semi-passive (stimulus present, no response required) ─────────────────
    "FunwithFractals": dict(
        require_stimulus=True,
        require_response=False,
        target_field=None,           # no RT; use subject-level metadata if needed
        anchor="stimulus_anchor",
        shift_after_stim=0.0,
        window_len=2.0,
        task_type="semi_passive",
        description="Passive visual stimulation with fractal onset events.",
    ),

    # ── Passive tasks (no stimulus / response) ────────────────────────────────
    "RestingState": dict(
        require_stimulus=False,
        require_response=False,
        target_field=None,           # use subject metadata (age, diagnosis, etc.)
        anchor=None,                 # sliding window — no event anchor
        shift_after_stim=0.0,
        window_len=4.0,              # longer windows common for resting-state
        task_type="passive",
        description="Eyes-open/closed resting state — no events.",
    ),
    "DiaryOfAWimpyKid": dict(
        require_stimulus=False,
        require_response=False,
        target_field=None,
        anchor=None,
        shift_after_stim=0.0,
        window_len=4.0,
        task_type="passive",
        description="Naturalistic movie watching — passive viewing.",
    ),
    "DespicableMe": dict(
        require_stimulus=False,
        require_response=False,
        target_field=None,
        anchor=None,
        shift_after_stim=0.0,
        window_len=4.0,
        task_type="passive",
        description="Naturalistic movie watching — passive viewing.",
    ),
    "ThePresent": dict(
        require_stimulus=False,
        require_response=False,
        target_field=None,
        anchor=None,
        shift_after_stim=0.0,
        window_len=4.0,
        task_type="passive",
        description="Naturalistic movie watching — passive viewing.",
    ),
}


def get_task_config(task: str) -> Dict[str, Any]:
    """
    Return the preprocessing config dict for a given task name.

    Parameters
    ----------
    task : str
        Task name (e.g. "contrastChangeDetection", "RestingState").

    Returns
    -------
    dict
        Config with keys: require_stimulus, require_response, target_field,
        anchor, shift_after_stim, window_len, task_type, description.

    Raises
    ------
    ValueError
        If the task name is not recognised.
    """
    if task not in TASK_CONFIGS:
        available = "\n  ".join(sorted(TASK_CONFIGS))
        raise ValueError(
            f"Unknown task: '{task}'.\nAvailable tasks:\n  {available}"
        )
    return TASK_CONFIGS[task]