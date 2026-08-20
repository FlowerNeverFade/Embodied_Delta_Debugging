from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from causal_failure_predicates import GoalPredicate, StateSnapshot

from custom_tasks import long_breakfast_curriculum_v2
from custom_tasks import long_horizon_breakfast_cleanup


_MODULES = (
    long_horizon_breakfast_cleanup,
    long_breakfast_curriculum_v2,
)


def _module(task_suite_name: str):
    name = str(task_suite_name)
    for module in _MODULES:
        if module.is_custom_suite(name):
            return module
    return None


def is_custom_suite(task_suite_name: str) -> bool:
    return _module(task_suite_name) is not None


def make_task_suite(task_suite_name: str):
    module = _module(task_suite_name)
    return None if module is None else module.make_task_suite(task_suite_name)


def custom_bddl_path(task_suite_name: str, task_id: int) -> Optional[Path]:
    module = _module(task_suite_name)
    return None if module is None else module.custom_bddl_path(task_suite_name, task_id)


def deterministic_reset_seed(task_suite_name: str, base_seed: int, init_state_id: int) -> int:
    module = _module(task_suite_name)
    if module is None:
        return int(base_seed)
    return int(module.deterministic_reset_seed(base_seed, init_state_id))


def max_steps_for_suite(task_suite_name: str) -> Optional[int]:
    module = _module(task_suite_name)
    if module is None:
        return None
    return module.max_steps_for_suite(task_suite_name)


def max_steps_for_task(task_suite_name: str, task_id: int) -> Optional[int]:
    module = _module(task_suite_name)
    if module is None:
        return None
    func = getattr(module, "max_steps_for_task", None)
    if func is None:
        return module.max_steps_for_suite(task_suite_name)
    return func(task_suite_name, task_id)


def stage_predicates_for_suite(
    task_suite_name: str, task_id: int = 0
) -> tuple[GoalPredicate, ...]:
    module = _module(task_suite_name)
    if module is None:
        return tuple()
    try:
        return module.stage_predicates_for_suite(task_suite_name, task_id)
    except TypeError:
        return module.stage_predicates_for_suite(task_suite_name)


def make_stage_tracker(
    task_suite_name: str,
    task_id: int = 0,
    snapshot: Optional[StateSnapshot] = None,
):
    module = _module(task_suite_name)
    if module is None:
        return None
    try:
        return module.make_stage_tracker(task_suite_name, task_id, snapshot)
    except TypeError:
        return module.make_stage_tracker(task_suite_name, snapshot)


def augment_snapshot_with_stage_truth(
    task_suite_name: str,
    snapshot: StateSnapshot,
    tracker,
    env,
    task_id: int = 0,
) -> StateSnapshot:
    module = _module(task_suite_name)
    if module is None:
        return snapshot
    try:
        return module.augment_snapshot_with_stage_truth(
            task_suite_name, snapshot, tracker, env, task_id
        )
    except TypeError:
        return module.augment_snapshot_with_stage_truth(
            task_suite_name, snapshot, tracker, env
        )


def stage_trace_from_snapshots(
    task_suite_name: str,
    snapshots: Sequence[StateSnapshot],
    task_id: int = 0,
) -> Optional[dict]:
    module = _module(task_suite_name)
    if module is None:
        return None
    try:
        return module.stage_trace_from_snapshots(snapshots, task_id)
    except TypeError:
        return module.stage_trace_from_snapshots(snapshots)


def stage_summary_from_snapshots(
    task_suite_name: str,
    snapshots: Sequence[StateSnapshot],
    task_id: int = 0,
) -> Optional[dict]:
    module = _module(task_suite_name)
    if module is None:
        return None
    func = getattr(module, "stage_summary_from_snapshots", None)
    if func is not None:
        return func(task_suite_name, snapshots, task_id)
    trace = stage_trace_from_snapshots(task_suite_name, snapshots, task_id)
    if not trace:
        return None
    return {
        "earliest_failed_stage": trace.get("earliest_failed_stage"),
        "stage_progress_count": trace.get("stage_progress_count"),
        "stage_first_completed_step": trace.get("stage_first_completed_step")
        or trace.get("first_completed_step"),
    }


def target_key_for_snapshot(
    task_suite_name: str,
    task_id: int,
    snapshot: Optional[StateSnapshot],
    obs: dict,
    task_language: str = "",
) -> Optional[str]:
    module = _module(task_suite_name)
    if module is None:
        return None
    func = getattr(module, "target_key_for_snapshot", None)
    if func is None:
        return None
    return func(task_suite_name, task_id, snapshot, obs, task_language)


def task_metadata(task_suite_name: str, task_id: Optional[int] = None) -> Optional[dict]:
    module = _module(task_suite_name)
    if module is None:
        return None
    try:
        return module.task_metadata(task_suite_name, task_id)
    except TypeError:
        return module.task_metadata()


def expert_repair_metadata(
    task_suite_name: str,
    task_id: int,
    snapshot: StateSnapshot,
) -> Optional[dict]:
    module = _module(task_suite_name)
    if module is None:
        return None
    func = getattr(module, "expert_repair_metadata", None)
    if func is None:
        return None
    return func(task_suite_name, task_id, snapshot)


def expert_action(
    task_suite_name: str,
    controller_state: dict,
    env,
    obs: dict,
    task_id: int,
    snapshot: StateSnapshot,
):
    module = _module(task_suite_name)
    if module is None:
        return None
    func = getattr(module, "expert_action", None)
    if func is None:
        return None
    return func(controller_state, env, obs, task_id, snapshot)


def initial_state_quality(
    task_suite_name: str,
    task_id: int,
    obs: dict,
    env,
    snapshot: Optional[StateSnapshot] = None,
) -> dict:
    module = _module(task_suite_name)
    if module is None:
        return {"valid": True, "reasons": [], "semantic_quality": "not_custom"}
    func = getattr(module, "initial_state_quality", None)
    if func is None:
        return {"valid": True, "reasons": [], "semantic_quality": "not_checked"}
    return func(task_suite_name, task_id, obs, env, snapshot)
