from __future__ import annotations

import argparse
import csv
import fcntl
import functools
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openpi_client import websocket_client_policy

from causal_failure_predicates import (
    FailureSignature,
    build_causal_units,
    get_goal_predicates,
    infer_failure_signature,
    semantic_quality_for_env,
)
from custom_tasks import registry as custom_tasks
from edd_types import CandidateSlice
from pi05_natural_failure_probe import (
    LIBERO_DUMMY_ACTION,
    Pi05Rollout,
    RuntimeProfile,
    _distance,
    _make_env,
    _policy_observation,
    _select_target_key,
    _semantic_snapshot,
    _set_state_and_obs,
    _state,
    _video_frame_from_obs,
    collect_pi05_rollout,
    find_failure_event,
    minimize_event_slice,
    replay_candidate,
)


PROJECT_ROOT = Path(
    os.environ.get("EDD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_datasets/pi0fast-libero-libero_10/outputs/v4_keyframe_repair_hunt20_20260604"
)
DEFAULT_POLICY_DIR = PROJECT_ROOT / "model_datasets/pi0fast-libero-libero_10/policy_overlay"
DEFAULT_DEMO_ROOT = Path(
    "/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"
)
DEFAULT_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _json_load(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def _parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _case_id(task_id: int, init_state_id: int, seed: int) -> str:
    return f"task{int(task_id):02d}_init{int(init_state_id):02d}_seed{int(seed):02d}"


def _case_seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _failure_signature_from_dict(data: Optional[dict]) -> FailureSignature:
    data = dict(data or {})
    anchor = data.get("anchor_window") or data.get("anchor_interval") or [0, 0]
    return FailureSignature(
        failure_type=str(data.get("failure_type") or "unknown"),
        failed_goal_predicates=tuple(
            str(x) for x in data.get("failed_goal_predicates") or []
        ),
        affected_objects=tuple(str(x) for x in data.get("affected_objects") or []),
        anchor_start=int(anchor[0] if len(anchor) > 0 else 0),
        anchor_end=int(anchor[1] if len(anchor) > 1 else 0),
        semantic_quality=str(data.get("semantic_quality") or "degraded"),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        mechanism=str(data.get("mechanism") or ""),
        evidence=dict(data.get("evidence") or {}),
    )


def _probe_args(
    args: argparse.Namespace,
    *,
    seed: int,
    output: Path,
    gpu_id: Optional[int] = None,
    replay_trials: Optional[int] = None,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.policy_host = str(args.policy_host)
    ns.policy_port = int(args.policy_port)
    ns.policy_config = str(args.policy_config)
    ns.policy_checkpoint = str(args.policy_dir)
    ns.task_suite_name = str(args.task_suite_name)
    ns.task_ids = list(getattr(args, "task_ids", []))
    ns.init_state_ids = list(getattr(args, "init_state_ids", []))
    ns.seed = int(seed)
    ns.gpu_id = int(args.gpu_id if gpu_id is None else gpu_id)
    ns.camera_size = int(args.camera_size)
    ns.resize_size = int(args.resize_size)
    ns.replan_steps = int(args.replan_steps)
    ns.num_steps_wait = int(args.num_steps_wait)
    ns.initial_state_max_attempts = 1
    ns.disable_initial_state_quality_filter = False
    ns.max_steps = int(args.max_steps)
    ns.event_window = int(args.event_window)
    ns.min_distance_delta = float(args.min_distance_delta)
    ns.continuation = "recorded"
    trials = int(replay_trials if replay_trials is not None else args.search_replay_trials)
    ns.replay_trials = trials
    ns.search_replay_trials = int(args.search_replay_trials)
    ns.confirm_replay_trials = int(args.search_confirm_trials)
    ns.repair_replay_trials = int(args.accept_trials)
    ns.same_failure_threshold = float(args.same_failure_threshold)
    ns.accept_same_failure_rate = float(args.accept_same_failure_rate)
    ns.causal_effect_threshold = float(args.causal_effect_threshold)
    ns.causal_chunk_size = int(args.causal_chunk_size)
    ns.causal_ablation_trials = int(args.search_confirm_trials)
    ns.causal_ablation_strategies = "hold"
    ns.causal_context_before = int(args.causal_context_before)
    ns.causal_context_after = int(args.causal_context_after)
    ns.causal_max_units = int(args.causal_max_units)
    ns.disable_replay_cache = False
    ns.replay_evaluation_timeout_seconds = float(args.replay_timeout_seconds)
    ns.verbose_replay_progress = bool(args.verbose_replay_progress)
    ns.progress_log_path = None
    ns.enable_sequential_trial_pruning = False
    ns.disable_hierarchical_causal_pruning = False
    ns.demo_dataset_root = Path(args.demo_dataset_root)
    ns.demo_repair_timeout_seconds = 30.0
    ns.scripted_expert_repair_max_steps = 80
    ns.expert_repair_max_actions_to_store = 128
    ns.skip_source_repair_if_policy_pass = True
    ns.disable_source_repair = True
    ns.defer_source_repair = True
    ns.stop_after_first_repair_valid_core = True
    ns.repair_scheduling_mode = "pass_hunt"
    ns.disable_rule_language_intervention = True
    ns.enable_visual_policy_mask = False
    ns.record_video = False
    ns.video_dir = None
    ns.video_prefix = ""
    ns.video_camera = str(args.video_camera)
    ns.video_fps = int(args.video_fps)
    ns.video_every_n = 1
    ns.video_codec = "libx264"
    ns.video_quality = 10
    ns.video_no_flip = False
    ns.output = Path(output)
    ns._runtime_profile = RuntimeProfile.create()
    ns._replay_evaluation_cache = {}
    return ns


def _planned_cases(args: argparse.Namespace) -> List[Tuple[int, int, int]]:
    cases = [
        (task_id, init_state_id, seed)
        for task_id in args.task_ids
        for init_state_id in args.init_state_ids
        for seed in args.seeds
    ]
    rng = random.Random(int(args.case_order_seed))
    rng.shuffle(cases)
    return cases


def _save_rollout_archive(
    args: argparse.Namespace,
    rollout: Pi05Rollout,
    *,
    event: Optional[Tuple[int, int, float]] = None,
    minimal_slice: Optional[Tuple[int, int]] = None,
) -> Optional[str]:
    if rollout is None or rollout.length <= 0 or not rollout.states_before_action:
        return None
    archive_dir = Path(args.output_dir) / "rollout_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    seed = int(rollout.reset_seed if rollout.reset_seed is not None else 0)
    case_id = _case_id(rollout.task_id, rollout.init_state_id, seed)
    archive_path = archive_dir / f"{case_id}_failure_rollout_archive.npz"
    metadata = {
        "schema_version": "v4-keyframe-hunt-rollout-archive-v1",
        "task_suite_name": rollout.task_suite_name,
        "task_id": int(rollout.task_id),
        "init_state_id": int(rollout.init_state_id),
        "seed": seed,
        "task_language": rollout.task_language,
        "target_key": rollout.target_key,
        "target_key_trace": list(rollout.target_key_trace),
        "distance_trace": [float(x) for x in rollout.distance_trace],
        "success": bool(rollout.success),
        "done_step": rollout.done_step,
        "reset_seed": rollout.reset_seed,
        "length": int(rollout.length),
        "failure_signature": rollout.failure_signature.to_dict(),
        "failure_event": None
        if event is None
        else {"window": [int(event[0]), int(event[1])], "confidence": float(event[2])},
        "minimal_same_failure_slice": None
        if minimal_slice is None
        else [int(minimal_slice[0]), int(minimal_slice[1])],
        "created_for": "v4_keyframe_repair_hunt20_20260604",
    }
    np.savez_compressed(
        archive_path,
        actions=np.asarray(rollout.actions, dtype=np.float32),
        states_before_action=np.asarray(rollout.states_before_action, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    sidecar = archive_path.with_suffix(".json")
    _json_dump(sidecar, metadata)
    rollout.rollout_archive_path = str(archive_path)
    return str(archive_path)


def _load_rollout_archive(
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    archive_path: Path,
) -> Pi05Rollout:
    data = np.load(str(archive_path), allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float32)
    states = [np.asarray(x, dtype=np.float64) for x in data["states_before_action"]]
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    task_suite_name = str(metadata.get("task_suite_name") or args.task_suite_name)
    task_id = int(metadata.get("task_id"))
    init_state_id = int(metadata.get("init_state_id"))
    task_language = str(metadata.get("task_language") or "")
    target_key = str(metadata.get("target_key") or "")
    env, _task_suite, _task = _make_env(probe_args, task_suite_name, task_id)
    snapshots = []
    distances: List[float] = []
    target_key_trace: List[str] = []
    try:
        env.reset()
        predicates = get_goal_predicates(env) + custom_tasks.stage_predicates_for_suite(
            task_suite_name,
            task_id,
        )
        semantic_quality = semantic_quality_for_env(env)
        stage_tracker = custom_tasks.make_stage_tracker(task_suite_name, task_id)
        for t, state in enumerate(states):
            obs = _set_state_and_obs(env, state)
            action = None if t == 0 or t - 1 >= len(actions) else actions[t - 1]
            snapshot = _semantic_snapshot(
                task_suite_name,
                t,
                obs,
                env=env,
                action=action,
                success=False,
                predicates=predicates,
                stage_tracker=stage_tracker,
                task_id=task_id,
            )
            snapshots.append(snapshot)
            selected_target = _select_target_key(
                obs,
                task_language,
                task_suite_name=task_suite_name,
                task_id=task_id,
                snapshot=snapshot,
            )
            target_key_trace.append(selected_target)
            distances.append(float(_distance(obs, target_key or selected_target)))
    finally:
        env.close()
    failure_signature = _failure_signature_from_dict(metadata.get("failure_signature"))
    if not failure_signature.failed_goal_predicates and snapshots:
        failure_signature = infer_failure_signature(
            snapshots,
            predicates=tuple(predicates),
            semantic_quality=semantic_quality,
            event_window=probe_args.event_window,
            task_language=task_language,
        )
    return Pi05Rollout(
        task_suite_name=task_suite_name,
        task_id=task_id,
        init_state_id=init_state_id,
        task_language=task_language,
        target_key=target_key or (target_key_trace[0] if target_key_trace else ""),
        target_key_trace=target_key_trace,
        actions=actions,
        states_before_action=states,
        snapshots=snapshots,
        goal_predicates=tuple(predicates),
        semantic_quality=str(metadata.get("semantic_quality") or semantic_quality),
        failure_signature=failure_signature,
        distance_trace=distances,
        success=bool(metadata.get("success", False)),
        done_step=metadata.get("done_step"),
        initial_state_quality={"valid": True, "reasons": []},
        initial_state_attempt=0,
        reset_seed=int(metadata.get("reset_seed", metadata.get("seed", probe_args.seed))),
        video_path=None,
        video_frames=0,
        rollout_archive_path=str(archive_path),
    )


def _candidate_priority(kind: str) -> int:
    order = {
        "repair_context_start": 0,
        "minimal_start": 1,
        "state_anchor_unit": 2,
        "contact_event": 3,
        "object_movement_event": 4,
        "target_object_hypothesis": 5,
        "gripper_transition": 6,
        "action_chunk": 7,
        "stage_phase_window": 8,
        "goal_predicate_anchor": 9,
        "failure_event": 10,
    }
    return order.get(kind, 20)


def _keyframe_candidate_sort_key(item: dict, minimal_start: int) -> Tuple[int, int, int]:
    return (
        int(item["priority"]),
        abs(int(item["keyframe"]) - int(minimal_start)),
        int(item["keyframe"]),
    )


def _row_keyframe(row: dict) -> int:
    return int(row["keyframe"])


def _row_case_keyframe(row: dict) -> Tuple[str, int]:
    return str(row["case_id"]), int(row["keyframe"])


def _add_keyframe_candidate(
    candidates: Dict[int, dict],
    rollout_length: int,
    step: object,
    *,
    source: str,
    interval: Optional[Tuple[int, int]] = None,
    unit: Optional[dict] = None,
) -> None:
    try:
        keyframe = int(step)
    except Exception:
        return
    if rollout_length <= 0:
        return
    keyframe = max(0, min(rollout_length - 1, keyframe))
    item = candidates.setdefault(
        keyframe,
        {
            "keyframe": int(keyframe),
            "sources": [],
            "intervals": [],
            "units": [],
            "priority": _candidate_priority(source),
        },
    )
    item["priority"] = min(int(item["priority"]), _candidate_priority(source))
    if source not in item["sources"]:
        item["sources"].append(source)
    if interval is not None:
        iv = [int(interval[0]), int(interval[1])]
        if iv not in item["intervals"]:
            item["intervals"].append(iv)
    if unit is not None:
        marker = json.dumps(unit, sort_keys=True, ensure_ascii=False)
        if marker not in {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in item["units"]}:
            item["units"].append(unit)


def _generate_keyframe_candidates(
    args: argparse.Namespace,
    rollout: Pi05Rollout,
    event: Optional[Tuple[int, int, float]],
    final_eval,
) -> List[dict]:
    candidates: Dict[int, dict] = {}
    length = int(rollout.length)
    anchor_start, anchor_end = rollout.failure_signature.anchor_interval()
    if event is not None:
        s, e, _ = event
        for off in (0, -16, -8, -4, 4, 8, 16):
            _add_keyframe_candidate(
                candidates,
                length,
                int(s) + off,
                source="failure_event",
                interval=(int(s), int(e)),
            )
        _add_keyframe_candidate(
            candidates,
            length,
            int(e) - 1,
            source="failure_event",
            interval=(int(s), int(e)),
        )
    if anchor_end > anchor_start:
        for off in (0, -16, -8, -4, 4, 8, 16):
            _add_keyframe_candidate(
                candidates,
                length,
                int(anchor_start) + off,
                source="failure_anchor",
                interval=(int(anchor_start), int(anchor_end)),
            )
        _add_keyframe_candidate(
            candidates,
            length,
            int(anchor_end) - 1,
            source="failure_anchor",
            interval=(int(anchor_start), int(anchor_end)),
        )

    candidate_slice = None
    if final_eval is not None and final_eval.candidate.span_start is not None:
        candidate_slice = final_eval.candidate
        s = int(final_eval.candidate.span_start)
        e = int(final_eval.candidate.span_end)
        for off in (0, -16, -8, -4, 4, 8, 16):
            _add_keyframe_candidate(
                candidates,
                length,
                s + off,
                source="minimal_start",
                interval=(s, e),
            )
        _add_keyframe_candidate(
            candidates,
            length,
            max(s, e - 1),
            source="minimal_end",
            interval=(s, e),
        )
    elif event is not None:
        candidate_slice = CandidateSlice.from_window(
            int(event[0]), int(event[1]), n_steps=length, level="failure_event"
        )

    if candidate_slice is not None:
        units = build_causal_units(
            candidate_slice,
            rollout.actions,
            rollout.snapshots,
            rollout.failure_signature,
            task_language=rollout.task_language,
            chunk_size=int(args.causal_chunk_size),
            max_units=int(args.causal_max_units),
            context_before=int(args.causal_context_before),
            context_after=int(args.causal_context_after),
        )
        for unit in units:
            unit_dict = unit.to_dict()
            s, e = int(unit.interval[0]), int(unit.interval[1])
            _add_keyframe_candidate(
                candidates,
                length,
                s,
                source=str(unit.kind),
                interval=(s, e),
                unit=unit_dict,
            )
            if unit.kind in {"contact_event", "object_movement_event", "gripper_transition"}:
                _add_keyframe_candidate(
                    candidates,
                    length,
                    max(s, e - 1),
                    source=str(unit.kind),
                    interval=(s, e),
                    unit=unit_dict,
                )

    minimal_start = (
        int(final_eval.candidate.span_start)
        if final_eval is not None and final_eval.candidate.span_start is not None
        else int(event[0])
        if event is not None
        else int(anchor_start)
        if anchor_end > anchor_start
        else 0
    )
    ordered = sorted(
        candidates.values(),
        key=functools.partial(
            _keyframe_candidate_sort_key,
            minimal_start=minimal_start,
        ),
    )
    return ordered[: max(1, int(args.max_keyframe_candidates))]


def _success_count(evaluation) -> int:
    return sum(1 for item in evaluation.trial_outcomes if item.get("success") is True)


def _failed_goal_predicates_from_eval(evaluation) -> List[str]:
    counter = Counter()
    for item in evaluation.trial_outcomes:
        for pred in item.get("failed_goal_predicates") or []:
            counter[str(pred)] += 1
    return [pred for pred, _count in counter.most_common()]


def _evaluate_keyframe(
    args: argparse.Namespace,
    rollout: Pi05Rollout,
    keyframe: int,
    client: websocket_client_policy.WebsocketClientPolicy,
    *,
    trials: int,
    output: Path,
    seed: int,
    stage_level: str,
):
    probe_args = _probe_args(args, seed=seed, output=output, replay_trials=trials)
    return replay_candidate(
        probe_args,
        rollout,
        int(keyframe),
        min(int(keyframe) + 1, int(rollout.length)),
        rollout.failure_signature,
        client=client,
        policy_from_step=int(keyframe),
        stage_level=stage_level,
        trials_override=int(trials),
        early_stop_objective="same_failure",
    )


def _accepted_manifest_path(output_dir: Path) -> Path:
    return output_dir / "accepted_cases_manifest.json"


def _accepted_count(output_dir: Path) -> int:
    data = _json_load(_accepted_manifest_path(output_dir), [])
    return len(data) if isinstance(data, list) else 0


def _try_accept_case(args: argparse.Namespace, row: dict) -> bool:
    output_dir = Path(args.output_dir)
    lock_path = output_dir / "accepted_cases_manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        current = _json_load(_accepted_manifest_path(output_dir), [])
        if not isinstance(current, list):
            current = []
        if len(current) >= int(args.accepted_target):
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            return False
        if any(str(item.get("case_id")) == str(row.get("case_id")) for item in current):
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            return False
        current.append(row)
        _json_dump(_accepted_manifest_path(output_dir), current)
        _append_jsonl(output_dir / "accepted_cases_manifest.jsonl", row)
        if len(current) >= int(args.accepted_target):
            (output_dir / "STOP").write_text(
                f"accepted_target_reached {len(current)}\n", encoding="utf-8"
            )
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return True


def _case_result_path(args: argparse.Namespace, case_id: str) -> Path:
    return Path(args.output_dir) / "case_results" / f"gpu{int(args.gpu_id)}" / f"{case_id}.json"


def _natural_failure_case(
    args: argparse.Namespace,
    task_id: int,
    init_state_id: int,
    seed: int,
    client: websocket_client_policy.WebsocketClientPolicy,
) -> dict:
    case_id = _case_id(task_id, init_state_id, seed)
    case_path = _case_result_path(args, case_id)
    if case_path.exists() and not bool(args.force):
        return {"case_id": case_id, "status": "skipped_existing"}

    start_wall = time.perf_counter()
    status = "unknown"
    event = None
    final_eval = None
    archive_path = None
    keyframe_candidates: List[dict] = []
    candidate_results: List[dict] = []
    accepted_row = None
    error = None
    rollout = None
    try:
        probe_args = _probe_args(
            args,
            seed=seed,
            output=case_path,
            replay_trials=int(args.search_replay_trials),
        )
        rollout = collect_pi05_rollout(
            probe_args,
            client,
            str(args.task_suite_name),
            int(task_id),
            int(init_state_id),
        )
        if rollout.success:
            status = "natural_success_skipped"
        elif not rollout.initial_state_valid:
            status = "invalid_initial_state"
        elif rollout.length <= 0:
            status = "empty_failure_rollout"
        else:
            event = find_failure_event(
                rollout,
                int(args.event_window),
                float(args.min_distance_delta),
            )
            if event is not None:
                final_eval, _trace = minimize_event_slice(
                    probe_args,
                    rollout,
                    event,
                    client=client,
                )
            minimal_slice = None
            if final_eval is not None and final_eval.candidate.span_start is not None:
                minimal_slice = (
                    int(final_eval.candidate.span_start),
                    int(final_eval.candidate.span_end),
                )
            archive_path = _save_rollout_archive(
                args,
                rollout,
                event=event,
                minimal_slice=minimal_slice,
            )
            keyframe_candidates = _generate_keyframe_candidates(
                args,
                rollout,
                event,
                final_eval,
            )
            status = "natural_failure_no_repair_keyframe"
            for candidate in keyframe_candidates:
                if (Path(args.output_dir) / "STOP").exists():
                    status = "stopped_after_target"
                    break
                keyframe = int(candidate["keyframe"])
                screen_eval = _evaluate_keyframe(
                    args,
                    rollout,
                    keyframe,
                    client,
                    trials=int(args.candidate_screen_trials),
                    output=case_path,
                    seed=seed,
                    stage_level="keyframe_raw_policy_replan_screen",
                )
                screen_success = _success_count(screen_eval)
                result = {
                    "keyframe": keyframe,
                    "candidate": candidate,
                    "screen_trials": int(screen_eval.executed_trials),
                    "screen_success_count": int(screen_success),
                    "screen_success_rate": float(
                        screen_success / max(1, screen_eval.executed_trials)
                    ),
                }
                if (
                    int(screen_eval.executed_trials) == int(args.candidate_screen_trials)
                    and screen_success == int(args.candidate_screen_trials)
                ):
                    accept_eval = _evaluate_keyframe(
                        args,
                        rollout,
                        keyframe,
                        client,
                        trials=int(args.accept_trials),
                        output=case_path,
                        seed=seed,
                        stage_level="keyframe_raw_policy_replan_accept_k5",
                    )
                    accept_success = _success_count(accept_eval)
                    result.update(
                        {
                            "accept_trials": int(accept_eval.executed_trials),
                            "accept_success_count": int(accept_success),
                            "accept_success_rate": float(
                                accept_success / max(1, accept_eval.executed_trials)
                            ),
                            "accept_trial_outcomes": [
                                dict(item) for item in accept_eval.trial_outcomes
                            ],
                            "accept_failed_goal_predicates": _failed_goal_predicates_from_eval(
                                accept_eval
                            ),
                        }
                    )
                    if (
                        int(accept_eval.executed_trials) == int(args.accept_trials)
                        and accept_success == int(args.accept_trials)
                    ):
                        accepted_row = {
                            "schema_version": "v4-keyframe-repair-hunt20-accepted-v1",
                            "case_id": case_id,
                            "task_suite_name": str(args.task_suite_name),
                            "task_id": int(task_id),
                            "init_state_id": int(init_state_id),
                            "seed": int(seed),
                            "task_language": rollout.task_language,
                            "archive_path": str(archive_path),
                            "archive_source": "natural_failed_rollout_recorded_by_hunt20",
                            "archive_success": bool(rollout.success),
                            "rollout_length": int(rollout.length),
                            "failure_signature": rollout.failure_signature.to_dict(),
                            "failure_event": None
                            if event is None
                            else {
                                "window": [int(event[0]), int(event[1])],
                                "confidence": float(event[2]),
                            },
                            "minimal_same_failure_slice": None
                            if minimal_slice is None
                            else [int(minimal_slice[0]), int(minimal_slice[1])],
                            "found_keyframe": int(keyframe),
                            "found_keyframe_candidate": candidate,
                            "found_keyframe_success_count": int(accept_success),
                            "found_keyframe_success_rate": 1.0,
                            "found_keyframe_trials": int(accept_eval.executed_trials),
                            "strict_acceptance": "raw_policy_replan_from_found_keyframe_k5_all_success",
                            "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        }
                        if _try_accept_case(args, accepted_row):
                            status = "accepted_strict_keyframe_repair"
                        else:
                            status = "strict_keyframe_repair_found_after_target_full"
                        result["accepted"] = bool(status == "accepted_strict_keyframe_repair")
                        candidate_results.append(result)
                        break
                candidate_results.append(result)
    except Exception as exc:
        status = "error"
        error = repr(exc)

    row = {
        "schema_version": "v4-keyframe-repair-hunt20-case-result-v1",
        "case_id": case_id,
        "task_suite_name": str(args.task_suite_name),
        "task_id": int(task_id),
        "init_state_id": int(init_state_id),
        "seed": int(seed),
        "status": status,
        "error": error,
        "archive_path": archive_path,
        "rollout_success": None if rollout is None else bool(rollout.success),
        "rollout_length": None if rollout is None else int(rollout.length),
        "failure_signature": None
        if rollout is None
        else rollout.failure_signature.to_dict(),
        "failure_event": None
        if event is None
        else {"window": [int(event[0]), int(event[1])], "confidence": float(event[2])},
        "minimal_same_failure_slice": None
        if final_eval is None or final_eval.candidate.span_start is None
        else [int(final_eval.candidate.span_start), int(final_eval.candidate.span_end)],
        "minimal_same_failure_rate": None
        if final_eval is None
        else float(final_eval.same_failure_rate),
        "num_keyframe_candidates": int(len(keyframe_candidates)),
        "keyframe_candidates": keyframe_candidates,
        "candidate_results": candidate_results,
        "accepted": accepted_row,
        "elapsed_seconds": float(time.perf_counter() - start_wall),
    }
    _json_dump(case_path, row)
    _append_jsonl(
        Path(args.output_dir) / "hunt_case_results.jsonl",
        {
            "case_id": case_id,
            "task_id": int(task_id),
            "init_state_id": int(init_state_id),
            "seed": int(seed),
            "status": status,
            "archive_path": archive_path,
            "elapsed_seconds": row["elapsed_seconds"],
        },
    )
    return row


def run_hunt_worker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = _planned_cases(args)
    _json_dump(
        output_dir / f"planned_cases_gpu{int(args.gpu_id)}.json",
        [
            {"task_id": t, "init_state_id": i, "seed": s}
            for t, i, s in planned
        ],
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "hunt-worker-dry-run",
                    "gpu_id": int(args.gpu_id),
                    "num_cases": len(planned),
                    "accepted_target": int(args.accepted_target),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    client = websocket_client_policy.WebsocketClientPolicy(
        str(args.policy_host), int(args.policy_port)
    )
    for task_id, init_state_id, seed in planned:
        if _accepted_count(output_dir) >= int(args.accepted_target):
            break
        if (output_dir / "STOP").exists():
            break
        _natural_failure_case(args, task_id, init_state_id, seed, client)


def _sample_keyframes_for_case(row: dict, args: argparse.Namespace) -> List[int]:
    found = int(row["found_keyframe"])
    length = int(row["rollout_length"])
    low = max(0, found - int(args.sweep_window))
    high = min(length - 1, found + int(args.sweep_window))
    population = [x for x in range(low, high + 1) if x != found]
    rng = random.Random(_case_seed(str(row["case_id"]), int(args.sweep_random_seed)))
    sampled = rng.sample(
        population,
        min(max(0, int(args.keyframes_per_case) - 1), len(population)),
    )
    return sorted(set([found] + sampled))


def _trial_rows_from_eval(
    accepted: dict,
    keyframe: int,
    evaluation,
    trial_offset: int = 0,
) -> List[dict]:
    rows = []
    found = int(accepted["found_keyframe"])
    minimal = accepted.get("minimal_same_failure_slice") or [None, None]
    event = (accepted.get("failure_event") or {}).get("window") or [None, None]
    for outcome in evaluation.trial_outcomes:
        rows.append(
            {
                "case_id": accepted["case_id"],
                "task_suite_name": accepted.get("task_suite_name"),
                "task_id": int(accepted["task_id"]),
                "init_state_id": int(accepted["init_state_id"]),
                "seed": int(accepted["seed"]),
                "task_language": accepted.get("task_language"),
                "archive_path": accepted.get("archive_path"),
                "archive_source": accepted.get("archive_source"),
                "found_keyframe": int(found),
                "keyframe": int(keyframe),
                "keyframe_offset_from_found": int(keyframe - found),
                "minimal_start": minimal[0],
                "minimal_end": minimal[1],
                "failure_event_start": event[0],
                "failure_event_end": event[1],
                "trial": int(trial_offset) + int(outcome.get("trial", 0)),
                "success": bool(outcome.get("success")),
                "failure_type": outcome.get("failure_type"),
                "failed_goal_predicates": list(outcome.get("failed_goal_predicates") or []),
                "failed_goal_count": int(outcome.get("failed_goal_count", 0)),
                "goal_progress": int(outcome.get("goal_progress", 0)),
            }
        )
    return rows


def _policy_error_trial_rows(
    accepted: dict,
    keyframe: int,
    trial_offset: int,
    count: int,
    error: object,
) -> List[dict]:
    found = int(accepted["found_keyframe"])
    minimal = accepted.get("minimal_same_failure_slice") or [None, None]
    event = (accepted.get("failure_event") or {}).get("window") or [None, None]
    rows = []
    for idx in range(max(0, int(count))):
        rows.append(
            {
                "case_id": accepted["case_id"],
                "task_suite_name": accepted.get("task_suite_name"),
                "task_id": int(accepted["task_id"]),
                "init_state_id": int(accepted["init_state_id"]),
                "seed": int(accepted["seed"]),
                "task_language": accepted.get("task_language"),
                "archive_path": accepted.get("archive_path"),
                "archive_source": accepted.get("archive_source"),
                "found_keyframe": int(found),
                "keyframe": int(keyframe),
                "keyframe_offset_from_found": int(keyframe - found),
                "minimal_start": minimal[0],
                "minimal_end": minimal[1],
                "failure_event_start": event[0],
                "failure_event_end": event[1],
                "trial": int(trial_offset) + int(idx),
                "success": False,
                "failure_type": "policy_inference_error",
                "failed_goal_predicates": [],
                "failed_goal_count": -1,
                "goal_progress": 0,
                "error": str(error),
            }
        )
    return rows


def _existing_sweep_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        counts[(str(row.get("case_id")), int(row.get("keyframe", -1)))] += 1
    return counts


def run_sweep_shard(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    accepted = _json_load(_accepted_manifest_path(output_dir), [])
    if not isinstance(accepted, list):
        raise RuntimeError("accepted_cases_manifest.json is missing or invalid")
    accepted = accepted[: int(args.accepted_target)]
    shard_cases = [
        row
        for idx, row in enumerate(accepted)
        if idx % int(args.num_shards) == int(args.shard_index)
    ]
    shard_dir = output_dir / "sweep_shards" / f"gpu{int(args.gpu_id)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "sweep-shard-dry-run",
                    "gpu_id": int(args.gpu_id),
                    "num_cases": len(shard_cases),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    client = websocket_client_policy.WebsocketClientPolicy(
        str(args.policy_host), int(args.policy_port)
    )
    results_path = shard_dir / "keyframe_sweep_results.jsonl"
    existing_counts = _existing_sweep_counts(results_path)
    for accepted_row in shard_cases:
        case_id = str(accepted_row["case_id"])
        probe_args = _probe_args(
            args,
            seed=int(accepted_row["seed"]),
            output=shard_dir / f"{case_id}_probe.json",
            replay_trials=int(args.sweep_trials),
        )
        rollout = _load_rollout_archive(
            args,
            probe_args,
            Path(str(accepted_row["archive_path"])),
        )
        keyframes = _sample_keyframes_for_case(accepted_row, args)
        case_manifest = {
            "case_id": case_id,
            "archive_path": accepted_row["archive_path"],
            "found_keyframe": int(accepted_row["found_keyframe"]),
            "keyframes": keyframes,
        }
        _json_dump(shard_dir / "cases" / case_id / "sweep_manifest.json", case_manifest)
        for keyframe in keyframes:
            existing = int(existing_counts[(case_id, int(keyframe))])
            remaining = max(0, int(args.sweep_trials) - existing)
            if remaining <= 0:
                continue
            try:
                evaluation = replay_candidate(
                    probe_args,
                    rollout,
                    int(keyframe),
                    min(int(keyframe) + 1, int(rollout.length)),
                    rollout.failure_signature,
                    client=client,
                    policy_from_step=int(keyframe),
                    stage_level="keyframe_random_sweep_raw_policy_replan_k5",
                    trials_override=int(remaining),
                    early_stop_objective="same_failure",
                )
                rows = _trial_rows_from_eval(
                    accepted_row,
                    int(keyframe),
                    evaluation,
                    trial_offset=existing,
                )
            except Exception as exc:
                rows = _policy_error_trial_rows(
                    accepted_row,
                    int(keyframe),
                    existing,
                    remaining,
                    exc,
                )
                _json_dump(
                    shard_dir / "cases" / case_id / f"keyframe_{int(keyframe):04d}_error.json",
                    {
                        "case_id": case_id,
                        "keyframe": int(keyframe),
                        "trial_offset": int(existing),
                        "remaining_trials": int(remaining),
                        "error": repr(exc),
                    },
                )
            for row in rows:
                _append_jsonl(results_path, row)
                existing_counts[(case_id, int(keyframe))] += 1
        if bool(args.make_videos):
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{case_id}_found_keyframe_raw_repair_review.mp4"
            if video_path.exists() and video_path.stat().st_size > 0:
                continue
            if video_path.exists() and video_path.stat().st_size == 0:
                video_path.unlink()
            try:
                _make_case_video(args, probe_args, rollout, accepted_row, client, video_path)
            except Exception as exc:
                _json_dump(
                    shard_dir / "cases" / case_id / "video_error.json",
                    {"case_id": case_id, "error": repr(exc)},
                )


def _frame_label(frame: np.ndarray, label: str) -> np.ndarray:
    arr = np.ascontiguousarray(frame.copy())
    try:
        import cv2

        cv2.rectangle(arr, (0, 0), (arr.shape[1], 30), (0, 0, 0), thickness=-1)
        cv2.putText(
            arr,
            str(label)[:80],
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    except Exception:
        pass
    return arr


def _recorded_frames(
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    rollout: Pi05Rollout,
    start: int,
) -> List[np.ndarray]:
    env, _suite, _task = _make_env(probe_args, rollout.task_suite_name, rollout.task_id)
    frames: List[np.ndarray] = []
    try:
        env.reset()
        obs = _set_state_and_obs(env, rollout.states_before_action[start])
        frames.append(_video_frame_from_obs(obs, args.video_camera, flip=True))
        for t in range(start, rollout.length):
            obs, _reward, done, _info = env.step(np.asarray(rollout.actions[t]).tolist())
            frames.append(_video_frame_from_obs(obs, args.video_camera, flip=True))
            if done:
                break
    finally:
        env.close()
    return frames


def _repair_frames(
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    rollout: Pi05Rollout,
    start: int,
    client: websocket_client_policy.WebsocketClientPolicy,
) -> List[np.ndarray]:
    env, _suite, _task = _make_env(probe_args, rollout.task_suite_name, rollout.task_id)
    frames: List[np.ndarray] = []
    try:
        env.reset()
        obs = _set_state_and_obs(env, rollout.states_before_action[start])
        frames.append(_video_frame_from_obs(obs, args.video_camera, flip=True))
        action_plan = []
        for _t in range(start, rollout.length):
            if not action_plan:
                element = _policy_observation(obs, rollout.task_language, args.resize_size)
                try:
                    action_chunk = client.infer(element)["actions"]
                except Exception:
                    break
                action_plan.extend(
                    list(np.asarray(action_chunk, dtype=np.float32)[: int(args.replan_steps)])
                )
            action = np.asarray(action_plan.pop(0), dtype=np.float32)
            obs, _reward, done, _info = env.step(action.tolist())
            frames.append(_video_frame_from_obs(obs, args.video_camera, flip=True))
            if done:
                break
    finally:
        env.close()
    return frames


def _make_case_video(
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    rollout: Pi05Rollout,
    accepted: dict,
    client: websocket_client_policy.WebsocketClientPolicy,
    path: Path,
) -> None:
    import imageio.v2 as imageio

    found = int(accepted["found_keyframe"])
    context_start = max(0, found - int(args.video_context_before))
    original = _recorded_frames(args, probe_args, rollout, context_start)
    repair = _repair_frames(args, probe_args, rollout, found, client)
    if not original:
        original = [np.zeros((args.camera_size, args.camera_size, 3), dtype=np.uint8)]
    if not repair:
        repair = [np.zeros_like(original[0])]
    final_panel = [repair[-1]]
    n = max(len(original), len(repair), int(args.video_min_frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=int(args.video_fps),
        codec="libx264",
        quality=8,
        macro_block_size=1,
        output_params=[
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "18",
        ],
    )
    try:
        for idx in range(n):
            left = original[min(idx, len(original) - 1)]
            mid = repair[min(idx, len(repair) - 1)]
            right = final_panel[0]
            frame = np.concatenate(
                [
                    _frame_label(left, "original failed rollout from context"),
                    _frame_label(mid, "raw policy repair from found keyframe"),
                    _frame_label(right, "repair final state"),
                ],
                axis=1,
            )
            writer.append_data(frame)
    finally:
        writer.close()


def _summarize_trials(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["keyframe"]))].append(row)
    summary = []
    for (_case_id, _keyframe), items in sorted(grouped.items()):
        first = items[0]
        success_count = sum(1 for row in items if row.get("success") is True)
        failed_predicates = Counter()
        failure_types = Counter()
        for row in items:
            failure_types[str(row.get("failure_type"))] += 1
            for pred in row.get("failed_goal_predicates") or []:
                failed_predicates[str(pred)] += 1
        executed = len(items)
        summary.append(
            {
                **{
                    key: first.get(key)
                    for key in [
                        "case_id",
                        "task_suite_name",
                        "task_id",
                        "init_state_id",
                        "seed",
                        "task_language",
                        "archive_path",
                        "archive_source",
                        "found_keyframe",
                        "keyframe",
                        "keyframe_offset_from_found",
                        "minimal_start",
                        "minimal_end",
                        "failure_event_start",
                        "failure_event_end",
                    ]
                },
                "planned_trials": int(executed),
                "executed_trials": int(executed),
                "success_count": int(success_count),
                "success_rate": float(success_count / max(1, executed)),
                "failure_type_counts": dict(failure_types),
                "failed_goal_predicate_counts": dict(failed_predicates),
            }
        )
    return summary


def _plot_summary(output_dir: Path, summary: Sequence[dict]) -> None:
    if not summary:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    by_case: Dict[str, List[dict]] = defaultdict(list)
    by_task: Dict[int, List[dict]] = defaultdict(list)
    for row in summary:
        by_case[str(row["case_id"])].append(row)
        by_task[int(row["task_id"])].append(row)

    case_ids = sorted(by_case)
    cols = 4
    rows_n = max(1, math.ceil(len(case_ids) / cols))
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(cols * 4.5, rows_n * 3.0), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for idx, case_id in enumerate(case_ids):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        rows = sorted(by_case[case_id], key=_row_keyframe)
        xs = [int(r["keyframe"]) for r in rows]
        ys = [float(r["success_rate"]) for r in rows]
        found = int(rows[0]["found_keyframe"])
        ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=3)
        ax.axvline(found, color="tab:green", linestyle="--", linewidth=1, label="found")
        if rows[0].get("minimal_start") is not None:
            ax.axvline(int(rows[0]["minimal_start"]), color="tab:red", linestyle=":", linewidth=1)
        if rows[0].get("minimal_end") is not None:
            ax.axvline(int(rows[0]["minimal_end"]), color="tab:red", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{case_id} | task{int(rows[0]['task_id']):02d}", fontsize=8)
        ax.set_xlabel("keyframe step")
        ax.set_ylabel("K=5 success rate")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(plots / "keyframe_success_by_case.png", dpi=180)
    plt.close(fig)

    task_ids = sorted(by_task)
    fig, axes = plt.subplots(
        len(task_ids), 1, figsize=(8.5, max(3, 2.5 * len(task_ids))), squeeze=False
    )
    for idx, task_id in enumerate(task_ids):
        ax = axes[idx][0]
        rows = sorted(by_task[task_id], key=_row_case_keyframe)
        for case_id in sorted({str(r["case_id"]) for r in rows}):
            case_rows = [r for r in rows if str(r["case_id"]) == case_id]
            xs = [int(r["keyframe_offset_from_found"]) for r in case_rows]
            ys = [float(r["success_rate"]) for r in case_rows]
            ax.plot(xs, ys, marker="o", linewidth=1, markersize=2.5, label=case_id)
        ax.axvline(0, color="tab:green", linestyle="--", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"task{task_id:02d}: success vs keyframe offset from found")
        ax.set_xlabel("keyframe - found_keyframe")
        ax.set_ylabel("success rate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "keyframe_success_by_task.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for case_id in case_ids:
        rows = sorted(by_case[case_id], key=_row_keyframe)
        ax.scatter(
            [int(r["keyframe_offset_from_found"]) for r in rows],
            [float(r["success_rate"]) for r in rows],
            s=16,
            alpha=0.75,
            label=case_id,
        )
    ax.axvline(0, color="tab:green", linestyle="--", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("keyframe offset from found keyframe")
    ax.set_ylabel("K=5 success rate")
    ax.set_title("All accepted cases: success distribution around found keyframe")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "keyframe_success_all_cases.png", dpi=180)
    plt.close(fig)


def _classification(case_rows: Sequence[dict]) -> dict:
    found = int(case_rows[0]["found_keyframe"])
    found_rows = [row for row in case_rows if int(row["keyframe"]) == found]
    found_rate = float(found_rows[0]["success_rate"]) if found_rows else 0.0
    others = [row for row in case_rows if int(row["keyframe"]) != found]
    mean_other = float(np.mean([float(row["success_rate"]) for row in others])) if others else 0.0
    full_success_points = [
        int(row["keyframe"]) for row in case_rows if float(row["success_rate"]) >= 1.0
    ]
    positive_points = [
        int(row["keyframe"]) for row in case_rows if float(row["success_rate"]) > 0.0
    ]
    full_ratio = len([x for x in full_success_points if x != found]) / max(1, len(others))
    if found_rate >= 1.0 and mean_other <= 0.25 and full_ratio <= 0.20:
        label = "sharp_keyframe"
    elif found_rate >= 1.0 and mean_other <= 0.65:
        label = "repair_basin"
    else:
        label = "weak_specificity"
    return {
        "classification": label,
        "found_keyframe_success_rate": found_rate,
        "nearby_mean_success_rate_excluding_found": mean_other,
        "num_full_success_keyframes": len(full_success_points),
        "full_success_keyframe_range": None
        if not full_success_points
        else [min(full_success_points), max(full_success_points)],
        "num_positive_keyframes": len(positive_points),
        "positive_keyframe_range": None
        if not positive_points
        else [min(positive_points), max(positive_points)],
        "found_is_earliest_full_success": bool(
            full_success_points and found == min(full_success_points)
        ),
        "found_is_unique_full_success": bool(full_success_points == [found]),
    }


def _write_analysis(output_dir: Path, summary: Sequence[dict], accepted: Sequence[dict]) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    by_case: Dict[str, List[dict]] = defaultdict(list)
    for row in summary:
        by_case[str(row["case_id"])].append(row)
    classifications = {
        case_id: _classification(rows) for case_id, rows in sorted(by_case.items())
    }
    _json_dump(analysis_dir / "keyframe_support_report.json", classifications)
    counts = Counter(item["classification"] for item in classifications.values())
    lines = [
        "# v4 Keyframe Repair Hunt20 + Random Sweep 分析",
        "",
        "## 口径",
        "",
        "- 模型：pi0fast-libero；任务套件：LIBERO-10。",
        "- accepted case 的强证据：同一条自然失败 rollout archive 上，从 v4 proposal 找到的 found keyframe 重新 query 原始 task raw policy，K=5 全部最终成功。",
        "- 随机 keyframe sweep 只读取 accepted archive，不允许 regenerated，因此横向比较的是同一个失败轨迹附近不同起点的可修复性。",
        "- 本实验不把 language / visual / demo repair 计入 keyframe 通过证据。",
        "",
        "## 总体统计",
        "",
        f"- accepted cases: {len(accepted)}",
        f"- evaluated keyframes: {len(summary)}",
        f"- classification: {dict(counts)}",
        "",
        "## 如何解释",
        "",
        "- `sharp_keyframe`：found keyframe K=5 成功，但附近大多数 keyframe 失败，最支持“我们找到了关键帧”。",
        "- `repair_basin`：found keyframe 成功，附近也有一段区域成功，支持“找到了可修复状态区域”，但不支持唯一关键帧。",
        "- `weak_specificity`：附近随机点成功率也很高，说明这个 case 更像任务本身从很多状态都能恢复，不适合作为强关键帧证据。",
        "",
        "## Per-case",
        "",
    ]
    for case_id, rows in sorted(by_case.items()):
        info = classifications[case_id]
        first = rows[0]
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- task: {int(first['task_id']):02d}, init: {int(first['init_state_id']):02d}, seed: {int(first['seed']):02d}",
                f"- found_keyframe: {int(first['found_keyframe'])}",
                f"- minimal_slice: [{first.get('minimal_start')}, {first.get('minimal_end')}]",
                f"- classification: `{info['classification']}`",
                f"- found_keyframe_success_rate: {info['found_keyframe_success_rate']:.2f}",
                f"- nearby_mean_success_rate_excluding_found: {info['nearby_mean_success_rate_excluding_found']:.2f}",
                f"- full_success_keyframes: {info['num_full_success_keyframes']}, range: {info['full_success_keyframe_range']}",
                f"- found_is_earliest_full_success: {info['found_is_earliest_full_success']}",
                f"- found_is_unique_full_success: {info['found_is_unique_full_success']}",
                "",
            ]
        )
    (analysis_dir / "keyframe_support_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def finalize_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows: List[dict] = []
    for path in sorted(output_dir.glob("sweep_shards/gpu*/keyframe_sweep_results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    result_path = output_dir / "keyframe_sweep_results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = _summarize_trials(rows)
    _json_dump(output_dir / "keyframe_success_summary.json", summary)
    csv_path = output_dir / "keyframe_success_summary.csv"
    fields = [
        "case_id",
        "task_id",
        "init_state_id",
        "seed",
        "found_keyframe",
        "keyframe",
        "keyframe_offset_from_found",
        "minimal_start",
        "minimal_end",
        "planned_trials",
        "executed_trials",
        "success_count",
        "success_rate",
        "archive_source",
        "archive_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key) for key in fields})
    _plot_summary(output_dir, summary)
    accepted = _json_load(_accepted_manifest_path(output_dir), [])
    if not isinstance(accepted, list):
        accepted = []
    _write_analysis(output_dir, summary, accepted)
    _json_dump(
        output_dir / "summary.json",
        {
            "schema_version": "v4-keyframe-repair-hunt20-summary-v1",
            "num_accepted_cases": len(accepted),
            "num_sweep_trials": len(rows),
            "num_keyframes": len(summary),
            "trials_per_keyframe": int(args.sweep_trials),
            "success_rate_mean": float(np.mean([r["success_rate"] for r in summary]))
            if summary
            else None,
        },
    )


def finalize_accepted(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    accepted = _json_load(_accepted_manifest_path(output_dir), [])
    if not isinstance(accepted, list):
        accepted = []
    accepted = accepted[: int(args.accepted_target)]
    _json_dump(_accepted_manifest_path(output_dir), accepted)
    (output_dir / "accepted_cases_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    _json_dump(
        output_dir / "accepted_summary.json",
        {
            "schema_version": "v4-keyframe-repair-hunt20-accepted-summary-v1",
            "num_accepted_cases": len(accepted),
            "accepted_target": int(args.accepted_target),
            "case_ids": [row.get("case_id") for row in accepted],
        },
    )


def dry_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _planned_cases(args)
    _json_dump(
        output_dir / "planned_cases_full.json",
        [{"task_id": t, "init_state_id": i, "seed": s} for t, i, s in cases],
    )
    print(
        json.dumps(
            {
                "schema_version": "v4-keyframe-repair-hunt20-dry-run-v1",
                "output_dir": str(output_dir),
                "num_cases": len(cases),
                "tasks": args.task_ids,
                "init_state_ids": args.init_state_ids,
                "seeds": args.seeds,
                "accepted_target": int(args.accepted_target),
                "sweep_trials": int(args.sweep_trials),
                "planned_sweep_trials_after_20": int(
                    args.accepted_target * args.keyframes_per_case * args.sweep_trials
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--init-state-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--case-order-seed", type=int, default=20260604)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8340)
    parser.add_argument("--policy-config", default="pi0_fast_libero")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--demo-dataset-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=32)
    parser.add_argument("--min-distance-delta", type=float, default=0.03)
    parser.add_argument("--same-failure-threshold", type=float, default=0.75)
    parser.add_argument("--accept-same-failure-rate", type=float, default=0.80)
    parser.add_argument("--causal-effect-threshold", type=float, default=0.30)
    parser.add_argument("--causal-chunk-size", type=int, default=5)
    parser.add_argument("--causal-context-before", type=int, default=48)
    parser.add_argument("--causal-context-after", type=int, default=8)
    parser.add_argument("--causal-max-units", type=int, default=32)
    parser.add_argument("--search-replay-trials", type=int, default=1)
    parser.add_argument("--search-confirm-trials", type=int, default=1)
    parser.add_argument("--candidate-screen-trials", type=int, default=1)
    parser.add_argument("--accept-trials", type=int, default=5)
    parser.add_argument("--sweep-trials", type=int, default=5)
    parser.add_argument("--accepted-target", type=int, default=20)
    parser.add_argument("--max-keyframe-candidates", type=int, default=32)
    parser.add_argument("--keyframes-per-case", type=int, default=21)
    parser.add_argument("--sweep-window", type=int, default=64)
    parser.add_argument("--sweep-random-seed", type=int, default=20260604)
    parser.add_argument("--replay-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--verbose-replay-progress", action="store_true")
    parser.add_argument("--video-camera", default="agentview_image")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-context-before", type=int, default=32)
    parser.add_argument("--video-min-frames", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hunt strict raw-policy repair keyframes, then sweep nearby keyframes."
        ,
        fromfile_prefix_chars="@",
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("dry-run", "hunt-worker", "sweep-shard", "finalize-sweep", "finalize-accepted"):
        p = sub.add_parser(name)
        _add_common_args(p)
        if name == "sweep-shard":
            p.add_argument("--shard-index", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
            p.add_argument("--make-videos", action="store_true")
    args = parser.parse_args(argv)
    args.task_ids = _parse_int_list(args.task_ids)
    args.init_state_ids = _parse_int_list(args.init_state_ids)
    args.seeds = _parse_int_list(args.seeds)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.mode == "dry-run":
        dry_run(args)
    elif args.mode == "hunt-worker":
        run_hunt_worker(args)
    elif args.mode == "sweep-shard":
        run_sweep_shard(args)
    elif args.mode == "finalize-sweep":
        finalize_sweep(args)
    elif args.mode == "finalize-accepted":
        finalize_accepted(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
