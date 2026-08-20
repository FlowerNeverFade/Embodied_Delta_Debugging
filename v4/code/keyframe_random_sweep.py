from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openpi_client import websocket_client_policy

from causal_failure_predicates import (
    FailureSignature,
    get_goal_predicates,
    infer_failure_signature,
    semantic_quality_for_env,
)
from custom_tasks import registry as custom_tasks
from pi05_natural_failure_probe import (
    Pi05Rollout,
    RuntimeProfile,
    _distance,
    _make_env,
    _policy_observation,  # imported to keep py_compile honest if helper API changes
    _select_target_key,
    _semantic_snapshot,
    _set_state_and_obs,
    collect_pi05_rollout,
    replay_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "model_datasets/pi0fast-libero-libero_10/outputs/showcase_strict_success_by_task_20260531/manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_datasets/pi0fast-libero-libero_10/outputs/keyframe_random_sweep_pi0fast_libero10_20260603"
)


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_int_pair(value: object, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, dict):
        value = value.get("interval")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return default


def _case_seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _failure_signature_from_dict(data: Optional[dict]) -> Optional[FailureSignature]:
    if not isinstance(data, dict):
        return None
    anchor = data.get("anchor_window") or data.get("anchor_interval") or [0, 0]
    return FailureSignature(
        failure_type=str(data.get("failure_type") or "unknown"),
        failed_goal_predicates=tuple(str(x) for x in data.get("failed_goal_predicates") or []),
        affected_objects=tuple(str(x) for x in data.get("affected_objects") or []),
        anchor_start=int(anchor[0] if len(anchor) > 0 else 0),
        anchor_end=int(anchor[1] if len(anchor) > 1 else 0),
        semantic_quality=str(data.get("semantic_quality") or "degraded"),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        mechanism=str(data.get("mechanism") or ""),
        evidence=dict(data.get("evidence") or {}),
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _translated_existing_path(raw: object) -> Optional[Path]:
    if not raw:
        return None
    path = Path(str(raw))
    if path.exists():
        return path
    text = str(path)
    replacements = [
        (
            "/data2/yanghaoyun/research/Embodied_Delta_Debugging",
            str(PROJECT_ROOT),
        ),
    ]
    for old, new in replacements:
        if text.startswith(old):
            candidate = Path(new + text[len(old) :])
            if candidate.exists():
                return candidate
    return None


def _load_case_rows(manifest_path: Path, case_ids: Sequence[str]) -> List[dict]:
    manifest = _load_json(manifest_path)
    rows = list(manifest.get("cases") or [])
    wanted = {str(x) for x in case_ids if str(x)}
    if wanted:
        rows = [row for row in rows if str(row.get("case_id")) in wanted]
    rows.sort(key=lambda row: str(row.get("case_id")))
    return rows


def _probe_args(args: argparse.Namespace, *, seed: int, output: Path) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.policy_host = args.policy_host
    ns.policy_port = int(args.policy_port)
    ns.policy_config = args.policy_config
    ns.policy_checkpoint = args.policy_dir
    ns.task_suite_name = args.task_suite_name
    ns.seed = int(seed)
    ns.gpu_id = int(args.gpu_id)
    ns.camera_size = int(args.camera_size)
    ns.resize_size = int(args.resize_size)
    ns.replan_steps = int(args.replan_steps)
    ns.num_steps_wait = int(args.num_steps_wait)
    ns.max_steps = int(args.max_steps)
    ns.event_window = int(args.event_window)
    ns.min_distance_delta = 0.03
    ns.continuation = "recorded"
    ns.replay_trials = int(args.trials)
    ns.search_replay_trials = int(args.trials)
    ns.confirm_replay_trials = int(args.trials)
    ns.repair_replay_trials = int(args.trials)
    ns.same_failure_threshold = 0.75
    ns.accept_same_failure_rate = 0.80
    ns.causal_effect_threshold = 0.30
    ns.causal_chunk_size = 5
    ns.causal_ablation_trials = int(args.trials)
    ns.causal_ablation_strategies = "hold,adjacent,gripper_correction"
    ns.causal_context_before = 48
    ns.causal_context_after = 8
    ns.causal_max_units = 18
    ns.replay_evaluation_timeout_seconds = float(args.replay_timeout_seconds)
    ns.verbose_replay_progress = bool(args.verbose_replay_progress)
    ns.progress_log_path = None
    ns.enable_sequential_trial_pruning = False
    ns.disable_hierarchical_causal_pruning = False
    ns.demo_dataset_root = args.demo_dataset_root
    ns.demo_repair_timeout_seconds = 30.0
    ns.initial_state_max_attempts = 1
    ns.disable_initial_state_quality_filter = False
    ns.scripted_expert_repair_max_steps = 80
    ns.skip_source_repair_if_policy_pass = True
    ns.stop_after_first_repair_valid_core = False
    ns.disable_source_repair = True
    ns.defer_source_repair = True
    ns.disable_rule_language_intervention = True
    ns.enable_visual_policy_mask = False
    ns.repair_scheduling_mode = "pass_hunt"
    ns.record_video = False
    ns.video_dir = None
    ns.video_prefix = ""
    ns.video_camera = "agentview_image"
    ns.video_fps = 30
    ns.video_every_n = 1
    ns.video_codec = "libx264"
    ns.video_quality = 10
    ns.video_no_flip = False
    ns.output = output
    ns._runtime_profile = RuntimeProfile.create()
    ns._replay_evaluation_cache = {}
    return ns


def _reconstruct_rollout_from_archive(
    args: argparse.Namespace,
    probe_args: argparse.Namespace,
    case: dict,
    case_review: dict,
    report: dict,
    archive_path: Path,
) -> Pi05Rollout:
    data = np.load(str(archive_path), allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float32)
    states = [np.asarray(x, dtype=np.float64) for x in data["states_before_action"]]
    metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    task_suite_name = str(metadata.get("task_suite_name") or args.task_suite_name)
    task_id = int(metadata.get("task_id", case.get("task_id")))
    init_state_id = int(metadata.get("init_state_id", case.get("init_state_id", 0)))
    task_language = str(metadata.get("task_language") or case_review.get("task_language") or "")
    target_key = str(metadata.get("target_key") or (report.get("selected_failed_rollout") or {}).get("target_key") or "")
    if not target_key:
        target_key = "robot0_eef_pos"

    env, _task_suite, _task = _make_env(probe_args, task_suite_name, task_id)
    snapshots = []
    distances = []
    target_key_trace = []
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
            snapshot = _semantic_snapshot(
                task_suite_name,
                t,
                obs,
                env=env,
                action=None if t == 0 or t - 1 >= len(actions) else actions[t - 1],
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

    failure_signature = _failure_signature_from_dict(report.get("original_failure_signature"))
    if failure_signature is None:
        failure_signature = infer_failure_signature(
            snapshots,
            predicates=predicates,
            semantic_quality=semantic_quality,
            event_window=probe_args.event_window,
            task_language=task_language,
        )
    selected = report.get("selected_failed_rollout") or {}
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
        semantic_quality=str(selected.get("semantic_quality") or semantic_quality),
        failure_signature=failure_signature,
        distance_trace=distances,
        success=bool(selected.get("success", metadata.get("success", False))),
        done_step=selected.get("done_step"),
        initial_state_quality=dict(selected.get("initial_state_quality") or {}),
        initial_state_attempt=int(selected.get("initial_state_attempt", 0) or 0),
        reset_seed=int(selected.get("reset_seed", metadata.get("reset_seed", probe_args.seed)) or probe_args.seed),
        video_path=selected.get("video_path"),
        video_frames=int(selected.get("video_frames", 0) or 0),
        rollout_archive_path=str(archive_path),
    )


def _load_or_regenerate_rollout(
    args: argparse.Namespace,
    case: dict,
    client: websocket_client_policy.WebsocketClientPolicy,
    case_dir: Path,
) -> Tuple[Pi05Rollout, dict]:
    case_review_path = _translated_existing_path(case["case_review_path"]) or Path(
        str(case["case_review_path"])
    )
    case_review = _load_json(case_review_path)
    report_path = _translated_existing_path(case_review.get("report_path"))
    report = _load_json(report_path) if report_path and report_path.exists() else {}
    seed = int(case_review.get("seed", case.get("seed", 7)))
    probe_args = _probe_args(args, seed=seed, output=case_dir / "regenerated_rollout.json")

    selected = report.get("selected_failed_rollout") or {}
    archive_path = _translated_existing_path(selected.get("rollout_archive_path"))
    if archive_path is not None:
        rollout = _reconstruct_rollout_from_archive(
            args,
            probe_args,
            case,
            case_review,
            report,
            archive_path,
        )
        return rollout, {
            "archive_source": "report_rollout_archive",
            "archive_path": str(archive_path),
            "report_path": None if report_path is None else str(report_path),
        }

    rollout = collect_pi05_rollout(
        probe_args,
        client,
        args.task_suite_name,
        int(case_review.get("task_id", case.get("task_id"))),
        int(case_review.get("init_state_id", case.get("init_state_id", 0))),
    )
    saved = _save_rollout_archive_local(probe_args, rollout)
    return rollout, {
        "archive_source": "regenerated",
        "archive_path": saved,
        "report_path": None if report_path is None else str(report_path),
        "regenerated_success": bool(rollout.success),
        "regenerated_length": int(rollout.length),
    }


def _save_rollout_archive_local(probe_args: argparse.Namespace, rollout: Pi05Rollout) -> Optional[str]:
    if rollout is None or rollout.length <= 0 or not rollout.states_before_action:
        return None
    archive_dir = Path(probe_args.output).parent / "rollout_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / (
        "task%02d_init%02d_seed%02d_rollout_archive.npz"
        % (
            int(rollout.task_id),
            int(rollout.init_state_id),
            int(rollout.reset_seed if rollout.reset_seed is not None else probe_args.seed),
        )
    )
    metadata = {
        "schema_version": "keyframe-sweep-rollout-archive-v1",
        "task_suite_name": rollout.task_suite_name,
        "task_id": int(rollout.task_id),
        "init_state_id": int(rollout.init_state_id),
        "task_language": rollout.task_language,
        "target_key": rollout.target_key,
        "reset_seed": None if rollout.reset_seed is None else int(rollout.reset_seed),
        "success": bool(rollout.success),
        "length": int(rollout.length),
    }
    np.savez_compressed(
        archive_path,
        actions=np.asarray(rollout.actions, dtype=np.float32),
        states_before_action=np.asarray(rollout.states_before_action, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    rollout.rollout_archive_path = str(archive_path)
    return str(archive_path)


def _sample_keyframes(
    case_id: str,
    rollout_length: int,
    repair_interval: Tuple[int, int],
    keyframes_per_case: int,
    window_before: int,
    window_after: int,
    random_seed: int,
) -> List[int]:
    a, b = repair_interval
    if rollout_length <= 0:
        return []
    low = max(0, int(a) - int(window_before))
    high = min(rollout_length - 1, int(b) + int(window_after))
    if high < low:
        low, high = 0, rollout_length - 1
    required = [min(max(0, int(a)), rollout_length - 1)]
    population = [x for x in range(low, high + 1) if x not in required]
    rng = random.Random(_case_seed(case_id, random_seed))
    n_random = max(0, int(keyframes_per_case) - len(required))
    sampled = rng.sample(population, min(n_random, len(population))) if population else []
    return sorted(set(required + sampled))


def _trial_rows_from_evaluation(
    case: dict,
    case_review: dict,
    keyframe: int,
    repair_interval: Tuple[int, int],
    minimal_interval: Tuple[int, int],
    archive_meta: dict,
    evaluation,
) -> List[dict]:
    rows = []
    for outcome in evaluation.trial_outcomes:
        rows.append(
            {
                "case_id": case["case_id"],
                "task_id": int(case["task_id"]),
                "init_state_id": int(case_review.get("init_state_id", case.get("init_state_id", 0))),
                "seed": int(case_review.get("seed", case.get("seed", 0))),
                "task_language": case_review.get("task_language"),
                "evidence_level": case.get("evidence_level"),
                "keyframe": int(keyframe),
                "keyframe_offset_from_repair_start": int(keyframe - repair_interval[0]),
                "repair_context_start": int(repair_interval[0]),
                "repair_context_end": int(repair_interval[1]),
                "minimal_start": int(minimal_interval[0]),
                "minimal_end": int(minimal_interval[1]),
                "trial": int(outcome.get("trial", 0)),
                "success": bool(outcome.get("success")),
                "failure_type": outcome.get("failure_type"),
                "failed_goal_predicates": list(outcome.get("failed_goal_predicates") or []),
                "failed_goal_count": int(outcome.get("failed_goal_count", 0)),
                "goal_progress": int(outcome.get("goal_progress", 0)),
                "archive_source": archive_meta.get("archive_source"),
                "archive_path": archive_meta.get("archive_path"),
            }
        )
    return rows


def _summarize_trials(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), int(row["keyframe"]))].append(row)
    summary = []
    for (_case_id, _keyframe), items in sorted(grouped.items()):
        first = items[0]
        success_count = sum(1 for row in items if row.get("success") is True)
        failure_types = Counter(str(row.get("failure_type")) for row in items)
        failed_predicates = Counter()
        for row in items:
            for pred in row.get("failed_goal_predicates") or []:
                failed_predicates[str(pred)] += 1
        executed = len(items)
        summary.append(
            {
                **{
                    key: first.get(key)
                    for key in [
                        "case_id",
                        "task_id",
                        "init_state_id",
                        "seed",
                        "task_language",
                        "evidence_level",
                        "keyframe",
                        "keyframe_offset_from_repair_start",
                        "repair_context_start",
                        "repair_context_end",
                        "minimal_start",
                        "minimal_end",
                        "archive_source",
                        "archive_path",
                    ]
                },
                "planned_trials": executed,
                "executed_trials": executed,
                "success_count": int(success_count),
                "success_rate": float(success_count / max(1, executed)),
                "failure_type_counts": dict(failure_types),
                "failed_goal_predicate_counts": dict(failed_predicates),
            }
        )
    return summary


def _write_summary(output_dir: Path, trial_rows: Sequence[dict]) -> List[dict]:
    summary = _summarize_trials(trial_rows)
    _json_dump(output_dir / "keyframe_success_summary.json", summary)
    csv_path = output_dir / "keyframe_success_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "task_id",
        "init_state_id",
        "seed",
        "evidence_level",
        "keyframe",
        "keyframe_offset_from_repair_start",
        "repair_context_start",
        "repair_context_end",
        "minimal_start",
        "minimal_end",
        "planned_trials",
        "executed_trials",
        "success_count",
        "success_rate",
        "archive_source",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row.get(key) for key in fields})
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
    cols = 3
    rows_n = math.ceil(len(case_ids) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 5.0, rows_n * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, case_id in enumerate(case_ids):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        rows = sorted(by_case[case_id], key=lambda x: int(x["keyframe"]))
        xs = [int(r["keyframe"]) for r in rows]
        ys = [float(r["success_rate"]) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3)
        repair_start = int(rows[0]["repair_context_start"])
        minimal_start = int(rows[0]["minimal_start"])
        minimal_end = int(rows[0]["minimal_end"])
        ax.axvline(repair_start, color="tab:green", linestyle="--", linewidth=1, label="repair start")
        ax.axvline(minimal_start, color="tab:red", linestyle=":", linewidth=1, label="minimal")
        ax.axvline(minimal_end, color="tab:red", linestyle=":", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{case_id} | task{int(rows[0]['task_id']):02d}", fontsize=9)
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
    fig, axes = plt.subplots(len(task_ids), 1, figsize=(8, max(3, 2.4 * len(task_ids))), squeeze=False)
    for idx, task_id in enumerate(task_ids):
        ax = axes[idx][0]
        rows = sorted(by_task[task_id], key=lambda x: (str(x["case_id"]), int(x["keyframe"])))
        for case_id in sorted({str(r["case_id"]) for r in rows}):
            case_rows = [r for r in rows if str(r["case_id"]) == case_id]
            xs = [int(r["keyframe_offset_from_repair_start"]) for r in case_rows]
            ys = [float(r["success_rate"]) for r in case_rows]
            ax.plot(xs, ys, marker="o", linewidth=1, markersize=2.5, label=case_id)
        ax.axvline(0, color="tab:green", linestyle="--", linewidth=1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"task{task_id:02d}: success vs keyframe offset")
        ax.set_xlabel("keyframe - repair_context_start")
        ax.set_ylabel("success rate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "keyframe_success_by_task.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for case_id in case_ids:
        rows = sorted(by_case[case_id], key=lambda x: int(x["keyframe"]))
        ax.scatter(
            [int(r["keyframe"]) for r in rows],
            [float(r["success_rate"]) for r in rows],
            s=16,
            alpha=0.75,
            label=case_id,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("keyframe step")
    ax.set_ylabel("K=5 success rate")
    ax.set_title("All selected cases: success distribution by keyframe")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "keyframe_success_all_cases.png", dpi=180)
    plt.close(fig)


def _planned_rows_for_cases(args: argparse.Namespace, cases: Sequence[dict]) -> List[dict]:
    planned = []
    for case in cases:
        case_review_path = _translated_existing_path(case["case_review_path"]) or Path(
            str(case["case_review_path"])
        )
        case_review = _load_json(case_review_path)
        repair_interval = _parse_int_pair(case_review.get("repair_replay_context"), (0, 1))
        minimal_interval = _parse_int_pair(case_review.get("minimal_slice"), (repair_interval[0], repair_interval[0] + 1))
        length = int(case_review.get("rollout_length") or 0)
        report_path = _translated_existing_path(case_review.get("report_path"))
        if report_path and report_path.exists():
            report = _load_json(report_path)
            length = int((report.get("selected_failed_rollout") or {}).get("length") or length)
        if length <= 0:
            length = max(repair_interval[1] + args.keyframe_window_after + 1, minimal_interval[1] + 1)
        keyframes = _sample_keyframes(
            str(case["case_id"]),
            length,
            repair_interval,
            args.keyframes_per_case,
            args.keyframe_window_before,
            args.keyframe_window_after,
            args.random_seed,
        )
        for keyframe in keyframes:
            planned.append(
                {
                    "case_id": case["case_id"],
                    "task_id": case.get("task_id"),
                    "init_state_id": case_review.get("init_state_id"),
                    "seed": case_review.get("seed"),
                    "evidence_level": case.get("evidence_level"),
                    "keyframe": int(keyframe),
                    "keyframe_offset_from_repair_start": int(keyframe - repair_interval[0]),
                    "repair_context": list(repair_interval),
                    "minimal_slice": list(minimal_interval),
                    "rollout_length": int(length),
                }
            )
    return planned


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_case_rows(Path(args.manifest), args.case_ids)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    planned = _planned_rows_for_cases(args, cases)
    _json_dump(output_dir / "planned_keyframes.json", planned)
    if args.dry_run:
        print(json.dumps({"num_cases": len(cases), "num_keyframes": len(planned)}, indent=2), flush=True)
        return

    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, int(args.policy_port))
    trial_rows: List[dict] = []
    results_path = output_dir / "keyframe_sweep_results.jsonl"
    case_manifest = []
    with results_path.open("a", encoding="utf-8") as results_file:
        for case in cases:
            case_id = str(case["case_id"])
            case_dir = output_dir / "cases" / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            case_review_path = _translated_existing_path(case["case_review_path"]) or Path(
                str(case["case_review_path"])
            )
            case_review = _load_json(case_review_path)
            repair_interval = _parse_int_pair(case_review.get("repair_replay_context"), (0, 1))
            minimal_interval = _parse_int_pair(case_review.get("minimal_slice"), (repair_interval[0], repair_interval[0] + 1))
            start_wall = time.perf_counter()
            rollout, archive_meta = _load_or_regenerate_rollout(args, case, client, case_dir)
            keyframes = _sample_keyframes(
                case_id,
                rollout.length,
                repair_interval,
                args.keyframes_per_case,
                args.keyframe_window_before,
                args.keyframe_window_after,
                args.random_seed,
            )
            case_manifest.append(
                {
                    "case_id": case_id,
                    "task_id": int(case["task_id"]),
                    "rollout_length": int(rollout.length),
                    "archive": archive_meta,
                    "repair_context": list(repair_interval),
                    "minimal_slice": list(minimal_interval),
                    "keyframes": keyframes,
                }
            )
            probe_args = _probe_args(
                args,
                seed=int(case_review.get("seed", case.get("seed", 7))),
                output=case_dir / "keyframe_sweep_probe.json",
            )
            for keyframe in keyframes:
                if keyframe < 0 or keyframe >= rollout.length:
                    continue
                evaluation = replay_candidate(
                    probe_args,
                    rollout,
                    int(keyframe),
                    min(int(keyframe) + 1, rollout.length),
                    rollout.failure_signature,
                    client=client,
                    policy_from_step=int(keyframe),
                    stage_level="keyframe_raw_policy_replan",
                    trials_override=int(args.trials),
                    early_stop_objective="same_failure",
                )
                rows = _trial_rows_from_evaluation(
                    case,
                    case_review,
                    int(keyframe),
                    repair_interval,
                    minimal_interval,
                    archive_meta,
                    evaluation,
                )
                for row in rows:
                    results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                results_file.flush()
                trial_rows.extend(rows)
            case_manifest[-1]["elapsed_seconds"] = float(time.perf_counter() - start_wall)
            _json_dump(output_dir / "case_manifest.json", case_manifest)
    summary = _write_summary(output_dir, trial_rows)
    _plot_summary(output_dir, summary)
    _json_dump(
        output_dir / "summary.json",
        {
            "schema_version": "keyframe-random-sweep-summary-v1",
            "output_dir": str(output_dir),
            "num_cases": len(cases),
            "num_keyframes": len(summary),
            "num_trials": len(trial_rows),
            "trials_per_keyframe": int(args.trials),
            "success_rate_mean": float(np.mean([row["success_rate"] for row in summary])) if summary else None,
        },
    )


def plot_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json or output_dir / "keyframe_success_summary.json")
    summary = _load_json(summary_path)
    _plot_summary(output_dir, summary)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random keyframe policy-replan success sweep.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids; empty means all manifest cases.")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8230)
    parser.add_argument("--policy-config", default="pi0_fast_libero")
    parser.add_argument("--policy-dir", type=Path, default=PROJECT_ROOT / "model_datasets/pi0fast-libero-libero_10/policy_overlay")
    parser.add_argument("--demo-dataset-root", type=Path, default=Path("/data2/yanghaoyun/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=32)
    parser.add_argument("--keyframes-per-case", type=int, default=21)
    parser.add_argument("--keyframe-window-before", type=int, default=48)
    parser.add_argument("--keyframe-window-after", type=int, default=48)
    parser.add_argument("--random-seed", type=int, default=20260603)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--replay-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--verbose-replay-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    args.case_ids = [x.strip() for x in str(args.case_ids).split(",") if x.strip()]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.plot_only:
        plot_only(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
