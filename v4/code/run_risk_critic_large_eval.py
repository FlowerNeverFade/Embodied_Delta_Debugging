from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from cost_summary import build_cost_summary, build_cost_summary_from_paths
from custom_tasks import registry as custom_tasks
from risk_critic_export import (
    CAUSAL_SCHEMAS,
    _strict_causal_pass,
    export_risk_critic_dataset,
    export_risk_critic_dataset_from_paths,
)
from train_risk_critic import _read_jsonl, train_and_evaluate


CODE_ROOT = Path(__file__).resolve().parent
V4_ROOT = CODE_ROOT.parent
PROJECT_ROOT = V4_ROOT.parent
OPENPI_PYTHON = Path("/root/autodl-tmp/research/openpi/.venv/bin/python")
LIBERO_PYTHON = Path("/root/autodl-tmp/envs/libero38/bin/python")
LEROBOT_PYTHON = Path("/root/miniconda3/bin/python")
OPENPI_ROOT = Path("/root/autodl-tmp/research/openpi")
DEFAULT_POLICY_DIR = Path("/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero")
DEFAULT_POLICY_CONFIG = "pi05_libero"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_datasets"
    / "pi0fast-libero-libero_10"
    / "outputs"
    / "risk_critic_ultra_video_causal_v4_targeted_k1_20260527"
)
DEFAULT_REPORT_DIR = DEFAULT_OUTPUT_DIR / "reports"
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "logs"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.jsonl"
DEFAULT_MASTER_LOG = DEFAULT_OUTPUT_DIR / "master.log"
DEFAULT_COST_SUMMARY = DEFAULT_OUTPUT_DIR / "cost_summary.json"
DEFAULT_EXPORT_PATH = DEFAULT_OUTPUT_DIR / "risk_critic_full_v1.jsonl"
DEFAULT_TRAIN_OUTPUT = DEFAULT_OUTPUT_DIR / "risk_critic_full_metrics.json"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.json"
DEFAULT_PORT = 8060
DEFAULT_TASK_IDS = tuple(range(10))
DEFAULT_INIT_STATE_IDS = (0, 1, 2)
DEFAULT_SEEDS = (7, 17, 27)
COMPLETED_REPORT_STATUSES = {
    "semantic_pass",
    "semantic_nonpass",
    "invalid_init",
    "skipped_existing",
}
EXPORTABLE_REPORT_STATUSES = {"semantic_pass", "semantic_nonpass", "skipped_existing"}
CAUSAL_SEMANTIC_VERSION = "global-multimodal-v4"
EXPECTED_CAUSAL_SCHEMA = "shed-cfs-causal-v4-global-multimodal"
RUNTIME_FINGERPRINT_KEYS = {
    "policy_host",
    "policy_port",
    "cuda_visible_devices",
    "xla_mem_fraction",
}


def _parse_int_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _format_int_list(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _env_without_proxy() -> dict:
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        env.pop(key, None)
    return env


def _first_cuda_device(cuda_visible_devices: object) -> str:
    first = str(cuda_visible_devices or "0").split(",", 1)[0].strip()
    return first or "0"


def _fingerprints_match(stored: object, expected: dict) -> bool:
    if not isinstance(stored, dict):
        return False

    def stable(fp: dict) -> dict:
        return {k: v for k, v in fp.items() if k not in RUNTIME_FINGERPRINT_KEYS}

    return stable(stored) == stable(expected)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if _port_open(host, port):
            return
        time.sleep(2.0)
    raise TimeoutError(f"Policy server at {host}:{port} did not become ready")


def _report_path(report_dir: Path, task_id: int, init_state_id: int, seed: int) -> Path:
    return report_dir / f"task{task_id:02d}_init{init_state_id:02d}_seed{seed:02d}_causal_v4.json"


def _log_path(log_dir: Path, task_id: int, init_state_id: int, seed: int) -> Path:
    return log_dir / f"task{task_id:02d}_init{init_state_id:02d}_seed{seed:02d}.log"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _nullable_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _report_bool(report: Optional[dict], *path: str) -> Optional[bool]:
    if not isinstance(report, dict):
        return None
    current: object = report
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return _nullable_bool(current)


def _active_video_dir(args: argparse.Namespace) -> Optional[Path]:
    if not args.record_video and args.video_dir is None:
        return None
    return Path(args.video_dir) if args.video_dir is not None else args.output_dir / "videos"


def _expected_video_path(
    args: argparse.Namespace, task_id: int, init_state_id: int, seed: int
) -> Optional[Path]:
    video_dir = _active_video_dir(args)
    if video_dir is None:
        return None
    prefix = f"task{task_id:02d}_init{init_state_id:02d}_seed{seed:02d}"
    return video_dir / f"{prefix}_task{task_id:02d}_init{init_state_id:02d}.mp4"


def _policy_metadata_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "policy_server_metadata.json"


def _policy_fingerprint(args: argparse.Namespace) -> dict:
    return {
        "policy_server_kind": str(args.policy_server_kind),
        "policy_host": str(args.policy_host),
        "policy_port": int(args.policy_port),
        "policy_config": str(args.policy_config),
        "policy_dir": str(Path(args.policy_dir)),
        "action_tokenizer_path": None
        if args.action_tokenizer_path is None
        else str(Path(args.action_tokenizer_path)),
        "text_tokenizer_path": None
        if args.text_tokenizer_path is None
        else str(Path(args.text_tokenizer_path)),
        "allow_hub_download": bool(args.allow_hub_download),
        "pytorch_device": str(args.pytorch_device or ""),
        "pytorch_compile_mode": str(args.pytorch_compile_mode or ""),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "xla_mem_fraction": float(args.xla_mem_fraction),
    }


def _case_fingerprint(
    args: argparse.Namespace, task_id: int, init_state_id: int, seed: int
) -> dict:
    video_dir = _active_video_dir(args)
    custom_metadata = (
        custom_tasks.task_metadata(str(args.task_suite_name))
        if custom_tasks.is_custom_suite(str(args.task_suite_name))
        else None
    )
    return {
        "schema_version": "risk-critic-run-fingerprint-v1",
        "causal_semantic_version": CAUSAL_SEMANTIC_VERSION,
        **_policy_fingerprint(args),
        "task_suite_name": str(args.task_suite_name),
        "custom_task_metadata": custom_metadata,
        "task_id": int(task_id),
        "init_state_id": int(init_state_id),
        "seed": int(seed),
        "replay_trials": int(args.replay_trials),
        "search_replay_trials": int(args.search_replay_trials),
        "confirm_replay_trials": int(args.confirm_replay_trials),
        "repair_replay_trials": int(args.repair_replay_trials),
        "probe_max_steps": int(args.probe_max_steps),
        "event_window": int(args.event_window),
        "causal_context_before": int(args.causal_context_before),
        "causal_context_after": int(args.causal_context_after),
        "causal_max_units": int(args.causal_max_units),
        "causal_ablation_trials": int(args.causal_ablation_trials),
        "causal_ablation_strategies": str(args.causal_ablation_strategies),
        "replay_cache_enabled": not bool(args.disable_replay_cache),
        "sequential_trial_pruning_enabled": bool(args.enable_sequential_trial_pruning),
        "hierarchical_causal_pruning_enabled": not bool(args.disable_hierarchical_causal_pruning),
        "continuation": str(args.continuation),
        "scripted_expert_repair_max_steps": int(args.scripted_expert_repair_max_steps),
        "demo_repair_timeout_seconds": float(args.demo_repair_timeout_seconds),
        "skip_source_repair_if_policy_pass": bool(args.skip_source_repair_if_policy_pass),
        "disable_source_repair": bool(args.disable_source_repair),
        "defer_source_repair": bool(args.defer_source_repair),
        "stop_after_first_repair_valid_core": bool(args.stop_after_first_repair_valid_core),
        "repair_scheduling_mode": str(args.repair_scheduling_mode),
        "disable_rule_language_intervention": bool(args.disable_rule_language_intervention),
        "enable_visual_policy_mask": bool(args.enable_visual_policy_mask),
        "top_k_minimal_sets": 5,
        "language_minimization": "phrase_level_bddl_delta_debug",
        "visual_projection": "policy_input_target_highlight_or_distractor_mask",
        "contact_evidence_priority": "mujoco_contact_then_degraded_proximity",
        "initial_state_max_attempts": int(args.initial_state_max_attempts),
        "disable_initial_state_quality_filter": bool(
            args.disable_initial_state_quality_filter
        ),
        "require_full_features": True,
        "record_video": bool(args.record_video or args.video_dir is not None),
        "require_video": bool(args.require_video),
        "video_dir": None if video_dir is None else str(video_dir),
        "video_camera": str(args.video_camera),
        "video_fps": int(args.video_fps),
        "video_every_n": int(args.video_every_n),
        "video_codec": str(args.video_codec),
        "video_quality": int(args.video_quality),
        "video_no_flip": bool(args.video_no_flip),
        "camera_size": int(args.camera_size),
    }


def _report_video_paths(report: dict) -> list[Path]:
    paths = []
    for rollout in report.get("rollout_summaries") or []:
        video_path = rollout.get("video_path")
        if video_path:
            paths.append(Path(video_path))
    return paths


def _positive_window_count(report: dict) -> int:
    return sum(
        1
        for window in report.get("risk_training_windows") or []
        if int(window.get("label") or 0) == 1
    )


def _positive_window_kinds(report: dict) -> list[str]:
    return sorted(
        {
            str(window.get("sample_kind") or "risk_window")
            for window in report.get("risk_training_windows") or []
            if int(window.get("label") or 0) == 1
        }
    )


def _has_full_feature_windows(report: dict) -> bool:
    windows = report.get("risk_training_windows") or []
    return bool(
        windows
        and all((window.get("features") or {}).get("feature_quality") == "full" for window in windows)
    )


def _is_reusable_report(
    path: Path,
    args: argparse.Namespace,
    task_id: int,
    init_state_id: int,
    seed: int,
) -> bool:
    report = _load_json(path)
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") != EXPECTED_CAUSAL_SCHEMA:
        return False
    invalid_init = (report.get("feasibility") or {}).get("verdict") == "invalid_initial_state"
    if not invalid_init and not _has_full_feature_windows(report):
        return False
    if not _fingerprints_match(
        report.get("runner_fingerprint"),
        _case_fingerprint(args, task_id, init_state_id, seed),
    ):
        return False
    if args.require_video:
        video_paths = _report_video_paths(report)
        if not video_paths or not all(path.exists() for path in video_paths):
            return False
    return True


def _semantic_status_from_report(report: Optional[dict], return_code: Optional[int]) -> Optional[str]:
    if not isinstance(report, dict) or report.get("schema_version") not in CAUSAL_SCHEMAS:
        return None
    if (report.get("feasibility") or {}).get("verdict") == "invalid_initial_state":
        return "invalid_init"
    natural_pass = _report_bool(report, "feasibility", "pi05_natural_pass")
    if natural_pass is True or return_code == 0:
        return "semantic_pass"
    if natural_pass is False or return_code == 1:
        return "semantic_nonpass"
    return None


def _annotate_report_with_runner_metadata(
    report_path: Path,
    args: argparse.Namespace,
    task_id: int,
    init_state_id: int,
    seed: int,
    return_code: Optional[int],
) -> None:
    report = _load_json(report_path)
    if not isinstance(report, dict):
        return
    report["runner_fingerprint"] = _case_fingerprint(args, task_id, init_state_id, seed)
    report["runner_case_metadata"] = {
        "schema_version": "risk-critic-runner-case-metadata-v1",
        "return_code": return_code,
        "semantic_status": _semantic_status_from_report(report, return_code),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(report_path, report)


def _case_command(args: argparse.Namespace, report_path: Path, task_id: int, init_state_id: int, seed: int) -> list[str]:
    cmd = [
        str(LIBERO_PYTHON),
        str(CODE_ROOT / "pi05_natural_failure_probe.py"),
        "--policy-host",
        str(args.policy_host),
        "--policy-port",
        str(args.policy_port),
        "--policy-config",
        str(args.policy_config),
        "--policy-checkpoint",
        str(args.policy_dir),
        "--task-suite-name",
        args.task_suite_name,
        "--task-ids",
        str(task_id),
        "--init-state-ids",
        str(init_state_id),
        "--seed",
        str(seed),
        "--replay-trials",
        str(args.replay_trials),
        "--search-replay-trials",
        str(args.search_replay_trials),
        "--confirm-replay-trials",
        str(args.confirm_replay_trials),
        "--repair-replay-trials",
        str(args.repair_replay_trials),
        "--event-window",
        str(args.event_window),
        "--causal-context-before",
        str(args.causal_context_before),
        "--causal-context-after",
        str(args.causal_context_after),
        "--causal-max-units",
        str(args.causal_max_units),
        "--causal-ablation-trials",
        str(args.causal_ablation_trials),
        "--causal-ablation-strategies",
        str(args.causal_ablation_strategies),
        "--continuation",
        args.continuation,
        "--scripted-expert-repair-max-steps",
        str(args.scripted_expert_repair_max_steps),
        "--demo-repair-timeout-seconds",
        str(args.demo_repair_timeout_seconds),
        "--initial-state-max-attempts",
        str(args.initial_state_max_attempts),
        "--camera-size",
        str(args.camera_size),
        "--gpu-id",
        _first_cuda_device(args.cuda_visible_devices),
        "--output",
        str(report_path),
    ]
    if int(args.probe_max_steps) > 0:
        cmd.extend(["--max-steps", str(args.probe_max_steps)])
    if args.disable_initial_state_quality_filter:
        cmd.append("--disable-initial-state-quality-filter")
    if args.skip_source_repair_if_policy_pass:
        cmd.append("--skip-source-repair-if-policy-pass")
    if args.disable_source_repair:
        cmd.append("--disable-source-repair")
    if args.defer_source_repair:
        cmd.append("--defer-source-repair")
    if args.disable_replay_cache:
        cmd.append("--disable-replay-cache")
    if not args.enable_sequential_trial_pruning:
        cmd.append("--disable-sequential-trial-pruning")
    if args.disable_hierarchical_causal_pruning:
        cmd.append("--disable-hierarchical-causal-pruning")
    if args.stop_after_first_repair_valid_core:
        cmd.append("--stop-after-first-repair-valid-core")
    cmd.extend(["--repair-scheduling-mode", str(args.repair_scheduling_mode)])
    if args.disable_rule_language_intervention:
        cmd.append("--disable-rule-language-intervention")
    if args.enable_visual_policy_mask:
        cmd.append("--enable-visual-policy-mask")
    if args.record_video or args.video_dir is not None:
        video_dir = _active_video_dir(args)
        assert video_dir is not None
        cmd.extend(
            [
                "--record-video",
                "--video-dir",
                str(video_dir),
                "--video-prefix",
                f"task{task_id:02d}_init{init_state_id:02d}_seed{seed:02d}",
                "--video-camera",
                str(args.video_camera),
                "--video-fps",
                str(args.video_fps),
                "--video-every-n",
                str(args.video_every_n),
                "--video-codec",
                str(args.video_codec),
                "--video-quality",
                str(args.video_quality),
            ]
        )
        if args.video_no_flip:
            cmd.append("--video-no-flip")
    return cmd


def _row_from_report(
    args: argparse.Namespace,
    report_path: Path,
    log_path: Path,
    task_id: int,
    init_state_id: int,
    seed: int,
    status: str,
    start: str,
    return_code: Optional[int] = None,
    wall_seconds: Optional[float] = None,
) -> dict:
    report = _load_json(report_path)
    if return_code is None and isinstance(report, dict):
        stored_return_code = (report.get("runner_case_metadata") or {}).get("return_code")
        if isinstance(stored_return_code, int):
            return_code = stored_return_code
    video_paths = [] if not isinstance(report, dict) else _report_video_paths(report)
    video_exists = bool(video_paths) and all(path.exists() for path in video_paths)
    positive_windows = 0 if not isinstance(report, dict) else _positive_window_count(report)
    semantic_status = _semantic_status_from_report(report, return_code)
    row = {
        "schema_version": "risk-critic-large-eval-manifest-v1",
        "task_id": int(task_id),
        "init_state_id": int(init_state_id),
        "seed": int(seed),
        "status": status,
        "semantic_status": semantic_status,
        "return_code": return_code,
        "report_path": str(report_path) if report_path.exists() else None,
        "log_path": str(log_path),
        "started_at": start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "report_schema_version": None if not isinstance(report, dict) else report.get("schema_version"),
        "natural_failure_found": None if not isinstance(report, dict) else (
            False
            if (report.get("feasibility") or {}).get("verdict") == "invalid_initial_state"
            else True
            if report.get("selected_failed_rollout") is not None
            else False
        ),
        "initial_state_valid": None
        if not isinstance(report, dict)
        else (
            False
            if (report.get("feasibility") or {}).get("verdict") == "invalid_initial_state"
            else None
        ),
        "same_failure_pass": _report_bool(report, "reproduction_statistics", "same_failure"),
        "causal_pass": None if not isinstance(report, dict) else _strict_causal_pass(report),
        "policy_strong_repair_valid_pass": None
        if not isinstance(report, dict)
        else bool(
            report.get("policy_strong_repair_valid_pass")
            or (report.get("causal_validation") or {}).get("policy_strong_repair_valid_pass")
        ),
        "demo_existence_repair_pass": None
        if not isinstance(report, dict)
        else bool(
            report.get("demo_existence_repair_pass")
            or (report.get("causal_validation") or {}).get("demo_existence_repair_pass")
        ),
        "full_success_repair_pass": None
        if not isinstance(report, dict)
        else bool(
            report.get("full_success_policy_repair_pass")
            or (report.get("causal_validation") or {}).get("full_success_policy_repair_pass")
            or report.get("full_success_repair_pass")
        ),
        "full_windows": None if not isinstance(report, dict) else len(report.get("risk_training_windows") or []),
        "positive_windows": positive_windows,
        "positive_source_report": bool(positive_windows > 0),
        "positive_window_kinds": [] if not isinstance(report, dict) else _positive_window_kinds(report),
        "video_path": str(video_paths[0]) if video_paths else None,
        "video_paths": [str(path) for path in video_paths],
        "video_exists": bool(video_exists),
        "runtime_profile": None if not isinstance(report, dict) else report.get("runtime_profile"),
    }
    return row


def _write_manifest_line(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _report_paths_from_rows(rows: Sequence[dict]) -> list[Path]:
    paths = []
    for row in rows:
        if row.get("status") not in EXPORTABLE_REPORT_STATUSES:
            continue
        if row.get("semantic_status") == "invalid_init":
            continue
        report_path = row.get("report_path")
        if report_path:
            path = Path(report_path)
            if path.exists():
                paths.append(path)
    return paths


def _quarantine_timeout_video(
    args: argparse.Namespace,
    task_id: int,
    init_state_id: int,
    seed: int,
) -> None:
    video_path = _expected_video_path(args, task_id, init_state_id, seed)
    if video_path is None or not video_path.exists():
        return
    quarantine = video_path.with_suffix(video_path.suffix + ".timeout_partial")
    try:
        if quarantine.exists():
            quarantine.unlink()
        video_path.rename(quarantine)
    except Exception:
        pass


def _run_case(args: argparse.Namespace, task_id: int, init_state_id: int, seed: int) -> dict:
    report_path = _report_path(args.report_dir, task_id, init_state_id, seed)
    log_path = _log_path(args.log_dir, task_id, init_state_id, seed)
    start = datetime.now(timezone.utc).isoformat()

    if args.resume and _is_reusable_report(report_path, args, task_id, init_state_id, seed):
        row = _row_from_report(
            args,
            report_path,
            log_path,
            task_id,
            init_state_id,
            seed,
            status="skipped_existing",
            start=start,
        )
        _write_manifest_line(args.manifest_path, row)
        return row

    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _case_command(args, report_path, task_id, init_state_id, seed)
    env = _env_without_proxy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "MUJOCO_EGL_DEVICE_ID": _first_cuda_device(args.cuda_visible_devices),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
    )
    probe_started = time.time()
    return_code = None
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(
            json.dumps(
                {
                    "event": "start_case",
                    "task_id": int(task_id),
                    "init_state_id": int(init_state_id),
                    "seed": int(seed),
                    "command": cmd,
                    "started_at": start,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log_f.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=None
                if args.case_timeout_seconds <= 0
                else float(args.case_timeout_seconds),
            )
            return_code = int(proc.returncode)
            finish_event = "finish_case"
        except subprocess.TimeoutExpired:
            return_code = None
            finish_event = "timeout_case"
            _quarantine_timeout_video(args, task_id, init_state_id, seed)
        log_f.write(
            json.dumps(
                {
                    "event": finish_event,
                    "task_id": int(task_id),
                    "init_state_id": int(init_state_id),
                    "seed": int(seed),
                    "return_code": return_code,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": float(time.time() - probe_started),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    if return_code is not None:
        _annotate_report_with_runner_metadata(
            report_path, args, task_id, init_state_id, seed, return_code
        )
    valid_report = (
        return_code is not None
        and _is_reusable_report(report_path, args, task_id, init_state_id, seed)
    )
    report = _load_json(report_path) if valid_report else None
    semantic_status = _semantic_status_from_report(report, return_code)
    if return_code is None:
        status = "timeout"
    elif valid_report and semantic_status in {"semantic_pass", "semantic_nonpass", "invalid_init"}:
        status = semantic_status
    else:
        status = "probe_failed"
    row = _row_from_report(
        args,
        report_path,
        log_path,
        task_id,
        init_state_id,
        seed,
        status=status,
        start=start,
        return_code=return_code,
        wall_seconds=float(time.time() - probe_started),
    )
    _write_manifest_line(args.manifest_path, row)
    return row


def _start_policy_server(args: argparse.Namespace, output_dir: Path) -> Optional[subprocess.Popen]:
    if _port_open(args.policy_host, args.policy_port):
        metadata = _load_json(_policy_metadata_path(args))
        if args.allow_unverified_policy_server:
            return None
        if (
            isinstance(metadata, dict)
            and metadata.get("policy_fingerprint") == _policy_fingerprint(args)
        ):
            return None
        raise RuntimeError(
            "Policy server port is already open but no matching runner metadata was found. "
            "Use --allow-unverified-policy-server only if you intentionally want to reuse it."
        )
    if not args.launch_policy_server:
        raise RuntimeError(
            f"No policy server is reachable at {args.policy_host}:{args.policy_port}, "
            "and --launch-policy-server was not provided."
        )
    if args.dry_run:
        return None
    server_log = output_dir / "policy_server.log"
    if args.policy_server_kind == "openpi":
        cmd = [
            str(OPENPI_PYTHON),
            "scripts/serve_policy.py",
            "--port",
            str(args.policy_port),
        ]
        if args.pytorch_device:
            cmd.extend(["--pytorch-device", str(args.pytorch_device)])
        if args.pytorch_compile_mode:
            cmd.extend(["--pytorch-compile-mode", str(args.pytorch_compile_mode)])
        cmd.extend(
            [
                "policy:checkpoint",
                f"--policy.config={args.policy_config}",
                f"--policy.dir={args.policy_dir}",
            ]
        )
        cwd = OPENPI_ROOT
    elif args.policy_server_kind == "lerobot_pi0fast":
        cmd = [
            str(LEROBOT_PYTHON),
            str(CODE_ROOT / "serve_lerobot_pi0fast_policy.py"),
            "--port",
            str(args.policy_port),
            "--policy-dir",
            str(args.policy_dir),
            "--device",
            str(args.pytorch_device or "cuda"),
            "--compile-mode",
            str(args.pytorch_compile_mode or "none"),
        ]
        if args.pytorch_compile_mode and args.pytorch_compile_mode != "none":
            cmd.append("--compile-model")
        if args.action_tokenizer_path is not None:
            cmd.extend(["--action-tokenizer-path", str(args.action_tokenizer_path)])
        if args.text_tokenizer_path is not None:
            cmd.extend(["--text-tokenizer-path", str(args.text_tokenizer_path)])
        if args.allow_hub_download:
            cmd.append("--allow-hub-download")
        else:
            cmd.append("--local-files-only")
        cwd = PROJECT_ROOT
    else:
        raise ValueError(f"Unsupported policy server kind: {args.policy_server_kind}")
    env = _env_without_proxy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(args.xla_mem_fraction),
        }
    )
    if not args.allow_hub_download:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    log_f = server_log.open("a", encoding="utf-8")
    server_proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    log_f.close()
    try:
        _wait_for_port(args.policy_host, args.policy_port, timeout=args.policy_ready_timeout)
    except Exception:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=30)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass
        raise
    (output_dir / "policy_server.pid").write_text(str(server_proc.pid), encoding="utf-8")
    _write_json(
        _policy_metadata_path(args),
        {
            "schema_version": "risk-critic-policy-server-metadata-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pid": int(server_proc.pid),
            "policy_fingerprint": _policy_fingerprint(args),
            "command": cmd,
        },
    )
    return server_proc


def run_large_eval(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = _active_video_dir(args)
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    policy_proc = None
    started_policy = False
    try:
        if not args.dry_run:
            policy_proc = _start_policy_server(args, args.output_dir)
            started_policy = policy_proc is not None

        rows = []
        planned = [
            (task_id, init_state_id, seed)
            for task_id in args.task_ids
            for init_state_id in args.init_state_ids
            for seed in args.seeds
        ]
        if args.shuffle_cases:
            rng = random.Random(int(args.case_order_seed))
            rng.shuffle(planned)
        if args.max_cases > 0:
            planned = planned[: args.max_cases]

        summary_path = args.output_dir / "planned_cases.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "risk-critic-large-eval-plan-v1",
                    "task_suite_name": args.task_suite_name,
                    "task_ids": args.task_ids,
                    "init_state_ids": args.init_state_ids,
                    "seeds": args.seeds,
                    "planned_cases": len(planned),
                    "report_dir": str(args.report_dir),
                    "log_dir": str(args.log_dir),
                    "video_dir": None if video_dir is None else str(video_dir),
                    "record_video": bool(args.record_video or args.video_dir is not None),
                    "camera_size": int(args.camera_size),
                    "video_fps": int(args.video_fps),
                    "video_quality": int(args.video_quality),
                    "positive_target": int(args.positive_target),
                    "min_cases_before_positive_stop": int(args.min_cases_before_positive_stop),
                    "shuffle_cases": bool(args.shuffle_cases),
                    "case_order_seed": int(args.case_order_seed),
                    "postprocess_scope": args.postprocess_scope,
                    "case_timeout_seconds": float(args.case_timeout_seconds),
                    "replay_trials": int(args.replay_trials),
                    "search_replay_trials": int(args.search_replay_trials),
                    "confirm_replay_trials": int(args.confirm_replay_trials),
                    "repair_replay_trials": int(args.repair_replay_trials),
                    "scripted_expert_repair_max_steps": int(args.scripted_expert_repair_max_steps),
                    "demo_repair_timeout_seconds": float(args.demo_repair_timeout_seconds),
                    "probe_max_steps": int(args.probe_max_steps),
                    "event_window": int(args.event_window),
                    "causal_context_before": int(args.causal_context_before),
                    "causal_context_after": int(args.causal_context_after),
                    "causal_max_units": int(args.causal_max_units),
                    "causal_ablation_trials": int(args.causal_ablation_trials),
                    "initial_state_max_attempts": int(args.initial_state_max_attempts),
                    "disable_initial_state_quality_filter": bool(
                        args.disable_initial_state_quality_filter
                    ),
                    "policy_server_kind": str(args.policy_server_kind),
                    "disable_rule_language_intervention": bool(
                        args.disable_rule_language_intervention
                    ),
                    "enable_visual_policy_mask": bool(args.enable_visual_policy_mask),
                    "causal_ablation_strategies": str(args.causal_ablation_strategies),
                    "disable_source_repair": bool(args.disable_source_repair),
                    "defer_source_repair": bool(args.defer_source_repair),
                    "repair_scheduling_mode": str(args.repair_scheduling_mode),
                    "replay_cache_enabled": not bool(args.disable_replay_cache),
                    "sequential_trial_pruning_enabled": bool(args.enable_sequential_trial_pruning),
                    "hierarchical_causal_pruning_enabled": not bool(
                        args.disable_hierarchical_causal_pruning
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if args.dry_run:
            for task_id, init_state_id, seed in planned:
                expected_video = _expected_video_path(args, task_id, init_state_id, seed)
                _write_manifest_line(
                    args.manifest_path,
                    {
                        "schema_version": "risk-critic-large-eval-manifest-v1",
                        "task_id": int(task_id),
                        "init_state_id": int(init_state_id),
                        "seed": int(seed),
                        "status": "planned",
                        "report_path": str(
                            _report_path(args.report_dir, task_id, init_state_id, seed)
                        ),
                        "log_path": str(_log_path(args.log_dir, task_id, init_state_id, seed)),
                        "video_path": None if expected_video is None else str(expected_video),
                        "video_exists": False,
                        "positive_windows": 0,
                        "positive_source_report": False,
                        "runner_fingerprint": _case_fingerprint(
                            args, task_id, init_state_id, seed
                        ),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            summary = {
                "schema_version": "risk-critic-large-eval-summary-v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(args.output_dir),
                "config": {
                    "task_suite_name": args.task_suite_name,
                    "task_ids": args.task_ids,
                    "init_state_ids": args.init_state_ids,
                    "seeds": args.seeds,
                    "dry_run": True,
                    "record_video": bool(args.record_video or args.video_dir is not None),
                    "video_dir": None if video_dir is None else str(video_dir),
                    "camera_size": int(args.camera_size),
                    "video_fps": int(args.video_fps),
                    "video_quality": int(args.video_quality),
                    "positive_target": int(args.positive_target),
                    "min_cases_before_positive_stop": int(args.min_cases_before_positive_stop),
                    "shuffle_cases": bool(args.shuffle_cases),
                    "case_order_seed": int(args.case_order_seed),
                    "postprocess_scope": args.postprocess_scope,
                    "case_timeout_seconds": float(args.case_timeout_seconds),
                    "replay_trials": int(args.replay_trials),
                    "search_replay_trials": int(args.search_replay_trials),
                    "confirm_replay_trials": int(args.confirm_replay_trials),
                    "repair_replay_trials": int(args.repair_replay_trials),
                    "scripted_expert_repair_max_steps": int(args.scripted_expert_repair_max_steps),
                    "demo_repair_timeout_seconds": float(args.demo_repair_timeout_seconds),
                    "probe_max_steps": int(args.probe_max_steps),
                    "event_window": int(args.event_window),
                    "causal_context_before": int(args.causal_context_before),
                    "causal_context_after": int(args.causal_context_after),
                    "causal_max_units": int(args.causal_max_units),
                    "causal_ablation_trials": int(args.causal_ablation_trials),
                    "initial_state_max_attempts": int(args.initial_state_max_attempts),
                    "disable_initial_state_quality_filter": bool(
                        args.disable_initial_state_quality_filter
                    ),
                    "policy_server_kind": str(args.policy_server_kind),
                    "causal_ablation_strategies": str(args.causal_ablation_strategies),
                    "disable_source_repair": bool(args.disable_source_repair),
                    "defer_source_repair": bool(args.defer_source_repair),
                    "repair_scheduling_mode": str(args.repair_scheduling_mode),
                    "replay_cache_enabled": not bool(args.disable_replay_cache),
                    "sequential_trial_pruning_enabled": bool(args.enable_sequential_trial_pruning),
                    "hierarchical_causal_pruning_enabled": not bool(
                        args.disable_hierarchical_causal_pruning
                    ),
                },
                "aggregate": {
                    "planned_cases": len(planned),
                    "processed_cases": 0,
                    "completed_reports": 0,
                    "semantic_pass": 0,
                    "semantic_nonpass": 0,
                    "skipped_existing": 0,
                    "invalid_init": 0,
                    "probe_failed": 0,
                    "timeout": 0,
                    "natural_failures": 0,
                    "natural_failure_missing": 0,
                    "same_failure_passes": 0,
                    "same_failure_missing": 0,
                    "causal_passes": 0,
                    "causal_missing": 0,
                    "policy_strong_repair_valid_passes": 0,
                    "demo_existence_repair_passes": 0,
                    "reports_with_full_windows": 0,
                    "positive_windows": 0,
                    "videos_written": 0,
                    "early_stopped": False,
                },
                "manifest_path": str(args.manifest_path),
                "report_dir": str(args.report_dir),
                "log_dir": str(args.log_dir),
                "rows": [],
            }
            args.summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return summary

        for task_id, init_state_id, seed in planned:
            rows.append(_run_case(args, task_id, init_state_id, seed))
            positive_windows = sum(int(row.get("positive_windows") or 0) for row in rows)
            processed_cases = len(rows)
            if (
                args.positive_target > 0
                and processed_cases >= args.min_cases_before_positive_stop
                and positive_windows >= args.positive_target
            ):
                break

        current_report_paths = _report_paths_from_rows(rows)
        if args.postprocess_scope == "report-dir":
            cost_summary = build_cost_summary(args.report_dir)
            export_summary = export_risk_critic_dataset(
                args.report_dir,
                args.export_path,
                require_full_features=True,
            )
        else:
            cost_summary = build_cost_summary_from_paths(
                current_report_paths, outputs_root=args.output_dir
            )
            export_summary = export_risk_critic_dataset_from_paths(
                current_report_paths,
                args.export_path,
                require_full_features=True,
                outputs_root=args.output_dir,
            )
        args.cost_summary_path.write_text(
            json.dumps(cost_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        samples = _read_jsonl(args.export_path)
        train_summary = train_and_evaluate(
            samples,
            seed=args.train_seed,
            val_fraction=args.val_fraction,
            steps=args.train_steps,
            feature_set="full_state_action_goal",
            split_by="source_report",
            min_class_count=args.min_class_count,
        )
        args.train_output.write_text(
            json.dumps(train_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        aggregate = {
            "planned_cases": len(planned),
            "processed_cases": len(rows),
            "completed_reports": sum(1 for row in rows if row["status"] in COMPLETED_REPORT_STATUSES),
            "semantic_pass": sum(1 for row in rows if row["status"] == "semantic_pass"),
            "semantic_nonpass": sum(1 for row in rows if row["status"] == "semantic_nonpass"),
            "skipped_existing": sum(1 for row in rows if row["status"] == "skipped_existing"),
            "probe_failed": sum(1 for row in rows if row["status"] == "probe_failed"),
            "timeout": sum(1 for row in rows if row["status"] == "timeout"),
            "invalid_init": sum(
                1
                for row in rows
                if row.get("status") == "invalid_init"
                or row.get("semantic_status") == "invalid_init"
            ),
            "natural_failures": sum(1 for row in rows if row.get("natural_failure_found") is True),
            "natural_failure_missing": sum(1 for row in rows if row.get("natural_failure_found") is None),
            "same_failure_passes": sum(1 for row in rows if row.get("same_failure_pass") is True),
            "same_failure_missing": sum(1 for row in rows if row.get("same_failure_pass") is None),
            "causal_passes": sum(1 for row in rows if row.get("causal_pass") is True),
            "causal_missing": sum(1 for row in rows if row.get("causal_pass") is None),
            "policy_strong_repair_valid_passes": sum(
                1 for row in rows if row.get("policy_strong_repair_valid_pass") is True
            ),
            "demo_existence_repair_passes": sum(
                1 for row in rows if row.get("demo_existence_repair_pass") is True
            ),
            "full_success_repair_passes": sum(
                1 for row in rows if row.get("full_success_repair_pass") is True
            ),
            "reports_with_full_windows": sum(
                1 for row in rows if isinstance(row.get("full_windows"), int) and row["full_windows"] > 0
            ),
            "positive_windows": sum(int(row.get("positive_windows") or 0) for row in rows),
            "positive_source_reports": sum(1 for row in rows if row.get("positive_source_report")),
            "videos_written": sum(1 for row in rows if row.get("video_exists")),
            "early_stopped": len(rows) < len(planned),
        }
        summary = {
            "schema_version": "risk-critic-large-eval-summary-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(args.output_dir),
            "config": {
                "task_suite_name": args.task_suite_name,
                "task_ids": args.task_ids,
                "init_state_ids": args.init_state_ids,
                "seeds": args.seeds,
                "policy_host": args.policy_host,
                "policy_port": args.policy_port,
                "policy_server_kind": args.policy_server_kind,
                "policy_config": args.policy_config,
                "policy_dir": str(args.policy_dir),
                "action_tokenizer_path": None
                if args.action_tokenizer_path is None
                else str(args.action_tokenizer_path),
                "text_tokenizer_path": None
                if args.text_tokenizer_path is None
                else str(args.text_tokenizer_path),
                "allow_hub_download": bool(args.allow_hub_download),
                "pytorch_device": args.pytorch_device,
                "pytorch_compile_mode": args.pytorch_compile_mode,
                "replay_trials": args.replay_trials,
                "search_replay_trials": args.search_replay_trials,
                "confirm_replay_trials": args.confirm_replay_trials,
                "repair_replay_trials": args.repair_replay_trials,
                "scripted_expert_repair_max_steps": args.scripted_expert_repair_max_steps,
                "probe_max_steps": args.probe_max_steps,
                "event_window": args.event_window,
                "causal_context_before": args.causal_context_before,
                "causal_context_after": args.causal_context_after,
                "causal_max_units": args.causal_max_units,
                "causal_ablation_trials": args.causal_ablation_trials,
                "causal_ablation_strategies": str(args.causal_ablation_strategies),
                "skip_source_repair_if_policy_pass": bool(args.skip_source_repair_if_policy_pass),
                "disable_source_repair": bool(args.disable_source_repair),
                "defer_source_repair": bool(args.defer_source_repair),
                "stop_after_first_repair_valid_core": bool(args.stop_after_first_repair_valid_core),
                "repair_scheduling_mode": str(args.repair_scheduling_mode),
                "disable_rule_language_intervention": bool(args.disable_rule_language_intervention),
                "enable_visual_policy_mask": bool(args.enable_visual_policy_mask),
                "replay_cache_enabled": not bool(args.disable_replay_cache),
                "sequential_trial_pruning_enabled": bool(args.enable_sequential_trial_pruning),
                "hierarchical_causal_pruning_enabled": not bool(args.disable_hierarchical_causal_pruning),
                "continuation": args.continuation,
                "resume": bool(args.resume),
                "require_video": bool(args.require_video),
                "record_video": bool(args.record_video or args.video_dir is not None),
                "video_dir": None if video_dir is None else str(video_dir),
                "video_camera": args.video_camera,
                "video_fps": int(args.video_fps),
                "video_every_n": int(args.video_every_n),
                "video_codec": args.video_codec,
                "video_quality": int(args.video_quality),
                "camera_size": int(args.camera_size),
                "positive_target": int(args.positive_target),
                "min_cases_before_positive_stop": int(args.min_cases_before_positive_stop),
                "shuffle_cases": bool(args.shuffle_cases),
                "case_order_seed": int(args.case_order_seed),
                "min_class_count": int(args.min_class_count),
                "feature_set": "full_state_action_goal",
                "split_by": "source_report",
                "postprocess_scope": args.postprocess_scope,
                "case_timeout_seconds": float(args.case_timeout_seconds),
                "allow_unverified_policy_server": bool(args.allow_unverified_policy_server),
            },
            "aggregate": aggregate,
            "cost_summary_path": str(args.cost_summary_path),
            "export_summary": export_summary,
            "train_summary": train_summary,
            "manifest_path": str(args.manifest_path),
            "report_dir": str(args.report_dir),
            "log_dir": str(args.log_dir),
            "rows": rows,
        }
        args.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
    finally:
        if started_policy and policy_proc is not None:
            try:
                policy_proc.terminate()
                policy_proc.wait(timeout=30)
            except Exception:
                try:
                    policy_proc.kill()
                except Exception:
                    pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run large-scale full-feature Risk Critic evaluation."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cost-summary-path", type=Path, default=DEFAULT_COST_SUMMARY)
    parser.add_argument("--export-path", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--policy-server-kind", choices=("openpi", "lerobot_pi0fast"), default="openpi")
    parser.add_argument("--policy-config", default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--action-tokenizer-path", type=Path, default=None)
    parser.add_argument("--text-tokenizer-path", type=Path, default=None)
    parser.add_argument("--allow-hub-download", action="store_true")
    parser.add_argument("--pytorch-device", default="")
    parser.add_argument("--pytorch-compile-mode", default="")
    parser.add_argument("--policy-ready-timeout", type=float, default=900.0)
    parser.add_argument("--launch-policy-server", action="store_true", default=False)
    parser.add_argument("--no-launch-policy-server", dest="launch_policy_server", action="store_false")
    parser.add_argument("--allow-unverified-policy-server", action="store_true")
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-ids", default=_format_int_list(DEFAULT_TASK_IDS))
    parser.add_argument("--init-state-ids", default=_format_int_list(DEFAULT_INIT_STATE_IDS))
    parser.add_argument("--seeds", default=_format_int_list(DEFAULT_SEEDS))
    parser.add_argument("--replay-trials", type=int, default=5)
    parser.add_argument("--search-replay-trials", type=int, default=1)
    parser.add_argument("--confirm-replay-trials", type=int, default=5)
    parser.add_argument("--repair-replay-trials", type=int, default=1)
    parser.add_argument("--scripted-expert-repair-max-steps", type=int, default=180)
    parser.add_argument("--demo-repair-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--initial-state-max-attempts", type=int, default=8)
    parser.add_argument("--disable-initial-state-quality-filter", action="store_true")
    parser.add_argument(
        "--probe-max-steps",
        type=int,
        default=0,
        help="Optional max-steps override passed to each probe. Default keeps the suite-specific horizon.",
    )
    parser.add_argument("--event-window", type=int, default=32)
    parser.add_argument("--causal-context-before", type=int, default=36)
    parser.add_argument("--causal-context-after", type=int, default=8)
    parser.add_argument("--causal-max-units", type=int, default=18)
    parser.add_argument("--causal-ablation-trials", type=int, default=5)
    parser.add_argument("--causal-ablation-strategies", default="hold")
    parser.add_argument("--disable-replay-cache", action="store_true")
    parser.add_argument(
        "--disable-sequential-trial-pruning",
        dest="enable_sequential_trial_pruning",
        action="store_false",
    )
    parser.set_defaults(enable_sequential_trial_pruning=True)
    parser.add_argument("--disable-hierarchical-causal-pruning", action="store_true")
    parser.add_argument("--skip-source-repair-if-policy-pass", action="store_true")
    parser.add_argument("--stop-after-first-repair-valid-core", action="store_true")
    parser.add_argument("--disable-source-repair", action="store_true")
    parser.add_argument("--defer-source-repair", action="store_true")
    parser.add_argument("--disable-rule-language-intervention", action="store_true")
    parser.add_argument("--enable-visual-policy-mask", action="store_true")
    parser.add_argument(
        "--repair-scheduling-mode",
        choices=("pass_hunt", "topk_complete"),
        default="pass_hunt",
    )
    parser.add_argument("--continuation", choices=("recorded", "policy"), default="recorded")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-camera", default="agentview_image")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-every-n", type=int, default=1)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=8)
    parser.add_argument("--video-no-flip", action="store_true")
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--positive-target", type=int, default=0)
    parser.add_argument("--min-cases-before-positive-stop", type=int, default=0)
    parser.add_argument("--shuffle-cases", action="store_true")
    parser.add_argument("--case-order-seed", type=int, default=0)
    parser.add_argument("--case-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--postprocess-scope",
        choices=("current-rows", "report-dir"),
        default="current-rows",
    )
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--xla-mem-fraction", type=float, default=0.55)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.report_dir = (
        args.output_dir / "reports"
        if Path(args.report_dir) == DEFAULT_REPORT_DIR
        else Path(args.report_dir)
    )
    args.log_dir = (
        args.output_dir / "logs" if Path(args.log_dir) == DEFAULT_LOG_DIR else Path(args.log_dir)
    )
    args.manifest_path = (
        args.output_dir / "manifest.jsonl"
        if Path(args.manifest_path) == DEFAULT_MANIFEST
        else Path(args.manifest_path)
    )
    args.cost_summary_path = (
        args.output_dir / "cost_summary.json"
        if Path(args.cost_summary_path) == DEFAULT_COST_SUMMARY
        else Path(args.cost_summary_path)
    )
    args.export_path = (
        args.output_dir / "risk_critic_full_v1.jsonl"
        if Path(args.export_path) == DEFAULT_EXPORT_PATH
        else Path(args.export_path)
    )
    args.train_output = (
        args.output_dir / "risk_critic_full_metrics.json"
        if Path(args.train_output) == DEFAULT_TRAIN_OUTPUT
        else Path(args.train_output)
    )
    args.summary_path = (
        args.output_dir / "summary.json"
        if Path(args.summary_path) == DEFAULT_SUMMARY_PATH
        else Path(args.summary_path)
    )
    args.policy_dir = Path(args.policy_dir)
    if args.action_tokenizer_path is not None:
        args.action_tokenizer_path = Path(args.action_tokenizer_path)
    if args.text_tokenizer_path is not None:
        args.text_tokenizer_path = Path(args.text_tokenizer_path)
    if args.video_dir is not None:
        args.video_dir = Path(args.video_dir)
    args.task_ids = _parse_int_list(str(args.task_ids))
    args.init_state_ids = _parse_int_list(str(args.init_state_ids))
    args.seeds = _parse_int_list(str(args.seeds))
    args.repair_replay_trials = max(1, int(args.repair_replay_trials))
    summary = run_large_eval(args)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.summary_path}")


if __name__ == "__main__":
    main()
