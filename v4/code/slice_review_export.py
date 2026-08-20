from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from risk_critic_export import _strict_causal_pass


CODE_ROOT = Path(__file__).resolve().parent
V4_ROOT = CODE_ROOT.parent
PROJECT_ROOT = V4_ROOT.parent
LEROBOT_PYTHON = Path(os.environ.get("LEROBOT_PYTHON", sys.executable))
DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "model_datasets"
    / "pi0fast-libero-libero_10"
    / "outputs"
    / "risk_critic_ultra_video_causal_v2_20260525"
    / "reports"
)
DEFAULT_OUTPUT_PARENT = (
    PROJECT_ROOT / "model_datasets" / "pi0fast-libero-libero_10" / "outputs"
)


@dataclass
class ReviewCase:
    report_path: Path
    report: dict
    case_id: str
    task_id: int
    init_state_id: int
    seed: int
    task_language: str
    source_video_path: Path
    rollout_length: int
    video_frames: int
    video_fps: int
    video_every_n: int
    minimal_start: int
    minimal_end: int
    repair_replay_start: int
    repair_replay_end: int
    repair_replay_source: str
    failure_type: str
    failed_goals: list[str]
    same_failure_rate: float
    base_same_failure_rate: float
    review_semantics: str
    full_success_repair_pass: bool
    necessity_core_units: list[dict]
    causal_core_units: list[dict]
    k_minimal_causal_sets: list[dict]
    repair_pass_variants: list[dict]
    best_counterfactual: dict


@dataclass
class ArchivedRollout:
    task_suite_name: str
    task_id: int
    init_state_id: int
    task_language: str
    target_key: str
    target_key_trace: list[str]
    actions: np.ndarray
    states_before_action: list[np.ndarray]
    failure_signature: object
    distance_trace: list[float]
    success: bool
    done_step: Optional[int]
    reset_seed: Optional[int]
    video_path: Optional[str] = None
    video_frames: int = 0
    snapshots: list[object] = field(default_factory=list)
    loaded_from_archive: bool = True
    archive_path: Optional[str] = None

    @property
    def length(self) -> int:
        return int(np.asarray(self.actions).shape[0])


def _env_without_proxy() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        env.pop(key, None)
    return env


def _first_cuda_device(cuda_visible_devices: object) -> str:
    first = str(cuda_visible_devices or "0").split(",", 1)[0].strip()
    return first or "0"


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except Exception:
        return False


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if _port_open(host, port):
            return
        time.sleep(2.0)
    raise TimeoutError(f"Policy server at {host}:{port} did not become ready")


def _start_lerobot_policy_server(args: argparse.Namespace, log_path: Path) -> Optional[subprocess.Popen]:
    if _port_open(args.policy_host, args.policy_port):
        if args.allow_existing_policy_server:
            return None
        raise RuntimeError(
            f"Policy port {args.policy_host}:{args.policy_port} is already open. "
            "Pass --allow-existing-policy-server to reuse it."
        )
    if not args.launch_policy_server:
        raise RuntimeError(
            "Replay recording requires a reachable policy server. "
            "Pass --launch-policy-server or start one separately."
        )
    cmd = [
        str(LEROBOT_PYTHON),
        str(CODE_ROOT / "serve_lerobot_pi0fast_policy.py"),
        "--port",
        str(args.policy_port),
        "--policy-dir",
        str(args.policy_dir),
        "--device",
        str(args.pytorch_device),
        "--compile-mode",
        str(args.pytorch_compile_mode),
        "--action-tokenizer-path",
        str(args.action_tokenizer_path),
        "--text-tokenizer-path",
        str(args.text_tokenizer_path),
        "--local-files-only",
    ]
    if args.pytorch_compile_mode and args.pytorch_compile_mode != "none":
        cmd.append("--compile-model")
    env = _env_without_proxy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.cuda_visible_devices),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(CODE_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    log_f.close()
    try:
        _wait_for_port(args.policy_host, args.policy_port, args.policy_ready_timeout)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        raise
    return proc


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _resolve_migrated_path(path: Path) -> Path:
    if _path_exists(path):
        return path
    text = str(path)
    candidates: list[Path] = []
    old_roots = [
        "/root/autodl-tmp/research/Embodied_Delta_Debugging",
        "/data2/yanghaoyun/research/Embodied_Delta_Debugging",
    ]
    for old_root in old_roots:
        if text.startswith(old_root):
            candidates.append(PROJECT_ROOT / text[len(old_root) + 1 :])
    marker = "Embodied_Delta_Debugging/"
    if marker in text:
        candidates.append(PROJECT_ROOT / text.split(marker, 1)[1])
    for candidate in candidates:
        if _path_exists(candidate):
            return candidate
    return path


def _failure_signature_from_dict(payload: dict):
    from causal_failure_predicates import FailureSignature

    anchor = payload.get("anchor_window") or payload.get("anchor_interval") or [0, 0]
    return FailureSignature(
        failure_type=str(payload.get("failure_type") or ""),
        mechanism=str(payload.get("mechanism") or payload.get("failure_type") or ""),
        failed_goal_predicates=tuple(str(x) for x in payload.get("failed_goal_predicates") or []),
        affected_objects=tuple(str(x) for x in payload.get("affected_objects") or []),
        anchor_start=int(anchor[0]) if len(anchor) >= 1 else 0,
        anchor_end=int(anchor[1]) if len(anchor) >= 2 else 0,
        semantic_quality=str(payload.get("semantic_quality") or "degraded"),
        confidence=float(payload.get("confidence") or 0.0),
        evidence=dict(payload.get("evidence") or {}),
    )


def _load_rollout_archive(case: ReviewCase) -> Optional[ArchivedRollout]:
    selected = case.report.get("selected_failed_rollout") or {}
    archive_value = selected.get("rollout_archive_path") or selected.get("archive_path")
    if not archive_value:
        return None
    archive_path = _resolve_migrated_path(Path(str(archive_value)))
    if not _path_exists(archive_path):
        return None
    data = np.load(archive_path, allow_pickle=False)
    try:
        metadata = json.loads(str(data["metadata_json"].item()))
        actions = np.asarray(data["actions"], dtype=np.float32)
        states_arr = np.asarray(data["states_before_action"], dtype=np.float64)
    finally:
        data.close()
    states = [np.asarray(item, dtype=np.float64) for item in states_arr]
    signature_payload = metadata.get("failure_signature") or case.report.get("original_failure_signature") or {}
    distance_trace = [float(x) for x in metadata.get("distance_trace") or []]
    if len(distance_trace) < actions.shape[0] + 1:
        distance_trace.extend([0.0] * (actions.shape[0] + 1 - len(distance_trace)))
    return ArchivedRollout(
        task_suite_name=str(metadata.get("task_suite_name") or selected.get("task_suite_name") or "libero_10"),
        task_id=int(metadata.get("task_id") if metadata.get("task_id") is not None else case.task_id),
        init_state_id=int(
            metadata.get("init_state_id")
            if metadata.get("init_state_id") is not None
            else case.init_state_id
        ),
        task_language=str(metadata.get("task_language") or case.task_language),
        target_key=str(metadata.get("target_key") or selected.get("target_key") or ""),
        target_key_trace=[str(x) for x in metadata.get("target_key_trace") or []],
        actions=actions,
        states_before_action=states,
        failure_signature=_failure_signature_from_dict(signature_payload),
        distance_trace=distance_trace,
        success=bool(metadata.get("success")),
        done_step=metadata.get("done_step"),
        reset_seed=metadata.get("reset_seed"),
        video_path=metadata.get("video_path"),
        video_frames=int(metadata.get("video_frames") or 0),
        archive_path=str(archive_path),
    )


def _report_paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*_causal_v*.json")))
            paths.extend(sorted(item.glob("*.json")))
        elif item.exists():
            paths.append(item)
    seen = set()
    unique = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _case_ids_from_path(path: Path) -> tuple[int, int, int]:
    match = re.search(r"task(\d+)_init(\d+)_seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse task/init/seed from {path.name}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]


def _first_interval(candidate: dict) -> tuple[int, int]:
    intervals = candidate.get("intervals") or []
    if intervals:
        return int(intervals[0][0]), int(intervals[0][1])
    span = candidate.get("span")
    if span:
        return int(span[0]), int(span[1])
    raise ValueError("Candidate has no intervals/span")


def _max_ce_core(cores: list[dict]) -> dict:
    if not cores:
        raise ValueError("Review report has no core units")
    return max(cores, key=lambda item: float(item.get("causal_effect") or 0.0))


def _interval_from_unit(unit: dict) -> Optional[tuple[int, int]]:
    interval = unit.get("interval")
    if not interval or len(interval) < 2:
        return None
    return int(interval[0]), int(interval[1])


def _interval_from_repair_variant(variant: dict) -> Optional[tuple[int, int]]:
    evaluation = variant.get("evaluation") or {}
    candidate = evaluation.get("candidate") or {}
    try:
        return _first_interval(candidate)
    except Exception:
        return None


def _all_repair_variants(report: dict) -> list[dict]:
    causal = report.get("causal_validation") or {}
    variants: list[dict] = []
    for key in (
        "policy_repair_pass_variants",
        "demo_repair_pass_variants",
        "repair_pass_variants",
    ):
        for item in report.get(key) or causal.get(key) or []:
            if isinstance(item, dict):
                variants.append(item)
    seen = set()
    unique = []
    for item in variants:
        digest = json.dumps(item, sort_keys=True, default=str)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(item)
    return unique


def _variant_interval_distance(
    variant: dict,
    start: Optional[int],
    end: Optional[int],
) -> tuple[int, int, str]:
    interval = _interval_from_repair_variant(variant)
    if interval is None or start is None:
        return (10**9, 10**9, str(variant.get("source") or ""))
    end_penalty = 0 if end is None else abs(int(interval[1]) - int(end))
    return (abs(int(interval[0]) - int(start)), end_penalty, str(variant.get("source") or ""))


def _select_repair_variant(
    case: ReviewCase,
    sources: Sequence[str],
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[dict]:
    source_set = {str(item) for item in sources}
    matches = [
        item
        for item in case.repair_pass_variants
        if str(item.get("source") or "") in source_set
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            not bool(item.get("repair_pass")),
            not bool((item.get("evaluation") or {}).get("success")),
            _variant_interval_distance(item, start, end),
        )
    )
    return matches[0]


def _source_metadata_without_actions(source_meta: object) -> object:
    if not isinstance(source_meta, dict):
        return source_meta
    return {
        k: v
        for k, v in source_meta.items()
        if k != "actions" and not isinstance(v, np.ndarray)
    }


def _safe_id(text: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return safe.strip("_") or "item"


def _k_minimal_sets_from_report(report: dict) -> list[dict]:
    causal = report.get("causal_validation") or {}
    raw_sets = report.get("k_minimal_causal_sets") or causal.get("k_minimal_causal_sets") or []
    sets = [item for item in raw_sets if isinstance(item, dict)]
    return sorted(
        sets,
        key=lambda item: (
            int(item.get("rank") or 10**9),
            str(item.get("set_id") or ""),
        ),
    )


def _interval_from_k_minimal_set(k_set: dict) -> Optional[tuple[int, int]]:
    for unit in k_set.get("units") or []:
        if not isinstance(unit, dict):
            continue
        interval = _interval_from_unit(unit)
        if interval is not None:
            return interval
    return None


def _unit_ids_from_k_minimal_set(k_set: dict) -> set[str]:
    ids: set[str] = set()
    for unit in k_set.get("units") or []:
        if isinstance(unit, dict) and unit.get("unit_id"):
            ids.add(str(unit.get("unit_id")))
    return ids


def _core_for_k_minimal_set(case: ReviewCase, k_set: dict) -> dict:
    unit_ids = _unit_ids_from_k_minimal_set(k_set)
    for core in list(case.causal_core_units or []) + list(case.necessity_core_units or []):
        unit = core.get("unit") or {}
        if str(unit.get("unit_id") or "") in unit_ids:
            return core
    interval = _interval_from_k_minimal_set(k_set) or (
        case.repair_replay_start,
        case.repair_replay_end,
    )
    units = [unit for unit in k_set.get("units") or [] if isinstance(unit, dict)]
    unit = dict(units[0]) if units else {}
    unit.setdefault("unit_id", str(k_set.get("set_id") or "k_minimal_set"))
    unit.setdefault("kind", "k_minimal_causal_set")
    unit.setdefault("interval", [int(interval[0]), int(interval[1])])
    return {
        "unit": unit,
        "causal_effect": float(k_set.get("causal_effect") or 0.0),
        "ablated_same_failure_rate": float(k_set.get("ablated_same_failure_rate") or 0.0),
        "repair_pass": bool(k_set.get("repair_valid")),
        "best_counterfactual": case.best_counterfactual,
    }


def _case_for_k_minimal_set(case: ReviewCase, k_set: dict) -> ReviewCase:
    rank = int(k_set.get("rank") or 0)
    interval = _interval_from_k_minimal_set(k_set) or (
        case.repair_replay_start,
        case.repair_replay_end,
    )
    core = _core_for_k_minimal_set(case, k_set)
    sources = ",".join(str(item) for item in k_set.get("repair_sources") or []) or "necessity"
    best_counterfactual = core.get("best_counterfactual") or case.best_counterfactual
    return replace(
        case,
        repair_replay_start=int(interval[0]),
        repair_replay_end=int(interval[1]),
        repair_replay_source=f"k_minimal_rank_{rank}:{sources}",
        causal_core_units=[core] if case.causal_core_units else [],
        necessity_core_units=[core] if not case.causal_core_units else case.necessity_core_units,
        best_counterfactual=best_counterfactual,
    )


def _summarize_k_minimal_set(k_set: dict) -> dict:
    units = []
    for unit in k_set.get("units") or []:
        if not isinstance(unit, dict):
            continue
        units.append(
            {
                "unit_id": unit.get("unit_id"),
                "kind": unit.get("kind"),
                "interval": unit.get("interval"),
                "phrase": unit.get("phrase") or unit.get("original_phrase"),
                "visual_intervention": unit.get("visual_intervention")
                or unit.get("mask_type"),
            }
        )
    return {
        "rank": int(k_set.get("rank") or 0),
        "set_id": str(k_set.get("set_id") or ""),
        "bounded_minimal": bool(k_set.get("bounded_minimal")),
        "minimality_scope": str(k_set.get("minimality_scope") or ""),
        "causal_effect": float(k_set.get("causal_effect") or 0.0),
        "same_failure_rate": float(k_set.get("same_failure_rate") or 0.0),
        "ablated_same_failure_rate": float(k_set.get("ablated_same_failure_rate") or 0.0),
        "repair_sources": [str(item) for item in k_set.get("repair_sources") or []],
        "repair_valid": bool(k_set.get("repair_valid")),
        "policy_strong_repair_valid": bool(k_set.get("policy_strong_repair_valid")),
        "full_success_repair": bool(k_set.get("full_success_repair")),
        "drop_one_unit_checks": k_set.get("drop_one_unit_checks") or [],
        "units": units,
    }


def _repair_replay_context(
    report: dict,
    repair_valid: bool,
    cores_for_review: list[dict],
    minimal_interval: tuple[int, int],
) -> tuple[int, int, str]:
    """Choose the state anchor used for repair videos.

    The minimal same-failure slice can be much shorter than the actual repair
    context.  For example, a two-step wrong-placement slice may only be
    repairable when the policy replans from the earlier grasp/transport core.
    Review videos should therefore use the interval that produced the repair
    evidence in the causal report, while still recording the minimal slice
    separately in case_review.json.
    """

    causal = report.get("causal_validation") or {}
    variant_groups = [
        report.get("policy_repair_pass_variants")
        or causal.get("policy_repair_pass_variants")
        or [],
        report.get("demo_repair_pass_variants")
        or causal.get("demo_repair_pass_variants")
        or [],
        report.get("repair_pass_variants") or causal.get("repair_pass_variants") or [],
    ]
    variants = [item for group in variant_groups for item in group if isinstance(item, dict)]
    if repair_valid:
        passing_variants = [item for item in variants if bool(item.get("repair_pass"))]
        passing_variants.sort(
            key=lambda item: (
                not bool((item.get("evaluation") or {}).get("success")),
                str(item.get("source") or ""),
            )
        )
        for variant in passing_variants:
            interval = _interval_from_repair_variant(variant)
            if interval is not None:
                return interval[0], interval[1], str(variant.get("source") or "repair_variant")

    if cores_for_review:
        best_core = _max_ce_core(cores_for_review)
        interval = _interval_from_unit(best_core.get("unit") or {})
        if interval is not None:
            return interval[0], interval[1], str(
                (best_core.get("unit") or {}).get("kind") or "causal_core"
            )

    return minimal_interval[0], minimal_interval[1], "minimal_slice"


def _same_failure_necessity_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    return bool(
        report.get("same_failure_necessity_pass")
        or causal.get("same_failure_necessity_pass")
    )


def _full_success_repair_pass(report: dict) -> bool:
    causal = report.get("causal_validation") or {}
    if report.get("schema_version") == "shed-cfs-causal-v4-global-multimodal":
        if bool(
            report.get("full_success_policy_repair_pass")
            or causal.get("full_success_policy_repair_pass")
            or causal.get("full_success_repair_pass")
        ):
            return True
        for item in (
            report.get("policy_repair_pass_variants")
            or causal.get("policy_repair_pass_variants")
            or []
        ):
            evaluation = item.get("evaluation") or {}
            repair_evidence = item.get("repair_evidence") or {}
            if bool(evaluation.get("success")) or bool(repair_evidence.get("success")):
                return True
        return False
    if report.get("schema_version") == "shed-cfs-causal-v3-multimodal":
        if bool(
            report.get("full_success_policy_repair_pass")
            or causal.get("full_success_policy_repair_pass")
        ):
            return True
        for item in (
            report.get("policy_repair_pass_variants")
            or causal.get("policy_repair_pass_variants")
            or []
        ):
            evaluation = item.get("evaluation") or {}
            repair_evidence = item.get("repair_evidence") or {}
            if bool(evaluation.get("success")) or bool(repair_evidence.get("success")):
                return True
        return False
    if bool(report.get("full_success_repair_pass") or causal.get("full_success_repair_pass")):
        return True
    for item in report.get("repair_pass_variants") or causal.get("repair_pass_variants") or []:
        evaluation = item.get("evaluation") or {}
        repair_evidence = item.get("repair_evidence") or {}
        if bool(evaluation.get("success")) or bool(repair_evidence.get("success")):
            return True
    for unit in report.get("causal_core_units") or causal.get("causal_core_units") or []:
        evidence = unit.get("repair_evidence") or {}
        best = unit.get("best_counterfactual") or {}
        evaluation = best.get("evaluation") or {}
        best_evidence = best.get("repair_evidence") or {}
        if (
            bool(evidence.get("success"))
            or bool(best_evidence.get("success"))
            or bool(evaluation.get("success"))
        ):
            return True
    return False


def _build_review_case(path: Path, include_necessity_only: bool = False) -> Optional[ReviewCase]:
    report = _load_json(path)
    repair_valid = _strict_causal_pass(report)
    full_success = _full_success_repair_pass(report)
    necessity_only = bool(
        include_necessity_only and not repair_valid and _same_failure_necessity_pass(report)
    )
    if not repair_valid and not necessity_only:
        return None
    task_id, init_state_id, seed = _case_ids_from_path(path)
    selected = report.get("selected_failed_rollout") or {}
    video_path = _resolve_migrated_path(Path(selected.get("video_path") or ""))
    if not _path_exists(video_path):
        raise FileNotFoundError(f"Source video missing for {path.name}: {video_path}")
    candidate = (
        (report.get("reproduction_statistics") or {}).get("candidate")
        or report.get("causal_failure_slice")
        or ((report.get("minimal_replay_context") or {}).get("candidate_actions") or {})
    )
    minimal_start, minimal_end = _first_interval(candidate)
    signature = report.get("original_failure_signature") or {}
    repro = report.get("reproduction_statistics") or {}
    causal = report.get("causal_validation") or {}
    necessity_cores = report.get("necessity_core_units") or causal.get("necessity_core_units") or []
    repair_cores = report.get("causal_core_units") or causal.get("causal_core_units") or []
    k_minimal_sets = _k_minimal_sets_from_report(report)
    cores_for_review = repair_cores if repair_valid else necessity_cores
    best_core = _max_ce_core(cores_for_review)
    repair_start, repair_end, repair_source = _repair_replay_context(
        report,
        repair_valid,
        cores_for_review,
        (minimal_start, minimal_end),
    )
    return ReviewCase(
        report_path=path,
        report=report,
        case_id=f"task{task_id:02d}_init{init_state_id:02d}_seed{seed:02d}",
        task_id=task_id,
        init_state_id=init_state_id,
        seed=seed,
        task_language=str(selected.get("task_language") or ""),
        source_video_path=video_path,
        rollout_length=int(selected.get("length") or 0),
        video_frames=int(selected.get("video_frames") or 0),
        video_fps=int((report.get("video_config") or {}).get("video_fps") or 30),
        video_every_n=int((report.get("video_config") or {}).get("video_every_n") or 1),
        minimal_start=minimal_start,
        minimal_end=minimal_end,
        repair_replay_start=repair_start,
        repair_replay_end=repair_end,
        repair_replay_source=repair_source,
        failure_type=str(signature.get("failure_type") or ""),
        failed_goals=[str(item) for item in signature.get("failed_goal_predicates") or []],
        same_failure_rate=float(repro.get("same_failure_rate") or 0.0),
        base_same_failure_rate=float(causal.get("base_same_failure_rate") or 0.0),
        review_semantics=(
            "repair_valid_success"
            if repair_valid and full_success
            else "repair_valid_causal_pass"
            if repair_valid
            else "same_failure_necessity_only"
        ),
        full_success_repair_pass=full_success,
        necessity_core_units=necessity_cores,
        causal_core_units=repair_cores,
        k_minimal_causal_sets=k_minimal_sets,
        repair_pass_variants=_all_repair_variants(report),
        best_counterfactual=best_core.get("best_counterfactual") or {},
    )


def collect_review_cases(
    paths: Sequence[Path],
    max_cases: int = 0,
    include_necessity_only: bool = False,
) -> list[ReviewCase]:
    cases = []
    for path in _report_paths(paths):
        case = _build_review_case(path, include_necessity_only=include_necessity_only)
        if case is not None:
            cases.append(case)
        if max_cases > 0 and len(cases) >= max_cases:
            break
    return cases


def frame_window_for_slice(
    start: int,
    end: int,
    fps: int,
    context_seconds: float,
    total_frames: int,
) -> tuple[int, int]:
    context = int(round(float(context_seconds) * int(fps)))
    first = max(0, int(start) - context)
    last = min(max(0, int(total_frames) - 1), int(end) + context)
    return first, max(first, last)


def _video_frame_count(path: Path) -> int:
    reader = imageio.get_reader(str(path))
    try:
        try:
            return int(reader.count_frames())
        except Exception:
            meta = reader.get_meta_data()
            nframes = meta.get("nframes")
            if isinstance(nframes, int) and nframes > 0:
                return nframes
            count = 0
            for _ in reader:
                count += 1
            return count
    finally:
        reader.close()


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def _put_text_lines(frame: np.ndarray, lines: Sequence[str], origin: tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_h = 17
    max_width = 0
    for line in lines:
        (w, _h), _baseline = cv2.getTextSize(line, font, scale, thickness)
        max_width = max(max_width, w)
    cv2.rectangle(
        frame,
        (x - 5, y - 14),
        (x + max_width + 6, y + line_h * len(lines) + 3),
        (0, 0, 0),
        -1,
    )
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def _intervals(case: ReviewCase) -> list[tuple[int, int, str]]:
    items = [(case.minimal_start, case.minimal_end, "minimal")]
    cores = case.causal_core_units or case.necessity_core_units
    for core in cores:
        unit = core.get("unit") or {}
        interval = unit.get("interval")
        kind = str(unit.get("kind") or "core")
        if interval and len(interval) == 2:
            items.append((int(interval[0]), int(interval[1]), kind))
    return items


def annotate_frame(
    frame: np.ndarray,
    case: ReviewCase,
    step: int,
    panel: str,
    variant: str,
    trial: Optional[int] = None,
) -> np.ndarray:
    out = _ensure_uint8(frame).copy()
    in_minimal = case.minimal_start <= step < case.minimal_end
    in_core = any(
        start <= step < end and label != "minimal" for start, end, label in _intervals(case)
    )
    color = (40, 180, 255) if in_minimal else (80, 220, 80) if in_core else (80, 80, 80)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 4)
    cores = case.causal_core_units or case.necessity_core_units
    ce_values = [float(core.get("causal_effect") or 0.0) for core in cores]
    ce_max = max(ce_values) if ce_values else 0.0
    core_text = ",".join(
        f"{(core.get('unit') or {}).get('kind','core')}:{(core.get('unit') or {}).get('interval')}"
        for core in cores[:2]
    )
    goal_text = " | ".join(case.failed_goals)
    if len(goal_text) > 76:
        goal_text = goal_text[:73] + "..."
    lines = [
        f"{panel} {variant}" + ("" if trial is None else f" trial={trial:02d}"),
        f"{case.case_id} step={step} minimal=[{case.minimal_start},{case.minimal_end})",
        f"context=[{case.repair_replay_start},{case.repair_replay_end}) source={case.repair_replay_source}",
        f"{case.review_semantics} type={case.failure_type} same={case.same_failure_rate:.2f} base={case.base_same_failure_rate:.2f} CEmax={ce_max:.2f}",
        f"goals={goal_text}",
        f"core={core_text}",
    ]
    _put_text_lines(out, lines, (8, 22))
    return out


def _write_video(frames: Iterable[np.ndarray], out_path: Path, fps: int, quality: int, codec: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "imageio_ffmpeg is required to write H.264 Main/yuv420p review videos"
        ) from exc
    count = 0
    # imageio quality=10 maps to x264 lossless, which H.264 Main profile
    # cannot encode. Keep the user-facing quality knob, but translate it to
    # a high-quality non-lossless CRF that is compatible with Main+yuv420p.
    q = max(1, min(10, int(quality)))
    crf = max(18, min(34, 28 - q))
    writer = imageio.get_writer(
        str(out_path),
        fps=int(fps),
        codec="libx264",
        quality=None,
        macro_block_size=1,
        output_params=[
            "-crf",
            str(crf),
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    try:
        for frame in frames:
            writer.append_data(_ensure_uint8(frame))
            count += 1
    finally:
        writer.close()
    return count


def _read_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    frames: list[np.ndarray] = []
    try:
        for frame in reader:
            frames.append(_ensure_uint8(frame))
    finally:
        reader.close()
    return frames


def _pad_to_shape(frame: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    arr = _ensure_uint8(frame)
    out = np.zeros(shape, dtype=np.uint8)
    h, w = arr.shape[:2]
    H, W = shape[:2]
    y0 = max(0, (H - h) // 2)
    x0 = max(0, (W - w) // 2)
    h2 = min(h, H)
    w2 = min(w, W)
    out[y0 : y0 + h2, x0 : x0 + w2] = arr[:h2, :w2]
    return out


def _make_concat_reel(
    segments: Sequence[tuple[str, Path]],
    out_path: Path,
    fps: int,
    quality: int,
    codec: str,
) -> int:
    usable = [(title, path) for title, path in segments if path is not None and path.exists()]
    if not usable:
        return 0
    shapes = []
    for _title, path in usable:
        reader = imageio.get_reader(str(path))
        try:
            shapes.append(_ensure_uint8(reader.get_data(0)).shape)
        except Exception:
            pass
        finally:
            reader.close()
    if not shapes:
        return 0
    shape = (
        max(s[0] for s in shapes),
        max(s[1] for s in shapes),
        3,
    )

    def frames() -> Iterable[np.ndarray]:
        for title, path in usable:
            title_card = np.full(shape, 24, dtype=np.uint8)
            _put_text_lines(title_card, [title], (24, 48))
            for _ in range(max(1, int(round(0.75 * fps)))):
                yield title_card.copy()
            reader = imageio.get_reader(str(path))
            try:
                for frame in reader:
                    yield _pad_to_shape(frame, shape)
            finally:
                reader.close()

    return _write_video(frames(), out_path, fps, quality, codec)


def _write_video_cv2(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> int:
    writer = None
    count = 0
    try:
        for frame in frames:
            arr = _ensure_uint8(frame)
            if writer is None:
                h, w = arr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open cv2 VideoWriter for {out_path}")
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return count


def make_original_clip(
    case: ReviewCase,
    out_path: Path,
    start_frame: int,
    end_frame: int,
    fps: int,
    quality: int,
    codec: str,
    panel: str,
) -> int:
    reader = imageio.get_reader(str(case.source_video_path))

    def frames() -> Iterable[np.ndarray]:
        try:
            for idx in range(start_frame, end_frame + 1):
                yield annotate_frame(reader.get_data(idx), case, idx, panel, "original")
        finally:
            reader.close()

    return _write_video(frames(), out_path, fps, quality, codec)


def _read_video_frame(reader, idx: int, last: Optional[np.ndarray]) -> np.ndarray:
    try:
        return _ensure_uint8(reader.get_data(idx))
    except Exception:
        if last is None:
            raise
        return last.copy()


def make_triptych(
    left_path: Path,
    middle_path: Path,
    right_path: Path,
    out_path: Path,
    labels: tuple[str, str, str],
    fps: int,
    quality: int,
    codec: str,
) -> int:
    readers = [imageio.get_reader(str(path)) for path in (left_path, middle_path, right_path)]
    counts = []
    for reader in readers:
        try:
            counts.append(int(reader.count_frames()))
        except Exception:
            counts.append(0)
    total = max(counts)
    last_frames: list[Optional[np.ndarray]] = [None, None, None]

    def frames() -> Iterable[np.ndarray]:
        try:
            for idx in range(total):
                panels = []
                for i, reader in enumerate(readers):
                    if counts[i] and idx >= counts[i] and last_frames[i] is not None:
                        frame = last_frames[i].copy()
                    else:
                        frame = _read_video_frame(reader, idx, last_frames[i])
                        last_frames[i] = frame
                    cv2.putText(
                        frame,
                        labels[i],
                        (10, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    panels.append(frame)
                yield np.concatenate(panels, axis=1)
        finally:
            for reader in readers:
                reader.close()

    return _write_video(frames(), out_path, fps, quality, codec)


def make_multipanel(
    paths: Sequence[Optional[Path]],
    out_path: Path,
    labels: Sequence[str],
    fps: int,
    quality: int,
    codec: str,
    columns: Optional[int] = None,
) -> int:
    readers = [None if path is None else imageio.get_reader(str(path)) for path in paths]
    counts = []
    sample_shape = None
    for reader in readers:
        if reader is None:
            counts.append(0)
            continue
        try:
            counts.append(int(reader.count_frames()))
        except Exception:
            counts.append(0)
        if sample_shape is None:
            try:
                sample_shape = _ensure_uint8(reader.get_data(0)).shape
            except Exception:
                pass
    total = max(counts) if counts else 0
    if total <= 0:
        total = 1
    if sample_shape is None:
        sample_shape = (512, 512, 3)
    last_frames: list[Optional[np.ndarray]] = [None for _ in readers]
    panel_count = max(len(readers), len(labels))
    grid_columns = int(columns or (3 if panel_count == 6 else max(1, panel_count)))
    grid_columns = max(1, min(grid_columns, max(1, panel_count)))
    grid_rows = int(math.ceil(panel_count / grid_columns))

    def unavailable_frame(label: str) -> np.ndarray:
        frame = np.full(sample_shape, 96, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (220, 180, 80), 4)
        _put_text_lines(
            frame,
            [
                label,
                "UNAVAILABLE",
                "component replay was not generated",
            ],
            (18, 36),
        )
        return frame

    def frames() -> Iterable[np.ndarray]:
        try:
            for idx in range(total):
                panels = []
                for i in range(panel_count):
                    reader = readers[i] if i < len(readers) else None
                    label = labels[i] if i < len(labels) else f"panel {i}"
                    if reader is None:
                        frame = unavailable_frame(label)
                    elif counts[i] and idx >= counts[i] and last_frames[i] is not None:
                        frame = last_frames[i].copy()
                    else:
                        frame = _read_video_frame(reader, min(idx, max(0, counts[i] - 1)), last_frames[i])
                        last_frames[i] = frame
                    frame = _pad_to_shape(frame, sample_shape)
                    cv2.putText(
                        frame,
                        label,
                        (10, frame.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    panels.append(frame)
                rows = []
                blank = unavailable_frame("")
                for row_idx in range(grid_rows):
                    start = row_idx * grid_columns
                    row_panels = panels[start : start + grid_columns]
                    while len(row_panels) < grid_columns:
                        row_panels.append(blank.copy())
                    rows.append(np.concatenate(row_panels, axis=1))
                yield np.concatenate(rows, axis=0)
        finally:
            for reader in readers:
                if reader is not None:
                    reader.close()

    return _write_video(frames(), out_path, fps, quality, codec)


def make_quadriptych(
    original_path: Path,
    minimal_path: Path,
    policy_repair_path: Optional[Path],
    source_repair_path: Optional[Path],
    out_path: Path,
    labels: tuple[str, str, str, str],
    fps: int,
    quality: int,
    codec: str,
) -> int:
    return make_multipanel(
        [original_path, minimal_path, policy_repair_path, source_repair_path],
        out_path,
        labels,
        fps,
        quality,
        codec,
    )


def _build_probe_args(args: argparse.Namespace, case: ReviewCase):
    from pi05_natural_failure_probe import RuntimeProfile

    ns = argparse.Namespace(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        policy_config=args.policy_config,
        policy_checkpoint=str(args.policy_dir),
        task_suite_name=args.task_suite_name,
        task_ids=[case.task_id],
        init_state_ids=[case.init_state_id],
        seed=case.seed,
        gpu_id=0,
        camera_size=args.camera_size,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        event_window=args.event_window,
        min_distance_delta=0.03,
        continuation="recorded",
        replay_trials=args.replay_trials_to_record,
        search_replay_trials=1,
        confirm_replay_trials=args.replay_trials_to_record,
        same_failure_threshold=0.75,
        accept_same_failure_rate=0.80,
        causal_effect_threshold=0.30,
        causal_chunk_size=5,
        demo_dataset_root=args.demo_dataset_root,
        demo_repair_timeout_seconds=args.demo_repair_timeout_seconds,
        record_video=False,
        video_dir=None,
        video_prefix="",
        video_camera=args.video_camera,
        video_fps=args.video_fps,
        video_every_n=1,
        video_codec=args.video_codec,
        video_quality=args.video_quality,
        video_no_flip=args.video_no_flip,
        output=Path("/tmp/slice_review_unused.json"),
    )
    ns._runtime_profile = RuntimeProfile.create()
    return ns


def _interval_check(
    name: str,
    interval: tuple[int, int],
    action_length: int,
    state_count: int,
    *,
    needs_action_end: bool = True,
) -> dict:
    start, end = int(interval[0]), int(interval[1])
    reasons: list[str] = []
    if start < 0 or end < start:
        reasons.append("invalid_interval")
    if start >= state_count:
        reasons.append("start_state_missing")
    if needs_action_end and end > action_length:
        reasons.append("end_exceeds_action_trace")
    if needs_action_end and start >= action_length:
        reasons.append("start_exceeds_action_trace")
    return {
        "name": str(name),
        "interval": [start, end],
        "needs_action_end": bool(needs_action_end),
        "in_bounds": not reasons,
        "reasons": reasons,
    }


def _coordinate_validation_for_rollout(
    case: ReviewCase,
    rollout,
    best_interval: tuple[int, int],
) -> dict:
    action_length = int(np.asarray(getattr(rollout, "actions", [])).shape[0])
    state_count = int(len(getattr(rollout, "states_before_action", []) or []))
    report_length = int(case.rollout_length or 0)
    length_matches = bool(report_length <= 0 or report_length == action_length)
    interval_checks = [
        _interval_check(
            "minimal_same_failure_slice",
            (case.minimal_start, case.minimal_end),
            action_length,
            state_count,
        ),
        _interval_check(
            "repair_replay_context",
            (case.repair_replay_start, case.repair_replay_end),
            action_length,
            state_count,
        ),
        _interval_check("destructive_core_unit", best_interval, action_length, state_count),
    ]
    reasons: list[str] = []
    if not length_matches:
        reasons.append("fresh_rollout_length_differs_from_report")
    for item in interval_checks:
        if not bool(item.get("in_bounds")):
            reasons.append(f"{item.get('name')}:out_of_bounds")
    strict = bool(length_matches and all(bool(item.get("in_bounds")) for item in interval_checks))
    return {
        "schema_version": "shed-cfs-review-coordinate-validation-v1",
        "strict_coordinate_match": strict,
        "coordinate_mismatch": not strict,
        "report_action_length": report_length,
        "fresh_action_length": action_length,
        "fresh_state_count": state_count,
        "length_matches_report": length_matches,
        "interval_checks": interval_checks,
        "reasons": reasons,
        "note": (
            "Strict source-aware evidence requires the review rollout action/state "
            "coordinate frame to match the causal report. Degraded videos may still "
            "be useful for inspection, but they are not counted as strict repair proof."
        ),
    }


def _interval_is_in_bounds(validation: dict, name: str) -> bool:
    for item in validation.get("interval_checks") or []:
        if item.get("name") == name:
            return bool(item.get("in_bounds"))
    return False


def _unavailable_replay_meta(
    variant: str,
    reason: str,
    *,
    trial: int,
    repair_source: Optional[str] = None,
    coordinate_validation: Optional[dict] = None,
    report_variant: Optional[dict] = None,
) -> dict:
    return {
        "available": False,
        "variant": str(variant),
        "trial": int(trial),
        "reason": str(reason),
        "repair_source": repair_source,
        "report_variant": report_variant,
        "coordinate_validation": coordinate_validation,
        "strict_source_aware_evidence": False,
        "coordinate_mismatch": bool(
            (coordinate_validation or {}).get("coordinate_mismatch")
        ),
    }


def _record_replay_component(
    args: argparse.Namespace,
    case: ReviewCase,
    rollout,
    start: int,
    replacements: Optional[dict[int, np.ndarray]],
    out_path: Path,
    variant: str,
    trial: int,
    policy_client=None,
    policy_from_step: Optional[int] = None,
    external_actions: Optional[np.ndarray] = None,
    prompt_override: Optional[str] = None,
    visual_intervention: Optional[dict] = None,
    repair_source: Optional[str] = None,
    coordinate_validation: Optional[dict] = None,
) -> dict:
    from causal_failure_predicates import (
        compare_failure_signatures,
        get_goal_predicates,
        infer_failure_signature,
        make_state_snapshot,
        semantic_quality_for_env,
    )
    from pi05_natural_failure_probe import (
        _distance,
        _make_env,
        _policy_observation,
        _set_state_and_obs,
        _video_frame_from_obs,
    )

    env, _task_suite, _task = _make_env(args, rollout.task_suite_name, rollout.task_id)
    frames_written = 0
    success = False
    try:
        env.reset()
        if start < 0 or start >= len(rollout.states_before_action):
            return _unavailable_replay_meta(
                variant,
                "start_state_missing_for_review_rollout",
                trial=trial,
                repair_source=repair_source,
                coordinate_validation=coordinate_validation,
            )
        obs = _set_state_and_obs(env, rollout.states_before_action[start])
        predicates = get_goal_predicates(env)
        semantic_quality = semantic_quality_for_env(env)
        snapshots = [
            make_state_snapshot(start, obs, env=env, success=False, predicates=predicates)
        ]

        rendered_frames = []
        try:
            frame = _video_frame_from_obs(obs, args.video_camera, flip=not bool(args.video_no_flip))
            rendered_frames.append(annotate_frame(frame, case, start, "REPLAY", variant, trial))
            frames_written += 1
            last_obs = obs
            action_plan: list[np.ndarray] = []
            external_arr = None
            if external_actions is not None:
                external_arr = np.asarray(external_actions, dtype=np.float32)
                if external_arr.ndim == 1 and external_arr.size > 0:
                    external_arr = external_arr.reshape(1, -1)
            for t in range(start, rollout.length):
                if external_arr is not None:
                    idx = t - start
                    if idx < 0 or idx >= external_arr.shape[0]:
                        break
                    action = np.asarray(external_arr[idx], dtype=np.float32)
                elif (
                    policy_from_step is not None
                    and t >= int(policy_from_step)
                    and policy_client is not None
                ):
                    if not action_plan:
                        element = _policy_observation(
                            last_obs,
                            prompt_override or rollout.task_language,
                            args.resize_size,
                            visual_intervention=visual_intervention,
                        )
                        chunk = policy_client.infer(element)["actions"]
                        action_plan.extend(
                            np.asarray(chunk, dtype=np.float32)[: args.replan_steps]
                        )
                    action = np.asarray(action_plan.pop(0), dtype=np.float32)
                else:
                    action = np.asarray(rollout.actions[t], dtype=np.float32)
                if replacements is not None and t in replacements:
                    action = np.asarray(replacements[t], dtype=np.float32)
                last_obs, _reward, done, _info = env.step(action.tolist())
                success = success or bool(done)
                snapshots.append(
                    make_state_snapshot(
                        t + 1,
                        last_obs,
                        env=env,
                        action=action,
                        success=success,
                        predicates=predicates,
                    )
                )
                frame = _video_frame_from_obs(last_obs, args.video_camera, flip=not bool(args.video_no_flip))
                rendered_frames.append(
                    annotate_frame(frame, case, t + 1, "REPLAY", variant, trial)
                )
                frames_written += 1
                if done:
                    break
        finally:
            pass
        _write_video(
            rendered_frames,
            out_path,
            args.video_fps,
            args.video_quality,
            args.video_codec,
        )
        signature = infer_failure_signature(
            snapshots,
            predicates=predicates,
            semantic_quality=semantic_quality,
            event_window=args.event_window,
            task_language=rollout.task_language,
        )
        evidence = compare_failure_signatures(
            rollout.failure_signature,
            signature,
            threshold=args.same_failure_threshold,
        )
        return {
            "available": True,
            "video_path": str(out_path),
            "frames": int(frames_written),
            "variant": str(variant),
            "repair_source": repair_source,
            "prompt_override_used": prompt_override is not None,
            "prompt_override": prompt_override,
            "visual_intervention_used": visual_intervention is not None,
            "visual_intervention": visual_intervention,
            "success": bool(success),
            "same_failure": bool(evidence.same_failure),
            "semantic_match_score": float(evidence.score),
            "failure_signature": signature.to_dict(),
            "end_distance": float(_distance(last_obs, rollout.target_key)),
            "coordinate_validation": coordinate_validation,
            "strict_source_aware_evidence": bool(
                (coordinate_validation or {}).get("strict_coordinate_match", True)
            ),
            "coordinate_mismatch": bool(
                (coordinate_validation or {}).get("coordinate_mismatch", False)
            ),
        }
    finally:
        env.close()


def _record_replay_triptychs(
    args: argparse.Namespace,
    case: ReviewCase,
    case_dir: Path,
    *,
    probe_args=None,
    client=None,
    rollout=None,
    filename_prefix: str = "",
    label_prefix: str = "",
) -> list[dict]:
    from openpi_client import websocket_client_policy
    from pi05_natural_failure_probe import (
        _demo_repair_actions,
        _replacement_actions,
        collect_pi05_rollout,
    )

    if probe_args is None:
        probe_args = _build_probe_args(args, case)
    if client is None:
        client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
    if rollout is None:
        rollout = _load_rollout_archive(case)
        if rollout is None:
            rollout = collect_pi05_rollout(
                probe_args,
                client,
                args.task_suite_name,
                case.task_id,
                case.init_state_id,
            )
    start = int(case.repair_replay_start)
    repair_end = int(case.repair_replay_end)
    best_strategy = str(case.best_counterfactual.get("strategy") or "hold")
    best_unit = case.best_counterfactual.get("unit_id") or ""
    best_interval = None
    cores = case.causal_core_units or case.necessity_core_units
    for core in cores:
        unit = core.get("unit") or {}
        if unit.get("unit_id") == best_unit:
            best_interval = tuple(int(v) for v in unit.get("interval"))
            break
    if best_interval is None:
        unit = (_max_ce_core(cores).get("unit") or {})
        best_interval = tuple(int(v) for v in unit.get("interval"))
    coordinate_validation = _coordinate_validation_for_rollout(case, rollout, best_interval)
    destructive_in_bounds = _interval_is_in_bounds(
        coordinate_validation, "destructive_core_unit"
    )
    replay_start_in_bounds = _interval_is_in_bounds(
        coordinate_validation, "repair_replay_context"
    )
    replacements = (
        _replacement_actions(rollout.actions, best_interval, best_strategy)
        if destructive_in_bounds
        else None
    )

    prefix = _safe_id(filename_prefix)
    prefix = "" if prefix == "item" and not filename_prefix else f"{prefix}_"
    label_prefix = str(label_prefix or "").strip()
    label_suffix = "" if not label_prefix else f"{label_prefix} "

    original_suffix = case_dir / f"{prefix}original_suffix.mp4"
    total_frames = case.video_frames or _video_frame_count(case.source_video_path)
    make_original_clip(
        case,
        original_suffix,
        start,
        total_frames - 1,
        args.video_fps,
        args.video_quality,
        args.video_codec,
        "ORIGINAL",
    )

    raw_variant = _select_repair_variant(
        case,
        ("policy_replan_from_pre_state",),
        start,
        repair_end,
    )
    language_variant = _select_repair_variant(
        case,
        ("policy_language_disambiguation_repair",),
        start,
        repair_end,
    )
    visual_variant = _select_repair_variant(
        case,
        ("policy_visual_mask_repair",),
        start,
        repair_end,
    )
    demo_variant = _select_repair_variant(
        case,
        ("success_or_demo_nn_repair", "demo_repair", "demo_or_success_repair"),
        start,
        repair_end,
    )

    language_intervention = None
    if language_variant is not None:
        language_intervention = language_variant.get("language_intervention")
    language_prompt = None
    if isinstance(language_intervention, dict):
        language_prompt = language_intervention.get("prompt")
    if language_prompt is not None:
        language_prompt = str(language_prompt)

    visual_intervention = None
    if visual_variant is not None:
        visual_intervention = visual_variant.get("visual_policy_mask_intervention")
    if not isinstance(visual_intervention, dict) or not bool(
        visual_intervention.get("applied_to_policy_input")
    ):
        visual_intervention = None

    source_meta = None if demo_variant is None else demo_variant.get("source_metadata")
    if (
        (not isinstance(source_meta, dict) or source_meta.get("actions") is None)
        and bool(getattr(args, "enable_demo_repair_lookup", False))
        and bool(getattr(rollout, "snapshots", []))
    ):
        source_meta = _demo_repair_actions(
            probe_args,
            rollout,
            start,
            max(1, rollout.length - start),
        )

    trials = []
    for trial in range(int(args.replay_trials_to_record)):
        minimal_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_minimal_replay.mp4"
        destructive_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_destructive_{best_strategy}.mp4"
        policy_repair_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_policy_repair.mp4"
        language_repair_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_language_repair.mp4"
        visual_repair_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_visual_repair.mp4"
        source_repair_path = case_dir / "components" / f"{prefix}trial_{trial:02d}_demo_or_success_repair.mp4"
        triptych_path = case_dir / f"{prefix}trial_{trial:02d}_destructive_triptych.mp4"
        multisource_path = case_dir / f"{prefix}trial_{trial:02d}_repair_multisource.mp4"
        minimal_panel_path: Optional[Path] = None
        destructive_panel_path: Optional[Path] = None
        policy_panel_path: Optional[Path] = None
        if replay_start_in_bounds:
            minimal_meta = _record_replay_component(
                probe_args,
                case,
                rollout,
                start,
                None,
                minimal_path,
                "minimal",
                trial,
                coordinate_validation=coordinate_validation,
            )
            if minimal_meta.get("available") is not False:
                minimal_panel_path = minimal_path
        else:
            minimal_meta = _unavailable_replay_meta(
                "minimal",
                "repair_replay_context_start_out_of_bounds",
                trial=trial,
                coordinate_validation=coordinate_validation,
            )
        if replay_start_in_bounds and destructive_in_bounds:
            destructive_meta = _record_replay_component(
                probe_args,
                case,
                rollout,
                start,
                replacements,
                destructive_path,
                f"destructive:{best_strategy}",
                trial,
                coordinate_validation=coordinate_validation,
            )
            if destructive_meta.get("available") is not False:
                destructive_panel_path = destructive_path
        else:
            destructive_meta = _unavailable_replay_meta(
                f"destructive:{best_strategy}",
                (
                    "destructive_core_unit_out_of_bounds"
                    if replay_start_in_bounds
                    else "repair_replay_context_start_out_of_bounds"
                ),
                trial=trial,
                coordinate_validation=coordinate_validation,
            )
        policy_panel_label = "raw policy repair"
        if str(getattr(args, "policy_repair_review_mode", "requery_policy")) == "recorded_error":
            policy_panel_label = "recorded error continuation (not repair)"
        if replay_start_in_bounds:
            if str(getattr(args, "policy_repair_review_mode", "requery_policy")) == "recorded_error":
                policy_repair_meta = _record_replay_component(
                    probe_args,
                    case,
                    rollout,
                    start,
                    None,
                    policy_repair_path,
                    "recorded_error_continuation",
                    trial,
                    repair_source="recorded_original_failure_actions",
                    coordinate_validation=coordinate_validation,
                )
                policy_repair_meta["counts_as_repair_evidence"] = False
                policy_repair_meta["review_mode"] = "recorded_error_continuation"
                policy_repair_meta["reason"] = (
                    "policy repair re-query disabled; panel follows the original "
                    "failed rollout actions from the repair context"
                )
                policy_repair_meta["report_variant"] = raw_variant
            else:
                policy_repair_meta = _record_replay_component(
                    probe_args,
                    case,
                    rollout,
                    start,
                    None,
                    policy_repair_path,
                    "policy_repair",
                    trial,
                    policy_client=client,
                    policy_from_step=start,
                    repair_source=None if raw_variant is None else str(raw_variant.get("source") or ""),
                    coordinate_validation=coordinate_validation,
                )
            if policy_repair_meta.get("available") is not False:
                policy_panel_path = policy_repair_path
        else:
            policy_repair_meta = _unavailable_replay_meta(
                (
                    "recorded_error_continuation"
                    if str(getattr(args, "policy_repair_review_mode", "requery_policy")) == "recorded_error"
                    else "policy_repair"
                ),
                "repair_replay_context_start_out_of_bounds",
                trial=trial,
                repair_source=None if raw_variant is None else str(raw_variant.get("source") or ""),
                coordinate_validation=coordinate_validation,
                report_variant=raw_variant,
            )
        language_repair_meta = {
            "available": False,
            "source": "policy_language_disambiguation_repair",
            "reason": "no_language_intervention_in_report",
            "report_variant": language_variant,
        }
        language_panel_path: Optional[Path] = None
        if language_prompt:
            if replay_start_in_bounds:
                language_repair_meta = _record_replay_component(
                    probe_args,
                    case,
                    rollout,
                    start,
                    None,
                    language_repair_path,
                    "language_phrase_repair",
                    trial,
                    policy_client=client,
                    policy_from_step=start,
                    prompt_override=language_prompt,
                    repair_source="policy_language_disambiguation_repair",
                    coordinate_validation=coordinate_validation,
                )
                language_repair_meta["report_variant"] = language_variant
                language_repair_meta["language_intervention"] = language_intervention
                if language_repair_meta.get("available") is not False:
                    language_panel_path = language_repair_path
            else:
                language_repair_meta = _unavailable_replay_meta(
                    "language_phrase_repair",
                    "repair_replay_context_start_out_of_bounds",
                    trial=trial,
                    repair_source="policy_language_disambiguation_repair",
                    coordinate_validation=coordinate_validation,
                    report_variant=language_variant,
                )
        visual_repair_meta = {
            "available": False,
            "source": "policy_visual_mask_repair",
            "reason": "no_applicable_visual_policy_mask_in_report",
            "report_variant": visual_variant,
        }
        visual_panel_path: Optional[Path] = None
        if visual_intervention is not None:
            if replay_start_in_bounds:
                visual_repair_meta = _record_replay_component(
                    probe_args,
                    case,
                    rollout,
                    start,
                    None,
                    visual_repair_path,
                    "visual_mask_repair",
                    trial,
                    policy_client=client,
                    policy_from_step=start,
                    visual_intervention=visual_intervention,
                    repair_source="policy_visual_mask_repair",
                    coordinate_validation=coordinate_validation,
                )
                visual_repair_meta["report_variant"] = visual_variant
                visual_repair_meta["visual_policy_mask_intervention"] = visual_intervention
                if visual_repair_meta.get("available") is not False:
                    visual_panel_path = visual_repair_path
            else:
                visual_repair_meta = _unavailable_replay_meta(
                    "visual_mask_repair",
                    "repair_replay_context_start_out_of_bounds",
                    trial=trial,
                    repair_source="policy_visual_mask_repair",
                    coordinate_validation=coordinate_validation,
                    report_variant=visual_variant,
                )
        source_repair_meta = {
            "available": False,
            "reason": None if source_meta is None else source_meta.get("reason"),
            "source": "success_or_demo_nn_repair",
            "source_metadata": _source_metadata_without_actions(source_meta),
            "report_variant": demo_variant,
        }
        source_panel_path: Optional[Path] = None
        if source_meta is not None and source_meta.get("actions") is not None:
            if replay_start_in_bounds:
                source_repair_meta = _record_replay_component(
                    probe_args,
                    case,
                    rollout,
                    start,
                    None,
                    source_repair_path,
                    "demo_or_success_repair",
                    trial,
                    external_actions=np.asarray(source_meta["actions"], dtype=np.float32),
                    repair_source="success_or_demo_nn_repair",
                    coordinate_validation=coordinate_validation,
                )
                source_repair_meta["source_metadata"] = _source_metadata_without_actions(source_meta)
                source_repair_meta["report_variant"] = demo_variant
                if source_repair_meta.get("available") is not False:
                    source_panel_path = source_repair_path
            else:
                source_repair_meta = _unavailable_replay_meta(
                    "demo_or_success_repair",
                    "repair_replay_context_start_out_of_bounds",
                    trial=trial,
                    repair_source="success_or_demo_nn_repair",
                    coordinate_validation=coordinate_validation,
                    report_variant=demo_variant,
                )
                source_repair_meta["source_metadata"] = _source_metadata_without_actions(source_meta)
        frames = make_multipanel(
            [original_suffix, minimal_panel_path, destructive_panel_path],
            triptych_path,
            [
                f"{label_suffix}original from {start}",
                f"{label_suffix}recorded replay from {start}",
                f"{label_suffix}destructive ablation {best_strategy} (not repair)",
            ],
            args.video_fps,
            args.video_quality,
            args.video_codec,
            columns=3,
        )
        repair_frames = make_multipanel(
            [
                original_suffix,
                minimal_panel_path,
                policy_panel_path,
                language_panel_path,
                visual_panel_path,
                source_panel_path,
            ],
            multisource_path,
            [
                f"{label_suffix}original from {start}",
                f"{label_suffix}recorded replay from {start}",
                f"{label_suffix}{policy_panel_label}",
                (
                    f"{label_suffix}language phrase repair"
                    if language_panel_path is not None
                    else f"{label_suffix}language unavailable"
                ),
                (
                    f"{label_suffix}visual mask repair"
                    if visual_panel_path is not None
                    else f"{label_suffix}visual unavailable"
                ),
                (
                    f"{label_suffix}demo/success repair"
                    if source_panel_path is not None
                    else f"{label_suffix}demo/success unavailable"
                ),
            ],
            args.video_fps,
            args.video_quality,
            args.video_codec,
        )
        trials.append(
            {
                "trial": int(trial),
                "triptych_video": str(triptych_path),
                "triptych_frames": int(frames),
                "destructive_triptych_video": str(triptych_path),
                "repair_multisource_video": str(multisource_path),
                "repair_multisource_frames": int(repair_frames),
                "repair_quadriptych_video": str(multisource_path),
                "repair_quadriptych_frames": int(repair_frames),
                "minimal_replay": minimal_meta,
                "destructive_ablation_replay": destructive_meta,
                "policy_repair_replay": policy_repair_meta,
                "raw_policy_repair_replay": policy_repair_meta,
                "language_phrase_repair_replay": language_repair_meta,
                "visual_mask_repair_replay": visual_repair_meta,
                "demo_or_success_repair_replay": source_repair_meta,
                "coordinate_validation": coordinate_validation,
                "repair_source_panels": {
                    "raw_policy": None if policy_panel_path is None else str(policy_panel_path),
                    "raw_policy_review_mode": str(
                        getattr(args, "policy_repair_review_mode", "requery_policy")
                    ),
                    "language_phrase": None
                    if language_panel_path is None
                    else str(language_panel_path),
                    "visual_mask": None
                    if visual_panel_path is None
                    else str(visual_panel_path),
                    "demo_or_success": None
                    if source_panel_path is None
                    else str(source_panel_path),
                },
                "repair_replay_context": {
                    "interval": [start, repair_end],
                    "source": case.repair_replay_source,
                    "note": (
                        "Review replay starts at the repair evidence interval, "
                        "which may be earlier than the minimal same-failure slice."
                    ),
                },
                "rollout_source": (
                    {
                        "source": "rollout_archive",
                        "archive_path": getattr(rollout, "archive_path", None),
                    }
                    if bool(getattr(rollout, "loaded_from_archive", False))
                    else {"source": "fresh_policy_rollout"}
                ),
            }
        )
    return trials


def _record_k_minimal_set_reviews(
    args: argparse.Namespace,
    case: ReviewCase,
    case_dir: Path,
) -> list[dict]:
    if int(args.review_top_k_sets) <= 0 or not case.k_minimal_causal_sets:
        return []

    from openpi_client import websocket_client_policy
    from pi05_natural_failure_probe import collect_pi05_rollout

    probe_args = _build_probe_args(args, case)
    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
    rollout = _load_rollout_archive(case)
    if rollout is None:
        rollout = collect_pi05_rollout(
            probe_args,
            client,
            args.task_suite_name,
            case.task_id,
            case.init_state_id,
        )

    reviews = []
    for k_set in case.k_minimal_causal_sets[: int(args.review_top_k_sets)]:
        rank = int(k_set.get("rank") or (len(reviews) + 1))
        set_id = _safe_id(k_set.get("set_id") or f"set_{rank:02d}")
        set_case = _case_for_k_minimal_set(case, k_set)
        set_dir = case_dir / "k_minimal_sets" / f"set_{rank:02d}_{set_id}"
        set_dir.mkdir(parents=True, exist_ok=True)
        sources = ",".join(str(item) for item in k_set.get("repair_sources") or []) or "necessity"
        interval = _interval_from_k_minimal_set(k_set) or (
            set_case.repair_replay_start,
            set_case.repair_replay_end,
        )
        label = (
            f"set{rank} [{interval[0]},{interval[1]}) "
            f"CE={float(k_set.get('causal_effect') or 0.0):.2f} {sources}"
        )
        trials = _record_replay_triptychs(
            args,
            set_case,
            set_dir,
            probe_args=probe_args,
            client=client,
            rollout=rollout,
            filename_prefix=f"set_{rank:02d}",
            label_prefix=label,
        )
        reviews.append(
            {
                "rank": rank,
                "set_id": str(k_set.get("set_id") or ""),
                "set_dir": str(set_dir),
                "summary": _summarize_k_minimal_set(k_set),
                "repair_replay_context": {
                    "interval": [int(set_case.repair_replay_start), int(set_case.repair_replay_end)],
                    "source": set_case.repair_replay_source,
                },
                "replay_trials": trials,
            }
        )
    return reviews


def _replay_failed_goals(meta: dict) -> list[str]:
    signature = meta.get("failure_signature") or {}
    return [str(item) for item in signature.get("failed_goal_predicates") or []]


def _replay_improves(case: ReviewCase, meta: dict) -> bool:
    if bool(meta.get("success")):
        return True
    failed = set(_replay_failed_goals(meta))
    original_failed = set(case.failed_goals)
    if not original_failed:
        return False
    return failed.issubset(original_failed) and len(failed) < len(original_failed)


def _summarize_recorded_repair_evidence(
    case: ReviewCase,
    replay_trials: Sequence[dict],
    k_minimal_set_reviews: Sequence[dict],
) -> dict:
    sources = {
        "raw_policy": "raw_policy_repair_replay",
        "language_phrase": "language_phrase_repair_replay",
        "visual_mask": "visual_mask_repair_replay",
        "demo_or_success": "demo_or_success_repair_replay",
    }
    summary: dict[str, object] = {
        "schema_version": "recorded-repair-evidence-v3-coordinate-strict",
        "any_success": False,
        "any_improvement": False,
        "observed_any_success": False,
        "observed_any_improvement": False,
        "strict_source_aware_only": True,
        "coordinate_mismatch_count": 0,
        "recorded_error_continuation_count": 0,
        "reported_full_success_repair_pass": bool(case.full_success_repair_pass),
        "reported_vs_recorded_mismatch": False,
        "non_repair_continuations": [],
        "sources": {},
    }
    source_summary = summary["sources"]
    assert isinstance(source_summary, dict)

    def ingest(scope: str, trial: dict) -> None:
        for source, key in sources.items():
            meta = trial.get(key) or {}
            if not isinstance(meta, dict) or meta.get("available") is False:
                continue
            if meta.get("counts_as_repair_evidence") is False:
                summary["recorded_error_continuation_count"] = int(
                    summary["recorded_error_continuation_count"]
                ) + 1
                continuations = summary["non_repair_continuations"]
                if isinstance(continuations, list) and len(continuations) < 5:
                    continuations.append(
                        {
                            "scope": scope,
                            "source_slot": source,
                            "trial": int(trial.get("trial") or 0),
                            "variant": meta.get("variant"),
                            "reason": meta.get("reason"),
                            "success": bool(meta.get("success")),
                            "failed_goal_predicates": _replay_failed_goals(meta),
                            "video_path": meta.get("video_path"),
                        }
                    )
                continue
            item = source_summary.setdefault(
                source,
                {
                    "executed_trials": 0,
                    "success_count": 0,
                    "improvement_count": 0,
                    "observed_success_count": 0,
                    "observed_improvement_count": 0,
                    "coordinate_mismatch_count": 0,
                    "examples": [],
                },
            )
            item["executed_trials"] = int(item["executed_trials"]) + 1
            success = bool(meta.get("success"))
            improves = _replay_improves(case, meta)
            strict_eligible = bool(meta.get("strict_source_aware_evidence", True))
            coordinate_mismatch = bool(meta.get("coordinate_mismatch"))
            if coordinate_mismatch:
                item["coordinate_mismatch_count"] = int(item["coordinate_mismatch_count"]) + 1
                summary["coordinate_mismatch_count"] = int(summary["coordinate_mismatch_count"]) + 1
            if success:
                item["observed_success_count"] = int(item["observed_success_count"]) + 1
                summary["observed_any_success"] = True
            if improves:
                item["observed_improvement_count"] = int(item["observed_improvement_count"]) + 1
                summary["observed_any_improvement"] = True
            if success and strict_eligible:
                item["success_count"] = int(item["success_count"]) + 1
                summary["any_success"] = True
            if improves and strict_eligible:
                item["improvement_count"] = int(item["improvement_count"]) + 1
                summary["any_improvement"] = True
            examples = item["examples"]
            if isinstance(examples, list) and len(examples) < 3:
                examples.append(
                    {
                        "scope": scope,
                        "trial": int(trial.get("trial") or 0),
                        "success": success,
                        "improves": improves,
                        "strict_source_aware_evidence": strict_eligible,
                        "coordinate_mismatch": coordinate_mismatch,
                        "failed_goal_predicates": _replay_failed_goals(meta),
                        "failure_type": (meta.get("failure_signature") or {}).get("failure_type"),
                        "video_path": meta.get("video_path"),
                    }
                )

    for trial in replay_trials:
        ingest("case", trial)
    for set_review in k_minimal_set_reviews:
        rank = int(set_review.get("rank") or 0)
        for trial in set_review.get("replay_trials") or []:
            ingest(f"k_minimal_set_{rank}", trial)

    summary["reported_vs_recorded_mismatch"] = bool(
        case.full_success_repair_pass and not summary["any_success"]
    )
    return summary


def _case_review_json(case: ReviewCase, out: dict) -> dict:
    recorded_summary = _summarize_recorded_repair_evidence(
        case,
        out.get("replay_trials") or [],
        out.get("k_minimal_set_reviews") or [],
    )
    return {
        "schema_version": "slice-review-case-v4-k-minimal-multisource",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case.case_id,
        "report_path": str(case.report_path),
        "review_semantics": case.review_semantics,
        "full_success_repair_pass": bool(case.full_success_repair_pass),
        "task_id": case.task_id,
        "init_state_id": case.init_state_id,
        "seed": case.seed,
        "task_language": case.task_language,
        "minimal_slice": {"interval": [case.minimal_start, case.minimal_end], "semantics": "[start,end)"},
        "repair_replay_context": {
            "interval": [case.repair_replay_start, case.repair_replay_end],
            "source": case.repair_replay_source,
            "semantics": (
                "State anchor used for repair/destructive review videos. "
                "This may be earlier than the minimal same-failure slice when "
                "the causal report's repair evidence comes from a broader core unit."
            ),
        },
        "failure_type": case.failure_type,
        "failed_goals": case.failed_goals,
        "same_failure_rate": case.same_failure_rate,
        "base_same_failure_rate": case.base_same_failure_rate,
        "necessity_core_units": case.necessity_core_units,
        "causal_core_units": case.causal_core_units,
        "k_minimal_causal_sets": [
            _summarize_k_minimal_set(item) for item in case.k_minimal_causal_sets
        ],
        "repair_pass_variants": case.repair_pass_variants,
        "recorded_repair_evidence": recorded_summary,
        "destructive_ablation_note": (
            "hold/adjacent/gripper-correction with recorded suffix is a destructive ablation, not a repair."
        ),
        **out,
        "human_review": {
            "is_true_minimal_slice": None,
            "slice_contains_visible_cause": None,
            "destructive_ablation_changes_same_failure": None,
            "policy_repair_improves_or_succeeds": None,
            "demo_or_success_repair_improves_or_succeeds": None,
            "notes": "",
        },
    }


def _write_index(output_dir: Path, cases: list[dict]) -> None:
    cards = []
    for case in cases:
        rel_context = Path(case["original_context_video"]).relative_to(output_dir)
        k_set_rows = []
        for item in case.get("k_minimal_causal_sets") or []:
            units = []
            for unit in item.get("units") or []:
                units.append(
                    "%s %s %s"
                    % (
                        html.escape(str(unit.get("kind") or "")),
                        html.escape(str(unit.get("unit_id") or "")),
                        html.escape(str(unit.get("interval") or "")),
                    )
                )
            k_set_rows.append(
                "<tr>"
                f"<td>{int(item.get('rank') or 0)}</td>"
                f"<td>{html.escape(str(item.get('set_id') or ''))}</td>"
                f"<td>{float(item.get('causal_effect') or 0.0):.2f}</td>"
                f"<td>{float(item.get('ablated_same_failure_rate') or 0.0):.2f}</td>"
                f"<td>{html.escape(','.join(str(x) for x in item.get('repair_sources') or []))}</td>"
                f"<td>{html.escape(str(bool(item.get('full_success_repair'))))}</td>"
                f"<td>{'<br>'.join(units)}</td>"
                "</tr>"
            )
        k_set_table = (
            "<table><thead><tr><th>Rank</th><th>Set</th><th>CE</th>"
            "<th>Ablated same-failure</th><th>Repair source</th>"
            "<th>Full success</th><th>Units</th></tr></thead><tbody>"
            + "".join(k_set_rows)
            + "</tbody></table>"
            if k_set_rows
            else "<p>No k-minimal causal sets were reported for this case.</p>"
        )
        k_set_video_tags = []
        for set_review in case.get("k_minimal_set_reviews") or []:
            rank = int(set_review.get("rank") or 0)
            summary = set_review.get("summary") or {}
            k_set_video_tags.append(
                "<section class='kset'>"
                f"<h4>k-minimal set rank {rank}: {html.escape(str(set_review.get('set_id') or ''))}</h4>"
                f"<p>CE={float(summary.get('causal_effect') or 0.0):.2f} "
                f"repair_sources={html.escape(','.join(str(x) for x in summary.get('repair_sources') or []))} "
                f"full_success={html.escape(str(bool(summary.get('full_success_repair'))))}</p>"
            )
            for trial in set_review.get("replay_trials") or []:
                destructive = trial.get("destructive_triptych_video") or trial.get("triptych_video")
                repair = trial.get("repair_multisource_video") or trial.get("repair_quadriptych_video")
                if destructive:
                    rel = Path(destructive).relative_to(output_dir)
                    k_set_video_tags.append(
                        f"<h5>Set {rank} Trial {int(trial.get('trial') or 0):02d} Destructive</h5>"
                        f"<video controls preload='metadata' src='{html.escape(str(rel))}'></video>"
                    )
                if repair:
                    rel = Path(repair).relative_to(output_dir)
                    k_set_video_tags.append(
                        f"<h5>Set {rank} Trial {int(trial.get('trial') or 0):02d} Repair Multi-Source</h5>"
                        f"<video controls preload='metadata' src='{html.escape(str(rel))}'></video>"
                    )
            k_set_video_tags.append("</section>")
        triptych_tags = []
        for trial in case.get("replay_trials", []):
            rel = Path(trial["triptych_video"]).relative_to(output_dir)
            repair_rel = trial.get("repair_multisource_video") or trial.get("repair_quadriptych_video")
            repair_tag = ""
            if repair_rel:
                repair_rel_path = Path(repair_rel).relative_to(output_dir)
                repair_tag = (
                    f"<h4>Trial {trial['trial']:02d} Repair Multi-Source</h4>"
                    f"<video controls preload='metadata' src='{html.escape(str(repair_rel_path))}'></video>"
                )
            triptych_tags.append(
                f"<h4>Trial {trial['trial']:02d} Destructive Ablation</h4>"
                f"<video controls preload='metadata' src='{html.escape(str(rel))}'></video>"
                f"{repair_tag}"
            )
        goals = "<br>".join(html.escape(goal) for goal in case.get("failed_goals", []))
        recorded = case.get("recorded_repair_evidence") or {}
        recorded_line = (
            f"<b>Recorded repair success in review videos:</b> {html.escape(str(bool(recorded.get('any_success'))))}<br>"
            f"<b>Recorded repair improvement:</b> {html.escape(str(bool(recorded.get('any_improvement'))))}<br>"
            f"<b>Report/video mismatch:</b> {html.escape(str(bool(recorded.get('reported_vs_recorded_mismatch'))))}<br>"
        )
        cards.append(
            f"""
<section class="case">
	  <h2>{html.escape(case['case_id'])}</h2>
	  <p><b>Review semantics:</b> {html.escape(case.get('review_semantics', ''))}<br>
	  <b>Reported full-success repair:</b> {html.escape(str(bool(case.get('full_success_repair_pass'))))}<br>
  {recorded_line}
	  <b>Failure:</b> {html.escape(case['failure_type'])}<br>
	  <b>Minimal:</b> {case['minimal_slice']['interval']} &nbsp;
  <b>same:</b> {case['same_failure_rate']:.2f} &nbsp;
  <b>base:</b> {case['base_same_failure_rate']:.2f}</p>
  <p><b>Failed goals:</b><br>{goals}</p>
  <h3>Original Context</h3>
  <video controls preload="metadata" src="{html.escape(str(rel_context))}"></video>
	  <h3>Replay Videos</h3>
	  <p>Right side of the three-panel video is destructive ablation, not a repair. Multi-source repair videos show original, recorded replay, raw policy, language phrase, visual mask, and demo/success panels side by side.</p>
	  {''.join(triptych_tags) if triptych_tags else '<p>Replay videos were not recorded for this run.</p>'}
  <h3>Top-k Minimal Causal Sets</h3>
  <p>Each row is a bounded-minimal causal set inside the generated multimodal proposal universe. The videos below let you compare rank 1-5 directly.</p>
  {k_set_table}
  {''.join(k_set_video_tags) if k_set_video_tags else '<p>No per-set videos were recorded for this run.</p>'}
</section>
"""
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Slice Review</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
.case {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 0 0 24px; }}
.kset {{ border-top: 1px solid #ddd; margin-top: 16px; padding-top: 10px; }}
video {{ width: 100%; max-width: 2048px; display: block; margin: 8px 0 18px; background: #000; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 13px; }}
th, td {{ border: 1px solid #d6d9df; padding: 6px 8px; vertical-align: top; }}
th {{ background: #eef1f5; text-align: left; }}
h2 {{ margin-top: 0; }}
</style></head><body>
	<h1>Split-Repair Causal Slice Review</h1>
<p>Generated at {html.escape(datetime.now(timezone.utc).isoformat())}. Cases: {len(cases)}</p>
{''.join(cards)}
</body></html>
"""
    (output_dir / "review_index.html").write_text(page, encoding="utf-8")


def _write_review_reels(
    output_dir: Path,
    cases: list[dict],
    fps: int,
    quality: int,
    codec: str,
) -> dict:
    repair_segments: list[tuple[str, Path]] = []
    necessity_segments: list[tuple[str, Path]] = []
    all_segments: list[tuple[str, Path]] = []

    def trial_has_recorded_success(trial: dict) -> bool:
        for key in (
            "raw_policy_repair_replay",
            "language_phrase_repair_replay",
            "visual_mask_repair_replay",
            "demo_or_success_repair_replay",
            "policy_repair_replay",
        ):
            meta = trial.get(key) or {}
            if isinstance(meta, dict) and bool(meta.get("success")):
                return True
        return False

    for case in cases:
        case_id = str(case.get("case_id") or "")
        semantics = str(case.get("review_semantics") or "")
        recorded = case.get("recorded_repair_evidence") or {}
        full_success = bool(recorded.get("any_success")) if recorded else bool(
            case.get("full_success_repair_pass")
        )
        context_raw = str(case.get("original_context_video") or "")
        context_path = Path(context_raw) if context_raw else None
        if context_path is not None and context_path.exists():
            all_segments.append((f"{case_id} original context", context_path))
        for trial in case.get("replay_trials") or []:
            trial_id = int(trial.get("trial") or 0)
            repair_raw = str(
                trial.get("repair_multisource_video")
                or trial.get("repair_quadriptych_video")
                or ""
            )
            destructive_raw = str(
                trial.get("destructive_triptych_video") or trial.get("triptych_video") or ""
            )
            repair_path = Path(repair_raw) if repair_raw else None
            destructive_path = Path(destructive_raw) if destructive_raw else None
            if (
                semantics in {"repair_valid_causal_pass", "repair_valid_success"}
                and full_success
                and repair_path is not None
                and repair_path.exists()
            ):
                repair_segments.append(
                    (f"{case_id} trial {trial_id:02d}: repair-valid success", repair_path)
                )
            if (
                semantics in {"repair_valid_causal_pass", "repair_valid_success"}
                and repair_path is not None
                and repair_path.exists()
            ):
                all_segments.append(
                    (f"{case_id} trial {trial_id:02d}: repair multi-source", repair_path)
                )
            if (
                semantics == "same_failure_necessity_only"
                and destructive_path is not None
                and destructive_path.exists()
            ):
                necessity_segments.append(
                    (f"{case_id} trial {trial_id:02d}: necessity-only, not repair-valid", destructive_path)
                )
                all_segments.append(
                    (f"{case_id} trial {trial_id:02d}: necessity-only contrast", destructive_path)
                )
        for set_review in case.get("k_minimal_set_reviews") or []:
            rank = int(set_review.get("rank") or 0)
            summary = set_review.get("summary") or {}
            full_success_set = bool(summary.get("full_success_repair"))
            for trial in set_review.get("replay_trials") or []:
                trial_id = int(trial.get("trial") or 0)
                repair_raw = str(
                    trial.get("repair_multisource_video")
                    or trial.get("repair_quadriptych_video")
                    or ""
                )
                destructive_raw = str(
                    trial.get("destructive_triptych_video") or trial.get("triptych_video") or ""
                )
                repair_path = Path(repair_raw) if repair_raw else None
                destructive_path = Path(destructive_raw) if destructive_raw else None
                if repair_path is not None and repair_path.exists():
                    title = f"{case_id} set {rank} trial {trial_id:02d}: k-minimal repair"
                    all_segments.append((title, repair_path))
                    if full_success_set and trial_has_recorded_success(trial):
                        repair_segments.append((f"{title} full-success", repair_path))
                elif destructive_path is not None and destructive_path.exists():
                    all_segments.append(
                        (
                            f"{case_id} set {rank} trial {trial_id:02d}: k-minimal destructive",
                            destructive_path,
                        )
                    )

    outputs = {}
    repair_reel = output_dir / "repair_valid_success_reel.mp4"
    necessity_reel = output_dir / "necessity_only_contrast_reel.mp4"
    all_reel = output_dir / "all_review_reel.mp4"
    repair_frames = _make_concat_reel(repair_segments, repair_reel, fps, quality, codec)
    necessity_frames = _make_concat_reel(necessity_segments, necessity_reel, fps, quality, codec)
    all_frames = _make_concat_reel(all_segments, all_reel, fps, quality, codec)
    for frames, path in (
        (repair_frames, repair_reel),
        (necessity_frames, necessity_reel),
        (all_frames, all_reel),
    ):
        if frames <= 0 and path.exists():
            path.unlink()
    outputs["repair_valid_success_reel"] = None if repair_frames <= 0 else str(repair_reel)
    outputs["repair_valid_success_reel_frames"] = int(repair_frames)
    outputs["necessity_only_contrast_reel"] = None if necessity_frames <= 0 else str(necessity_reel)
    outputs["necessity_only_contrast_reel_frames"] = int(necessity_frames)
    outputs["all_review_reel"] = None if all_frames <= 0 else str(all_reel)
    outputs["all_review_reel_frames"] = int(all_frames)
    return outputs


def export_reviews(args: argparse.Namespace) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _first_cuda_device(args.cuda_visible_devices))
    inputs = list(args.report_path or [])
    if args.report_dir is not None:
        inputs.append(args.report_dir)
    cases = collect_review_cases(
        inputs,
        max_cases=args.max_cases,
        include_necessity_only=args.include_necessity_only,
    )
    if not cases:
        raise RuntimeError("No causal pass reports found")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_proc = None
    if args.record_replays:
        policy_proc = _start_lerobot_policy_server(args, output_dir / "policy_server.log")
    reviews = []
    try:
        for case in cases:
            if case.video_every_n != 1 and not args.allow_degraded_video:
                raise RuntimeError(
                    f"{case.case_id} has video_every_n={case.video_every_n}; step/frame mapping is degraded."
                )
            case_dir = output_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            total_frames = case.video_frames or _video_frame_count(case.source_video_path)
            context_start = min(case.minimal_start, case.repair_replay_start)
            context_end = max(case.minimal_end, case.repair_replay_end)
            clip_start, clip_end = frame_window_for_slice(
                context_start,
                context_end,
                case.video_fps,
                args.context_seconds,
                total_frames,
            )
            context_video = case_dir / "original_context.mp4"
            context_frames = make_original_clip(
                case,
                context_video,
                clip_start,
                clip_end,
                args.video_fps,
                args.video_quality,
                args.video_codec,
                "ORIGINAL",
            )
            replay_trials = []
            k_minimal_set_reviews = []
            if args.record_replays:
                replay_trials = _record_replay_triptychs(args, case, case_dir)
                k_minimal_set_reviews = _record_k_minimal_set_reviews(args, case, case_dir)
            review = _case_review_json(
                case,
                {
                    "source_video_path": str(case.source_video_path),
                    "original_context_video": str(context_video),
                    "original_context_window": [clip_start, clip_end],
                    "original_context_frames": int(context_frames),
                    "replay_trials": replay_trials,
                    "k_minimal_set_reviews": k_minimal_set_reviews,
                },
            )
            review_path = case_dir / "case_review.json"
            review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
            review["case_review_path"] = str(review_path)
            reviews.append(review)
        reels = _write_review_reels(
            output_dir,
            reviews,
            args.video_fps,
            args.video_quality,
            args.video_codec,
        )
        manifest = {
            "schema_version": "slice-review-manifest-v4-k-minimal-multisource",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "record_replays": bool(args.record_replays),
            "include_necessity_only": bool(args.include_necessity_only),
            "replay_trials_to_record": int(args.replay_trials_to_record),
            "policy_repair_review_mode": str(args.policy_repair_review_mode),
            "review_top_k_sets": int(args.review_top_k_sets),
            "num_cases": len(reviews),
            "reels": reels,
            "cases": reviews,
        }
        (output_dir / "review_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_index(output_dir, reviews)
        return manifest
    finally:
        if policy_proc is not None:
            try:
                policy_proc.terminate()
                policy_proc.wait(timeout=30)
            except Exception:
                policy_proc.kill()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Export split-repair human-review videos for SHED-CFS slices.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--report-path", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PARENT / f"slice_review_{timestamp}")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--include-necessity-only",
        action="store_true",
        help="Also export same-failure necessity cases that are not repair-valid causal passes.",
    )
    parser.add_argument("--context-seconds", type=float, default=3.0)
    parser.add_argument("--record-replays", action="store_true", default=True)
    parser.add_argument("--no-record-replays", dest="record_replays", action="store_false")
    parser.add_argument("--replay-trials-to-record", type=int, default=5)
    parser.add_argument(
        "--policy-repair-review-mode",
        choices=("requery_policy", "recorded_error"),
        default="requery_policy",
        help=(
            "How to fill the raw-policy panel in review videos. requery_policy asks "
            "the policy server from the repair context; recorded_error follows the "
            "original failed rollout actions from that context and marks the panel "
            "as non-repair evidence."
        ),
    )
    parser.add_argument(
        "--review-top-k-sets",
        type=int,
        default=5,
        help=(
            "Generate per-set review videos for the top K k_minimal_causal_sets. "
            "Use 0 to keep only the legacy best-set review videos."
        ),
    )
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8060)
    parser.add_argument("--policy-config", default="pi0_fast_libero")
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=PROJECT_ROOT / "model_datasets/pi0fast-libero-libero_10/policy_overlay",
    )
    parser.add_argument(
        "--action-tokenizer-path",
        type=Path,
        default=PROJECT_ROOT
        / "model_datasets/pi0fast-libero-libero_10/tokenizers/jadechoghari_tokenizer-lib-mean",
    )
    parser.add_argument(
        "--text-tokenizer-path",
        type=Path,
        default=Path("/root/autodl-tmp/research/VLA_SKILL/model/google/paligemma-3b-pt-224"),
    )
    parser.add_argument("--launch-policy-server", action="store_true")
    parser.add_argument("--allow-existing-policy-server", action="store_true")
    parser.add_argument("--policy-ready-timeout", type=float, default=900.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--pytorch-device", default="cuda")
    parser.add_argument("--pytorch-compile-mode", default="none")
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=32)
    parser.add_argument("--same-failure-threshold", type=float, default=0.75)
    parser.add_argument(
        "--demo-dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"),
    )
    parser.add_argument("--demo-repair-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--enable-demo-repair-lookup",
        action="store_true",
        help=(
            "If a report does not already contain demo/source repair actions, "
            "run a fresh nearest-neighbor demo lookup for the review panel. "
            "Disabled by default because it can scan a large local dataset."
        ),
    )
    parser.add_argument("--video-camera", default="agentview_image")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=10)
    parser.add_argument("--video-no-flip", action="store_true")
    parser.add_argument("--allow-degraded-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = export_reviews(args)
    print(json.dumps({"num_cases": manifest["num_cases"], "output_dir": manifest["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
