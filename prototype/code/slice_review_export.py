from __future__ import annotations

import argparse
import html
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from risk_critic_export import _strict_causal_pass


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_PYTHON = Path("/root/miniconda3/bin/python")
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
    repair_pass_variants: list[dict]
    best_counterfactual: dict


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
        cwd=str(PROJECT_ROOT),
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
    video_path = Path(selected.get("video_path") or "")
    if not video_path.exists():
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
        repair_pass_variants=list(
            report.get("repair_pass_variants")
            or causal.get("repair_pass_variants")
            or []
        ),
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
    except ImportError:
        return _write_video_cv2(frames, out_path, fps)
    count = 0
    writer = imageio.get_writer(
        str(out_path),
        fps=int(fps),
        codec=str(codec),
        quality=int(quality),
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
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
                for i, reader in enumerate(readers):
                    if reader is None:
                        frame = unavailable_frame(labels[i])
                    elif counts[i] and idx >= counts[i] and last_frames[i] is not None:
                        frame = last_frames[i].copy()
                    else:
                        frame = _read_video_frame(reader, min(idx, max(0, counts[i] - 1)), last_frames[i])
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
                        element = _policy_observation(last_obs, rollout.task_language, args.resize_size)
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
            "video_path": str(out_path),
            "frames": int(frames_written),
            "success": bool(success),
            "same_failure": bool(evidence.same_failure),
            "semantic_match_score": float(evidence.score),
            "failure_signature": signature.to_dict(),
            "end_distance": float(_distance(last_obs, rollout.target_key)),
        }
    finally:
        env.close()


def _record_replay_triptychs(args: argparse.Namespace, case: ReviewCase, case_dir: Path) -> list[dict]:
    from openpi_client import websocket_client_policy
    from pi05_natural_failure_probe import (
        _demo_repair_actions,
        _replacement_actions,
        collect_pi05_rollout,
    )

    probe_args = _build_probe_args(args, case)
    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
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
    replacements = _replacement_actions(rollout.actions, best_interval, best_strategy)

    original_suffix = case_dir / "original_suffix.mp4"
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

    preferred_source = None
    for variant in case.repair_pass_variants:
        evaluation = variant.get("evaluation") or {}
        evidence = variant.get("repair_evidence") or {}
        if bool(evaluation.get("success")) or bool(evidence.get("success")):
            preferred_source = variant
            break
    source_meta = None
    if preferred_source is not None:
        source_meta = preferred_source.get("source_metadata")
    if not isinstance(source_meta, dict) or source_meta.get("actions") is None:
        source_meta = _demo_repair_actions(
            probe_args,
            rollout,
            start,
            max(1, rollout.length - start),
        )

    trials = []
    for trial in range(int(args.replay_trials_to_record)):
        minimal_path = case_dir / "components" / f"trial_{trial:02d}_minimal_replay.mp4"
        destructive_path = case_dir / "components" / f"trial_{trial:02d}_destructive_{best_strategy}.mp4"
        policy_repair_path = case_dir / "components" / f"trial_{trial:02d}_policy_repair.mp4"
        source_repair_path = case_dir / "components" / f"trial_{trial:02d}_demo_or_success_repair.mp4"
        triptych_path = case_dir / f"trial_{trial:02d}_destructive_triptych.mp4"
        quadriptych_path = case_dir / f"trial_{trial:02d}_repair_quadriptych.mp4"
        minimal_meta = _record_replay_component(
            probe_args, case, rollout, start, None, minimal_path, "minimal", trial
        )
        destructive_meta = _record_replay_component(
            probe_args,
            case,
            rollout,
            start,
            replacements,
            destructive_path,
            f"destructive:{best_strategy}",
            trial,
        )
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
        )
        source_repair_meta = {
            "available": False,
            "reason": None if source_meta is None else source_meta.get("reason"),
            "source_metadata": source_meta,
        }
        source_panel_path: Optional[Path] = None
        if source_meta is not None and source_meta.get("actions") is not None:
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
            )
            source_repair_meta["source_metadata"] = {
                k: v
                for k, v in source_meta.items()
                if k != "actions" and not isinstance(v, np.ndarray)
            }
            source_panel_path = source_repair_path
        frames = make_triptych(
            original_suffix,
            minimal_path,
            destructive_path,
            triptych_path,
            (
                f"original from {start}",
                f"recorded replay from {start}",
                f"destructive ablation {best_strategy} (not repair)",
            ),
            args.video_fps,
            args.video_quality,
            args.video_codec,
        )
        repair_frames = make_quadriptych(
            original_suffix,
            minimal_path,
            policy_repair_path,
            source_panel_path,
            quadriptych_path,
            (
                f"original from {start}",
                f"recorded replay from {start}",
                f"policy repair from {start}",
                "demo/success repair" if source_panel_path is not None else "demo/success unavailable",
            ),
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
                "repair_quadriptych_video": str(quadriptych_path),
                "repair_quadriptych_frames": int(repair_frames),
                "minimal_replay": minimal_meta,
                "destructive_ablation_replay": destructive_meta,
                "policy_repair_replay": policy_repair_meta,
                "demo_or_success_repair_replay": source_repair_meta,
                "repair_replay_context": {
                    "interval": [start, repair_end],
                    "source": case.repair_replay_source,
                    "note": (
                        "Review replay starts at the repair evidence interval, "
                        "which may be earlier than the minimal same-failure slice."
                    ),
                },
            }
        )
    return trials


def _case_review_json(case: ReviewCase, out: dict) -> dict:
    return {
        "schema_version": "slice-review-case-v2-split-repair",
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
        "repair_pass_variants": case.repair_pass_variants,
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
        triptych_tags = []
        for trial in case.get("replay_trials", []):
            rel = Path(trial["triptych_video"]).relative_to(output_dir)
            repair_rel = trial.get("repair_quadriptych_video")
            repair_tag = ""
            if repair_rel:
                repair_rel_path = Path(repair_rel).relative_to(output_dir)
                repair_tag = (
                    f"<h4>Trial {trial['trial']:02d} Repair Quad</h4>"
                    f"<video controls preload='metadata' src='{html.escape(str(repair_rel_path))}'></video>"
                )
            triptych_tags.append(
                f"<h4>Trial {trial['trial']:02d} Destructive Ablation</h4>"
                f"<video controls preload='metadata' src='{html.escape(str(rel))}'></video>"
                f"{repair_tag}"
            )
        goals = "<br>".join(html.escape(goal) for goal in case.get("failed_goals", []))
        cards.append(
            f"""
<section class="case">
	  <h2>{html.escape(case['case_id'])}</h2>
	  <p><b>Review semantics:</b> {html.escape(case.get('review_semantics', ''))}<br>
	  <b>Full-success repair:</b> {html.escape(str(bool(case.get('full_success_repair_pass'))))}<br>
	  <b>Failure:</b> {html.escape(case['failure_type'])}<br>
	  <b>Minimal:</b> {case['minimal_slice']['interval']} &nbsp;
  <b>same:</b> {case['same_failure_rate']:.2f} &nbsp;
  <b>base:</b> {case['base_same_failure_rate']:.2f}</p>
  <p><b>Failed goals:</b><br>{goals}</p>
  <h3>Original Context</h3>
  <video controls preload="metadata" src="{html.escape(str(rel_context))}"></video>
	  <h3>Replay Videos</h3>
	  <p>Right side of the three-panel video is destructive ablation, not a repair. Four-panel videos show repair attempts.</p>
	  {''.join(triptych_tags) if triptych_tags else '<p>Replay videos were not recorded for this run.</p>'}
</section>
"""
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Slice Review</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f6f7f9; color: #111; }}
.case {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 0 0 24px; }}
video {{ width: 100%; max-width: 2048px; display: block; margin: 8px 0 18px; background: #000; }}
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

    for case in cases:
        case_id = str(case.get("case_id") or "")
        semantics = str(case.get("review_semantics") or "")
        full_success = bool(case.get("full_success_repair_pass"))
        context_path = Path(case.get("original_context_video") or "")
        if context_path.exists():
            all_segments.append((f"{case_id} original context", context_path))
        for trial in case.get("replay_trials") or []:
            trial_id = int(trial.get("trial") or 0)
            repair_path = Path(trial.get("repair_quadriptych_video") or "")
            destructive_path = Path(trial.get("destructive_triptych_video") or trial.get("triptych_video") or "")
            if semantics in {"repair_valid_causal_pass", "repair_valid_success"} and full_success and repair_path.exists():
                repair_segments.append(
                    (f"{case_id} trial {trial_id:02d}: repair-valid success", repair_path)
                )
            if semantics in {"repair_valid_causal_pass", "repair_valid_success"} and repair_path.exists():
                all_segments.append(
                    (f"{case_id} trial {trial_id:02d}: repair quadriptych", repair_path)
                )
            if semantics == "same_failure_necessity_only" and destructive_path.exists():
                necessity_segments.append(
                    (f"{case_id} trial {trial_id:02d}: necessity-only, not repair-valid", destructive_path)
                )
                all_segments.append(
                    (f"{case_id} trial {trial_id:02d}: necessity-only contrast", destructive_path)
                )

    outputs = {}
    repair_reel = output_dir / "repair_valid_success_reel.mp4"
    necessity_reel = output_dir / "necessity_only_contrast_reel.mp4"
    all_reel = output_dir / "all_review_reel.mp4"
    repair_frames = _make_concat_reel(repair_segments, repair_reel, fps, quality, codec)
    necessity_frames = _make_concat_reel(necessity_segments, necessity_reel, fps, quality, codec)
    all_frames = _make_concat_reel(all_segments, all_reel, fps, quality, codec)
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
            if args.record_replays:
                replay_trials = _record_replay_triptychs(args, case, case_dir)
            review = _case_review_json(
                case,
                {
                    "source_video_path": str(case.source_video_path),
                    "original_context_video": str(context_video),
                    "original_context_window": [clip_start, clip_end],
                    "original_context_frames": int(context_frames),
                    "replay_trials": replay_trials,
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
            "schema_version": "slice-review-manifest-v2-split-repair",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "record_replays": bool(args.record_replays),
            "include_necessity_only": bool(args.include_necessity_only),
            "replay_trials_to_record": int(args.replay_trials_to_record),
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
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8020)
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
