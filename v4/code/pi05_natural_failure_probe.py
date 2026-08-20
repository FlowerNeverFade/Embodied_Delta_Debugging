from __future__ import annotations

import argparse
import collections
from contextlib import contextmanager
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from causal_failure_predicates import (
    CausalUnit,
    CausalUnitResult,
    CausalValidationResult,
    FailureSignature,
    GoalPredicate,
    SameFailureResult,
    StateSnapshot,
    build_global_multimodal_units,
    build_causal_units,
    candidate_overlaps_failure_anchor,
    compare_failure_signatures,
    get_goal_predicates,
    infer_failure_signature,
    make_causal_validation_result,
    make_state_snapshot,
    semantic_quality_for_env,
)
from custom_tasks import registry as custom_tasks
from edd_types import CandidateSlice


DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs/pi05_natural_failure_probe.json"
)
PROJECT_ROOT = Path(
    os.environ.get(
        "EDD_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[2]),
    )
)
OPENPI_PYTHON = Path(os.environ.get("OPENPI_PYTHON", sys.executable))
DEFAULT_DEMO_DATASET_ROOT = Path(
    "/root/autodl-tmp/research/VLA_SKILL/datasets/HuggingFaceVLA_libero"
)
CAUSAL_SCHEMA_VERSION = "shed-cfs-causal-v4-global-multimodal"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class Pi05Rollout:
    task_suite_name: str
    task_id: int
    init_state_id: int
    task_language: str
    target_key: str
    target_key_trace: List[str]
    actions: np.ndarray
    states_before_action: List[np.ndarray]
    snapshots: List[StateSnapshot]
    goal_predicates: Tuple[GoalPredicate, ...]
    semantic_quality: str
    failure_signature: FailureSignature
    distance_trace: List[float]
    success: bool
    done_step: Optional[int]
    initial_state_quality: dict = field(
        default_factory=lambda: {"valid": True, "reasons": []}
    )
    initial_state_attempt: int = 0
    reset_seed: Optional[int] = None
    video_path: Optional[str] = None
    video_frames: int = 0
    rollout_archive_path: Optional[str] = None

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def initial_state_valid(self) -> bool:
        return bool((self.initial_state_quality or {}).get("valid", True))


@dataclass
class ReplayEvaluation:
    candidate: CandidateSlice
    same_failure: bool
    same_failure_rate: float
    failure_rate: float
    trials: int
    success: bool
    start_distance: float
    end_distance: float
    distance_delta: float
    steps: int
    signature: FailureSignature
    same_failure_evidence: SameFailureResult
    planned_trials: int = 0
    executed_trials: int = 0
    same_failure_count: int = 0
    same_failure_rate_lower_bound: float = 0.0
    same_failure_rate_upper_bound: float = 0.0
    trial_outcomes: Tuple[Dict[str, object], ...] = ()
    early_stop_reason: Optional[str] = None
    from_cache: bool = False
    cache_key: Optional[str] = None
    repair_valid_count: int = 0
    repair_valid_rate_lower_bound: float = 0.0
    repair_valid_rate_upper_bound: float = 0.0

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "same_failure": bool(self.same_failure),
            "same_failure_rate": float(self.same_failure_rate),
            "failure_rate": float(self.failure_rate),
            "trials": int(self.trials),
            "planned_trials": int(self.planned_trials or self.trials),
            "executed_trials": int(self.executed_trials or self.trials),
            "same_failure_count": int(self.same_failure_count),
            "same_failure_rate_bounds": [
                float(self.same_failure_rate_lower_bound),
                float(self.same_failure_rate_upper_bound),
            ],
            "repair_valid_count": int(self.repair_valid_count),
            "repair_trial_bounds": [
                float(self.repair_valid_rate_lower_bound),
                float(self.repair_valid_rate_upper_bound),
            ],
            "trial_outcomes": [dict(item) for item in self.trial_outcomes],
            "early_stop_reason": self.early_stop_reason,
            "from_cache": bool(self.from_cache),
            "cache_key": self.cache_key,
            "success": bool(self.success),
            "start_distance": float(self.start_distance),
            "end_distance": float(self.end_distance),
            "distance_delta": float(self.distance_delta),
            "steps": int(self.steps),
            "failure_signature": self.signature.to_dict(),
            "same_failure_evidence": self.same_failure_evidence.to_dict(),
        }


@dataclass
class RuntimeProfile:
    started_at: str
    wall_start: float
    durations: Dict[str, float]
    counters: Dict[str, int]

    @classmethod
    def create(cls) -> "RuntimeProfile":
        return cls(
            started_at=datetime.now(timezone.utc).isoformat(),
            wall_start=time.perf_counter(),
            durations={},
            counters={},
        )

    @contextmanager
    def timed(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[key] = self.durations.get(key, 0.0) + (
                time.perf_counter() - start
            )

    def incr(self, key: str, value: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0) + int(value))

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_wall_seconds": float(time.perf_counter() - self.wall_start),
            "durations_seconds": {k: float(v) for k, v in sorted(self.durations.items())},
            "counters": {k: int(v) for k, v in sorted(self.counters.items())},
        }


def _runtime(args) -> Optional[RuntimeProfile]:
    return getattr(args, "_runtime_profile", None)


@contextmanager
def _timed(args, key: str):
    runtime = _runtime(args)
    if runtime is None:
        yield
    else:
        with runtime.timed(key):
            yield


def _incr(args, key: str, value: int = 1) -> None:
    runtime = _runtime(args)
    if runtime is not None:
        runtime.incr(key, value)


def _set_counter(args, key: str, value: int) -> None:
    runtime = _runtime(args)
    if runtime is not None:
        runtime.counters[key] = int(value)


def _progress(args, event: str, **fields: object) -> None:
    if not bool(getattr(args, "verbose_replay_progress", False)) and not getattr(
        args, "progress_log_path", None
    ):
        return
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if bool(getattr(args, "verbose_replay_progress", False)):
        print(line, file=sys.stderr, flush=True)
    progress_path = getattr(args, "progress_log_path", None)
    if progress_path:
        try:
            progress_path = Path(progress_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=np.float32)
        return {
            "__ndarray__": True,
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _stable_jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _stable_jsonable(value.to_dict())
    return repr(value)


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        _stable_jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replay_cache(args) -> Optional[Dict[str, ReplayEvaluation]]:
    if bool(getattr(args, "disable_replay_cache", False)):
        return None
    cache = getattr(args, "_replay_evaluation_cache", None)
    if cache is None:
        cache = {}
        setattr(args, "_replay_evaluation_cache", cache)
    return cache


def _replay_cache_key(
    args,
    rollout: Pi05Rollout,
    start: int,
    end: int,
    reference_signature: FailureSignature,
    client: Optional[websocket_client_policy.WebsocketClientPolicy],
    action_replacements: Optional[Dict[int, np.ndarray]],
    external_actions: Optional[np.ndarray],
    policy_from_step: Optional[int],
    prompt_override: Optional[str],
    visual_intervention: Optional[dict],
    trials: int,
    early_stop_objective: str = "same_failure",
    ce_reference_rate: Optional[float] = None,
    ce_threshold: Optional[float] = None,
) -> str:
    repair_like_replay = str(early_stop_objective) == "repair_valid"
    physical_end = None if repair_like_replay else int(end)
    key = {
        "schema": "replay-cache-key-v1",
        "task_suite_name": rollout.task_suite_name,
        "task_id": int(rollout.task_id),
        "init_state_id": int(rollout.init_state_id),
        "reset_seed": rollout.reset_seed,
        "start": int(start),
        "physical_end": physical_end,
        "rollout_length": int(rollout.length),
        "trials": int(trials),
        "continuation": getattr(args, "continuation", "recorded"),
        "accept_same_failure_rate": float(args.accept_same_failure_rate),
        "same_failure_threshold": float(args.same_failure_threshold),
        "early_stop_objective": str(early_stop_objective),
        "ce_reference_rate": None
        if ce_reference_rate is None
        else float(ce_reference_rate),
        "ce_threshold": None if ce_threshold is None else float(ce_threshold),
        "replay_evaluation_timeout_seconds": float(
            getattr(args, "replay_evaluation_timeout_seconds", 0.0) or 0.0
        ),
        "event_window": int(args.event_window),
        "policy_from_step": None if policy_from_step is None else int(policy_from_step),
        "policy_client_available": client is not None,
        "replan_steps": int(args.replan_steps),
        "prompt_override": prompt_override,
        "visual_intervention": visual_intervention,
        "action_replacements": action_replacements,
        "external_actions": external_actions,
        "reference_signature": reference_signature.to_dict(),
    }
    return _stable_digest(key)


def _restage_replay_evaluation(
    evaluation: ReplayEvaluation,
    start: int,
    end: int,
    n_steps: int,
    stage_level: str,
    from_cache: bool,
) -> ReplayEvaluation:
    return replace(
        evaluation,
        candidate=CandidateSlice.from_window(start, end, n_steps=n_steps, level=stage_level),
        steps=int(max(0, end - start)),
        from_cache=bool(from_cache),
    )


def _same_failure_rate_bounds(same_count: int, executed: int, planned: int) -> Tuple[float, float]:
    planned = max(1, int(planned))
    executed = max(0, min(int(executed), planned))
    remaining = max(0, planned - executed)
    return float(same_count / planned), float((same_count + remaining) / planned)


def _sequential_trial_stop_reason(
    lower_bound: float,
    upper_bound: float,
    accept_same_failure_rate: float,
    *,
    objective: str = "same_failure",
    ce_reference_rate: Optional[float] = None,
    ce_threshold: Optional[float] = None,
) -> Optional[str]:
    eps = 1e-9
    if objective == "causal_effect":
        if ce_reference_rate is None or ce_threshold is None:
            return None
        cutoff = float(ce_reference_rate) - float(ce_threshold)
        if upper_bound <= cutoff + eps:
            return "causal_effect_threshold_already_met"
        if lower_bound > cutoff + eps:
            return "causal_effect_threshold_impossible"
        return None
    if objective == "repair_valid":
        threshold = float(accept_same_failure_rate)
        if lower_bound >= threshold - eps:
            return "repair_valid_threshold_already_met"
        if upper_bound < threshold - eps:
            return "repair_valid_threshold_impossible"
        return None
    threshold = float(accept_same_failure_rate)
    if lower_bound >= threshold - eps:
        return "same_failure_threshold_already_met"
    if upper_bound < threshold - eps:
        return "same_failure_threshold_impossible"
    return None


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def _apply_visual_policy_intervention(
    image: np.ndarray,
    intervention: Optional[dict],
    image_key: str,
) -> np.ndarray:
    if not isinstance(intervention, dict):
        return image
    rect = intervention.get("%s_rect" % image_key) or intervention.get("image_rect")
    mode = str(intervention.get("mode") or "")
    if not rect or mode not in {"highlight_target", "mask_distractor"}:
        return image
    arr = np.asarray(image).copy()
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in rect]
    except Exception:
        return image
    h, w = arr.shape[:2]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return image
    if mode == "mask_distractor":
        patch = arr[y0:y1, x0:x1]
        if patch.size:
            patch[:] = np.asarray([96, 96, 96], dtype=arr.dtype)
        return np.ascontiguousarray(arr)
    color = np.asarray([0, 255, 0], dtype=arr.dtype)
    thickness = max(2, min(h, w) // 80)
    arr[y0 : min(y0 + thickness, y1), x0:x1] = color
    arr[max(y1 - thickness, y0) : y1, x0:x1] = color
    arr[y0:y1, x0 : min(x0 + thickness, x1)] = color
    arr[y0:y1, max(x1 - thickness, x0) : x1] = color
    return np.ascontiguousarray(arr)


def _policy_observation(
    obs: dict,
    task_description: str,
    resize_size: int,
    visual_intervention: Optional[dict] = None,
) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    img = _apply_visual_policy_intervention(img, visual_intervention, "agent")
    wrist_img = _apply_visual_policy_intervention(
        wrist_img, visual_intervention, "wrist"
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def _video_frame_from_obs(obs: dict, camera_key: str, flip: bool = True) -> np.ndarray:
    if camera_key not in obs:
        raise KeyError(
            "Video camera key %r not found in observation. Available image-like keys: %s"
            % (
                camera_key,
                sorted(
                    key
                    for key, value in obs.items()
                    if isinstance(value, np.ndarray) and value.ndim in (2, 3)
                ),
            )
        )
    frame = np.asarray(obs[camera_key])
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim != 3:
        raise ValueError("Video frame %s has shape %s" % (camera_key, frame.shape))
    if frame.shape[2] > 3:
        frame = frame[:, :, :3]
    if flip:
        frame = frame[::-1, ::-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _rollout_video_path(args, task_id: int, init_state_id: int) -> Optional[Path]:
    if not args.record_video and args.video_dir is None:
        return None
    video_dir = Path(args.video_dir) if args.video_dir is not None else args.output.parent / "videos"
    prefix = args.video_prefix or args.output.stem
    return video_dir / ("%s_task%02d_init%02d.mp4" % (prefix, int(task_id), int(init_state_id)))


@contextmanager
def _rollout_video_writer(args, task_id: int, init_state_id: int):
    path = _rollout_video_path(args, task_id, init_state_id)
    if path is None:
        yield None, None
        return
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "imageio is required for --record-video. Install imageio[ffmpeg] in the LIBERO env."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=int(args.video_fps),
        codec=str(args.video_codec),
        quality=int(args.video_quality),
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p"],
    )
    try:
        yield writer, path
    finally:
        writer.close()


def _state(env) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def _set_state(env, state: np.ndarray) -> None:
    env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
    env.sim.forward()


def _set_state_and_obs(env, state: np.ndarray) -> dict:
    if hasattr(env, "regenerate_obs_from_state"):
        return env.regenerate_obs_from_state(np.asarray(state, dtype=np.float64))
    _set_state(env, state)
    base = getattr(env, "env", env)
    if hasattr(base, "_post_process"):
        base._post_process()
    if hasattr(base, "_update_observables"):
        base._update_observables(force=True)
    if hasattr(base, "_get_observations"):
        return base._get_observations()
    raise RuntimeError("Cannot regenerate observation from simulator state")


def _save_rollout_archive(args, rollout: Pi05Rollout) -> Optional[str]:
    """Persist exact simulator states/actions for source-aware review replay."""

    if rollout is None or rollout.length <= 0 or not rollout.states_before_action:
        return None
    archive_dir = args.output.parent / "rollout_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / (
        f"{args.output.stem}_task{rollout.task_id:02d}_"
        f"init{rollout.init_state_id:02d}_seed{int(args.seed):02d}.npz"
    )
    states = np.asarray(rollout.states_before_action, dtype=np.float64)
    actions = np.asarray(rollout.actions, dtype=np.float32)
    metadata = {
        "schema_version": "pi05-rollout-archive-v1",
        "task_suite_name": rollout.task_suite_name,
        "task_id": int(rollout.task_id),
        "init_state_id": int(rollout.init_state_id),
        "task_language": rollout.task_language,
        "target_key": rollout.target_key,
        "target_key_trace": list(rollout.target_key_trace),
        "distance_trace": [float(x) for x in rollout.distance_trace],
        "success": bool(rollout.success),
        "done_step": rollout.done_step,
        "reset_seed": rollout.reset_seed,
        "length": int(rollout.length),
        "failure_signature": rollout.failure_signature.to_dict(),
        "video_path": rollout.video_path,
        "video_frames": int(rollout.video_frames),
    }
    np.savez_compressed(
        archive_path,
        actions=actions,
        states_before_action=states,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    rollout.rollout_archive_path = str(archive_path)
    return str(archive_path)


def _object_pos_keys(obs: dict) -> List[str]:
    keys = []
    for key, value in obs.items():
        if not key.endswith("_pos"):
            continue
        if "_to_" in key:
            continue
        if key.startswith("robot") or key.startswith("gripper"):
            continue
        arr = np.asarray(value)
        if arr.shape == (3,):
            keys.append(key)
    return sorted(keys)


def _select_target_key(
    obs: dict,
    task_language: str,
    task_suite_name: str = "",
    task_id: int = 0,
    snapshot: Optional[StateSnapshot] = None,
) -> str:
    keys = _object_pos_keys(obs)
    if not keys:
        raise RuntimeError("No object position keys found in LIBERO observation")

    stage_key = custom_tasks.target_key_for_snapshot(
        task_suite_name,
        task_id,
        snapshot,
        obs,
        task_language,
    )
    if stage_key in keys:
        return str(stage_key)

    language = task_language.lower()
    aliases = [
        ("black_bowl", ["black bowl", "bowl"]),
        ("akita_black_bowl", ["black bowl", "bowl"]),
        ("yellow_and_white_mug", ["yellow and white mug", "yellow mug"]),
        ("white_mug", ["white mug", "mug"]),
        ("red_mug", ["red mug"]),
        ("alphabet_soup", ["alphabet soup"]),
        ("cream_cheese", ["cream cheese"]),
        ("tomato_sauce", ["tomato sauce"]),
        ("bbq_sauce", ["bbq sauce"]),
        ("ketchup", ["ketchup"]),
        ("butter", ["butter"]),
        ("milk", ["milk"]),
        ("orange_juice", ["orange juice"]),
        ("chocolate_pudding", ["chocolate pudding"]),
        ("salad_dressing", ["salad dressing"]),
        ("wine_bottle", ["wine bottle"]),
        ("book", ["book"]),
        ("moka_pot", ["moka pot"]),
    ]
    for key_fragment, phrases in aliases:
        if any(phrase in language for phrase in phrases):
            matching = [key for key in keys if key_fragment in key]
            if matching:
                return matching[0]

    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return min(keys, key=lambda k: float(np.linalg.norm(eef - np.asarray(obs[k]))))


def _distance(obs: dict, target_key: str) -> float:
    return float(
        np.linalg.norm(
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            - np.asarray(obs[target_key], dtype=np.float64)
        )
    )


def _make_env(args, task_suite_name: str, task_id: int):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.gpu_id))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    custom_suite = custom_tasks.make_task_suite(task_suite_name)
    if custom_suite is not None:
        task_suite = custom_suite
        task = task_suite.get_task(task_id)
        bddl_file = custom_tasks.custom_bddl_path(task_suite_name, task_id)
    else:
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        task = task_suite.get_task(task_id)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    env.seed(args.seed)
    return env, task_suite, task


def _max_steps_for_suite(args, task_suite_name: str) -> int:
    if args.max_steps:
        return int(args.max_steps)
    custom_max = custom_tasks.max_steps_for_suite(task_suite_name)
    if custom_max is not None:
        return int(custom_max)
    return int(MAX_STEPS_BY_SUITE.get(task_suite_name, 520))


def _max_steps_for_task(args, task_suite_name: str, task_id: int) -> int:
    if args.max_steps:
        return int(args.max_steps)
    custom_max = custom_tasks.max_steps_for_task(task_suite_name, task_id)
    if custom_max is not None:
        return int(custom_max)
    return _max_steps_for_suite(args, task_suite_name)


def _semantic_snapshot(
    task_suite_name: str,
    t: int,
    obs: dict,
    env,
    action: Optional[Sequence[float]],
    success: bool,
    predicates: Sequence[GoalPredicate],
    stage_tracker=None,
    task_id: int = 0,
) -> StateSnapshot:
    snapshot = make_state_snapshot(
        t,
        obs,
        env=env,
        action=action,
        success=success,
        predicates=predicates,
    )
    return custom_tasks.augment_snapshot_with_stage_truth(
        task_suite_name,
        snapshot,
        stage_tracker,
        env,
        task_id,
    )


def collect_pi05_rollout(
    args,
    client: websocket_client_policy.WebsocketClientPolicy,
    task_suite_name: str,
    task_id: int,
    init_state_id: int,
) -> Pi05Rollout:
    env, task_suite, task = _make_env(args, task_suite_name, task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    init_state_id = min(init_state_id, len(initial_states) - 1)
    max_steps = _max_steps_for_task(args, task_suite_name, task_id)
    try:
        custom_suite = custom_tasks.is_custom_suite(task_suite_name)
        max_attempts = 1
        if custom_suite and not args.disable_initial_state_quality_filter:
            max_attempts = max(1, int(args.initial_state_max_attempts))
        chosen = None
        attempt_records = []
        for attempt in range(max_attempts):
            reset_seed = (
                custom_tasks.deterministic_reset_seed(
                    task_suite_name, args.seed, init_state_id
                )
                + 7919 * int(attempt)
                if custom_suite
                else int(args.seed)
            )
            env.seed(reset_seed)
            env.reset()
            _incr(args, "env_resets")
            if len(initial_states) > 0 and initial_states[init_state_id] is not None:
                obs = env.set_init_state(initial_states[init_state_id])
            else:
                obs = env.reset()
            for _ in range(args.num_steps_wait):
                obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

            predicates = get_goal_predicates(env) + custom_tasks.stage_predicates_for_suite(
                task_suite_name,
                task_id,
            )
            semantic_quality = semantic_quality_for_env(env)
            stage_tracker = custom_tasks.make_stage_tracker(task_suite_name, task_id)
            initial_snapshot = _semantic_snapshot(
                task_suite_name,
                0,
                obs,
                env=env,
                action=None,
                success=False,
                predicates=predicates,
                stage_tracker=stage_tracker,
                task_id=task_id,
            )
            target_key = _select_target_key(
                obs,
                task.language,
                task_suite_name=task_suite_name,
                task_id=task_id,
                snapshot=initial_snapshot,
            )
            initial_quality = custom_tasks.initial_state_quality(
                task_suite_name,
                task_id,
                obs,
                env,
                initial_snapshot,
            )
            initial_quality = dict(initial_quality or {"valid": True, "reasons": []})
            initial_quality["attempt"] = int(attempt)
            initial_quality["reset_seed"] = int(reset_seed)
            initial_quality["target_key"] = target_key
            attempt_records.append(dict(initial_quality))
            chosen = (
                obs,
                predicates,
                semantic_quality,
                stage_tracker,
                initial_snapshot,
                target_key,
                initial_quality,
                attempt,
                reset_seed,
            )
            if initial_quality.get("valid", True) or args.disable_initial_state_quality_filter:
                break

        assert chosen is not None
        (
            obs,
            predicates,
            semantic_quality,
            stage_tracker,
            initial_snapshot,
            target_key,
            initial_quality,
            initial_attempt,
            reset_seed,
        ) = chosen
        initial_quality = dict(initial_quality)
        initial_quality["attempts"] = [dict(record) for record in attempt_records]
        action_plan: collections.deque[np.ndarray] = collections.deque()
        actions: List[np.ndarray] = []
        states: List[np.ndarray] = []
        target_key_trace = [target_key]
        distances = [_distance(obs, target_key)]
        snapshots = [
            initial_snapshot
        ]
        success = False
        done_step = None
        video_path: Optional[str] = None
        video_frames = 0

        with _rollout_video_writer(args, task_id, init_state_id) as (video_writer, path):
            if path is not None:
                video_path = str(path)

            def append_video_frame(frame_obs: dict, step_index: int) -> None:
                nonlocal video_frames
                if video_writer is None:
                    return
                every_n = max(1, int(args.video_every_n))
                if int(step_index) % every_n != 0:
                    return
                video_writer.append_data(
                    _video_frame_from_obs(
                        frame_obs,
                        args.video_camera,
                        flip=not bool(args.video_no_flip),
                    )
                )
                video_frames += 1

            append_video_frame(obs, 0)

            if not initial_quality.get("valid", True):
                failure_signature = infer_failure_signature(
                    snapshots,
                    predicates=predicates,
                    semantic_quality=semantic_quality,
                    event_window=args.event_window,
                    task_language=task.language,
                )
                return Pi05Rollout(
                    task_suite_name=task_suite_name,
                    task_id=int(task_id),
                    init_state_id=int(init_state_id),
                    task_language=task.language,
                    target_key=target_key,
                    target_key_trace=target_key_trace,
                    actions=np.zeros((0, 7), dtype=np.float32),
                    states_before_action=[],
                    snapshots=snapshots,
                    goal_predicates=predicates,
                    semantic_quality=semantic_quality,
                    failure_signature=failure_signature,
                    distance_trace=[float(x) for x in distances],
                    success=False,
                    done_step=None,
                    initial_state_quality=initial_quality,
                    initial_state_attempt=int(initial_attempt),
                    reset_seed=int(reset_seed),
                    video_path=video_path,
                    video_frames=int(video_frames),
                )

            for t in range(max_steps):
                if not action_plan:
                    element = _policy_observation(obs, task.language, args.resize_size)
                    _incr(args, "policy_queries")
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(
                        np.asarray(action_chunk, dtype=np.float32)[: args.replan_steps]
                    )

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                states.append(_state(env))
                actions.append(action)
                obs, _reward, done, _info = env.step(action.tolist())
                _incr(args, "rollout_steps")
                append_video_frame(obs, t + 1)
                step_snapshot = _semantic_snapshot(
                    task_suite_name,
                    t + 1,
                    obs,
                    env=env,
                    action=action,
                    success=bool(done),
                    predicates=predicates,
                    stage_tracker=stage_tracker,
                    task_id=task_id,
                )
                step_target_key = _select_target_key(
                    obs,
                    task.language,
                    task_suite_name=task_suite_name,
                    task_id=task_id,
                    snapshot=step_snapshot,
                )
                target_key_trace.append(step_target_key)
                distances.append(_distance(obs, step_target_key))
                semantic_success = bool(done) and bool(step_snapshot.goal_truth) and all(
                    bool(v) for v in step_snapshot.goal_truth.values()
                )
                if done:
                    success = bool(semantic_success)
                    done_step = t
                    step_snapshot = replace(step_snapshot, success=success)
                snapshots.append(step_snapshot)
                if done:
                    break

        failure_signature = infer_failure_signature(
            snapshots,
            predicates=predicates,
            semantic_quality=semantic_quality,
            event_window=args.event_window,
            task_language=task.language,
        )
        return Pi05Rollout(
            task_suite_name=task_suite_name,
            task_id=int(task_id),
            init_state_id=int(init_state_id),
            task_language=task.language,
            target_key=target_key,
            target_key_trace=target_key_trace,
            actions=np.asarray(actions, dtype=np.float32),
            states_before_action=states,
            snapshots=snapshots,
            goal_predicates=predicates,
            semantic_quality=semantic_quality,
            failure_signature=failure_signature,
            distance_trace=[float(x) for x in distances],
            success=bool(success),
            done_step=done_step,
            initial_state_quality=initial_quality,
            initial_state_attempt=int(initial_attempt),
            reset_seed=int(reset_seed),
            video_path=video_path,
            video_frames=int(video_frames),
        )
    finally:
        env.close()


def find_failure_event(rollout: Pi05Rollout, window: int, min_delta: float) -> Optional[Tuple[int, int, float]]:
    anchor_start, anchor_end = rollout.failure_signature.anchor_interval()
    if anchor_end > anchor_start:
        return (
            max(0, anchor_start),
            min(rollout.length, anchor_end),
            float(rollout.failure_signature.confidence),
        )
    distances = np.asarray(rollout.distance_trace, dtype=np.float64)
    best = None
    for start in range(0, max(1, rollout.length - window + 1)):
        end = min(rollout.length, start + window)
        delta = float(distances[end] - distances[start])
        if best is None or delta > best[2]:
            best = (start, end, delta)
    if best is not None and best[2] >= min_delta:
        return best
    return None


def replay_candidate(
    args,
    rollout: Pi05Rollout,
    start: int,
    end: int,
    reference_signature: FailureSignature,
    client: Optional[websocket_client_policy.WebsocketClientPolicy] = None,
    action_replacements: Optional[Dict[int, np.ndarray]] = None,
    external_actions: Optional[np.ndarray] = None,
    policy_from_step: Optional[int] = None,
    prompt_override: Optional[str] = None,
    visual_intervention: Optional[dict] = None,
    stage_level: str = "pi05_natural_replay",
    trials_override: Optional[int] = None,
    early_stop_objective: str = "same_failure",
    ce_reference_rate: Optional[float] = None,
    ce_threshold: Optional[float] = None,
) -> ReplayEvaluation:
    candidate = CandidateSlice.from_window(
        start, end, n_steps=rollout.length, level=stage_level
    )
    trials = max(1, int(trials_override if trials_override is not None else args.replay_trials))
    cache_key = _replay_cache_key(
        args,
        rollout,
        start,
        end,
        reference_signature,
        client,
        action_replacements,
        external_actions,
        policy_from_step,
        prompt_override,
        visual_intervention,
        trials,
        early_stop_objective,
        ce_reference_rate,
        ce_threshold,
    )
    cache = _replay_cache(args)
    _incr(args, "replay_requests")
    timeout_seconds = float(
        getattr(args, "replay_evaluation_timeout_seconds", 0.0) or 0.0
    )
    replay_wall_start = time.perf_counter()
    _progress(
        args,
        "replay_start",
        stage_level=stage_level,
        start=int(start),
        end=int(end),
        trials=int(trials),
        objective=str(early_stop_objective),
        timeout_seconds=timeout_seconds,
        policy_from_step=None if policy_from_step is None else int(policy_from_step),
        has_external_actions=external_actions is not None,
        has_action_replacements=bool(action_replacements),
    )
    if cache is not None:
        _incr(args, "replay_cache_lookups")
        cached = cache.get(cache_key)
        if cached is not None:
            _incr(args, "replay_cache_hits")
            _progress(
                args,
                "replay_cache_hit",
                stage_level=stage_level,
                start=int(start),
                end=int(end),
                cache_key=cache_key,
            )
            return _restage_replay_evaluation(
                cached,
                start,
                end,
                rollout.length,
                stage_level,
                from_cache=True,
            )
        _incr(args, "replay_cache_misses")
    _incr(args, "replay_evaluations")
    _incr(args, "replay_trials_planned", trials)
    same_count = 0
    failure_count = 0
    repair_valid_count = 0
    executed_trials = 0
    trial_outcomes: List[Dict[str, object]] = []
    early_stop_reason = None
    representative_signature = reference_signature
    representative_evidence = compare_failure_signatures(
        reference_signature, reference_signature, threshold=args.same_failure_threshold
    )
    representative_success = False
    representative_end_distance = float(rollout.distance_trace[min(rollout.length, start)])
    external_arr = None
    if external_actions is not None:
        external_arr = np.asarray(external_actions, dtype=np.float32)
        if external_arr.ndim == 1 and external_arr.size > 0:
            external_arr = external_arr.reshape(1, -1)
    replay_stop = rollout.length
    if external_arr is not None:
        replay_stop = min(rollout.length, start + int(external_arr.shape[0]))

    timed_out = False
    for trial_index in range(trials):
        if timeout_seconds > 0 and time.perf_counter() - replay_wall_start >= timeout_seconds:
            timed_out = True
            early_stop_reason = "replay_evaluation_timeout"
            _incr(args, "replay_evaluation_timeouts")
            _progress(
                args,
                "replay_timeout",
                stage_level=stage_level,
                start=int(start),
                end=int(end),
                executed_trials=int(executed_trials),
                elapsed_seconds=float(time.perf_counter() - replay_wall_start),
            )
            break
        _progress(
            args,
            "replay_trial_start",
            stage_level=stage_level,
            trial=int(trial_index),
            start=int(start),
            end=int(end),
        )
        env, _task_suite, _task = _make_env(args, rollout.task_suite_name, rollout.task_id)
        try:
            env.reset()
            _incr(args, "env_resets")
            obs = _set_state_and_obs(env, rollout.states_before_action[start])
            predicates = get_goal_predicates(env) + custom_tasks.stage_predicates_for_suite(
                rollout.task_suite_name,
                rollout.task_id,
            )
            semantic_quality = semantic_quality_for_env(env)
            stage_tracker = custom_tasks.make_stage_tracker(
                rollout.task_suite_name,
                rollout.task_id,
                rollout.snapshots[min(start, len(rollout.snapshots) - 1)],
            )
            snapshots = [
                _semantic_snapshot(
                    rollout.task_suite_name,
                    start,
                    obs,
                    env=env,
                    action=None,
                    success=False,
                    predicates=predicates,
                    stage_tracker=stage_tracker,
                    task_id=rollout.task_id,
                )
            ]
            success = False
            last_obs = obs
            action_plan: collections.deque[np.ndarray] = collections.deque()
            for t in range(start, replay_stop):
                if (
                    timeout_seconds > 0
                    and time.perf_counter() - replay_wall_start >= timeout_seconds
                ):
                    timed_out = True
                    early_stop_reason = "replay_evaluation_timeout"
                    _incr(args, "replay_evaluation_timeouts")
                    _progress(
                        args,
                        "replay_timeout",
                        stage_level=stage_level,
                        start=int(start),
                        end=int(end),
                        trial=int(trial_index),
                        step=int(t),
                        executed_trials=int(executed_trials),
                        elapsed_seconds=float(time.perf_counter() - replay_wall_start),
                    )
                    break
                if external_arr is not None:
                    idx = t - start
                    if idx < 0 or idx >= external_arr.shape[0]:
                        break
                    action = np.asarray(external_arr[idx], dtype=np.float32)
                elif (
                    (
                        policy_from_step is not None
                        and t >= int(policy_from_step)
                    )
                    or (
                        args.continuation == "policy"
                        and t >= end
                    )
                ) and client is not None:
                    if not action_plan:
                        element = _policy_observation(
                            last_obs,
                            prompt_override or rollout.task_language,
                            args.resize_size,
                            visual_intervention=visual_intervention,
                        )
                        _incr(args, "policy_queries")
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(
                            np.asarray(action_chunk, dtype=np.float32)[: args.replan_steps]
                        )
                    action = np.asarray(action_plan.popleft(), dtype=np.float32)
                else:
                    action = np.asarray(rollout.actions[t], dtype=np.float32)
                if action_replacements is not None and t in action_replacements:
                    action = np.asarray(action_replacements[t], dtype=np.float32)

                last_obs, _reward, done, _info = env.step(action.tolist())
                _incr(args, "simulator_suffix_steps")
                step_snapshot = _semantic_snapshot(
                    rollout.task_suite_name,
                    t + 1,
                    last_obs,
                    env=env,
                    action=action,
                    success=bool(done),
                    predicates=predicates,
                    stage_tracker=stage_tracker,
                    task_id=rollout.task_id,
                )
                semantic_success = bool(done) and bool(step_snapshot.goal_truth) and all(
                    bool(v) for v in step_snapshot.goal_truth.values()
                )
                success = success or bool(semantic_success)
                snapshots.append(replace(step_snapshot, success=success))
                if done:
                    break
            if timed_out:
                break

            signature = infer_failure_signature(
                snapshots,
                predicates=predicates,
                semantic_quality=semantic_quality,
                event_window=args.event_window,
                task_language=rollout.task_language,
            )
            evidence = compare_failure_signatures(
                reference_signature,
                signature,
                threshold=args.same_failure_threshold,
            )
            if not success:
                failure_count += 1
            if evidence.same_failure:
                same_count += 1
            executed_trials += 1
            _incr(args, "replay_trials_executed")
            lower_bound, upper_bound = _same_failure_rate_bounds(
                same_count, executed_trials, trials
            )
            trial_outcome = {
                "trial": int(trial_index),
                "success": bool(success),
                "same_failure": bool(evidence.same_failure),
                "same_failure_score": float(evidence.score),
                "failure_type": signature.failure_type,
                "failed_goal_predicates": list(signature.failed_goal_predicates),
                "failed_goal_count": int(_failed_goal_count(signature)),
                "goal_progress": int(_final_goal_progress(signature)),
                "affected_objects": list(signature.affected_objects),
                "same_failure_rate_bounds": [lower_bound, upper_bound],
            }
            repair_valid = _trial_repair_pass(reference_signature, trial_outcome)
            trial_outcome["repair_valid"] = bool(repair_valid)
            if repair_valid:
                repair_valid_count += 1
            repair_lower_bound, repair_upper_bound = _same_failure_rate_bounds(
                repair_valid_count, executed_trials, trials
            )
            trial_outcome["repair_trial_bounds"] = [
                repair_lower_bound,
                repair_upper_bound,
            ]
            trial_outcomes.append(trial_outcome)
            _progress(
                args,
                "replay_trial_done",
                stage_level=stage_level,
                trial=int(trial_index),
                success=bool(success),
                same_failure=bool(evidence.same_failure),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                repair_valid=bool(repair_valid),
                repair_lower_bound=float(repair_lower_bound),
                repair_upper_bound=float(repair_upper_bound),
                executed_trials=int(executed_trials),
                failure_type=signature.failure_type,
                failed_goal_count=int(_failed_goal_count(signature)),
            )
            representative_signature = signature
            representative_evidence = evidence
            representative_success = bool(success)
            representative_end_distance = _distance(last_obs, rollout.target_key)
            if (
                bool(getattr(args, "enable_sequential_trial_pruning", True))
                and (
                    "repair" not in str(stage_level)
                    or str(early_stop_objective) == "repair_valid"
                )
            ):
                objective_lower_bound = lower_bound
                objective_upper_bound = upper_bound
                if str(early_stop_objective) == "repair_valid":
                    objective_lower_bound = repair_lower_bound
                    objective_upper_bound = repair_upper_bound
                stop_reason = _sequential_trial_stop_reason(
                    objective_lower_bound,
                    objective_upper_bound,
                    float(args.accept_same_failure_rate),
                    objective=str(early_stop_objective),
                    ce_reference_rate=ce_reference_rate,
                    ce_threshold=ce_threshold,
                )
                if stop_reason is not None:
                    early_stop_reason = stop_reason
                    _incr(args, "sequential_trial_early_stops")
                    _progress(
                        args,
                        "replay_early_stop",
                        stage_level=stage_level,
                        reason=stop_reason,
                        lower_bound=float(objective_lower_bound),
                        upper_bound=float(objective_upper_bound),
                        executed_trials=int(executed_trials),
                    )
                    break
        finally:
            env.close()

    start_distance = float(rollout.distance_trace[start])
    delta = float(representative_end_distance - start_distance)
    lower_bound, upper_bound = _same_failure_rate_bounds(same_count, executed_trials, trials)
    repair_lower_bound, repair_upper_bound = _same_failure_rate_bounds(
        repair_valid_count, executed_trials, trials
    )
    same_failure = bool(lower_bound >= float(args.accept_same_failure_rate))
    same_failure_rate = float(lower_bound if same_failure else upper_bound)
    same_failure = (
        same_failure
        and candidate_overlaps_failure_anchor(candidate, reference_signature)
    )
    evaluation = ReplayEvaluation(
        candidate=candidate,
        same_failure=bool(same_failure),
        same_failure_rate=same_failure_rate,
        failure_rate=float(failure_count / max(1, executed_trials)),
        trials=int(executed_trials),
        success=bool(representative_success),
        start_distance=start_distance,
        end_distance=float(representative_end_distance),
        distance_delta=delta,
        steps=int(max(0, end - start)),
        signature=representative_signature,
        same_failure_evidence=representative_evidence,
        planned_trials=int(trials),
        executed_trials=int(executed_trials),
        same_failure_count=int(same_count),
        same_failure_rate_lower_bound=float(lower_bound),
        same_failure_rate_upper_bound=float(upper_bound),
        trial_outcomes=tuple(trial_outcomes),
        early_stop_reason=early_stop_reason,
        from_cache=False,
        cache_key=cache_key,
        repair_valid_count=int(repair_valid_count),
        repair_valid_rate_lower_bound=float(repair_lower_bound),
        repair_valid_rate_upper_bound=float(repair_upper_bound),
    )
    if cache is not None:
        cache[cache_key] = evaluation
    _progress(
        args,
        "replay_done",
        stage_level=stage_level,
        start=int(start),
        end=int(end),
        same_failure=bool(evaluation.same_failure),
        same_failure_rate=float(evaluation.same_failure_rate),
        repair_valid_count=int(evaluation.repair_valid_count),
        repair_lower_bound=float(evaluation.repair_valid_rate_lower_bound),
        repair_upper_bound=float(evaluation.repair_valid_rate_upper_bound),
        executed_trials=int(evaluation.executed_trials),
        planned_trials=int(evaluation.planned_trials),
        early_stop_reason=evaluation.early_stop_reason,
        elapsed_seconds=float(time.perf_counter() - replay_wall_start),
    )
    return evaluation


def _replacement_actions(
    actions: np.ndarray,
    interval: Tuple[int, int],
    strategy: str,
) -> Dict[int, np.ndarray]:
    start, end = interval
    replacements: Dict[int, np.ndarray] = {}
    if end <= start:
        return replacements
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[0] <= 0 or start >= arr.shape[0]:
        return replacements
    before = arr[max(0, start - 1)].copy()
    after = arr[min(arr.shape[0] - 1, end)].copy() if end < arr.shape[0] else before.copy()
    for t in range(start, min(end, arr.shape[0])):
        action = arr[t].copy()
        if strategy == "hold":
            action[:] = 0.0
            if action.shape[0] >= 7:
                action[6] = before[6]
        elif strategy == "adjacent":
            action[:] = before if t - start < (end - start) / 2 else after
        elif strategy == "gripper_correction":
            action[: min(6, action.shape[0])] = 0.0
            if action.shape[0] >= 7:
                action[6] = -float(np.sign(action[6]) or 1.0)
        else:
            raise ValueError("Unknown replacement strategy: %s" % strategy)
        replacements[t] = action
    return replacements


def _nearest_success_repair_actions(
    rollouts: Sequence[Pi05Rollout],
    selected: Pi05Rollout,
    start: int,
    max_steps: int,
) -> Optional[dict]:
    if not rollouts:
        return None
    ref = selected.snapshots[min(int(start), len(selected.snapshots) - 1)]
    ref_eef = np.asarray(ref.eef_pos, dtype=np.float64)
    best = None
    for rollout in rollouts:
        if rollout is selected or not rollout.success:
            continue
        if rollout.task_language != selected.task_language:
            continue
        for idx, snapshot in enumerate(rollout.snapshots[:-1]):
            dist = float(np.linalg.norm(np.asarray(snapshot.eef_pos, dtype=np.float64) - ref_eef))
            if best is None or dist < best["state_l2"]:
                best = {"rollout": rollout, "start": idx, "state_l2": dist}
    if best is None:
        return None
    rollout = best["rollout"]
    demo_start = min(int(best["start"]), rollout.length)
    demo_end = min(rollout.length, demo_start + max(1, int(max_steps)))
    actions = np.asarray(rollout.actions[demo_start:demo_end], dtype=np.float32)
    if actions.size == 0:
        return None
    requested_steps = max(1, int(max_steps))
    return {
        "source": "successful_rollout_nearest_neighbor",
        "task_id": int(rollout.task_id),
        "init_state_id": int(rollout.init_state_id),
        "nearest_action_index": int(demo_start),
        "nearest_state_l2": float(best["state_l2"]),
        "num_actions": int(actions.shape[0]),
        "requested_steps": int(requested_steps),
        "action_suffix_complete": bool(actions.shape[0] >= requested_steps),
        "actions": actions,
    }


def _demo_repair_actions(
    args,
    rollout: Pi05Rollout,
    start: int,
    max_steps: int,
) -> Optional[dict]:
    dataset_root = Path(args.demo_dataset_root)
    if not dataset_root.exists() or not OPENPI_PYTHON.exists():
        return None
    snapshot_index = min(int(start), len(rollout.snapshots) - 1)
    eef = None
    for idx in [snapshot_index, *range(max(0, snapshot_index - 5), min(len(rollout.snapshots), snapshot_index + 6))]:
        candidate = getattr(rollout.snapshots[idx], "eef_pos", None)
        if candidate is None:
            continue
        arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            eef = arr[:3]
            break
    if eef is None:
        return {
            "source": "lerobot_demo_nearest_neighbor",
            "available": False,
            "reason": "eef_pos_unavailable_for_demo_lookup",
        }
    eef_arg = ",".join(str(float(x)) for x in eef[:3])
    cmd = [
        str(OPENPI_PYTHON),
        str(PROJECT_ROOT / "libero_demo_repair_source.py"),
        "--dataset-root",
        str(dataset_root),
        "--task-language",
        rollout.task_language,
        f"--eef-pos={eef_arg}",
        "--max-steps",
        str(max_steps),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=float(args.demo_repair_timeout_seconds),
        )
    except Exception as exc:
        return {
            "source": "lerobot_demo_nearest_neighbor",
            "available": False,
            "reason": "demo_lookup_exception:%s" % type(exc).__name__,
        }
    if proc.returncode != 0:
        return {
            "source": "lerobot_demo_nearest_neighbor",
            "available": False,
            "reason": "demo_lookup_failed",
            "stderr_tail": proc.stderr[-1000:],
        }
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "source": "lerobot_demo_nearest_neighbor",
            "available": False,
            "reason": "demo_lookup_invalid_json",
            "stdout_tail": proc.stdout[-1000:],
        }
    if not payload.get("available"):
        return payload
    actions = np.asarray(payload.get("actions") or [], dtype=np.float32)
    payload = dict(payload)
    payload["actions"] = actions
    return payload


def _scripted_expert_repair(
    args,
    rollout: Pi05Rollout,
    start: int,
    max_steps: int,
) -> dict:
    snapshot = rollout.snapshots[min(int(start), len(rollout.snapshots) - 1)]
    metadata = custom_tasks.expert_repair_metadata(
        rollout.task_suite_name,
        rollout.task_id,
        snapshot,
    )
    if not metadata or not metadata.get("available"):
        return {
            "source": "scripted_stage_expert_repair",
            "repair_pass": False,
            "unavailable": True,
            "source_metadata": metadata,
        }

    expert_max_steps = min(
        max(1, int(max_steps)),
        max(1, int(getattr(args, "scripted_expert_repair_max_steps", 180))),
    )
    trials = max(1, int(args.repair_replay_trials))
    _incr(args, "scripted_expert_repair_evaluations")
    _incr(args, "scripted_expert_repair_trials", trials)

    candidate = CandidateSlice.from_window(
        start,
        min(rollout.length, start + expert_max_steps),
        n_steps=rollout.length,
        level="scripted_stage_expert_repair",
    )
    same_count = 0
    failure_count = 0
    representative_signature = rollout.failure_signature
    representative_evidence = compare_failure_signatures(
        rollout.failure_signature,
        rollout.failure_signature,
        threshold=args.same_failure_threshold,
    )
    representative_success = False
    representative_end_distance = float(rollout.distance_trace[min(rollout.length, start)])
    representative_actions: List[np.ndarray] = []
    representative_controller_state: dict = {}

    for _trial in range(trials):
        env, _task_suite, _task = _make_env(args, rollout.task_suite_name, rollout.task_id)
        try:
            env.reset()
            _incr(args, "env_resets")
            obs = _set_state_and_obs(env, rollout.states_before_action[start])
            predicates = get_goal_predicates(env) + custom_tasks.stage_predicates_for_suite(
                rollout.task_suite_name,
                rollout.task_id,
            )
            semantic_quality = semantic_quality_for_env(env)
            stage_tracker = custom_tasks.make_stage_tracker(
                rollout.task_suite_name,
                rollout.task_id,
                snapshot,
            )
            snapshots = [
                _semantic_snapshot(
                    rollout.task_suite_name,
                    start,
                    obs,
                    env=env,
                    action=None,
                    success=False,
                    predicates=predicates,
                    stage_tracker=stage_tracker,
                    task_id=rollout.task_id,
                )
            ]
            controller_state: dict = {}
            actions: List[np.ndarray] = []
            success = False
            last_obs = obs
            for local_t in range(expert_max_steps):
                current_snapshot = snapshots[-1]
                action = custom_tasks.expert_action(
                    rollout.task_suite_name,
                    controller_state,
                    env,
                    last_obs,
                    rollout.task_id,
                    current_snapshot,
                )
                if action is None:
                    break
                action = np.asarray(action, dtype=np.float32)
                actions.append(action)
                last_obs, _reward, done, _info = env.step(action.tolist())
                _incr(args, "simulator_suffix_steps")
                step_snapshot = _semantic_snapshot(
                    rollout.task_suite_name,
                    start + local_t + 1,
                    last_obs,
                    env=env,
                    action=action,
                    success=bool(done),
                    predicates=predicates,
                    stage_tracker=stage_tracker,
                    task_id=rollout.task_id,
                )
                semantic_success = bool(done) and bool(step_snapshot.goal_truth) and all(
                    bool(v) for v in step_snapshot.goal_truth.values()
                )
                success = success or bool(semantic_success)
                snapshots.append(replace(step_snapshot, success=success))
                if done:
                    break

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
            if not success:
                failure_count += 1
            if evidence.same_failure:
                same_count += 1
            representative_signature = signature
            representative_evidence = evidence
            representative_success = bool(success)
            representative_end_distance = _distance(last_obs, rollout.target_key)
            representative_actions = actions
            representative_controller_state = dict(controller_state)
        finally:
            env.close()

    evaluation = ReplayEvaluation(
        candidate=candidate,
        same_failure=bool(same_count / trials >= float(args.accept_same_failure_rate)),
        same_failure_rate=float(same_count / trials),
        failure_rate=float(failure_count / trials),
        trials=int(trials),
        success=bool(representative_success),
        start_distance=float(rollout.distance_trace[start]),
        end_distance=float(representative_end_distance),
        distance_delta=float(representative_end_distance - float(rollout.distance_trace[start])),
        steps=int(len(representative_actions)),
        signature=representative_signature,
        same_failure_evidence=representative_evidence,
    )
    repair_evidence = _counterfactual_repair_evidence(rollout.failure_signature, evaluation)
    max_store = max(0, int(getattr(args, "expert_repair_max_actions_to_store", 128)))
    return {
        "source": "scripted_stage_expert_repair",
        "repair_pass": bool(repair_evidence["repair_pass"]),
        "evaluation": evaluation.to_dict(),
        "repair_evidence": repair_evidence,
        "source_metadata": {
            **metadata,
            "num_actions": int(len(representative_actions)),
            "actions_truncated": bool(len(representative_actions) > max_store),
            "actions": [
                np.asarray(action, dtype=np.float32).reshape(-1).tolist()
                for action in representative_actions[:max_store]
            ],
            "controller_final_state": representative_controller_state,
        },
    }


def _action_summary(actions: np.ndarray) -> dict:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1 and arr.size > 0:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return {
            "num_actions": 0,
            "action_dim": 0,
            "mean": [],
            "std": [],
            "max_abs": [],
            "gripper_transitions": 0,
        }
    gripper_transitions = 0
    if arr.shape[1] >= 7 and arr.shape[0] > 1:
        gripper_transitions = int(np.count_nonzero(np.abs(np.diff(arr[:, 6])) > 0.2))
    return {
        "num_actions": int(arr.shape[0]),
        "action_dim": int(arr.shape[1]),
        "mean": [float(x) for x in arr.mean(axis=0)],
        "std": [float(x) for x in arr.std(axis=0)],
        "max_abs": [float(x) for x in np.max(np.abs(arr), axis=0)],
        "gripper_transitions": gripper_transitions,
    }


def _snapshot_training_features(snapshot: StateSnapshot) -> dict:
    data = snapshot.to_dict()
    return {
        "t": data["t"],
        "success": data["success"],
        "eef_pos": data["eef_pos"],
        "gripper_qpos": data["gripper_qpos"],
        "goal_truth": data["goal_truth"],
        "stage_features": _stage_features_from_goal_truth(data["goal_truth"]),
        "object_positions": data["object_positions"],
    }


def _stage_features_from_goal_truth(goal_truth: dict) -> dict:
    labels = sorted(
        key
        for key in (goal_truth or {})
        if str(key).startswith("Stage") and str(key) != "Stageorder valid_sequence_so_far"
    )
    earliest_failed = next(
        (label for label in labels if not bool(goal_truth.get(label, False))),
        None,
    )
    return {
        "stage_labels": labels,
        "stage_progress_count": sum(1 for label in labels if bool(goal_truth.get(label, False))),
        "earliest_failed_stage": earliest_failed,
        "stage_order_valid": bool(goal_truth.get("Stageorder valid_sequence_so_far", True)),
    }


def _numeric_delta(before: object, after: object, length: int) -> List[float]:
    a = np.asarray(before if before is not None else [], dtype=np.float32).reshape(-1)
    b = np.asarray(after if after is not None else [], dtype=np.float32).reshape(-1)
    if a.size < length:
        a = np.pad(a, (0, length - a.size))
    if b.size < length:
        b = np.pad(b, (0, length - b.size))
    return [float(x) for x in (b[:length] - a[:length])]


def _state_delta_features(pre: dict, post: dict) -> dict:
    pre_goal = pre.get("goal_truth") or {}
    post_goal = post.get("goal_truth") or {}
    goal_delta = {
        key: int(bool(post_goal.get(key, False))) - int(bool(pre_goal.get(key, False)))
        for key in sorted(set(pre_goal) | set(post_goal))
    }

    pre_objects = pre.get("object_positions") or {}
    post_objects = post.get("object_positions") or {}
    object_delta = {}
    for key in sorted(set(pre_objects) | set(post_objects)):
        before = pre_objects.get(key)
        after = post_objects.get(key)
        if before is None or after is None:
            continue
        delta = _numeric_delta(before, after, 3)
        object_delta[key] = {
            "delta": delta,
            "l2": float(np.linalg.norm(np.asarray(delta, dtype=np.float32))),
        }

    return {
        "eef_delta": _numeric_delta(pre.get("eef_pos"), post.get("eef_pos"), 3),
        "gripper_delta": _numeric_delta(
            pre.get("gripper_qpos"), post.get("gripper_qpos"), 2
        ),
        "goal_truth_delta": goal_delta,
        "object_position_delta": object_delta,
    }


def _actions_for_window(
    rollout: Pi05Rollout,
    start: int,
    end: int,
    action_replacements: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    arr = np.asarray(rollout.actions[start:end], dtype=np.float32).copy()
    if action_replacements is None or arr.size == 0:
        return arr
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    dim = arr.shape[1]
    for t, replacement in action_replacements.items():
        idx = int(t) - int(start)
        if idx < 0 or idx >= arr.shape[0]:
            continue
        rep = np.asarray(replacement, dtype=np.float32).reshape(-1)
        if rep.size < dim:
            rep = np.pad(rep, (0, dim - rep.size))
        arr[idx] = rep[:dim]
    return arr


def _slice_training_features(
    rollout: Pi05Rollout,
    candidate: Optional[CandidateSlice],
    max_raw_actions: int = 128,
    action_replacements: Optional[Dict[int, np.ndarray]] = None,
) -> Optional[dict]:
    if candidate is None or candidate.span_start is None or candidate.span_end is None:
        return None
    start = max(0, min(int(candidate.span_start), rollout.length))
    end = max(start, min(int(candidate.span_end), rollout.length))
    action_chunk = _actions_for_window(
        rollout,
        start,
        end,
        action_replacements=action_replacements,
    )
    truncated = action_chunk.shape[0] > int(max_raw_actions)
    raw_actions = action_chunk[:max_raw_actions].tolist()
    pre_snapshot = _snapshot_training_features(
        rollout.snapshots[min(start, len(rollout.snapshots) - 1)]
    )
    post_snapshot = _snapshot_training_features(
        rollout.snapshots[min(end, len(rollout.snapshots) - 1)]
    )
    return {
        "feature_quality": "full",
        "candidate_window": [start, end],
        "candidate_actions": raw_actions,
        "candidate_actions_truncated": bool(truncated),
        "action_summary": _action_summary(action_chunk),
        "pre_state_features": pre_snapshot,
        "post_state_features": post_snapshot,
        "state_delta_features": _state_delta_features(pre_snapshot, post_snapshot),
    }


def _candidate_window(
    rollout: Pi05Rollout,
    start: int,
    end: int,
    level: str,
) -> CandidateSlice:
    return CandidateSlice.from_window(start, end, n_steps=rollout.length, level=level)


def _interval_overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(int(a[0]), int(b[0])) < min(int(a[1]), int(b[1]))


def _even_windows(
    length: int,
    window_size: int,
    max_windows: int,
    exclude: Sequence[Tuple[int, int]] = (),
) -> List[Tuple[int, int]]:
    if length <= 1 or max_windows <= 0:
        return []
    size = max(1, min(int(window_size), int(length)))
    max_start = max(0, int(length) - size)
    if max_start == 0:
        starts = [0]
    else:
        count = min(int(max_windows) * 3, max_start + 1)
        starts = sorted(
            {
                int(round(i * max_start / max(1, count - 1)))
                for i in range(count)
            }
        )
    windows: List[Tuple[int, int]] = []
    for start in starts:
        end = min(int(length), int(start) + size)
        if any(_interval_overlaps((start, end), item) for item in exclude):
            continue
        windows.append((start, end))
        if len(windows) >= int(max_windows):
            break
    return windows


def _risk_training_window_sample(
    rollout: Pi05Rollout,
    candidate: CandidateSlice,
    sample_kind: str,
    label: int,
    label_source: str,
    sample_id: str,
    same_failure_rate: float = 0.0,
    failure_rate: float = 0.0,
    causal_effect: Optional[float] = None,
    causal_validation_passed: bool = False,
    action_replacements: Optional[Dict[int, np.ndarray]] = None,
    extra: Optional[dict] = None,
) -> Optional[dict]:
    features = _slice_training_features(
        rollout,
        candidate,
        action_replacements=action_replacements,
    )
    if features is None:
        return None
    sample = {
        "sample_id": sample_id,
        "sample_kind": sample_kind,
        "label": int(label),
        "label_source": label_source,
        "task": {
            "task_suite_name": rollout.task_suite_name,
            "task_id": int(rollout.task_id),
            "init_state_id": int(rollout.init_state_id),
            "task_language": rollout.task_language,
        },
        "candidate": candidate.to_dict(),
        "features": features,
        "same_failure_rate": float(same_failure_rate),
        "failure_rate": float(failure_rate),
        "causal_effect": None if causal_effect is None else float(causal_effect),
        "causal_validation_passed": bool(causal_validation_passed),
    }
    if extra:
        sample.update(extra)
    return sample


def build_risk_training_windows(
    args,
    rollouts: Sequence[Pi05Rollout],
    selected: Optional[Pi05Rollout],
    event: Optional[Tuple[int, int, float]],
    final_eval: Optional[ReplayEvaluation],
    causal_validation: Optional[CausalValidationResult],
) -> List[dict]:
    windows: List[dict] = []
    window_size = max(2, int(args.event_window))

    for rollout in rollouts:
        if not rollout.initial_state_valid:
            continue
        if not rollout.success:
            continue
        for i, (start, end) in enumerate(
            _even_windows(rollout.length, window_size, max_windows=5)
        ):
            candidate = _candidate_window(
                rollout, start, end, level="successful_rollout_window"
            )
            sample = _risk_training_window_sample(
                rollout,
                candidate,
                sample_kind="successful_rollout_window",
                label=0,
                label_source="successful_rollout_negative_window",
                sample_id="success_%d_%d_%03d" % (rollout.task_id, rollout.init_state_id, i),
            )
            if sample is not None:
                windows.append(sample)

    if selected is not None and event is not None:
        exclude = [(int(event[0]), int(event[1]))]
        for i, (start, end) in enumerate(
            _even_windows(selected.length, window_size, max_windows=5, exclude=exclude)
        ):
            candidate = _candidate_window(
                selected, start, end, level="failure_event_outside_window"
            )
            sample = _risk_training_window_sample(
                selected,
                candidate,
                sample_kind="failure_event_outside_window",
                label=0,
                label_source="outside_failure_event_negative_window",
                sample_id="outside_%d_%d_%03d"
                % (selected.task_id, selected.init_state_id, i),
            )
            if sample is not None:
                windows.append(sample)

    if selected is not None and final_eval is not None:
        same_rate = float(final_eval.same_failure_rate)
        same_failure_ready = bool(
            final_eval.same_failure and same_rate >= float(args.accept_same_failure_rate)
        )
        same_failure_necessity_ready = bool(
            same_failure_ready
            and causal_validation is not None
            and causal_validation.same_failure_necessity_pass
        )
        if same_failure_necessity_ready:
            sample = _risk_training_window_sample(
                selected,
                final_eval.candidate,
                sample_kind="same_failure_necessity_slice",
                label=1,
                label_source="semantic_same_failure_minimal_slice",
                sample_id="minimal_%d_%d" % (selected.task_id, selected.init_state_id),
                same_failure_rate=same_rate,
                failure_rate=float(final_eval.failure_rate),
                causal_validation_passed=bool(
                    causal_validation is not None
                    and causal_validation.same_failure_necessity_pass
                ),
            )
        elif not same_failure_ready:
            sample = _risk_training_window_sample(
                selected,
                final_eval.candidate,
                sample_kind="non_matching_candidate_slice",
                label=0,
                label_source="same_failure_rejected_candidate",
                sample_id="nonmatching_%d_%d" % (selected.task_id, selected.init_state_id),
                same_failure_rate=same_rate,
                failure_rate=float(final_eval.failure_rate),
            )
        else:
            sample = None
        if sample is not None:
            windows.append(sample)

    if selected is not None and causal_validation is not None:
        for result in causal_validation.necessity_core_units:
            start, end = result.unit.interval
            candidate = _candidate_window(
                selected, start, end, level="same_failure_necessity_core"
            )
            sample = _risk_training_window_sample(
                selected,
                candidate,
                sample_kind="same_failure_necessity_core",
                label=1,
                label_source="destructive_ablation_same_failure_necessity",
                sample_id=result.unit.unit_id,
                same_failure_rate=float(result.base_same_failure_rate),
                causal_effect=float(result.causal_effect),
                causal_validation_passed=bool(causal_validation.same_failure_necessity_pass),
                extra={"causal_unit": result.unit.to_dict()},
            )
            if sample is not None:
                windows.append(sample)

        repair_sources = (
            (
                "policy_raw_repair_valid_core",
                "raw_policy_repair_valid_causal_core_unit",
                causal_validation.raw_policy_core_units,
            ),
            (
                "policy_language_phrase_repair_valid_core",
                "language_phrase_repair_valid_causal_core_unit",
                causal_validation.language_phrase_core_units,
            ),
            (
                "policy_visual_mask_repair_valid_core",
                "visual_mask_repair_valid_causal_core_unit",
                causal_validation.visual_mask_core_units,
            ),
            (
                "demo_existence_repair_valid_core",
                "demo_existence_repair_valid_causal_core_unit",
                causal_validation.demo_existence_core_units,
            ),
        )
        seen_repair_samples = set()
        for sample_kind, label_source, units in repair_sources:
            for result in units:
                key = (sample_kind, result.unit.unit_id)
                if key in seen_repair_samples:
                    continue
                seen_repair_samples.add(key)
                start, end = result.unit.interval
                candidate = _candidate_window(
                    selected, start, end, level=sample_kind
                )
                sample = _risk_training_window_sample(
                    selected,
                    candidate,
                    sample_kind=sample_kind,
                    label=1,
                    label_source=label_source,
                    sample_id="repair_%s_%s" % (sample_kind, result.unit.unit_id),
                    same_failure_rate=float(result.base_same_failure_rate),
                    causal_effect=float(result.causal_effect),
                    causal_validation_passed=bool(causal_validation.passed),
                    extra={
                        "causal_unit": result.unit.to_dict(),
                        "repair_evidence": result.repair_evidence,
                        "positive_source": sample_kind,
                    },
                )
                if sample is not None:
                    windows.append(sample)

    return windows


def _goal_trace_payload(signature: FailureSignature) -> dict:
    evidence = signature.evidence or {}
    trace = evidence.get("goal_trace") if isinstance(evidence, dict) else {}
    return trace if isinstance(trace, dict) else {}


def _final_goal_progress(signature: FailureSignature) -> int:
    trace = _goal_trace_payload(signature)
    progress = trace.get("progress_counts")
    if isinstance(progress, list) and progress:
        return int(progress[-1])
    final_truth = trace.get("final_truth")
    if isinstance(final_truth, dict):
        return sum(1 for value in final_truth.values() if bool(value))
    return 0


def _failed_goal_count(signature: FailureSignature) -> int:
    trace = _goal_trace_payload(signature)
    failed = trace.get("failed_final_predicates")
    if isinstance(failed, list):
        return len(failed)
    return len(signature.failed_goal_predicates)


def _failed_goal_set(signature: FailureSignature) -> set:
    trace = _goal_trace_payload(signature)
    failed = trace.get("failed_final_predicates")
    if isinstance(failed, list):
        return {str(item) for item in failed}
    return {str(item) for item in signature.failed_goal_predicates}


def _trial_repair_pass(reference_signature: FailureSignature, outcome: Dict[str, object]) -> bool:
    base_failed = _failed_goal_count(reference_signature)
    base_failed_set = _failed_goal_set(reference_signature)
    base_progress = _final_goal_progress(reference_signature)
    outcome_failed_set = {str(x) for x in outcome.get("failed_goal_predicates", [])}
    outcome_failed = int(outcome.get("failed_goal_count", len(outcome_failed_set)))
    outcome_progress = int(outcome.get("goal_progress", 0))
    outcome_affected = set(str(x) for x in outcome.get("affected_objects", []))
    outcome_non_worsening = bool(
        outcome_failed <= base_failed
        and outcome_progress >= base_progress
        and outcome_failed_set.issubset(base_failed_set)
        and len(outcome_affected - set(reference_signature.affected_objects)) == 0
    )
    outcome_improved = bool(
        outcome.get("success")
        or outcome_failed < base_failed
        or outcome_progress > base_progress
    )
    return bool(outcome.get("success") or (outcome_non_worsening and outcome_improved))


def _counterfactual_repair_evidence(
    reference_signature: FailureSignature,
    evaluation: ReplayEvaluation,
) -> dict:
    base_failed = _failed_goal_count(reference_signature)
    cf_failed = _failed_goal_count(evaluation.signature)
    base_failed_set = _failed_goal_set(reference_signature)
    cf_failed_set = _failed_goal_set(evaluation.signature)
    base_progress = _final_goal_progress(reference_signature)
    cf_progress = _final_goal_progress(evaluation.signature)
    failed_goal_subset = bool(cf_failed_set.issubset(base_failed_set))
    affected_growth = max(
        0,
        len(set(evaluation.signature.affected_objects))
        - len(set(reference_signature.affected_objects)),
    )
    non_worsening = bool(
        cf_failed <= base_failed
        and cf_progress >= base_progress
        and failed_goal_subset
        and affected_growth == 0
    )
    improved = bool(
        evaluation.success
        or cf_failed < base_failed
        or cf_progress > base_progress
    )
    trial_repairs = []
    for outcome in evaluation.trial_outcomes:
        trial_repairs.append(_trial_repair_pass(reference_signature, outcome))
    planned = max(1, int(evaluation.planned_trials or evaluation.trials or 1))
    repair_count = int(sum(1 for item in trial_repairs if item))
    repair_rate = float(repair_count / planned)
    aggregate_repair_pass = bool(repair_rate >= 0.80)
    representative_repair_pass = bool(evaluation.success or (non_worsening and improved))
    repair_pass = bool(
        aggregate_repair_pass
        or (
            representative_repair_pass
            and planned <= 1
        )
    )
    return {
        "repair_pass": repair_pass,
        "success": bool(
            aggregate_repair_pass
            and any(bool(outcome.get("success")) for outcome in evaluation.trial_outcomes)
        )
        if evaluation.trial_outcomes
        else bool(evaluation.success),
        "repair_pass_rate": repair_rate,
        "repair_pass_count": repair_count,
        "repair_pass_planned_trials": planned,
        "repair_trial_outcomes": trial_repairs,
        "base_failed_goal_count": int(base_failed),
        "counterfactual_failed_goal_count": int(cf_failed),
        "base_failed_goal_predicates": sorted(base_failed_set),
        "counterfactual_failed_goal_predicates": sorted(cf_failed_set),
        "failed_goal_subset_of_original": failed_goal_subset,
        "failed_goal_growth": max(0, len(cf_failed_set - base_failed_set)),
        "base_goal_progress": int(base_progress),
        "counterfactual_goal_progress": int(cf_progress),
        "affected_object_growth": int(affected_growth),
        "non_worsening": non_worsening,
        "improved": improved,
        "counterfactual_failure_type": evaluation.signature.failure_type,
    }


def _objects_and_regions_from_failure(signature: FailureSignature) -> Tuple[List[str], List[str]]:
    objects = []
    regions = []
    for pred in signature.failed_goal_predicates:
        parts = str(pred).split()
        for arg in parts[1:]:
            if arg.endswith("_region"):
                regions.append(arg)
            elif arg:
                objects.append(arg)
    for obj in signature.affected_objects:
        if obj and obj not in objects:
            objects.append(str(obj))
    return sorted(set(objects)), sorted(set(regions))


def _moved_distractors_from_signature(signature: FailureSignature) -> List[str]:
    motion = (signature.evidence or {}).get("motion", {})
    non_target = motion.get("non_target_objects") if isinstance(motion, dict) else {}
    if not isinstance(non_target, dict):
        return []
    distractors = []
    for obj, evidence in non_target.items():
        if not isinstance(evidence, dict):
            continue
        if float(evidence.get("displacement", 0.0)) >= 0.03:
            distractors.append(str(obj))
    return sorted(distractors)


def _phrase_span(text: str, phrase: str) -> Optional[List[int]]:
    idx = text.lower().find(str(phrase).lower())
    if idx < 0:
        return None
    return [int(idx), int(idx + len(str(phrase)))]


def _language_phrase_candidates(
    task_language: str,
    objects: Sequence[str],
    regions: Sequence[str],
    distractors: Sequence[str],
) -> List[dict]:
    candidates: List[dict] = []
    for kind, values in (
        ("intended_object", objects),
        ("goal_region", regions),
        ("avoid_distractor", distractors),
    ):
        for value in values:
            span = _phrase_span(task_language, str(value))
            candidates.append(
                {
                    "kind": kind,
                    "phrase": str(value),
                    "char_span": span,
                    "found_in_original_prompt": span is not None,
                }
            )
    return candidates


def _rule_language_intervention(
    rollout: Pi05Rollout,
    unit: object,
) -> dict:
    objects, regions = _objects_and_regions_from_failure(rollout.failure_signature)
    unit_evidence = getattr(unit, "evidence", {}) or {}
    target_object = unit_evidence.get("target_object")
    if target_object and str(target_object) not in objects:
        objects.insert(0, str(target_object))
    distractors = _moved_distractors_from_signature(rollout.failure_signature)
    if unit_evidence.get("role") == "moved_distractor":
        distractor = unit_evidence.get("target_object")
        if distractor and str(distractor) not in distractors:
            distractors.insert(0, str(distractor))
    original_prompt = str(rollout.task_language).strip()
    phrase_candidates = _language_phrase_candidates(
        original_prompt,
        objects[:4],
        regions[:4],
        distractors[:4],
    )
    selected_phrase = next(
        (
            item
            for item in phrase_candidates
            if item.get("kind") in {"intended_object", "avoid_distractor"}
        ),
        phrase_candidates[0] if phrase_candidates else None,
    )
    clauses = [original_prompt]
    if objects:
        clauses.append("Intended target object(s): %s." % ", ".join(objects[:4]))
    if regions:
        clauses.append("Intended goal region(s): %s." % ", ".join(regions[:4]))
    if distractors:
        clauses.append("Avoid moving distractor object(s): %s." % ", ".join(distractors[:4]))
    clauses.append(
        "Continue from the current simulator state and complete the original goal."
    )
    prompt = " ".join(clauses)
    return {
        "schema_version": "shed-cfs-language-phrase-intervention-v1",
        "unit_id": getattr(unit, "unit_id", None),
        "source": "phrase_level_bddl_disambiguation",
        "prompt": prompt,
        "original_prompt": original_prompt,
        "selected_phrase": selected_phrase,
        "phrase_candidates": phrase_candidates,
        "prompt_diff": {
            "operation": "append_disambiguating_phrase_clauses",
            "added_text": prompt[len(original_prompt) :].strip(),
            "phrase_level_delta_debug": True,
        },
        "intended_objects": objects,
        "goal_regions": regions,
        "avoid_distractors": distractors,
    }


def _visual_policy_mask_intervention(
    args,
    rollout: Pi05Rollout,
    unit: object,
) -> dict:
    unit_evidence = getattr(unit, "evidence", {}) or {}
    target = unit_evidence.get("target_object")
    if not bool(getattr(args, "enable_visual_policy_mask", False)):
        return {
            "schema_version": "shed-cfs-visual-policy-mask-v1",
            "unit_id": getattr(unit, "unit_id", None),
            "enabled": False,
            "visual_quality": "skipped",
            "reason": "visual_policy_mask_disabled",
        }
    if target is None:
        objects, _regions = _objects_and_regions_from_failure(rollout.failure_signature)
        target = objects[0] if objects else None
    snapshot_index = max(
        0,
        min(
            int(getattr(unit, "interval", (0, 1))[0]),
            max(0, len(rollout.snapshots) - 1),
        ),
    )
    snapshot = rollout.snapshots[snapshot_index] if rollout.snapshots else None
    rect = None
    visual_quality = "degraded"
    reason = "object_to_policy_image_projection_unavailable"
    projection_source = "unavailable"
    if (
        target
        and snapshot is not None
        and target in getattr(snapshot, "object_positions", {})
    ):
        # v4 keeps visual interventions honest: if a calibrated MuJoCo camera
        # projection is unavailable in the stored rollout, use a conservative
        # workspace-to-image heuristic and mark the evidence as degraded.
        pos = np.asarray(snapshot.object_positions[target], dtype=np.float64)
        size = int(getattr(args, "resize_size", 224))
        x = int(np.clip((pos[0] + 0.35) / 0.70 * size, 0, size - 1))
        y = int(np.clip((0.85 - pos[1]) / 0.70 * size, 0, size - 1))
        half = max(10, size // 16)
        rect = [
            int(max(0, x - half)),
            int(max(0, y - half)),
            int(min(size, x + half)),
            int(min(size, y + half)),
        ]
        visual_quality = "degraded_workspace_projection"
        reason = "calibrated_mujoco_camera_projection_unavailable_used_workspace_heuristic"
        projection_source = "workspace_xy_heuristic"
    if rect is not None:
        role = str(unit_evidence.get("role") or "")
        mode = "mask_distractor" if role == "moved_distractor" else "highlight_target"
        return {
            "schema_version": "shed-cfs-visual-grounding-mask-v1",
            "unit_id": getattr(unit, "unit_id", None),
            "enabled": True,
            "target_object": target,
            "mode": mode,
            "image_rect": rect,
            "agent_rect": rect,
            "wrist_rect": rect,
            "visual_quality": visual_quality,
            "projection_source": projection_source,
            "reason": reason,
            "applied_to_policy_input": True,
            "physical_world_unchanged": True,
        }
    return {
        "schema_version": "shed-cfs-visual-grounding-mask-v1",
        "unit_id": getattr(unit, "unit_id", None),
        "enabled": True,
        "target_object": target,
        "visual_quality": visual_quality,
        "projection_source": projection_source,
        "reason": reason,
        "applied_to_policy_input": False,
        "physical_world_unchanged": True,
    }


def _causal_ablation_strategies(args) -> Tuple[str, ...]:
    allowed = {"hold", "adjacent", "gripper_correction"}
    text = str(getattr(args, "causal_ablation_strategies", "") or "")
    strategies = tuple(item.strip() for item in text.split(",") if item.strip())
    if not strategies:
        strategies = ("hold", "adjacent", "gripper_correction")
    invalid = [item for item in strategies if item not in allowed]
    if invalid:
        raise ValueError(
            "Unsupported causal ablation strategies %s. Allowed: %s"
            % (invalid, sorted(allowed))
        )
    return strategies


def _causal_unit_group(unit: CausalUnit) -> str:
    kind = str(unit.kind)
    if kind in {"action_chunk", "stage_phase_window"}:
        return "action"
    if kind == "gripper_transition":
        return "gripper"
    if kind == "contact_event":
        return "contact"
    if kind in {"object_movement_event", "target_object_hypothesis"}:
        return "object_motion"
    if kind in {"goal_predicate_anchor", "goal_predicate_transition"}:
        return "goal"
    if kind == "language_phrase":
        return "language"
    if kind == "visual_grounding_mask":
        return "visual"
    if kind == "state_anchor_unit":
        return "state_anchor"
    return kind or "unknown"


def _ordered_unit_groups(units: Sequence[CausalUnit]) -> List[Tuple[str, List[CausalUnit]]]:
    grouped: Dict[str, List[CausalUnit]] = collections.OrderedDict()
    for unit in units:
        grouped.setdefault(_causal_unit_group(unit), []).append(unit)
    return [(name, members) for name, members in grouped.items()]


def _ablation_window_for_units(
    final_eval: ReplayEvaluation,
    units: Sequence[CausalUnit],
) -> Tuple[int, int]:
    starts = [int(unit.interval[0]) for unit in units]
    ends = [int(unit.interval[1]) for unit in units]
    start = min(int(final_eval.candidate.span_start or min(starts)), min(starts))
    end = max(int(final_eval.candidate.span_end or max(ends)), max(ends))
    return start, end


def _replacement_actions_for_units(
    actions: np.ndarray,
    units: Sequence[CausalUnit],
    strategy: str,
) -> Dict[int, np.ndarray]:
    replacements: Dict[int, np.ndarray] = {}
    for unit in units:
        replacements.update(_replacement_actions(actions, unit.interval, strategy))
    return replacements


def _evaluate_destructive_ablation(
    args,
    rollout: Pi05Rollout,
    final_eval: ReplayEvaluation,
    units: Sequence[CausalUnit],
    client: Optional[websocket_client_policy.WebsocketClientPolicy],
    stage_prefix: str,
) -> Tuple[ReplayEvaluation, str]:
    best_eval = None
    best_strategy = ""
    ablation_start, ablation_end = _ablation_window_for_units(final_eval, units)
    _progress(
        args,
        "destructive_ablation_start",
        stage_prefix=stage_prefix,
        unit_ids=[unit.unit_id for unit in units],
        unit_kinds=sorted({unit.kind for unit in units}),
        ablation_start=int(ablation_start),
        ablation_end=int(ablation_end),
        strategies=_causal_ablation_strategies(args),
    )
    for strategy in _causal_ablation_strategies(args):
        _incr(args, "causal_ablation_evaluations")
        _progress(
            args,
            "destructive_ablation_strategy_start",
            stage_prefix=stage_prefix,
            strategy=strategy,
            unit_ids=[unit.unit_id for unit in units],
        )
        replacements = _replacement_actions_for_units(rollout.actions, units, strategy)
        evaluation = replay_candidate(
            args,
            rollout,
            ablation_start,
            ablation_end,
            rollout.failure_signature,
            client=client,
            action_replacements=replacements,
            stage_level="%s_%s" % (stage_prefix, strategy),
            trials_override=args.causal_ablation_trials,
            early_stop_objective="causal_effect",
            ce_reference_rate=final_eval.same_failure_rate,
            ce_threshold=args.causal_effect_threshold,
        )
        if best_eval is None or evaluation.same_failure_rate < best_eval.same_failure_rate:
            best_eval = evaluation
            best_strategy = strategy
        _progress(
            args,
            "destructive_ablation_strategy_done",
            stage_prefix=stage_prefix,
            strategy=strategy,
            same_failure_rate=float(evaluation.same_failure_rate),
            executed_trials=int(evaluation.executed_trials),
            early_stop_reason=evaluation.early_stop_reason,
        )
    assert best_eval is not None
    _progress(
        args,
        "destructive_ablation_done",
        stage_prefix=stage_prefix,
        best_strategy=best_strategy,
        best_same_failure_rate=float(best_eval.same_failure_rate),
        base_same_failure_rate=float(final_eval.same_failure_rate),
        causal_effect=float(final_eval.same_failure_rate - best_eval.same_failure_rate),
    )
    return best_eval, best_strategy


def _should_run_language_repair(unit: CausalUnit, raw_policy_pass: bool) -> bool:
    return bool((not raw_policy_pass) or unit.kind in {"language_phrase", "target_object_hypothesis"})


def _should_run_visual_repair(unit: CausalUnit, raw_policy_pass: bool) -> bool:
    return bool(
        (not raw_policy_pass)
        or unit.kind in {"visual_grounding_mask", "target_object_hypothesis", "contact_event"}
    )


def _repair_scheduling_mode(args) -> str:
    mode = str(getattr(args, "repair_scheduling_mode", "pass_hunt") or "pass_hunt")
    if mode not in {"pass_hunt", "topk_complete"}:
        raise ValueError("Unsupported repair scheduling mode: %s" % mode)
    return mode


def _repair_scheduler_event(
    args,
    trace: List[Dict[str, object]],
    event: str,
    **fields: object,
) -> None:
    item = {"event": str(event), **fields}
    trace.append(item)
    _progress(args, "repair_scheduler_%s" % event, **fields)


def _unit_repair_priority(record: Dict[str, object], rollout: Pi05Rollout) -> Tuple[object, ...]:
    unit = record["unit"]
    assert isinstance(unit, CausalUnit)
    start, end = unit.interval
    kind_rank = {
        "target_object_hypothesis": 0,
        "goal_predicate_anchor": 1,
        "goal_predicate_transition": 1,
        "contact_event": 2,
        "object_movement_event": 3,
        "gripper_transition": 4,
        "stage_phase_window": 5,
        "action_chunk": 6,
        "language_phrase": 7,
        "visual_grounding_mask": 8,
        "state_anchor_unit": 9,
    }.get(str(unit.kind), 20)
    horizon = max(1, int(rollout.length) - int(start))
    length = max(1, int(end) - int(start))
    return (
        -float(record.get("causal_effect", 0.0)),
        int(kind_rank),
        int(horizon),
        int(length),
        str(unit.unit_id),
    )


def _deferred_repair_item(source: str, reason: str) -> Dict[str, object]:
    return {
        "source": source,
        "repair_pass": False,
        "skipped": True,
        "deferred": True,
        "deferred_then_confirmed": False,
        "reason": reason,
    }


def _unit_result_without_repair(
    rollout: Pi05Rollout,
    record: Dict[str, object],
    reason: str,
    *,
    necessity_pass: bool,
) -> CausalUnitResult:
    unit = record["unit"]
    assert isinstance(unit, CausalUnit)
    best_eval = record["best_eval"]
    assert isinstance(best_eval, ReplayEvaluation)
    best_strategy = str(record["best_strategy"])
    ce = float(record["causal_effect"])
    repair_evaluations: Tuple[Dict[str, object], ...] = tuple()
    repair_evidence = {
        "repair_pass": False,
        "policy_repair_pass": False,
        "demo_existence_repair_pass": False,
        "repair_skipped_reason": reason,
    }
    if necessity_pass:
        repair_evaluations = (
            _deferred_repair_item(
                "policy_replan_from_pre_state",
                reason,
            ),
            _deferred_repair_item(
                "success_or_demo_nn_repair",
                reason,
            ),
        )
        repair_evidence = {
            **repair_evidence,
            "deferred_repair_sources": [item["source"] for item in repair_evaluations],
        }
    return CausalUnitResult(
        unit=unit,
        base_same_failure_rate=float(record["base_same_failure_rate"]),
        ablated_same_failure_rate=best_eval.same_failure_rate,
        causal_effect=ce,
        is_causal_core=False,
        is_necessity_core=bool(necessity_pass),
        repair_pass=False,
        repair_evidence=repair_evidence,
        repair_evaluations=repair_evaluations,
        policy_strong_repair_pass=False,
        demo_existence_repair_pass=False,
        best_counterfactual={
            "strategy": best_strategy,
            "unit_id": unit.unit_id,
            "destructive_ablation": True,
            "description": "Recorded-suffix action replacement; not a repair continuation.",
            "evaluation": best_eval.to_dict(),
            "repair_pass": False,
            "repair_evidence": _counterfactual_repair_evidence(
                rollout.failure_signature,
                best_eval,
            ),
        },
    )


def _unit_result_with_repair(
    rollout: Pi05Rollout,
    record: Dict[str, object],
    repair_evaluations: Sequence[Dict[str, object]],
) -> CausalUnitResult:
    unit = record["unit"]
    assert isinstance(unit, CausalUnit)
    best_eval = record["best_eval"]
    assert isinstance(best_eval, ReplayEvaluation)
    best_strategy = str(record["best_strategy"])
    ce = float(record["causal_effect"])
    repair_evaluations = tuple(repair_evaluations)
    raw_policy_pass = any(
        item.get("source") == "policy_replan_from_pre_state"
        and bool(item.get("repair_pass"))
        for item in repair_evaluations
    )
    language_phrase_pass = any(
        item.get("source") == "policy_language_disambiguation_repair"
        and bool(item.get("repair_pass"))
        for item in repair_evaluations
    )
    visual_mask_pass = any(
        item.get("source") == "policy_visual_mask_repair"
        and bool(item.get("repair_pass"))
        for item in repair_evaluations
    )
    policy_pass = bool(raw_policy_pass or language_phrase_pass or visual_mask_pass)
    source_pass = any(
        item.get("source")
        in {
            "scripted_stage_expert_repair",
            "success_or_demo_nn_repair",
        }
        and bool(item.get("repair_pass"))
        for item in repair_evaluations
    )
    best_repair = next(
        (
            item
            for item in repair_evaluations
            if item.get("source")
            in {
                "policy_replan_from_pre_state",
                "policy_language_disambiguation_repair",
                "policy_visual_mask_repair",
            }
            and bool(item.get("repair_pass"))
        ),
        None,
    )
    if best_repair is None:
        best_repair = next(
            (item for item in repair_evaluations if bool(item.get("repair_pass"))),
            None,
        )
    repair_evidence = {
        "repair_pass": bool(policy_pass),
        "policy_repair_pass": bool(policy_pass),
        "raw_policy_repair_pass": bool(raw_policy_pass),
        "language_phrase_repair_pass": bool(language_phrase_pass),
        "visual_mask_repair_pass": bool(visual_mask_pass),
        "demo_existence_repair_pass": bool(source_pass),
        "scripted_expert_repair_pass": any(
            item.get("source") == "scripted_stage_expert_repair"
            and bool(item.get("repair_pass"))
            for item in repair_evaluations
        ),
        "success_or_demo_repair_pass": any(
            item.get("source") == "success_or_demo_nn_repair"
            and bool(item.get("repair_pass"))
            for item in repair_evaluations
        ),
        "both_source_repair_pass": bool(policy_pass and source_pass),
        "best_repair_source": None if best_repair is None else best_repair.get("source"),
        "best_repair_evidence": None
        if best_repair is None
        else best_repair.get("repair_evidence"),
        "policy_strong_definition": (
            "v4 ranks raw policy, phrase-level language intervention, and "
            "policy-input visual grounding mask repair separately."
        ),
    }
    return CausalUnitResult(
        unit=unit,
        base_same_failure_rate=float(record["base_same_failure_rate"]),
        ablated_same_failure_rate=best_eval.same_failure_rate,
        causal_effect=ce,
        is_causal_core=bool(policy_pass),
        is_necessity_core=True,
        repair_pass=bool(policy_pass),
        repair_evidence=repair_evidence,
        repair_evaluations=repair_evaluations,
        policy_strong_repair_pass=bool(policy_pass),
        demo_existence_repair_pass=bool(source_pass),
        raw_policy_repair_pass=bool(raw_policy_pass),
        language_phrase_repair_pass=bool(language_phrase_pass),
        visual_mask_repair_pass=bool(visual_mask_pass),
        best_counterfactual={
            "strategy": best_strategy,
            "unit_id": unit.unit_id,
            "destructive_ablation": True,
            "description": "Recorded-suffix action replacement; not a repair continuation.",
            "evaluation": best_eval.to_dict(),
            "repair_pass": False,
            "repair_evidence": _counterfactual_repair_evidence(
                rollout.failure_signature,
                best_eval,
            ),
        },
    )


def _run_repair_for_necessity_record(
    args,
    rollout: Pi05Rollout,
    record: Dict[str, object],
    all_rollouts: Sequence[Pi05Rollout],
    client: Optional[websocket_client_policy.WebsocketClientPolicy],
    policy_repair_cache: Dict[Tuple[object, ...], dict],
    source_repair_cache: Dict[Tuple[object, ...], dict],
    scheduler_trace: List[Dict[str, object]],
) -> CausalUnitResult:
    unit = record["unit"]
    assert isinstance(unit, CausalUnit)
    repair_start = int(unit.interval[0])
    repair_end = int(unit.interval[1])
    repair_max_steps = max(1, rollout.length - repair_start)
    repair_evaluations: List[Dict[str, object]] = []
    mode = _repair_scheduling_mode(args)

    _repair_scheduler_event(
        args,
        scheduler_trace,
        "unit_repair_start",
        unit_id=unit.unit_id,
        kind=unit.kind,
        repair_start=int(repair_start),
        repair_end=int(repair_end),
        repair_max_steps=int(repair_max_steps),
        mode=mode,
    )
    raw_policy_pass_now = False
    if client is not None:
        repair_key = ("policy_raw", int(repair_start), int(repair_max_steps))
        if repair_key not in policy_repair_cache:
            _progress(
                args,
                "repair_source_start",
                unit_id=unit.unit_id,
                source="policy_replan_from_pre_state",
                repair_start=int(repair_start),
                repair_end=int(repair_end),
                repair_trials=int(args.repair_replay_trials),
            )
            policy_repair = replay_candidate(
                args,
                rollout,
                repair_start,
                repair_end,
                rollout.failure_signature,
                client=client,
                policy_from_step=repair_start,
                stage_level="policy_replan_from_pre_state",
                trials_override=args.repair_replay_trials,
                early_stop_objective="repair_valid",
            )
            policy_evidence = _counterfactual_repair_evidence(
                rollout.failure_signature,
                policy_repair,
            )
            policy_repair_cache[repair_key] = {
                "source": "policy_replan_from_pre_state",
                "repair_pass": bool(policy_evidence["repair_pass"]),
                "evaluation": policy_repair.to_dict(),
                "repair_evidence": policy_evidence,
            }
            _progress(
                args,
                "repair_source_done",
                unit_id=unit.unit_id,
                source="policy_replan_from_pre_state",
                repair_pass=bool(policy_evidence["repair_pass"]),
                success=bool(policy_evidence.get("success")),
            )
        repair_evaluations.append(policy_repair_cache[repair_key])
        raw_policy_pass_now = bool(policy_repair_cache[repair_key].get("repair_pass"))

        if (
            not bool(getattr(args, "disable_rule_language_intervention", False))
            and _should_run_language_repair(unit, raw_policy_pass_now)
        ):
            language = _rule_language_intervention(rollout, unit)
            lang_key = (
                "policy_language",
                int(repair_start),
                int(repair_max_steps),
                _stable_digest(language["prompt"]),
            )
            if lang_key not in policy_repair_cache:
                _progress(
                    args,
                    "repair_source_start",
                    unit_id=unit.unit_id,
                    source="policy_language_disambiguation_repair",
                    repair_start=int(repair_start),
                    repair_end=int(repair_end),
                    repair_trials=int(args.repair_replay_trials),
                )
                language_repair = replay_candidate(
                    args,
                    rollout,
                    repair_start,
                    repair_end,
                    rollout.failure_signature,
                    client=client,
                    policy_from_step=repair_start,
                    prompt_override=language["prompt"],
                    stage_level="policy_language_disambiguation_repair",
                    trials_override=args.repair_replay_trials,
                    early_stop_objective="repair_valid",
                )
                language_evidence = _counterfactual_repair_evidence(
                    rollout.failure_signature,
                    language_repair,
                )
                policy_repair_cache[lang_key] = {
                    "source": "policy_language_disambiguation_repair",
                    "repair_pass": bool(language_evidence["repair_pass"]),
                    "evaluation": language_repair.to_dict(),
                    "repair_evidence": language_evidence,
                    "language_intervention": language,
                }
                _progress(
                    args,
                    "repair_source_done",
                    unit_id=unit.unit_id,
                    source="policy_language_disambiguation_repair",
                    repair_pass=bool(language_evidence["repair_pass"]),
                    success=bool(language_evidence.get("success")),
                )
            repair_evaluations.append(policy_repair_cache[lang_key])

        if _should_run_visual_repair(unit, raw_policy_pass_now):
            visual = _visual_policy_mask_intervention(args, rollout, unit)
            if bool(visual.get("applied_to_policy_input")):
                visual_key = (
                    "policy_visual",
                    int(repair_start),
                    int(repair_max_steps),
                    str(visual.get("unit_id")),
                    str(visual.get("mode")),
                )
                if visual_key not in policy_repair_cache:
                    _progress(
                        args,
                        "repair_source_start",
                        unit_id=unit.unit_id,
                        source="policy_visual_mask_repair",
                        repair_start=int(repair_start),
                        repair_end=int(repair_end),
                        repair_trials=int(args.repair_replay_trials),
                    )
                    visual_repair = replay_candidate(
                        args,
                        rollout,
                        repair_start,
                        repair_end,
                        rollout.failure_signature,
                        client=client,
                        policy_from_step=repair_start,
                        visual_intervention=visual,
                        stage_level="policy_visual_mask_repair",
                        trials_override=args.repair_replay_trials,
                        early_stop_objective="repair_valid",
                    )
                    visual_evidence = _counterfactual_repair_evidence(
                        rollout.failure_signature,
                        visual_repair,
                    )
                    policy_repair_cache[visual_key] = {
                        "source": "policy_visual_mask_repair",
                        "repair_pass": bool(visual_evidence["repair_pass"]),
                        "evaluation": visual_repair.to_dict(),
                        "repair_evidence": visual_evidence,
                        "visual_policy_mask_intervention": visual,
                    }
                    _progress(
                        args,
                        "repair_source_done",
                        unit_id=unit.unit_id,
                        source="policy_visual_mask_repair",
                        repair_pass=bool(visual_evidence["repair_pass"]),
                        success=bool(visual_evidence.get("success")),
                    )
                repair_evaluations.append(policy_repair_cache[visual_key])
            else:
                _progress(
                    args,
                    "repair_source_unavailable",
                    unit_id=unit.unit_id,
                    source="policy_visual_mask_repair",
                    reason=visual.get("reason"),
                )
                repair_evaluations.append(
                    {
                        "source": "policy_visual_mask_repair",
                        "repair_pass": False,
                        "unavailable": True,
                        "reason": visual.get("reason"),
                        "visual_policy_mask_intervention": visual,
                    }
                )
        else:
            repair_evaluations.append(
                {
                    "source": "policy_visual_mask_repair",
                    "repair_pass": False,
                    "skipped": True,
                    "reason": "raw_policy_repair_already_passed_and_unit_not_visual_relevant",
                }
            )
    else:
        repair_evaluations.append(
            {
                "source": "policy_replan_from_pre_state",
                "repair_pass": False,
                "unavailable": True,
                "reason": "policy_client_unavailable",
            }
        )

    policy_pass = any(
        item.get("source")
        in {
            "policy_replan_from_pre_state",
            "policy_language_disambiguation_repair",
            "policy_visual_mask_repair",
        }
        and bool(item.get("repair_pass"))
        for item in repair_evaluations
    )
    source_skip_reason = None
    if bool(getattr(args, "disable_source_repair", False)):
        source_skip_reason = "source_repair_disabled"
    elif mode == "pass_hunt":
        source_skip_reason = "source_repair_deferred_by_pass_hunt_scheduler"
    elif bool(getattr(args, "defer_source_repair", False)):
        source_skip_reason = "source_repair_deferred_until_confirmation_or_review"
    elif bool(policy_pass) and bool(getattr(args, "skip_source_repair_if_policy_pass", False)):
        source_skip_reason = "policy_repair_already_passed"
    if source_skip_reason is not None:
        repair_evaluations.append(
            _deferred_repair_item("scripted_stage_expert_repair", source_skip_reason)
        )
        repair_evaluations.append(
            _deferred_repair_item("success_or_demo_nn_repair", source_skip_reason)
        )
    else:
        scripted_expert = _scripted_expert_repair(
            args,
            rollout,
            repair_start,
            repair_max_steps,
        )
        if scripted_expert:
            repair_evaluations.append(scripted_expert)

        source_key = ("success_or_demo", int(repair_start), int(repair_max_steps))
        if source_key not in source_repair_cache:
            source_meta = _nearest_success_repair_actions(
                all_rollouts,
                rollout,
                repair_start,
                repair_max_steps,
            )
            if source_meta is None:
                source_meta = _demo_repair_actions(args, rollout, repair_start, repair_max_steps)

            if source_meta is not None and source_meta.get("actions") is not None:
                _progress(
                    args,
                    "repair_source_start",
                    unit_id=unit.unit_id,
                    source="success_or_demo_nn_repair",
                    repair_start=int(repair_start),
                    repair_end=int(repair_end),
                    repair_trials=int(args.repair_replay_trials),
                )
                source_actions = np.asarray(source_meta["actions"], dtype=np.float32)
                if source_actions.ndim == 1 and source_actions.size > 0:
                    source_actions = source_actions.reshape(1, -1)
                source_truncated = bool(source_actions.shape[0] < repair_max_steps)
                source_eval = replay_candidate(
                    args,
                    rollout,
                    repair_start,
                    repair_end,
                    rollout.failure_signature,
                    client=None,
                    external_actions=source_actions,
                    stage_level="success_or_demo_nn_repair",
                    trials_override=args.repair_replay_trials,
                    early_stop_objective="repair_valid",
                )
                source_evidence = _counterfactual_repair_evidence(
                    rollout.failure_signature,
                    source_eval,
                )
                if source_truncated and not bool(source_evidence.get("success")):
                    source_evidence = {
                        **source_evidence,
                        "repair_pass": False,
                        "truncated_action_suffix": True,
                        "repair_rejected_reason": "source_action_suffix_shorter_than_remaining_horizon",
                    }
                source_public_meta = {
                    k: v
                    for k, v in source_meta.items()
                    if k != "actions" and not isinstance(v, np.ndarray)
                }
                source_repair_cache[source_key] = {
                    "source": "success_or_demo_nn_repair",
                    "repair_pass": bool(source_evidence["repair_pass"]),
                    "evaluation": source_eval.to_dict(),
                    "repair_evidence": source_evidence,
                    "source_metadata": {
                        **source_public_meta,
                        "truncated_action_suffix": source_truncated,
                    },
                }
                _progress(
                    args,
                    "repair_source_done",
                    unit_id=unit.unit_id,
                    source="success_or_demo_nn_repair",
                    repair_pass=bool(source_evidence["repair_pass"]),
                    success=bool(source_evidence.get("success")),
                    truncated_action_suffix=bool(source_truncated),
                )
            else:
                source_repair_cache[source_key] = {
                    "source": "success_or_demo_nn_repair",
                    "repair_pass": False,
                    "unavailable": True,
                    "reason": None if source_meta is None else source_meta.get("reason"),
                    "source_metadata": source_meta,
                }
        repair_evaluations.append(source_repair_cache[source_key])

    result = _unit_result_with_repair(rollout, record, repair_evaluations)
    _repair_scheduler_event(
        args,
        scheduler_trace,
        "unit_repair_done",
        unit_id=unit.unit_id,
        policy_pass=bool(result.policy_strong_repair_pass),
        raw_policy_pass=bool(result.raw_policy_repair_pass),
        language_phrase_pass=bool(result.language_phrase_repair_pass),
        visual_mask_pass=bool(result.visual_mask_repair_pass),
        demo_existence_pass=bool(result.demo_existence_repair_pass),
    )
    return result


def validate_causal_units(
    args,
    rollout: Pi05Rollout,
    final_eval: ReplayEvaluation,
    all_rollouts: Sequence[Pi05Rollout] = (),
    client: Optional[websocket_client_policy.WebsocketClientPolicy] = None,
) -> CausalValidationResult:
    units = build_causal_units(
        final_eval.candidate,
        rollout.actions,
        rollout.snapshots,
        rollout.failure_signature,
        task_language=rollout.task_language,
        chunk_size=args.causal_chunk_size,
        max_units=args.causal_max_units,
        context_before=args.causal_context_before,
        context_after=args.causal_context_after,
    )
    _progress(
        args,
        "causal_validation_start",
        task_id=int(rollout.task_id),
        init_state_id=int(rollout.init_state_id),
        seed=None if rollout.reset_seed is None else int(rollout.reset_seed),
        final_same_failure_rate=float(final_eval.same_failure_rate),
        candidate_units=int(len(units)),
        causal_max_units=int(args.causal_max_units),
        causal_ablation_trials=int(args.causal_ablation_trials),
    )
    results: List[CausalUnitResult] = []
    policy_repair_cache: Dict[Tuple[object, ...], dict] = {}
    source_repair_cache: Dict[Tuple[object, ...], dict] = {}
    hierarchical_pruning_trace: List[Dict[str, object]] = []
    repair_scheduler_trace: List[Dict[str, object]] = []
    ablation_records: List[Dict[str, object]] = []
    necessity_records: List[Dict[str, object]] = []
    units_to_validate: List[CausalUnit] = []
    repair_mode = _repair_scheduling_mode(args)
    _repair_scheduler_event(
        args,
        repair_scheduler_trace,
        "start",
        mode=repair_mode,
        repair_replay_trials=int(args.repair_replay_trials),
    )
    if bool(getattr(args, "disable_hierarchical_causal_pruning", False)):
        units_to_validate = list(units)
    else:
        for group_name, group_units in _ordered_unit_groups(units):
            _progress(
                args,
                "causal_group_start",
                group=group_name,
                unit_count=int(len(group_units)),
                unit_ids=[unit.unit_id for unit in group_units],
            )
            if len(group_units) <= 1:
                units_to_validate.extend(group_units)
                hierarchical_pruning_trace.append(
                    {
                        "group": group_name,
                        "status": "singleton_no_group_test",
                        "unit_ids": [unit.unit_id for unit in group_units],
                    }
                )
                _progress(
                    args,
                    "causal_group_done",
                    group=group_name,
                    status="singleton_no_group_test",
                    unit_count=int(len(group_units)),
                )
                continue
            _incr(args, "hierarchical_group_evaluations")
            group_eval, group_strategy = _evaluate_destructive_ablation(
                args,
                rollout,
                final_eval,
                group_units,
                client,
                stage_prefix="causal_group_ablation_%s" % group_name,
            )
            group_ce = float(final_eval.same_failure_rate - group_eval.same_failure_rate)
            trace_item = {
                "group": group_name,
                "status": "expanded" if group_ce >= args.causal_effect_threshold else "pruned",
                "unit_ids": [unit.unit_id for unit in group_units],
                "unit_kinds": sorted({unit.kind for unit in group_units}),
                "best_strategy": group_strategy,
                "base_same_failure_rate": float(final_eval.same_failure_rate),
                "ablated_same_failure_rate": float(group_eval.same_failure_rate),
                "causal_effect": group_ce,
                "evaluation": group_eval.to_dict(),
            }
            hierarchical_pruning_trace.append(trace_item)
            _progress(
                args,
                "causal_group_done",
                group=group_name,
                status=str(trace_item["status"]),
                causal_effect=float(group_ce),
                ablated_same_failure_rate=float(group_eval.same_failure_rate),
                executed_trials=int(group_eval.executed_trials),
                early_stop_reason=group_eval.early_stop_reason,
            )
            if group_ce >= args.causal_effect_threshold:
                units_to_validate.extend(group_units)
            else:
                _incr(args, "hierarchical_pruned_units", len(group_units))
    for unit in units_to_validate:
        _progress(
            args,
            "causal_unit_start",
            unit_id=unit.unit_id,
            kind=unit.kind,
            interval=list(unit.interval),
        )
        best_eval, best_strategy = _evaluate_destructive_ablation(
            args,
            rollout,
            final_eval,
            [unit],
            client,
            stage_prefix="causal_ablation_%s" % unit.unit_id,
        )
        ce = float(final_eval.same_failure_rate - best_eval.same_failure_rate)
        necessity_pass = bool(ce >= args.causal_effect_threshold)
        _progress(
            args,
            "causal_unit_ablation_done",
            unit_id=unit.unit_id,
            kind=unit.kind,
            best_strategy=best_strategy,
            causal_effect=float(ce),
            necessity_pass=bool(necessity_pass),
            ablated_same_failure_rate=float(best_eval.same_failure_rate),
            executed_trials=int(best_eval.executed_trials),
            early_stop_reason=best_eval.early_stop_reason,
        )
        record = {
            "unit": unit,
            "best_eval": best_eval,
            "best_strategy": best_strategy,
            "causal_effect": ce,
            "base_same_failure_rate": final_eval.same_failure_rate,
            "necessity_pass": necessity_pass,
        }
        ablation_records.append(record)
        if not necessity_pass:
            results.append(
                _unit_result_without_repair(
                    rollout,
                    record,
                    "unit_not_same_failure_necessity_core",
                    necessity_pass=False,
                )
            )
            _progress(
                args,
                "causal_unit_done",
                unit_id=unit.unit_id,
                kind=unit.kind,
                necessity_pass=False,
                repair_pass=False,
                skip_reason="unit_not_same_failure_necessity_core",
            )
            continue
        necessity_records.append(record)

    ordered_necessity_records = sorted(
        necessity_records,
        key=lambda item: _unit_repair_priority(item, rollout),
    )
    _repair_scheduler_event(
        args,
        repair_scheduler_trace,
        "necessity_units_ranked",
        mode=repair_mode,
        unit_ids=[record["unit"].unit_id for record in ordered_necessity_records],
    )
    policy_pass_found = False
    for record in ordered_necessity_records:
        unit = record["unit"]
        assert isinstance(unit, CausalUnit)
        if repair_mode == "pass_hunt" and policy_pass_found:
            deferred_result = _unit_result_without_repair(
                rollout,
                record,
                "policy_repair_deferred_after_first_policy_pass",
                necessity_pass=True,
            )
            results.append(deferred_result)
            _repair_scheduler_event(
                args,
                repair_scheduler_trace,
                "unit_repair_deferred",
                unit_id=unit.unit_id,
                reason="policy_repair_deferred_after_first_policy_pass",
            )
            _progress(
                args,
                "causal_unit_done",
                unit_id=unit.unit_id,
                kind=unit.kind,
                necessity_pass=True,
                repair_pass=False,
                skip_reason="policy_repair_deferred_after_first_policy_pass",
            )
            continue
        result = _run_repair_for_necessity_record(
            args,
            rollout,
            record,
            all_rollouts,
            client,
            policy_repair_cache,
            source_repair_cache,
            repair_scheduler_trace,
        )
        results.append(result)
        _progress(
            args,
            "causal_unit_done",
            unit_id=unit.unit_id,
            kind=unit.kind,
            necessity_pass=True,
            repair_pass=bool(result.repair_pass),
            policy_pass=bool(result.policy_strong_repair_pass),
            demo_existence_pass=bool(result.demo_existence_repair_pass),
            raw_policy_pass=bool(result.raw_policy_repair_pass),
            language_phrase_pass=bool(result.language_phrase_repair_pass),
            visual_mask_pass=bool(result.visual_mask_repair_pass),
        )
        if result.policy_strong_repair_pass:
            policy_pass_found = True
            _repair_scheduler_event(
                args,
                repair_scheduler_trace,
                "policy_pass_found",
                unit_id=unit.unit_id,
                mode=repair_mode,
            )
            if bool(getattr(args, "stop_after_first_repair_valid_core", False)):
                _repair_scheduler_event(
                    args,
                    repair_scheduler_trace,
                    "stop_after_first_repair_valid_core",
                    unit_id=unit.unit_id,
                )
    validation = make_causal_validation_result(
        final_eval.same_failure_rate,
        results,
        ce_threshold=args.causal_effect_threshold,
        hierarchical_pruning_trace=tuple(hierarchical_pruning_trace),
        repair_scheduler_trace=tuple(repair_scheduler_trace),
    )
    _progress(
        args,
        "causal_validation_done",
        same_failure_necessity_pass=bool(validation.same_failure_necessity_pass),
        repair_valid_causal_pass=bool(validation.repair_valid_causal_pass),
        policy_strong_repair_valid_pass=bool(validation.policy_strong_repair_valid_pass),
        demo_existence_repair_pass=bool(validation.demo_existence_repair_pass),
        unit_results=int(len(validation.unit_results)),
        causal_core_units=int(len(validation.causal_core_units)),
    )
    return validation


def minimize_event_slice(
    args,
    rollout: Pi05Rollout,
    event: Tuple[int, int, float],
    client: Optional[websocket_client_policy.WebsocketClientPolicy] = None,
) -> Tuple[ReplayEvaluation, List[dict]]:
    start, end, _delta = event
    trace: List[dict] = []

    def accepts(s: int, e: int, stage: str) -> bool:
        ev = replay_candidate(
            args,
            rollout,
            s,
            e,
            rollout.failure_signature,
            client=client,
            trials_override=args.search_replay_trials,
        )
        trace.append({"stage": stage, **ev.to_dict()})
        return ev.same_failure

    full_eval = replay_candidate(
        args,
        rollout,
        start,
        end,
        rollout.failure_signature,
        client=client,
        trials_override=args.search_replay_trials,
    )
    trace.append({"stage": "event_window", **full_eval.to_dict()})
    if not full_eval.same_failure:
        return full_eval, trace

    step = max(1, (end - start) // 2)
    while step >= 1 and end - start > 1:
        progressed = False
        while end - start - step >= 1 and accepts(start + step, end, "drop_prefix"):
            start += step
            progressed = True
        while end - start - step >= 1 and accepts(start, end - step, "drop_suffix"):
            end -= step
            progressed = True
        if not progressed:
            step //= 2

    final_eval = replay_candidate(
        args,
        rollout,
        start,
        end,
        rollout.failure_signature,
        client=client,
        trials_override=args.confirm_replay_trials,
        stage_level="minimal_pi05_natural_slice",
    )
    trace.append({"stage": "minimal_pi05_natural_slice", **final_eval.to_dict()})
    return final_eval, trace


def _estimate_search_suffix_steps(
    selected: Optional[Pi05Rollout],
    trace: Sequence[dict],
) -> int:
    if selected is None:
        return 0
    total = 0
    for item in trace:
        candidate = item.get("candidate") or {}
        span = candidate.get("span")
        if not span:
            continue
        trials = int(item.get("trials", 1))
        total += trials * max(0, selected.length - int(span[0]))
    return int(total)


def _estimate_ablation_suffix_steps(
    selected: Optional[Pi05Rollout],
    causal_validation: Optional[CausalValidationResult],
) -> int:
    if selected is None or causal_validation is None:
        return 0
    total = 0
    for result in causal_validation.unit_results:
        evaluation = result.best_counterfactual.get("evaluation", {})
        candidate = evaluation.get("candidate", {})
        span = candidate.get("span")
        if not span:
            continue
        trials = int(evaluation.get("trials", 1))
        total += trials * max(0, selected.length - int(span[0]))
    return int(total)


def build_cost_summary(
    selected: Optional[Pi05Rollout],
    event: Optional[Tuple[int, int, float]],
    final_eval: Optional[ReplayEvaluation],
    trace: Sequence[dict],
    causal_validation: Optional[CausalValidationResult],
    runtime_profile: Optional[dict],
    causal_validation_skipped_reason: Optional[str],
) -> dict:
    counters = (runtime_profile or {}).get("counters", {})
    durations = (runtime_profile or {}).get("durations_seconds", {})
    search_suffix_steps = _estimate_search_suffix_steps(selected, trace)
    ablation_suffix_steps = _estimate_ablation_suffix_steps(selected, causal_validation)
    return {
        "total_wall_seconds": (runtime_profile or {}).get("total_wall_seconds"),
        "durations_seconds": durations,
        "policy_queries": int(counters.get("policy_queries", 0)),
        "env_resets": int(counters.get("env_resets", 0)),
        "rollout_steps": int(counters.get("rollout_steps", 0)),
        "replay_requests": int(counters.get("replay_requests", 0)),
        "replay_evaluations": int(counters.get("replay_evaluations", 0)),
        "replay_trials": int(counters.get("replay_trials_executed", 0)),
        "replay_trials_planned": int(counters.get("replay_trials_planned", 0)),
        "replay_trials_executed": int(counters.get("replay_trials_executed", 0)),
        "replay_cache_lookups": int(counters.get("replay_cache_lookups", 0)),
        "replay_cache_hits": int(counters.get("replay_cache_hits", 0)),
        "replay_cache_misses": int(counters.get("replay_cache_misses", 0)),
        "replay_cache_size": int(counters.get("replay_cache_size", 0)),
        "sequential_trial_early_stops": int(counters.get("sequential_trial_early_stops", 0)),
        "hierarchical_group_evaluations": int(counters.get("hierarchical_group_evaluations", 0)),
        "hierarchical_pruned_units": int(counters.get("hierarchical_pruned_units", 0)),
        "causal_ablation_evaluations": int(counters.get("causal_ablation_evaluations", 0)),
        "simulator_suffix_steps_measured": int(counters.get("simulator_suffix_steps", 0)),
        "simulator_suffix_steps_estimated_from_search": int(search_suffix_steps),
        "simulator_suffix_steps_estimated_from_ablation_best": int(ablation_suffix_steps),
        "natural_failure_found": bool(selected is not None and not selected.success),
        "same_failure_pass": bool(final_eval is not None and final_eval.same_failure),
        "same_failure_necessity_pass": bool(
            causal_validation is not None
            and causal_validation.same_failure_necessity_pass
        ),
        "repair_valid_causal_pass": bool(
            causal_validation is not None and causal_validation.passed
        ),
        "policy_strong_repair_valid_pass": bool(
            causal_validation is not None
            and causal_validation.policy_strong_repair_valid_pass
        ),
        "demo_existence_repair_pass": bool(
            causal_validation is not None
            and causal_validation.demo_existence_repair_pass
        ),
        "causal_validation_pass": bool(
            causal_validation is not None and causal_validation.passed
        ),
        "causal_validation_skipped_reason": causal_validation_skipped_reason,
        "trajectory_reduction_ratio": None
        if selected is None or final_eval is None or final_eval.steps <= 0
        else float(selected.length / max(1, final_eval.steps)),
        "event_reduction_ratio": None
        if event is None or final_eval is None or final_eval.steps <= 0
        else float((event[1] - event[0]) / max(1, final_eval.steps)),
    }


def _causal_validation_items(
    causal_validation: Optional[CausalValidationResult],
    key: str,
) -> List[dict]:
    if causal_validation is None:
        return []
    items: List[dict] = []
    seen = set()
    for result in causal_validation.unit_results:
        for evaluation in result.repair_evaluations:
            value = evaluation.get(key)
            if not isinstance(value, dict):
                continue
            marker = json.dumps(value, sort_keys=True, ensure_ascii=False)
            if marker in seen:
                continue
            seen.add(marker)
            items.append(value)
    return items


def build_report(
    args,
    rollouts: List[Pi05Rollout],
    selected: Optional[Pi05Rollout],
    event: Optional[Tuple[int, int, float]],
    final_eval: Optional[ReplayEvaluation],
    trace: List[dict],
    causal_validation: Optional[CausalValidationResult],
    causal_validation_skipped_reason: Optional[str],
    runtime_profile: Optional[dict],
) -> dict:
    found = selected is not None and event is not None and final_eval is not None
    invalid_rollouts = [r for r in rollouts if not r.initial_state_valid]
    invalid_initial_state_only = bool(rollouts) and len(invalid_rollouts) == len(rollouts)
    passed = bool(
        found
        and (not selected.success)
        and final_eval.same_failure
        and final_eval.steps < (event[1] - event[0])
        and causal_validation is not None
        and causal_validation.policy_strong_repair_valid_pass
        and selected.semantic_quality != "degraded"
    )
    raw_policy_pass = bool(
        found
        and causal_validation is not None
        and causal_validation.raw_policy_repair_valid_pass
        and selected.semantic_quality != "degraded"
    )
    language_phrase_pass = bool(
        found
        and causal_validation is not None
        and causal_validation.language_phrase_repair_valid_pass
        and selected.semantic_quality != "degraded"
    )
    visual_mask_pass = bool(
        found
        and causal_validation is not None
        and causal_validation.visual_mask_repair_valid_pass
        and selected.semantic_quality != "degraded"
    )
    demo_existence_pass = bool(
        found
        and causal_validation is not None
        and causal_validation.demo_existence_repair_pass
        and selected.semantic_quality != "degraded"
    )
    global_units = []
    k_minimal_sets = []
    if causal_validation is not None:
        global_units = list(build_global_multimodal_units(causal_validation.unit_results))
        k_minimal_sets = list(causal_validation.k_minimal_causal_sets)
    same_failure_necessity_pass = bool(
        found
        and causal_validation is not None
        and causal_validation.same_failure_necessity_pass
        and selected.semantic_quality != "degraded"
    )
    return {
        "schema_version": CAUSAL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "pi0fast/pi05 natural LIBERO rollout via websocket policy server",
        "policy_server": {
            "host": args.policy_host,
            "port": int(args.policy_port),
            "config": args.policy_config,
            "checkpoint": args.policy_checkpoint,
        },
        "search_config": {
            "task_suite_name": args.task_suite_name,
            "task_ids": args.task_ids,
            "init_state_ids": args.init_state_ids,
            "max_steps": int(_max_steps_for_suite(args, args.task_suite_name)),
            "event_window": int(args.event_window),
            "replan_steps": int(args.replan_steps),
            "continuation": args.continuation,
            "replay_trials": int(args.replay_trials),
            "search_replay_trials": int(args.search_replay_trials),
            "confirm_replay_trials": int(args.confirm_replay_trials),
            "repair_replay_trials": int(args.repair_replay_trials),
            "same_failure_threshold": float(args.same_failure_threshold),
            "accept_same_failure_rate": float(args.accept_same_failure_rate),
            "causal_context_before": int(args.causal_context_before),
            "causal_context_after": int(args.causal_context_after),
            "causal_max_units": int(args.causal_max_units),
            "causal_ablation_trials": int(args.causal_ablation_trials),
            "replay_cache_enabled": not bool(args.disable_replay_cache),
            "sequential_trial_pruning_enabled": bool(args.enable_sequential_trial_pruning),
            "hierarchical_causal_pruning_enabled": not bool(args.disable_hierarchical_causal_pruning),
            "defer_source_repair": bool(args.defer_source_repair),
            "skip_source_repair_if_policy_pass": bool(args.skip_source_repair_if_policy_pass),
            "stop_after_first_repair_valid_core": bool(args.stop_after_first_repair_valid_core),
            "repair_scheduling_mode": str(args.repair_scheduling_mode),
            "disable_rule_language_intervention": bool(args.disable_rule_language_intervention),
            "enable_visual_policy_mask": bool(args.enable_visual_policy_mask),
            "causal_semantic_version": "global-multimodal-v4",
            "top_k_minimal_sets": 5,
            "language_minimization": "phrase_level_bddl_delta_debug",
            "visual_intervention": "policy_input_target_highlight_or_distractor_mask",
            "contact_evidence_priority": "mujoco_contact_then_degraded_proximity",
        },
        "video_config": {
            "record_video": bool(args.record_video or args.video_dir is not None),
            "video_dir": None if args.video_dir is None else str(args.video_dir),
            "video_camera": args.video_camera,
            "video_fps": int(args.video_fps),
            "video_every_n": int(args.video_every_n),
        },
        "custom_task": (
            custom_tasks.task_metadata(args.task_suite_name)
            if custom_tasks.is_custom_suite(args.task_suite_name)
            else None
        ),
        "rollout_summaries": [
            {
                "task_id": r.task_id,
                "init_state_id": r.init_state_id,
                "task_language": r.task_language,
                "target_key": r.target_key,
                "target_key_trace_sample": r.target_key_trace[:20],
                "length": r.length,
                "success": r.success,
                "done_step": r.done_step,
                "semantic_quality": r.semantic_quality,
                "failure_signature": r.failure_signature.to_dict(),
                "initial_distance": r.distance_trace[0],
                "final_distance": r.distance_trace[-1],
                "video_path": r.video_path,
                "video_frames": int(r.video_frames),
                "initial_state_quality": r.initial_state_quality,
                "initial_state_attempt": int(r.initial_state_attempt),
                "reset_seed": None if r.reset_seed is None else int(r.reset_seed),
                "stage_oracle_trace": (
                    custom_tasks.stage_trace_from_snapshots(
                        r.task_suite_name, r.snapshots, r.task_id
                    )
                    if custom_tasks.is_custom_suite(r.task_suite_name)
                    else None
                ),
                "stage_failure_summary": custom_tasks.stage_summary_from_snapshots(
                    r.task_suite_name, r.snapshots, r.task_id
                ),
                "max_window_event": find_failure_event(
                    r, args.event_window, args.min_distance_delta
                ),
            }
            for r in rollouts
        ],
        "selected_failed_rollout": None
        if selected is None
        else {
            "task_id": selected.task_id,
            "init_state_id": selected.init_state_id,
            "task_language": selected.task_language,
            "target_key": selected.target_key,
            "target_key_trace_sample": selected.target_key_trace[:20],
            "length": selected.length,
            "success": selected.success,
            "done_step": selected.done_step,
            "semantic_quality": selected.semantic_quality,
            "initial_distance": selected.distance_trace[0],
            "final_distance": selected.distance_trace[-1],
            "video_path": selected.video_path,
            "video_frames": int(selected.video_frames),
            "rollout_archive_path": selected.rollout_archive_path,
            "rollout_archive_schema": (
                "pi05-rollout-archive-v1"
                if selected.rollout_archive_path is not None
                else None
            ),
            "initial_state_quality": selected.initial_state_quality,
            "initial_state_attempt": int(selected.initial_state_attempt),
            "reset_seed": None if selected.reset_seed is None else int(selected.reset_seed),
            "stage_oracle_trace": (
                custom_tasks.stage_trace_from_snapshots(
                    selected.task_suite_name, selected.snapshots, selected.task_id
                )
                if custom_tasks.is_custom_suite(selected.task_suite_name)
                else None
            ),
            "stage_failure_summary": custom_tasks.stage_summary_from_snapshots(
                selected.task_suite_name, selected.snapshots, selected.task_id
            ),
        },
        "failure_predicate": {
            "type": "causal-v4 global multimodal bounded-minimal causal set validation",
            "acceptance": "same_failure_rate >= %.2f and semantic_match_score >= %.2f"
            % (args.accept_same_failure_rate, args.same_failure_threshold),
            "causal_v4_semantics": {
                "minimal_same_failure_slice": (
                    "Shortest replay slice found so far that still reproduces the same semantic failure."
                ),
                "causal_core_units": (
                    "Action/contact/state/language/visual units validated by destructive and repair interventions."
                ),
                "repair_replay_context": (
                    "Repair replay can start before the minimal slice; this context is not itself claimed as the minimal failure cause."
                ),
                "same_failure_necessity_pass": (
                    "Destructive ablation reduces same-failure reproduction; useful for risk detection."
                ),
                "repair_source_ranking": [
                    "policy_raw",
                    "policy_language_phrase",
                    "policy_visual_mask",
                    "demo_existence",
                ],
                "demo_existence_repair_pass": (
                    "Scripted, success-rollout, or demo nearest-neighbor actions show a repair exists; this is auxiliary, not policy-strong."
                ),
                "destructive_ablation_note": (
                    "hold/adjacent/gripper replacements followed by recorded suffix are not counted as repairs."
                ),
            },
            "failure_taxonomy": [
                "unsatisfied_goal_predicates_at_timeout",
                "wrong_object",
                "grasp_miss_no_transport",
                "premature_release_or_slip",
                "wrong_placement",
                "stagnation_timeout",
                "unsafe_contact",
                "order_violation",
            ],
        },
        "failure_event": None
        if event is None
        else {"window": [event[0], event[1]], "semantic_confidence": event[2]},
        "original_failure_signature": None
        if selected is None
        else selected.failure_signature.to_dict(),
        "goal_predicate_trace": None
        if selected is None
        else selected.failure_signature.evidence.get("goal_trace", {}),
        "earliest_failed_stage": None
        if selected is None
        else (
            custom_tasks.stage_summary_from_snapshots(
                selected.task_suite_name, selected.snapshots, selected.task_id
            )
            or {}
        ).get("earliest_failed_stage"),
        "stage_progress_count": None
        if selected is None
        else (
            custom_tasks.stage_summary_from_snapshots(
                selected.task_suite_name, selected.snapshots, selected.task_id
            )
            or {}
        ).get("stage_progress_count"),
        "stage_first_completed_step": None
        if selected is None
        else (
            custom_tasks.stage_summary_from_snapshots(
                selected.task_suite_name, selected.snapshots, selected.task_id
            )
            or {}
        ).get("stage_first_completed_step"),
        "minimal_same_failure_slice": None
        if final_eval is None
        else {
            **final_eval.candidate.to_dict(),
            "semantics": "minimal_same_failure_slice",
            "same_failure_rate": float(final_eval.same_failure_rate),
        },
        "causal_failure_slice": None if final_eval is None else final_eval.candidate.to_dict(),
        "minimal_replay_context": None
        if final_eval is None
        else {
            "pre_state": "simulator state before slice start",
            "candidate_actions": final_eval.candidate.to_dict(),
            "continuation": (
                "recorded suffix actions from original failed rollout"
                if args.continuation == "recorded"
                else "policy re-query suffix after the candidate window"
            ),
        },
        "repair_replay_context": None
        if final_eval is None
        else {
            "semantics": "repair_replay_context_may_be_larger_than_minimal_slice",
            "minimal_slice": final_eval.candidate.to_dict(),
            "preferred_context_interval": (
                None
                if not k_minimal_sets
                else (k_minimal_sets[0].get("units") or [{}])[0].get("interval")
            ),
            "pre_state": "simulator state before preferred context start",
            "continuation": "policy replan or source repair actions from context start",
        },
        "slice_training_features": None
        if selected is None or final_eval is None
        else _slice_training_features(selected, final_eval.candidate),
        "risk_training_windows": build_risk_training_windows(
            args,
            rollouts,
            selected,
            event,
            final_eval,
            causal_validation,
        ),
        "reproduction_statistics": None if final_eval is None else final_eval.to_dict(),
        "same_failure_evidence": None
        if final_eval is None
        else final_eval.same_failure_evidence.to_dict(),
        "causal_validation": None
        if causal_validation is None
        else causal_validation.to_dict(),
        "causal_validation_skipped_reason": causal_validation_skipped_reason,
        "global_multimodal_units": global_units,
        "multimodal_candidate_units": global_units,
        "k_minimal_causal_sets": k_minimal_sets,
        "hierarchical_pruning_trace": []
        if causal_validation is None
        else list(causal_validation.hierarchical_pruning_trace),
        "repair_scheduler_trace": []
        if causal_validation is None
        else list(causal_validation.repair_scheduler_trace),
        "deferred_repair_sources": []
        if causal_validation is None
        else list(causal_validation.deferred_repair_sources),
        "stage_object_search_trace": {
            "failure_anchor": None
            if selected is None
            else list(selected.failure_signature.anchor_interval()),
            "target_key_trace_sample": [] if selected is None else selected.target_key_trace[:40],
            "unit_kinds": []
            if causal_validation is None
            else sorted({r.unit.kind for r in causal_validation.unit_results}),
        },
        "language_interventions": _causal_validation_items(
            causal_validation,
            "language_intervention",
        ),
        "visual_policy_mask_interventions": _causal_validation_items(
            causal_validation,
            "visual_policy_mask_intervention",
        ),
        "contact_event_trace": []
        if selected is None
        else [
            {
                "t": int(s.t),
                "contacts": list(s.contacts),
                "contact_records": [
                    record for record in getattr(s, "contact_records", ())
                ],
            }
            for s in selected.snapshots
            if getattr(s, "contacts", ())
        ],
        "state_anchor_units": []
        if causal_validation is None
        else [
            r.unit.to_dict()
            for r in causal_validation.unit_results
            if r.unit.kind == "state_anchor_unit"
        ],
        "same_failure_necessity_pass": bool(same_failure_necessity_pass),
        "repair_valid_causal_pass": bool(passed),
        "any_policy_repair_valid_pass": bool(passed),
        "policy_raw_repair_valid_pass": bool(raw_policy_pass),
        "policy_language_phrase_repair_valid_pass": bool(language_phrase_pass),
        "policy_visual_mask_repair_valid_pass": bool(visual_mask_pass),
        "policy_strong_repair_valid_pass": bool(passed),
        "demo_existence_repair_pass": bool(demo_existence_pass),
        "repair_source_ranking": {
            "policy_raw": bool(raw_policy_pass),
            "policy_language_phrase": bool(language_phrase_pass),
            "policy_visual_mask": bool(visual_mask_pass),
            "demo_existence": bool(demo_existence_pass),
        },
        "k5_confirmation_status": (
            "confirmed_k5"
            if int(args.confirm_replay_trials) >= 5 and passed
            else "candidate_k%d" % int(args.confirm_replay_trials)
            if found and final_eval is not None and final_eval.same_failure
            else "not_applicable"
        ),
        "full_success_repair_pass": bool(
            passed
            and causal_validation is not None
            and causal_validation.full_success_policy_repair_pass
        ),
        "full_success_policy_repair_pass": bool(
            passed
            and causal_validation is not None
            and causal_validation.full_success_policy_repair_pass
        ),
        "full_success_demo_repair_pass": bool(
            causal_validation is not None
            and causal_validation.full_success_demo_repair_pass
        ),
        "initial_state_quality_summary": {
            "num_rollouts": int(len(rollouts)),
            "valid_rollouts": int(sum(1 for r in rollouts if r.initial_state_valid)),
            "invalid_rollouts": int(len(invalid_rollouts)),
            "invalid_rollout_ids": [
                {
                    "task_id": int(r.task_id),
                    "init_state_id": int(r.init_state_id),
                    "reasons": list((r.initial_state_quality or {}).get("reasons") or []),
                }
                for r in invalid_rollouts
            ],
        },
        "necessity_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.necessity_core_units],
        "causal_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.causal_core_units],
        "repair_valid_causal_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.policy_strong_core_units],
        "policy_strong_causal_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.policy_strong_core_units],
        "raw_policy_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.raw_policy_core_units],
        "language_phrase_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.language_phrase_core_units],
        "visual_mask_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.visual_mask_core_units],
        "demo_existence_core_units": []
        if causal_validation is None
        else [r.to_dict() for r in causal_validation.demo_existence_core_units],
        "destructive_ablation_variants": []
        if causal_validation is None
        else list(causal_validation.destructive_ablation_variants),
        "repair_pass_variants": []
        if causal_validation is None
        else list(causal_validation.repair_pass_variants),
        "counterfactual_pass_variants": []
        if causal_validation is None
        else list(causal_validation.counterfactual_pass_variants),
        "metrics": None
        if not found
        else {
            "event_reduction_ratio": float((event[1] - event[0]) / max(1, final_eval.steps)),
            "trajectory_reduction_ratio": float(selected.length / max(1, final_eval.steps)),
            "replay_evaluations": len(trace),
            "same_failure_rate": float(final_eval.same_failure_rate),
            "failure_rate": float(final_eval.failure_rate),
        },
        "search_trace": trace,
        "runtime_profile": runtime_profile,
        "cost_summary": build_cost_summary(
            selected,
            event,
            final_eval,
            trace,
            causal_validation,
            runtime_profile,
            causal_validation_skipped_reason,
        ),
        "feasibility": {
            "pi05_natural_pass": bool(passed),
            "same_failure_necessity_pass": bool(same_failure_necessity_pass),
            "repair_valid_causal_pass": bool(passed),
            "policy_strong_repair_valid_pass": bool(passed),
            "any_policy_repair_valid_pass": bool(passed),
            "policy_raw_repair_valid_pass": bool(raw_policy_pass),
            "policy_language_phrase_repair_valid_pass": bool(language_phrase_pass),
            "policy_visual_mask_repair_valid_pass": bool(visual_mask_pass),
            "demo_existence_repair_pass": bool(demo_existence_pass),
            "full_success_repair_pass": bool(
                passed
                and causal_validation is not None
                and causal_validation.full_success_policy_repair_pass
            ),
            "verdict": (
                "pi05_natural_feasible"
                if passed
                else "invalid_initial_state"
                if invalid_initial_state_only
                else "pi05_natural_not_validated"
            ),
            "interpretation": (
                "A natural policy failed rollout produced a causal-v4 repair-valid global multimodal causal set."
                if passed
                else "The rollout was rejected before policy evaluation because the initial state was not suitable for causal slicing or training."
                if invalid_initial_state_only
                else "No suitable pi05 natural failure slice passed repair-valid causal intervention validation; same-failure necessity may still be present."
            ),
        },
        "limitations": [
            "Causality is simulator-intervention evidence, not real-robot causal proof.",
            "Contact evidence is optional; BDDL predicates and state traces are the hard criteria.",
        ],
    }


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and minimize a natural pi05_libero failure rollout in LIBERO."
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument(
        "--policy-checkpoint",
        default="/root/autodl-tmp/research/VLA_SKILL/model/pi05_libero",
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default="0,1,2")
    parser.add_argument("--init-state-ids", default="0,1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument(
        "--initial-state-max-attempts",
        type=int,
        default=8,
        help="For custom suites, retry deterministic reset seeds until the initial state passes quality checks.",
    )
    parser.add_argument(
        "--disable-initial-state-quality-filter",
        action="store_true",
        help="Disable custom-suite initial-state rejection; useful only for debugging bad scenes.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=24)
    parser.add_argument("--min-distance-delta", type=float, default=0.03)
    parser.add_argument("--continuation", choices=("recorded", "policy"), default="recorded")
    parser.add_argument("--replay-trials", type=int, default=5)
    parser.add_argument(
        "--search-replay-trials",
        type=int,
        default=1,
        help="Trials per candidate during greedy minimization; keep low for cost.",
    )
    parser.add_argument(
        "--confirm-replay-trials",
        type=int,
        default=None,
        help="Trials for final same-failure confirmation and causal ablations. Defaults to --replay-trials.",
    )
    parser.add_argument(
        "--repair-replay-trials",
        type=int,
        default=1,
        help="Trials for policy/demo repair continuations. Same-failure and CE confirmation still use --confirm-replay-trials.",
    )
    parser.add_argument("--same-failure-threshold", type=float, default=0.75)
    parser.add_argument("--accept-same-failure-rate", type=float, default=0.80)
    parser.add_argument("--causal-effect-threshold", type=float, default=0.30)
    parser.add_argument("--causal-chunk-size", type=int, default=5)
    parser.add_argument(
        "--causal-ablation-trials",
        type=int,
        default=None,
        help="Trials for destructive ablation CE. Defaults to --confirm-replay-trials.",
    )
    parser.add_argument(
        "--causal-ablation-strategies",
        default="hold,adjacent,gripper_correction",
        help=(
            "Comma-separated destructive ablation strategies. Use 'hold' for "
            "fast candidate search; use all strategies for stronger confirmation."
        ),
    )
    parser.add_argument(
        "--causal-context-before",
        type=int,
        default=36,
        help="Include this many steps before the minimal slice/failure anchor when building causal units.",
    )
    parser.add_argument(
        "--causal-context-after",
        type=int,
        default=8,
        help="Include this many steps after the minimal slice/failure anchor when building causal units.",
    )
    parser.add_argument(
        "--causal-max-units",
        type=int,
        default=18,
        help="Maximum causal units to ablate/repair-test after context expansion.",
    )
    parser.add_argument(
        "--disable-replay-cache",
        action="store_true",
        help="Disable exact within-case replay evaluation cache.",
    )
    parser.add_argument(
        "--replay-evaluation-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock timeout for a single replay evaluation. "
            "0 disables the timeout; useful for profiling stuck K=5 cases."
        ),
    )
    parser.add_argument(
        "--verbose-replay-progress",
        action="store_true",
        help="Print JSONL progress events for replay, ablation, and repair stages.",
    )
    parser.add_argument(
        "--progress-log-path",
        type=Path,
        default=None,
        help="Optional JSONL path for replay/causal validation progress events.",
    )
    parser.add_argument(
        "--disable-sequential-trial-pruning",
        dest="enable_sequential_trial_pruning",
        action="store_false",
        help="Run every planned same-failure trial instead of exact threshold early stopping.",
    )
    parser.set_defaults(enable_sequential_trial_pruning=True)
    parser.add_argument(
        "--disable-hierarchical-causal-pruning",
        action="store_true",
        help="Validate every generated causal unit without conservative group pruning.",
    )
    parser.add_argument("--demo-dataset-root", type=Path, default=DEFAULT_DEMO_DATASET_ROOT)
    parser.add_argument("--demo-repair-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--scripted-expert-repair-max-steps",
        type=int,
        default=180,
        help="Maximum online primitive steps for a curriculum-v2 scripted repair attempt.",
    )
    parser.add_argument(
        "--expert-repair-max-actions-to-store",
        type=int,
        default=128,
        help="Store at most this many scripted repair actions in the JSON report.",
    )
    parser.add_argument(
        "--skip-source-repair-if-policy-pass",
        action="store_true",
        help=(
            "When policy replan already gives a repair-valid variant for a unit, "
            "skip scripted/demo repair for that unit. This keeps repair-valid "
            "semantics strict while avoiding redundant expensive continuations."
        ),
    )
    parser.add_argument(
        "--disable-source-repair",
        action="store_true",
        help=(
            "Skip scripted/demo/success-NN repair attempts. This preserves policy "
            "repair evidence and speeds up candidate search, but disables "
            "demo_existence repair evidence."
        ),
    )
    parser.add_argument(
        "--defer-source-repair",
        action="store_true",
        help=(
            "Mark scripted/demo/success-NN repair as deferred for candidate mining; "
            "K=5 confirmation/review should rerun without this flag."
        ),
    )
    parser.add_argument(
        "--stop-after-first-repair-valid-core",
        action="store_true",
        help=(
            "Stop causal-unit validation after the first unit that has both "
            "same-failure necessity and repair-valid evidence."
        ),
    )
    parser.add_argument(
        "--repair-scheduling-mode",
        choices=("pass_hunt", "topk_complete"),
        default="pass_hunt",
        help=(
            "pass_hunt prioritizes the first strict policy repair pass and defers "
            "expensive source repair; topk_complete fills all repair sources for "
            "confirmed review/gold reports."
        ),
    )
    parser.add_argument(
        "--disable-rule-language-intervention",
        action="store_true",
        help="Disable causal-v4 phrase-level prompt disambiguation repair attempts.",
    )
    parser.add_argument(
        "--enable-visual-policy-mask",
        action="store_true",
        help=(
            "Try causal-v4 policy-input visual mask/highlight interventions when "
            "a reliable object projection is available. Degraded cases are reported "
            "but skipped."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record an mp4 for each natural rollout. Defaults to OUTPUT_PARENT/videos.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Directory for rollout mp4 files. Setting this also enables video recording.",
    )
    parser.add_argument(
        "--video-prefix",
        default="",
        help="Prefix for video filenames. Defaults to the output JSON stem.",
    )
    parser.add_argument("--video-camera", default="agentview_image")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-every-n", type=int, default=1)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=8)
    parser.add_argument(
        "--video-no-flip",
        action="store_true",
        help="Do not apply the same 180-degree image flip used for policy observations.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.task_ids = parse_int_list(args.task_ids)
    args.init_state_ids = parse_int_list(args.init_state_ids)
    if args.confirm_replay_trials is None:
        args.confirm_replay_trials = args.replay_trials
    if args.causal_ablation_trials is None:
        args.causal_ablation_trials = args.confirm_replay_trials
    args.repair_replay_trials = max(1, int(args.repair_replay_trials))
    args.causal_ablation_trials = max(1, int(args.causal_ablation_trials))
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args._runtime_profile = RuntimeProfile.create()
    client = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)

    rollouts: List[Pi05Rollout] = []
    selected = None
    selected_event = None
    final_eval = None
    causal_validation = None
    causal_validation_skipped_reason = None
    trace: List[dict] = []

    for task_id in args.task_ids:
        for init_state_id in args.init_state_ids:
            with _timed(args, "rollout_seconds"):
                rollout = collect_pi05_rollout(
                    args, client, args.task_suite_name, task_id, init_state_id
                )
            rollouts.append(rollout)
            with _timed(args, "event_search_seconds"):
                event = None
                if rollout.initial_state_valid:
                    event = find_failure_event(
                        rollout, args.event_window, args.min_distance_delta
                    )
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "init_state_id": init_state_id,
                        "initial_state_valid": rollout.initial_state_valid,
                        "initial_state_quality": rollout.initial_state_quality,
                        "success": rollout.success,
                        "length": rollout.length,
                        "target_key": rollout.target_key,
                        "initial_distance": rollout.distance_trace[0],
                        "final_distance": rollout.distance_trace[-1],
                        "event": event,
                        "failure_signature": rollout.failure_signature.to_dict(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not rollout.initial_state_valid:
                continue
            if (not rollout.success) and event is not None:
                selected = rollout
                selected_event = event
                with _timed(args, "minimization_seconds"):
                    final_eval, trace = minimize_event_slice(
                        args, rollout, event, client=client
                    )
                if final_eval.same_failure:
                    with _timed(args, "causal_ablation_seconds"):
                        causal_validation = validate_causal_units(
                            args,
                            rollout,
                            final_eval,
                            all_rollouts=rollouts,
                            client=client,
                        )
                else:
                    causal_validation_skipped_reason = (
                        "final_minimal_slice_did_not_reproduce_same_failure"
                    )
                break
        if selected is not None:
            break

    _set_counter(args, "replay_cache_size", len(getattr(args, "_replay_evaluation_cache", {}) or {}))
    if selected is not None:
        _save_rollout_archive(args, selected)
    runtime_profile = args._runtime_profile.to_dict()
    report = build_report(
        args,
        rollouts,
        selected,
        selected_event,
        final_eval,
        trace,
        causal_validation,
        causal_validation_skipped_reason,
        runtime_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["feasibility"], indent=2, ensure_ascii=False), flush=True)
    if report["selected_failed_rollout"] is not None:
        print(json.dumps(report["selected_failed_rollout"], indent=2, ensure_ascii=False), flush=True)
        print(json.dumps(report["failure_event"], indent=2, ensure_ascii=False), flush=True)
        print(json.dumps(report["causal_failure_slice"], indent=2, ensure_ascii=False), flush=True)
        print(json.dumps(report["reproduction_statistics"], indent=2, ensure_ascii=False), flush=True)
    print(f"Wrote {args.output}", flush=True)
    if not report["feasibility"]["pi05_natural_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
