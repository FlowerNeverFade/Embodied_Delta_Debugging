from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from causal_failure_predicates import (
    CausalUnitResult,
    CausalValidationResult,
    FailureSignature,
    GoalPredicate,
    SameFailureResult,
    StateSnapshot,
    build_causal_units,
    candidate_overlaps_failure_anchor,
    compare_failure_signatures,
    get_goal_predicates,
    infer_failure_signature,
    make_causal_validation_result,
    make_state_snapshot,
    semantic_quality_for_env,
)
from edd_types import CandidateSlice


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "real_sim_failure_probe.json"


@dataclass
class FailedRollout:
    actions: np.ndarray
    states_before_action: List[np.ndarray]
    snapshots: List[StateSnapshot]
    goal_predicates: Tuple[GoalPredicate, ...]
    semantic_quality: str
    failure_signature: FailureSignature
    target_key: str
    task_language: str
    initial_distance: float
    final_distance: float
    success: bool
    distance_trace: List[float]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


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

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "same_failure": bool(self.same_failure),
            "same_failure_rate": float(self.same_failure_rate),
            "failure_rate": float(self.failure_rate),
            "trials": int(self.trials),
            "success": bool(self.success),
            "start_distance": float(self.start_distance),
            "end_distance": float(self.end_distance),
            "distance_delta": float(self.distance_delta),
            "steps": int(self.steps),
            "failure_signature": self.signature.to_dict(),
            "same_failure_evidence": self.same_failure_evidence.to_dict(),
        }


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(float(quat[3])) / den).astype(np.float32)


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


def _select_target_key(obs: dict) -> str:
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    keys = _object_pos_keys(obs)
    if not keys:
        raise RuntimeError("No object position keys found in LIBERO observation")
    # For this first real-failure probe, treat the nearest movable object as the
    # object the bad policy should move away from. This avoids injecting an
    # answer while still creating a concrete same-failure predicate.
    return min(keys, key=lambda k: float(np.linalg.norm(eef - np.asarray(obs[k]))))


def _distance(obs: dict, target_key: str) -> float:
    return float(
        np.linalg.norm(
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            - np.asarray(obs[target_key], dtype=np.float64)
        )
    )


def _move_away_action(obs: dict, target_key: str, scale: float) -> np.ndarray:
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    target = np.asarray(obs[target_key], dtype=np.float64)
    direction = eef - target
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(direction[:3] * float(scale), -1.0, 1.0)
    action[6] = -1.0
    return action


def _make_env(args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.gpu_id))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    env.seed(args.seed)
    initial_states = task_suite.get_task_init_states(args.task_id)
    return env, task_suite, task, initial_states


def collect_real_failed_rollout(args) -> Tuple[FailedRollout, dict]:
    env, _task_suite, task, initial_states = _make_env(args)
    try:
        env.reset()
        init_state_id = min(args.init_state_id, len(initial_states) - 1)
        obs = env.set_init_state(initial_states[init_state_id])
        dummy_action = [0.0] * 6 + [-1.0]
        for _ in range(args.num_steps_wait):
            obs, _reward, _done, _info = env.step(dummy_action)

        target_key = _select_target_key(obs)
        initial_distance = _distance(obs, target_key)
        predicates = get_goal_predicates(env)
        semantic_quality = semantic_quality_for_env(env)
        actions = []
        states = []
        distances = [initial_distance]
        snapshots = [
            make_state_snapshot(
                0,
                obs,
                env=env,
                success=False,
                predicates=predicates,
            )
        ]
        success = False

        for step_id in range(args.max_steps):
            states.append(_state(env))
            action = _move_away_action(obs, target_key, scale=args.action_scale)
            actions.append(action)
            obs, _reward, done, _info = env.step(action.tolist())
            success = success or bool(done)
            distances.append(_distance(obs, target_key))
            snapshots.append(
                make_state_snapshot(
                    step_id + 1,
                    obs,
                    env=env,
                    action=action,
                    success=success,
                    predicates=predicates,
                )
            )
            if done:
                break

        failure_signature = infer_failure_signature(
            snapshots,
            predicates=predicates,
            semantic_quality=semantic_quality,
            event_window=args.event_window,
            task_language=task.language,
        )
        rollout = FailedRollout(
            actions=np.asarray(actions, dtype=np.float32),
            states_before_action=states,
            snapshots=snapshots,
            goal_predicates=predicates,
            semantic_quality=semantic_quality,
            failure_signature=failure_signature,
            target_key=target_key,
            task_language=task.language,
            initial_distance=float(initial_distance),
            final_distance=float(distances[-1]),
            success=bool(success),
            distance_trace=[float(x) for x in distances],
        )
        meta = {
            "task_suite_name": args.task_suite_name,
            "task_id": int(args.task_id),
            "task_language": task.language,
            "init_state_id": int(init_state_id),
            "target_key": target_key,
            "policy": "observation-driven move-away baseline",
        }
        return rollout, meta
    finally:
        env.close()


def replay_candidate(
    args,
    rollout: FailedRollout,
    start: int,
    end: int,
    reference_signature: FailureSignature,
    action_replacements: Optional[Dict[int, np.ndarray]] = None,
    stage_level: str = "real_sim_replay",
) -> ReplayEvaluation:
    candidate = CandidateSlice.from_window(
        start, end, n_steps=rollout.length, level=stage_level
    )
    trials = max(1, int(args.replay_trials))
    same_count = 0
    failure_count = 0
    representative_signature = reference_signature
    representative_evidence = compare_failure_signatures(
        reference_signature, reference_signature, threshold=args.same_failure_threshold
    )
    representative_success = False
    representative_end_distance = float(rollout.distance_trace[min(rollout.length, start)])

    for _trial in range(trials):
        env, _task_suite, _task, _initial_states = _make_env(args)
        try:
            env.reset()
            obs = _set_state_and_obs(env, rollout.states_before_action[start])
            predicates = get_goal_predicates(env)
            semantic_quality = semantic_quality_for_env(env)
            snapshots = [
                make_state_snapshot(
                    start,
                    obs,
                    env=env,
                    success=False,
                    predicates=predicates,
                )
            ]
            success = False
            last_obs = obs
            for t in range(start, rollout.length):
                action = np.asarray(rollout.actions[t], dtype=np.float32)
                if action_replacements is not None and t in action_replacements:
                    action = np.asarray(action_replacements[t], dtype=np.float32)
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
                reference_signature,
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
        finally:
            env.close()

    start_distance = float(rollout.distance_trace[start])
    delta = float(representative_end_distance - start_distance)
    same_failure_rate = float(same_count / trials)
    same_failure = (
        same_failure_rate >= float(args.accept_same_failure_rate)
        and candidate_overlaps_failure_anchor(candidate, reference_signature)
    )
    return ReplayEvaluation(
        candidate=candidate,
        same_failure=bool(same_failure),
        same_failure_rate=same_failure_rate,
        failure_rate=float(failure_count / trials),
        trials=int(trials),
        success=bool(representative_success),
        start_distance=start_distance,
        end_distance=float(representative_end_distance),
        distance_delta=delta,
        steps=int(max(0, end - start)),
        signature=representative_signature,
        same_failure_evidence=representative_evidence,
    )


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


def validate_causal_units(
    args,
    rollout: FailedRollout,
    final_eval: ReplayEvaluation,
) -> CausalValidationResult:
    units = build_causal_units(
        final_eval.candidate,
        rollout.actions,
        rollout.snapshots,
        rollout.failure_signature,
        chunk_size=args.causal_chunk_size,
    )
    results: List[CausalUnitResult] = []
    for unit in units:
        best_eval = None
        best_strategy = ""
        for strategy in ("hold", "adjacent", "gripper_correction"):
            replacements = _replacement_actions(rollout.actions, unit.interval, strategy)
            evaluation = replay_candidate(
                args,
                rollout,
                final_eval.candidate.span_start or 0,
                final_eval.candidate.span_end or rollout.length,
                rollout.failure_signature,
                action_replacements=replacements,
                stage_level="causal_ablation_%s" % strategy,
            )
            if best_eval is None or evaluation.same_failure_rate < best_eval.same_failure_rate:
                best_eval = evaluation
                best_strategy = strategy
        assert best_eval is not None
        ce = float(final_eval.same_failure_rate - best_eval.same_failure_rate)
        results.append(
            CausalUnitResult(
                unit=unit,
                base_same_failure_rate=final_eval.same_failure_rate,
                ablated_same_failure_rate=best_eval.same_failure_rate,
                causal_effect=ce,
                is_causal_core=bool(ce >= args.causal_effect_threshold),
                best_counterfactual={
                    "strategy": best_strategy,
                    "unit_id": unit.unit_id,
                    "evaluation": best_eval.to_dict(),
                },
            )
        )
    return make_causal_validation_result(
        final_eval.same_failure_rate,
        results,
        ce_threshold=args.causal_effect_threshold,
    )


def minimize_real_failure(args, rollout: FailedRollout) -> Tuple[ReplayEvaluation, List[dict]]:
    start, end = 0, rollout.length
    trace: List[dict] = []

    def accepts(s: int, e: int, stage: str) -> bool:
        ev = replay_candidate(args, rollout, s, e, rollout.failure_signature)
        trace.append({"stage": stage, **ev.to_dict()})
        return ev.same_failure

    full_eval = replay_candidate(args, rollout, start, end, rollout.failure_signature)
    trace.append({"stage": "full_failed_rollout", **full_eval.to_dict()})
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

    final_eval = replay_candidate(args, rollout, start, end, rollout.failure_signature)
    trace.append({"stage": "minimal_real_sim_slice", **final_eval.to_dict()})
    return final_eval, trace


def build_report(
    args,
    rollout: FailedRollout,
    meta: dict,
    final_eval: ReplayEvaluation,
    trace: List[dict],
    causal_validation: CausalValidationResult,
) -> dict:
    full_delta = float(rollout.final_distance - rollout.initial_distance)
    passed = (
        (not rollout.success)
        and final_eval.same_failure
        and final_eval.steps < rollout.length
        and causal_validation.passed
        and rollout.semantic_quality != "degraded"
    )
    return {
        "schema_version": "shed-cfs-causal-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "LIBERO simulator execution; no controlled success-demo injection",
        "task_policy_environment": meta,
        "failure_predicate": {
            "type": "semantic same-failure predicate with intervention validation",
            "acceptance": "same_failure_rate >= %.2f and semantic_match_score >= %.2f"
            % (args.accept_same_failure_rate, args.same_failure_threshold),
            "failure_taxonomy": [
                "unsatisfied_goal_predicates_at_timeout",
                "wrong_object",
                "grasp_miss_no_transport",
                "premature_release_or_slip",
                "wrong_placement",
                "stagnation_timeout",
                "unsafe_contact",
            ],
        },
        "original_failed_rollout": {
            "length": int(rollout.length),
            "success": bool(rollout.success),
            "semantic_quality": rollout.semantic_quality,
            "initial_distance": float(rollout.initial_distance),
            "final_distance": float(rollout.final_distance),
            "distance_delta": full_delta,
            "distance_trace_first_last": [
                float(rollout.distance_trace[0]),
                float(rollout.distance_trace[-1]),
            ],
        },
        "original_failure_signature": rollout.failure_signature.to_dict(),
        "goal_predicate_trace": rollout.failure_signature.evidence.get("goal_trace", {}),
        "causal_failure_slice": final_eval.candidate.to_dict(),
        "minimal_replay_context": {
            "pre_state": "simulator state before slice start",
            "candidate_actions": final_eval.candidate.to_dict(),
            "continuation": "recorded suffix actions from original failed rollout",
        },
        "reproduction_statistics": final_eval.to_dict(),
        "same_failure_evidence": final_eval.same_failure_evidence.to_dict(),
        "causal_validation": causal_validation.to_dict(),
        "causal_core_units": [r.to_dict() for r in causal_validation.causal_core_units],
        "counterfactual_pass_variants": list(causal_validation.counterfactual_pass_variants),
        "metrics": {
            "reduction_ratio": float(rollout.length / max(1, final_eval.steps)),
            "replay_evaluations": len(trace),
            "same_failure_rate": float(final_eval.same_failure_rate),
            "failure_rate": float(final_eval.failure_rate),
        },
        "search_trace": trace,
        "feasibility": {
            "real_sim_pass": bool(passed),
            "verdict": "real_sim_feasible" if passed else "real_sim_not_yet_validated",
            "interpretation": (
                "A non-injected simulator failure was minimized, semantically reproduced, and intervention-validated."
                if passed
                else "The simulator ran, but semantic reproduction or causal intervention validation did not pass."
            ),
        },
        "limitations": [
            "The bad policy is scripted, not yet a VLA model rollout.",
            "Causality is simulator-intervention evidence, not real-robot causal proof.",
            "Contact evidence is optional; BDDL predicates and state traces are the hard criteria.",
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SHED-CFS on a real LIBERO simulator failure without injected demo corruption."
    )
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--action-scale", type=float, default=0.55)
    parser.add_argument("--min-distance-delta", type=float, default=0.02)
    parser.add_argument("--event-window", type=int, default=24)
    parser.add_argument("--replay-trials", type=int, default=5)
    parser.add_argument("--same-failure-threshold", type=float, default=0.75)
    parser.add_argument("--accept-same-failure-rate", type=float, default=0.80)
    parser.add_argument("--causal-effect-threshold", type=float, default=0.30)
    parser.add_argument("--causal-chunk-size", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rollout, meta = collect_real_failed_rollout(args)
    final_eval, trace = minimize_real_failure(args, rollout)
    causal_validation = validate_causal_units(args, rollout, final_eval)
    report = build_report(args, rollout, meta, final_eval, trace, causal_validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["original_failed_rollout"], indent=2))
    print(json.dumps(report["causal_failure_slice"], indent=2))
    print(json.dumps(report["reproduction_statistics"], indent=2))
    print(json.dumps(report["feasibility"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    if not report["feasibility"]["real_sim_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
