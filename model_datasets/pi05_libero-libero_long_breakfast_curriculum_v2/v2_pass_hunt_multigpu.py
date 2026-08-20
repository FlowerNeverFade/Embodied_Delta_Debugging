from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "prototype" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cost_summary import build_cost_summary_from_paths
from run_risk_critic_large_eval import (
    DEFAULT_POLICY_CONFIG,
    DEFAULT_POLICY_DIR,
    _active_video_dir,
    _case_fingerprint,
    _env_without_proxy,
    _format_int_list,
    _first_cuda_device,
    _load_json,
    _log_path,
    _policy_fingerprint,
    _report_path,
    _report_paths_from_rows,
    _run_case,
    _start_policy_server,
)
from risk_critic_export import export_risk_critic_dataset_from_paths
from train_risk_critic import _read_jsonl, train_and_evaluate


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "model_datasets"
    / "pi05_libero-libero_long_breakfast_curriculum_v2"
    / "outputs"
    / "v2_pass_hunt_20260526"
)


@dataclass(frozen=True)
class CaseSpec:
    task_id: int
    init_state_id: int
    seed: int


def _parse_int_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _plan_cases(task_ids: Sequence[int], init_state_ids: Sequence[int], seeds: Sequence[int], shuffle: bool, seed: int) -> list[CaseSpec]:
    cases = [CaseSpec(t, i, s) for t in task_ids for i in init_state_ids for s in seeds]
    if shuffle:
        import random

        rng = random.Random(int(seed))
        rng.shuffle(cases)
    return cases


def _worker_args(args: argparse.Namespace, gpu: int, port: int, worker_dir: Path) -> SimpleNamespace:
    ns = SimpleNamespace(**vars(args))
    ns.cuda_visible_devices = str(gpu)
    ns.policy_port = int(port)
    ns.output_dir = worker_dir
    ns.report_dir = worker_dir / "reports"
    ns.log_dir = worker_dir / "logs"
    ns.manifest_path = worker_dir / "manifest.jsonl"
    ns.cost_summary_path = worker_dir / "cost_summary.json"
    ns.export_path = worker_dir / "risk_critic_full_v1.jsonl"
    ns.train_output = worker_dir / "risk_critic_full_metrics.json"
    ns.summary_path = worker_dir / "summary.json"
    ns.video_dir = worker_dir / "videos"
    ns.record_video = True
    ns.require_video = True
    ns.launch_policy_server = True
    ns.allow_unverified_policy_server = False
    ns.dry_run = False
    ns.policy_ready_timeout = float(getattr(args, "policy_ready_timeout", 900.0))
    return ns


def _worker_main(
    worker_index: int,
    gpu: int,
    port: int,
    args: argparse.Namespace,
    plan_queue: mp.Queue,
    result_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    worker_dir = args.output_dir / f"gpu{gpu}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker = _worker_args(args, gpu, port, worker_dir)
    policy_proc = None
    try:
        policy_proc = _start_policy_server(worker, worker_dir)
        while not stop_event.is_set():
            try:
                item = plan_queue.get(timeout=1.0)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
            if item is None:
                break
            case = CaseSpec(*item)
            row = _run_case(worker, case.task_id, case.init_state_id, case.seed)
            row["worker_index"] = int(worker_index)
            row["worker_gpu"] = int(gpu)
            row["worker_port"] = int(port)
            result_queue.put(row)
    finally:
        if policy_proc is not None:
            try:
                policy_proc.terminate()
                policy_proc.wait(timeout=30)
            except Exception:
                try:
                    policy_proc.kill()
                except Exception:
                    pass
        result_queue.put({"type": "worker_done", "worker_index": int(worker_index)})


def _aggregate_rows(output_dir: Path, rows: list[dict], args: argparse.Namespace) -> dict:
    report_paths = _report_paths_from_rows(rows)
    cost_summary = build_cost_summary_from_paths(report_paths, outputs_root=output_dir)
    _write_json(output_dir / "cost_summary.json", cost_summary)

    export_summary = export_risk_critic_dataset_from_paths(
        report_paths,
        output_dir / "risk_critic_full_v1.jsonl",
        require_full_features=True,
        outputs_root=output_dir,
    )
    _write_json(output_dir / "risk_critic_export_summary.json", export_summary)

    samples = _read_jsonl(output_dir / "risk_critic_full_v1.jsonl")
    train_summary = train_and_evaluate(
        samples,
        seed=args.train_seed,
        val_fraction=args.val_fraction,
        steps=args.train_steps,
        feature_set="full_state_action_goal",
        split_by="source_report",
        min_class_count=args.min_class_count,
    )
    _write_json(output_dir / "risk_critic_full_metrics.json", train_summary)

    summary = {
        "schema_version": "v2-pass-hunt-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "aggregate": {
            "cases": len(rows),
            "semantic_pass": sum(1 for row in rows if row.get("status") == "semantic_pass"),
            "semantic_nonpass": sum(1 for row in rows if row.get("status") == "semantic_nonpass"),
            "timeout": sum(1 for row in rows if row.get("status") == "timeout"),
            "probe_failed": sum(1 for row in rows if row.get("status") == "probe_failed"),
            "full_success_repair_pass": sum(
                1 for row in rows if row.get("full_success_repair_pass") is True
            ),
            "same_failure_pass": sum(1 for row in rows if row.get("same_failure_pass") is True),
            "causal_pass": sum(1 for row in rows if row.get("causal_pass") is True),
            "positive_windows": sum(int(row.get("positive_windows") or 0) for row in rows),
        },
        "cost_summary": cost_summary,
        "export_summary": export_summary,
        "train_summary": train_summary,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic multi-GPU search for v2 repair-valid passes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-suite-name", default="libero_long_breakfast_curriculum_v2")
    parser.add_argument("--task-ids", default="0,1,2,3")
    parser.add_argument("--init-state-ids", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="7,17")
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--policy-ports", default="8060,8061,8062")
    parser.add_argument("--policy-server-kind", choices=("openpi", "lerobot_pi0fast"), default="lerobot_pi0fast")
    parser.add_argument("--policy-config", default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--action-tokenizer-path", type=Path, default=None)
    parser.add_argument("--text-tokenizer-path", type=Path, default=None)
    parser.add_argument("--allow-hub-download", action="store_true")
    parser.add_argument("--pytorch-device", default="cuda")
    parser.add_argument("--pytorch-compile-mode", default="none")
    parser.add_argument("--xla-mem-fraction", type=float, default=0.55)
    parser.add_argument("--replay-trials", type=int, default=5)
    parser.add_argument("--search-replay-trials", type=int, default=1)
    parser.add_argument("--confirm-replay-trials", type=int, default=5)
    parser.add_argument("--repair-replay-trials", type=int, default=1)
    parser.add_argument("--scripted-expert-repair-max-steps", type=int, default=180)
    parser.add_argument("--initial-state-max-attempts", type=int, default=8)
    parser.add_argument("--disable-initial-state-quality-filter", action="store_true")
    parser.add_argument("--probe-max-steps", type=int, default=0)
    parser.add_argument("--event-window", type=int, default=32)
    parser.add_argument("--causal-context-before", type=int, default=36)
    parser.add_argument("--causal-context-after", type=int, default=8)
    parser.add_argument("--causal-max-units", type=int, default=18)
    parser.add_argument("--causal-ablation-trials", type=int, default=1)
    parser.add_argument("--continuation", choices=("recorded", "policy"), default="recorded")
    parser.add_argument("--camera-size", type=int, default=512)
    parser.add_argument("--record-video", action="store_true", default=True)
    parser.add_argument("--video-camera", default="agentview_image")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-every-n", type=int, default=1)
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-quality", type=int, default=10)
    parser.add_argument("--video-no-flip", action="store_true")
    parser.add_argument("--require-video", action="store_true", default=True)
    parser.add_argument("--positive-target", type=int, default=3)
    parser.add_argument("--min-cases-before-positive-stop", type=int, default=60)
    parser.add_argument("--shuffle-cases", action="store_true", default=False)
    parser.add_argument("--case-order-seed", type=int, default=20260526)
    parser.add_argument("--case-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--policy-ready-timeout", type=float, default=900.0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--min-class-count", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videos").mkdir(parents=True, exist_ok=True)

    args.task_ids = _parse_int_list(args.task_ids)
    args.init_state_ids = _parse_int_list(args.init_state_ids)
    args.seeds = _parse_int_list(args.seeds)
    args.gpu_ids = _parse_int_list(args.gpu_ids)
    args.policy_ports = _parse_int_list(args.policy_ports)
    if len(args.gpu_ids) != len(args.policy_ports):
        raise SystemExit("gpu-ids and policy-ports must have the same length")

    planned = _plan_cases(
        args.task_ids,
        args.init_state_ids,
        args.seeds,
        bool(args.shuffle_cases),
        int(args.case_order_seed),
    )
    if args.max_cases > 0:
        planned = planned[: int(args.max_cases)]
    _write_json(
        args.output_dir / "planned_cases.json",
        {
            "schema_version": "v2-pass-hunt-plan-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "planned_cases": [asdict(case) for case in planned],
            "task_ids": args.task_ids,
            "init_state_ids": args.init_state_ids,
            "seeds": args.seeds,
            "gpu_ids": args.gpu_ids,
            "policy_ports": args.policy_ports,
        },
    )

    plan_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue()
    stop_event = mp.Event()
    for case in planned:
        plan_queue.put((case.task_id, case.init_state_id, case.seed))

    workers = []
    for worker_index, (gpu, port) in enumerate(zip(args.gpu_ids, args.policy_ports)):
        proc = mp.Process(
            target=_worker_main,
            args=(worker_index, gpu, port, args, plan_queue, result_queue, stop_event),
            daemon=True,
        )
        proc.start()
        workers.append(proc)
    for _ in workers:
        plan_queue.put(None)

    rows: list[dict] = []
    done_workers = 0
    positive_full_success = 0
    manifest_path = args.output_dir / "manifest.jsonl"
    while done_workers < len(workers):
        try:
            item = result_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        if item.get("type") == "worker_done":
            done_workers += 1
            continue
        rows.append(item)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if item.get("full_success_repair_pass") is True:
            positive_full_success += 1
        processed = len(rows)
        if (
            args.positive_target > 0
            and processed >= int(args.min_cases_before_positive_stop)
            and positive_full_success >= int(args.positive_target)
        ):
            stop_event.set()

    for proc in workers:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()

    summary = _aggregate_rows(args.output_dir, rows, args)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
