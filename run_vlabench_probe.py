from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from data_probe import DEFAULT_DATASET_ROOT
from run_offline_probe import DEFAULT_OUTPUT_DIR, parse_args, run_probe


DEFAULT_VLABENCH_ROOT = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/dataset/vlabench_unified"
)
DEFAULT_VLABENCH_OUTPUT_DIR = Path(
    "/root/autodl-tmp/research/Embodied_Delta_Debugging/outputs/vlabench_probe"
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.dataset_root == DEFAULT_DATASET_ROOT:
        args.dataset_root = DEFAULT_VLABENCH_ROOT
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_VLABENCH_OUTPUT_DIR
    if args.dataset_name == "LIBERO LeRobot":
        args.dataset_name = "VLABench unified"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_probe(args)

    import json

    print(json.dumps(summary["aggregate_metrics"], indent=2))
    print(json.dumps(summary["feasibility"], indent=2, ensure_ascii=False))
    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
