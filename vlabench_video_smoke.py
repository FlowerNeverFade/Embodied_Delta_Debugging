from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pyarrow.parquet as pq

from run_vlabench_probe import DEFAULT_VLABENCH_OUTPUT_DIR, DEFAULT_VLABENCH_ROOT


DEFAULT_VIDEO_KEY = "observation.images.image"


def _episode_video_ref(root: Path, episode_index: int, video_key: str) -> dict:
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    prefix = f"videos/{video_key}"
    columns = [
        "episode_index",
        "length",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    frame = pq.read_table(meta_path, columns=columns).to_pandas()
    row = frame.loc[frame["episode_index"] == int(episode_index)]
    if row.empty:
        raise KeyError(f"Episode {episode_index} not found in {meta_path}")

    record = row.iloc[0]
    chunk_index = int(record[f"{prefix}/chunk_index"])
    file_index = int(record[f"{prefix}/file_index"])
    video_path = (
        root
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    return {
        "episode_index": int(record["episode_index"]),
        "length": int(record["length"]),
        "video_key": video_key,
        "video_path": str(video_path),
        "from_timestamp": float(record[f"{prefix}/from_timestamp"]),
        "to_timestamp": float(record[f"{prefix}/to_timestamp"]),
    }


def decode_episode_frame(root: Path, episode_index: int, video_key: str) -> dict:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to decode VLABench AV1 video files") from exc

    ref = _episode_video_ref(root, episode_index=episode_index, video_key=video_key)
    video_path = Path(ref["video_path"])
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 10.0
        target_frame = max(0, int(round(ref["from_timestamp"] * fps)))
        decoded = None
        for index, frame in enumerate(container.decode(stream)):
            if index >= target_frame:
                decoded = frame.to_ndarray(format="rgb24")
                break

        if decoded is None:
            raise RuntimeError(
                f"Could not decode target frame {target_frame} from {video_path}"
            )

        ref.update(
            {
                "codec": stream.codec_context.name,
                "fps": fps,
                "target_frame_index": int(target_frame),
                "frame_shape": [int(v) for v in decoded.shape],
                "frame_dtype": str(decoded.dtype),
                "frame_mean": float(decoded.mean()),
                "frame_std": float(decoded.std()),
            }
        )
    return ref


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode one VLABench video frame.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_VLABENCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VLABENCH_OUTPUT_DIR)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--video-key", type=str, default=DEFAULT_VIDEO_KEY)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = decode_episode_frame(
        root=args.dataset_root,
        episode_index=args.episode,
        video_key=args.video_key,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "video_smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote video smoke report to {output_path}")


if __name__ == "__main__":
    main()
