"""Prepare a bounded, video-disjoint YouTubeASL trial from SLiM's training manifest."""

import argparse
import csv
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_rtmpose import STREAMS, process  # noqa: E402


def prepare(args):
    manifest = Path(args.manifest).resolve()
    pose_dir = Path(args.pose_dir).resolve()
    output = Path(args.output).resolve()
    if args.train_clips < 1 or args.valid_clips < 1 or args.workers < 1:
        raise ValueError("clip counts and workers must be positive")
    records, seen = [], set()
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            name = row["name"]
            if name in seen or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid or duplicate sample ID: {name!r}")
            seen.add(name)
            if row["num_frames"] >= 2:
                records.append((name, row["video_id"]))
    random.Random(args.seed).shuffle(records)
    splits = {"train": [], "valid": []}
    valid_videos = set()
    # Draw the held-out clips first, then exclude all their source videos from training.
    for name, video in records:
        if (pose_dir / f"{name}.pkl").is_file():
            splits["valid"].append(name)
            valid_videos.add(video)
            if len(splits["valid"]) == args.valid_clips:
                break
    for name, video in records:
        if video not in valid_videos and (pose_dir / f"{name}.pkl").is_file():
            splits["train"].append(name)
            if len(splits["train"]) == args.train_clips:
                break
    if len(splits["train"]) != args.train_clips or len(splits["valid"]) != args.valid_clips:
        raise ValueError("not enough available poses for the requested video-disjoint splits")
    metadata = {"source_manifest": str(manifest),
                "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "pose_dir": str(pose_dir), "seed": args.seed, "threshold": args.threshold,
                "validation_kind": "video-disjoint holdout drawn from the source training manifest",
                "splits": splits}
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "selection.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("existing selection differs; choose a fresh output directory")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    streams = output / "streams"
    for stream in STREAMS:
        (streams / stream).mkdir(parents=True, exist_ok=True)
    names = splits["train"] + splits["valid"]
    jobs = [(name, pose_dir / f"{name}.pkl", streams, args.threshold, 2) for name in names]
    counts = {}
    with Pool(args.workers) as pool:
        for index, status in enumerate(pool.imap(process, jobs, chunksize=16), 1):
            if status not in {"written", "kept"}:
                raise ValueError(status)
            counts[status] = counts.get(status, 0) + 1
            if index % 512 == 0 or index == len(jobs):
                print(json.dumps({"converted": index, "total": len(jobs), **counts}), flush=True)
    for split, ids in splits.items():
        frames = 0
        with (output / f"{split}.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", *STREAMS, "frames"], delimiter="\t")
            writer.writeheader()
            for name in ids:
                row = {"id": name}
                lengths = set()
                for stream, dimension in zip(STREAMS, (14, 42, 42)):
                    path = streams / stream / f"{name}.npy"
                    array = np.load(path, allow_pickle=False)
                    if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
                        raise ValueError(f"invalid coordinates: {path}")
                    lengths.add(len(array))
                    row[stream] = str(path)
                if len(lengths) != 1 or min(lengths) < 2:
                    raise ValueError(f"unaligned or empty streams: {name}")
                row["frames"] = lengths.pop()
                frames += row["frames"]
                writer.writerow(row)
        print(json.dumps({"split": split, "clips": len(ids), "frames": frames}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pose-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-clips", type=int, default=4096)
    parser.add_argument("--valid-clips", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
