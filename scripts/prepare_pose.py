"""Build aligned pose manifests and train-only k-means targets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shubert_pose.data import read_manifest  # noqa: E402


def manifest(args):
    streams = {}
    for value in args.stream:
        name, directory = value.split("=", 1)
        if name in streams or not name or name in {"id", "frames"} or not name.isidentifier():
            raise ValueError(f"invalid or repeated stream name: {name}")
        streams[name] = Path(directory).resolve()
    inventories = {name: {p.stem: p for p in directory.glob("*.npy")} for name, directory in streams.items()}
    if args.ids:
        ids = Path(args.ids).read_text().splitlines()
    else:
        ids = sorted(next(iter(inventories.values())))
        if any(set(inventory) != set(ids) for inventory in inventories.values()):
            raise ValueError("stream directories have different IDs; supply --ids to select an explicit split")
    if not ids or len(set(ids)) != len(ids) or any(not i or Path(i).name != i or i in {".", ".."} for i in ids):
        raise ValueError("sample IDs must be unique, nonempty filenames")
    output = Path(args.output)
    rows, dimensions = [], {}
    for sample_id in ids:
        row, lengths = {"id": sample_id}, set()
        for name, inventory in inventories.items():
            if sample_id not in inventory:
                raise ValueError(f"missing {name} coordinates for {sample_id}")
            path = inventory[sample_id]
            feature = np.load(path, mmap_mode="r", allow_pickle=False)
            if feature.ndim != 2 or min(feature.shape) < 1 or not np.isfinite(feature).all():
                raise ValueError(f"{path}: require finite, nonempty [T, D] coordinates")
            if dimensions.setdefault(name, feature.shape[1]) != feature.shape[1]:
                raise ValueError(f"{path}: inconsistent stream dimension")
            lengths.add(feature.shape[0])
            row[name] = str(path)
        if len(lengths) != 1:
            raise ValueError(f"unaligned stream frame counts for {sample_id}")
        row["frames"] = lengths.pop()
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *streams, "frames"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest": str(output), "samples": len(rows), "dimensions": dimensions}))


def cluster(args):
    if args.clusters < 2 or args.sample_frames < args.clusters:
        raise ValueError("require sample_frames >= clusters >= 2")
    if "train" not in args.splits or len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must include train and have no duplicates")
    root = Path(args.data)
    manifests = {split: read_manifest(root / f"{split}.tsv", args.streams) for split in args.splits}
    train = manifests["train"]
    total = sum(row["frames"] for row in train)
    if total < args.clusters:
        raise ValueError("not enough training frames for the requested clusters")
    rng = np.random.default_rng(args.seed)
    # Sample global frame indices uniformly without loading the full corpus.
    indices = np.sort(rng.choice(total, min(total, args.sample_frames), replace=False))
    model_dir = root / "codebooks"
    model_dir.mkdir(exist_ok=True)
    metadata = {"fit_split": "train", "seed": args.seed, "clusters": args.clusters,
                "sample_frames": len(indices), "streams": {}, "manifest_sha256": {}}
    for stream in args.streams:
        sampled, offset, dimension = [], 0, None
        for row in train:
            feature = np.load(row["paths"][stream], mmap_mode="r", allow_pickle=False)
            if feature.ndim != 2 or feature.shape[0] != row["frames"] or not np.isfinite(feature).all():
                raise ValueError(f"invalid coordinates: {row['paths'][stream]}")
            if dimension is not None and feature.shape[1] != dimension:
                raise ValueError(f"{stream}: inconsistent dimensions")
            dimension = feature.shape[1]
            begin, end = np.searchsorted(indices, [offset, offset + row["frames"]])
            sampled.append(np.asarray(feature[indices[begin:end] - offset], dtype=np.float32))
            offset += row["frames"]
        model = MiniBatchKMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10,
                                batch_size=min(10000, len(indices)))
        model.fit(np.concatenate(sampled))
        joblib.dump(model, model_dir / f"{stream}.joblib")
        metadata["streams"][stream] = dimension
        for split, rows in manifests.items():
            with (root / f"{split}.{stream}.km").open("w") as handle:
                for row in rows:
                    feature = np.load(row["paths"][stream], allow_pickle=False)
                    if feature.shape != (row["frames"], dimension) or not np.isfinite(feature).all():
                        raise ValueError(f"invalid coordinates: {row['paths'][stream]}")
                    targets = model.predict(np.asarray(feature, dtype=np.float32))
                    handle.write(" ".join(map(str, targets.tolist())) + "\n")
    for split in args.splits:
        metadata["manifest_sha256"][split] = hashlib.sha256((root / f"{split}.tsv").read_bytes()).hexdigest()
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("manifest", help="align existing per-stream .npy files by exact sample ID")
    prepare.add_argument("--stream", action="append", required=True, help="NAME=DIRECTORY; repeat for each stream")
    prepare.add_argument("--ids", help="split IDs, one per line; order is preserved")
    prepare.add_argument("--output", required=True, help="e.g. data/pose/train.tsv")
    prepare.set_defaults(run=manifest)
    fit = commands.add_parser("cluster", help="fit train-only codebooks and label each requested split")
    fit.add_argument("--data", required=True)
    fit.add_argument("--streams", nargs="+", default=["body", "left_hand", "right_hand"])
    fit.add_argument("--splits", nargs="+", default=["train"])
    fit.add_argument("--clusters", type=int, default=256)
    fit.add_argument("--sample-frames", type=int, default=100000)
    fit.add_argument("--seed", type=int, default=0)
    fit.set_defaults(run=cluster)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
