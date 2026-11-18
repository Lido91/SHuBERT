"""Prepare all usable clips in SLiM's sign69 pack for 150-epoch pose pretraining."""

import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from convert_rtmpose import fill_missing

STREAM_DIMS = {"body": 14, "left_hand": 42, "right_hand": 42}
BODY = [0, 3, 4, 5, 6, 7, 8]
HANDS = {"left_hand": slice(9, 30), "right_hand": slice(30, 51)}
FACE = slice(51, 69)


def stream_dimensions(include_face=False):
    return dict(STREAM_DIMS, **({"face": 36} if include_face else {}))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert_sign69(raw, threshold=0.3, include_face=False):
    """Nose-centred body frame; wrist-relative hands; unit = half the shoulder width.

    The 69-joint pack omits the eye landmarks required by convert_rtmpose.py's
    original signing-space origin. This normalization uses only actual packed
    landmarks, with no fabricated eye coordinates, and gets new codebooks.
    """
    if raw.ndim != 3 or raw.shape[1:] != (69, 3) or len(raw) < 1:
        raise ValueError("expected nonempty [T, 69, 3] raw pixel coordinates and confidence")
    valid = np.isfinite(raw).all(-1) & (raw[..., 2] > threshold)
    xy = np.where(valid[..., None], raw[..., :2], 0).astype(np.float32)
    anchors = [fill_missing(xy[:, j], valid[:, j]) for j in (0, 3, 4)]
    nose, left, right = anchors
    unit = np.linalg.norm(right - left, axis=-1) / 2
    if not valid[:, [0, 3, 4]].any(0).all() or not (unit > 1e-6).any():
        return {s: np.zeros((len(raw), d), np.float32) for s, d in stream_dimensions(include_face).items()}
    unit = fill_missing(unit, unit > 1e-6)[:, None]
    body = np.stack([fill_missing(xy[:, j], valid[:, j]) for j in BODY], axis=1)
    body = (body - nose[:, None]) / (unit[:, None] * np.array([6, 7], np.float32))
    body *= valid[:, BODY].any(0)[None, :, None]
    streams = {"body": body.reshape(len(raw), 14)}
    for stream, joints in HANDS.items():
        hand, seen = xy[:, joints], valid[:, joints]
        detected = seen[:, 0] & (seen.mean(-1) >= 0.5)
        hand = (hand - hand[:, :1]) / unit[:, None]
        streams[stream] = np.where((seen & detected[:, None])[..., None], hand, 0).reshape(len(raw), 42)
    if include_face:
        face, seen = xy[:, FACE], valid[:, FACE]
        # The last packed face point is the facial nose anchor (WholeBody ID 53).
        detected = seen[:, -1] & (seen.mean(-1) >= 0.5)
        face = (face - face[:, -1:]) / unit[:, None]
        streams["face"] = np.where((seen & detected[:, None])[..., None], face, 0).reshape(len(raw), 36)
    return {s: np.ascontiguousarray(value, dtype=np.float32) for s, value in streams.items()}


def init_conversion(source, output, threshold, include_face=False):
    global RAW, OFFSETS, FEATURES, DONE, THRESHOLD, INCLUDE_FACE
    source, output = Path(source), Path(output)
    RAW = np.load(source / "poses.npy", mmap_mode="r")
    OFFSETS = np.load(source / "offsets.npy", mmap_mode="r")
    FEATURES = {s: np.load(output / f"{s}.npy", mmap_mode="r+") for s in stream_dimensions(include_face)}
    DONE = np.load(output / "converted.npy", mmap_mode="r+")
    THRESHOLD = threshold
    INCLUDE_FACE = include_face


def convert_block(bounds):
    completed = []
    for index in range(*bounds):
        if DONE[index]:
            continue
        begin, end = map(int, OFFSETS[index:index + 2])
        if end - begin < 2:
            completed.append((index, 2))
            continue
        streams = convert_sign69(np.asarray(RAW[begin:end]), THRESHOLD, INCLUDE_FACE)
        for stream in FEATURES:
            FEATURES[stream][begin:end] = streams[stream]
        completed.append((index, 1 if np.any(streams["body"]) else 2))
    for array in FEATURES.values():
        array.flush()
    for index, status in completed:
        DONE[index] = status
    DONE.flush()
    return bounds[1] - bounds[0]


def assign_stream(job):
    output, stream = job
    output = Path(output)
    marker = output / f"{stream}.labels.done"
    if marker.exists():
        return stream
    model = joblib.load(output / "codebooks" / f"{stream}.joblib")
    feature = np.load(output / f"{stream}.npy", mmap_mode="r")
    targets = np.lib.format.open_memmap(output / f"{stream}.labels.npy", mode="w+", dtype=np.uint16,
                                      shape=(len(feature),))
    for begin in range(0, len(feature), 131072):
        end = min(len(feature), begin + 131072)
        targets[begin:end] = model.predict(np.asarray(feature[begin:end]))
        if begin % (131072 * 64) == 0:
            print(json.dumps({"label_stream": stream, "frames": end, "total": len(feature)}), flush=True)
    targets.flush()
    marker.write_text("complete\n")
    return stream


def prepare(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    include_face = getattr(args, "include_face", False)
    dimensions = stream_dimensions(include_face)
    source_meta = json.loads((source / "meta.json").read_text())
    expected_joints = [0] + list(range(3, 11)) + list(range(91, 133)) + list(range(23, 40, 2)) + list(range(83, 91)) + [53]
    if (source_meta.get("format") != "slim-sign-pack-v1" or source_meta.get("layout") != "unisign69"
            or source_meta["coco_wholebody_indices"] != expected_joints
            or source_meta["channels"] != ["x", "y", "score"]):
        raise ValueError("source must be a raw, correctly ordered SLiM sign69 pack")
    if args.clusters < 2 or args.clusters > 65536 or args.sample_frames < args.clusters or args.workers < 1:
        raise ValueError("invalid clusters, sample size, or workers")
    offsets = np.load(source / "offsets.npy", allow_pickle=False)
    raw = np.load(source / "poses.npy", mmap_mode="r", allow_pickle=False)
    count, total = len(offsets) - 1, int(offsets[-1])
    if (count != source_meta["num_clips"] or total != source_meta["num_frames"]
            or raw.shape != (total, 69, 3) or offsets[0] != 0 or (np.diff(offsets) < 1).any()
            or len(source_meta["names"]) != count):
        raise ValueError("source pack arrays and metadata do not align")
    selection_config = {"source": str(source), "source_metadata_sha256": digest(source / "meta.json"),
                        "source_offsets_sha256": digest(source / "offsets.npy"),
                        "normalization": "sign69-nose-centered-v1", "threshold": args.threshold,
                        "clusters": args.clusters, "sample_frames": args.sample_frames,
                        "seed": args.seed, "valid_videos": args.valid_videos}
    if include_face:
        selection_config.update(dimensions=dimensions,
                                face_normalization="sign69-face-nose-half-shoulder-v1")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "preparation.json"
    if config_path.exists() and json.loads(config_path.read_text()) != selection_config:
        raise ValueError("preparation settings changed; choose a fresh output directory")
    config_path.write_text(json.dumps(selection_config, indent=2) + "\n")
    if (output / "ready.json").exists():
        print((output / "ready.json").read_text(), flush=True)
        return
    for stream, dimension in dimensions.items():
        path = output / f"{stream}.npy"
        if not path.exists():
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(total, dimension))
            array.flush()
    done_path = output / "converted.npy"
    if not done_path.exists():
        array = np.lib.format.open_memmap(done_path, mode="w+", dtype=np.uint8, shape=(count,))
        array[:] = 0
        array.flush()
    print(json.dumps({"phase": "convert", "clips": count, "frames": total}), flush=True)
    blocks = [(begin, min(count, begin + 1024)) for begin in range(0, count, 1024)]
    with Pool(args.workers, initializer=init_conversion,
              initargs=(str(source), str(output), args.threshold, include_face)) as pool:
        completed = 0
        for length in pool.imap_unordered(convert_block, blocks):
            completed += length
            if completed % 10240 == 0 or completed == count:
                print(json.dumps({"converted_clips": completed, "total": count}), flush=True)
    done = np.load(done_path, mmap_mode="r")
    videos = np.array([name.rsplit("-", 1)[0] for name in source_meta["names"]])
    candidates = np.unique(videos[done == 1])
    if not 0 < args.valid_videos < len(candidates):
        raise ValueError("valid_videos must leave videos in both splits")
    rng = np.random.default_rng(args.seed)
    valid_videos = rng.choice(candidates, args.valid_videos, replace=False)
    held_out = np.isin(videos, valid_videos)
    indices = {"train": np.flatnonzero((done == 1) & ~held_out),
               "valid": np.flatnonzero((done == 1) & held_out)}
    np.save(output / "offsets.npy", offsets)
    for split, chosen in indices.items():
        np.save(output / f"{split}.indices.npy", chosen)
    lengths = offsets[indices["train"] + 1] - offsets[indices["train"]]
    cumulative = np.concatenate(([0], np.cumsum(lengths)))
    selected = np.sort(rng.choice(int(cumulative[-1]), min(args.sample_frames, int(cumulative[-1])), replace=False))
    owners = np.searchsorted(cumulative[1:], selected, side="right")
    global_frames = offsets[indices["train"][owners]] + selected - cumulative[owners]
    codebooks = output / "codebooks"
    codebooks.mkdir(exist_ok=True)
    for stream in dimensions:
        path = codebooks / f"{stream}.joblib"
        if not path.exists():
            feature = np.load(output / f"{stream}.npy", mmap_mode="r")
            sampled = np.asarray(feature[global_frames], dtype=np.float32)
            if not np.isfinite(sampled).all():
                raise ValueError(f"{stream}: non-finite clustering inputs")
            model = MiniBatchKMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10,
                                    batch_size=min(10000, len(sampled)))
            model.fit(sampled)
            joblib.dump(model, path)
        print(json.dumps({"phase": "fitted", "stream": stream, "training_sample_frames": len(selected)}), flush=True)
    with Pool(min(len(dimensions), args.workers)) as pool:
        for stream in pool.imap_unordered(assign_stream, [(str(output), s) for s in dimensions]):
            print(json.dumps({"phase": "labels_complete", "stream": stream}), flush=True)
    base = {"format": "shubert-pose-pack-v1", "dimensions": dimensions, "codebook_size": args.clusters,
            "normalization": selection_config["normalization"], "fit_split": "train",
            "features": {s: str(output / f"{s}.npy") for s in dimensions},
            "labels": {s: str(output / f"{s}.labels.npy") for s in dimensions},
            "offsets": str(output / "offsets.npy"), "offsets_sha256": digest(output / "offsets.npy")}
    if include_face:
        base.update(face_normalization=selection_config["face_normalization"],
                    face_coco_wholebody_indices=source_meta["coco_wholebody_indices"][51:69])
    summary = {"source_clips": count, "source_frames": total, "excluded_clips": int((done != 1).sum()),
               "held_out_videos": valid_videos.tolist(), "splits": {}}
    for split, chosen in indices.items():
        metadata = dict(base, indices=str(output / f"{split}.indices.npy"),
                        indices_sha256=digest(output / f"{split}.indices.npy"))
        (output / f"{split}.pack.json").write_text(json.dumps(metadata, indent=2) + "\n")
        summary["splits"][split] = {"clips": len(chosen),
                                    "frames": int((offsets[chosen + 1] - offsets[chosen]).sum())}
    # Use Fairseq's actual batching rule to size the LR schedule; epochs remain the stopping criterion.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shubert_pose.packed_dataset import PackedPoseDataset
    data = PackedPoseDataset(output / "train.pack.json", list(dimensions), list(dimensions.values()),
                             args.clusters, max_frames=256, shuffle=False)
    batches = data.batch_by_size(data.ordered_indices(), max_tokens=15000, required_batch_size_multiple=1)
    summary["microbatches_per_epoch"] = len(batches)
    summary["updates_per_epoch"] = math.ceil(math.ceil(len(batches) / 4) / 2)
    summary["epochs"] = 150
    summary["total_updates_for_lr"] = summary["updates_per_epoch"] * 150
    summary["warmup_updates"] = min(32000, max(1, int(summary["total_updates_for_lr"] * 0.08)))
    temporary = output / "ready.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "ready.json")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--sample-frames", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-videos", type=int, default=32)
    parser.add_argument("--include-face", action="store_true", help="include the 18 available facial points as 36D")
    prepare(parser.parse_args())
