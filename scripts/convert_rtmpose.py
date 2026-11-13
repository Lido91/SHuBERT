"""Convert COCO-WholeBody-133 RTMPose pickles into pose-only SHuBERT coordinate streams.

Input: one pickle per clip, {"keypoints": [T, 1, 133, 2] or [T, 133, 2], "scores": [T, 1, 133] or
[T, 133]}, with optional integer "start"/"end" frame bounds. Output: OUTPUT/STREAM/ID.npy, float32:

  body        [T, 14]  nose, shoulders, elbows, wrists in the signing-space frame of
                       features/body_features.py (shoulder-width units, centred on the signer)
  left_hand   [T, 42]  21 hand joints relative to the hand's wrist, in the same shoulder-width unit
  right_hand  [T, 42]

Hand location lives only in the body stream (its wrists); the hand streams carry shape and
orientation, so their k-means targets describe handshape rather than position in the frame.
"""

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import pickle

import numpy as np

# COCO-WholeBody indices, in the joint order of the original 14D body-posture stream.
BODY = [0, 5, 6, 7, 8, 9, 10]  # nose, L/R shoulder, L/R elbow, L/R wrist
NOSE, LEFT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1, 5, 6
HANDS = {"left_hand": slice(91, 112), "right_hand": slice(112, 133)}
STREAMS = ["body", *HANDS]


def fill_missing(values, valid):
    """Hold the last valid row; rows before the first valid one take its value; no valid row gives zeros."""
    if not valid.any():
        return np.zeros_like(values)
    index = np.where(valid, np.arange(len(valid)), -1)
    index = np.maximum.accumulate(index)
    index[index < 0] = np.argmax(valid)
    return values[index]


def convert(keypoints, scores, threshold=0.3):
    """[T, 133, 2] pixel coordinates and [T, 133] confidences -> {stream: [T, D] float32}."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2) or scores.shape != keypoints.shape[:2]:
        raise ValueError(f"expected [T, 133, 2] keypoints and [T, 133] scores, got {keypoints.shape}, {scores.shape}")
    valid = np.isfinite(keypoints).all(-1) & np.isfinite(scores) & (scores > threshold)
    keypoints = np.where(valid[..., None], keypoints, 0)

    # Signing-space frame: one head unit is half the shoulder width; the box is 6 x 7 head units,
    # horizontally centred on the nose and starting half a unit above the eye.
    reference = [NOSE, LEFT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER]
    anchors = np.stack([fill_missing(keypoints[:, j], valid[:, j]) for j in reference], axis=1)
    nose, eye, left_shoulder, right_shoulder = anchors.transpose(1, 0, 2)
    unit = np.linalg.norm(right_shoulder - left_shoulder, axis=-1) / 2
    framed = valid[:, reference].any(0).all() and (unit > 1e-6).any()
    if not framed:
        return {"body": np.zeros((len(keypoints), 14), np.float32),
                **{s: np.zeros((len(keypoints), 42), np.float32) for s in HANDS}}
    unit = fill_missing(unit, unit > 1e-6)[:, None]
    origin = np.stack([nose[:, 0] - 3 * unit[:, 0], eye[:, 1] - 0.5 * unit[:, 0]], axis=-1)
    size = unit * np.array([6.0, 7.0], dtype=np.float32)

    body = np.stack([fill_missing(keypoints[:, j], valid[:, j]) for j in BODY], axis=1)
    body = (body - origin[:, None]) / size[:, None] - 0.5
    present = np.stack([valid[:, j] for j in BODY], axis=1).any(0)  # joints never detected stay zero
    streams = {"body": (body * present[None, :, None]).reshape(len(body), 14)}

    for name, joints in HANDS.items():
        hand, seen = keypoints[:, joints], valid[:, joints]
        # A hand counts as detected when its wrist and most of its joints are confident.
        # Undetected frames are zeros: "no hand" stays a distinct, clusterable state.
        detected = seen[:, 0] & (seen.mean(-1) >= 0.5)
        hand = (hand - hand[:, :1]) / unit[:, None]
        hand = np.where((seen & detected[:, None])[..., None], hand, 0)
        streams[name] = hand.reshape(len(hand), 42)
    return {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in streams.items()}


def load_pose(path):
    with open(path, "rb") as handle:
        pose = pickle.load(handle)
    keypoints = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    if keypoints.ndim == 4:
        keypoints, scores = keypoints[:, 0], scores[:, 0]
    start, end = pose.get("start", 0), pose.get("end", len(scores))
    return keypoints[start:end], scores[start:end]


def save(path, array):
    # Write then rename, so an interrupted run never leaves a truncated file that a rerun would keep.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def process(job):
    sample_id, path, output, threshold, min_frames = job
    targets = {s: output / s / f"{sample_id}.npy" for s in STREAMS}
    if all(target.exists() for target in targets.values()):
        return "kept"
    try:
        keypoints, scores = load_pose(path)
        if len(scores) < min_frames:
            return "short"
        streams = convert(keypoints, scores, threshold)
    except Exception as error:  # one unreadable pickle must not stop a corpus-sized run
        return f"failed\t{path}\t{type(error).__name__}: {error}"
    for stream, target in targets.items():
        save(target, streams[stream])
    return "written"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="directory of .pkl pose files, searched recursively")
    parser.add_argument("--output", required=True, help="receives body/, left_hand/, right_hand/")
    parser.add_argument("--threshold", type=float, default=0.3, help="keypoint confidence threshold")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, output = Path(args.input).resolve(), Path(args.output).resolve()
    paths = sorted(root.rglob("*.pkl"))
    if not paths:
        raise ValueError(f"{root}: no .pkl files")
    # Nested inputs keep unique IDs: sub/dir/clip.pkl -> sub__dir__clip.
    ids = ["__".join(p.relative_to(root).with_suffix("").parts) for p in paths]
    if len(set(ids)) != len(ids):
        raise ValueError("relative paths do not give unique sample IDs")
    for stream in STREAMS:
        (output / stream).mkdir(parents=True, exist_ok=True)
    jobs = [(i, p, output, args.threshold, args.min_frames) for i, p in zip(ids, paths)]
    counts, failures = {}, []
    with Pool(args.workers) as pool:
        for status in pool.imap_unordered(process, jobs, chunksize=64):
            key = status.split("\t", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failures.append(status.split("\t", 1)[1])
    if failures:
        (output / "failed.txt").write_text("\n".join(failures) + "\n")
    print(json.dumps({"output": str(output), "files": len(jobs), **counts}))


if __name__ == "__main__":
    main()
