"""Convert verified WLASL packs with exactly the formal pretraining normalization."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_youtubeasl_pack import BODY, HANDS, STREAM_DIMS, convert_sign69  # noqa: E402
from shubert_pose.wlasl import atomic_json, sha256  # noqa: E402


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    spec = {"source": str(source.resolve()), "normalization": "sign69-nose-centered-v1",
            "threshold": 0.3, "body_indices": BODY,
            "hands": {k: [v.start, v.stop] for k, v in HANDS.items()}, "splits": {}}
    seen, classes = set(), None
    for split in ("train", "dev", "test"):
        root, destination = source / split, output / split
        meta = json.loads((root / "meta.json").read_text())
        expected = [0] + list(range(3, 11)) + list(range(91, 133)) + list(range(23, 40, 2)) + list(range(83, 91)) + [53]
        if (meta.get("format") != "slim-sign-pack-v1" or meta["layout"] != "unisign69"
                or meta["channels"] != ["x", "y", "score"] or meta["coco_wholebody_indices"] != expected
                or meta.get("skipped") or len(set(meta["names"])) != len(meta["names"])
                or seen.intersection(meta["names"])):
            raise ValueError(f"invalid/incomplete/overlapping source split: {split}")
        seen.update(meta["names"])
        classes = meta["classes"] if classes is None else classes
        if (classes != meta["classes"] or len(classes) != 2000 or len(meta["labels"]) != len(meta["names"])
                or min(meta["labels"]) < 0 or max(meta["labels"]) >= 2000):
            raise ValueError("WLASL2000 labels/vocabulary mismatch")
        offsets = np.load(root / "offsets.npy", allow_pickle=False)
        raw = np.load(root / "poses.npy", mmap_mode="r", allow_pickle=False)
        if (len(offsets) != len(meta["names"]) + 1 or offsets[0] != 0 or (np.diff(offsets) < 1).any()
                or raw.shape != (int(offsets[-1]), 69, 3)):
            raise ValueError("source arrays/metadata are misaligned")
        entry = {"samples": len(meta["names"]), "frames": int(offsets[-1]),
                 "source_meta_sha256": sha256(root / "meta.json"),
                 "source_offsets_sha256": sha256(root / "offsets.npy"),
                 "source_poses_sha256": sha256(root / "poses.npy")}
        marker = destination / "ready.json"
        if marker.exists():
            if json.loads(marker.read_text()) != entry:
                raise ValueError("existing converted split has different source data")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            arrays = {s: np.lib.format.open_memmap(destination / (s + ".npy.tmp"), mode="w+", dtype=np.float32,
                                                  shape=(len(raw), d)) for s, d in STREAM_DIMS.items()}
            zero_body = 0
            for index in range(len(offsets) - 1):
                begin, end = map(int, offsets[index:index + 2])
                converted = convert_sign69(np.asarray(raw[begin:end]))
                zero_body += not np.any(converted["body"])
                for stream, array in converted.items():
                    if not np.isfinite(array).all():
                        raise ValueError(f"nonfinite coordinates: {split}/{meta['names'][index]}")
                    arrays[stream][begin:end] = array
                if index % 2000 == 0:
                    print(json.dumps({"split": split, "converted": index, "total": len(meta["names"])}), flush=True)
            for stream, array in arrays.items():
                array.flush()
                (destination / (stream + ".npy.tmp")).replace(destination / (stream + ".npy"))
            shutil.copyfile(root / "offsets.npy", destination / "offsets.npy")
            atomic_json(destination / "meta.json", meta)
            atomic_json(destination / "conversion.json", {"zero_body_clips_kept": int(zero_body),
                                                         "samples_filtered": 0})
            atomic_json(marker, entry)
        spec["splits"][split] = entry
    atomic_json(output / "ready.json", spec)
    print("WLASL_READY " + json.dumps(spec), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/scratch/lidowu91/SLiM/data/WLASL/packed"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
