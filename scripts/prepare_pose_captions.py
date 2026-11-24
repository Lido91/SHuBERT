"""Tokenize paired captions into arrays indexed by packed clip, for semantic alignment pretraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pack = json.loads(Path(args.pack).read_text())
    if pack.get("format") != "shubert-pose-pack-v1":
        raise ValueError("--pack must be a packed pose manifest such as train.pack.json")
    if args.max_tokens < 1:
        raise ValueError("require --max-tokens >= 1")
    identity = {"records": [str(Path(path).resolve()) for path in args.records],
                "records_sha256": [digest(path) for path in args.records],
                "pack_offsets_sha256": pack["offsets_sha256"], "text_model": args.text_model,
                "max_tokens": args.max_tokens}
    manifest = output / "captions.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["identity"] != identity:
            raise ValueError("caption settings changed; choose a fresh output directory")
        print(json.dumps(saved), flush=True)
        return
    clips = len(np.load(pack["offsets"], mmap_mode="r", allow_pickle=False)) - 1
    texts = {}
    for path in args.records:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index, text = int(row["pack_index"]), row["text"].strip()
                if not 0 <= index < clips or index in texts:
                    raise ValueError(f"{path}: duplicate or out-of-range pack_index {index}")
                if text:
                    texts[index] = text
    if not texts:
        raise ValueError("no captions found")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    order = sorted(texts)
    lengths = np.zeros(clips, dtype=np.int64)
    encoded = []
    for begin in range(0, len(order), 4096):
        chunk = order[begin:begin + 4096]
        # As in SeRV: truncate to the token budget, which includes the end-of-sequence token.
        for index, ids in zip(chunk, tokenizer([texts[i] for i in chunk], truncation=True,
                                               max_length=args.max_tokens)["input_ids"]):
            if not ids or min(ids) < 0 or max(ids) >= len(tokenizer):
                raise ValueError(f"invalid tokens for pack_index {index}")
            lengths[index] = len(ids)
            encoded.append(np.asarray(ids, dtype=np.int32))
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    for name, array in (("tokens", np.concatenate(encoded)), ("offsets", offsets)):
        temporary = output / f"captions.{name}.tmp.npy"
        np.save(temporary, array)
        temporary.replace(output / f"captions.{name}.npy")
    summary = {"format": "shubert-pose-captions-v1", "identity": identity,
               "pack_offsets_sha256": pack["offsets_sha256"], "text_model": args.text_model,
               "max_tokens": args.max_tokens, "vocabulary": len(tokenizer), "clips": clips,
               "captioned_clips": len(order), "tokens_total": int(offsets[-1]),
               "captions_at_token_limit": int((lengths == args.max_tokens).sum())}
    for name in ("tokens", "offsets"):
        summary[name] = str(output / f"captions.{name}.npy")
        summary[f"{name}_sha256"] = digest(summary[name])
    temporary = output / "captions.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(manifest)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True,
                        help="JSON lines with pack_index and text, indexing the same clips as --pack")
    parser.add_argument("--pack", required=True, help="packed pose manifest, e.g. DATA/train.pack.json")
    parser.add_argument("--text-model", required=True, help="T5/mT5 name or local directory used for alignment")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())
