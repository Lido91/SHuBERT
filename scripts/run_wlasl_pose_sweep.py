"""Run a validated three-checkpoint WLASL comparison inside the four-GPU allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--selection-split", choices=("dev", "test"), default="dev")
    parser.add_argument("--warm-start-root", type=Path)
    parser.add_argument("--finetune-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--selection-start-epoch", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "sweep.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ALREADY_RUNNING " + str(output), flush=True)
        return
    write_json(output / "progress.json", {"phase": "tests", "updated_unix": time.time()})
    with (output / "tests.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_wlasl_pose.py", "-v"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(output / "progress.json", {"phase": "preparing_data", "updated_unix": time.time()})
    data = args.data.resolve() if args.data else output / "data"
    if args.data:
        if not (data / "ready.json").is_file():
            raise FileNotFoundError("the supplied prepared data directory is incomplete")
    else:
        with (output / "prepare.log").open("a") as log:
            subprocess.run([sys.executable, "-u", "scripts/prepare_wlasl_pose.py", "--output", str(data)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoints = [args.checkpoints.resolve() / f"checkpoint{epoch}.pt" for epoch in (141, 142, 143)]
    if not all(path.is_file() for path in checkpoints):
        raise FileNotFoundError("one of the three requested epoch snapshots is missing")
    import hashlib
    sources = []
    for epoch, path in zip((141, 142, 143), checkpoints):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        sources.append({"epoch": epoch, "path": str(path), "sha256": digest.hexdigest()})
    write_json(output / "checkpoints.json", sources)
    write_json(output / "progress.json", {"phase": "gpu_smoke", "updated_unix": time.time()})
    base = [sys.executable, "-u", "scripts/eval_wlasl_pose.py", "--data", str(data), "--resume",
            "--selection-split", args.selection_split, "--finetune-mode", args.finetune_mode,
            "--selection-start-epoch", str(args.selection_start_epoch)]
    smoke_warm = (["--warm-start", str(args.warm_start_root / "epoch141.pth")]
                  if args.warm_start_root else [])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="3")
    with (output / "smoke.log").open("a") as log:
        subprocess.run(base + ["--checkpoint", str(checkpoints[0]), "--output-dir", str(output / "smoke"),
                               "--mode", "smoke", "--smoke-steps", "2", "--workers", "0"] + smoke_warm,
                       cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    jobs = []
    for device, (epoch, path) in enumerate(zip((141, 142, 143), checkpoints)):
        log = (output / f"epoch{epoch}.log").open("a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), TRITON_CACHE_DIR=f"/tmp/shubert-wlasl-{os.getuid()}-{device}")
        warm = (["--warm-start", str(args.warm_start_root / f"epoch{epoch}.pth")]
                if args.warm_start_root else [])
        process = subprocess.Popen(base + ["--checkpoint", str(path), "--output-dir", str(output / f"epoch{epoch}")] + warm,
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((epoch, process, log))
    while True:
        status = [{"epoch": epoch, "pid": process.pid, "exit_code": process.poll()} for epoch, process, _ in jobs]
        write_json(output / "progress.json", {"phase": "evaluating", "jobs": status, "updated_unix": time.time()})
        if all(process.poll() is not None for _, process, _ in jobs):
            break
        time.sleep(15)
    for _, _, log in jobs:
        log.close()
    results = []
    for epoch, process, _ in jobs:
        name = "test_metrics.json" if args.selection_split == "dev" else "test_as_dev_metrics.json"
        path = output / f"epoch{epoch}" / name
        results.append({"pretrain_epoch": epoch, "exit_code": process.returncode,
                        "result": json.loads(path.read_text()) if path.is_file() else None})
    write_json(output / "comparison.json", results)
    complete = all(row["exit_code"] == 0 and row["result"] is not None for row in results)
    write_json(output / "progress.json", {"phase": "complete" if complete else "failed", "results": results,
                                          "updated_unix": time.time()})
    if not complete:
        raise RuntimeError("one or more WLASL runs failed; inspect epoch logs")
    print("COMPARISON_COMPLETE " + str(output / "comparison.json"), flush=True)


if __name__ == "__main__":
    main()
