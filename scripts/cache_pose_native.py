"""Populate the node-local library cache referenced by the existing SLiM PyTorch runtime."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import time


def main():
    source = Path("/scratch/lidowu91/Uni-Sign-Lido/envs/unisign/lib/python3.9/site-packages/torch/lib")
    destination = Path("/tmp") / f"slim-unisign-native-{os.getuid()}"
    destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    def stage(path):
        target = destination / path.name
        if target.exists() and target.stat().st_size == path.stat().st_size:
            return
        temporary = target.with_name(f"{target.name}.{os.getpid()}.copying")
        shutil.copyfile(path, temporary)
        temporary.replace(target)

    with (destination / ".cache.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stage, [path for path in source.iterdir() if path.is_file()]))
    print(f"PyTorch native cache ready: {destination}; {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
