"""Run one resumable PathNet frame batch for one UVG-CWI-DQPC sequence."""

import argparse
import subprocess
import sys
from pathlib import Path

import run_full_dataset_method as full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method-prefix", default="pathnet_chunked_full")
    args = parser.parse_args()

    frames = full.common_frames(args.dataset_root, args.sequence)
    start = args.batch_index * args.batch_size
    selected = frames[start:start + args.batch_size]
    if not selected:
        print(
            f"[pathnet-batch] sequence={args.sequence} batch={args.batch_index} "
            f"has no frames (sequence total={len(frames)}); exiting"
        )
        return

    method_name = f"{args.method_prefix}_batch_{args.batch_index:02d}"
    cmd = [
        sys.executable,
        "-u",
        str(full.REPO_ROOT / "scripts" / "run_pathnet_selected_frames.py"),
        "--sequence",
        args.sequence,
        "--frames",
        *selected,
        "--dataset-root",
        str(args.dataset_root),
        "--results-root",
        str(args.results_root),
        "--method-name",
        method_name,
        "--chunk-size",
        "50000",
        "--batch-size",
        "512",
        "--knn",
        "128",
        "--iterations",
        "2",
    ]
    print(
        f"[pathnet-batch] sequence={args.sequence} batch={args.batch_index} "
        f"frames={selected[0]}..{selected[-1]} count={len(selected)}",
        flush=True,
    )
    subprocess.run(cmd, cwd=full.REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
