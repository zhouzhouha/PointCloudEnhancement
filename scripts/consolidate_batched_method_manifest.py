"""Atomically consolidate an audited immutable batch manifest with symlinks."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-total", type=int, default=2152)
    args = parser.parse_args()

    if not args.run_name or args.run_name in {".", ".."} or Path(args.run_name).name != args.run_name:
        raise ValueError("run-name must be one non-special path component")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    output_root = (args.results_root / "method_outputs").resolve()
    target = output_root / args.run_name
    staging = output_root / f".{args.run_name}.part-{os.getpid()}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite consolidated output: {target}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"staging path already exists: {staging}")

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sequence", "frame", "status", "selected_file"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest requires columns {sorted(required)}")
    if len(rows) != args.expected_total:
        raise RuntimeError(f"expected {args.expected_total} manifest rows, found {len(rows)}")

    sources: dict[tuple[str, str], Path] = {}
    for row in rows:
        key = (row["sequence"], row["frame"])
        if key in sources:
            raise RuntimeError(f"duplicate manifest key: {key}")
        if row["status"] not in {"present", "duplicate"}:
            raise RuntimeError(f"non-consolidatable manifest row {key}: status={row['status']!r}")
        source = Path(row["selected_file"]).resolve(strict=True)
        if not source.is_file() or source.suffix.lower() != ".ply":
            raise ValueError(f"selected source is not a PLY file: {source}")
        if not is_relative_to(source, output_root):
            raise ValueError(f"selected source is outside method_outputs: {source}")
        sources[key] = source

    try:
        for (sequence, frame), source in sorted(sources.items()):
            directory = staging / sequence / "15fps"
            directory.mkdir(parents=True, exist_ok=True)
            link = directory / f"frame_{frame}.ply"
            link.symlink_to(os.path.relpath(source, start=directory))
        staging.replace(target)
    finally:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)

    link_count = sum(1 for path in target.glob("*/15fps/frame_*.ply") if path.is_symlink())
    if link_count != args.expected_total:
        raise RuntimeError(f"post-consolidation check found {link_count} links")
    print(f"consolidated {link_count} immutable links at {target}")


if __name__ == "__main__":
    main()
