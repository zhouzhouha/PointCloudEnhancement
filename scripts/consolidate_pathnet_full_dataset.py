"""Consolidate resumable PathNet batches without duplicating large PLY files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


SEQUENCES = (
    "BlueSpeech", "BlueVolley", "BouncingBlue", "FitFluencer",
    "GoodVision", "Mannequin", "OrangeKettlebell", "PinkNoir",
    "TicTacToe", "TrumanShow", "VictoryHeart", "VirtualLife",
)


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method", default="pathnet_chunked_full")
    parser.add_argument(
        "--batch-prefixes", nargs="+", default=["pathnet_chunked_full_batch_"],
        help="Immutable batch-directory prefixes to merge in the given order",
    )
    parser.add_argument("--expected-total", type=int)
    args = parser.parse_args()

    output_root = args.results_root / "method_outputs"
    metric_root = args.results_root / "uvg_cwi_dqpc"
    consolidated_root = output_root / args.method
    if consolidated_root.exists() or consolidated_root.is_symlink():
        raise FileExistsError(f"refusing to modify existing consolidated output: {consolidated_root}")
    for sequence in SEQUENCES:
        existing_metrics = metric_root / sequence / args.method
        if existing_metrics.exists() or existing_metrics.is_symlink():
            raise FileExistsError(f"refusing to modify existing metrics: {existing_metrics}")

    batch_dirs: list[Path] = []
    for prefix in args.batch_prefixes:
        batch_dirs.extend(sorted(output_root.glob(f"{prefix}*")))
    if not batch_dirs:
        raise FileNotFoundError(f"no batch directories found for prefixes {args.batch_prefixes}")

    total = 0
    for sequence in SEQUENCES:
        sources: dict[str, Path] = {}
        metric_rows: list[dict[str, str]] = []
        fieldnames: list[str] | None = None
        for batch_dir in batch_dirs:
            for path in sorted((batch_dir / sequence / "15fps").glob("frame_*.ply")):
                frame = frame_id(path)
                if frame in sources:
                    raise RuntimeError(f"duplicate {sequence} frame {frame}: {sources[frame]} and {path}")
                sources[frame] = path.resolve()

            csv_path = metric_root / sequence / batch_dir.name / "per_frame_metrics.csv"
            if csv_path.exists():
                with csv_path.open(newline="", encoding="ascii") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = fieldnames or list(reader.fieldnames or [])
                    for row in reader:
                        row["method"] = "cg_baseline" if row["method"] == "cg_baseline" else args.method
                        metric_rows.append(row)

        target = output_root / args.method / sequence / "15fps"
        target.mkdir(parents=True, exist_ok=True)
        for frame, source in sorted(sources.items()):
            link = target / f"frame_{frame}.ply"
            if link.is_symlink() and link.resolve() == source:
                continue
            if link.exists() or link.is_symlink():
                raise RuntimeError(f"refusing to replace {link}")
            link.symlink_to(os.path.relpath(source, start=target))

        for row in metric_rows:
            if row["method"] == args.method:
                row["pred_file"] = str(target / f"frame_{row['frame']}.ply")

        expected_rows = 2 * len(sources)
        if len(metric_rows) != expected_rows:
            raise RuntimeError(f"{sequence}: {len(sources)} outputs but {len(metric_rows)} metric rows")
        out_metrics = metric_root / sequence / args.method / "per_frame_metrics.csv"
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        if out_metrics.exists():
            raise FileExistsError(f"refusing to overwrite {out_metrics}")
        with out_metrics.open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(metric_rows, key=lambda row: (row["frame"], row["method"])))
        print(f"{sequence}: outputs={len(sources)} metric_rows={len(metric_rows)}")
        total += len(sources)

    if args.expected_total is not None and total != args.expected_total:
        raise RuntimeError(f"expected {args.expected_total} outputs but consolidated {total}")
    print(f"total_outputs={total}")


if __name__ == "__main__":
    main()
