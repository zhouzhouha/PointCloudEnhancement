"""Merge PointCleanNet full170 chunk outputs into one benchmark folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

import run_mag_selected_frames as common


def read_csv(path: Path):
    with path.open("r", newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge PointCleanNet full170 chunk outputs")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--source-prefix", default="pointcleannet_full170_chunk_")
    parser.add_argument("--target-method", default="pointcleannet_full170")
    parser.add_argument("--chunks", type=int, default=17)
    args = parser.parse_args()

    repo = common.REPO_ROOT
    target_out = repo / "results" / "method_outputs" / args.target_method / args.sequence / "15fps"
    target_metrics = repo / "results" / "uvg_cwi_dqpc" / args.sequence / args.target_method
    target_out.mkdir(parents=True, exist_ok=True)
    target_metrics.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_counts = []
    all_comparisons = []
    copied = []

    for chunk in range(args.chunks):
        method = f"{args.source_prefix}{chunk}"
        src_out = repo / "results" / "method_outputs" / method / args.sequence / "15fps"
        src_metrics = repo / "results" / "uvg_cwi_dqpc" / args.sequence / method
        if not src_metrics.exists():
            raise FileNotFoundError(src_metrics)
        for ply in sorted(src_out.glob("frame_*.ply")):
            dest = target_out / ply.name
            shutil.copy2(ply, dest)
            copied.append(str(dest))
        all_metrics.extend(read_csv(src_metrics / "per_frame_metrics.csv"))
        all_counts.extend(read_csv(src_metrics / "point_counts.csv"))
        comparison_files = sorted(src_metrics.glob("baseline_vs_*_by_frame.csv"))
        if comparison_files:
            all_comparisons.extend(read_csv(comparison_files[0]))

    all_metrics = sorted(all_metrics, key=lambda row: (row.get("frame", ""), row.get("method", "")))
    all_counts = sorted(all_counts, key=lambda row: row.get("frame", ""))
    all_comparisons = sorted(all_comparisons, key=lambda row: (row.get("frame", ""), row.get("metric", "")))
    write_csv(target_metrics / "per_frame_metrics.csv", all_metrics)
    write_csv(target_metrics / "point_counts.csv", all_counts)
    write_csv(target_metrics / f"baseline_vs_{args.target_method}_by_frame.csv", all_comparisons)

    normalized_rows = []
    for row in all_metrics:
        normalized = dict(row)
        if normalized.get("method", "").startswith(args.source_prefix):
            normalized["method"] = args.target_method
        normalized_rows.append(normalized)
    common.write_summary(normalized_rows, target_metrics / "summary_metrics.csv")
    (target_metrics / "run_config.json").write_text(
        json.dumps({
            "sequence": args.sequence,
            "source_prefix": args.source_prefix,
            "target_method": args.target_method,
            "chunks": args.chunks,
            "copied_files": len(copied),
            "mean_points": float(np.mean([int(row["output_points"]) for row in all_counts])) if all_counts else None,
        }, indent=2),
        encoding="ascii",
    )
    print(f"Merged {len(copied)} PLY files into {target_out}")
    print(f"Wrote merged metrics to {target_metrics}")


if __name__ == "__main__":
    main()
