"""Summarize full-dataset method results by sequence first, then dataset."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


GEOMETRY_SKIP = {"method", "sequence", "frame", "pred_file", "gt_file"}
TEXTURE_SKIP = {"method", "sequence", "frame", "pred_file", "ref_file"}
HIGHER_IS_BETTER = {
    "P_5",
    "P_10",
    "P_20",
    "R_5",
    "R_10",
    "R_20",
    "F_5",
    "F_10",
    "F_20",
    "N_Acc",
    "N_Comp",
    "normals",
    "y_psnr",
    "u_psnr",
    "v_psnr",
    "yuv_psnr_mean",
    "projection_ssim_mean",
    "pcqm",
}


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def metric_delta(metric: str, baseline: float, method: float) -> float:
    if math.isnan(baseline) or math.isnan(method):
        return math.nan
    if metric in HIGHER_IS_BETTER:
        return method - baseline
    return baseline - method


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open(encoding="ascii", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(rows: list[dict[str, str]], skip: set[str], out_prefix: Path) -> None:
    if not rows:
        print(f"[skip] no rows for {out_prefix}")
        return
    metrics = [key for key in rows[0] if key not in skip]
    by_method_sequence: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method_sequence[(row["method"], row["sequence"])].append(row)

    sequence_rows = []
    for (method, sequence), grouped in sorted(by_method_sequence.items()):
        out = {"method": method, "sequence": sequence, "frame_count": len(grouped)}
        for metric in metrics:
            out[metric] = mean([as_float(row.get(metric, "")) for row in grouped])
        sequence_rows.append(out)

    by_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in sequence_rows:
        by_method[str(row["method"])].append(row)

    dataset_rows = []
    for method, grouped in sorted(by_method.items()):
        out = {"method": method, "sequence_count": len(grouped)}
        for metric in metrics:
            out[metric] = mean([float(row[metric]) for row in grouped])
        dataset_rows.append(out)

    delta_rows = []
    baselines = {
        str(row["sequence"]): row
        for row in sequence_rows
        if row["method"] == "cg_baseline"
    }
    for row in sequence_rows:
        method = str(row["method"])
        if method == "cg_baseline":
            continue
        baseline = baselines.get(str(row["sequence"]))
        if baseline is None:
            continue
        out = {"method": method, "sequence": row["sequence"], "frame_count": row["frame_count"]}
        for metric in metrics:
            out[f"d{metric}"] = metric_delta(metric, float(baseline[metric]), float(row[metric]))
        delta_rows.append(out)

    by_delta_method: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in delta_rows:
        by_delta_method[str(row["method"])].append(row)
    dataset_delta_rows = []
    delta_metrics = [f"d{metric}" for metric in metrics]
    for method, grouped in sorted(by_delta_method.items()):
        out = {"method": method, "sequence_count": len(grouped)}
        for metric in delta_metrics:
            out[metric] = mean([float(row[metric]) for row in grouped])
        dataset_delta_rows.append(out)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_prefix.with_name(out_prefix.name + "_per_sequence.csv"), sequence_rows)
    write_csv(out_prefix.with_name(out_prefix.name + "_dataset_mean.csv"), dataset_rows)
    if delta_rows:
        write_csv(out_prefix.with_name(out_prefix.name + "_delta_per_sequence.csv"), delta_rows)
        write_csv(out_prefix.with_name(out_prefix.name + "_delta_dataset_mean.csv"), dataset_delta_rows)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", nargs="*", type=Path, default=[])
    parser.add_argument("--texture", nargs="*", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("results/full_dataset/summary"))
    args = parser.parse_args()

    summarize(read_rows(args.geometry), GEOMETRY_SKIP, args.out_dir / "geometry")
    summarize(read_rows(args.texture), TEXTURE_SKIP, args.out_dir / "texture_perceptual")


if __name__ == "__main__":
    main()
