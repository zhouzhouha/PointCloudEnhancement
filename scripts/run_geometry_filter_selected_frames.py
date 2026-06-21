"""Run point-cloud geometry filtering on selected UVG-CWI-DQPC frames.

This baseline avoids mesh reconstruction. It filters the CG point cloud directly
and transfers RGB from the original CG frame to every output point with kNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
METRIC_DIR = REPO_ROOT / "third_party" / "UVG-CWI-Metric"

sys.path.insert(0, str(METRIC_DIR))
from metrics import eval_pointcloud  # noqa: E402


LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamferL2_old", "chamfer-L1", "chamfer-L2"}
HIGHER_IS_BETTER = {
    "N_Acc",
    "N_Comp",
    "normals",
    "P_5",
    "R_5",
    "F_5",
    "P_10",
    "R_10",
    "F_10",
    "P_20",
    "R_20",
    "F_20",
}


def find_frame(directory: Path, frame: str) -> Path:
    matches = sorted(directory.glob(f"*_{frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def compare_metric(metric: str, baseline: float, method: float):
    if not math.isfinite(float(baseline)) or not math.isfinite(float(method)):
        return float("nan"), False
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


def transfer_colors_knn(source_points, source_colors, target_points, k: int):
    if source_colors is None or len(source_colors) == 0:
        return None
    k = max(1, min(int(k), len(source_points)))
    tree = cKDTree(source_points)
    distances, indices = tree.query(target_points, k=k, workers=8)
    if k == 1:
        return source_colors[indices]

    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / distances
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkc->nc", weights, source_colors[indices])


def write_summary(rows, summary_csv: Path):
    metrics = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    methods = sorted({row["method"] for row in rows})
    with summary_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "mean", "std", "count"])
        writer.writeheader()
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            for metric in metrics:
                values = np.array([float(row[metric]) for row in method_rows], dtype=float)
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values),
                        "count": len(values),
                    }
                )


def enhance_frame(cg_path: Path, out_path: Path, args):
    source = o3d.io.read_point_cloud(str(cg_path))
    if not source.has_points():
        raise ValueError(f"No points loaded from {cg_path}")

    source_points = np.asarray(source.points)
    source_colors = np.asarray(source.colors) if source.has_colors() else None

    pcd = source
    if args.sor_neighbors > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.sor_neighbors,
            std_ratio=args.sor_std_ratio,
        )
    if args.radius > 0 and args.radius_neighbors > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=args.radius_neighbors,
            radius=args.radius,
        )
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    target_points = np.asarray(pcd.points)
    if len(target_points) == 0:
        raise ValueError(f"Filtering removed all points from {cg_path}")

    target_colors = transfer_colors_knn(source_points, source_colors, target_points, args.color_knn)
    if target_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(target_colors, 0.0, 1.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False, compressed=False)
    return len(source_points), len(target_points), bool(target_colors is not None)


def main():
    parser = argparse.ArgumentParser(description="Run direct point-cloud geometry filtering")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--method-name", default="geometry_filter_sor")
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--radius-neighbors", type=int, default=16)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--color-knn", type=int, default=1)
    args = parser.parse_args()

    out_root = REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    out_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    rows = []
    comparisons = []
    counts = []
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"

    for frame in args.frames:
        cg = find_frame(cg_dir, frame)
        he = find_frame(he_dir, frame)
        out = out_root / f"frame_{frame}.ply"
        n_in, n_out, has_color = enhance_frame(cg, out, args)
        counts.append({"frame": frame, "input_points": n_in, "output_points": n_out, "has_color": has_color})
        print(f"{frame}: {n_in} -> {n_out} points, color={has_color}, output={out}", flush=True)

        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        filtered = eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **filtered})
        for metric in baseline:
            delta, improved = compare_metric(metric, baseline[metric], filtered[metric])
            comparisons.append(
                {
                    "frame": frame,
                    "metric": metric,
                    "baseline": baseline[metric],
                    args.method_name: filtered[metric],
                    "delta_for_better": delta,
                    f"{args.method_name}_improved": improved,
                }
            )

    metric_names = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    per_frame_csv = metric_root / "per_frame_metrics.csv"
    comparison_csv = metric_root / f"baseline_vs_{args.method_name}_by_frame.csv"
    summary_csv = metric_root / "summary_metrics.csv"
    counts_csv = metric_root / "point_counts.csv"
    config_json = metric_root / "run_config.json"

    with per_frame_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with comparison_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "metric", "baseline", args.method_name, "delta_for_better", f"{args.method_name}_improved"],
        )
        writer.writeheader()
        writer.writerows(comparisons)
    with counts_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "output_points", "has_color"])
        writer.writeheader()
        writer.writerows(counts)
    write_summary(rows, summary_csv)
    config_json.write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")

    print(f"Per-frame metrics: {per_frame_csv}")
    print(f"Comparison metrics: {comparison_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"Point counts: {counts_csv}")


if __name__ == "__main__":
    main()
