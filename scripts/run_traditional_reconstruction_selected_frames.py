"""Run traditional non-deep-learning reconstruction on selected UVG frames.

Methods:
- SPSR / Screened Poisson Surface Reconstruction
- BPA / Ball Pivoting Algorithm
- Alpha shapes

The adapter preserves the benchmark protocol: reconstruct geometry from CG,
sample back to the input point count, transfer RGB from CG by nearest neighbor
`k=1`, and evaluate against HE with the UVG-CWI metric runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
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


def read_uvg_xyzrgb(ply_path: Path):
    with ply_path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {ply_path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {ply_path}")

        record = struct.Struct("<dddBBB")
        points = np.empty((vertex_count, 3), dtype=np.float64)
        colors = np.empty((vertex_count, 3), dtype=np.uint8)
        for idx in range(vertex_count):
            x, y, z, r, g, b = record.unpack(handle.read(record.size))
            points[idx] = (x, y, z)
            colors[idx] = (r, g, b)
    return points, colors


def write_xyzrgb_ply(path: Path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            handle.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def transfer_nearest_colors(source_points, source_colors, target_points):
    tree = cKDTree(source_points)
    _distances, indices = tree.query(target_points, k=1, workers=8)
    return source_colors[indices]


def estimate_normals(pcd, knn: int):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=knn)
    return pcd


def auto_bpa_radii(pcd):
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_dist = float(np.mean(distances))
    return [avg_dist, avg_dist * 2.0, avg_dist * 4.0]


def auto_alpha(pcd):
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(np.mean(distances) * 3.0)


def reconstruct_points(source_points, args):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_points)
    estimate_normals(pcd, args.knn)

    method = args.reconstruction_method.lower()
    if method == "spsr":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=args.poisson_depth,
            width=0,
            scale=args.poisson_scale,
            linear_fit=args.poisson_linear_fit,
        )
        if args.density_quantile > 0:
            densities = np.asarray(densities)
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, args.density_quantile))
    elif method == "bpa":
        radii = args.bpa_radii if args.bpa_radii else auto_bpa_radii(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
    elif method == "alpha_shape":
        alpha = args.alpha if args.alpha > 0 else auto_alpha(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    else:
        raise ValueError(f"Unknown traditional method: {args.reconstruction_method}")

    if len(mesh.triangles) == 0:
        raise RuntimeError(f"{args.reconstruction_method} produced an empty mesh")
    sampled = mesh.sample_points_uniformly(number_of_points=len(source_points))
    target_points = np.asarray(sampled.points)
    if len(target_points) == 0:
        raise RuntimeError(f"{args.reconstruction_method} sampled zero output points")
    return target_points, len(mesh.vertices), len(mesh.triangles)


def compare_metric(metric: str, baseline: float, method: float):
    if not math.isfinite(float(baseline)) or not math.isfinite(float(method)):
        return float("nan"), False
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


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


def main():
    parser = argparse.ArgumentParser(description="Run traditional reconstruction selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--reconstruction-method", choices=["spsr", "bpa", "alpha_shape"], default="spsr")
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--knn", type=int, default=30)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-scale", type=float, default=1.1)
    parser.add_argument("--poisson-linear-fit", action="store_true")
    parser.add_argument("--density-quantile", type=float, default=0.01)
    parser.add_argument("--bpa-radii", type=float, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    if args.method_name is None:
        args.method_name = f"traditional_{args.reconstruction_method}"

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
        source_points, source_colors = read_uvg_xyzrgb(cg)
        print(f"[{args.method_name}] frame={frame} input_points={len(source_points)}", flush=True)
        target_points, mesh_vertices, mesh_triangles = reconstruct_points(source_points, args)
        target_colors = transfer_nearest_colors(source_points, source_colors, target_points)
        out = out_root / f"frame_{frame}.ply"
        write_xyzrgb_ply(out, target_points, target_colors)
        counts.append(
            {
                "frame": frame,
                "input_points": len(source_points),
                "output_points": len(target_points),
                "mesh_vertices": mesh_vertices,
                "mesh_triangles": mesh_triangles,
                "has_color": True,
            }
        )
        print(
            f"{frame}: {len(source_points)} -> {len(target_points)} points, "
            f"mesh={mesh_vertices}v/{mesh_triangles}t output={out}",
            flush=True,
        )

        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        reconstructed = eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **reconstructed})
        for metric in baseline:
            delta, improved = compare_metric(metric, baseline[metric], reconstructed[metric])
            comparisons.append(
                {
                    "frame": frame,
                    "metric": metric,
                    "baseline": baseline[metric],
                    args.method_name: reconstructed[metric],
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
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "output_points", "mesh_vertices", "mesh_triangles", "has_color"])
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
