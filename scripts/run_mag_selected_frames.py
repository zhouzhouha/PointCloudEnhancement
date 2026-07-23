"""Run MAG denoising on selected UVG-CWI-DQPC frames.

Adapter around MAG's official `test_large.py`. It converts UVG CG frames to
XYZ, runs the pretrained MAG checkpoint with official large-cloud inference,
then transfers RGB from the original CG frame by nearest neighbor (`k=1`).
"""

import argparse
import csv
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
MAG_ROOT = REPO_ROOT / "third_party" / "enhancement" / "MAG"
MAG_CKPT = MAG_ROOT / "pretrained" / "ckpt.pt"
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
        points = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 3), dtype=np.uint8)
        for idx in range(vertex_count):
            x, y, z, r, g, b = record.unpack(handle.read(record.size))
            points[idx] = (x, y, z)
            colors[idx] = (r, g, b)
    return points, colors


def write_xyz(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt="%.10f")


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


def run(cmd, cwd):
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


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
                finite_values = values[np.isfinite(values)]
                mean = float(np.mean(finite_values)) if finite_values.size else float("nan")
                std = float(np.std(finite_values)) if finite_values.size else float("nan")
                writer.writerow({"method": method, "metric": metric, "mean": mean, "std": std, "count": len(values)})


def main():
    parser = argparse.ArgumentParser(description="Run MAG selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--method-name", default="mag")
    parser.add_argument("--cluster-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    if not MAG_CKPT.exists():
        raise FileNotFoundError(MAG_CKPT)

    work_root = REPO_ROOT / "results" / "work" / args.method_name / args.sequence / "15fps"
    input_root = work_root / "input"
    mag_root = work_root / "mag_results"
    out_root = REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [input_root, mag_root, out_root, metric_root]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    comparisons = []
    counts = []
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"

    for frame in args.frames:
        cg = find_frame(cg_dir, frame)
        he = find_frame(he_dir, frame)
        source_points, source_colors = read_uvg_xyzrgb(cg)
        stem = f"{args.sequence}_frame_{frame}"
        input_xyz = input_root / f"{stem}.xyz"
        output_xyz = mag_root / f"{stem}.denoised.xyz"
        write_xyz(input_xyz, source_points)

        run(
            [
                sys.executable,
                "test_large.py",
                "--ckpt",
                MAG_CKPT,
                "--input_xyz",
                input_xyz,
                "--output_xyz",
                output_xyz,
                "--device",
                "cuda",
                "--seed",
                args.seed,
                "--cluster_size",
                args.cluster_size,
            ],
            cwd=MAG_ROOT,
        )

        denoised_points = np.loadtxt(output_xyz).astype(np.float32)
        denoised_colors = transfer_nearest_colors(source_points, source_colors, denoised_points)
        out = out_root / f"frame_{frame}.ply"
        write_xyzrgb_ply(out, denoised_points, denoised_colors)
        counts.append({"frame": frame, "input_points": len(source_points), "output_points": len(denoised_points), "has_color": True})
        print(f"{frame}: {len(source_points)} -> {len(denoised_points)} points, output={out}", flush=True)

        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        method_metrics = eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **method_metrics})
        for metric in baseline:
            delta, improved = compare_metric(metric, baseline[metric], method_metrics[metric])
            comparisons.append({"frame": frame, "metric": metric, "baseline": baseline[metric], args.method_name: method_metrics[metric], "delta_for_better": delta, f"{args.method_name}_improved": improved})

    metric_names = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    with (metric_root / "per_frame_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with (metric_root / f"baseline_vs_{args.method_name}_by_frame.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "metric", "baseline", args.method_name, "delta_for_better", f"{args.method_name}_improved"])
        writer.writeheader()
        writer.writerows(comparisons)
    with (metric_root / "point_counts.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "output_points", "has_color"])
        writer.writeheader()
        writer.writerows(counts)
    write_summary(rows, metric_root / "summary_metrics.csv")
    (metric_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


if __name__ == "__main__":
    main()
