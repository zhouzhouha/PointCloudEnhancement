"""Run GQE-Net color enhancement on selected UVG-CWI-DQPC frames.

GQE-Net enhances color attributes. This adapter stages UVG CG frames in the
filename layout expected by the official evaluation script, runs the released
Y/U/V checkpoints, copies the predicted XYZRGB PLY to the benchmark output
folder, and evaluates it with the same metric runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
GQE_ROOT = REPO_ROOT / "third_party" / "enhancement" / "GQE-Net"
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
                writer.writerow({"method": method, "metric": metric, "mean": np.nanmean(values), "std": np.nanstd(values), "count": len(values)})


def main():
    parser = argparse.ArgumentParser(description="Run GQE-Net selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--method-name", default="gqenet")
    parser.add_argument("--test-batch-size", type=int, default=8)
    args = parser.parse_args()

    model_paths = [
        GQE_ROOT / "pths" / "final_2023" / "GQE-Net" / "2023-07-25" / "y" / "model_6.pth",
        GQE_ROOT / "pths" / "final_2023" / "GQE-Net" / "2023-07-28" / "u" / "model_55.pth",
        GQE_ROOT / "pths" / "final_2023" / "GQE-Net" / "2023-07-31" / "v" / "model_92.pth",
    ]
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    work_root = REPO_ROOT / "results" / "work" / args.method_name / args.sequence / "15fps"
    ori_root = work_root / "ori"
    rec_root = work_root / "rec"
    pred_root = work_root / "pred"
    log_root = work_root / "logs"
    out_root = REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [ori_root, rec_root, pred_root, log_root, out_root, metric_root]:
        path.mkdir(parents=True, exist_ok=True)

    test_txt = work_root / "testFile.txt"
    rows = []
    comparisons = []
    counts = []
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"

    rec_names = []
    for frame in args.frames:
        cg = find_frame(cg_dir, frame)
        points, colors = read_uvg_xyzrgb(cg)
        ori_name = f"frame_{frame}.ply"
        rec_name = f"frame_{frame}_r01.ply"
        write_xyzrgb_ply(ori_root / ori_name, points, colors)
        write_xyzrgb_ply(rec_root / rec_name, points, colors)
        rec_names.append(rec_name)
        counts.append({"frame": frame, "input_points": len(points), "output_points": len(points), "has_color": True})

    test_txt.write_text("\n".join(rec_names) + "\n", encoding="ascii")

    run(
        [
            sys.executable,
            "main_mix.py",
            "--log_path_test",
            log_root,
            "--test_ply_txt",
            test_txt,
            "--test_ori_ply",
            ori_root,
            "--test_rec_ply",
            rec_root,
            "--model1_path",
            model_paths[0],
            "--model2_path",
            model_paths[1],
            "--model3_path",
            model_paths[2],
            "--pred_path",
            pred_root,
            "--eval",
            "1",
            "--test_batch_size",
            args.test_batch_size,
        ],
        cwd=GQE_ROOT,
    )

    for frame, rec_name in zip(args.frames, rec_names):
        cg = find_frame(cg_dir, frame)
        he = find_frame(he_dir, frame)
        pred = pred_root / rec_name
        out = out_root / f"frame_{frame}.ply"
        if not pred.exists():
            raise FileNotFoundError(pred)
        shutil.copy2(pred, out)
        print(f"{frame}: output={out}", flush=True)

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
