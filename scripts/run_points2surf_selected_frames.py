"""Run pretrained Points2Surf on selected UVG-CWI-DQPC frames and evaluate outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
P2S_ROOT = REPO_ROOT / "third_party" / "SCUTSurface" / "reconstruction" / "Points2Surf"
METRIC_DIR = REPO_ROOT / "third_party" / "UVG-CWI-Metric"

sys.path.insert(0, str(METRIC_DIR))
from metrics import eval_pointcloud  # noqa: E402


LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamferL2_old", "chamfer-L1", "chamfer-L2"}
HIGHER_IS_BETTER = {"N_Acc", "N_Comp", "normals", "P_5", "R_5", "F_5", "P_10", "R_10", "F_10", "P_20", "R_20", "F_20"}


def run(cmd, cwd=REPO_ROOT):
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


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
                writer.writerow({
                    "method": method,
                    "metric": metric,
                    "mean": np.nanmean(values),
                    "std": np.nanstd(values),
                    "count": len(values),
                })


def main():
    parser = argparse.ArgumentParser(description="Run Points2Surf selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--input-points", type=int, default=200000)
    parser.add_argument("--sample-points", type=int, default=200000)
    parser.add_argument("--query-grid-resolution", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=3)
    parser.add_argument("--certainty-threshold", type=int, default=13)
    parser.add_argument("--sigma", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=501)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    dataset_name = f"uvg_{args.sequence.lower()}"
    p2s_dataset = P2S_ROOT / "datasets" / dataset_name
    p2s_out = P2S_ROOT / "results" / dataset_name / f"p2s_max_res{args.query_grid_resolution}"
    out_root = REPO_ROOT / "results" / "method_outputs" / "points2surf" / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / "points2surf"
    out_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable,
        "scripts/prepare_uvg_frames_for_points2surf.py",
        "--sequence",
        args.sequence,
        "--frames",
        *args.frames,
        "--dataset-root",
        args.dataset_root,
        "--max-points",
        args.input_points,
        "--out-root",
        p2s_dataset,
    ])

    run([
        sys.executable,
        "-m",
        "source.points_to_surf_eval",
        "--indir",
        p2s_dataset,
        "--outdir",
        p2s_out,
        "--dataset",
        "testset.txt",
        "--reconstruction",
        "True",
        "--query_grid_resolution",
        args.query_grid_resolution,
        "--epsilon",
        args.epsilon,
        "--certainty_threshold",
        args.certainty_threshold,
        "--sigma",
        args.sigma,
        "--modeldir",
        "models",
        "--models",
        "p2s_max",
        "--modelpostfix",
        "_model_249.pth",
        "--batchSize",
        args.batch_size,
        "--workers",
        args.workers,
        "--cache_capacity",
        "2",
        "--gpu_idx",
        "0",
    ], cwd=P2S_ROOT)

    run([
        sys.executable,
        "-c",
        (
            "from source import sdf; "
            f"sdf.implicit_surface_to_mesh_directory('{p2s_out / 'rec' / 'dist_ms'}', "
            f"'{p2s_out / 'rec' / 'query_pts_ms'}', "
            f"'{p2s_out / 'rec' / 'vol'}', "
            f"'{p2s_out / 'rec' / 'mesh'}', "
            f"{args.query_grid_resolution}, {args.sigma}, {args.certainty_threshold}, {args.workers})"
        ),
    ], cwd=P2S_ROOT)

    rows = []
    comparisons = []
    for frame in args.frames:
        stem = f"frame_{frame}"
        mesh = p2s_out / "rec" / "mesh" / f"{stem}.ply"
        if not mesh.exists():
            print(f"[failed] missing Points2Surf mesh for {frame}: {mesh}", flush=True)
            continue
        out_mesh = out_root / f"frame_{frame}_mesh.ply"
        out_points = out_root / f"frame_{frame}.ply"
        run([
            sys.executable,
            "scripts/denormalize_igr_output.py",
            "--mesh",
            mesh,
            "--normalization",
            p2s_dataset / "normalization" / f"{stem}.json",
            "--out-mesh",
            out_mesh,
            "--out-points",
            out_points,
            "--sample-points",
            args.sample_points,
            "--seed",
            frame,
        ])

        cg = find_frame(args.dataset_root / args.sequence / "cg" / "15fps", frame)
        he = find_frame(args.dataset_root / args.sequence / "he" / "15fps", frame)
        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        p2s = eval_pointcloud(str(out_points), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": "points2surf", "sequence": args.sequence, "frame": frame, "pred_file": str(out_points), "gt_file": str(he), **p2s})
        for metric in baseline:
            delta, improved = compare_metric(metric, baseline[metric], p2s[metric])
            comparisons.append({
                "frame": frame,
                "metric": metric,
                "baseline": baseline[metric],
                "points2surf": p2s[metric],
                "delta_for_better": delta,
                "points2surf_improved": improved,
            })

    if not rows:
        raise RuntimeError("Points2Surf produced no evaluable meshes")

    metric_names = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    per_frame_csv = metric_root / "per_frame_metrics.csv"
    comparison_csv = metric_root / "baseline_vs_points2surf_by_frame.csv"
    summary_csv = metric_root / "summary_metrics.csv"
    config_json = metric_root / "run_config.json"
    with per_frame_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with comparison_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "metric", "baseline", "points2surf", "delta_for_better", "points2surf_improved"])
        writer.writeheader()
        writer.writerows(comparisons)
    write_summary(rows, summary_csv)
    config_json.write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")
    print(f"Per-frame metrics: {per_frame_csv}")
    print(f"Comparison metrics: {comparison_csv}")
    print(f"Summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()
