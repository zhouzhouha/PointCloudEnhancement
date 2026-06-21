"""Run P2P-Bridge object denoising on UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_mag_selected_frames as common


P2P_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "P2P-Bridge"
P2P_CKPT = P2P_ROOT / "pretrained" / "PVDS_PUNet" / "latest.pth"
P2P_DEPS = P2P_ROOT / "python_deps"


def run_p2p(input_xyz: Path, output_xyz: Path, steps: int, k: int, seed: int) -> None:
    env = os.environ.copy()
    extra_pythonpath = [
        str(P2P_DEPS),
        str(P2P_ROOT),
        str(P2P_ROOT / "third_party" / "openpoints" / "cpp" / "pointnet2_batch"),
        str(P2P_ROOT / "metrics" / "emd_assignment"),
        str(P2P_ROOT / "metrics" / "PyTorchEMD"),
        str(P2P_ROOT / "metrics" / "chamfer3D"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(path for path in extra_pythonpath if path)
    cmd = [
        sys.executable,
        "denoise_object.py",
        "--data_path",
        input_xyz,
        "--save_path",
        output_xyz,
        "--model_path",
        P2P_CKPT,
        "--steps",
        steps,
        "--k",
        k,
        "--seed",
        seed,
        "--gpu",
        "cuda:0",
    ]
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(P2P_ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2P-Bridge selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="p2p_bridge_pvds_punet")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=common.REPO_ROOT / "results")
    args = parser.parse_args()

    if not P2P_CKPT.exists():
        raise FileNotFoundError(P2P_CKPT)

    work_root = args.results_root / "work" / args.method_name / args.sequence / "15fps"
    input_root = work_root / "input"
    p2p_root = work_root / "p2p_bridge_results"
    out_root = args.results_root / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = args.results_root / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [input_root, p2p_root, out_root, metric_root]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    comparisons = []
    counts = []
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"

    for frame in args.frames:
        cg = common.find_frame(cg_dir, frame)
        he = common.find_frame(he_dir, frame)
        source_points, source_colors = common.read_uvg_xyzrgb(cg)
        stem = f"{args.sequence}_frame_{frame}"
        input_xyz = input_root / f"{stem}.xyz"
        output_xyz = p2p_root / f"{stem}.denoised.xyz"
        common.write_xyz(input_xyz, source_points)

        run_p2p(input_xyz, output_xyz, args.steps, args.k, args.seed)

        denoised_points = np.loadtxt(output_xyz).astype(np.float32)
        denoised_colors = common.transfer_nearest_colors(source_points, source_colors, denoised_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, denoised_points, denoised_colors)
        counts.append({"frame": frame, "input_points": len(source_points), "output_points": len(denoised_points), "has_color": True})
        print(f"{frame}: {len(source_points)} -> {len(denoised_points)} points, output={out}", flush=True)

        baseline = common.eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        method_metrics = common.eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **method_metrics})
        for metric in baseline:
            delta, improved = common.compare_metric(metric, baseline[metric], method_metrics[metric])
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
    common.write_summary(rows, metric_root / "summary_metrics.csv")
    (metric_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


if __name__ == "__main__":
    main()
