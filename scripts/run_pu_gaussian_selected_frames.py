"""Run PU-Gaussian upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

import run_mag_selected_frames as common


PUGAUSSIAN_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PU-Gaussian"
PUGAUSSIAN_CKPTS = {
    "pu1k": PUGAUSSIAN_ROOT / "pretrained_model" / "pu_gaussian_pu1k_Best.pth",
    "pugan": PUGAUSSIAN_ROOT / "pretrained_model" / "pu_gaussian_pugan_Best.pth",
}


def run_official_inference(
    input_path: Path,
    raw_output_path: Path,
    ckpt: Path,
    patch_size: int,
    patch_rate: int,
    up_ratio: int,
    num_samples: int,
    distribution: str,
    training_stage: int,
) -> None:
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PUGAUSSIAN_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [
        sys.executable,
        "-u",
        "infer.py",
        "--inference_input_path",
        input_path,
        "--inference_output_path",
        raw_output_path,
        "--ckpt",
        ckpt,
        "--return_color",
        "--patch_size",
        patch_size,
        "--patch_rate",
        patch_rate,
        "--r",
        up_ratio,
        "--num_samples",
        num_samples,
        "--distribution",
        distribution,
        "--training_stage",
        training_stage,
    ]
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    last_error = None
    for attempt in range(1, 4):
        if attempt > 1:
            wait_seconds = 30 * attempt
            print(f"[retry] PU-Gaussian inference attempt {attempt}/3 after {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
        result = subprocess.run([str(x) for x in cmd], cwd=str(PUGAUSSIAN_ROOT), env=env)
        if result.returncode == 0:
            return
        last_error = subprocess.CalledProcessError(result.returncode, [str(x) for x in cmd])
    raise last_error


def read_open3d_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        raise RuntimeError(f"PU-Gaussian wrote an empty point cloud: {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PU-Gaussian selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pu_gaussian_pu1k_4x")
    parser.add_argument("--checkpoint", choices=sorted(PUGAUSSIAN_CKPTS), default="pu1k")
    parser.add_argument("--patch-size", type=int, default=10000)
    parser.add_argument("--patch-rate", type=int, default=3)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--distribution", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--training-stage", type=int, default=2)
    args = parser.parse_args()

    ckpt = PUGAUSSIAN_CKPTS[args.checkpoint]
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    out_root = common.REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    raw_root = common.REPO_ROOT / "results" / "work" / args.method_name / args.sequence / "15fps" / "raw_pu_gaussian"
    metric_root = common.REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [out_root, raw_root, metric_root]:
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
        raw_output = raw_root / f"frame_{frame}_raw.ply"
        run_official_inference(
            input_path=cg,
            raw_output_path=raw_output,
            ckpt=ckpt,
            patch_size=args.patch_size,
            patch_rate=args.patch_rate,
            up_ratio=args.up_ratio,
            num_samples=args.num_samples,
            distribution=args.distribution,
            training_stage=args.training_stage,
        )
        enhanced_points = read_open3d_points(raw_output)
        enhanced_colors = common.transfer_nearest_colors(source_points, source_colors, enhanced_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, enhanced_points, enhanced_colors)
        counts.append({"frame": frame, "input_points": len(source_points), "output_points": len(enhanced_points), "has_color": True})
        print(f"{frame}: {len(source_points)} -> {len(enhanced_points)} points, output={out}", flush=True)

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
