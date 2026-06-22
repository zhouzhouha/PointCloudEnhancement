"""Run PD-LTS denoising on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import run_mag_selected_frames as common


PDLTS_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PD-LTS"
PDLTS_CKPTS = {
    "light": PDLTS_ROOT / "product" / "ckpt" / "Denoiseflow-light-FBM.ckpt",
    "heavy": PDLTS_ROOT / "product" / "ckpt" / "Denoiseflow-heavy-FBM.ckpt",
}


def run_pdlts(
    input_xyz: Path,
    output_xyz: Path,
    variant: str,
    patch_size: int,
    niters: int,
    seed_k: int,
    seed: int,
) -> None:
    ckpt = PDLTS_CKPTS[variant]
    script = PDLTS_ROOT / "models" / f"model_{variant}" / "denoise.py"
    cmd = [
        sys.executable,
        script,
        "--input",
        input_xyz,
        "--output",
        output_xyz,
        "--patch_size",
        patch_size,
        "--niters",
        niters,
        "--seed_k",
        seed_k,
        "--seed",
        seed,
        "--device",
        "cuda",
        "--ckpt",
        ckpt,
    ]
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(PDLTS_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PD-LTS selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pdlts_light_fbm")
    parser.add_argument("--variant", choices=sorted(PDLTS_CKPTS), default="light")
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument("--seed-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--results-root", type=Path, default=common.REPO_ROOT / "results")
    args = parser.parse_args()

    ckpt = PDLTS_CKPTS[args.variant]
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    work_root = args.results_root / "work" / args.method_name / args.sequence / "15fps"
    input_root = work_root / "input"
    pdlts_root = work_root / "pdlts_results"
    out_root = args.results_root / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = args.results_root / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [input_root, pdlts_root, out_root, metric_root]:
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
        output_xyz = pdlts_root / f"{stem}.denoised.xyz"
        common.write_xyz(input_xyz, source_points)

        run_pdlts(input_xyz, output_xyz, args.variant, args.patch_size, args.niters, args.seed_k, args.seed)

        denoised_points = np.loadtxt(output_xyz).astype(np.float32)
        denoised_colors = common.transfer_nearest_colors(source_points, source_colors, denoised_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, denoised_points, denoised_colors)
        counts.append(
            {
                "frame": frame,
                "input_points": len(source_points),
                "output_points": len(denoised_points),
                "has_color": True,
            }
        )
        print(f"{frame}: {len(source_points)} -> {len(denoised_points)} points, output={out}", flush=True)

        baseline = common.eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        method_metrics = common.eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **method_metrics})
        for metric in baseline:
            delta, improved = common.compare_metric(metric, baseline[metric], method_metrics[metric])
            comparisons.append(
                {
                    "frame": frame,
                    "metric": metric,
                    "baseline": baseline[metric],
                    args.method_name: method_metrics[metric],
                    "delta_for_better": delta,
                    f"{args.method_name}_improved": improved,
                }
            )

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
