"""Run a small IGR parameter sweep on one UVG-CWI-DQPC frame."""

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
IGR_CODE = REPO_ROOT / "third_party" / "SCUTSurface" / "reconstruction" / "IGR" / "code"
METRIC_DIR = REPO_ROOT / "third_party" / "UVG-CWI-Metric"

sys.path.insert(0, str(METRIC_DIR))
from metrics import eval_pointcloud  # noqa: E402


LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamferL2_old", "chamfer-L1", "chamfer-L2"}
HIGHER_IS_BETTER = {"N_Acc", "N_Comp", "normals", "P_5", "R_5", "F_5", "P_10", "R_10", "F_10", "P_20", "R_20", "F_20"}


VARIANTS = [
    {"name": "igr100_50k_res64_sigma18_lam01", "epochs": 100, "points": 50000, "batch": 8192, "resolution": 64, "sigma": 1.8, "grad_lambda": 0.1},
    {"name": "igr300_100k_res96_sigma08_lam01", "epochs": 300, "points": 100000, "batch": 8192, "resolution": 96, "sigma": 0.8, "grad_lambda": 0.1},
    {"name": "igr300_200k_res128_sigma05_lam005", "epochs": 300, "points": 200000, "batch": 8192, "resolution": 128, "sigma": 0.5, "grad_lambda": 0.05},
    {"name": "igr500_200k_res128_sigma08_lam01", "epochs": 500, "points": 200000, "batch": 8192, "resolution": 128, "sigma": 0.8, "grad_lambda": 0.1},
]


def run(cmd, cwd=REPO_ROOT):
    print(f"[cmd] {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def find_frame(directory: Path, frame: str) -> Path:
    matches = sorted(directory.glob(f"*_{frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def write_config(sequence: str, frame: str, variant: dict) -> str:
    conf_name = f"{sequence}_frame_{frame}_{variant['name']}.conf"
    conf_path = IGR_CODE / "reconstruction" / conf_name
    text = f"""train{{
    input_path = ../data/uvg_kettlebell/points/{sequence}/frame_{frame}.npy
    d_in = 3
    plot_frequency = 100000
    checkpoint_frequency = 100000
    status_frequency = 25
    weight_decay = 0
    learning_rate_schedule = [{{
                                "Type" : "Step",
                                "Initial" : 0.005,
                                "Interval" : 2000,
                                "Factor" : 0.5
                                }}]
    network_class = model.network.ImplicitNet
}}

plot{{
    resolution = {variant['resolution']}
    mc_value = 0.0
    is_uniform_grid = True
    verbose = False
    save_html = False
    save_ply = True
    overwrite = True
    connected = True
}}
network{{
    inputs{{
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
        skip_in = [4]
        geometric_init= True
        radius_init = 1
        beta=100
    }}
    sampler{{
        sampler_type = NormalPerPoint
        properties{{
            global_sigma = {variant['sigma']}
            }}
        }}
    loss{{
        lambda = {variant['grad_lambda']}
        normals_lambda = 0
    }}
}}
"""
    conf_path.write_text(text, encoding="ascii")
    return conf_name


def latest_eval_mesh(expname: str) -> Path:
    exp_dir = IGR_CODE.parent / "exps" / expname
    meshes = sorted(exp_dir.glob("*/evaluation/**/*.ply"))
    if not meshes:
        raise FileNotFoundError(f"No IGR eval mesh found in {exp_dir}")
    return meshes[-1]


def compare(metric: str, baseline: float, method: float):
    if not math.isfinite(float(baseline)) or not math.isfinite(float(method)):
        return float("nan"), False
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


def main():
    parser = argparse.ArgumentParser(description="Run IGR parameter sweep")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sample-points", type=int, default=200000)
    args = parser.parse_args()

    out_root = REPO_ROOT / "results" / "method_outputs" / "igr_sweep" / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / "igr_sweep"
    out_root.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)

    cg = find_frame(args.dataset_root / args.sequence / "cg" / "15fps", args.frame)
    he = find_frame(args.dataset_root / args.sequence / "he" / "15fps", args.frame)
    baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])

    rows = []
    comparisons = []

    for variant in VARIANTS:
        expname = f"{args.sequence}_frame_{args.frame}_{variant['name']}"
        print(f"\n=== IGR sweep {expname} ===", flush=True)
        conf_name = write_config(args.sequence, args.frame, variant)

        run([
            sys.executable,
            "scripts/prepare_uvg_frame_for_igr.py",
            "--sequence",
            args.sequence,
            "--frame",
            args.frame,
            "--dataset-root",
            args.dataset_root,
            "--max-points",
            variant["points"],
        ])

        run([
            sys.executable,
            "reconstruction/run.py",
            "--points_batch",
            variant["batch"],
            "--nepoch",
            variant["epochs"],
            "--conf",
            conf_name,
            "--expname",
            expname,
            "--gpu",
            "0",
        ], cwd=IGR_CODE)

        run([
            sys.executable,
            "reconstruction/run.py",
            "--eval",
            "--checkpoint",
            "latest",
            "--conf",
            conf_name,
            "--expname",
            expname,
            "--gpu",
            "0",
        ], cwd=IGR_CODE)

        mesh = latest_eval_mesh(expname)
        out_mesh = out_root / f"frame_{args.frame}_{variant['name']}_mesh.ply"
        out_points = out_root / f"frame_{args.frame}_{variant['name']}.ply"
        run([
            sys.executable,
            "scripts/denormalize_igr_output.py",
            "--mesh",
            mesh,
            "--normalization",
            IGR_CODE.parent / "data" / "uvg_kettlebell" / "points" / args.sequence / f"frame_{args.frame}_normalization.json",
            "--out-mesh",
            out_mesh,
            "--out-points",
            out_points,
            "--sample-points",
            args.sample_points,
            "--seed",
            args.frame,
        ])

        metrics = eval_pointcloud(str(out_points), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": variant["name"], "sequence": args.sequence, "frame": args.frame, "pred_file": str(out_points), "gt_file": str(he), **metrics})
        for metric, baseline_value in baseline.items():
            delta, improved = compare(metric, baseline_value, metrics[metric])
            comparisons.append({
                "variant": variant["name"],
                "metric": metric,
                "baseline": baseline_value,
                "igr": metrics[metric],
                "delta_for_better": delta,
                "igr_improved": improved,
            })

    metric_names = list(baseline.keys())
    per_frame_csv = metric_root / f"frame_{args.frame}_per_variant_metrics.csv"
    comparison_csv = metric_root / f"frame_{args.frame}_baseline_vs_igr_sweep.csv"
    config_json = metric_root / f"frame_{args.frame}_sweep_config.json"
    with per_frame_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with comparison_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "metric", "baseline", "igr", "delta_for_better", "igr_improved"])
        writer.writeheader()
        writer.writerows(comparisons)
    config_json.write_text(json.dumps({"sequence": args.sequence, "frame": args.frame, "variants": VARIANTS}, indent=2), encoding="ascii")

    print(f"Per-variant metrics: {per_frame_csv}")
    print(f"Comparison metrics: {comparison_csv}")


if __name__ == "__main__":
    main()
