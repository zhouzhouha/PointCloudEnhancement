"""Run SAL on selected UVG-CWI-DQPC frames and evaluate with UVG-CWI/Metric."""

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
SAL_ROOT = REPO_ROOT / "third_party" / "SCUTSurface" / "reconstruction" / "SAL"
SAL_CODE = SAL_ROOT / "code"
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


def write_sal_config(frame: str, expname: str, resolution: int) -> Path:
    config = f"""train{{
    plot_frequency = 0
    preprocess = True
    auto_decoder = False
    latent_size = 0
    expname = {expname}
    dataset_path = ../data/uvg_kettlebell/points/OrangeKettlebell/frame_{frame}.xyz
    adjust_lr = False
    dataset = datasets.recon_dataset.ReconDataSet
    data_split = none

    learning_rate_schedule = [{{ "Type" : "Step",
                              "Initial" : 0.0005,
                               "Interval" : 500,
                                "Factor" : 0.5
                            }},
                            {{
                                "Type" : "Step",
                                "Initial" : 0.001,
                                "Interval" : 500,
                                "Factor" : 0.5
                            }}]
    network_class = model.network.SALNetwork
}}

plot{{
    resolution = {resolution}
    mc_value = 0.0
    is_uniform_grid = True
    verbose = False
    save_html = False
    save_ply = True
    overwrite = True
}}

network{{
    decode_mnfld_pnts = False
    encoder{{

    }}
    decoder
    {{
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
        dropout = []
        dropout_prob =  0.2
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        latent_in = []
        xyz_in_all = False
        activation = None

        latent_dropout = False
        weight_norm = True
    }}

    loss{{
        loss_type = model.loss.SALLoss
        properties{{
            manifold_pnts_weight = 0
            unsigned = True
        }}
    }}
}}
"""
    path = SAL_CODE / "confs" / f"{expname}.conf"
    path.write_text(config, encoding="ascii")
    return path


def latest_timestamp(expname: str) -> str:
    exp_dir = SAL_ROOT / "exps" / expname
    timestamps = [p.name for p in exp_dir.iterdir() if p.is_dir()]
    if not timestamps:
        raise FileNotFoundError(f"No timestamps found in {exp_dir}")
    return sorted(timestamps)[-1]


def has_latest_checkpoint(expname: str) -> bool:
    exp_dir = SAL_ROOT / "exps" / expname
    if not exp_dir.exists():
        return False
    for timestamp_dir in exp_dir.iterdir():
        checkpoint = timestamp_dir / "checkpoints" / "ModelParameters" / "latest.pth"
        if checkpoint.exists():
            return True
    return False


def find_sal_eval_mesh(expname: str, timestamp: str) -> Path:
    eval_dir = SAL_ROOT / "exps" / expname / timestamp / "evaluation" / "none"
    meshes = sorted(eval_dir.glob("**/*.ply"))
    if len(meshes) != 1:
        raise FileNotFoundError(f"Expected one SAL eval PLY in {eval_dir}, found {len(meshes)}")
    return meshes[0]


def find_sal_eval_mesh_or_none(expname: str, timestamp: str):
    try:
        return find_sal_eval_mesh(expname, timestamp)
    except FileNotFoundError:
        return None


def compare_metric(metric: str, baseline: float, method: float):
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


def write_summary(rows, summary_csv: Path):
    metrics = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    methods = sorted({row["method"] for row in rows})
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
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


def finite_metric(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Run selected SAL frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--input-points", type=int, default=200000)
    parser.add_argument("--sample-points", type=int, default=200000)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--mc-values", nargs="+", type=float, default=[0.0, 0.0025, 0.005, 0.01, 0.02])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = parser.parse_args()

    out_root = REPO_ROOT / "results" / "method_outputs" / "sal" / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / "sal_selected10"
    metric_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    per_frame_csv = metric_root / "per_frame_metrics.csv"
    comparison_csv = metric_root / "baseline_vs_sal_by_frame.csv"
    summary_csv = metric_root / "summary_metrics.csv"
    config_json = metric_root / "run_config.json"

    rows = []
    comparisons = []

    for frame in args.frames:
        expname = f"{args.sequence}_frame_{frame}_sal{args.epochs}_{args.input_points // 1000}k"
        print(f"\n=== SAL {args.sequence} frame {frame} ({expname}) ===", flush=True)

        run([
            sys.executable,
            "scripts/prepare_uvg_frame_for_sal.py",
            "--sequence",
            args.sequence,
            "--frame",
            frame,
            "--dataset-root",
            args.dataset_root,
            "--max-points",
            args.input_points,
        ])

        conf = write_sal_config(frame, expname, args.resolution)

        if not has_latest_checkpoint(expname):
            run([
                sys.executable,
                "training/exp_runner.py",
                "--batch_size",
                "1",
                "--nepoch",
                args.epochs,
                "--conf",
                conf.relative_to(SAL_CODE),
                "--workers",
                "1",
                "--gpu",
                "all",
            ], cwd=SAL_CODE)
        else:
            print(f"[skip] latest checkpoint already exists for {expname}", flush=True)

        timestamp = latest_timestamp(expname)
        mesh_out = out_root / f"frame_{frame}_mesh.ply"
        vertex_out = out_root / f"frame_{frame}_vertices.ply"
        surface_out = out_root / f"frame_{frame}.ply"
        xyz = SAL_ROOT / "data" / "uvg_kettlebell" / "points" / args.sequence / f"frame_{frame}.xyz"

        sal_mesh_norm = None
        if not surface_out.exists():
            for mc_value in args.mc_values:
                print(f"[eval] {expname} mc_value={mc_value}", flush=True)
                run([
                    sys.executable,
                    "evaluate/evaluate.py",
                    "--exp_name",
                    expname,
                    "--conf",
                    conf.relative_to(SAL_CODE),
                    "--checkpoint",
                    "latest",
                    "--split",
                    "none",
                    "--gpu",
                    "0",
                    "--resolution",
                    args.resolution,
                    "--mc_value",
                    mc_value,
                ], cwd=SAL_CODE)
                sal_mesh_norm = find_sal_eval_mesh_or_none(expname, timestamp)
                if sal_mesh_norm is not None:
                    break
            if sal_mesh_norm is None:
                print(f"[failed] no SAL mesh extracted for {expname}; skipping frame", flush=True)
                continue
        else:
            print(f"[skip] method output already exists for frame {frame}: {surface_out}", flush=True)

        if not surface_out.exists():
            run([
                sys.executable,
                "scripts/denormalize_sal_output.py",
                "--sal-input-xyz",
                xyz,
                "--sal-output-ply",
                sal_mesh_norm,
                "--out-mesh-ply",
                mesh_out,
                "--out-pointcloud-ply",
                vertex_out,
            ])
            run([
                sys.executable,
                "scripts/sample_ascii_mesh_ply.py",
                "--mesh",
                mesh_out,
                "--out",
                surface_out,
                "--points",
                args.sample_points,
                "--seed",
                frame,
            ])

        cg = find_frame(args.dataset_root / args.sequence / "cg" / "15fps", frame)
        he = find_frame(args.dataset_root / args.sequence / "he" / "15fps", frame)
        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        sal = eval_pointcloud(str(surface_out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": "sal", "sequence": args.sequence, "frame": frame, "pred_file": str(surface_out), "gt_file": str(he), **sal})
        for metric in baseline:
            if finite_metric(baseline[metric]) and finite_metric(sal[metric]):
                delta, improved = compare_metric(metric, baseline[metric], sal[metric])
            else:
                delta, improved = float("nan"), False
            comparisons.append({
                "frame": frame,
                "metric": metric,
                "baseline": baseline[metric],
                "sal": sal[metric],
                "delta_for_better": delta,
                "sal_improved": improved,
            })

        metric_names = list(baseline.keys())
        with per_frame_csv.open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
            writer.writeheader()
            writer.writerows(rows)
        with comparison_csv.open("w", newline="", encoding="ascii") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "metric", "baseline", "sal", "delta_for_better", "sal_improved"])
            writer.writeheader()
            writer.writerows(comparisons)
        write_summary(rows, summary_csv)

    with config_json.open("w", encoding="ascii") as handle:
        json.dump({
            "sequence": args.sequence,
            "frames": args.frames,
            "epochs": args.epochs,
            "input_points": args.input_points,
            "sample_points": args.sample_points,
            "resolution": args.resolution,
            "mc_values": args.mc_values,
            "dataset_root": str(args.dataset_root),
            "method_output_root": str(out_root),
            "per_frame_csv": str(per_frame_csv),
            "comparison_csv": str(comparison_csv),
            "summary_csv": str(summary_csv),
        }, handle, indent=2)

    print(f"Per-frame metrics: {per_frame_csv}")
    print(f"Comparison metrics: {comparison_csv}")
    print(f"Summary metrics: {summary_csv}")


if __name__ == "__main__":
    main()
