"""Run Deep-RS denoising on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

import run_mag_selected_frames as common


DEEPRS_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "deep-rs"
DEEPRS_CONFIG = DEEPRS_ROOT / "configs" / "denoise.yml"
DEEPRS_CKPT = DEEPRS_ROOT / "ckpts" / "denoise.pth"


def load_deeprs(device: str):
    sys.path.insert(0, str(DEEPRS_ROOT))
    from models.resampler import PointSetResampler  # noqa: WPS433

    with DEEPRS_CONFIG.open("r", encoding="ascii") as handle:
        config = EasyDict(yaml.safe_load(handle))
    model = PointSetResampler(config.model).to(device)
    model.load_state_dict(torch.load(str(DEEPRS_CKPT), map_location=device))
    model.eval()
    return model


def denoise_frame(model, points: np.ndarray, device: str, cluster_size: int, seed: int) -> np.ndarray:
    from utils.denoise import denoise_large_pointcloud  # noqa: WPS433

    tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
    with torch.no_grad():
        denoised = denoise_large_pointcloud(model, tensor, cluster_size=cluster_size, seed=seed)
    return denoised.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run Deep-RS selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="deeprs_denoise")
    parser.add_argument("--cluster-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not DEEPRS_CKPT.exists():
        raise FileNotFoundError(DEEPRS_CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Deep-RS CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_deeprs(args.device)

    out_root = common.REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = common.REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [out_root, metric_root]:
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
        denoised_points = denoise_frame(model, source_points, args.device, args.cluster_size, args.seed)
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
