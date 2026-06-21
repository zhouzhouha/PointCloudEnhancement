"""Run PoinTr completion on selected UVG-CWI-DQPC frames.

This is a domain-transfer adapter around the official PoinTr implementation.
UVG frames are much larger than ShapeNet-style completion inputs, so the
adapter deterministically samples 2048 XYZ points, applies the official
ShapeNet normalization, runs the pretrained PoinTr model, denormalizes the
completed points, and transfers RGB from the original CG frame with k=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common


POINTR_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PoinTr"
DEFAULT_POINTR_CONFIG = POINTR_ROOT / "cfgs" / "ShapeNet55_models" / "PoinTr.yaml"
DEFAULT_POINTR_CKPT = POINTR_ROOT / "pretrained" / "PoinTr_ShapeNet55.pth"


def load_pointr(device: str, config_path: Path, checkpoint_path: Path):
    sys.path.insert(0, str(POINTR_ROOT))
    old_cwd = Path.cwd()
    try:
        # PoinTr config files use relative _base_ paths from the repo root.
        import os

        os.chdir(POINTR_ROOT)
        from tools import builder  # noqa: WPS433
        from utils.config import cfg_from_yaml_file  # noqa: WPS433

        config = cfg_from_yaml_file(str(config_path.relative_to(POINTR_ROOT)))
        model = builder.model_builder(config.model)
        builder.load_model(model, str(checkpoint_path))
        model.to(device)
        model.eval()
        return model
    finally:
        import os

        os.chdir(old_cwd)


def normalize_shapenet(points: np.ndarray):
    centroid = np.mean(points, axis=0, keepdims=True)
    normalized = points - centroid
    scale = np.max(np.sqrt(np.sum(normalized ** 2, axis=1)))
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate point cloud.")
    return (normalized / scale).astype(np.float32), centroid.astype(np.float32), float(scale)


def sample_input(points: np.ndarray, n_points: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(points) >= n_points:
        indices = rng.choice(len(points), size=n_points, replace=False)
    else:
        indices = rng.choice(len(points), size=n_points, replace=True)
    return points[indices].astype(np.float32)


def complete_frame(model, points: np.ndarray, device: str, n_input: int, seed: int):
    sampled = sample_input(points, n_input, seed)
    normalized, centroid, scale = normalize_shapenet(sampled)
    tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)[-1].squeeze(0).detach().cpu().numpy().astype(np.float32)
    output = output * scale + centroid
    return output.astype(np.float32), sampled


def main():
    parser = argparse.ArgumentParser(description="Run PoinTr selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pointr_shapenet55")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_POINTR_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_POINTR_CKPT)
    parser.add_argument("--n-input", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.model_config = args.model_config.resolve()
    args.checkpoint = args.checkpoint.resolve()

    if not args.model_config.exists():
        raise FileNotFoundError(args.model_config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("PoinTr smoke requires CUDA.")

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_pointr(device, args.model_config, args.checkpoint)

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
        completed_points, sampled_points = complete_frame(
            model,
            source_points,
            device=device,
            n_input=args.n_input,
            seed=args.seed + int(frame),
        )
        completed_colors = common.transfer_nearest_colors(source_points, source_colors, completed_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, completed_points, completed_colors)
        counts.append({
            "frame": frame,
            "input_points": len(source_points),
            "sampled_input_points": len(sampled_points),
            "output_points": len(completed_points),
            "has_color": True,
        })
        print(f"{frame}: {len(source_points)} -> sample {len(sampled_points)} -> {len(completed_points)} points, output={out}", flush=True)

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
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "sampled_input_points", "output_points", "has_color"])
        writer.writeheader()
        writer.writerows(counts)
    common.write_summary(rows, metric_root / "summary_metrics.csv")
    (metric_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


if __name__ == "__main__":
    main()
