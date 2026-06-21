"""Run octree upsampling-cleaning on selected UVG-CWI-DQPC frames."""

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


OUNET_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "upsample-clean"
OUNET_CONFIG = OUNET_ROOT / "configs" / "upsample-clean.yaml"
OUNET_CKPT = OUNET_ROOT / "logs" / "puc" / "checkpoints" / "ounet.pth"
OUNET_DEPS = OUNET_ROOT / "python_deps"


def load_model(device: str):
    sys.path.insert(0, str(OUNET_DEPS))
    sys.path.insert(0, str(OUNET_ROOT))
    from model import OUNet  # noqa: WPS433

    with OUNET_CONFIG.open("r", encoding="utf-8") as handle:
        config = EasyDict(yaml.safe_load(handle))
    flags = config.MODEL
    # The released checkpoint uses this depth/channel schedule; the checked-in
    # YAML is larger and does not match the public weights.
    flags.depth = 8
    flags.full_depth = 2
    flags.channels = [0, 0, 256, 256, 128, 128, 64, 64, 32]
    model = OUNet(flags).to(device)
    model.load_state_dict(torch.load(str(OUNET_CKPT), map_location=device))
    model.eval()
    return model, flags


def normalize_points(points: np.ndarray):
    center = (points.max(axis=0, keepdims=True) + points.min(axis=0, keepdims=True)) * 0.5
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate point cloud.")
    # Keep the official transform's *1.7 scaling comfortably inside [-1, 1].
    normalized = centered / (2.0 * scale)
    return normalized.astype(np.float32), center.astype(np.float32), float(scale)


def octree2pts(model, octree):
    depth = octree.depth
    signal = octree.features[depth]
    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + signal
    return xyz / 2 ** (depth - 1) - 1.0


def run_frame(model, points: np.ndarray, device: str):
    from ocnn.octree import Octree, Points  # noqa: WPS433

    normalized, center, scale = normalize_points(points)
    pcd_in = torch.as_tensor(normalized * 1.7, dtype=torch.float32)
    features = torch.zeros((pcd_in.shape[0], 1), dtype=torch.float32)
    point_obj = Points(pcd_in, features=features)
    point_obj.clip(min=-1, max=1)
    octree = Octree(8, 2)
    octree.build_octree(point_obj)
    octree.construct_all_neigh()
    octree = octree.to(device)
    with torch.no_grad():
        output = model(octree, update_octree=True)
        out_norm = octree2pts(model, output["octree_out"]) / 1.7
    out_points = out_norm.detach().cpu().numpy().astype(np.float32) * (2.0 * scale) + center
    return out_points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run upsample-clean selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="upsample_clean_ounet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not OUNET_CKPT.exists():
        raise FileNotFoundError(OUNET_CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("upsample-clean CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, _flags = load_model(args.device)

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
        enhanced_points = run_frame(model, source_points, args.device)
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
