"""Run APU-LDI local-distance-indicator upsampling on selected UVG frames.

This adapter uses only the pretrained local LDI network. It does not run the
paper's per-shape global-field optimization path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points

import run_mag_selected_frames as common


APULDI_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "APU-LDI"
LDI_ROOT = APULDI_ROOT / "local_distance_indicator"
CKPT = LDI_ROOT / "pretrained_local" / "pu1k_local" / "ckpt" / "ckpt-epoch-60.pth"


def parse_model_args():
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        sys.path.insert(0, str(LDI_ROOT))
        from args.pu1k_args import parse_pu1k_args  # noqa: WPS433

        return parse_pu1k_args()
    finally:
        sys.argv = old_argv


def fps_numpy(points: np.ndarray, count: int, device: str) -> np.ndarray:
    if len(points) <= count:
        return points.astype(np.float32)
    tensor = torch.as_tensor(points[None], dtype=torch.float32, device=device)
    sampled, _idx = sample_farthest_points(tensor, K=count)
    return sampled[0].detach().cpu().numpy().astype(np.float32)


def load_model(device: str):
    sys.path.insert(0, str(LDI_ROOT))
    from models.P2PNet_Attention import P2PNet  # noqa: WPS433

    args = parse_model_args()
    args.ckpt_path = str(CKPT)
    args.up_rate = 4
    args.double_4X = False
    args.truncate_distance = True

    model = P2PNet(args).to(device)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return args, model


def run_apuldi(args, model, sampled: np.ndarray, device: str):
    from einops import rearrange  # noqa: WPS433
    from models.utils import normalize_point_cloud  # noqa: WPS433
    from test import pcd_upsample  # noqa: WPS433

    input_pcd = torch.as_tensor(sampled, dtype=torch.float32, device=device)
    input_pcd = rearrange(input_pcd, "n c -> 1 c n").contiguous()
    input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
    with torch.enable_grad():
        output = pcd_upsample(args, model, input_pcd)
    output = centroid + output * furthest_distance
    output = rearrange(output.squeeze(0), "c n -> n c").contiguous()
    return output.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APU-LDI local selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="apuldi_local_pu1k_4x_2048")
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", type=Path, default=common.REPO_ROOT / "results")
    args = parser.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("APU-LDI CUDA inference requested but CUDA is unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_args, model = load_model(args.device)

    out_root = args.results_root / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = args.results_root / "uvg_cwi_dqpc" / args.sequence / args.method_name
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
        sampled = fps_numpy(source_points, args.input_points, args.device)
        enhanced_points = run_apuldi(model_args, model, sampled, args.device)
        enhanced_colors = common.transfer_nearest_colors(source_points, source_colors, enhanced_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, enhanced_points, enhanced_colors)
        counts.append({
            "frame": frame,
            "input_points": len(source_points),
            "sampled_input_points": len(sampled),
            "output_points": len(enhanced_points),
            "has_color": True,
        })
        print(f"{frame}: {len(source_points)} -> sampled {len(sampled)} -> {len(enhanced_points)} points, output={out}", flush=True)

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
