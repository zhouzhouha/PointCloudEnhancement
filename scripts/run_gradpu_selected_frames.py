"""Run Grad-PU upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


GRADPU_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "Grad-PU"
GRADPU_CKPT = GRADPU_ROOT / "pretrained_model" / "pu1k" / "ckpt" / "ckpt-epoch-60.pth"


def load_gradpu():
    sys.path.insert(0, str(GRADPU_ROOT))
    from args.pu1k_args import parse_pu1k_args  # noqa: WPS433
    from models.P2PNet import P2PNet  # noqa: WPS433
    from test import pcd_upsample  # noqa: WPS433

    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_pu1k_args()
    finally:
        sys.argv = original_argv
    return P2PNet, pcd_upsample, model_args


def upsample_frame(model, pcd_upsample, model_args, points):
    from einops import rearrange
    from models.utils import normalize_point_cloud

    input_pcd = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    input_pcd = rearrange(input_pcd, "n c -> c n").contiguous().unsqueeze(0)
    input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
    upsampled = pcd_upsample(model_args, model, input_pcd)
    upsampled = centroid + upsampled * furthest_distance
    upsampled = rearrange(upsampled.squeeze(0), "c n -> n c").contiguous()
    return upsampled.detach().cpu().numpy().astype(np.float32)


def upsample_chunked(model, pcd_upsample, model_args, points, chunk_size: int):
    if chunk_size <= 0 or len(points) <= chunk_size:
        return upsample_frame(model, pcd_upsample, model_args, points)
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=2048)
    outputs = []
    for chunk_id, idx in enumerate(chunks):
        pred = upsample_frame(model, pcd_upsample, model_args, points[idx])
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run Grad-PU selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=common.REPO_ROOT / "results")
    parser.add_argument("--method-name", default="gradpu")
    parser.add_argument("--dataset", default="pu1k", choices=["pu1k"])
    parser.add_argument("--up-rate", type=int, default=4)
    parser.add_argument("--truncate-distance", action="store_true", default=True)
    parser.add_argument("--no-truncate-distance", dest="truncate_distance", action="store_false")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 keeps the original full-frame wrapper; positive values use a large-frame chunk wrapper.")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    if not GRADPU_CKPT.exists():
        raise FileNotFoundError(GRADPU_CKPT)
    if not torch.cuda.is_available():
        raise RuntimeError("Grad-PU requires CUDA for this smoke adapter.")

    P2PNet, pcd_upsample, model_args = load_gradpu()
    model_args.ckpt_path = str(GRADPU_CKPT)
    model_args.up_rate = args.up_rate
    model_args.truncate_distance = args.truncate_distance
    model_args.double_4X = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = P2PNet(model_args).cuda()
    model.load_state_dict(torch.load(str(GRADPU_CKPT), map_location="cuda"))
    model.eval()

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
        upsampled_points = upsample_chunked(model, pcd_upsample, model_args, source_points, args.chunk_size)
        upsampled_colors = common.transfer_nearest_colors(source_points, source_colors, upsampled_points)
        out = out_root / f"frame_{frame}.ply"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite immutable output: {out}")
        partial = out.with_name(f".{out.name}.part-{os.getpid()}")
        try:
            common.write_xyzrgb_ply(partial, upsampled_points, upsampled_colors)
            partial.replace(out)
        finally:
            partial.unlink(missing_ok=True)
        counts.append({"frame": frame, "input_points": len(source_points), "output_points": len(upsampled_points), "has_color": True})
        print(f"{frame}: {len(source_points)} -> {len(upsampled_points)} points, output={out}", flush=True)

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
