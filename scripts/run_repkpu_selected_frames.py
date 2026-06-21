"""Run RepKPU upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


REPKPU_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "RepKPU"
DEFAULT_CKPT = REPKPU_ROOT / "pretrain" / "exp1" / "ckpt-best.pth"


def load_model(device: str, checkpoint: Path):
    sys.path.insert(0, str(REPKPU_ROOT))
    from cfgs.upsampling import parse_pu1k_args  # noqa: WPS433
    from cfgs.utils import reset_model_args  # noqa: WPS433
    from models.repkpu import RepKPU  # noqa: WPS433

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]]
        cfg = parse_pu1k_args()
    finally:
        sys.argv = old_argv
    runtime_args = argparse.Namespace()
    reset_model_args(cfg, runtime_args)
    model = RepKPU(runtime_args).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, runtime_args


def repkpu_upsampling(model, cfg, input_pcd: torch.Tensor, up_rate: int, patch_rate: int):
    sys.path.insert(0, str(REPKPU_ROOT))
    from models.utils import FPS, extract_knn_patch, normalize_point_cloud  # noqa: WPS433

    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = cfg.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    with torch.no_grad():
        coarse_pts, _reg_loss = model.forward(patches)
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, "b c n -> c (b n)").contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1] * up_rate)
    return coarse_pts


def normalize_chunk(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate RepKPU chunk.")
    return (centered / scale).astype(np.float32), center.astype(np.float32), float(scale)


def run_chunked(model, cfg, points: np.ndarray, device: str, chunk_size: int, up_rate: int, patch_rate: int):
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=cfg.num_points)
    for chunk_id, idx in enumerate(chunks):
        chunk_norm, center, scale = normalize_chunk(points[idx])
        tensor = torch.as_tensor(chunk_norm.T[None], dtype=torch.float32, device=device)
        pred = repkpu_upsampling(model, cfg, tensor, up_rate=up_rate, patch_rate=patch_rate)
        pred_np = pred.squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        pred_np = pred_np * scale + center
        outputs.append(pred_np)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred_np)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepKPU selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="repkpu_pu1k_4x")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--up-rate", type=int, default=4)
    parser.add_argument("--patch-rate", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RepKPU CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(21)
    np.random.seed(21)
    model, cfg = load_model(args.device, args.checkpoint)

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
        enhanced_points = run_chunked(model, cfg, source_points, args.device, args.chunk_size, args.up_rate, args.patch_rate)
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
