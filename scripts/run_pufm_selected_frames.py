"""Run PUFM upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


PUFM_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PUFM"


def setup_imports() -> None:
    os.environ.setdefault("PCE_FORCE_PUFM_FALLBACKS", "1")
    sys.path.insert(0, str(PUFM_ROOT))


def load_model(args, device: str):
    setup_imports()
    from models.diffusion import PUFM, PUFM_w_attn  # noqa: WPS433

    if args.model == "pufm":
        model = PUFM(args).to(device)
    elif args.model == "pufm_w_attn":
        model = PUFM_w_attn(args).to(device)
    else:
        raise ValueError(f"Unsupported PUFM model {args.model}")
    checkpoint = PUFM_ROOT / "pretrained_model" / f"{args.model}.pth"
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def pufm_update_whole(model, interpolated_pcd: torch.Tensor, steps: int):
    updated_pcd = interpolated_pcd.clone()
    batch_size = updated_pcd.shape[0]
    with torch.no_grad():
        for step_id in range(steps):
            alpha = step_id / steps * torch.ones(batch_size, device=updated_pcd.device)
            pred = model(updated_pcd, interpolated_pcd, alpha)
            updated_pcd = updated_pcd + (1.0 / steps) * pred
    return torch.clamp(updated_pcd, -1.0, 1.0)


def run_chunk(model, points: np.ndarray, args, device: str):
    setup_imports()
    from models.utils import midpoint_interpolate, normalize_point_cloud  # noqa: WPS433

    tensor = torch.as_tensor(points.T[None], dtype=torch.float32, device=device).contiguous()
    tensor, centroid, furthest_distance = normalize_point_cloud(tensor)
    interpolated = midpoint_interpolate(args, tensor)
    updated = pufm_update_whole(model, interpolated, args.steps)
    updated = centroid + updated * furthest_distance
    return rearrange(updated.squeeze(0), "c n -> n c").detach().cpu().numpy().astype(np.float32)


def run_chunked(model, points: np.ndarray, args, device: str):
    chunks = iter_chunks(points, chunk_size=args.chunk_size, min_points=max(args.num_points, 256))
    outputs = []
    for chunk_id, idx in enumerate(chunks):
        pred = run_chunk(model, points[idx], args, device)
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUFM selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pufm_pugan_4x")
    parser.add_argument("--model", choices=["pufm", "pufm_w_attn"], default="pufm")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--up-rate", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PUFM CUDA inference requested but CUDA is unavailable.")
    checkpoint = PUFM_ROOT / "pretrained_model" / f"{args.model}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args, args.device)

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
        enhanced_points = run_chunked(model, source_points, args, args.device)
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
