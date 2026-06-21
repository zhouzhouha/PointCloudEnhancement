"""Run PUCRN on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common


PUCRN_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "3DVision" / "PointUpsampling" / "PUCRN"
CKPT = PUCRN_ROOT / "model" / "release" / "model.pth"


def morton_order(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0, keepdims=True)
    spans = points.max(axis=0, keepdims=True) - mins
    spans[spans <= 0] = 1.0
    quantized = np.clip(((points - mins) / spans * 1023.0).astype(np.uint32), 0, 1023)
    code = np.zeros(len(points), dtype=np.uint32)
    for bit in range(10):
        code |= ((quantized[:, 0] >> bit) & 1) << (3 * bit)
        code |= ((quantized[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((quantized[:, 2] >> bit) & 1) << (3 * bit + 2)
    return np.argsort(code, kind="mergesort")


def iter_chunks(points: np.ndarray, chunk_size: int, min_points: int):
    order = morton_order(points)
    chunks = [order[start : start + chunk_size] for start in range(0, len(order), chunk_size)]
    if len(chunks) > 1 and len(chunks[-1]) < min_points:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    return chunks


def normalize_chunk(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate PUCRN chunk.")
    return (centered / scale).astype(np.float32), center, scale


def load_model(device: str, up_ratio: int):
    sys.path.insert(0, str(PUCRN_ROOT / "pointnet2_ops_lib"))
    sys.path.insert(0, str(PUCRN_ROOT))
    from network.PUCRN import CRNet  # noqa: WPS433

    model = CRNet(up_ratio).to(device)
    state = torch.load(str(CKPT), map_location=device)["net_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_chunked(model, points: np.ndarray, device: str, chunk_size: int):
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=256)
    for chunk_id, idx in enumerate(chunks):
        chunk_norm, center, scale = normalize_chunk(points[idx])
        tensor = torch.as_tensor(chunk_norm.T[None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(tensor)
        if pred.size(1) == 3:
            pred = pred.transpose(2, 1).contiguous()
        pred_np = pred[0].detach().cpu().numpy().astype(np.float32) * scale + center
        outputs.append(pred_np.astype(np.float32))
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred_np)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUCRN selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pucrn_pu1k_4x")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PUCRN CUDA inference requested but CUDA is unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args.device, args.up_ratio)

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
        enhanced_points = run_chunked(model, source_points, args.device, args.chunk_size)
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
