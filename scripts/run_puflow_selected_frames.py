"""Run PU-Flow upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common


PUFLOW_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "puflow"
PUFLOW_CKPT = PUFLOW_ROOT / "pretrain" / "puflow-x4-pugeo.pt"


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


def load_model(device: str):
    sys.path.insert(0, str(PUFLOW_ROOT))
    from modules.discrete.interpflow import PointInterpFlow  # noqa: WPS433

    model = PointInterpFlow(pc_channel=3)
    model.load_state_dict(torch.load(str(PUFLOW_CKPT), map_location=device))
    model.set_to_initialized_state()
    model = model.to(device)
    model.eval()
    return model


def run_chunked(model, points: np.ndarray, device: str, chunk_size: int, num_patch: int, up_ratio: int):
    sys.path.insert(0, str(PUFLOW_ROOT))
    from modules.utils.patch import PatchHelper  # noqa: WPS433

    patch_helper = PatchHelper(num_patch, patch_expand_ratio=4)
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=num_patch)
    with torch.no_grad():
        for chunk_id, idx in enumerate(chunks):
            chunk = torch.as_tensor(points[idx], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            pred = patch_helper.upsample(model, chunk, npoint=len(idx), upratio=up_ratio, jitter=False)
            outputs.append(pred.squeeze(0).detach().cpu().numpy().astype(np.float32))
            print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {outputs[-1].shape[0]}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run PU-Flow selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="puflow_discrete")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--num-patch", type=int, default=256)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not PUFLOW_CKPT.exists():
        raise FileNotFoundError(PUFLOW_CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PU-Flow CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_model(args.device)

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
        enhanced_points = run_chunked(model, source_points, args.device, args.chunk_size, args.num_patch, args.up_ratio)
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
