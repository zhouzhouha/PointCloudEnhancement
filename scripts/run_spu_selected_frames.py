"""Run Semantic Point Cloud Upsampling (SPU) on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common


SPU_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "SPU"
SPU_CLS_CKPT = SPU_ROOT / "PretrainModel" / "PointNetModelNet40.parm"
SPU_UP_CKPTS = {
    2: SPU_ROOT / "savedModel" / "SPU_Final2X_pointnet_att.parm",
    4: SPU_ROOT / "savedModel" / "SPU_Final4X_pointnet_att.parm",
    8: SPU_ROOT / "savedModel" / "SPU_Final8X_pointnet_att.parm",
    16: SPU_ROOT / "savedModel" / "SPU_Final16X_pointnet_att.parm",
}


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


def load_models(device: str, scale: int, up_ckpt: Path | None):
    sys.path.insert(0, str(SPU_ROOT))
    from ClsNetwork.PointNetcls import PoiintNet  # noqa: WPS433
    from option.train_option import get_train_options  # noqa: WPS433

    module = importlib.import_module(f"arch.SPUfinal{scale}x")
    generator_cls = module.Generator

    cls_model = PoiintNet(k=40, normal_channel=False).to(device)
    cls_model.load_state_dict(torch.load(str(SPU_CLS_CKPT), map_location=device))
    cls_model.eval()

    ckpt = up_ckpt or SPU_UP_CKPTS[scale]
    generator = generator_cls(get_train_options()).to(device)
    generator.load_state_dict(torch.load(str(ckpt), map_location=device))
    generator.eval()
    return cls_model, generator


def normalize_chunk(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    scale = np.linalg.norm(centered, axis=1).max()
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate SPU chunk.")
    return (centered / scale).astype(np.float32), center.astype(np.float32), float(scale)


def run_chunked(cls_model, generator, points: np.ndarray, device: str, chunk_size: int):
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=32)
    for chunk_id, idx in enumerate(chunks):
        chunk_norm, center, scale = normalize_chunk(points[idx])
        tensor = torch.as_tensor(chunk_norm.T[None], dtype=torch.float32, device=device)
        tensor.requires_grad_(True)
        cls_model.zero_grad(set_to_none=True)
        generator.zero_grad(set_to_none=True)
        logits, _global_feat, _cls_feat = cls_model(tensor)
        grad, = torch.autograd.grad(logits, tensor, torch.ones_like(logits), retain_graph=False)
        with torch.no_grad():
            pred = generator(tensor, grad.detach()).squeeze(0).transpose(0, 1).contiguous()
        pred_np = pred.detach().cpu().numpy().astype(np.float32) * scale + center
        outputs.append(pred_np)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred_np)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run SPU selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="spu_pointnet_4x")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--scale", type=int, choices=sorted(SPU_UP_CKPTS), default=4)
    parser.add_argument("--up-ckpt", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not SPU_CLS_CKPT.exists():
        raise FileNotFoundError(SPU_CLS_CKPT)
    up_ckpt = args.up_ckpt or SPU_UP_CKPTS[args.scale]
    if not up_ckpt.exists():
        raise FileNotFoundError(up_ckpt)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SPU CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cls_model, generator = load_models(args.device, args.scale, args.up_ckpt)

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
        enhanced_points = run_chunked(cls_model, generator, source_points, args.device, args.chunk_size)
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
