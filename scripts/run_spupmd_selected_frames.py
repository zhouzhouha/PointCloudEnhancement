"""Run SPU-PMD upsampling on selected UVG-CWI-DQPC frames."""

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


SPUPMD_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "SPU-PMD"
SPUPMD_CKPT = SPUPMD_ROOT / "pretrained" / "spupmd_pu1k_pretrained"


def setup_imports() -> None:
    os.environ.setdefault("PCE_FORCE_SPUPMD_FALLBACKS", "1")
    sys.path.insert(0, str(SPUPMD_ROOT))
    sys.path.insert(0, str(SPUPMD_ROOT / "pointnet2_ops_lib"))


def normalize_point_cloud(data: np.ndarray):
    centroid = np.mean(data, axis=1, keepdims=True)
    centered = data - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(centered**2, axis=-1)), axis=1, keepdims=True)
    furthest_distance = np.expand_dims(np.maximum(furthest_distance, 1e-8), axis=-1)
    return centered / furthest_distance, centroid, furthest_distance


def load_model(args, device: str):
    setup_imports()
    from network.SPUPMD import SPUPMDNet  # noqa: WPS433

    model = SPUPMDNet(args.up_ratio).to(device)
    state = torch.load(str(args.checkpoint), map_location=device)["net_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_chunk(model, points: np.ndarray, args, device: str):
    setup_imports()
    from network import operations  # noqa: WPS433

    data = points.astype(np.float32)[None, ...]
    data, centroid, furthest_distance = normalize_point_cloud(data)
    input_pc = torch.from_numpy(data).transpose(2, 1).to(device=device).float().contiguous()

    num_patches = max(1, int(input_pc.shape[2] / args.num_point * args.patch_num_ratio))
    with torch.no_grad():
        _idx, seeds = operations.fps_subsample(input_pc, num_patches, NCHW=True)
        patches, _, _ = operations.group_knn(args.num_point, seeds, input_pc, NCHW=True)
        pred_list = []
        for patch_id in range(num_patches):
            patch = patches[:, :, patch_id, :]
            patch, patch_centroid, patch_scale = operations.normalize_point_batch(patch, NCHW=True)
            pred = model(patch.detach())
            if pred.size(1) != 3:
                pred = pred.transpose(2, 1).contiguous()
            pred = pred * patch_scale + patch_centroid
            pred_list.append(pred)
        pred_pc = torch.cat(pred_list, dim=-1)
        _idx, pred_pc = operations.fps_subsample(pred_pc, len(points) * args.up_ratio, NCHW=True)
    pred_pc = pred_pc.transpose(2, 1).detach().cpu().numpy()
    pred_pc = pred_pc * furthest_distance + centroid
    return pred_pc[0].astype(np.float32)


def run_chunked(model, points: np.ndarray, args, device: str):
    chunks = iter_chunks(points, chunk_size=args.chunk_size, min_points=args.num_point)
    outputs = []
    for chunk_id, idx in enumerate(chunks):
        pred = predict_chunk(model, points[idx], args, device)
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPU-PMD selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="spupmd_pu1k_4x")
    parser.add_argument("--checkpoint", type=Path, default=SPUPMD_CKPT)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--num-point", type=int, default=256)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--patch-num-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SPU-PMD CUDA inference requested but CUDA is unavailable.")

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
