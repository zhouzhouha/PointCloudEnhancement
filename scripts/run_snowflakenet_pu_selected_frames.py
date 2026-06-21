"""Run SnowflakeNet point-cloud upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


SNOWFLAKENET_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "SnowflakeNet"
SNOWFLAKENET_PU_ROOT = SNOWFLAKENET_ROOT / "PU"
DEFAULT_CKPT = SNOWFLAKENET_ROOT / "pretrained" / "pu" / "ckpt-pu.pth"
DEFAULT_UP_FACTORS = [1, 2, 2, 1]


def load_model(device: str, checkpoint: Path, up_factors: list[int]):
    sys.path.insert(0, str(SNOWFLAKENET_ROOT))
    sys.path.insert(0, str(SNOWFLAKENET_PU_ROOT))
    from models.model_pu import ModelPU  # noqa: WPS433

    model = ModelPU(up_factors=up_factors)
    checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
    state = checkpoint_data.get("model", checkpoint_data)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def normalize_tensor(points: torch.Tensor):
    centroid = points.mean(dim=1, keepdim=True)
    centered = points - centroid
    scale = torch.sqrt(torch.sum(centered**2, dim=-1, keepdim=True)).max(dim=1, keepdim=True)[0]
    return centered / scale, centroid, scale


def upsample_chunk(model, chunk_points: np.ndarray, device: str, patch_num_ratio: int, num_per_patch: int, up_ratio: int):
    sys.path.insert(0, str(SNOWFLAKENET_ROOT))
    sys.path.insert(0, str(SNOWFLAKENET_PU_ROOT))
    from PU.utils import fps_subsample, patch_extraction  # noqa: WPS433

    points = torch.as_tensor(chunk_points, dtype=torch.float32, device=device).unsqueeze(0)
    normalized, centroid, scale = normalize_tensor(points)
    patch_points = patch_extraction(normalized, num_per_patch=num_per_patch, patch_num_ratio=patch_num_ratio)
    normalized_patch_points, patch_centroid, patch_scale = normalize_tensor(patch_points)
    with torch.no_grad():
        predictions = model(normalized_patch_points)
        normalized_upsampled = predictions[-1]
    upsampled_patch_points = normalized_upsampled * patch_scale + patch_centroid
    upsampled_points = upsampled_patch_points.reshape(1, -1, 3).contiguous()
    pred = fps_subsample(upsampled_points, chunk_points.shape[0] * up_ratio)
    pred = pred * scale + centroid
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)


def run_chunked(model, points: np.ndarray, device: str, chunk_size: int, patch_num_ratio: int, num_per_patch: int, up_ratio: int):
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=num_per_patch)
    for chunk_id, idx in enumerate(chunks):
        pred = upsample_chunk(model, points[idx], device, patch_num_ratio, num_per_patch, up_ratio)
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SnowflakeNet-PU selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="snowflakenet_pu_4x")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--patch-num-ratio", type=int, default=3)
    parser.add_argument("--num-per-patch", type=int, default=256)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SnowflakeNet-PU CUDA smoke requested but CUDA is unavailable.")

    torch.manual_seed(2026)
    np.random.seed(2026)
    model = load_model(args.device, args.checkpoint, DEFAULT_UP_FACTORS)

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
        enhanced_points = run_chunked(
            model,
            source_points,
            args.device,
            args.chunk_size,
            args.patch_num_ratio,
            args.num_per_patch,
            args.up_ratio,
        )
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
