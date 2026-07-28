"""Run PUDM upsampling on selected UVG-CWI-DQPC frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


PUDM_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PUDM"
DEFAULT_CONFIG = PUDM_ROOT / "pointnet2" / "exp_configs" / "PU1K.json"
DEFAULT_CKPT = PUDM_ROOT / "pointnet2" / "pkls" / "pu1k.pkl"


def merge_rows(path: Path, new_rows: list[dict], key_fields: list[str], fieldnames: list[str]) -> None:
    """Write CSV rows while preserving prior rows outside the current resume set."""
    merged: dict[tuple[str, ...], dict] = {}
    if path.exists():
        with path.open(newline="", encoding="ascii") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                merged[tuple(row.get(field, "") for field in key_fields)] = row
    for row in new_rows:
        merged[tuple(str(row.get(field, "")) for field in key_fields)] = row
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda row: tuple(row.get(field, "") for field in key_fields)))


def setup_pudm_imports() -> None:
    os.environ.setdefault("PCE_FORCE_PUDM_FALLBACKS", "1")
    sys.path.insert(0, str(PUDM_ROOT))
    sys.path.insert(0, str(PUDM_ROOT / "pointnet2_ops_lib"))
    sys.path.insert(0, str(PUDM_ROOT / "pointnet2"))


def normalize_chunk(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate PUDM chunk.")
    return (centered / scale).astype(np.float32), center, scale


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    setup_pudm_imports()
    from json_reader import restore_string_to_list_in_a_dict  # noqa: WPS433
    from models.pointnet2_with_pcld_condition import PointNet2CloudCondition  # noqa: WPS433
    from util import calc_diffusion_hyperparams  # noqa: WPS433

    config = restore_string_to_list_in_a_dict(json.loads(config_path.read_text(encoding="ascii")))
    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    for key, value in list(diffusion_hyperparams.items()):
        if key != "T":
            diffusion_hyperparams[key] = value.to(device)

    model = PointNet2CloudCondition(pointnet_config).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, diffusion_hyperparams


def run_pudm_chunk(model, diffusion_hyperparams, points: np.ndarray, device: str, up_rate: int, gamma: float, step: int):
    setup_pudm_imports()
    from util import sampling_ddim  # noqa: WPS433

    points_norm, center, scale = normalize_chunk(points)
    condition = torch.as_tensor(points_norm[None], dtype=torch.float32, device=device).contiguous()
    label = torch.full((1,), fill_value=up_rate - 1, dtype=torch.long, device=device)
    num_points = condition.shape[1] * up_rate
    model.reset_cond_features()
    with torch.no_grad():
        generated, _condition_pre, _z = sampling_ddim(
            net=model,
            size=(1, num_points, 3),
            diffusion_hyperparams=diffusion_hyperparams,
            print_every_n_steps=max(1, step // 3),
            label=label,
            condition=condition,
            R=up_rate,
            gamma=gamma,
            step=step,
        )
    pred = generated[0].detach().cpu().numpy().astype(np.float32)
    return pred * scale + center


def run_chunked(model, diffusion_hyperparams, points: np.ndarray, device: str, chunk_size: int, up_rate: int, gamma: float, step: int):
    outputs = []
    chunks = iter_chunks(points, chunk_size=chunk_size, min_points=chunk_size)
    for chunk_id, idx in enumerate(chunks):
        pred = run_pudm_chunk(model, diffusion_hyperparams, points[idx], device, up_rate, gamma, step)
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def write_binary_xyzrgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 255).astype(np.float64, copy=False) / 255.0)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f"Failed to write binary PLY: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PUDM selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pudm_pu1k_4x")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--up-rate", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PUDM CUDA inference requested but CUDA is unavailable.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, diffusion_hyperparams = load_model(args.config, args.checkpoint, args.device)

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
            diffusion_hyperparams,
            source_points,
            args.device,
            args.chunk_size,
            args.up_rate,
            args.gamma,
            args.step,
        )
        enhanced_colors = common.transfer_nearest_colors(source_points, source_colors, enhanced_points)
        out = out_root / f"frame_{frame}.ply"
        write_binary_xyzrgb_ply(out, enhanced_points, enhanced_colors)
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
    metrics_path = metric_root / "per_frame_metrics.csv"
    comparison_path = metric_root / f"baseline_vs_{args.method_name}_by_frame.csv"
    counts_path = metric_root / "point_counts.csv"
    merge_rows(metrics_path, rows, ["method", "frame"], ["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
    merge_rows(
        comparison_path,
        comparisons,
        ["frame", "metric"],
        ["frame", "metric", "baseline", args.method_name, "delta_for_better", f"{args.method_name}_improved"],
    )
    merge_rows(counts_path, counts, ["frame"], ["frame", "input_points", "output_points", "has_color"])
    with metrics_path.open(newline="", encoding="ascii") as handle:
        summary_rows = list(csv.DictReader(handle))
    common.write_summary(summary_rows, metric_root / "summary_metrics.csv")
    (metric_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


if __name__ == "__main__":
    main()
