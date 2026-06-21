"""Run CRCIR upsampling-after-compression mode on selected UVG frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points, sample_farthest_points

import run_mag_selected_frames as common
from run_puflow_selected_frames import iter_chunks


CRCIR_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "CRCIR_for_PCGC"
CRCIR_DEPS = CRCIR_ROOT / "python_deps"
DEFAULT_CONFIG = CRCIR_ROOT / "result" / "ex0_hyper_5e_3" / "config.yaml"
DEFAULT_CKPT = CRCIR_ROOT / "result" / "ex0_hyper_5e_3" / "checkpoint_best.pth"


def setup_imports() -> None:
    for path in [CRCIR_DEPS, CRCIR_ROOT]:
        sys.path.insert(0, str(path))


def get_scale_table(min_value=0.11, max_value=256, levels=64):
    return torch.exp(torch.linspace(math.log(min_value), math.log(max_value), levels))


def load_components(config_path: Path, checkpoint_path: Path, device: str):
    setup_imports()
    from src import config  # noqa: WPS433

    cfg = config.load_config(str(config_path))
    encoder = config.get_encoder(cfg).to(device)
    decoder = config.get_decoder(cfg).to(device)
    compressor = config.get_compressor(cfg).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    compressor.load_state_dict(checkpoint["compressor"])
    encoder.eval()
    decoder.eval()
    compressor.eval()
    compressor.entropy_bottleneck.update(force=True)
    if cfg["model"]["compressor"] == "hyper":
        compressor.gaussian_conditional.update_scale_table(get_scale_table().to(device))
        compressor.gaussian_conditional.update()
    return cfg, encoder, decoder, compressor


def normalize_01(points: torch.Tensor):
    shift = torch.mean(points, dim=0)
    centered = points - shift
    max_coord = torch.max(centered)
    min_coord = torch.min(centered)
    normed = (centered - min_coord) / (max_coord - min_coord)
    return normed, shift, max_coord, min_coord


def denormalize_from_minus1(points: torch.Tensor, shift: torch.Tensor, max_coord: torch.Tensor, min_coord: torch.Tensor):
    return (points / 2.0 + 0.5) * (max_coord - min_coord) + min_coord + shift


def golden_interpolation(points: torch.Tensor, ratio: int):
    _dist, _idx, coords = knn_points(points, points, K=ratio, return_nn=True)
    neighbors = coords[:, :, 0:ratio, :]
    return points.unsqueeze(2) + (neighbors - points.unsqueeze(2)) * ((3 - math.sqrt(5)) / 2)


def golden_interpolation2(points: torch.Tensor, ratio: int):
    _dist, _idx, coords = knn_points(points, points, K=ratio, return_nn=True)
    neighbors1 = coords[:, :, 0:8, :]
    midpoints1 = points.unsqueeze(2) + (neighbors1 - points.unsqueeze(2)) * ((3 - math.sqrt(5)) / 2)
    neighbors2 = coords[:, :, 8:ratio, :]
    midpoints2 = points.unsqueeze(2) + (neighbors2 - points.unsqueeze(2)) * (0.75 * (3 - math.sqrt(5)) / 2)
    return torch.cat([midpoints1, midpoints2], dim=2)


def interpolate(points: torch.Tensor, ratio: int):
    if ratio <= 8:
        return golden_interpolation(points, ratio)
    return golden_interpolation2(points, ratio)


def run_chunk(points: np.ndarray, cfg, encoder, decoder, compressor, args, device: str):
    chunk = torch.as_tensor(points, dtype=torch.float32, device=device)
    xyzs, shift, max_coord, min_coord = normalize_01(chunk)
    gt_pcd = (2.0 * xyzs - 1.0).unsqueeze(0)
    sparse_count = max(16, int(points.shape[0] // args.encoder_downsample))
    sparse_01, _sparse_idx = sample_farthest_points(xyzs.unsqueeze(0), K=sparse_count)
    sparse_pcd = 2.0 * sparse_01 - 1.0

    c1 = args.encoder_downsample
    c2 = args.decoder_multiplier
    with torch.no_grad():
        interpolated = interpolate(sparse_pcd, c1)
        reshaped = interpolated.view(1, -1, 3)
        _dist, _idx, nns = knn_points(reshaped, gt_pcd, K=1, return_nn=True)
        residuals = nns.squeeze(-2) - reshaped
        residuals_cluster = residuals.view(1, sparse_count, c1, 3)
        local_centers = torch.mean(interpolated, dim=2)
        central_diff = interpolated - local_centers.unsqueeze(2)
        feats = encoder(residuals_cluster, central_diff)

        if cfg["model"]["compressor"] == "ffp":
            latents, _g_bpp, _enc_time, eb_size = compressor.compress(
                g_coords=sparse_pcd,
                g_feats=feats,
                points_num=points.shape[0],
            )
            feats, _dec_time = compressor.decompress(
                g_coords=sparse_pcd,
                g_latents_str=latents,
                points_num=points.shape[0],
                eb_size=eb_size,
            )
        else:
            g_latents, h_latents, _g_bpp, _h_bpp, _enc_time, eb_size = compressor.compress(
                g_coords=sparse_pcd,
                g_feats=feats,
                points_num=points.shape[0],
            )
            feats, _dec_time = compressor.decompress(
                g_coords=sparse_pcd,
                g_latents_str=g_latents,
                h_latents_str=h_latents,
                points_num=points.shape[0],
                eb_size=eb_size,
            )

        total_ratio = c1 * c2
        interpolated2 = interpolate(sparse_pcd, total_ratio)
        central_diff2 = interpolated2 - local_centers.unsqueeze(2)
        feat_dim = feats.shape[2]
        pred_residuals = decoder(
            central_diff2.view(1, -1, 3),
            feats.unsqueeze(2).repeat(1, 1, total_ratio, 1).view(1, -1, feat_dim),
        )
        pred = interpolated2.view(1, -1, 3) + pred_residuals
        pred = denormalize_from_minus1(pred.squeeze(0), shift, max_coord, min_coord)
    return pred.detach().cpu().numpy().astype(np.float32)


def run_chunked(points: np.ndarray, cfg, encoder, decoder, compressor, args, device: str):
    chunks = iter_chunks(points, chunk_size=args.chunk_size, min_points=args.chunk_size)
    outputs = []
    for chunk_id, idx in enumerate(chunks):
        pred = run_chunk(points[idx], cfg, encoder, decoder, compressor, args, device)
        outputs.append(pred)
        print(f"  chunk {chunk_id + 1}/{len(chunks)}: {len(idx)} -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRCIR upsampling-after-compression selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="crcir_aftercomp_4x")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--encoder-downsample", type=int, default=2)
    parser.add_argument("--decoder-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CRCIR CUDA inference requested but CUDA is unavailable.")
    for path in [args.config, args.checkpoint, CRCIR_DEPS]:
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg, encoder, decoder, compressor = load_components(args.config, args.checkpoint, args.device)

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
        enhanced_points = run_chunked(source_points, cfg, encoder, decoder, compressor, args, args.device)
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
