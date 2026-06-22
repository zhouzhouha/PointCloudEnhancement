"""Run PathNet denoising on selected UVG-CWI-DQPC frames.

PathNet's released test path is built around 10K-50K point clouds and applies
PCA normalization to every 128-neighbor patch. This adapter keeps the official
checkpoint/model unchanged, processes dense UVG frames in Morton chunks, then
transfers RGB from the original CG frame by nearest neighbor (k=1).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import spatial

import run_mag_selected_frames as common


PATHNET_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PathNet"
PATHNET_CKPT = PATHNET_ROOT / "log" / "path-denoise" / "model" / "checkpoints" / "best_model.pth"


def morton_order(points: np.ndarray):
    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-6)
    q = np.clip(((points - mins) / spans * 1023.0).astype(np.uint32), 0, 1023)
    codes = np.zeros(len(points), dtype=np.uint32)
    for bit in range(10):
        codes |= ((q[:, 0] >> bit) & 1) << (3 * bit)
        codes |= ((q[:, 1] >> bit) & 1) << (3 * bit + 1)
        codes |= ((q[:, 2] >> bit) & 1) << (3 * bit + 2)
    return np.argsort(codes, kind="mergesort")


def pca_alignment(points: np.ndarray):
    covariance = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return points @ basis, basis.T


def pca_normalize(patches: np.ndarray):
    normalized = np.zeros(patches.shape, dtype=np.float32)
    matrices_inv = np.zeros((patches.shape[0], 3, 3), dtype=np.float32)
    centroids = patches[:, 0:1, :]
    patches = patches - centroids
    scales = np.max(np.sqrt(np.sum(patches * patches, axis=2)), axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    patches = patches / scales[:, None, :]
    for idx in range(patches.shape[0]):
        aligned, matrix_inv = pca_alignment(patches[idx])
        normalized[idx] = aligned
        matrices_inv[idx] = matrix_inv
    return normalized, matrices_inv, scales


def load_pathnet(device: str):
    sys.path.insert(0, str(PATHNET_ROOT / "models"))
    import model as pathnet_model  # noqa: WPS433

    checkpoint = torch.load(PATHNET_CKPT, map_location=device)
    denoiser = pathnet_model.get_model(block_num=6, path_num=2).to(device)
    analyser = pathnet_model.get_analyser(block_num=6, path_num=2).to(device)
    denoiser.load_state_dict(checkpoint["denoiser_model_state_dict"])
    analyser.load_state_dict(checkpoint["analyser_model_state_dict"])
    denoiser.eval()
    analyser.eval()
    return denoiser, analyser


def denoise_points(points: np.ndarray, denoiser, analyser, device: str, batch_size: int, knn: int, iterations: int):
    current = points.astype(np.float32, copy=True)
    block_num = 6
    for _iter_idx in range(iterations):
        tree = spatial.cKDTree(current)
        _dist, indices = tree.query(current, k=min(knn, len(current)), workers=8)
        if indices.ndim == 1:
            indices = indices[:, None]
        if indices.shape[1] < knn:
            pad = np.repeat(indices[:, -1:], knn - indices.shape[1], axis=1)
            indices = np.concatenate([indices, pad], axis=1)
        patches = current[indices]
        normalized, matrices_inv, scales = pca_normalize(patches)

        normalized_trans = []
        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                batch = torch.as_tensor(normalized[start:start + batch_size], dtype=torch.float32, device=device)
                batch = batch.transpose(2, 1).contiguous()
                trans_m, _path_m, _prob_m = denoiser(batch, analyser, 0)
                trans = torch.cat(trans_m, dim=0).reshape(block_num, -1, 3).transpose(1, 0)
                normalized_trans.append(trans.detach().cpu().numpy().astype(np.float32))
        normalized_trans_all = np.concatenate(normalized_trans, axis=0)
        trans_all = np.matmul(matrices_inv, normalized_trans_all.transpose(0, 2, 1)).transpose(0, 2, 1)
        trans_all = trans_all * scales[:, None, :]
        current = current.reshape(-1, 1, 3) - trans_all
        current = current[:, -1, :].astype(np.float32)
    return current


def run_pathnet_large_frame(points: np.ndarray, denoiser, analyser, device: str, chunk_size: int, batch_size: int, knn: int, iterations: int):
    order = morton_order(points)
    output = np.empty_like(points, dtype=np.float32)
    for chunk_idx, start in enumerate(range(0, len(order), chunk_size)):
        idx = order[start:start + chunk_size]
        print(f"chunk {chunk_idx}: denoising {len(idx)} points", flush=True)
        output[idx] = denoise_points(points[idx], denoiser, analyser, device, batch_size, knn, iterations)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PathNet selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pathnet_chunked")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--knn", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PathNet CUDA inference requested but CUDA is unavailable.")
    if not PATHNET_CKPT.exists():
        raise FileNotFoundError(PATHNET_CKPT)

    denoiser, analyser = load_pathnet(args.device)
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
        denoised_points = run_pathnet_large_frame(
            source_points,
            denoiser,
            analyser,
            args.device,
            args.chunk_size,
            args.batch_size,
            args.knn,
            args.iterations,
        )
        denoised_colors = common.transfer_nearest_colors(source_points, source_colors, denoised_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, denoised_points, denoised_colors)
        counts.append({"frame": frame, "input_points": len(source_points), "output_points": len(denoised_points), "has_color": True})
        print(f"{frame}: {len(source_points)} -> {len(denoised_points)} points, output={out}", flush=True)

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
