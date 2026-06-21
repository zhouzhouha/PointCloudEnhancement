"""Run Neural Points arbitrary upsampling on selected UVG-CWI-DQPC frames.

The released Neural Points test path is built around 256-point object patches
and a 16x output ratio. This adapter keeps that inference shape: FPS-sample a
dense UVG frame to 2048 points, split it into eight Morton-ordered 256-point
patches, run the released pretrained model, then transfer RGB from CG by k=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points

import run_mag_selected_frames as common


NEURALPOINTS_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "NeuralPoints"
MODEL_ROOT = NEURALPOINTS_ROOT / "model" / "conpu_v6"
CKPT = MODEL_ROOT / "pre_trained" / "v3.pt"


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


def fps_numpy(points: np.ndarray, count: int, device: str):
    if len(points) <= count:
        return points.astype(np.float32)
    tensor = torch.as_tensor(points[None], dtype=torch.float32, device=device)
    sampled, _idx = sample_farthest_points(tensor, K=count)
    return sampled[0].detach().cpu().numpy().astype(np.float32)


def normalize_patch(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True).astype(np.float32)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate NeuralPoints patch.")
    return (centered / scale).astype(np.float32), center, scale


def model_args():
    return Namespace(
        num_point=256,
        training_up_ratio=16,
        testing_up_ratio=16,
        over_sampling_scale=4,
        emb_dims=512,
        pe_out_L=5,
        feature_unfolding_nei_num=4,
        if_bn=0,
        neighbor_k=10,
        mlp_fitting_str="256 128 64",
        glue_neighbor=4,
        proj_neighbor=4,
        if_fix_sample=0,
        if_use_siren=0,
    )


def load_model(device: str):
    sys.path.insert(0, str(MODEL_ROOT))
    sys.path.insert(0, str(NEURALPOINTS_ROOT / "code"))
    sys.path.insert(0, str(NEURALPOINTS_ROOT))
    from network import Net_conpu_v7  # noqa: WPS433

    model = Net_conpu_v7(model_args()).to(device)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_neuralpoints(model, source_points: np.ndarray, input_points: int, device: str):
    sampled = fps_numpy(source_points, input_points, device)
    order = morton_order(sampled)
    if input_points % 256 != 0:
        raise ValueError("NeuralPoints input_points must be a multiple of 256.")

    outputs = []
    for patch_id, start in enumerate(range(0, input_points, 256)):
        patch = sampled[order[start : start + 256]]
        patch_norm, center, scale = normalize_patch(patch)
        tensor = torch.as_tensor(patch_norm[None], dtype=torch.float32, device=device)
        with torch.enable_grad():
            _dense_all, _normals_all, _uv, _query, _query_normals, glued, _glued_normals = model(tensor)
        pred = glued.detach().cpu().numpy()[0].astype(np.float32) * scale + center
        outputs.append(pred.astype(np.float32))
        print(f"  patch {patch_id + 1}/{input_points // 256}: 256 -> {len(pred)}", flush=True)
    return np.concatenate(outputs, axis=0).astype(np.float32), sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuralPoints selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="neuralpoints_16x_2048")
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("NeuralPoints CUDA inference requested but CUDA is unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
        enhanced_points, sampled_input = run_neuralpoints(model, source_points, args.input_points, args.device)
        enhanced_colors = common.transfer_nearest_colors(source_points, source_colors, enhanced_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, enhanced_points, enhanced_colors)
        counts.append({
            "frame": frame,
            "input_points": len(source_points),
            "sampled_input_points": len(sampled_input),
            "output_points": len(enhanced_points),
            "has_color": True,
        })
        print(f"{frame}: {len(source_points)} -> sampled {len(sampled_input)} -> {len(enhanced_points)} points, output={out}", flush=True)

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
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "sampled_input_points", "output_points", "has_color"])
        writer.writeheader()
        writer.writerows(counts)
    common.write_summary(rows, metric_root / "summary_metrics.csv")
    (metric_root / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")


if __name__ == "__main__":
    main()
