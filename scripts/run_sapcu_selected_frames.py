"""Run SAPCU upsampling on selected UVG-CWI-DQPC frames.

SAPCU's released inference code targets 2048-point object clouds and its dense
seed binary is capped at about 5000 input points. This adapter therefore runs a
fixed-size smoke path: FPS-sample each dense UVG frame to 2048 points, apply the
official 4x generator, FPS the generated cloud to 8192 points, then transfer RGB
from the original CG frame by nearest neighbor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points

import run_mag_selected_frames as common


SAPCU_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "sapcu"


def fps_numpy(points: np.ndarray, count: int, device: str):
    if len(points) <= count:
        return points.astype(np.float32)
    tensor = torch.as_tensor(points[None], dtype=torch.float32, device=device)
    sampled, _idx = sample_farthest_points(tensor, K=count)
    return sampled[0].detach().cpu().numpy().astype(np.float32)


def load_generator(device: str):
    sys.path.insert(0, str(SAPCU_ROOT))
    import fd.checkpoints  # noqa: WPS433
    import fd.config  # noqa: WPS433
    import fn.checkpoints  # noqa: WPS433
    import fn.config  # noqa: WPS433
    from generation import Generator3D6  # noqa: WPS433

    cfg_fn = fn.config.load_config(str(SAPCU_ROOT / "configs" / "fn.yaml"))
    cfg_fd = fd.config.load_config(str(SAPCU_ROOT / "configs" / "fd.yaml"))
    torch_device = torch.device(device)
    model_fn = fn.config.get_model(cfg_fn, torch_device)
    model_fd = fd.config.get_model(cfg_fd, torch_device)
    fn.checkpoints.CheckpointIO(str(SAPCU_ROOT / "out" / "fn"), model=model_fn).load("model_best.pt")
    fd.checkpoints.CheckpointIO(str(SAPCU_ROOT / "out" / "fd"), model=model_fd).load("model_best.pt")
    model_fn.eval()
    model_fd.eval()
    return Generator3D6(model_fn, model_fd, torch_device)


def normalize_bbox(points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    loc = ((mins + maxs) / 2.0).astype(np.float32)
    scale = float((maxs - mins).max())
    if scale <= 0:
        raise ValueError("Cannot normalize a degenerate SAPCU input.")
    return ((points - loc) / scale).astype(np.float32), loc, scale


def run_sapcu(generator, source_points: np.ndarray, input_points: int, output_points: int, device: str):
    sampled = fps_numpy(source_points, input_points, device)
    norm, loc, scale = normalize_bbox(sampled)

    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="sapcu_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "dense").symlink_to(SAPCU_ROOT / "dense")
        try:
            os.chdir(tmp_path)
            np.savetxt("test.xyz", norm, fmt="%.10f")
            generated = np.asarray(generator.upsample(norm[None]), dtype=np.float32)
        finally:
            os.chdir(old_cwd)

    generated = generated * scale + loc
    generated = fps_numpy(generated, output_points, device)
    return generated.astype(np.float32), sampled.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAPCU selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=common.REPO_ROOT / "results")
    parser.add_argument("--method-name", default="sapcu_4x_2048")
    parser.add_argument("--input-points", type=int, default=2048)
    parser.add_argument("--output-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAPCU CUDA inference requested but CUDA is unavailable.")
    for checkpoint in [SAPCU_ROOT / "out" / "fn" / "model_best.pt", SAPCU_ROOT / "out" / "fd" / "model_best.pt"]:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = load_generator(args.device)

    out_root = args.results_root / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = args.results_root / "uvg_cwi_dqpc" / args.sequence / args.method_name
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
        enhanced_points, sampled_input = run_sapcu(generator, source_points, args.input_points, args.output_points, args.device)
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
