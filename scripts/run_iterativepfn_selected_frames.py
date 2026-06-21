"""Run IterativePFN denoising on selected UVG-CWI-DQPC frames.

This adapter keeps IterativePFN's official pretrained checkpoint and default
test-time denoising parameters, then handles UVG-CWI-DQPC file conversion,
nearest-neighbor RGB transfer (`k=1`), and UVG metric reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import run_mag_selected_frames as common


ITERATIVEPFN_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "IterativePFN"
ITERATIVEPFN_CKPT = ITERATIVEPFN_ROOT / "pretrained" / "denoisenet-ep-99.ckpt"


def load_iterativepfn():
    sys.path.insert(0, str(ITERATIVEPFN_ROOT))
    from models.denoise import DenoiseNet  # noqa: WPS433
    from utils.misc import seed_all  # noqa: WPS433
    from utils.transforms import NormalizeUnitSphere  # noqa: WPS433

    return DenoiseNet, NormalizeUnitSphere, seed_all


def denoise_frame(model, normalize, points, args):
    pcl_noisy = torch.as_tensor(points, dtype=torch.float32, device=args.device)
    pcl_noisy, center, scale = normalize.normalize(pcl_noisy)
    pcl_next = pcl_noisy

    with torch.no_grad():
        model.eval()
        for _ in range(args.niters):
            if args.patch_stitching:
                pcl_next = model.patch_based_denoise(
                    pcl_noisy=pcl_next,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    num_modules_to_use=args.num_modules_to_use,
                )
            else:
                pcl_next = model.patch_based_denoise_without_stitching(
                    pcl_noisy=pcl_next,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    num_modules_to_use=args.num_modules_to_use,
                )

    denoised = pcl_next * scale + center
    return denoised.detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run IterativePFN selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="iterativepfn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument("--num-modules-to-use", type=int, default=None)
    parser.add_argument("--patch-stitching", dest="patch_stitching", action="store_true", default=True)
    parser.add_argument("--no-patch-stitching", dest="patch_stitching", action="store_false")
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--seed-k-alpha", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    if not ITERATIVEPFN_CKPT.exists():
        raise FileNotFoundError(ITERATIVEPFN_CKPT)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("IterativePFN requested CUDA, but CUDA is not available.")

    DenoiseNet, NormalizeUnitSphere, seed_all = load_iterativepfn()
    seed_all(args.seed)
    model = DenoiseNet.load_from_checkpoint(str(ITERATIVEPFN_CKPT), map_location=args.device)
    model = model.to(args.device)

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

        denoised_points = denoise_frame(model, NormalizeUnitSphere, source_points, args)
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
