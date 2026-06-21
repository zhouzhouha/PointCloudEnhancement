"""Run PMP-Net++ completion on selected UVG-CWI-DQPC frames.

PMP-Net/PMP-Net++ deform an input point set rather than generating a dense
scene-sized cloud directly. For PCN inference, the official code repeats
2048-point completion passes and reshapes them toward a 16384-point output.
This adapter follows that domain-transfer idea: sample normalized UVG points,
run several 2048-point passes with the official checkpoint, concatenate the
outputs, denormalize, transfer RGB from CG by k=1, and evaluate against HE.
"""

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
from run_pointr_selected_frames import normalize_shapenet, sample_input


PMP_ROOT = common.REPO_ROOT / "third_party" / "enhancement" / "PMP-Net"
DEFAULT_CKPT = PMP_ROOT / "pretrained" / "pcn" / "ckpt-best-pmpplus.pth"


def load_pmpnet(device: str, checkpoint_path: Path, model_variant: str):
    sys.path.insert(0, str(PMP_ROOT))
    old_cwd = Path.cwd()
    try:
        os.chdir(PMP_ROOT)
        from models.model import PMPNet, PMPNetPlus  # noqa: WPS433

        model_cls = PMPNetPlus if model_variant == "pmpnetplus" else PMPNet
        model = model_cls(dataset="ShapeNet")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model
    finally:
        os.chdir(old_cwd)


def complete_frame(model, points: np.ndarray, device: str, n_input: int, n_repeats: int, seed: int):
    sampled_base = sample_input(points, n_input * n_repeats, seed)
    normalized, centroid, scale = normalize_shapenet(sampled_base)
    rng = np.random.default_rng(seed)
    batches = []
    sampled_inputs = []
    for _idx in range(n_repeats):
        indices = rng.choice(len(normalized), size=n_input, replace=len(normalized) < n_input)
        batch = normalized[indices].astype(np.float32)
        batches.append(batch)
        sampled_inputs.append(sampled_base[indices])
    tensor = torch.from_numpy(np.stack(batches, axis=0)).to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model(tensor)[0][-1].detach().cpu().numpy().astype(np.float32)
    completed = outputs.reshape(-1, 3) * scale + centroid
    sampled_inputs = np.concatenate(sampled_inputs, axis=0)
    return completed.astype(np.float32), sampled_inputs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Run PMP-Net++ selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--method-name", default="pmpnetplus_pcn")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--model-variant", choices=["pmpnetplus", "pmpnet"], default="pmpnetplus")
    parser.add_argument("--n-input", type=int, default=2048)
    parser.add_argument("--n-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("PMP-Net smoke requires CUDA.")

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_pmpnet(device, args.checkpoint, args.model_variant)

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
        completed_points, sampled_points = complete_frame(
            model,
            source_points,
            device=device,
            n_input=args.n_input,
            n_repeats=args.n_repeats,
            seed=args.seed + int(frame),
        )
        completed_colors = common.transfer_nearest_colors(source_points, source_colors, completed_points)
        out = out_root / f"frame_{frame}.ply"
        common.write_xyzrgb_ply(out, completed_points, completed_colors)
        counts.append({
            "frame": frame,
            "input_points": len(source_points),
            "sampled_input_points": len(sampled_points),
            "output_points": len(completed_points),
            "has_color": True,
        })
        print(
            f"{frame}: {len(source_points)} -> sample {len(sampled_points)} -> "
            f"{len(completed_points)} points, output={out}",
            flush=True,
        )

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
