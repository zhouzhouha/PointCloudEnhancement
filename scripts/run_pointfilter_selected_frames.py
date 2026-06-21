"""Run Pointfilter on selected UVG-CWI-DQPC frames.

This adapter prepares UVG CG frames as `.npy`, calls the released Pointfilter
test routine with its pretrained model and default evaluation settings, then
writes XYZRGB PLY files by copying the nearest original CG color (`k=1`) to
every filtered point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
POINTFILTER_ROOT = REPO_ROOT / "third_party" / "enhancement" / "Pointfilter"
POINTFILTER_MODEL_DIR = POINTFILTER_ROOT / "Summary" / "pre_train_model"
METRIC_DIR = REPO_ROOT / "third_party" / "UVG-CWI-Metric"

sys.path.insert(0, str(METRIC_DIR))
from metrics import eval_pointcloud  # noqa: E402


LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamferL2_old", "chamfer-L1", "chamfer-L2"}
HIGHER_IS_BETTER = {
    "N_Acc",
    "N_Comp",
    "normals",
    "P_5",
    "R_5",
    "F_5",
    "P_10",
    "R_10",
    "F_10",
    "P_20",
    "R_20",
    "F_20",
}


def find_frame(directory: Path, frame: str) -> Path:
    matches = sorted(directory.glob(f"*_{frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def read_uvg_xyzrgb(ply_path: Path):
    with ply_path.open("rb") as handle:
        vertex_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {ply_path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {ply_path}")

        record = struct.Struct("<dddBBB")
        points = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 3), dtype=np.uint8)
        for idx in range(vertex_count):
            x, y, z, r, g, b = record.unpack(handle.read(record.size))
            points[idx] = (x, y, z)
            colors[idx] = (r, g, b)
    return points, colors


def write_xyzrgb_ply(path: Path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            handle.write(f"{x:.8f} {y:.8f} {z:.8f} {int(r)} {int(g)} {int(b)}\n")


def read_ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {path}")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])


def transfer_nearest_colors(source_points, source_colors, target_points):
    tree = cKDTree(source_points)
    _distances, indices = tree.query(target_points, k=1, workers=8)
    return source_colors[indices]


def compare_metric(metric: str, baseline: float, method: float):
    if not math.isfinite(float(baseline)) or not math.isfinite(float(method)):
        return float("nan"), False
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


def write_summary(rows, summary_csv: Path):
    metrics = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    methods = sorted({row["method"] for row in rows})
    with summary_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "mean", "std", "count"])
        writer.writeheader()
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            for metric in metrics:
                values = np.array([float(row[metric]) for row in method_rows], dtype=float)
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": np.nanmean(values),
                        "std": np.nanstd(values),
                        "count": len(values),
                    }
                )


def run_pointfilter(test_root: Path, save_root: Path, shape_base: str, args):
    sys.path.insert(0, str(POINTFILTER_ROOT))
    import torch  # noqa: PLC0415
    from Pointfilter_DataLoader import PointcloudPatchDataset  # noqa: PLC0415
    from Pointfilter_Network_Architecture import pointfilternet  # noqa: PLC0415

    def collate_skip_none(batch):
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    with (test_root / "test.txt").open("r", encoding="ascii") as handle:
        shape_names = [line.strip() for line in handle if line.strip()]
    save_root.mkdir(parents=True, exist_ok=True)

    model_filename = args.eval_dir / "model_full_ae.pth"
    checkpoint = torch.load(model_filename)
    pointfilter_eval = pointfilternet().cuda()
    pointfilter_eval.load_state_dict(checkpoint["state_dict"])
    pointfilter_eval.eval()

    for shape_name in shape_names:
        print(shape_name, flush=True)
        original_noise_pts = np.load(test_root / f"{shape_name}.npy")
        np.save(save_root / f"{shape_name}_pred_iter_0.npy", original_noise_pts.astype("float32"))
        for eval_index in range(args.eval_iter_nums):
            print(eval_index, flush=True)
            test_dataset = PointcloudPatchDataset(
                root=str(save_root),
                shape_name=f"{shape_name}_pred_iter_{eval_index}",
                patch_radius=args.patch_radius,
                train_state="evaluation",
            )
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=int(args.workers),
                collate_fn=collate_skip_none,
            )

            patch_radius = test_dataset.patch_radius_absolute
            pred_chunks = []
            skipped_batches = 0
            for data_tuple in test_dataloader:
                if data_tuple is None:
                    skipped_batches += 1
                    continue
                noise_patch, noise_inv, noise_disp = data_tuple
                noise_patch = noise_patch.float().cuda()
                noise_inv = noise_inv.float().cuda()
                noise_patch = noise_patch.transpose(2, 1).contiguous()
                with torch.no_grad():
                    predict = pointfilter_eval(noise_patch)
                    predict = predict.unsqueeze(2)
                    predict = torch.bmm(noise_inv, predict)
                pred_chunks.append(np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy())

            if not pred_chunks:
                raise RuntimeError(f"Pointfilter produced no valid patches for {shape_name} iteration {eval_index}")
            if skipped_batches:
                print(f"[pointfilter] skipped {skipped_batches} empty batches", flush=True)
            pred_pts = np.concatenate(pred_chunks, axis=0).astype("float32")
            np.save(save_root / f"{shape_name}_pred_iter_{eval_index + 1}.npy", pred_pts)

    return save_root / f"{shape_base}_pred_iter_{args.eval_iter_nums}.npy"


def main():
    parser = argparse.ArgumentParser(description="Run Pointfilter selected frames")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--method-name", default="pointfilter")
    parser.add_argument("--eval-dir", type=Path, default=POINTFILTER_MODEL_DIR)
    parser.add_argument("--eval-iter-nums", type=int, default=2)
    parser.add_argument("--patch-radius", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing-output", action="store_true")
    args = parser.parse_args()

    if not (args.eval_dir / "model_full_ae.pth").exists():
        raise FileNotFoundError(f"Missing Pointfilter pretrained model in {args.eval_dir}")

    work_root = REPO_ROOT / "results" / "work" / args.method_name / args.sequence / "15fps"
    input_root = work_root / "input"
    pf_root = work_root / "pointfilter_results"
    out_root = REPO_ROOT / "results" / "method_outputs" / args.method_name / args.sequence / "15fps"
    metric_root = REPO_ROOT / "results" / "uvg_cwi_dqpc" / args.sequence / args.method_name
    for path in [input_root, pf_root, out_root, metric_root]:
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    comparisons = []
    counts = []
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"

    for frame in args.frames:
        cg = find_frame(cg_dir, frame)
        he = find_frame(he_dir, frame)
        source_points, source_colors = read_uvg_xyzrgb(cg)
        shape_base = f"{args.sequence}_frame_{frame}"
        out = out_root / f"frame_{frame}.ply"
        if args.skip_existing_output and out.exists():
            output_points = read_ply_vertex_count(out)
            counts.append(
                {
                    "frame": frame,
                    "input_points": len(source_points),
                    "output_points": output_points,
                    "has_color": True,
                }
            )
            print(f"[pointfilter] frame={frame} reusing existing output={out}", flush=True)
        else:
            frame_input_root = input_root / frame
            frame_save_root = pf_root / frame
            frame_input_root.mkdir(parents=True, exist_ok=True)
            frame_save_root.mkdir(parents=True, exist_ok=True)
            np.save(frame_input_root / f"{shape_base}.npy", source_points.astype("float32"))
            (frame_input_root / "test.txt").write_text(f"{shape_base}\n", encoding="ascii")

            print(f"[pointfilter] frame={frame} input={frame_input_root / (shape_base + '.npy')}", flush=True)
            filtered_npy = run_pointfilter(frame_input_root, frame_save_root, shape_base, args)
            filtered_points = np.load(filtered_npy).astype(np.float32)
            filtered_colors = transfer_nearest_colors(source_points, source_colors, filtered_points)
            write_xyzrgb_ply(out, filtered_points, filtered_colors)
            counts.append(
                {
                    "frame": frame,
                    "input_points": len(source_points),
                    "output_points": len(filtered_points),
                    "has_color": True,
                }
            )
            print(f"{frame}: {len(source_points)} -> {len(filtered_points)} points, output={out}", flush=True)

        baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        filtered = eval_pointcloud(str(out), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
        rows.append({"method": "cg_baseline", "sequence": args.sequence, "frame": frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        rows.append({"method": args.method_name, "sequence": args.sequence, "frame": frame, "pred_file": str(out), "gt_file": str(he), **filtered})
        for metric in baseline:
            delta, improved = compare_metric(metric, baseline[metric], filtered[metric])
            comparisons.append(
                {
                    "frame": frame,
                    "metric": metric,
                    "baseline": baseline[metric],
                    args.method_name: filtered[metric],
                    "delta_for_better": delta,
                    f"{args.method_name}_improved": improved,
                }
            )

    metric_names = [key for key in rows[0] if key not in {"method", "sequence", "frame", "pred_file", "gt_file"}]
    per_frame_csv = metric_root / "per_frame_metrics.csv"
    comparison_csv = metric_root / f"baseline_vs_{args.method_name}_by_frame.csv"
    summary_csv = metric_root / "summary_metrics.csv"
    counts_csv = metric_root / "point_counts.csv"
    config_json = metric_root / "run_config.json"

    with per_frame_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerows(rows)
    with comparison_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "metric", "baseline", args.method_name, "delta_for_better", f"{args.method_name}_improved"],
        )
        writer.writeheader()
        writer.writerows(comparisons)
    with counts_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "input_points", "output_points", "has_color"])
        writer.writeheader()
        writer.writerows(counts)
    write_summary(rows, summary_csv)
    config_json.write_text(json.dumps(vars(args), indent=2, default=str), encoding="ascii")

    print(f"Per-frame metrics: {per_frame_csv}")
    print(f"Comparison metrics: {comparison_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"Point counts: {counts_csv}")


if __name__ == "__main__":
    main()
