"""Evaluate color/texture and projection perceptual metrics for UVG-CWI-DQPC.

Metrics:
- YUV PSNR on nearest-neighbor color correspondence from reference HE to test.
- Six-view orthographic projection SSIM.
- Six-view orthographic projection LPIPS.
- Optional PCQM via the external MEPP-team/PCQM binary.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

import run_mag_selected_frames as common


REPO_ROOT = Path(__file__).resolve().parents[1]
PCQM_BIN = Path(os.environ.get(
    "PCE_PCQM_BIN",
    REPO_ROOT / "third_party" / "metrics" / "PCQM" / "build" / "PCQM",
))
TYPE_FORMATS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}
VIEWS = {
    "x_pos": (0, 1, 2, 1.0),
    "x_neg": (0, 1, 2, -1.0),
    "y_pos": (1, 0, 2, 1.0),
    "y_neg": (1, 0, 2, -1.0),
    "z_pos": (2, 0, 1, 1.0),
    "z_neg": (2, 0, 1, -1.0),
}


def read_xyzrgb_ply(path: Path):
    with path.open("rb") as handle:
        fmt = None
        vertex_count = None
        props = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Missing PLY end_header in {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"List vertex properties are unsupported in {path}")
                props.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise ValueError(f"Missing PLY format or vertex count in {path}")

        names = [name for name, _typ in props]
        color_names = []
        for candidates in [("red", "r"), ("green", "g"), ("blue", "b")]:
            for candidate in candidates:
                if candidate in names:
                    color_names.append(candidate)
                    break
        needed = ["x", "y", "z", *color_names]
        missing = [name for name in needed if name not in names]
        if missing:
            raise ValueError(f"Missing PLY properties {missing} in {path}")

        if fmt == "ascii":
            data = np.loadtxt(handle, dtype=np.float64, max_rows=vertex_count)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            points = data[:, [names.index("x"), names.index("y"), names.index("z")]].astype(np.float32)
            colors = data[:, [names.index(c) for c in color_names]].astype(np.float32)
        elif fmt == "binary_little_endian":
            record = struct.Struct("<" + "".join(TYPE_FORMATS[typ] for _name, typ in props))
            points = np.empty((vertex_count, 3), dtype=np.float32)
            colors = np.empty((vertex_count, 3), dtype=np.float32)
            xyz_idx = [names.index("x"), names.index("y"), names.index("z")]
            color_idx = [names.index(c) for c in color_names]
            for idx in range(vertex_count):
                values = record.unpack(handle.read(record.size))
                points[idx] = [values[i] for i in xyz_idx]
                colors[idx] = [values[i] for i in color_idx]
        else:
            raise ValueError(f"Unsupported PLY format {fmt} in {path}")

    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    return points, colors


def write_ascii_xyzrgb_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    common.write_xyzrgb_ply(path, points.astype(np.float32), colors.astype(np.uint8))


def rgb_to_yuv_bt709(colors: np.ndarray):
    rgb = colors.astype(np.float64)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = (b - y) / (2.0 * (1.0 - 0.0722)) + 128.0
    v = (r - y) / (2.0 * (1.0 - 0.2126)) + 128.0
    return np.stack([y, u, v], axis=1)


def psnr_from_mse(mse: float, peak: float = 255.0):
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


def yuv_psnr_nn(pred_points, pred_colors, ref_points, ref_colors):
    indices = cKDTree(ref_points).query(pred_points, k=1, workers=8)[1]
    pred_yuv = rgb_to_yuv_bt709(pred_colors)
    ref_yuv = rgb_to_yuv_bt709(ref_colors[indices])
    mse = np.mean((pred_yuv - ref_yuv) ** 2, axis=0)
    return {
        "y_psnr": psnr_from_mse(float(mse[0])),
        "u_psnr": psnr_from_mse(float(mse[1])),
        "v_psnr": psnr_from_mse(float(mse[2])),
        "yuv_psnr_mean": float(np.mean([psnr_from_mse(float(x)) for x in mse])),
    }


def project_view(points, colors, view, bounds, image_size):
    depth_axis, u_axis, v_axis, sign = VIEWS[view]
    u_min, u_max, v_min, v_max = bounds
    eps = 1e-8
    u = (points[:, u_axis] - u_min) / max(u_max - u_min, eps)
    v = (points[:, v_axis] - v_min) / max(v_max - v_min, eps)
    x = np.clip(np.floor(u * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    y = np.clip(np.floor((1.0 - v) * (image_size - 1)).astype(np.int64), 0, image_size - 1)
    depth = sign * points[:, depth_axis]
    order = np.argsort(depth)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    image[y[order], x[order]] = colors[order]
    mask[y[order], x[order]] = 255
    return image, mask


def view_bounds(a_points, b_points, view):
    _depth_axis, u_axis, v_axis, _sign = VIEWS[view]
    points = np.vstack([a_points[:, [u_axis, v_axis]], b_points[:, [u_axis, v_axis]]])
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.02, 1e-6)
    return mins[0] - pad[0], maxs[0] + pad[0], mins[1] - pad[1], maxs[1] + pad[1]


def image_to_lpips_tensor(image, device):
    arr = image.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def projection_metrics(pred_points, pred_colors, ref_points, ref_colors, image_size, lpips_model, device, render_dir=None, prefix=""):
    rows = []
    for view in VIEWS:
        bounds = view_bounds(pred_points, ref_points, view)
        pred_img, pred_mask = project_view(pred_points, pred_colors, view, bounds, image_size)
        ref_img, ref_mask = project_view(ref_points, ref_colors, view, bounds, image_size)
        ssim = structural_similarity(ref_img, pred_img, channel_axis=2, data_range=255)
        if lpips_model is None:
            lpips_value = float("nan")
        else:
            with torch.no_grad():
                lpips_value = float(lpips_model(image_to_lpips_tensor(ref_img, device), image_to_lpips_tensor(pred_img, device)).item())
        if render_dir is not None:
            from PIL import Image
            render_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ref_img).save(render_dir / f"{prefix}_{view}_ref.png")
            Image.fromarray(pred_img).save(render_dir / f"{prefix}_{view}_pred.png")
            Image.fromarray(ref_mask).save(render_dir / f"{prefix}_{view}_ref_mask.png")
            Image.fromarray(pred_mask).save(render_dir / f"{prefix}_{view}_pred_mask.png")
        rows.append({"view": view, "projection_ssim": float(ssim), "projection_lpips": lpips_value})
    return rows


def run_pcqm(pred_points, pred_colors, ref_points, ref_colors, tmp_root: Path, stem: str):
    if not PCQM_BIN.exists():
        return float("nan")
    ref_ply = tmp_root / f"{stem}_ref.ply"
    pred_ply = tmp_root / f"{stem}_pred.ply"
    write_ascii_xyzrgb_ply(ref_ply, ref_points, ref_colors)
    write_ascii_xyzrgb_ply(pred_ply, pred_points, pred_colors)
    resource_root = PCQM_BIN.parent
    resource_names = (
        "L_data.txt",
        "RegularGridInit_0_0_1.txt",
        "RegularGridInit_0_0_2.txt",
        "RegularGrid_0_0_1.txt",
        "RegularGrid_0_0_2.txt",
    )
    for name in resource_names:
        target = tmp_root / name
        if not target.exists():
            source = resource_root / name
            if not source.exists():
                source = resource_root.parent / "resources" / name
            if not source.exists():
                raise FileNotFoundError(f"missing PCQM resource: {name}")
            shutil.copy2(source, target)
    # The upstream README documents "reference registered", but main.cpp assigns
    # argv[2] to the reference and argv[1] to the registered/distorted cloud.
    # PCQM always writes features_extracted.csv in its working directory.
    # Use the task-private temporary directory so concurrent jobs cannot race.
    subprocess.run([str(PCQM_BIN), str(pred_ply), str(ref_ply), "-r", "0.004", "-knn", "20", "-rx", "2.0"], cwd=str(tmp_root), check=True)
    csv_path = tmp_root / "features_extracted.csv"
    if not csv_path.exists():
        return float("nan")
    rows = list(csv.DictReader(csv_path.open(encoding="ascii"), delimiter=";"))
    if not rows:
        return float("nan")
    return float(rows[-1]["PCQM"])


def summarize(rows, out_csv: Path):
    metric_names = [k for k in rows[0] if k not in {"method", "sequence", "frame", "pred_file", "ref_file"}]
    methods = sorted({r["method"] for r in rows})
    with out_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "mean", "std", "count"])
        writer.writeheader()
        for method in methods:
            method_rows = [r for r in rows if r["method"] == method]
            for metric in metric_names:
                values = np.array([float(r[metric]) for r in method_rows], dtype=float)
                writer.writerow({"method": method, "metric": metric, "mean": np.nanmean(values), "std": np.nanstd(values), "count": len(values)})


def main():
    parser = argparse.ArgumentParser(description="Run texture/perceptual metrics")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=["0000"])
    parser.add_argument("--methods", nargs="+", required=True, help="Method names under results/method_outputs")
    parser.add_argument("--dataset-root", type=Path, default=common.DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results",
                        help="Root containing method_outputs; defaults to the repository results directory.")
    parser.add_argument("--run-name", default="selected10")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Directory for large temporary PCQM PLY files; defaults to $TMPDIR if set.")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--skip-pcqm", action="store_true")
    parser.add_argument("--save-renders", action="store_true")
    args = parser.parse_args()

    out_root = args.results_root / "texture_perceptual_metrics" / args.sequence / args.run_name
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    render_root = out_root / "renders" if args.save_renders else None

    lpips_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.skip_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    he_dir = args.dataset_root / args.sequence / "he" / "15fps"
    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    all_rows = []
    view_rows = []
    tmp_base = args.tmp_dir or Path(os.environ.get("TMPDIR", out_root))
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix="pcqm_", dir=str(tmp_base)))
    try:
        for frame in args.frames:
            ref_path = common.find_frame(he_dir, frame)
            ref_points, ref_colors = read_xyzrgb_ply(ref_path)
            candidates = [("cg_baseline", common.find_frame(cg_dir, frame))]
            for method in args.methods:
                candidates.append((method, args.results_root / "method_outputs" / method / args.sequence / "15fps" / f"frame_{frame}.ply"))
            for method, pred_path in candidates:
                if not pred_path.exists():
                    print(f"[skip] missing {method} frame {frame}: {pred_path}", flush=True)
                    continue
                pred_points, pred_colors = read_xyzrgb_ply(pred_path)
                row = {
                    "method": method,
                    "sequence": args.sequence,
                    "frame": frame,
                    "pred_file": str(pred_path),
                    "ref_file": str(ref_path),
                    "point_count": len(pred_points),
                }
                row.update(yuv_psnr_nn(pred_points, pred_colors, ref_points, ref_colors))
                projections = projection_metrics(
                    pred_points,
                    pred_colors,
                    ref_points,
                    ref_colors,
                    args.image_size,
                    lpips_model,
                    device,
                    render_root,
                    f"{method}_{frame}",
                )
                row["projection_ssim_mean"] = float(np.nanmean([r["projection_ssim"] for r in projections]))
                row["projection_lpips_mean"] = float(np.nanmean([r["projection_lpips"] for r in projections]))
                if args.skip_pcqm:
                    row["pcqm"] = float("nan")
                else:
                    row["pcqm"] = run_pcqm(pred_points, pred_colors, ref_points, ref_colors, tmp_parent, f"{method}_{frame}")
                all_rows.append(row)
                for projection in projections:
                    view_rows.append({"method": method, "sequence": args.sequence, "frame": frame, **projection})
                print(f"{method} {frame}: Y={row['y_psnr']:.4f}, SSIM={row['projection_ssim_mean']:.6f}, LPIPS={row['projection_lpips_mean']:.6f}, PCQM={row['pcqm']}", flush=True)
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)

    if not all_rows:
        raise RuntimeError("No metric rows were produced.")
    fieldnames = list(all_rows[0].keys())
    with (out_root / "per_frame_texture_perceptual_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    with (out_root / "per_view_projection_metrics.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "view", "projection_ssim", "projection_lpips"])
        writer.writeheader()
        writer.writerows(view_rows)
    summarize(all_rows, out_root / "summary_texture_perceptual_metrics.csv")


if __name__ == "__main__":
    main()
