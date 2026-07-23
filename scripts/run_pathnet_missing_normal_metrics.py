"""Compute kNN/PCA normal metrics for one resumable PathNet frame batch."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "UVG-CWI-Metric"))
from metrics import distance_p2p, estimate_normals_knn, load_ply_points  # noqa: E402


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def normal_scores(pred_path: Path, ref_points: np.ndarray, ref_normals: np.ndarray, k: int, batch_size: int):
    points, normals = load_ply_points(str(pred_path))
    if normals is None or len(normals) != len(points):
        normals = estimate_normals_knn(points, k=k, batch_size=batch_size)
    comp = distance_p2p(ref_points, ref_normals, points, normals)[1].mean()
    acc = distance_p2p(points, normals, ref_points, ref_normals)[1].mean()
    return {"N_Acc": float(acc), "N_Comp": float(comp), "normals": float(0.5 * (acc + comp))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method", default="pathnet_chunked_full")
    parser.add_argument("--normal-k", type=int, default=20)
    parser.add_argument("--normal-batch-size", type=int, default=20000)
    parser.add_argument("--normal-cache-root", type=Path)
    parser.add_argument("--run-id", default="legacy")
    args = parser.parse_args()

    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    he_dir = args.dataset_root / args.sequence / "he" / "15fps"
    pred_dir = args.results_root / "method_outputs" / args.method / args.sequence / "15fps"
    cg = {frame_id(path): path for path in cg_dir.glob("*.ply")}
    he = {frame_id(path): path for path in he_dir.glob("*.ply")}
    pred = {frame_id(path): path for path in pred_dir.glob("*.ply")}
    frames = sorted(set(cg) & set(he) & set(pred))
    selected = frames[args.batch_index * args.batch_size:(args.batch_index + 1) * args.batch_size]
    if not selected:
        print(f"[skip] {args.sequence} batch {args.batch_index} has no frames")
        return

    rows = []
    for index, frame in enumerate(selected, 1):
        ref_points, source_normals = load_ply_points(str(he[frame]))
        if args.normal_cache_root is not None:
            cache_path = args.normal_cache_root / "pca_knn20" / args.sequence / "15fps" / f"frame_{frame}.normals.npy"
            if not cache_path.exists():
                raise FileNotFoundError(cache_path)
            ref_normals = np.load(cache_path, allow_pickle=False)
            if ref_normals.shape != (len(ref_points), 3):
                raise ValueError(f"normal cache shape mismatch: {cache_path}")
        elif source_normals is not None and len(source_normals) == len(ref_points):
            ref_normals = source_normals
        else:
            ref_normals = estimate_normals_knn(ref_points, k=args.normal_k, batch_size=args.normal_batch_size)
        for method, path in (("cg_baseline", cg[frame]), (args.method, pred[frame])):
            rows.append({"method": method, "sequence": args.sequence, "frame": frame,
                         "pred_file": str(path), "gt_file": str(he[frame]),
                         **normal_scores(path, ref_points, ref_normals, args.normal_k, args.normal_batch_size)})
        print(f"[{index}/{len(selected)}] {args.sequence} {frame}", flush=True)

    out = args.results_root / "normal_metrics" / args.method / args.run_id / args.sequence / f"batch_{args.batch_index:02d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {out}")
    with out.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
