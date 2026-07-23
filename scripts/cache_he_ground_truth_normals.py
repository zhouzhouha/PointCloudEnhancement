"""Persist reusable HE ground-truth PCA normals; never cache method normals."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "UVG-CWI-Metric"))
from metrics import estimate_normals_knn, load_ply_points  # noqa: E402


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()
    if args.neighbors != 20:
        raise ValueError("This benchmark's paper-supported PCA normal protocol uses 20 neighbors")

    he_dir = args.dataset_root / args.sequence / "he" / "15fps"
    sources = sorted(he_dir.glob("*.ply"), key=frame_id)
    if not sources:
        raise FileNotFoundError(f"no HE PLY files in {he_dir}")
    target_dir = args.cache_root / "pca_knn20" / args.sequence / "15fps"
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(sources, 1):
        frame = frame_id(source)
        normal_path = target_dir / f"frame_{frame}.normals.npy"
        metadata_path = target_dir / f"frame_{frame}.json"
        if normal_path.exists() and metadata_path.exists():
            normals = np.load(normal_path, mmap_mode="r")
            metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            if normals.shape == (metadata["point_count"], 3) and metadata["neighbors"] == 20:
                print(f"[{index}/{len(sources)}] reuse {args.sequence} {frame}", flush=True)
                continue
            raise RuntimeError(f"invalid existing cache entry: {normal_path}")
        if normal_path.exists() or metadata_path.exists():
            raise RuntimeError(f"partial cache entry; refusing overwrite: {normal_path}")

        points, normals = load_ply_points(str(source))
        source_has_normals = normals is not None and len(normals) == len(points)
        if not source_has_normals:
            normals = estimate_normals_knn(points, k=20, batch_size=args.batch_size)
        normals = np.asarray(normals, dtype=np.float32)
        tmp_normal = normal_path.with_suffix(normal_path.suffix + f".{os.getpid()}.tmp")
        with tmp_normal.open("wb") as handle:
            np.save(handle, normals, allow_pickle=False)
        tmp_normal.replace(normal_path)
        metadata = {
            "sequence": args.sequence,
            "frame": frame,
            "source": str(source),
            "point_count": int(len(points)),
            "estimator": "knn_pca_smallest_eigenvector",
            "neighbors": 20,
            "dtype": "float32",
            "source_had_normals": bool(source_has_normals),
        }
        tmp_metadata = metadata_path.with_suffix(metadata_path.suffix + f".{os.getpid()}.tmp")
        tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
        tmp_metadata.replace(metadata_path)
        print(f"[{index}/{len(sources)}] cached {args.sequence} {frame}", flush=True)


if __name__ == "__main__":
    main()
