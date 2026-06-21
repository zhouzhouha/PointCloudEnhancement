"""Prepare one UVG-CWI-DQPC frame for IGR reconstruction."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")


def read_uvg_xyz(ply_path: Path):
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
        for idx in range(vertex_count):
            x, y, z, _r, _g, _b = record.unpack(handle.read(record.size))
            points[idx] = (x, y, z)
    return points


def main():
    parser = argparse.ArgumentParser(description="Prepare normalized UVG frame for IGR")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--out-dir", type=Path, default=Path("third_party/SCUTSurface/reconstruction/IGR/data/uvg_kettlebell"))
    args = parser.parse_args()

    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    matches = sorted(cg_dir.glob(f"*_{args.frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CG frame for {args.sequence} {args.frame}, found {len(matches)}")

    points = read_uvg_xyz(matches[0])
    if len(points) > args.max_points:
        index = np.linspace(0, len(points) - 1, args.max_points).astype(np.int64)
        points = points[index]

    center = points.mean(axis=0)
    scale = np.abs(points - center).max()
    normalized = (points - center) / scale

    points_dir = args.out_dir / "points" / args.sequence
    points_dir.mkdir(parents=True, exist_ok=True)
    npy_path = points_dir / f"frame_{args.frame}.npy"
    meta_path = points_dir / f"frame_{args.frame}_normalization.json"
    np.save(npy_path, normalized.astype(np.float32))
    meta_path.write_text(json.dumps({
        "source": str(matches[0]),
        "center": center.tolist(),
        "scale": float(scale),
        "points": int(len(points)),
    }, indent=2), encoding="ascii")

    print(f"Input: {matches[0]}")
    print(f"Output: {npy_path}")
    print(f"Metadata: {meta_path}")
    print(f"Points: {len(points)}")
    print(f"Center: {center.tolist()}")
    print(f"Scale: {float(scale)}")


if __name__ == "__main__":
    main()
