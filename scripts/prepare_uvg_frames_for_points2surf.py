"""Prepare selected UVG-CWI-DQPC frames for Points2Surf reconstruction."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
P2S_ROOT = Path("third_party/SCUTSurface/reconstruction/Points2Surf")


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


def find_frame(directory: Path, frame: str) -> Path:
    matches = sorted(directory.glob(f"*_{frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Prepare UVG frames as Points2Surf .xyz.npy files")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frames", nargs="+", default=[f"{i:04d}" for i in range(0, 100, 10)])
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--max-points", type=int, default=200000)
    parser.add_argument("--out-root", type=Path, default=P2S_ROOT / "datasets" / "uvg_orangekettlebell")
    args = parser.parse_args()

    pts_dir = args.out_root / "04_pts"
    meta_dir = args.out_root / "normalization"
    pts_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    stems = []
    for frame in args.frames:
        source = find_frame(cg_dir, frame)
        points = read_uvg_xyz(source)
        if len(points) > args.max_points:
            indices = np.linspace(0, len(points) - 1, args.max_points).astype(np.int64)
            points = points[indices]

        center = points.mean(axis=0)
        scale = float(np.abs(points - center).max())
        normalized = ((points - center) / scale).astype(np.float32)

        stem = f"frame_{frame}"
        np.save(pts_dir / f"{stem}.xyz.npy", normalized)
        (meta_dir / f"{stem}.json").write_text(json.dumps({
            "source": str(source),
            "center": center.tolist(),
            "scale": scale,
            "points": int(len(points)),
        }, indent=2), encoding="ascii")
        stems.append(stem)
        print(f"{frame}: {source} -> {pts_dir / f'{stem}.xyz.npy'} ({len(points)} points)")

    (args.out_root / "testset.txt").write_text("\n".join(stems) + "\n", encoding="ascii")
    print(f"Test set: {args.out_root / 'testset.txt'}")


if __name__ == "__main__":
    main()
