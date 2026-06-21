"""Prepare one UVG-CWI-DQPC frame for SAL.

The UVG-CWI-DQPC PLY files are binary little-endian with double XYZ and
uchar RGB. SAL only needs geometry for the current configs, so this script
extracts XYZ and writes an .xyz file under SAL/data.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
SAL_ROOT = Path("third_party/SCUTSurface/reconstruction/SAL")


def read_uvg_xyz(ply_path: Path):
    with ply_path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {ply_path}")
            header_lines.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Unsupported PLY format in {ply_path}")

        vertex_count = None
        for line in header_lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {ply_path}")

        record = struct.Struct("<dddBBB")
        points = []
        for _ in range(vertex_count):
            chunk = handle.read(record.size)
            if len(chunk) != record.size:
                raise ValueError(f"Unexpected end of vertex data in {ply_path}")
            x, y, z, _r, _g, _b = record.unpack(chunk)
            points.append((x, y, z))

    return points


def write_xyz(points, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as handle:
        for x, y, z in points:
            handle.write(f"{x:.10f} {y:.10f} {z:.10f}\n")


def limit_points(points, max_points: int | None):
    if max_points is None or len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]


def main():
    parser = argparse.ArgumentParser(description="Prepare one UVG-CWI-DQPC CG frame for SAL")
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sal-root", type=Path, default=SAL_ROOT)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    cg_dir = args.dataset_root / args.sequence / "cg" / "15fps"
    matches = sorted(cg_dir.glob(f"*_{args.frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CG frame for {args.sequence} {args.frame}, found {len(matches)}")

    output_path = (
        args.sal_root
        / "data"
        / "uvg_kettlebell"
        / "points"
        / args.sequence
        / f"frame_{args.frame}.xyz"
    )

    points = limit_points(read_uvg_xyz(matches[0]), args.max_points)
    write_xyz(points, output_path)

    print(f"Input: {matches[0]}")
    print(f"Output: {output_path}")
    print(f"Points: {len(points)}")


if __name__ == "__main__":
    main()
