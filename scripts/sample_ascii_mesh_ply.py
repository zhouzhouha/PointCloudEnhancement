"""Sample points uniformly from an ASCII triangular mesh PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ascii_mesh_ply(path: Path):
    with path.open("r", encoding="ascii") as handle:
        line = handle.readline().strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        vertex_count = None
        face_count = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {path}")
            text = line.strip()
            if text == "format ascii 1.0":
                continue
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
            elif text.startswith("element face "):
                face_count = int(text.split()[-1])
            elif text == "end_header":
                break

        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing vertex or face count in {path}")

        vertices = np.array([[float(v) for v in handle.readline().split()[:3]] for _ in range(vertex_count)])
        faces = []
        for _ in range(face_count):
            parts = handle.readline().split()
            if int(parts[0]) != 3:
                raise ValueError("Only triangular faces are supported")
            faces.append([int(v) for v in parts[1:4]])
        return vertices, np.array(faces, dtype=np.int64)


def sample_surface(vertices, faces, num_points, seed):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > 0
    tri = tri[valid]
    areas = areas[valid]
    probs = areas / areas.sum()

    rng = np.random.default_rng(seed)
    choices = rng.choice(len(tri), size=num_points, p=probs)
    selected = tri[choices]
    u = rng.random(num_points)
    v = rng.random(num_points)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])


def write_pointcloud_ply(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Sample points from an ASCII mesh PLY")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vertices, faces = read_ascii_mesh_ply(args.mesh)
    points = sample_surface(vertices, faces, args.points, args.seed)
    write_pointcloud_ply(args.out, points)
    print(f"Mesh: {args.mesh}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces: {len(faces)}")
    print(f"Sampled points: {len(points)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
