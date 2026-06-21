"""Denormalize an IGR ASCII/Binary mesh PLY and write benchmark PLY outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def write_ascii_mesh_ply(path: Path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


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
    parser = argparse.ArgumentParser(description="Denormalize IGR mesh output")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out-mesh", type=Path, required=True)
    parser.add_argument("--out-points", type=Path, required=True)
    parser.add_argument("--sample-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = json.loads(args.normalization.read_text(encoding="ascii"))
    center = np.array(meta["center"], dtype=np.float64)
    scale = float(meta["scale"])
    mesh = trimesh.load(args.mesh, process=False)
    vertices = np.asarray(mesh.vertices) * scale + center
    faces = np.asarray(mesh.faces)
    write_ascii_mesh_ply(args.out_mesh, vertices, faces)
    denorm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, _ = trimesh.sample.sample_surface(denorm_mesh, args.sample_points, seed=args.seed)
    write_pointcloud_ply(args.out_points, points)

    print(f"Mesh: {args.mesh}")
    print(f"Output mesh: {args.out_mesh}")
    print(f"Output points: {args.out_points}")
    print(f"Sampled points: {len(points)}")


if __name__ == "__main__":
    main()
