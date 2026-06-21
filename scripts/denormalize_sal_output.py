"""Denormalize one SAL output mesh back to UVG-CWI-DQPC coordinates."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def load_xyz_stats(path: Path):
    count = 0
    total = [0.0, 0.0, 0.0]
    points = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.strip():
                continue
            x, y, z = [float(v) for v in line.split()[:3]]
            points.append((x, y, z))
            total[0] += x
            total[1] += y
            total[2] += z
            count += 1

    if count == 0:
        raise ValueError(f"No points in {path}")

    center = [v / count for v in total]
    scale = max(abs(coord - center[i]) for point in points for i, coord in enumerate(point))
    return center, scale


def read_binary_sal_ply(path: Path):
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Missing PLY end_header in {path}")
            text = line.decode("ascii").strip()
            header.append(text)
            if text == "end_header":
                break

        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"Unsupported PLY format in {path}")

        vertex_count = None
        face_count = 0
        for line in header:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])

        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")

        vertex_struct = struct.Struct("<fff")
        vertices = [vertex_struct.unpack(handle.read(vertex_struct.size)) for _ in range(vertex_count)]

        faces = []
        for _ in range(face_count):
            count = struct.unpack("<B", handle.read(1))[0]
            if count != 3:
                raise ValueError(f"Only triangular faces are supported, got {count}")
            faces.append(struct.unpack("<iii", handle.read(12)))

    return vertices, faces


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


def write_ascii_pointcloud_ply(path: Path, vertices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in vertices:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Denormalize SAL binary mesh PLY")
    parser.add_argument("--sal-input-xyz", type=Path, required=True)
    parser.add_argument("--sal-output-ply", type=Path, required=True)
    parser.add_argument("--out-mesh-ply", type=Path, required=True)
    parser.add_argument("--out-pointcloud-ply", type=Path, required=True)
    args = parser.parse_args()

    center, scale = load_xyz_stats(args.sal_input_xyz)
    vertices, faces = read_binary_sal_ply(args.sal_output_ply)
    denormalized = [
        (x * scale + center[0], y * scale + center[1], z * scale + center[2])
        for x, y, z in vertices
    ]

    write_ascii_mesh_ply(args.out_mesh_ply, denormalized, faces)
    write_ascii_pointcloud_ply(args.out_pointcloud_ply, denormalized)

    print(f"Center: {center}")
    print(f"Scale: {scale}")
    print(f"Input vertices: {len(vertices)}")
    print(f"Input faces: {len(faces)}")
    print(f"Mesh output: {args.out_mesh_ply}")
    print(f"Point-cloud output: {args.out_pointcloud_ply}")


if __name__ == "__main__":
    main()
