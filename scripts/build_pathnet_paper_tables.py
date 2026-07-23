"""Validate and build immutable paper-ready PathNet objective-metric tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SEQUENCES = (
    "BlueSpeech", "BlueVolley", "BouncingBlue", "FitFluencer",
    "GoodVision", "Mannequin", "OrangeKettlebell", "PinkNoir",
    "TicTacToe", "TrumanShow", "VictoryHeart", "VirtualLife",
)
KEYS = ("method", "sequence", "frame")
META = {"method", "sequence", "frame", "pred_file", "gt_file", "ref_file"}
HIGHER_IS_BETTER = {
    "N_Acc", "N_Comp", "normals",
    "P_5", "R_5", "F_5", "P_10", "R_10", "F_10", "P_20", "R_20", "F_20",
    "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_mean", "projection_ssim_mean", "pcqm",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="ascii") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[name] for name in KEYS)  # type: ignore[return-value]


def numeric(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {label}: {value!r}")
    return result


def index_unique(rows: list[dict[str, str]], label: str) -> dict[tuple[str, str, str], dict[str, str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        item_key = key(row)
        if item_key in indexed:
            raise RuntimeError(f"duplicate {label} key: {item_key}")
        indexed[item_key] = row
    return indexed


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate(rows: list[dict[str, object]], metrics: list[str], group_names: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[name]) for name in group_names)].append(row)
    output = []
    for group_key, grouped in sorted(groups.items()):
        item: dict[str, object] = dict(zip(group_names, group_key))
        item["frame_count"] = len(grouped)
        for metric in metrics:
            item[metric] = mean([float(row[metric]) for row in grouped])
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--normal-run", required=True)
    parser.add_argument("--texture-run", required=True)
    parser.add_argument("--pcqm-run", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=2152)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to modify non-empty paper table directory: {args.out_dir}")

    geometry_rows: list[dict[str, str]] = []
    normal_rows: list[dict[str, str]] = []
    texture_rows: list[dict[str, str]] = []
    pcqm_rows: list[dict[str, str]] = []
    for sequence in SEQUENCES:
        geometry_rows += read_csv(args.results_root / "uvg_cwi_dqpc" / sequence / args.method / "per_frame_metrics.csv")
        normal_dir = args.results_root / "normal_metrics" / args.method / args.normal_run / sequence
        normal_files = sorted(normal_dir.glob("batch_*.csv"))
        if not normal_files:
            raise FileNotFoundError(normal_dir)
        for path in normal_files:
            normal_rows += read_csv(path)
        texture_rows += read_csv(
            args.results_root / "texture_perceptual_metrics" / sequence / args.method /
            args.texture_run / "per_frame_texture_perceptual_metrics.csv"
        )
        pcqm_rows += read_csv(
            args.results_root / "objective_metrics" / args.method / args.pcqm_run /
            "pcqm" / sequence / "per_frame_pcqm.csv"
        )

    expected_rows = 2 * args.expected_frames
    collections = {
        "geometry": geometry_rows, "normal": normal_rows,
        "texture": texture_rows, "pcqm": pcqm_rows,
    }
    indexed = {name: index_unique(rows, name) for name, rows in collections.items()}
    geometry_keys = set(indexed["geometry"])
    for name, table in indexed.items():
        if len(table) != expected_rows:
            raise RuntimeError(f"{name}: expected {expected_rows} rows, found {len(table)}")
        if set(table) != geometry_keys:
            raise RuntimeError(f"{name}: key coverage differs from geometry")

    merged: list[dict[str, object]] = []
    geometry_metrics = [name for name in geometry_rows[0] if name not in META and name not in {"N_Acc", "N_Comp", "normals"}]
    normal_metrics = ["N_Acc", "N_Comp", "normals"]
    texture_metrics = [
        name for name in texture_rows[0]
        if name not in META and name not in {"point_count", "pcqm"}
    ]
    metrics = geometry_metrics + normal_metrics + texture_metrics + ["pcqm"]
    for item_key in sorted(geometry_keys, key=lambda value: (value[1], value[2], value[0])):
        geometry = indexed["geometry"][item_key]
        normal = indexed["normal"][item_key]
        texture = indexed["texture"][item_key]
        pcqm = indexed["pcqm"][item_key]
        row: dict[str, object] = {name: geometry[name] for name in KEYS}
        row["pred_file"] = geometry["pred_file"]
        row["gt_file"] = geometry["gt_file"]
        for metric in geometry_metrics:
            row[metric] = numeric(geometry[metric], f"geometry {item_key} {metric}")
        for metric in normal_metrics:
            row[metric] = numeric(normal[metric], f"normal {item_key} {metric}")
        for metric in texture_metrics:
            row[metric] = numeric(texture[metric], f"texture {item_key} {metric}")
        row["pcqm"] = numeric(pcqm["pcqm"], f"pcqm {item_key}")
        merged.append(row)

    per_sequence = aggregate(merged, metrics, ("method", "sequence"))
    dataset_mean = aggregate(per_sequence, metrics, ("method",))
    baseline = {str(row["sequence"]): row for row in per_sequence if row["method"] == "cg_baseline"}
    delta_sequence: list[dict[str, object]] = []
    for row in per_sequence:
        if row["method"] == "cg_baseline":
            continue
        base = baseline[str(row["sequence"])]
        delta: dict[str, object] = {
            "method": row["method"], "sequence": row["sequence"], "frame_count": row["frame_count"]
        }
        for metric in metrics:
            method_value, baseline_value = float(row[metric]), float(base[metric])
            delta[f"d{metric}"] = method_value - baseline_value if metric in HIGHER_IS_BETTER else baseline_value - method_value
        delta_sequence.append(delta)
    delta_metrics = [f"d{metric}" for metric in metrics]
    delta_dataset = aggregate(delta_sequence, delta_metrics, ("method",))

    write_csv(args.out_dir / "pathnet_per_frame_all_metrics.csv", merged)
    write_csv(args.out_dir / "pathnet_per_sequence_metrics.csv", per_sequence)
    write_csv(args.out_dir / "pathnet_dataset_sequence_mean.csv", dataset_mean)
    write_csv(args.out_dir / "pathnet_delta_per_sequence.csv", delta_sequence)
    write_csv(args.out_dir / "pathnet_delta_dataset_sequence_mean.csv", delta_dataset)


if __name__ == "__main__":
    main()
