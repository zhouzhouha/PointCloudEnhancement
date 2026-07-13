"""Build a UVG-CWI-DQPC dataset manifest for full-dataset benchmarking."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_SEQUENCES = {
    "BlueSpeech": 169,
    "BlueVolley": 171,
    "BouncingBlue": 157,
    "FitFluencer": 201,
    "GoodVision": 168,
    "Mannequin": 188,
    "OrangeKettlebell": 170,
    "PinkNoir": 201,
    "TicTacToe": 165,
    "TrumanShow": 171,
    "VictoryHeart": 197,
    "VirtualLife": 196,
}


def frame_id(path: Path) -> str:
    return path.stem.split("_")[-1]


def find_frames(directory: Path) -> dict[str, Path]:
    return {frame_id(path): path for path in sorted(directory.glob("*.ply"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/full_dataset/uvg_cwi_dqpc_he15_cgv2_15_manifest.csv"),
    )
    args = parser.parse_args()

    rows = []
    for sequence, expected_count in EXPECTED_SEQUENCES.items():
        cg_dir = args.dataset_root / sequence / "cg" / "15fps"
        he_dir = args.dataset_root / sequence / "he" / "15fps"
        cg_frames = find_frames(cg_dir) if cg_dir.exists() else {}
        he_frames = find_frames(he_dir) if he_dir.exists() else {}
        common_frames = sorted(set(cg_frames) & set(he_frames))
        missing_cg = sorted(set(he_frames) - set(cg_frames))
        missing_he = sorted(set(cg_frames) - set(he_frames))
        for frame in common_frames:
            rows.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "cg_v2_file": str(cg_frames[frame]),
                    "he_file": str(he_frames[frame]),
                    "expected_sequence_frames": expected_count,
                    "sequence_common_frames": len(common_frames),
                    "sequence_cg_frames": len(cg_frames),
                    "sequence_he_frames": len(he_frames),
                    "missing_cg_count": len(missing_cg),
                    "missing_he_count": len(missing_he),
                }
            )
        print(
            f"{sequence}: common={len(common_frames)} cg={len(cg_frames)} he={len(he_frames)} "
            f"expected={expected_count} missing_cg={len(missing_cg)} missing_he={len(missing_he)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "frame",
        "cg_v2_file",
        "he_file",
        "expected_sequence_frames",
        "sequence_common_frames",
        "sequence_cg_frames",
        "sequence_he_frames",
        "missing_cg_count",
        "missing_he_count",
    ]
    with args.out.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} frame pairs to {args.out}")


if __name__ == "__main__":
    main()
