"""Audit paired-frame coverage across ordered immutable method-output batches."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


SEQUENCES = (
    "BlueSpeech", "BlueVolley", "BouncingBlue", "FitFluencer",
    "GoodVision", "Mannequin", "OrangeKettlebell", "PinkNoir",
    "TicTacToe", "TrumanShow", "VictoryHeart", "VirtualLife",
)


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def frames(directory: Path) -> dict[str, Path]:
    return {frame_id(path): path for path in sorted(directory.glob("*.ply"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--directory-globs", nargs="+", required=True,
        help="Ordered method-output directory globs; earlier matches take precedence",
    )
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--expected-total", type=int, default=2152)
    parser.add_argument("--allow-duplicates", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    output_root = args.results_root / "method_outputs"
    directories: list[Path] = []
    seen_directories: set[Path] = set()
    for pattern in args.directory_globs:
        for directory in sorted(output_root.glob(pattern)):
            if directory.is_dir() and directory not in seen_directories:
                directories.append(directory)
                seen_directories.add(directory)
    if not directories:
        raise FileNotFoundError(f"no method-output directories matched {args.directory_globs}")

    manifest: list[dict[str, object]] = []
    paired_total = present_total = missing_total = duplicate_total = unexpected_total = 0
    for sequence in SEQUENCES:
        cg = frames(args.dataset_root / sequence / "cg" / "15fps")
        he = frames(args.dataset_root / sequence / "he" / "15fps")
        paired = sorted(set(cg) & set(he))
        paired_set = set(paired)
        sources: dict[str, list[Path]] = defaultdict(list)
        for directory in directories:
            for path in sorted((directory / sequence / "15fps").glob("frame_*.ply")):
                sources[frame_id(path)].append(path)

        present = missing = duplicates = unexpected = 0
        for frame in paired:
            candidates = sources.get(frame, [])
            if candidates:
                present += 1
                duplicates += max(0, len(candidates) - 1)
            else:
                missing += 1
            manifest.append({
                "sequence": sequence,
                "frame": frame,
                "status": "missing" if not candidates else ("duplicate" if len(candidates) > 1 else "present"),
                "selected_file": str(candidates[0]) if candidates else "",
                "alternate_count": max(0, len(candidates) - 1),
                "alternate_files": "|".join(str(path) for path in candidates[1:]),
                "cg_file": str(cg[frame]),
                "he_file": str(he[frame]),
            })
        for frame in sorted(set(sources) - paired_set):
            unexpected += len(sources[frame])
            manifest.append({
                "sequence": sequence,
                "frame": frame,
                "status": "unexpected",
                "selected_file": str(sources[frame][0]),
                "alternate_count": max(0, len(sources[frame]) - 1),
                "alternate_files": "|".join(str(path) for path in sources[frame][1:]),
                "cg_file": "",
                "he_file": "",
            })

        print(
            f"{sequence}: paired={len(paired)} present={present} missing={missing} "
            f"duplicate_alternates={duplicates} unexpected={unexpected}"
        )
        paired_total += len(paired)
        present_total += present
        missing_total += missing
        duplicate_total += duplicates
        unexpected_total += unexpected

    if args.manifest_out is not None:
        if args.manifest_out.exists():
            raise FileExistsError(f"refusing to overwrite manifest: {args.manifest_out}")
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        partial = args.manifest_out.with_name(f".{args.manifest_out.name}.part-{os.getpid()}")
        try:
            with partial.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
            partial.replace(args.manifest_out)
        finally:
            partial.unlink(missing_ok=True)
        print(f"wrote {len(manifest)} rows to {args.manifest_out}")

    print(
        f"TOTAL paired={paired_total} present={present_total} missing={missing_total} "
        f"duplicate_alternates={duplicate_total} unexpected={unexpected_total}"
    )
    if paired_total != args.expected_total:
        raise RuntimeError(f"expected {args.expected_total} paired frames, found {paired_total}")
    if unexpected_total:
        raise RuntimeError(f"found {unexpected_total} outputs outside paired-frame coverage")
    if duplicate_total and not args.allow_duplicates:
        raise RuntimeError(f"found {duplicate_total} alternate duplicate outputs")
    if missing_total and not args.allow_incomplete:
        raise RuntimeError(f"missing {missing_total} paired-frame outputs")


if __name__ == "__main__":
    main()
