"""Run isolated PCQM scoring for PathNet and its CG baseline."""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
from pathlib import Path

from run_texture_perceptual_metrics import read_xyzrgb_ply, run_pcqm


def frame_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--method", default="pathnet_chunked_full")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    he = {frame_id(path): path for path in (args.dataset_root / args.sequence / "he/15fps").glob("*.ply")}
    cg = {frame_id(path): path for path in (args.dataset_root / args.sequence / "cg/15fps").glob("*.ply")}
    pred = {frame_id(path): path for path in (args.results_root / "method_outputs" / args.method / args.sequence / "15fps").glob("*.ply")}
    frames = sorted(set(he) & set(cg) & set(pred))
    if args.max_frames is not None:
        frames = frames[:args.max_frames]
    if not frames:
        raise RuntimeError(f"no common frames for {args.sequence}")
    out = args.results_root / "objective_metrics" / args.method / args.run_id / "pcqm" / args.sequence / "per_frame_pcqm.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")

    rows = []
    with tempfile.TemporaryDirectory(prefix=f"pcqm_{args.sequence}_", dir=args.tmp_dir) as tmp:
        tmp_path = Path(tmp)
        for index, frame in enumerate(frames, 1):
            ref_points, ref_colors = read_xyzrgb_ply(he[frame])
            for method, path in (("cg_baseline", cg[frame]), (args.method, pred[frame])):
                points, colors = read_xyzrgb_ply(path)
                value = run_pcqm(points, colors, ref_points, ref_colors, tmp_path, f"{method}_{frame}")
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite PCQM for {method} {args.sequence} {frame}")
                rows.append({"method": method, "sequence": args.sequence, "frame": frame,
                             "pred_file": str(path), "gt_file": str(he[frame]), "pcqm": value})
            print(f"[{index}/{len(frames)}] {args.sequence} {frame}", flush=True)
    with out.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
