"""Compare one method output against the CG baseline with UVG-CWI/Metric."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
METRIC_DIR = REPO_ROOT / "third_party" / "UVG-CWI-Metric"

sys.path.insert(0, str(METRIC_DIR))
from metrics import eval_pointcloud  # noqa: E402


LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamferL2_old", "chamfer-L1", "chamfer-L2"}
HIGHER_IS_BETTER = {
    "N_Acc", "N_Comp", "normals",
    "P_5", "R_5", "F_5",
    "P_10", "R_10", "F_10",
    "P_20", "R_20", "F_20",
}


def find_frame(directory: Path, frame: str) -> Path:
    matches = sorted(directory.glob(f"*_{frame}.ply"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one frame {frame} in {directory}, found {len(matches)}")
    return matches[0]


def compare_metric(metric: str, baseline: float, method: float):
    if metric in LOWER_IS_BETTER:
        return baseline - method, method < baseline
    if metric in HIGHER_IS_BETTER:
        return method - baseline, method > baseline
    return method - baseline, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate one method frame against CG baseline")
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequence", default="OrangeKettlebell")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--method-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cg = find_frame(args.dataset_root / args.sequence / "cg" / "15fps", args.frame)
    he = find_frame(args.dataset_root / args.sequence / "he" / "15fps", args.frame)
    method_output = args.method_output
    if not method_output.exists():
        raise FileNotFoundError(method_output)

    baseline = eval_pointcloud(str(cg), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])
    method_metrics = eval_pointcloud(str(method_output), str(he), samplepoint=0, eval_type="ply", thresholds=[5, 10, 20])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_method_csv = args.out_dir / f"frame_{args.frame}_uvg_metric.csv"
    comparison_csv = args.out_dir / f"frame_{args.frame}_baseline_vs_{args.method}.csv"
    config_json = args.out_dir / f"frame_{args.frame}_run_config.json"

    metric_names = list(baseline.keys())
    with per_method_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "sequence", "frame", "pred_file", "gt_file", *metric_names])
        writer.writeheader()
        writer.writerow({"method": "cg_baseline", "sequence": args.sequence, "frame": args.frame, "pred_file": str(cg), "gt_file": str(he), **baseline})
        writer.writerow({"method": args.method, "sequence": args.sequence, "frame": args.frame, "pred_file": str(method_output), "gt_file": str(he), **method_metrics})

    with comparison_csv.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "baseline", args.method, "delta_for_better", f"{args.method}_improved"])
        writer.writeheader()
        for metric in metric_names:
            delta, improved = compare_metric(metric, baseline[metric], method_metrics[metric])
            writer.writerow({
                "metric": metric,
                "baseline": baseline[metric],
                args.method: method_metrics[metric],
                "delta_for_better": delta,
                f"{args.method}_improved": improved,
            })

    config_json.write_text(json.dumps({
        "metric_repo": str(METRIC_DIR),
        "method": args.method,
        "sequence": args.sequence,
        "frame": args.frame,
        "cg": str(cg),
        "he": str(he),
        "method_output": str(method_output),
        "thresholds": [5, 10, 20],
        "per_method_csv": str(per_method_csv),
        "comparison_csv": str(comparison_csv),
    }, indent=2), encoding="ascii")

    print(f"Per-method metrics: {per_method_csv}")
    print(f"Comparison metrics: {comparison_csv}")
    print(f"Run config: {config_json}")
    print()
    for metric in ["CD_Acc", "CD_Comp", "chamfer-L1", "chamfer-L2", "F_5", "F_10", "F_20"]:
        delta, improved = compare_metric(metric, baseline[metric], method_metrics[metric])
        print(f"{metric}: baseline={baseline[metric]} {args.method}={method_metrics[metric]} delta_for_better={delta} improved={improved}")


if __name__ == "__main__":
    main()
