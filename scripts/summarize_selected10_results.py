"""Create a compact selected-10 benchmark summary for notes/Overleaf."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOM_ROOT = REPO_ROOT / "results" / "uvg_cwi_dqpc" / "OrangeKettlebell"
TEXTURE_ROOT = REPO_ROOT / "results" / "texture_perceptual_metrics" / "OrangeKettlebell"
OUT_CSV = REPO_ROOT / "results" / "selected10_method_summary.csv"
OUT_MD = REPO_ROOT / "skills" / "methods" / "SELECTED10_RESULTS_SUMMARY.md"


METHODS = [
    ("PointCleanNet", "Denoising", "pointcleannet_selected10", "Candidate"),
    ("Pointfilter", "Denoising", "pointfilter_selected10", "Negative baseline"),
    ("MAG", "Denoising", "mag_selected10", "Candidate"),
    ("Score-Denoise", "Denoising", "score_denoise_selected10", "Candidate"),
    ("GQE-Net", "Color / texture enhancement", "gqenet_selected10", "Candidate for texture"),
    ("Octree upsample-clean", "Joint upsampling / cleaning", "upsample_clean_ounet_selected10", "Candidate"),
    ("PU-Flow", "Upsampling", "puflow_discrete_selected10", "Candidate"),
    ("SPU", "Upsampling", "spu_pointnet_4x_selected10", "Candidate"),
    ("RepKPU", "Upsampling", "repkpu_pu1k_4x_selected10", "Candidate"),
    ("SnowflakeNet-PU", "Upsampling", "snowflakenet_pu_4x_selected10", "Candidate"),
    ("PU-Gaussian", "Upsampling", "pu_gaussian_pu1k_4x_selected10", "Candidate"),
    ("Grad-PU chunked", "Upsampling", "gradpu_chunked_4x_selected10", "Candidate / non-default chunked"),
    ("PUFM", "Upsampling", "pufm_pugan_4x_selected10", "Candidate"),
    ("PUDM", "Upsampling", "pudm_pu1k_4x_selected10", "Mixed / likely negative"),
    ("SPU-PMD", "Upsampling", "spupmd_pu1k_4x_selected10", "Candidate"),
    ("PUCRN", "Upsampling", "pucrn_pu1k_4x_selected10", "Candidate"),
    ("CRCIR after-compression 4x", "Compression-derived enhancement", "crcir_aftercomp_4x_selected10", "Candidate"),
    ("BPA", "Traditional reconstruction", "traditional_bpa", "Optional baseline"),
    ("SPSR", "Traditional reconstruction", "traditional_spsr", "Optional baseline"),
    ("SOR", "Traditional filtering", "geometry_filter_sor", "Negative baseline"),
    ("SAL", "Implicit reconstruction", "sal_selected10", "Negative"),
    ("Points2Surf", "Implicit reconstruction", "points2surf", "Negative"),
    ("PoinTr", "Completion", "pointr_shapenet55", "Negative domain transfer"),
]

GEOMETRY_METRICS = ["CD_Acc", "CD_Comp", "chamfer-L1", "F_10", "F_20"]
TEXTURE_METRICS = ["yuv_psnr_mean", "projection_ssim_mean", "projection_lpips_mean", "pcqm"]
LOWER_IS_BETTER = {"CD_Acc", "CD_Comp", "chamfer-L1", "chamfer-L2", "chamferL2_old", "projection_lpips_mean"}


def read_summary(path: Path):
    if not path.exists():
        return {}
    data = {}
    with path.open("r", newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            data.setdefault(row["method"], {})[row["metric"]] = float(row["mean"])
    return data


def read_texture_summaries(root: Path):
    data = {}
    for path in sorted(root.glob("*/summary_texture_perceptual_metrics.csv")):
        if path.parent.name == "smoke_0000":
            continue
        for method, values in read_summary(path).items():
            # Keep the baseline once; method names are unique across selected-10 runs.
            data.setdefault(method, values)
    return data


def method_values(summary, method_dir: str):
    if method_dir in summary:
        return summary[method_dir]
    for name, values in summary.items():
        if name != "cg_baseline":
            return values
    return {}


def delta(metric: str, baseline: float | None, value: float | None):
    if baseline is None or value is None:
        return None
    return baseline - value if metric in LOWER_IS_BETTER else value - baseline


def fmt(value):
    if value is None:
        return ""
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def main():
    texture = read_texture_summaries(TEXTURE_ROOT)
    rows = []
    for display_name, category, method_dir, decision in METHODS:
        geom_path = GEOM_ROOT / method_dir / "summary_metrics.csv"
        geom = read_summary(geom_path)
        baseline_geom = geom.get("cg_baseline", {})
        method_geom = method_values(geom, method_dir)
        baseline_texture = texture.get("cg_baseline", {})
        method_texture = texture.get(method_dir, {})

        row = {
            "method": display_name,
            "category": category,
            "method_dir": method_dir,
            "decision": decision,
            "geometry_summary": str(geom_path) if geom_path.exists() else "",
        }
        for metric in GEOMETRY_METRICS:
            base = baseline_geom.get(metric)
            val = method_geom.get(metric)
            row[f"{metric}_baseline"] = base
            row[f"{metric}_method"] = val
            row[f"{metric}_delta_for_better"] = delta(metric, base, val)
        for metric in TEXTURE_METRICS:
            base = baseline_texture.get(metric)
            val = method_texture.get(metric)
            row[f"{metric}_baseline"] = base
            row[f"{metric}_method"] = val
            row[f"{metric}_delta_for_better"] = delta(metric, base, val)
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Selected-10 Benchmark Summary",
        "",
        "Scope: `OrangeKettlebell`, frames `0000 0010 0020 0030 0040 0050 0060 0070 0080 0090`.",
        "",
        "Positive `delta` means the method improved over the CG baseline for that metric. Lower-is-better metrics are inverted before computing delta.",
        "",
        "| Method | Category | Decision | dCD-Acc | dCD-Comp | dChamfer-L1 | dF10 | dF20 | dYUV-PSNR | dProj-SSIM | dLPIPS | dPCQM |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {category} | {decision} | {d_cd_acc} | {d_cd_comp} | {d_chamfer} | {d_f10} | {d_f20} | {d_yuv} | {d_ssim} | {d_lpips} | {d_pcqm} |".format(
                method=row["method"],
                category=row["category"],
                decision=row["decision"],
                d_cd_acc=fmt(row["CD_Acc_delta_for_better"]),
                d_cd_comp=fmt(row["CD_Comp_delta_for_better"]),
                d_chamfer=fmt(row["chamfer-L1_delta_for_better"]),
                d_f10=fmt(row["F_10_delta_for_better"]),
                d_f20=fmt(row["F_20_delta_for_better"]),
                d_yuv=fmt(row["yuv_psnr_mean_delta_for_better"]),
                d_ssim=fmt(row["projection_ssim_mean_delta_for_better"]),
                d_lpips=fmt(row["projection_lpips_mean_delta_for_better"]),
                d_pcqm=fmt(row["pcqm_delta_for_better"]),
            )
        )
    lines.extend([
        "",
        "Detailed machine-readable table: `results/selected10_method_summary.csv`.",
        "Texture/perceptual metrics are available only for methods included in the texture metric pass.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
