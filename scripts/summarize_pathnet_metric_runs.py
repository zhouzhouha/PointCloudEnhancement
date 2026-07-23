"""Validate and summarize the completed PathNet normal and texture metric runs."""

from pathlib import Path

import pandas as pd


def report(name, table, metrics):
    print(name, "rows", len(table), "frames_by_method", table.groupby("method").size().to_dict())
    print(name, "nan_counts", table[metrics].isna().sum().to_dict())
    print(table.groupby("method")[metrics].mean().to_csv())


def main():
    scratch = Path("/scratch/project_465003117/PointCloudEnhancement/full_dataset/results")
    repo = Path(__file__).resolve().parents[1]
    normal_paths = sorted((scratch / "normal_metrics/pathnet_chunked_full").glob("*/*.csv"))
    normals = pd.concat([pd.read_csv(path) for path in normal_paths], ignore_index=True)
    report("normal", normals, ["N_Acc", "N_Comp", "normals"])

    texture_paths = sorted((repo / "results/texture_perceptual_metrics").glob(
        "*/pathnet_chunked_full/run_20010566/per_frame_texture_perceptual_metrics.csv"
    ))
    texture = pd.concat([pd.read_csv(path) for path in texture_paths], ignore_index=True)
    report("texture", texture, [
        "y_psnr", "u_psnr", "v_psnr", "yuv_psnr_mean",
        "projection_ssim_mean", "projection_lpips_mean", "pcqm",
    ])


if __name__ == "__main__":
    main()
