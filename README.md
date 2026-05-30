# PointCloudEnhancement

This repository provides tools for point cloud quality evaluation and enhancement, built upon the [SCUTSurface benchmark](https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code).

## Overview

The repository includes:
- **Point cloud metrics computation**: Chamfer distance, Hausdorff distance, Precision/Recall/F-score
- **SCUTSurface integration**: Complete surface reconstruction benchmark suite (in `third_party/SCUTSurface/`)
  - Synthetic data generation (object-level and scene-level)
  - Point cloud preprocessing (outlier removal, denoising, resampling)
  - Reconstruction methods (SAL, IGR, LIG, Points2Surf, etc.)
  - Evaluation metrics (vanilla and neural metrics)

## Quick Start

### Setup

Clone the repository with submodules:

```powershell
git clone --recurse-submodules https://github.com/zhouzhouha/PointCloudEnhancement.git
cd PointCloudEnhancement
```

If you already cloned without `--recurse-submodules`, initialize the SCUTSurface submodule:

```powershell
git submodule update --init --recursive
```

### Using the Metrics Launcher
--------

Use the PowerShell launcher `run_metrics.ps1` from the repository root. The script will ensure
the project's virtual environment exists, install `requirements.txt`, and run the metrics.

Examples:

Run a single pair (PLY files):

```powershell
.\run_metrics.ps1 -GtDir "UVG-CWI-DQPC/OrangeKettlebell/CG/15fps/OrangeKettlebell_..._0000.ply" -RecDir "UVG-CWI-DQPC/OrangeKettlebell/HE/15fps/OrangeKettlebell_..._0000.ply" -OutCsv onepair_results.csv -ColorWeight 0
```

Run directories (matching basenames):

```powershell
.\run_metrics.ps1 -GtDir "path\to\gt_dir" -RecDir "path\to\rec_dir" -OutCsv results.csv
```

Notes
-----
- The launcher has a `-ColorWeight` argument (default 0). Set to 0 to ignore color and compute
  distances on XYZ only.
- Use `-OutCsv` to control output path. Use `-Append` to append results to an existing CSV.
- The venv is created at `./venv` and used automatically by the launcher.

## SCUTSurface Integration

The repository includes the complete SCUTSurface benchmark suite as a git submodule in `third_party/SCUTSurface/`.

### Available Tools

**1. Build Dataset**
- **Scan & Synthesis** (object-level and scene-level):
  - Generate synthetic scanned point clouds using BlenSor
  - Support for various artifacts (noise, outliers, missing data, misalignment)
  - See `third_party/SCUTSurface/build_dataset/scan_and_synthesis/`
  
- **Preprocessing**:
  - Outlier removal (PCL required)
  - Denoising (CGAL required)
  - Resampling (FPS)
  - Format conversion (TXT to PLY)
  - See `third_party/SCUTSurface/build_dataset/preprocessing/`

**2. Reconstruction Methods**
- Multiple state-of-the-art methods included as submodules:
  - SAL, IGR, DeepSDF, Occupancy Networks
  - Local Implicit Grid (LIG), Points2Surf, DeepMLS
  - See `third_party/SCUTSurface/reconstruction/`

**3. Evaluation Metrics**
- **Vanilla Metrics**: Chamfer Distance, F-score, Normal Consistency Score
- **Neural Metrics**: Neural Feature Similarity (NFS)
- See `third_party/SCUTSurface/metrics/`

### Usage Examples

For detailed instructions on each module, refer to the README files in the respective subdirectories:
- [Preprocessing Guide](third_party/SCUTSurface/build_dataset/preprocessing/README.md)
- [Object-level Scanning](third_party/SCUTSurface/build_dataset/scan_and_synthesis/object_level/README.md)
- [Scene-level Scanning](third_party/SCUTSurface/build_dataset/scan_and_synthesis/scene_level/README.md)
- [Metrics Evaluation](third_party/SCUTSurface/metrics/README.md)
- [Reconstruction Methods](third_party/SCUTSurface/reconstruction/README.md)

### Citation

If you use SCUTSurface in your research, please cite:
```
@article{huang2024surface,
  title={Surface Reconstruction Benchmark from Point Clouds: A Survey and a Benchmark},
  author={Huang, Zhangjin and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

## License

This project includes code from SCUTSurface (MIT License). See `third_party/SCUTSurface/LICENSE` for details.
