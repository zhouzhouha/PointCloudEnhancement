# GQE-Net

## Source

- Paper: GQE-Net: A Graph-based Quality Enhancement Network for Point Cloud Color Attribute
- Paper link: `https://doi.org/10.1109/TIP.2023.3330753`
- Repository: `https://github.com/xjr998/GQE-Net`
- Local checkout: `third_party/enhancement/GQE-Net`

## Category

- External enhancement method
- Color / texture / attribute enhancement
- Keeps input geometry and predicts color attributes

## Model

Released pretrained checkpoints are present locally:

- Y channel: `third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-25/y/model_6.pth`
- U channel: `third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-28/u/model_55.pth`
- V channel: `third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-31/v/model_92.pth`

## UVG-CWI-DQPC Adaptation

The adapter stages the UVG CG PLY in GQE-Net's expected original/reconstructed
directory layout and runs the official evaluation script. For the enhancement
input, CG is used as the distorted/reconstructed point cloud. HE is not given to
the method; HE is only used afterward by the benchmark evaluator.

Because GQE-Net itself fuses overlapping color patches by averaging, that
averaging is part of the official method and is left unchanged. The `k=1` color
transfer rule is not used here because this is not a geometry-only method.

## Status

- Integration status: smoke completed after compatibility fix
- Job script: `jobs/gqenet_orangekettlebell_0000_smoke.slurm`
- Failed SLURM job: `24000031` on `gpu_a100`
- Failure reason: official script used CUDA indices to index CPU tensors under current PyTorch.
- Compatibility fix: keep indexed point/color tensors and overlap counters on the same CUDA device; model weights and algorithm are unchanged.
- Completed SLURM job: `24000230` on `gpu_a100`
- Adapter: `scripts/run_gqenet_selected_frames.py`
- Output: `results/method_outputs/gqenet/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/gqenet/summary_metrics.csv`
- Result: geometry metrics are essentially unchanged, as expected for a color-only method. Texture-specific metrics are still needed.

## Potential Adjustment

The current UVG-CWI metric runner is mostly geometry-oriented, so it may not
show the benefit of a color-only method. A later texture benchmark should add
attribute PSNR or nearest-neighbor color error against HE.

## Selected-10 Expansion

- Job script: `jobs/gqenet_orangekettlebell_selected10.slurm`
- Method name: `gqenet_selected10`
- Frames: `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`
