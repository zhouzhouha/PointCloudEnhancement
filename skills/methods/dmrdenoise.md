# DMRDenoise

## Source

- Paper: Differentiable Manifold Reconstruction for Point Cloud Denoising
- Local checkout: `third_party/enhancement/DMRDenoise`
- Survey category: point cloud denoising

## Category

- External enhancement method
- Denoising
- Geometry-only

## Model

Released pretrained checkpoints are present locally:

- `third_party/enhancement/DMRDenoise/pretrained/supervised/epoch=153.ckpt`
- `third_party/enhancement/DMRDenoise/pretrained/unsupervised/epoch=141.ckpt`

The smoke used the official supervised checkpoint and official/default denoising settings.

## UVG-CWI-DQPC Adaptation

The adapter prepares CG frames as `.xyz`, calls the official `denoise.py`, then would transfer RGB from the original CG frame with nearest neighbor `k=1` and write XYZRGB PLY.

Compatibility changes only:

- Installed missing `pytorch-lightning==0.7.6` and `h5py` into `torch_env`.
- Built the repository's bundled EMD extension.
- Wrapped the removed sklearn `KMeans(..., n_jobs=...)` argument for compatibility with the installed sklearn version.

No method tuning was applied.

## Status

- Integration status: smoke failed
- SLURM job: `23527266`
- Log: `logs/dmrdenoise_ok_0000_smoke_23527266.out`
- Input generated: `results/work/dmrdenoise/OrangeKettlebell/15fps/input/OrangeKettlebell_frame_0000.xyz`
- Final enhanced output: not generated

## Failure Reason

The official model failed during inference before writing `dmr_results/OrangeKettlebell_frame_0000.denoised.xyz`.

Error:

```text
RuntimeError: Sizes of tensors must match except in dimension 3. Expected size 23 but got size 24 for tensor number 1 in the list.
```

This happens inside DMRDenoise dynamic graph convolution after its large-cloud clustering/downsampling path. For one local patch, the effective neighborhood size becomes smaller than the checkpoint/network branch expects. Because the official script crashes before saving the denoised `.xyz`, the adapter cannot transfer colors or write:

```text
results/method_outputs/dmrdenoise/OrangeKettlebell/15fps/frame_0000.ply
```

## Decision

Keep DMRDenoise as failed/incompatible under official default settings for this UVG frame. Do not tune `patch_size`, `cluster_size`, KNN, or network internals unless we explicitly label a later run as a non-default ablation.
