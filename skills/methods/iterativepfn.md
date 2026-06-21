# IterativePFN

- Paper title: IterativePFN: True Iterative Point Cloud Filtering.
- Repository: `https://github.com/ddsediri/IterativePFN`.
- Category: geometry denoising / point cloud filtering.
- Local code: `third_party/enhancement/IterativePFN`.
- Checkpoint: `third_party/enhancement/IterativePFN/pretrained/denoisenet-ep-99.ckpt`.
- Benchmark adapter: `scripts/run_iterativepfn_selected_frames.py`.

## Benchmark Setup

- Input: UVG-CWI-DQPC CG `.ply` frames converted to XYZ tensors.
- Output: denoised XYZ with the same point count, written back as XYZRGB `.ply`.
- Texture: nearest-neighbor color transfer from the original CG input (`k=1`).
- Metrics: UVG-CWI metric runner against HE reference.
- Inference settings: official pretrained checkpoint and default `test.py`
  denoising parameters: `patch_size=1000`, `niters=1`, `patch_stitching=True`,
  `seed_k=6`, `seed_k_alpha=10`, `num_modules_to_use=None`.

## Notes

- The official implementation uses PyTorch Lightning. The local environment was
  pinned to `pytorch-lightning==1.7.6` and `torchmetrics==0.9.3` so the provided
  checkpoint can be loaded.
- The official patch-stitching path contains a hard-coded `.cuda()`, so smoke
  and selected-frame runs should be scheduled on GPU.
