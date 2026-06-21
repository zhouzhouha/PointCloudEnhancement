# Score-Denoise

## Source

- Paper: Score-Based Point Cloud Denoising
- Repository: `https://github.com/luost26/score-denoise`
- Local checkout: `third_party/enhancement/score-denoise`

## Category

- External enhancement method
- Geometry denoising
- Geometry-only output, so UVG benchmark output uses nearest CG color transfer with `k=1`

## Model

- Released pretrained checkpoint: `third_party/enhancement/score-denoise/pretrained/ckpt.pt`

## UVG-CWI-DQPC Adaptation

The adapter should mirror MAG because MAG is implemented on top of Score-Denoise:

1. Convert UVG CG XYZRGB PLY to geometry-only `.xyz`.
2. Run official `test_large.py` with the released checkpoint.
3. Transfer RGB from original CG to output points by nearest neighbor `k=1`.
4. Write XYZRGB PLY and evaluate with UVG-CWI metrics.

No method tuning is applied.

## Status

- Integration status: smoke job submitted
- Environment status: required PyTorch3D and torch-cluster dependencies are now installed in `torch_env`.
- Job script: `jobs/score_denoise_orangekettlebell_0000_smoke.slurm`
- Active SLURM job: `24001558` on `gpu_a100`
- Adapter: `scripts/run_score_denoise_selected_frames.py`
- Output target: `results/method_outputs/score_denoise/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics target: `results/uvg_cwi_dqpc/OrangeKettlebell/score_denoise/summary_metrics.csv`

## Selected-10 Expansion

- Job script: `jobs/score_denoise_orangekettlebell_selected10.slurm`
- Method name: `score_denoise_selected10`
- Frames: `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`
