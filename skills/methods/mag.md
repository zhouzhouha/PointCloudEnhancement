# MAG

## Source

- Paper: Point Cloud Denoising via Momentum Ascent in Gradient Fields
- Paper link: `https://ieeexplore.ieee.org/abstract/document/10222122`
- Repository: `https://github.com/IndigoPurple/MAG`
- Local checkout: `third_party/enhancement/MAG`

## Category

- External enhancement method
- Geometry denoising
- Geometry-only output, so UVG benchmark output uses nearest CG color transfer with `k=1`

## Model

- Released pretrained checkpoint: `third_party/enhancement/MAG/pretrained/ckpt.pt`

## UVG-CWI-DQPC Adaptation

The adapter only converts input/output and evaluates:

1. Convert UVG CG XYZRGB PLY to geometry-only `.xyz`.
2. Run MAG official `test_large.py` with the released checkpoint.
3. Transfer RGB from the original CG frame to each MAG output point using nearest neighbor `k=1`, no averaging.
4. Write XYZRGB PLY and evaluate against HE with UVG-CWI metrics.

No method tuning is applied.

## Status

- Integration status: smoke completed
- Job script: `jobs/mag_orangekettlebell_0000_smoke.slurm`
- Completed SLURM job: `24000030` on `gpu_a100`
- Adapter: `scripts/run_mag_selected_frames.py`
- Output: `results/method_outputs/mag/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/mag/summary_metrics.csv`
- Result: frame `0000` slightly improved `CD_Acc`, `chamfer-L1`, `chamfer-L2`, and `F_20`, but slightly worsened `CD_Comp` and `F_10`.

## Potential Adjustment

If the official large-cloud run is too slow or fails on 500k-point frames, the only benchmark-safe adjustment is scheduling/splitting frames. Changing `cluster_size`, denoising iterations, or KNN parameters should be labeled as a non-default ablation.

## Selected-10 Expansion

- Job script: `jobs/mag_orangekettlebell_selected10.slurm`
- Method name: `mag_selected10`
- Frames: `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`
