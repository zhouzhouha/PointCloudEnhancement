# PointCleanNet

## Source

- Paper: PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds
- Upstream category: external point-cloud denoising/enhancement method
- Local checkout: `third_party/enhancement/PointCleanNet`
- Official inference script used: `third_party/enhancement/PointCleanNet/noise_removal/eval_pcpnet.py`

## Category

- Denoising
- Outlier removal
- External enhancement baseline

This method is not part of the SCUTSurface/SUSTech-first reconstruction list. It was started as an external denoising baseline because pretrained weights are available. Keep it recorded separately from the SCUTSurface method order.

## Model

Released pretrained denoising model:

- `third_party/enhancement/PointCleanNet/models/denoisingModel/PointCleanNet_model.pth`
- `third_party/enhancement/PointCleanNet/models/denoisingModel/PointCleanNet_params.pth`

## UVG-CWI-DQPC Adaptation

The adapter only handles input/output and benchmarking:

1. Convert UVG-CWI-DQPC CG XYZRGB PLY to geometry-only `.xyz`.
2. Run PointCleanNet official `eval_pcpnet.py`.
3. Use the released three-pass refinement protocol.
4. Transfer RGB from the original CG frame to each enhanced point with nearest neighbor `k=1`.
5. Write XYZRGB PLY for metric and visual inspection.
6. Evaluate CG baseline and PointCleanNet output against HE reference using the UVG-CWI metric runner.

No method tuning is applied for benchmarking.

## Run Command

Smoke job for OrangeKettlebell frame `0000`:

```bash
sbatch jobs/pointcleannet_orangekettlebell_0000_smoke.slurm
```

Adapter:

```bash
python -u scripts/run_pointcleannet_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pointcleannet \
  --nrun 3 \
  --workers 1 \
  --cache-capacity 1
```

## Output

Expected output after completion:

- `results/method_outputs/pointcleannet/OrangeKettlebell/15fps/frame_0000.ply`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointcleannet/per_frame_metrics.csv`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointcleannet/summary_metrics.csv`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointcleannet/run_config.json`

Working files:

- `results/work/pointcleannet/OrangeKettlebell/15fps/input/`
- `results/work/pointcleannet/OrangeKettlebell/15fps/pcn_results/`

## Status

- Integration status: external smoke completed
- SLURM job: `23526613`
- Log: `logs/pointcleannet_ok_0000_smoke_23526613.out`
- Result: frame `0000` completed with `502941 -> 502941` points.
- Output: `results/method_outputs/pointcleannet/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pointcleannet/summary_metrics.csv`
- Observed metric behavior: improved most distance/F-score metrics on frame `0000`, with small drops in `P_5` and `R_20`.
- Expansion status: do not expand beyond smoke until metrics are reviewed and the SCUTSurface/SUSTech-first list is back on track.

## Notes

This method should be reported under external enhancement methods, not as a SCUTSurface/SUSTech method. If retained in benchmark tables, label it clearly as a pretrained external denoising baseline.

Selected-10 expansion:

- Job script: `jobs/pointcleannet_orangekettlebell_selected10.slurm`
- Method name: `pointcleannet_selected10`
- Frames: `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`
- SLURM job: `23980358`
- Current status observed: running on `gpu_h100` node `gcn99`
- Intermediate output observed: `results/work/pointcleannet_selected10/OrangeKettlebell/15fps/pcn_results/OrangeKettlebell_frame_0000_0.xyz`
