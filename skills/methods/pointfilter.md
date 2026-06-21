# Pointfilter

## Source

- Paper: Pointfilter: Point Cloud Filtering via Encoder-Decoder Modeling
- Local checkout: `third_party/enhancement/Pointfilter`
- Survey category: point cloud denoising/filtering

## Category

- External enhancement method
- Denoising / filtering
- Geometry-only

## Model

Released pretrained model:

- `third_party/enhancement/Pointfilter/Summary/pre_train_model/model_full_ae.pth`

The checkpoint loads successfully in the current `torch_env`.

## UVG-CWI-DQPC Adaptation

The adapter only handles input/output and benchmarking:

1. Convert UVG-CWI-DQPC CG XYZRGB PLY to geometry-only `.npy`.
2. Use Pointfilter's released evaluation logic with the released model/dataset classes.
3. Use the released/default smoke settings from the official script: 2 filtering iterations, patch radius `0.05`, batch size `64`.
4. Transfer RGB from original CG to filtered points using nearest neighbor `k=1`.
5. Write XYZRGB PLY.
6. Evaluate against HE reference with the UVG-CWI metric runner.

No method tuning is applied for benchmarking.

Runtime compatibility note: the original evaluation path can return `None` for sparse local patches and then crash in PyTorch's default DataLoader collation. The adapter skips those invalid patches; this does not change the model weights or filtering parameters.

## Run Command

Smoke job:

```bash
sbatch jobs/pointfilter_orangekettlebell_0000_smoke.slurm
```

Adapter:

```bash
python -u scripts/run_pointfilter_selected_frames.py \
  --sequence OrangeKettlebell \
  --frames 0000 \
  --method-name pointfilter
```

## Output

Expected after completion:

- `results/method_outputs/pointfilter/OrangeKettlebell/15fps/frame_0000.ply`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointfilter/per_frame_metrics.csv`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointfilter/summary_metrics.csv`
- `results/uvg_cwi_dqpc/OrangeKettlebell/pointfilter/run_config.json`

Working files:

- `results/work/pointfilter/OrangeKettlebell/15fps/input/`
- `results/work/pointfilter/OrangeKettlebell/15fps/pointfilter_results/`

## Status

- Integration status: external smoke completed after adapter compatibility fix
- Failed SLURM job: `23529203`
- Failed log: `logs/pointfilter_ok_0000_smoke_23529203.out`
- Failure reason: DataLoader collation failed because sparse local patches returned `None`.
- Current adapter status: skip invalid local patches and keep official pretrained/default filtering settings.
- Completed SLURM job: `23580016`
- Completed log: `logs/pointfilter_ok_0000_smoke_23580016.out`
- Output: `results/method_outputs/pointfilter/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pointfilter/summary_metrics.csv`
- Result: `CD_Acc` improved slightly, but `CD_Comp`, Chamfer, recall, and F-scores worsened. Do not expand beyond smoke unless it is needed as a negative/failure baseline.

## Notes

Pointfilter is a better fit for UVG-CWI-DQPC than object completion methods because it preserves the input point count and filters geometry directly. It still ignores color, so the benchmark output uses `k=1` color transfer from the CG input.
