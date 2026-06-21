# SAL

## Source

- Repository: `third_party/SCUTSurface/reconstruction/SAL`
- Upstream: `https://github.com/matanatz/SAL`
- SCUTSurface source list: `https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/tree/main/reconstruction`
- Paper: Sign Agnostic Learning of Shapes From Raw Data
- Category: learning-based implicit surface reconstruction
- Selected for benchmark: yes, first SCUTSurface method

## Purpose

SAL learns an implicit shape representation from raw point clouds and reconstructs a surface. In this benchmark it is used as the first SCUTSurface reconstruction method to test whether reconstruction-based enhancement improves CG point clouds when evaluated against HE references.

## Environment

- Python / CUDA / compiler requirements: todo, read SAL setup and Snellius guidance before creating the environment.
- Required pretrained model: todo, determine whether pretrained weights are available or whether per-sequence/per-frame training is needed.
- Snellius module or conda environment: todo.
- Expected hardware: likely GPU for training/inference; confirm from SAL scripts.

Read the Snellius skill reference before writing environment commands or SLURM scripts.

## Input

- Expected input file type: point cloud, likely normalized format required by SAL preprocessing.
- Expected point attributes: geometry positions; color support unclear and likely not used by SAL.
- Expected point count: todo.
- Expected coordinate scale: SAL may normalize to unit sphere; output must be resized back to UVG-CWI-DQPC scale.
- Does it use color: likely no.
- Does it use normals: todo.
- Does it process single frames or full sequences: likely single frames; sequence handling must be added by wrapper.

## UVG-CWI-DQPC Adaptation

- Dataset root: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC`
- Toy sequence: `OrangeKettlebell`
- Input path pattern: `<dataset_root>/<sequence>/cg/15fps/*.ply`
- Reference path pattern: `<dataset_root>/<sequence>/he/15fps/*.ply`
- Frame pairing rule: pair CG and HE by final zero-padded frame id.
- Preprocessing required: convert each CG frame into SAL input format; record normalization transform.
- Postprocessing required: convert SAL reconstruction output back to PLY and restore original coordinate scale with `third_party/SCUTSurface/reconstruction/resize.py` or an equivalent recorded inverse transform.

## Output

- Output file type: PLY point cloud or mesh-derived point cloud, depending on SAL reconstruction output.
- Output path pattern: `results/method_outputs/sal/<sequence>/15fps/*.ply`
- Preserved attributes: geometry only unless color transfer is added.
- Generated attributes: reconstructed geometry.
- Failure cases: normalization mismatch, mesh-to-point-cloud sampling mismatch, lost color, per-frame training too slow, topology artifacts.

## Run Command

```bash
sbatch jobs/sal_orangekettlebell_0000_smoke.slurm
```

## Metrics

Evaluate output against the HE reference with UVG-CWI/Metric.

- Per-frame output: `results/uvg_cwi_dqpc/<sequence>/sal/per_frame_metrics.csv`
- Per-sequence summary: `results/uvg_cwi_dqpc/<sequence>/sal/summary_metrics.csv`
- Runtime logging: record preprocessing, reconstruction, postprocessing, and metric runtime separately.

## Status

- Integration status: train/eval smoke queued
- Toy frame tested: training smoke completed in SLURM job `23389811`; train/eval smoke queued as SLURM job `23430737`
- 5-frame test passed: no
- Full `OrangeKettlebell` sequence passed: no
- All sequences passed: no

## Notes

SAL is the first method because the user requested starting with SCUTSurface reconstruction methods, and SAL is the only method currently present under `third_party/SCUTSurface/reconstruction/` in this checkout.

Smoke-test details:

- Job script: `jobs/sal_orangekettlebell_0000_smoke.slurm`
- Partition: `gpu_mig`
- Resources: 1 MIG GPU, 8 CPUs, 1 hour
- Input frame: `OrangeKettlebell` frame `0000`
- Input preparation script: `scripts/prepare_uvg_frame_for_sal.py`
- Smoke input point cap: 50,000 points to reduce preprocessing cost
- Log: `logs/sal_ok_0000_smoke_23389811.out`
- Checkpoint directory: `third_party/SCUTSurface/reconstruction/SAL/exps/OrangeKettlebell_frame_0000/2026_06_02_23_06_55/checkpoints/`
- Result: completed short training through epoch 6 and wrote model/optimizer checkpoints. Mesh extraction was not run in this smoke job.

Train/eval smoke-test details:

- Job script: `jobs/sal_orangekettlebell_0000_train_eval.slurm`
- Job ID: `23430737`
- Partition: `gpu_mig`
- Resources: 1 MIG GPU, 8 CPUs, 1.5 hours
- Training length: 20 epochs on capped 50,000-point frame input
- Evaluation resolution: 64
- Expected log: `logs/sal_ok_0000_train_eval_23430737.out`
- Expected mesh directory: `third_party/SCUTSurface/reconstruction/SAL/exps/OrangeKettlebell_frame_0000/<timestamp>/evaluation/none/`

Metric smoke-test details:

- Metric source: `third_party/UVG-CWI-Metric/metrics.py`
- Evaluation script: `scripts/evaluate_sal_vs_baseline_uvg_metric.py`
- Per-method CSV: `results/uvg_cwi_dqpc/OrangeKettlebell/sal/frame_0000_uvg_metric.csv`
- Baseline-vs-SAL CSV: `results/uvg_cwi_dqpc/OrangeKettlebell/sal/frame_0000_baseline_vs_sal.csv`
- Result for frame `0000`: SAL is worse than the CG baseline for the smoke setting. This is not a final SAL result because it used only 20 epochs, a 50,000-point capped input, and a low-resolution evaluation mesh.

Surface-sampled metric check:

- Surface-sampled output: `results/method_outputs/sal/OrangeKettlebell/15fps/frame_0000_surface200k.ply`
- CSV: `results/uvg_cwi_dqpc/OrangeKettlebell/sal_surface200k/frame_0000_baseline_vs_sal.csv`
- Result: still worse than CG baseline, so the poor result is not only caused by using sparse mesh vertices.

Selected better setting before any full-frame/full-sequence SAL run:

- Input cap: 200,000 CG points for the toy frame instead of 50,000.
- Training: 500 epochs for `OrangeKettlebell` frame `0000`.
- Plotting during training: disabled with `plot_frequency = 0`; only run final evaluation after training.
- Evaluation resolution: start at 128; try 192 only if 128 is stable and visually reasonable.
- Output for metrics/visualization: save both the reconstructed mesh and a 200,000-point surface-sampled PLY.
- Resource request: continue using `gpu_mig` first to save quota; escalate only if memory/runtime fails.

Selected 10-frame run:

- Job script: `jobs/sal_orangekettlebell_selected10_500e.slurm`
- Job ID: `23475682`
- Frames: `0000, 0010, 0020, 0030, 0040, 0050, 0060, 0070, 0080, 0090`
- Settings: 200,000 input points, 500 epochs, evaluation resolution 128, 200,000 surface-sampled output points.
- Partition/resources: `gpu_mig`, 1 MIG GPU, 8 CPUs, 12 hours.
- Expected log: `logs/sal_ok_selected10_500e_23475682.out`
- Generated visual PLY outputs: `results/method_outputs/sal/OrangeKettlebell/15fps/frame_<frame>.ply`
- Generated mesh outputs: `results/method_outputs/sal/OrangeKettlebell/15fps/frame_<frame>_mesh.ply`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/sal_selected10/per_frame_metrics.csv`
- Summary: `results/uvg_cwi_dqpc/OrangeKettlebell/sal_selected10/summary_metrics.csv`

GPU-resubmitted 10-frame run:

- Reason: `accinfo` showed the account has GPU products (`gpu_a100`, `gpu_h100`, `gpu_mig`) but no CPU partition product; SAL also requires GPU because the implementation calls `.cuda()`.
- Old `gpu_mig` job `23475682` started frame `0000` and was canceled while evaluating, before the 10-frame run completed.
- New job script: `jobs/sal_orangekettlebell_selected10_500e_a100.slurm`
- New job ID: `23475964`
- Partition/resources: `gpu_a100`, 1 A100 GPU, 18 CPUs, 12 hours.
- Status observed: running on node `gcn71`.
- Expected log: `logs/sal_ok_selected10_500e_a100_23475964.out`
