# IGR

## Source

- Repository: `third_party/SCUTSurface/reconstruction/IGR`
- Upstream: `https://github.com/amosgropp/IGR`
- SCUTSurface source list: `https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/tree/main/reconstruction`
- Paper: Implicit Geometric Regularization for Learning Shapes
- Category: learning-based implicit surface reconstruction
- Selected for benchmark: yes, second SCUTSurface method after SAL

## Purpose

IGR learns an implicit function for surface reconstruction from point clouds. In this benchmark it is used after SAL to compare implicit reconstruction methods on real-world dynamic UVG-CWI-DQPC CG captures, evaluated against HE references.

## Environment

- Python / CUDA / compiler requirements: Python with PyTorch, pyhocon, plotly, scikit-image, trimesh, scipy, GPUtil.
- Required pretrained model: not used for the reconstruction smoke test; IGR optimizes a shape-specific implicit function from the input frame.
- Snellius module or conda environment: `torch_env`.
- Expected hardware: GPU required by the current implementation because `reconstruction/run.py` calls `.cuda()` directly.

Read the Snellius skill reference before writing environment commands or SLURM scripts.

## Input

- Expected input file type: point cloud, likely normalized point samples.
- Expected point attributes: geometry positions; color likely not used.
- Expected point count: smoke test uses 50,000 normalized CG points for frame `0000`.
- Expected coordinate scale: normalized to unit scale before IGR; output is restored to UVG-CWI-DQPC scale using saved normalization metadata.
- Does it use color: likely no.
- Does it use normals: no for the smoke test; `normals_lambda = 0`.
- Does it process single frames or full sequences: likely single frames; sequence wrapper required.

## UVG-CWI-DQPC Adaptation

- Dataset root: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC`
- Toy sequence: `OrangeKettlebell`
- Input path pattern: `<dataset_root>/<sequence>/cg/15fps/*.ply`
- Reference path pattern: `<dataset_root>/<sequence>/he/15fps/*.ply`
- Frame pairing rule: pair CG and HE by final zero-padded frame id.
- Preprocessing required: convert each CG frame into IGR input format; record normalization transform.
- Postprocessing required: convert reconstructed surface/output to PLY and restore original coordinate scale.

## Output

- Output file type: PLY point cloud or mesh-derived point cloud, depending on IGR output.
- Output path pattern: `results/method_outputs/igr/<sequence>/15fps/*.ply`
- Preserved attributes: geometry only unless color transfer is added.
- Generated attributes: reconstructed geometry.
- Failure cases: normalization mismatch, mesh sampling mismatch, lost color, per-frame optimization too slow, topology artifacts.

## Run Command

```bash
sbatch jobs/igr_orangekettlebell_0000_smoke.slurm
```

## Metrics

Evaluate output against the HE reference with UVG-CWI/Metric.

- Per-frame output: `results/uvg_cwi_dqpc/<sequence>/igr/per_frame_metrics.csv`
- Per-sequence summary: `results/uvg_cwi_dqpc/<sequence>/igr/summary_metrics.csv`
- Runtime logging: record preprocessing, reconstruction, postprocessing, and metric runtime separately.

## Status

- Integration status: attempted and set aside
- Toy frame tested: completed for frame `0000`
- Parameter sweep tested: completed for frame `0000`
- 5-frame test passed: no
- Full `OrangeKettlebell` sequence passed: no
- All sequences passed: no
- Decision: do not expand to selected-10 under the current benchmark pass because all tested variants were worse than the CG baseline.

## Notes

IGR is second because the SCUTSurface reconstruction README lists it after SAL and the user specified `https://github.com/amosgropp/IGR` as the next method.

Smoke-test details:

- Job script: `jobs/igr_orangekettlebell_0000_smoke.slurm`
- Job ID: `23475762`
- Partition: `gpu_mig`
- Resources: 1 MIG GPU, 8 CPUs, 2 hours
- Input frame: `OrangeKettlebell` frame `0000`
- Input preparation script: `scripts/prepare_uvg_frame_for_igr.py`
- Smoke input point cap: 50,000 points
- Training length: 100 epochs
- Evaluation resolution: 64
- Expected log: `logs/igr_ok_0000_smoke_23475762.out`
- Generated visual PLY output: `results/method_outputs/igr/OrangeKettlebell/15fps/frame_0000.ply`
- Generated mesh output: `results/method_outputs/igr/OrangeKettlebell/15fps/frame_0000_mesh.ply`

GPU-resubmission/accounting note:

- `accinfo` showed active products are `gpu_a100`, `gpu_h100`, `gpu_mig`, `cbuild`, and `staging`, with no CPU partition product listed.
- IGR requires GPU in this implementation because `reconstruction/run.py` calls `.cuda()` directly.
- The original `gpu_mig` job `23475762` completed and produced frame `0000` outputs.
- A replacement H100 job was briefly submitted, first as `23475965` with 18 CPUs; Snellius warned that this would be billed as 2 H100 GPUs, so it was canceled.
- It was resubmitted as `23475978` with 16 CPUs so it billed as 1 H100 GPU; this run also completed and overwrote/confirmed the frame `0000` outputs.

Sweep result:

- Job script: `jobs/igr_orangekettlebell_0000_sweep.slurm`
- Job ID: `23485809`
- Output directory: `results/method_outputs/igr_sweep/OrangeKettlebell/15fps/`
- Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/igr_sweep/frame_0000_baseline_vs_igr_sweep.csv`
- Result: all sweep variants worsened the frame `0000` CG baseline across Chamfer and F-score metrics. The best-looking variant still had much worse `chamfer-L1` and `F_20`, so selected-10 expansion is not currently justified.
