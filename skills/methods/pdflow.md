# PD-Flow

## Source

- Paper: PD-Flow: A Point Cloud Denoising Framework with Normalizing Flows
- Paper link: `https://arxiv.org/abs/2203.05940`
- Repository: `https://github.com/unknownue/pdflow`
- Local checkout: `third_party/enhancement/pdflow`

## Category

- External enhancement method
- Geometry denoising
- Geometry-only output, so UVG benchmark output uses nearest CG color transfer with `k=1`

## Model

- Released pretrained checkpoint: `third_party/enhancement/pdflow/pretrain/pdflow-score-LCC.pt`

## UVG-CWI-DQPC Adaptation

The adapter only handles input/output and benchmarking:

1. Convert UVG CG XYZRGB PLY to geometry-only `.xyz`.
2. Run official `models/deflow/denoise.py`.
3. Transfer RGB from original CG to output points by nearest neighbor `k=1`.
4. Write XYZRGB PLY and evaluate with UVG-CWI metrics.

No method tuning is applied.

## Compatibility Notes

- `kaolin` is listed by PD-Flow as training-only, but the repo imported it at module load. The import was made lazy so inference can run without installing kaolin.
- The bundled PyTorch Chamfer extension requires `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST=8.0` on Snellius A100.

## Status

- Integration status: smoke job submitted
- Job script: `jobs/pdflow_orangekettlebell_0000_smoke.slurm`
- Active SLURM job: `24003019` on `gpu_a100`
- Adapter: `scripts/run_pdflow_selected_frames.py`
- Output target: `results/method_outputs/pdflow/OrangeKettlebell/15fps/frame_0000.ply`
- Metrics target: `results/uvg_cwi_dqpc/OrangeKettlebell/pdflow/summary_metrics.csv`
