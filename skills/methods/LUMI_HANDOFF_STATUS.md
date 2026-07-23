# LUMI Handoff Status

Date: 2026-07-13

## Scope

This repo contains the benchmark scripts, SLURM templates, method status notes,
and compact CSV summaries needed to resume the UVG-CWI-DQPC point cloud
enhancement benchmark on another cluster. Large generated PLY outputs, dataset
archives, checkpoints, logs, and third-party cloned repositories are intentionally
not committed to Git.

## Dataset

- Dataset: UVG-CWI-DQPC.
- Reference input: high-end point cloud, `HE_15`.
- Degraded input: consumer-grade point cloud v2, `CGv2_15`.
- Full-dataset manifest: `results/full_dataset/uvg_cwi_dqpc_he15_cgv2_15_manifest.csv`.
- Paired frames in manifest: 2152.
- Dataset-level aggregation rule: average all frames inside each sequence first,
  then average the 12 sequence means.

## Latest Full-Dataset Geometry Counts

Completed 12/12 sequence summaries:

- `apuldi_local_pu1k_4x_2048_full`
- `crcir_aftercomp_4x_full`
- `neuralpoints_16x_2048_full`
- `pucrn_pu1k_4x_full`
- `puflow_discrete_full`
- `pufm_pugan_4x_full`
- `spu_pointnet_4x_full`
- `upsample_clean_ounet_full`

Partial sequence summaries:

- `gradpu_chunked_4x_full`: 11/12
- `pdlts_light_fbm_full`: 10/12
- `pathnet_chunked_full`: 10/12
- `snowflakenet_pu_4x_full`: 8/12
- `repkpu_pu1k_4x_full`: 8/12
- `gqenet_full`: 8/12
- `score_denoise_full`: 7/12
- `mag_full`: 7/12
- `pu_gaussian_pu1k_4x_full`: 4/12
- `pudm_pu1k_4x_full`: 3/12

Methods with no complete sequence summary in the first pass:

- `pdflow_full_dataset`
- `pc2pu_4x_chunks256_full`
- `p2p_bridge_pvds_punet_full`
- `spupmd_pu1k_4x_full`
- `pointcleannet_full`

## Compact Result Files Committed

- `results/full_dataset/summary_completed12_interim/geometry_dataset_mean.csv`
- `results/full_dataset/summary_completed12_interim/geometry_delta_dataset_mean.csv`
- `results/full_dataset/summary_completed12_interim/geometry_per_sequence.csv`
- `results/full_dataset/summary_completed12_interim/geometry_delta_per_sequence.csv`
- `results/full_dataset/full_dataset_method_jobs.csv`
- `results/full_dataset/full_dataset_postprocess_jobs.csv`
- `results/selected10_method_summary.csv`

Texture/perceptual full-dataset outputs were not produced before quota ran out.
Both GPU and CPU-staging texture jobs were cancelled by Snellius with
`AdminComment=reason=budget`.

## Current Practical Next Steps On LUMI

1. Recreate or mount the UVG-CWI-DQPC dataset with the same `HE_15`/`CGv2_15`
   layout.
2. Recreate method environments and checkpoints from
   `skills/methods/ENVIRONMENT_REPRODUCIBILITY.md` and
   `skills/methods/BENCHMARK_METHOD_STATUS.md`.
3. Retry missing full-dataset method/sequence pairs, starting with the methods
   already included in the survey paper.
4. Run `jobs/full_dataset_texture_array_cpu.slurm` or a LUMI-adapted equivalent
   after geometry outputs are present.
5. Run `scripts/summarize_full_dataset_results.py` using per-frame geometry CSVs
   and texture/perceptual CSVs.

## LUMI Scheduler Adaptation

Scheduler-only adaptation completed on 2026-07-13. No enhancement method logic
was changed.

- Checkout: `/project/project_465003117/xuemei/PointCloudEnhancement` on branch
  `agent/lumi-benchmark-handoff`.
- Shared job setup: `jobs/lumi_job_env.sh`.
- Adapted templates: `jobs/full_dataset_method_array.slurm`,
  `jobs/full_dataset_texture_array.slurm`,
  `jobs/full_dataset_texture_array_cpu.slurm`, and
  `jobs/full_dataset_summarize.slurm`. The dataset download/layout jobs and the
  three completed-12 interim postprocessing jobs have also been adapted.
- GPU jobs use the NVIDIA LUMI-D `lumid` partition with one A40 GPU and the
  project account. CPU jobs use `small`. LUMI-D has a 12-hour maximum, so GPU
  templates use 12-hour limits.
- Default dataset location:
  `/scratch/project_465003117/data/UVG_CWI_DQPC/UVG-CWI-DQPC`. LUMI project
  storage has a 50 GB quota, which is not suitable for the archives plus
  extracted point clouds; scratch currently provides the required capacity.
- Heavy results, temporary files, and extension builds default to
  `/scratch/project_465003117/PointCloudEnhancement/full_dataset`; compact CSV
  summaries remain in the repository under `results/full_dataset/`.
- Runtime base being created from
  `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel`, matching the Snellius PyTorch
  2.4.1 / CUDA 12.1 stack as closely as LUMI-D permits. Expected SIF path:
  `/project/project_465003117/containers/pce-pytorch-2.4.1-cu121-devel.sif`.
- Expected overlay environment:
  `/project/project_465003117/envs/PointCloudEnhancement/lumi_cuda_torch_env`.
- Core environment requirements are pinned in
  `skills/methods/lumi_cuda_metric_requirements.txt`. Open3D is intentionally
  not in this core venv: both 0.19 Linux wheels tested require X11 libraries
  absent from the CUDA base image. PathNet and UVG-CWI/Metric do not need it;
  methods that do need it require a separately recorded runtime layer.
- All paths and the container/venv locations can be overridden at submission
  time with the `PCE_*` variables defined in `jobs/lumi_job_env.sh`.
- Shell syntax and Slurm headers were validated. `sbatch --test-only` accepted
  the adapted templates.

LUMI-D compatibility boundary: the A40 is NVIDIA/CUDA and therefore closer to
Snellius than LUMI-G, but LUMI-D is intended mainly for visualization, has an
older software stack, and limits jobs to 12 hours. Methods that previously
needed A100/H100 memory or more than 12 hours still need targeted feasibility
tests; do not silently change their algorithms.

## Active LUMI Jobs

Submitted on 2026-07-13:

- `19842018`: downloaded and extracted the 24 selected archives (`HE_15` and
  `CGv2_15` for 12 sequences) into scratch; completed successfully.
- `19842019`: canonicalize `he/15fps` and `cg/15fps` links and rebuild the
  paired-frame manifest. Completed successfully.
- `19842808`: built the PCQM binary from upstream commit
  `2a2c4105f2683d82fea69a6c6edb228b25a696c8`; completed successfully.
- `19843196`: created the CUDA 12.1 core container venv; completed
  successfully. Earlier attempts `19842798` and `19843136` exposed the Open3D
  X11 dependency and are retained in logs as failed bootstrap diagnostics.
- `19843220`: validated the environment on an NVIDIA A40. PyTorch reported
  CUDA 12.1, device capability 8.6, and completed a CUDA matrix multiply.
- `19843269`: PathNet OrangeKettlebell frame `0000` compatibility smoke on
  LUMI-D; completed successfully in 5m28s and wrote only to scratch. It
  preserved `502941` points, produced `per_frame_metrics.csv` and
  `summary_metrics.csv`, and was objectively mixed (for example, Chamfer-L2 and
  F10 improved slightly while Chamfer-L1 and F20 worsened slightly).

At the observed PathNet smoke rate, even the shortest 157-frame sequence would
take about 14.3 hours if frame cost stayed constant, exceeding LUMI-D's 12-hour
limit. Therefore PathNet retries must be split into resumable frame batches and
merged into one sequence summary; do not submit the current one-sequence array
entry unchanged. The two missing PathNet sequence names were not retained in
the committed Snellius summaries and must be recovered before targeted retry.

The LUMI manifest contains 2,152 pairs and exactly matches the Snellius counts.
`BlueSpeech`, `GoodVision`, and `TrumanShow` each have one CG frame without a
paired HE frame: `0168`, `0167`, and `0171`, respectively.

Storage grant clarification: the grant PDF specifies 31,000 LUMI TB-hours,
equivalent to about 3.54 TB held continuously for one year. The live allocation
provides 50 TiB of scratch quota, 2 TiB of flash, and 50 GiB of project storage.
The 237 GB dataset is stored under scratch and consumes the grant's TB-hours;
there is no separate 4 TB mount exposed by LUMI.

No full-sequence retry, texture, or final-summary job has been submitted on
LUMI yet. The PathNet job above is a one-frame compatibility smoke only.

Update 2026-07-14: PathNet full-dataset recreation has started as resumable
50-frame batches because a whole sequence cannot fit LUMI-D's 12-hour limit.
Job array `19853978` is the first wave (task offset 0), covering BlueSpeech,
BlueVolley, and BouncingBlue with at most two A40 tasks running concurrently.
The launcher is `jobs/pathnet_full_dataset_batched_lumid.slurm`; each batch has
an isolated method/output directory so concurrent metric CSV writes cannot
overwrite each other. Later waves use offsets 12, 24, and 36 after earlier
tasks drain, because the LUMI-D association permits at most 16 submitted jobs.
Batch outputs still need to be merged into `pathnet_chunked_full` after all
four waves complete.

Continuation update 2026-07-14: tasks `0` through `3` of job `19853978`
completed successfully, tasks `4` and `5` were running, and tasks `6` through
`11` remained queued at the latest check. The next eight global tasks were
submitted as job array `19871221` with `PCE_TASK_OFFSET=12`, covering all four
batches of `FitFluencer` and `GoodVision`. Job `19871221` has an
`afterany:19853978` dependency and a two-task concurrency limit, so it starts
only after the first wave drains. The remaining global tasks `20` through `47`
must be submitted later in capacity-safe waves; do not duplicate offsets `0`
through `19`.

Parallel incomplete-method compatibility smokes on 2026-07-14 established the
remaining transfer/environment blockers without performing full-sequence work:

- GradPU `19854153`: missing
  `third_party/enhancement/Grad-PU/pretrained_model/pu1k/ckpt/ckpt-epoch-60.pth`.
- GQE-Net `19854154`: missing the Y/U/V checkpoint tree under
  `third_party/enhancement/GQE-Net/pths/final_2023/GQE-Net/2023-07-25/`.
- Score-Denoise `19854155`: missing
  `third_party/enhancement/score-denoise/pretrained/ckpt.pt`.
- PD-LTS `19854157`: checkpoint is present, but the LUMI core environment lacks
  the required `pytorch3d` package.
- P2P-Bridge `19854158`: missing
  `third_party/enhancement/P2P-Bridge/pretrained/PVDS_PUNet/latest.pth`.
- PUDM `19854159`: its repository/config/checkpoint payload is incomplete;
  `third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json` was the first
  missing path reported.

Only the PathNet and PD-LTS checkpoints were transferred with the handoff. Do
not submit full-sequence jobs for the other incomplete methods until their
recorded repositories/checkpoints have been restored. PathNet continues to run;
PD-LTS needs a compatible PyTorch3D installation or a separately validated
inference fallback before retry.

Only compact Snellius summary CSVs were committed. Enhanced full-dataset PLY
outputs and third-party method repositories were not transferred, so LUMI
texture/perceptual jobs must wait until the corresponding geometry outputs have
been recreated or copied from an external retained Snellius location.

## Restored Method Repositories (2026-07-17)

The user supplied `/project/project_465003117/lumi_missing_method_repos.tar.gz`
with SHA-256
`af4bbfff46dab16be834d3e547cb5a5d8291937ae00a366abac0f64dfcd9bd42`.
The archive contained GradPU, GQE-Net, Score-Denoise, P2P-Bridge, and PUDM.
It was inspected for absolute paths, parent traversal, and symbolic links,
extracted to a staging directory, and merged with `rsync --ignore-existing` so
no existing repository file was overwritten.

Restored-asset smoke jobs:

- P2P-Bridge `19973872`: checkpoint found; stopped at missing `open3d`.
- PD-LTS `19973873`: PyTorch3D and the `pila` CUDA extension loaded/built;
  stopped at missing bundled top-level `emd` extension.
- GQE-Net `19973874`: Y/U/V checkpoints found; stopped at missing `sewar`.
- Score-Denoise `19973875`: checkpoint found; stopped at missing
  `torch_cluster`.
- GradPU `19973876`: checkpoint found; stopped at missing `open3d`.
- PUDM `19973877`: config/checkpoint found and PyTorch3D fallbacks loaded;
  stopped at missing `open3d`.

CPU dependency build job `19973893` installs GQE-Net's pure-Python `sewar`
dependency and compiles PD-LTS's bundled CUDA EMD extension without consuming a
GPU allocation. PathNet global tasks `20` through `31` were submitted as job
array `19973883`, covering Mannequin, OrangeKettlebell, and PinkNoir.
