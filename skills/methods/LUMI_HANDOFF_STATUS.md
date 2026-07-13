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
