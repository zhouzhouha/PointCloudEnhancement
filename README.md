# UVG-CWI-DQPC Point Cloud Enhancement Benchmark Plan

This root README is the working plan for benchmarking point cloud enhancement on the UVG-CWI-DQPC dataset for a TMM special issue submission. All historical Markdown notes have been moved under `skills/`; before changing code or running experiments, read `skills/README.md` and the relevant files under `skills/third_party/SCUTSurface/`.

## Current Context

- Dataset root: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC`
- Toy validation sequence: `OrangeKettlebell`
- Toy low-quality input: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC/OrangeKettlebell/cg/15fps`
- Toy high-quality/reference target: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC/OrangeKettlebell/he/15fps`
- Frame count checked on 2026-06-02: `170` CG frames and `170` HE frames.

Use `OrangeKettlebell` as the toy test because the other sequences should follow the same folder and pairing logic. For every method, use the same 10-frame protocol first: frames `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, and `0090`. These 10 frames are the toy benchmark and the initial quantitative comparison set. The first goal is not to report final paper numbers; it is to prove every metric is implemented, paired, scaled, logged, and reproducible on a consistent frame subset.

Current active scope: keep running and recording methods on the `OrangeKettlebell`
selected-10 benchmark only. Full-sequence/full-dataset jobs should not be
submitted until the method list has been written up and selected from the
10-frame objective and subjective results. The accidentally submitted full170
jobs `24030797`, `24030799`, `24030800`, and `24030801` were cancelled on
2026-06-19.

Explicit exception: after the user read the PD-Flow paper, PD-Flow was attempted
on the full 170-frame `OrangeKettlebell` sequence as a broader ablation for the
future UVG-CWI-DQPC run, despite its mixed/negative smoke result. The job was
submitted on 2026-06-21 as `24076038` with method name `pdflow_full170` and
failed after frame `0000` because the workspace hit disk quota while writing the
output PLY. PD-Flow must receive geometry only, so the adapter feeds XYZ `(N,3)`
to the model and then transfers RGB to the enhanced geometry by nearest-neighbor
`k=1` from the original CG frame, with no color averaging. The full170 ablation
was resubmitted as job `24076375` with work files, method outputs, and metrics
redirected to `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results`.

Survey exclusion tracking:

- SuperPC: paper read and excluded from the main benchmark because it is a
  multimodal image-and-point-cloud conditioned method, not a point-cloud-only
  enhancement method.
- FBNet: paper read and excluded from the survey benchmark because it is a
  partial point cloud completion method. Its input/output assumption is partial
  completion rather than degraded dense point cloud enhancement, so its smoke
  result is kept only as a domain-transfer reference and should not be counted
  as an included benchmark method.

## Metric Scope

For this benchmark, the primary metric implementation is `https://github.com/UVG-CWI/Metric`. SCUTSurface metrics in this repo are kept as references or cross-checks, not as the main evaluation path. The initial metric suite is:

- Accuracy / completeness: one-way nearest-neighbor distances, predicted to reference and reference to predicted.
- Chamfer distance: `chamfer-L1`, `chamfer-L2`, and legacy `chamferL2_old`.
- Threshold metrics: precision, recall, and F-score at `5`, `10`, and `20` distance units, with possible extension to thresholds used by the paper.
- Normal metrics: `N_Acc`, `N_Comp`, and combined normal correctness.
- SCUTSurface neural metric: Neural Feature Similarity, if the pretrained model and input format can be recovered and validated.

Use the SCUTSurface reconstruction methods first: `https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/tree/main/reconstruction`. The enhancement-method survey repository, `https://github.com/LilydotEE/Point_cloud_quality_enhancement`, is a secondary source for extending the benchmark after the SCUTSurface reconstruction methods are integrated. This benchmark will run representative methods by category, including learning-based, geometry-based, and hybrid approaches, and will compare how each family handles real-world dynamic point clouds with geometry noise, missing regions, temporal flickering, and color distortions.

## Immediate Engineering Plan

1. Restore the root-level workflow entry point.
   - Keep this `README.md` outside `skills/`.
   - Do not move new working documentation into `skills/` unless it is background/reference material.
   - Use `skills/METRICS_IMPLEMENTATION_PLAN.md` for metric-by-metric implementation status.
   - Use `skills/methods/` for one Markdown file per enhancement method.

2. Build the UVG-CWI metric runner.
   - Clone or vendor `https://github.com/UVG-CWI/Metric` into a controlled location, then use its `metrics.py` as the primary evaluator.
   - Adapt the runner to the real dataset path above instead of `dataset/UVG-CWI-DQPC/...`.
   - CG and HE files must be paired by frame id because their filenames differ by quality tag.
   - Write outputs under `results/uvg_cwi_dqpc/OrangeKettlebell/`.

3. Validate units and scale.
   - Inspect representative PLY headers and coordinate ranges.
   - Decide whether thresholds `5`, `10`, `20` mean millimeters in the dataset coordinate system or need conversion.
   - Record the decision in the output metadata CSV/JSON.

4. Run the fixed 10-frame toy benchmark.
   - For each method, reconstruct/enhance only frames `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, and `0090`.
   - Verify no crashes, no empty point clouds, all metric columns present, stable pairing, and plausible metric ranges.
   - Keep this 10-frame setting for method-to-method comparison unless the protocol is explicitly expanded later.
   - If a one-frame smoke test gives mixed results rather than a clear crash or clearly unusable output, run the selected-10 benchmark and compute the full objective metrics before making a keep/drop decision. Do not block selected-10 expansion on subjective visual inspection for mixed methods; visual inspection is recorded afterward as supporting evidence.

5. Cross-check metric correctness.
   - Compare one or more frames against the local SCUTSurface metric implementation as a secondary sanity check.
   - Add tiny synthetic tests for identical clouds, shifted clouds, empty clouds, and known threshold behavior.
   - Confirm F-score uses per-point distance arrays, not scalar mean distances.

6. Prepare Snellius batch execution.
   - Keep login-node work limited to file inspection and tiny runs.
   - Use `sbatch` for full-sequence metrics.
   - Stage heavy temporary outputs to `$TMPDIR` when needed, then copy CSV/JSON summaries back to the repo or project storage.
   - Default GPU smoke/selected-10 jobs can use `gpu_a100`; if a method fails on A100 with CUDA OOM and the method is still worth testing, retry once on `gpu_h100`, which is the largest-memory GPU fallback in this benchmark. Record both attempts and do not silently switch to chunking/patching unless that wrapper is explicitly marked as a compatibility adaptation.

7. Expand after the toy sequence is validated.
   - Apply the exact same runner to all UVG-CWI-DQPC sequences.
   - Generate per-frame CSV, per-sequence summary CSV, and an all-sequence aggregate table.
   - Keep raw metric outputs separate from paper-ready tables.

8. Benchmark reconstruction/enhancement methods by category.
   - Start with SCUTSurface reconstruction methods from `third_party/SCUTSurface/reconstruction/` and `https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/tree/main/reconstruction`.
   - First method to integrate: SAL from `https://github.com/matanatz/SAL`.
   - Second method to integrate: IGR from `https://github.com/amosgropp/IGR`.
   - Then add the remaining SCUTSurface-listed methods with one implementation per method. Prefer a PyTorch implementation when both TensorFlow and PyTorch versions exist; skip or hold a method when the usable PyTorch repository cannot be verified or the setup cost is too high for the current benchmarking pass.
   - Keep the ordered method/link/status table in `skills/methods/BENCHMARK_METHOD_STATUS.md`.
   - Use `https://github.com/LilydotEE/Point_cloud_quality_enhancement` and related repositories as the secondary source list for additional SOTA enhancement categories.
   - Organize models into learning-based, geometry-based, and hybrid approaches; if useful, further split them into reconstruction, denoising, completion, up-sampling / super-resolution, and color / texture enhancement.
   - For each method, create one Markdown file under `skills/methods/` documenting its input, output, environment, sequence adaptation, command, and benchmark status.
   - For each method, evaluate CG input vs HE reference and enhanced output vs HE reference with the same metric runner.
   - Report quantitative metrics, runtime, and practical failure cases so the benchmark explains both performance and generalization to real captured dynamic point clouds.
   - If a method requires pretrained models, datasets, or external assets that cannot be downloaded from Snellius/Codex due network, host, permission, quota, or authentication issues, do not guess, silently skip, or train from scratch. Always record the exact download URL, filename, checksum if available, expected local placement, and method name in the method Markdown file under `skills/methods/` and `BENCHMARK_METHOD_STATUS.md`, then ask the user to download it manually and place it in that folder.

## Expected Toy Output

For `OrangeKettlebell`, each successful method deliverable should contain:

- `results/method_outputs/<method>/OrangeKettlebell/15fps/frame_0000.ply` through `frame_0090.ply`, sampled every 10 frames.
- `results/uvg_cwi_dqpc/OrangeKettlebell/<method>/per_frame_metrics.csv` or equivalent method-specific per-frame CSV.
- `results/uvg_cwi_dqpc/OrangeKettlebell/<method>/summary_metrics.csv` or equivalent method-specific summary CSV.
- `results/uvg_cwi_dqpc/OrangeKettlebell/<method>/run_config.json` or equivalent command/config metadata.
- A short console log or SLURM log showing command, environment, and completion status.

These files prove that the method output and metric implementation work before using the results in the TMM submission. The generated PLY files are kept so each method can also be visually compared against CG input and HE reference.

Decision rule: selected-10 objective metrics are the first decision gate for any method that passes basic smoke-test validity. Subjective inspection is still required before final paper inclusion, but mixed smoke-test methods should not be stopped before selected-10 metrics are computed.

Publication-year screening rule: the active benchmark shortlist should prioritize
point-cloud-only enhancement methods from the past five years, using 2021 and
later as the default cutoff. Methods published before 2021 are treated as older
baselines or survey-background methods. Mark those methods as requiring paper
reading, and do not spend new benchmarking time on them unless they already show
positive objective performance on the current UVG selected-10 protocol or the
user explicitly promotes them for historical/comparative coverage.

Selected-10 aggregate notes for Overleaf/method decisions:

- Human-readable summary: `skills/methods/SELECTED10_RESULTS_SUMMARY.md`
- Machine-readable summary: `results/selected10_method_summary.csv`

## Environment Reproducibility

Current smoke/selected-10 jobs mostly use the shared Snellius conda environment `torch_env`. This is acceptable for rapid compatibility testing, but it is not enough for the final whole-dataset rerun unless each method's environment is frozen and recorded.

For each new enhancement method, inspect the method README and setup files before installing anything. Extract the stated Python, CUDA, PyTorch/TensorFlow, compiler, and package requirements, then compare them against existing environments. Reuse an existing compatible environment whenever possible to save space and reduce dependency drift. Create a new method-specific environment only when the README requirements conflict with existing environments or when custom CUDA/TensorFlow dependencies are fragile enough that isolation is safer.

Environment selection protocol:

- Read the method README, requirements file, and official inference instructions first.
- Check existing conda/venv environments and local method dependency folders.
- Reuse `torch_env` for modern PyTorch methods when required versions are compatible.
- Use method-local `python_deps` only for small pure-Python dependency additions or packages that do not need to replace core PyTorch/CUDA packages.
- Create a dedicated environment for old PyTorch/CUDA, TensorFlow, or custom-extension methods if reuse would break other methods.
- Skip or hold TF1/custom-op methods unless a maintained PyTorch path or explicitly approved TF environment exists.
- Record why the chosen environment was reused or created before running selected-10.
- TensorFlow checkpoint ZIPs downloaded for PU-GCN/PU-Net/MPU/PUGAN-style
  upsampling are not equivalent to PyTorch `.pth` weights. If these methods are
  used, create/record a dedicated TF1/custom-op environment or mark them held.
- Current decision: skip the legacy TensorFlow PU-GCN/PU-Net/MPU/PUGAN path for
  now. The downloaded/extracted checkpoints are kept for provenance, but no TF1
  environment should be built unless the user explicitly reopens this method.

Storage policy:

- Keep CSV/JSON summaries, logs, scripts, and lightweight smoke outputs in this
  repository.
- Move large selected-10/full-sequence PLY output folders to project storage
  under `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/` and leave
  symlinks from the original `results/method_outputs/<method>/` path.
- For future full-sequence runs, write heavy work/output files directly to
  project storage or `$TMPDIR` and copy back only final summaries and symlinks.

Before whole-dataset execution, record one environment entry per method in the method status/protocol notes:

- Conda environment name used by the Slurm job.
- Python, PyTorch, CUDA, PyTorch3D, Open3D, NumPy, SciPy, and other method-critical package versions.
- Whether the method uses the shared `torch_env` or a dedicated method environment.
- All compatibility patches applied to third-party code, especially replacements for old TensorFlow, PointNet++, Chamfer, EMD, FPS, KNN, or custom CUDA extensions.
- Checkpoint filenames, local paths, and source URLs.
- If any checkpoint or external asset cannot be downloaded automatically, record the source URL and exact expected local path, then ask the user to download it manually before continuing that method.
- GPU partition used for validated smoke/selected-10 runs, including any A100 OOM -> H100 retry.

For final full-dataset benchmarking, prefer dedicated per-method environments for methods with old or fragile dependencies. Methods already stable in `torch_env` can continue to use it only if the exact package versions and patches are documented.
