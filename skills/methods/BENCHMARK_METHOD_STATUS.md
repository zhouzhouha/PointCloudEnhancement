# Benchmark Method Status

This file tracks each enhancement/reconstruction method tried for UVG-CWI-DQPC
`OrangeKettlebell`, why it was selected, whether it ran, and what adjustment
would be needed before reading or rerunning the paper/code.

Benchmark rules:

- Keep method internals and official/default inference settings unchanged.
- Only adapt file format, sequence looping, large-frame wrapping, and metrics.
- For geometry-only outputs, transfer RGB from CG input with nearest neighbor
  `k=1`, no averaging.
- Skip TensorFlow 1.x/custom-op methods unless a usable PyTorch version with
  pretrained weights exists.
- Keep SCUTSurface/SUSTech-first reconstruction methods separate from external
  enhancement methods.

Source/provenance labels used below:

- `SCUTSurface/SUSTech`: selected from the SCUTSurface/SUSTech reconstruction list.
- `LilydotEE survey repo`: listed in `https://github.com/LilydotEE/Point_cloud_quality_enhancement`.
- `Independent paper/repo search`: found outside the LilydotEE repo during this benchmark.
- `Traditional baseline`: classical/non-learning baseline selected locally for comparison.

Environment record:

- Most current jobs use the shared Snellius conda environment `torch_env`.
- Frozen package records are stored under `skills/methods/env/`.
- Per-method environment notes and full-dataset rerun requirements are tracked in
  `skills/methods/ENVIRONMENT_REPRODUCIBILITY.md`.

Current scope note:

- Scheduled continuation 2026-07-23 03:26 EEST: the coverage auditor's final
  CSV publication was hardened to match the immutable pipeline: it now opens a
  unique hidden `.part-PID` file exclusively, writes the complete manifest,
  atomically renames it on the same filesystem, and removes a failed partial.
  Exact ROCm Python 3.12 compilation and `git diff --check` pass.
- Scheduled continuation 2026-07-23 03:25 EEST: immutable final-output helper
  `scripts/consolidate_batched_method_manifest.py` was added. It requires an
  exact 2,152-row audit manifest with unique sequence/frame keys and only
  `present`/explicit-precedence `duplicate` statuses; it validates every PLY
  source inside `method_outputs`, refuses an existing run name, builds relative
  symlinks under a hidden same-filesystem staging directory, and atomically
  publishes the complete new run directory. A first negative-test command was
  rejected before execution by the command safety layer because it contained
  shell deletion for temporary cleanup; inspection confirmed it created no
  manifest or target. Non-mutating validation with the exact ROCm Python 3.12
  environment passed compilation, CLI parsing, and `git diff --check`.
  Consolidation remains intentionally disabled until strict coverage reaches
  zero missing frames.
- Scheduled continuation 2026-07-23 03:23 EEST: reusable read-only auditor
  `scripts/audit_batched_method_coverage.py` was added because the PathNet
  consolidator cannot represent ordered immutable retry sources. It validates
  the authoritative CG/HE intersection, records an explicit selected source
  and all alternates per sequence/frame, rejects unexpected outputs, and can
  reject duplicates or incomplete coverage. It compiles under the exact ROCm
  Python 3.12 environment and passes `git diff --check`. An incomplete-safe
  audit over the ordered current/old Grad-PU prefixes proved exactly 2,152
  paired frames, 269 currently present outputs (168 BlueSpeech, 101
  BlueVolley), 1,883 missing, zero duplicate alternates, and zero outputs
  outside paired coverage. No manifest was published yet because the active
  immutable batches are still changing; final strict mode will require zero
  missing frames before consolidation.
- Scheduled continuation 2026-07-23 03:21 EEST: expanded-array accounting
  showed 96 live `small-g` tasks before the next submission, not the 30
  compressed `squeue` lines. The complete 201-frame FitFluencer sequence was
  then added without exceeding the 210-task association ceiling: inference
  array `20161361` covers exact global tasks 63-83 at two-way concurrency,
  followed by corresponding-task k=20 normal array `20161362`,
  texture/perceptual array `20161363`, and real-PCQM array `20161364`.
  FitFluencer inference is success-gated on BouncingBlue inference `20161316`.
  The queue now contains 180 expanded tasks, so further sequence submission is
  intentionally deferred until capacity is released. Active BlueVolley task
  31 remained healthy and advanced to 421 cumulative chunks (75/342 on frame
  0101) with no detected NaN, OOM, traceback, or scheduler error.
- Scheduled continuation 2026-07-23 03:20 EEST: live association inspection
  confirmed `small-g` is up with a 210-task submission allowance (CPU `small`
  and NVIDIA `lumid` remain at zero), and the production MI250X task continued
  normally through BlueVolley frame 0101. To preserve autonomous progress
  without recomputation, the exact 157-frame BouncingBlue wave was queued as
  global tasks 42-57: inference array `20161316` (two-way concurrency), k=20
  normal array `20161317`, texture/perceptual array `20161318`, and real-PCQM
  array `20161319`. Inference is success-gated on completion of the remaining
  BlueVolley inference array `20161262`; each metric task is corresponding-task
  gated. All use the existing immutable ROCm prefix and distinct sequence paths
  and Slurm-derived metric run IDs.
- Scheduled continuation 2026-07-23 03:17 EEST: inspection after frame 0100
  exposed a publication-resilience issue, not a metric failure: Grad-PU's ASCII
  writer created the final PLY pathname before all bytes were flushed. Slurm
  success dependencies prevented any metric consumer from seeing that partial
  file, and the completed frame already passed full validation, but a killed
  future job could otherwise leave a misleading final-named artifact. The
  Grad-PU adapter now rejects an existing final output, writes to a hidden
  same-filesystem `.part-PID` path, and atomically renames only after a complete
  write; failed partials are cleaned. The login-node Python 3.6 compile attempt
  could not parse the script's pre-existing future-annotations directive. This
  known runtime mismatch was resolved by compiling with the exact ROCm job
  environment (Python 3.12), which passed along with `git diff --check`; no
  partial file remains. The already-running task loaded the earlier adapter,
  while queued wave `20161262` will use atomic publication.
- Scheduled continuation 2026-07-23 03:16 EEST: the first production ROCm
  output from task `20161079_31`, BlueVolley frame 0100, was written after all
  346 chunks. A full streaming validation—not only a header check—parsed all
  2,834,464 ASCII vertices, confirmed the exact 4x ratio to the 708,616-point
  CG input, found zero nonfinite XYZ coordinates, zero out-of-range RGB values,
  and zero trailing records. The task continues with geometry evaluation and
  the remaining nine frames; no result is yet being classified as a complete
  ten-frame batch.
- Scheduled continuation 2026-07-23 03:13 EEST: an authoritative paired-frame
  audit found 2,152 dataset pairs, 268 existing immutable Grad-PU outputs
  (168 BlueSpeech and 100 BlueVolley), and 1,884 outputs still required for
  full Grad-PU coverage. Active MI250X task `20161079_31` reached chunk 240/346
  on BlueVolley frame 0100 with no traceback, OOM, fatal message, or nonfinite
  report; its stderr contains only PyTorch's trusted-checkpoint deprecation
  warning. The next exact BlueVolley wave, global tasks 32-38, was submitted as
  inference array `20161262` with two-way GPU concurrency and is success-gated
  on task 31. Corresponding immutable k=20 normal array `20161263`,
  texture/perceptual array `20161264`, and real-PCQM array `20161265` use
  per-task `aftercorr` dependencies. This wave covers the remaining 61
  BlueVolley frames after task 31 without recomputing the 100 older outputs;
  all metric stages retain separate Slurm-derived run IDs.
- Scheduled continuation 2026-07-23 03:11 EEST: the rebuilt ROCm-native PCQM
  binary passed the full import/link smoke, and real execution job `20161113`
  completed with exit code 0 after loading an actual 2,673,059-point reference
  and 2,834,464-point Grad-PU result. Its immutable CSV passed strict validation
  with two rows, two unique method/sequence/frame keys, no duplicates, and two
  finite PCQM values: 0.133755 for the CG baseline and 0.134773 for Grad-PU.
  No PCQM NaN or runtime error remains. Grad-PU production job `20161079_31`
  is genuinely using an MI250X and reached chunk 197/346 on the first of ten
  BlueVolley frames with no traceback, OOM, fatal message, or nonfinite report.
  The k=20 normal job `20161114_31` and
  texture/perceptual job `20161115_31` remain success-gated on production;
  PCQM job `20161116_31` additionally requires the real-PCQM smoke and texture
  stage. No partial artifact is being treated as complete, and every dependent
  stage retains a separate immutable run ID.
- Scheduled continuation 2026-07-23: the final PathNet VirtualLife PCQM task
  `20060550_11` and paper-table job `20060761` completed with exit code 0.
  VirtualLife passed exact validation with 392 rows, 392 unique
  method/sequence/frame keys, all 392 PCQM values finite, and no duplicates.
  The five immutable PathNet paper tables passed their exact expected counts
  (4,304 per-frame rows, 24 per-sequence rows, two dataset means, 12
  per-sequence deltas, and one dataset delta); all 103,296 required per-frame
  geometry, k=20 normal, texture/perceptual, and PCQM values were finite.
  All 17 newly released Grad-PU batch PCQM files passed exact 20-row validation
  (340/340 finite scores), and their corresponding 17 texture/perceptual and
  projection-view files passed exact 20-row and 120-row validation. Strict
  validation initially found four NaN cells in each 16-row texture summary;
  inspection showed these are only the mean/std cells of the two intentional
  `--skip-pcqm` placeholder rows. Real PCQM is stored separately and finite, so
  the immutable texture summaries were not overwritten; later consolidation
  must merge the real PCQM CSVs.
  The active queue was empty and no failed, OOM, timeout, or node-failure job
  was found. Coverage is complete for all 168 Grad-PU BlueSpeech frames and
  100/171 BlueVolley frames. Submission of the next CUDA/A40 wave was rejected
  before creation by `AssocMaxSubmitJobLimit`: live associations now expose
  zero submit slots on both `small` and `lumid`, while `small-g` and
  `standard-g` remain enabled. A deterministic native-PyTorch kNN/FPS fallback
  was therefore added for Grad-PU when PyTorch3D is unavailable, without
  changing the model or checkpoint, and immutable one-frame MI250X/ROCm smoke
  job `20160842` was submitted on `dev-g`. It loaded the official checkpoint,
  advanced to 78/346 chunks with each 2,048-point chunk producing the expected
  8,192 points, and showed no traceback, OOM, or nonfinite error. Separate
  production template `jobs/gradpu_all2152_rocm_batches.slurm` passes shell and
  Slurm preflight for missing BlueVolley task 31 under new immutable prefix
  `gradpu_chunked_4x_full_20260723_rocm1`. Production remains gated on the
  smoke's final shape, finiteness, and objective-metric validation. Three
  separate ROCm post-processing templates for k=20 normals,
  texture/perceptual metrics, and real PCQM also pass shell/Slurm preflight.
  Post-processing environment smoke `20160989` failed before metric execution:
  first, its import audit found missing `skimage`; it also confirmed the prior
  SciPy warning that NumPy 2.5.1 exceeds SciPy's supported `<2.3` range. No
  result artifact was written. First repair job `20161001` installed NumPy
  2.2.6, scikit-image 0.25.2, and LPIPS 0.1.4 and passed their direct imports,
  but pip then reported that NumPy 2.2.6 still conflicts with the existing
  CuPy, Numba, Outlines, and vLLM constraints. Corrective dependency job
  `20161006` is therefore queued to pin NumPy 1.26.4, which satisfies those
  packages plus SciPy/scikit-image. The same LPIPS/normal/PCQM smoke must pass
  after that correction before post-processing is released. A subsequent
  `pip check` exposed one final constraint conflict (`plyfile 1.1.4` required
  NumPy >=2); final repair job `20161024` pinned `plyfile 1.0.3` and tifffile
  2025.6.11 alongside NumPy 1.26.4 and completed with `No broken requirements
  found`. The premature postprocess retry `20161018` was cancelled before it
  ran, and replacement `20161027` is correctly `afterok`-gated on the final
  repair. Grad-PU inference completed 346/346 chunks and wrote frame 0100 with
  exactly 2,834,464 points, four times the 708,616-point input. A first local
  validator incorrectly assumed binary PLY and stopped without changing data;
  the format-aware retry parsed the actual ASCII PLY and verified all XYZ
  coordinates finite, all colors integral in [0,255], and the exact 4x count.
  The smoke job remains inside its geometry-metric stage, so ten-frame
  production is still gated on its final CSV and exit status.
  The smoke subsequently completed with exit code 0 in 12m05s. Its two-row
  geometry CSV passed exact validation with 28 finite geometry/F-score values,
  six intentional normal placeholders, and no duplicate keys; the point-count
  and run-config records also passed. Production task 31 then required two
  launcher repairs before real work began: job `20161054_31` could not resolve
  the container venv during frame discovery because `/project` was not bound,
  and job `20161065_31` saw empty dataset symlinks because their `/scratch`
  targets were not bound. Both exited before creating an output directory.
  The launcher now binds project, flash, and scratch and captures discovery
  status outside process substitution. Retry `20161079_31` discovered the
  exact ten BlueVolley frames 0100-0109 and is actively running on MI250X.
  Postprocess smoke `20161027` confirmed finite MI250X LPIPS and k=20 normal
  output but isolated a PCQM GLIBC mismatch: the CUDA-container binary requires
  GLIBC 2.32/2.34 unavailable in the ROCm image. Separate build job `20161084`
  compiled unchanged PCQM source inside that ROCm image into immutable
  `build_lumi_rocm` and passed its complete link check. End-to-end postprocess
  retry `20161085` is success-gated behind its Slurm epilog. After adding the
  missing scratch bind, production retry `20161079_31` correctly selected
  BlueVolley frames 0100-0109 and is actively processing the first frame on an
  MI250X; no traceback or nonfinite error is present. Its k=20, texture, and
  PCQM dependents remained intentionally unsubmitted until `20161085` passed
  with finite LPIPS, finite 64x3 k=20 normals, and a complete PCQM link check.
  A real one-frame PCQM execution smoke is running as `20161113` rather than
  relying only on linkage. Production k=20 normal job `20161114_31` and
  texture/perceptual job `20161115_31` are correctly `afterok`-gated on
  inference `20161079_31`; real-PCQM job `20161116_31` requires both the
  one-frame execution smoke and texture job to succeed. Every stage writes a
  separate run-ID directory and retains the k=20 ground-truth normal cache.
- Scheduled continuation 2026-07-22: Grad-PU BlueVolley batch-9 real-PCQM
  task `20123244_30` completed with exit code 0 after all ten frame pairs. Its
  separate immutable CSV passed exact validation with 20 rows, 20 unique
  method/sequence/frame keys, all 20 PCQM values finite, and zero duplicates
  or nonfinite values. PathNet VirtualLife advanced to 377/392 finite
  emissions. The PathNet paper-table job and the remaining separate Grad-PU
  texture/PCQM chains remain correctly success-gated on the full PathNet PCQM
  array. No NaN/infinite value, traceback, OOM, timeout, node failure, or
  dependency error was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-8 real-PCQM task
  `20123244_29` completed with exit code 0. Its immutable BlueVolley CSV passed
  exact validation with 20 rows, 20 unique method/sequence/frame keys, all 20
  PCQM values finite, and zero duplicates or nonfinite values. The intentional
  `%1` array limit released batch-9 task `20123244_30`, which started normally.
  PathNet VirtualLife advanced to 327/392 finite emissions (196 paired frames,
  with one CG-baseline and one PathNet score per frame), while the PathNet
  table/consolidation job remained correctly dependency-gated on that final
  sequence. No unexpected NaN/infinite value, traceback, OOM, timeout, node
  failure, or dependency error was found. A login-node Python 3.6 compile check
  rejected `from __future__ import annotations`; this was confirmed to be a
  diagnostic-runtime mismatch rather than a pipeline defect by compiling the
  same scripts successfully with the actual CPU container's Python 3.11.9.
  Older pending Grad-PU texture/PCQM arrays were also audited and are
  intentionally success-gated on the complete PathNet PCQM array, not stuck
  in `DependencyNeverSatisfied`; their immutable run IDs remain separate.
  Grad-PU batch-9 advanced to 19/20 finite PCQM values. These PCQM stages are
  appropriately CPU-scheduled; GPU allocations remain reserved for compatible
  neural inference. The queued PathNet paper-table stage was preflighted: its
  target directory is absent, overwrite guards are active, its Slurm script
  passes `bash -n`, and the builder compiles with the actual Python 3.11.9 CPU
  container. It remains correctly `afterok`-gated on all PathNet PCQM tasks.
  All 11 completed PathNet sequence PCQM CSVs were revalidated against their
  exact paired-frame-derived row counts: 3,912/3,912 rows and unique keys were
  present, every score was finite, and there were zero duplicates. VirtualLife
  is the only remaining PathNet PCQM sequence.
  Grad-PU batch-9's preceding immutable stages were also revalidated exactly:
  geometry has 20 unique rows and 280 finite geometry/F-score values plus 60
  intentional normal placeholders; the separate `pca_knn20` CSV has 20 unique
  rows and 60 finite normal values; texture/perceptual has 20 unique rows and
  120 finite values plus 20 intentional PCQM placeholders. Thus real PCQM is
  its only unfinished objective-metric stage.
  Both active log files were confirmed to have current modification times and
  ongoing frame output, ruling out a silent scheduler-visible stall.
  Grad-PU batch-9's sixth comparison is unusually long, but its live `srun`
  step reported about nine CPU-minutes, 2.1 GiB RSS, active disk I/O, an empty
  stderr, and 10:44 elapsed against a 12-hour limit. This is active PCQM
  computation on a 2.52-million-point prediction, not a detected hang. The
  comparison subsequently completed with a finite value and the job advanced
  normally to the next frame.
  A BlueVolley PCQM coverage audit corrected two local path assumptions
  (`seq -w` did not zero-pad this range, and batches 0-4 are not members of
  the current retry1 PCQM run). Direct discovery showed batches 5-8 complete;
  all four immutable CSVs passed exact 20-row/20-key finite validation (80/80
  values, no duplicates or nonfinite scores). Batches 0-4 remain intentionally
  queued in the separate `20123099` chain behind PathNet-dependent texture
  jobs; batch 9 is the active current retry1 task. No benchmark output was
  changed by the failed audit path lookups.
- Scheduled continuation 2026-07-22: PathNet VictoryHeart real-PCQM task
  `20060550_10` completed with exit code 0. Its immutable sequence CSV passed
  exact validation with 394 rows, 394 unique method/sequence/frame keys, every
  PCQM value finite, and zero duplicates or nonfinite values. VirtualLife
  continued to 165 finite emissions; the PathNet table/consolidation job
  remains correctly dependency-gated on that final sequence only. Grad-PU
  batch-8 real PCQM reached 17 finite emissions. No unexpected NaN/infinite
  value, traceback, OOM, timeout, node failure, or dependency error was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-7 real-PCQM task
  `20123244_28` completed with exit code 0. Its separate immutable CSV passed
  exact validation with 20 rows, 20 unique method/sequence/frame keys, all 20
  PCQM values finite, and zero duplicates or nonfinite values. The intentional
  `%1` array limit immediately released batch-8 task `20123244_29`, which
  started normally and advanced to ten finite PCQM values. PathNet
  VictoryHeart advanced to 389 finite emissions and VirtualLife to 124;
  consolidation remains correctly dependency-gated until both finish. No
  manual resubmission, overwrite, unexpected NaN/infinite value, traceback,
  OOM, timeout, node failure, or dependency error was found.
- Scheduled continuation 2026-07-22: IterativePFN final partial batch-16
  real-PCQM task `20125032_16` completed with exit code 0. Its separate
  immutable CSV passed exact validation with 16 rows, 16 unique
  method/sequence/frame keys, all 16 PCQM values finite, and zero duplicates
  or nonfinite values. This completes geometry, k=20 normals,
  texture/perceptual metrics, and real PCQM for the manifest-defined eight
  BlueSpeech frames in this batch. Grad-PU batch-7 remained healthy at 19/20
  finite PCQM emissions while computing its final row; PathNet VictoryHeart
  reached 338 and VirtualLife 62. No unexpected NaN/infinite value, traceback,
  OOM, timeout, node failure, or dependency error was found.
- Scheduled continuation 2026-07-22: IterativePFN final partial batch-16 k=20
  normal and texture/perceptual jobs `20125030_16`/`20125031_16` completed with
  exit code 0. The immutable normal CSV passed exact validation with 16 rows,
  16 unique keys, all 48 normal values finite, and no duplicates or nonfinite
  values. The separate texture CSV passed with 16 unique rows, all 96
  Y/U/V/YUV-PSNR, projection-SSIM, and LPIPS values finite, and exactly 16
  intentional `--skip-pcqm` placeholders. Real-PCQM task `20125032_16` then
  released and advanced to 11 finite emissions. Grad-PU batch-7 real PCQM
  remained at 17 finite emissions while computing the next row; PathNet
  VictoryHeart reached 330 and VirtualLife 54. No enhanced
  normal cache was retained, and no unexpected NaN/infinite value, traceback,
  OOM, timeout, node failure, or dependency error was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-9 k=20 normal task
  `20123242_30` completed with exit code 0. Its separate immutable
  `run_20123242_pca_knn20` CSV passed exact validation with 20
  baseline/method rows, 20 unique keys, all 60 `N_Acc`, `N_Comp`, and combined
  normal values finite, and zero duplicates or nonfinite values. This confirms
  that both the texture/perceptual and k=20 normal branches for this batch are
  complete; its real-PCQM task remains released behind the intentional `%1`
  limit. No enhanced normal cache was retained by this metric stage.
- Scheduled continuation 2026-07-22: IterativePFN final partial BlueSpeech
  batch-16 inference job `20125029_16` completed with exit code 0 and produced
  its manifest-defined 8/8 frames (`0160`-`0167`), correcting the earlier
  interim denominator of ten. Its immutable geometry CSV passed exact
  validation with 16 baseline/method rows, 16 unique keys, all 224 required
  geometry values finite, zero duplicates or unexpected nonfinite values, and
  exactly 48 intentional normal placeholders. Its k=20 normal and
  texture/perceptual tasks `20125030_16`/`20125031_16` started automatically;
  real PCQM remains correctly dependency-gated behind texture completion.
  Grad-PU batch-9 k=20 normals reached 9/10 frames, Grad-PU batch-7 real PCQM
  13 finite emissions, PathNet VictoryHeart 300, and VirtualLife 27. No
  unexpected NaN/infinite value, traceback, OOM, timeout, node failure, or
  dependency error was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-9 texture/perceptual task
  `20123243_30` completed with exit code 0. Its separate immutable CSV passed
  exact validation with 20 baseline/method rows, 20 unique keys, all 120
  Y/U/V/YUV-PSNR, projection-SSIM, and LPIPS values finite, zero duplicates,
  and exactly 20 intentional `--skip-pcqm` placeholders. Its real-PCQM task
  is released and waits only behind the intentional `%1` array limit. The
  matching k=20 normal job reached 8/10 frames. Grad-PU batch-7 real PCQM
  reached 13 finite emissions; PathNet VictoryHeart reached 295 and
  VirtualLife 23. IterativePFN batch-16 inference completed its expected 8/8 immutable
  outputs. No unexpected NaN/infinite value, traceback, OOM, timeout, node
  failure, or dependency error was found, and no duplicate submission or
  repair was required.
- Scheduled continuation 2026-07-22: IterativePFN batch-14 real-PCQM task
  `20125032_14` completed with exit code 0. Its immutable CSV passed exact
  validation with 20 rows, 20 unique method/sequence/frame keys, all 20 PCQM
  values finite, and zero duplicates or nonfinite values. Grad-PU batch-9 k=20
  normals reached 3/10 frames and its texture/perceptual stage produced four
  method rows with only intentional `--skip-pcqm` placeholders. Grad-PU
  batch-7 real PCQM reached nine finite emissions. PathNet VictoryHeart
  advanced to 272 finite emissions and newly released VirtualLife to five;
  IterativePFN batch-16 inference remained healthy at 6/8 outputs. No
  unexpected NaN/infinite value, traceback, OOM, timeout, node failure, or
  dependency error was found.
- Scheduled continuation 2026-07-22: PathNet TicTacToe real-PCQM task
  `20060550_8` completed with exit code 0. Its immutable CSV passed exact
  validation with 330 rows, 330 unique method/sequence/frame keys, every PCQM
  value finite, and zero duplicates or nonfinite values; the intentional `%2`
  limit then released VirtualLife task `20060550_11`, which started normally.
  VictoryHeart advanced to 268 finite PCQM emissions. Grad-PU batch-9 k=20
  normal and texture jobs `20123242_30`/`20123243_30` are running after their
  validated inference dependency; texture logs contain only the expected
  `--skip-pcqm` placeholders, with real PCQM isolated downstream. Grad-PU
  batch-7 reached eight finite real-PCQM emissions, IterativePFN batch-14
  reached 17, and IterativePFN batch-16 inference reached 6/8 outputs. No
  unexpected NaN/infinite value, traceback, OOM, timeout, node failure, or
  dependency error was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-9 inference job
  `20123239_30` produced all ten requested BlueVolley frames and completed
  with exit code 0. Its newly written immutable geometry CSV passed exact
  validation with 20 baseline/method rows, 20 unique keys, all 280 required
  geometry values finite, zero duplicates or unexpected nonfinite values, and
  exactly 60 intentional normal placeholders. Its k=20 normal task
  `20123242_30` was then released automatically and started; texture and
  real-PCQM remain correctly chained behind it. IterativePFN batch-16 advanced
  to 5/10 outputs.
  IterativePFN batch-14 real PCQM reached six finite emissions, Grad-PU batch-7
  PCQM four, and PathNet reached 322 finite TicTacToe and 246 VictoryHeart
  emissions. No traceback, OOM, timeout, node failure, unexpected NaN/infinite
  value, or dependency error was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-13 real PCQM task
  `20125181_13` completed with exit code 0. Its immutable CSV passed exact
  validation with 20 rows, 20 unique method/sequence/frame keys, all 20 PCQM
  values finite, and zero duplicates or nonfinite values. IterativePFN
  batch-14 PCQM advanced to five finite emissions; Grad-PU batch-7 PCQM had
  three. PathNet reached 319 finite TicTacToe and 239 VictoryHeart emissions.
  A40 inference remained healthy at 9/10 Grad-PU batch-9 outputs and 4/10
  IterativePFN batch-16 outputs. No traceback, OOM, timeout, node failure,
  unexpected NaN/infinite value, or dependency error was found, so the
  existing immutable dependency chains were left running.
- Scheduled continuation 2026-07-22: IterativePFN batch-15 real PCQM task
  `20125032_15` completed with exit code 0. Its immutable result CSV passed
  exact validation with 20 rows, 20 unique method/sequence/frame keys, all 20
  PCQM values finite, and zero duplicates or nonfinite values. Batch-14 and
  batch-13 PCQM are active with one and 16 finite emissions, respectively.
  Grad-PU batch-7 PCQM is active and batch 8 remains queued behind the
  intentional `%1` concurrency limit. GPU inference advanced to 9/10 outputs
  for Grad-PU batch 9 and 4/10 for IterativePFN batch 16. PathNet reached 317
  finite TicTacToe emissions out of 330 and 234 VictoryHeart emissions out of
  394. No traceback, OOM, timeout, node failure, unexpected NaN/infinite
  value, or new dependency error was found; all downstream normal, texture,
  and PCQM stages remain dependency-queued under immutable run names.
- Scheduled continuation 2026-07-22: Grad-PU batch-6 real PCQM task
  `20123244_27` completed with exit code 0. Its immutable CSV passed exact
  validation with 20 rows, 20 unique method/sequence/frame keys, all 20 PCQM
  values finite, and zero duplicates or nonfinite values. Batch-7 PCQM task
  `20123244_28` started automatically, while batch 8 remains queued behind the
  intentional `%1` limit. IterativePFN batch-15 and batch-13 PCQM advanced to
  18 and 10 finite emissions. PathNet reached 312 finite TicTacToe and 224
  finite VictoryHeart emissions. GPU inference remained healthy at 8/10 for
  Grad-PU batch 9 and 3/10 for IterativePFN batch 16. No unexpected
  NaN/infinite value, failure, or repair was found.
- Scheduled continuation 2026-07-22: all active real-PCQM streams remained
  finite. Grad-PU batch-6 reached 19/20 emissions and is inside the native
  PCQM statistics computation for its final row; no stall or error is shown.
  IterativePFN batch-15 reached 11 finite PCQM emissions and batch-13 reached
  three. PathNet reached 308 finite TicTacToe and 215 finite VictoryHeart
  emissions. GPU inference advanced to 8/10 for Grad-PU batch 9 and 3/10 for
  IterativePFN batch 16. No unexpected NaN/infinite value, traceback, OOM,
  timeout, dependency failure, or repair was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-14 texture task
  `20125031_14` completed with exit code 0 and passed exact validation with 20
  rows, 20 unique keys, all 120 Y/U/V/YUV, projection-SSIM, and LPIPS values
  finite, zero duplicates, and exactly 20 intentional `--skip-pcqm`
  placeholders. Batch-12 real PCQM task `20125181_12` also completed and
  passed with 20 unique rows and all 20 PCQM values finite; batch-13 PCQM
  started automatically. Batch-14 PCQM remains queued behind its separate
  intentional `%1` limit. Before completion, PathNet reached 303 finite
  TicTacToe and 204 finite VictoryHeart emissions, Grad-PU batch-6 reached 17,
  and IterativePFN batch-15 reached three finite PCQM emissions. GPU inference
  remained healthy at 7/10 for Grad-PU batch 9 and 2/10 for IterativePFN
  batch 16. No unexpected NaN/infinite value or failure was found.
- Scheduled continuation 2026-07-22: dependency release remained healthy.
  IterativePFN batch-14 texture/perceptual evaluation reached ten rows, all
  YUV-PSNR, projection-SSIM, and LPIPS values finite with only intentional
  PCQM placeholders. Batch-15 real PCQM task `20125032_15` started and is in
  initialization. Batch-12 PCQM advanced to 14 finite emissions, Grad-PU
  batch-6 to 17, and PathNet to 300 finite TicTacToe and 199 finite
  VictoryHeart emissions, all with zero NaN/infinite values. GPU inference
  reached 7/10 for Grad-PU batch 9 and 2/10 for IterativePFN batch 16. No
  traceback, OOM, timeout, dependency failure, or repair was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-14 and batch-15 k=20
  normal jobs `20125030_14`/`_15` completed with exit code 0. Each immutable
  CSV passed exact validation with 20 rows, 20 unique keys, all 60
  `N_Acc`/`N_Comp`/combined-normal values finite, and zero duplicates or
  nonfinite values. Batch-15 texture task `20125031_15` also completed and
  passed with 20 unique rows, all 120 texture/perceptual values finite, and
  exactly 20 intentional `--skip-pcqm` placeholders. Batch-14 texture started
  automatically, while batch-15 real PCQM remains dependency-gated only
  through Slurm epilog. PathNet reached 297 finite TicTacToe and 192 finite
  VictoryHeart emissions; Grad-PU batch-6 had 15 and IterativePFN batch-12 had
  nine finite real PCQM emissions. GPU inference advanced to 7/10 for Grad-PU
  batch 9 and remains healthy for IterativePFN batch 16. No unexpected
  NaN/infinite value or failure was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-14/batch-15 k=20
  normal evaluation advanced to 6/10 for both batches. Batch-15
  texture/perceptual evaluation reached 13 rows, with every YUV-PSNR,
  projection-SSIM, and LPIPS value finite and only intentional PCQM
  placeholders. Batch-12 real PCQM reached six finite emissions. PathNet
  reached 295 finite TicTacToe and 187 finite VictoryHeart emissions, while
  Grad-PU batch-6 reached 15 finite PCQM emissions. GPU inference remains
  healthy at 6/10 for Grad-PU batch 9 and 1/10 for IterativePFN batch 16.
  No unexpected NaN/infinite value, traceback, OOM, timeout, dependency
  failure, or repair was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-14/batch-15 k=20
  normal tasks started and advanced to 2/10 and 3/10. Batch-15
  texture/perceptual evaluation started with four rows, all YUV-PSNR,
  projection-SSIM, and LPIPS values finite and only intentional PCQM
  placeholders; batch-14 texture is queued behind the intentional `%1`
  limit. Batch-12 real PCQM started with two finite emissions. GPU inference
  advanced to 6/10 for Grad-PU batch 9, while IterativePFN batch 16 produced
  its first immutable output. PathNet reached 292 finite TicTacToe and 182
  finite VictoryHeart emissions; Grad-PU batch-6 reached 14 finite PCQM
  emissions. Explicit parsing found zero NaN/infinite real PCQM values and no
  new failure, so no repair was required.
- Scheduled continuation 2026-07-22: IterativePFN batch-14 inference
  `20125029_14` completed with exit code 0 and all ten immutable outputs. Its
  geometry CSV passed exact validation with 20 rows, 20 unique keys, all 280
  geometry values finite, zero duplicates, and exactly 60 intentional normal
  placeholders. Real PCQM jobs for retry5 batch 11 (`20125181_11`) and retry5
  batch 8 (`20125334_8`) also completed with exit code 0; each immutable CSV
  passed with 20 unique rows and all 20 PCQM values finite. Batch-15 k=20 and
  texture tasks are scheduler-eligible with `Priority`; batch-14 dependents
  remain held only through its Slurm epilog. Remaining PathNet PCQM reached
  289 finite TicTacToe and 174 finite VictoryHeart emissions, while Grad-PU
  batch-6 reached 13 finite emissions. No unexpected NaN/infinite value,
  failure, or repair was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-15 inference
  `20125029_15` completed with exit code 0 and all ten immutable outputs. Its
  geometry CSV passed exact validation with 20 rows, 20 unique keys, all 280
  geometry values finite, zero duplicates, and exactly 60 intentional normal
  placeholders. The prior all-NaN summary-warning fix was confirmed effective:
  this completed run emitted no `nanmean`/`nanstd` warning. Batch 16 started
  automatically on CUDA A40; batch 14 remains healthy at 9/10. Real PCQM
  streams contained 285 finite PathNet TicTacToe, 166 finite VictoryHeart,
  11 finite Grad-PU batch-6, and 14 finite values each for IterativePFN
  batches 8 and 11, with zero NaN/infinite values. No new failure or repair
  was required.
- Scheduled continuation 2026-07-22: IterativePFN base batch-3 real PCQM
  task `20125329_3` completed with exit code 0 and passed exact validation with
  20 rows, 20 unique method/sequence/frame keys, all 20 PCQM values finite,
  and zero duplicates or nonfinite values. Grad-PU batch-7 texture task
  `20123243_28` also completed and passed with 20 unique rows, all 120
  Y/U/V/YUV, projection-SSIM, and LPIPS values finite, and exactly 20
  intentional `--skip-pcqm` placeholders. Its separate PCQM task remains
  correctly gated only through Slurm epilog. At inspection, GPU inference was
  4/10 for Grad-PU batch 9 and 8/10 and 9/10 for IterativePFN batches 14/15.
  PathNet had 279 finite TicTacToe and 155 finite VictoryHeart PCQM emissions;
  Grad-PU batch-6 had ten, IterativePFN batch-8 had six, and batch-11 had six
  finite real PCQM emissions. No unexpected NaN/infinite value or job failure
  was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-13 texture task
  `20125180_13` completed with exit code 0. Its immutable flash CSV passed
  exact validation with 20 rows, 20 unique method/sequence/frame keys, all
  120 Y/U/V/YUV, projection-SSIM, and LPIPS values finite, zero duplicates,
  and exactly 20 intentional `--skip-pcqm` placeholders. Its separate PCQM
  task remains correctly dependency-gated through Slurm epilog. Grad-PU
  batch-7 texture advanced to 9/20 finite rows. GPU inference advanced to
  4/10 for Grad-PU batch 9 and 8/10 for both IterativePFN batches 14 and 15.
  PathNet reached 274 finite TicTacToe and 146 finite VictoryHeart PCQM
  emissions; Grad-PU batch-6 had nine and IterativePFN base batch-3 had ten
  finite real PCQM emissions. No unexpected NaN/infinite value, failure, or
  repair was found.
- Scheduled continuation 2026-07-22: real IterativePFN PCQM tasks for retry5
  batch 10 (`20125181_10`) and retry5 batch 6 (`20125334_6`) completed with
  exit code 0. Each immutable CSV passed exact validation with 20 rows, 20
  unique method/sequence/frame keys, all 20 PCQM values finite, zero
  duplicates, and zero nonfinite values. Batch-8 PCQM is queued behind the
  intentional `%1` limit while batch-6 exits epilog; base batch-3 PCQM is
  active. Grad-PU batch-7 and IterativePFN batch-13 texture jobs started and
  emitted 2 and 3 rows, respectively, with all texture/perceptual values
  finite and only expected `--skip-pcqm` placeholders. PathNet reached 270
  finite TicTacToe and 137 finite VictoryHeart emissions; Grad-PU batch-6
  reached seven finite real PCQM emissions. No unexpected NaN/infinite value,
  failure, or repair was found.
- Scheduled continuation 2026-07-22: Grad-PU batch-8 and batch-7 k=20
  normal jobs `20123242_29`/`_28` completed with exit code 0. Each immutable
  CSV passed exact validation with 20 rows, 20 unique keys, all 60
  `N_Acc`/`N_Comp`/combined-normal values finite, and zero duplicates or
  nonfinite values. IterativePFN batch-12 texture task `20125180_12` completed
  and passed with 20 unique rows, 120 finite texture/perceptual values, and
  exactly 20 intentional PCQM placeholders. Base batch-1 real PCQM task
  `20125329_1` completed and passed with 20 unique finite PCQM rows; batch-3
  PCQM is queued behind its intentional `%1` limit. Active PCQM streams had
  266 finite PathNet TicTacToe, 128 finite VictoryHeart, six finite Grad-PU
  batch-6, 12 finite IterativePFN batch-6, and 14 finite IterativePFN batch-10
  emissions, with zero NaN/infinite values. No failure or repair was required.
- Scheduled continuation 2026-07-22: Grad-PU batch-8 texture/perceptual job
  `20123243_29` completed with exit code 0. Its immutable flash CSV passed
  exact validation with 20 rows, 20 unique keys, all 120 Y/U/V/YUV,
  projection-SSIM, and LPIPS values finite, zero duplicates, and exactly 20
  intentional `--skip-pcqm` placeholders. Its separate PCQM task remains
  dependency-gated only while the completed texture task exits Slurm epilog;
  batch-7 texture is queued behind the intentional `%1` limit. Grad-PU
  batch-7/batch-8 k=20 normal evaluation advanced to 7/10 and 8/10.
  PathNet reached 262 finite TicTacToe and 123 finite VictoryHeart emissions;
  Grad-PU batch-6 reached five, and IterativePFN batch 1/6/10 reached
  16/7/9 finite real PCQM emissions. Explicit parsing found zero
  NaN/infinite real PCQM values and no new job failure.
- Scheduled continuation 2026-07-22: IterativePFN batch-11 texture task
  `20125180_11` completed with exit code 0 and passed validation with 20 rows,
  20 unique keys, all 120 texture/perceptual values finite, zero duplicates,
  and exactly 20 intentional `--skip-pcqm` placeholders. Real IterativePFN
  PCQM tasks for retry5 batch 9 (`20125181_9`) and retry6 batch 7
  (`20125336_7`) also completed and passed exact validation; each CSV has 20
  unique rows and all 20 PCQM values finite. Batch-12 texture started
  automatically, while batch-11 PCQM is queued behind the intentional `%1`
  limit. Grad-PU batch-7/batch-8 k=20
  normal jobs advanced to 3/10 and 4/10, respectively; batch-8 texture had 11
  rows with all YUV-PSNR/projection-SSIM/LPIPS values finite and only expected
  PCQM placeholders. PathNet reached 255 finite TicTacToe and 111 finite
  VictoryHeart PCQM emissions; Grad-PU batch-6 had three and IterativePFN
  batch-1 had six finite real PCQM emissions. No unexpected NaN/infinite
  value, failure, or repair was found.
- Scheduled continuation 2026-07-22: repaired IterativePFN texture tasks for
  base batch 3 (`20125328_3`) and retry5 batch 8 (`20125333_8`) completed with
  exit code 0. Each CSV passed validation with 20 rows, 20 unique keys, all
  120 texture/perceptual values finite, zero duplicates, and exactly 20
  intentional `--skip-pcqm` placeholders. Three real IterativePFN PCQM tasks
  also completed and passed exact validation: base batch 0 (`20125329_0`),
  retry1 batch 2 (`20125332_2`), and retry5 batch 5 (`20125334_5`); each has
  20 rows, 20 unique keys, and all 20 PCQM values finite. Base batch-1 PCQM
  started automatically. Grad-PU batches 7/8 k=20 normal jobs are active,
  batch-8 texture metrics are active, and Grad-PU batch 9 is running on CUDA
  A40. Remaining real PCQM streams had 251 finite PathNet TicTacToe, 103
  finite VictoryHeart, one finite Grad-PU batch-6, 19 finite IterativePFN
  batch-9, and 17 finite IterativePFN batch-7 emissions, with zero
  NaN/infinite values. No failure or repair was required.
- Scheduled continuation 2026-07-22: Grad-PU inference batches 7 and 8
  (`20123239_28`/`_29`) completed with exit code 0. Each immutable geometry
  CSV passed exact validation with 20 rows, 20 unique keys, all 280 geometry
  values finite, zero duplicates, and exactly 60 intentional normal
  placeholders. Batch-8 k=20 normal and texture/perceptual tasks
  `20123242_29`/`20123243_29` started after its Slurm epilog; batch-7
  dependents remain correctly held only while its epilog completes. Grad-PU
  batch 9 started on CUDA A40. Repaired IterativePFN batch-10 texture job
  `20125180_10` also completed with exit code 0; its CSV passed validation
  with 20 unique rows, all 120 texture/perceptual values finite, and exactly
  20 intentional PCQM placeholders. Repaired texture batches 3 and 8 continue
  producing finite values. Real PCQM streams contained 247 finite
  PathNet TicTacToe, 96 finite PathNet VictoryHeart, one finite Grad-PU
  batch-6, and 13/15/12/13/10 finite IterativePFN batch 9/0/2/5/7 emissions,
  respectively, with zero NaN/infinite values. No job failure or repair was
  required.
- Scheduled continuation 2026-07-22: corrected IterativePFN texture tasks for
  base batch 1 (`20125328_1`) and retry5 batch 6 (`20125333_6`) completed with
  exit code 0. Each flash CSV passed exact validation with 20 rows, 20 unique
  keys, all 120 texture/perceptual values finite, zero duplicates, and exactly
  20 intentional `--skip-pcqm` placeholders. Grad-PU batch-5 PCQM job
  `20123244_26` completed with exit code 0; its immutable CSV passed validation
  with 20 rows, 20 unique keys, and all 20 PCQM values finite. Batch-6 PCQM
  task `20123244_27` started automatically. Active IterativePFN PCQM tasks had
  9, 10, 7, 7, and 5 finite emissions for batches 9, 0, 2, 5, and 7,
  respectively, with zero NaN/infinite values. PathNet reached 243 finite
  TicTacToe and 88 finite VictoryHeart emissions. Grad-PU batch 8 produced all
  ten immutable inference outputs and is finalizing geometry metrics; batch 7
  has nine outputs. No failure or corrective resubmission was required.
- Scheduled continuation 2026-07-22: four corrected IterativePFN
  texture/perceptual tasks completed with exit code 0: base batch 0
  (`20125328_0`), retry1 batch 2 (`20125330_2`), retry5 batch 5
  (`20125333_5`), and retry6 batch 7 (`20125335_7`). Each immutable flash CSV
  passed exact validation with 20 rows, 20 unique keys, all 120 Y/U/V/YUV,
  projection-SSIM, and LPIPS values finite, zero duplicates, and exactly 20
  intentional `--skip-pcqm` placeholders. Successor texture tasks for base
  batch 1 and retry5 batch 6 started automatically. IterativePFN batch-9 real
  PCQM task `20125181_9` started and emitted three finite values with zero
  NaN/infinite values. PathNet reached 238 finite TicTacToe and 78 finite
  VictoryHeart emissions; Grad-PU batch-5 reached 18 finite PCQM emissions.
  No new failure or repair was required.
- Scheduled continuation 2026-07-22: repaired IterativePFN batch-9 texture
  task `20125180_9` completed with exit code 0 and automatically released
  batch 10. Its immutable flash CSV passed exact validation with 20 rows, 20
  unique method/sequence/frame keys, all 120 Y/U/V/YUV-mean,
  projection-SSIM, and LPIPS values finite, zero duplicates, and exactly 20
  expected `--skip-pcqm` placeholders. The first validator attempt used
  nonexistent shorthand fields `projection_ssim`/`lpips`; rerunning with the
  actual schema fields `projection_ssim_mean`/`projection_lpips_mean` passed,
  confirming this was a command error rather than a data error. PCQM task
  `20125181_9` remains correctly dependency-gated only while Slurm finishes
  the texture-task epilog. The four older repaired texture arrays are also
  producing finite values: at inspection they had 4, 4, 4, and 2 rows, all
  finite for texture/perceptual metrics and only intentional PCQM NaNs.
  PathNet reached 230 finite TicTacToe and 64 finite VictoryHeart PCQM
  emissions; Grad-PU batch-5 reached 15 finite PCQM emissions. No unexpected
  NaN/infinite value or job failure was found.
- Scheduled continuation 2026-07-22: repaired the remaining unrelated-PathNet
  dependency affecting eight completed IterativePFN batches. Confirmed batches
  0, 1, 2, 3, 5, 6, 7, and 8 had complete immutable inference outputs but no
  texture or PCQM result files because never-started jobs `20122707` through
  `20122714` were gated on the full PathNet array. Those blocked metric jobs
  were cancelled and replaced with prefix-correct immutable arrays: base
  batches 0/1/3 use texture `20125328` and PCQM `20125329`; retry1 batch 2 uses
  `20125330`/`20125332`; retry5 batches 5/6/8 use `20125333`/`20125334`; and
  retry6 batch 7 uses `20125335`/`20125336`. The four first texture tasks are
  running concurrently on CPU nodes, and each PCQM array was verified to have
  the intended `aftercorr` dependency. The previously repaired batch-9
  texture job is also active and emitting finite YUV-PSNR,
  projection-SSIM, and LPIPS values. Its logged PCQM NaNs are expected
  `--skip-pcqm` placeholders; separate PCQM job `20125181` remains linked.
  No unexpected NaN/infinite metric, traceback, OOM, timeout, or node failure
  was found.
- Scheduled continuation 2026-07-22: IterativePFN batch-13 k=20 normal job
  `20122039_13` completed with exit code 0. Its flash CSV passed exact
  validation with 20 rows, 20 unique method/sequence/frame keys, all 60
  `N_Acc`/`N_Comp`/combined-normal values finite, and zero duplicates or
  unexpected nonfinite values. Inspection found that completed IterativePFN
  batches 9-13 had texture tasks `20123096_[9-13]` incorrectly gated on the
  unrelated full PathNet array; none had started or written metric results.
  The five blocked texture tasks and their never-started PCQM dependents
  `20123097_[9-13]` were cancelled. Replacement immutable texture/perceptual
  array `20125180_[9-13%1]` was submitted without the unrelated dependency and
  started task 9; replacement PCQM array `20125181_[9-13%1]` is correctly
  linked by `aftercorr:20125180`. At repair time, PathNet had 221 finite
  TicTacToe and 46 finite VictoryHeart PCQM emissions, Grad-PU batch-5 had 12
  finite PCQM emissions, Grad-PU batches 7/8 had 7/10 outputs, and new
  IterativePFN batches 14/15 had each produced their first immutable output.
  Explicit parsing found no NaN/infinite PCQM emission.
- Scheduled continuation 2026-07-22: no new failed, OOM, timeout, node-failure,
  traceback, CUDA-memory, or dependency error was found. Grad-PU batches 7
  and 8 advanced to 7/10 immutable outputs each. IterativePFN batch-13 k=20
  normal evaluation advanced to 5/10; new CUDA A40 inference batches 14 and
  15 are initialized and running, with batch 16 queued behind the intentional
  `%2` limit. PathNet PCQM reached 218 finite TicTacToe emissions and 41
  finite VictoryHeart emissions, while Grad-PU batch-5 PCQM reached 11 finite
  emissions. Explicit parsing found zero NaN/infinite PCQM emissions. No
  resubmission or dependency change was required.
- Scheduled continuation 2026-07-22: IterativePFN batch 13 completed with
  exit code 0 and all ten immutable outputs. Direct flash-side validation of
  its geometry CSV proved 20 rows, 20 unique method/sequence/frame keys, 280
  finite geometry values, zero unexpected nonfinite values, and exactly 60
  intentional normal placeholders. Its k=20 normal job `20122039_13` then
  started, and controller `20122911` completed successfully after submitting
  the next inference wave `20125029_[14-16%2]`, matching k=20 array `20125030`,
  texture/perceptual array `20125031`, and separate PCQM array `20125032`.
  IterativePFN batches 14 and 15 are running on CUDA A40 and batch 16 is
  queued. Grad-PU batches 7 and 8 each reached 6/10 immutable outputs. PathNet
  PCQM reached 213 finite TicTacToe emissions and 32 finite VictoryHeart
  emissions; Grad-PU batch-5 PCQM reached nine finite emissions. No metric
  emission was NaN or infinite. The geometry summary did emit warnings for
  its deliberately all-NaN normal placeholder columns; the attempted and
  adopted fix in `run_mag_selected_frames.py` filters finite values and writes
  an explicit NaN mean/std when none exist. A focused test verified correct
  finite aggregation and intentional NaN preservation with zero warnings;
  Python compilation and `git diff --check` passed.
- Scheduled continuation 2026-07-22: live Slurm accounting showed no new
  FAILED, OUT_OF_MEMORY, TIMEOUT, or NODE_FAIL jobs. PathNet PCQM reached
  208 finite TicTacToe emissions (104/165 paired frames) and 22 finite
  VictoryHeart emissions (11/197 paired frames); VirtualLife remains queued
  behind the intentional `%2` array limit. IterativePFN batch 13 produced
  9/10 immutable outputs and is processing its final frame. Grad-PU batches
  7 and 8 remain active, with batch 7 producing its next immutable output and
  batch 8 actively processing chunks; batch 9 remains queued behind `%2`.
  Batch-5 separate PCQM retains seven finite completed emissions and is inside
  the native curvature/statistics computation for its next result. Batch-6
  k=20 normal and texture/perceptual CSVs remain completed and validated;
  batch-6 separate PCQM is queued behind the intentional `%1` metric limit.
  Python compilation, shell syntax checks, and `git diff --check` passed. No
  unexpected NaN/infinite value or actionable pipeline error was found, so no
  repair or unsafe resubmission was needed.
- Scheduled continuation 2026-07-22: Grad-PU batch-6 k=20 normal and
  texture/perceptual jobs `20123242_27` and `20123243_27` completed with exit
  code 0. Flash-side validator `20124878` proved both immutable CSVs have 20
  rows, 20 unique method/sequence/frame keys, and zero duplicates: all 60
  `N_Acc`/`N_Comp`/combined-normal values and all 60
  YUV-PSNR/projection-SSIM/LPIPS values are finite, with exactly 20 expected
  PCQM placeholders and zero unexpected nonfinite values. Separate batch-6
  PCQM is queued behind the `%1` limit while batch-5 PCQM advanced to seven
  real finite emissions. Grad-PU batches 7 and 8 each reached 5/10 outputs;
  IterativePFN batch 13 reached 9/10. PathNet PCQM advanced to 204 finite
  TicTacToe and 14 finite VictoryHeart emissions, with zero NaN/infinite
  values. No repair was required.
- Scheduled continuation 2026-07-22: new PathNet VictoryHeart task
  `20060550_10` emitted four PCQM values, all finite. TicTacToe advanced to 199
  finite emissions; the validated TrumanShow result remains complete at 342.
  Grad-PU batch-5 separate PCQM advanced to five real finite emissions with
  zero NaN/infinite values. Batch-6 k=20 normals reached frame 8/10, and its
  texture/perceptual job emitted 17 rows with every YUV-PSNR,
  projection-SSIM, and LPIPS value finite and only intentional PCQM
  placeholders. Grad-PU batches 7 and 8 each reached 4/10 immutable outputs;
  IterativePFN batch 13 remains healthy at 8/10. No traceback, OOM,
  CUDA-visibility, dependency, or pipeline error was found; no repair was
  required.
- Scheduled continuation 2026-07-22: IterativePFN batch 13 advanced to 8/10
  immutable outputs. Grad-PU batch-6 k=20 normals reached frame 7/10, and its
  texture/perceptual job emitted 11 rows with every YUV-PSNR,
  projection-SSIM, and LPIPS value finite and only intentional PCQM
  placeholders. Batch-5 separate PCQM retains three finite real emissions and
  zero NaN/infinite values while computing the next result. Grad-PU batches 7
  and 8 remain healthy at 4/10 and 3/10 outputs. PathNet PCQM advanced to 195
  finite TicTacToe and 342 finite TrumanShow emissions. TrumanShow task
  `20060550_9` then completed with exit code 0; flash-side validator `20124839`
  proved its committed CSV has exactly 342 rows for 171 paired frames, 342
  unique method/sequence/frame keys, zero duplicates, and all 342 PCQM values
  finite. VictoryHeart task `20060550_10` started automatically as the `%2`
  slot released. No unexpected nonfinite value, traceback, OOM,
  CUDA-visibility, dependency, or pipeline error was found; no repair was
  required.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 separate PCQM advanced
  to three real emissions; explicit parsing confirmed all three finite and
  zero NaN/infinite values. Batch-6 k=20 normals reached frame 6/10, and its
  texture/perceptual job emitted eight rows with all YUV-PSNR,
  projection-SSIM, and LPIPS values finite and only intentional PCQM
  placeholders. Grad-PU batches 7 and 8 reached 4/10 and 3/10 immutable
  outputs. IterativePFN batch 13 remains healthy at 7/10 while processing its
  next frame. PathNet PCQM advanced to 192 finite TicTacToe and 337 finite
  TrumanShow emissions. No unexpected nonfinite value, traceback, OOM,
  CUDA-visibility, dependency, or pipeline error was found; no repair was
  required.
- Scheduled continuation 2026-07-22: Grad-PU batch-6 k=20 normals advanced
  through frame 5/10. Its texture/perceptual job emitted seven current rows;
  all YUV-PSNR, projection-SSIM, and LPIPS values are finite and all seven
  PCQM cells are intentional placeholders. Batch-5 separate PCQM retains one
  finite real emission while computing the next result. Grad-PU batches 7 and
  8 remain healthy at 3/10 outputs each, and IterativePFN batch 13 remains at
  7/10 while processing its next frame. PathNet PCQM advanced to 190 finite
  TicTacToe and 333 finite TrumanShow emissions. No unexpected NaN/infinite
  value, traceback, OOM, CUDA-visibility, dependency, or pipeline error was
  found; no repair was required.
- Scheduled continuation 2026-07-22: IterativePFN batch 13 advanced to 7/10
  immutable outputs. Grad-PU batches 7 and 8 each reached 3/10 outputs.
  Batch-6 texture/perceptual emitted four rows (two baseline/method pairs), all
  with finite YUV-PSNR, projection-SSIM, and LPIPS and only intentional PCQM
  placeholders; its k=20 job remains healthy at frame 4/10 while processing a
  larger frame. Batch-5 separate PCQM retains one finite real emission and is
  actively computing the next result. PathNet PCQM advanced to 188 finite
  TicTacToe and 328 finite TrumanShow emissions. No unexpected NaN/infinite
  value, traceback, OOM, CUDA-visibility, dependency, or pipeline error was
  found; no repair was required.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 separate PCQM job
  `20123244_26` emitted its first real PCQM value and explicit parsing
  confirmed it finite. Batch-6 texture/perceptual emitted two rows (one
  baseline/method pair), both with finite YUV-PSNR, projection-SSIM, and LPIPS
  and only the intentional PCQM placeholders; its k=20 normals advanced
  through frame 4/10. Grad-PU batches 7 and 8 reached 3/10 and 2/10 immutable
  outputs, respectively. IterativePFN batch 13 remains healthy at 6/10.
  PathNet PCQM advanced to 185 finite TicTacToe and 324 finite TrumanShow
  emissions. No unexpected NaN/infinite value, traceback, OOM,
  CUDA-visibility, dependency, or pipeline error was found, so no repair was
  required.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 texture/perceptual job
  `20123243_26` completed with exit code 0. Flash-side validator `20124683`
  proved its immutable CSV has 20 rows, 20 unique method/sequence/frame keys,
  zero duplicates, 60 finite YUV-PSNR/projection-SSIM/LPIPS values, and exactly
  20 expected PCQM placeholders with no unexpected nonfinite value. Its
  separate PCQM job `20123244_26` released normally and is initializing;
  batch-6 texture/perceptual also started. Batch-6 k=20 normals reached frame
  3/10. Grad-PU batches 7 and 8 each reached 2/10 outputs, and IterativePFN
  batch 13 reached 6/10. PathNet PCQM advanced to 181 finite TicTacToe and 315
  finite TrumanShow emissions, with zero NaN/infinite values. No repair was
  required; all dependencies remain intact.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 k=20 normal job
  `20123242_26` completed with exit code 0. Flash-side validator `20124657`
  proved its immutable CSV has 20 rows, 20 unique method/sequence/frame keys,
  zero duplicates, 60 finite `N_Acc`/`N_Comp`/combined-normal values, and zero
  NaN/infinite values. Batch-6 k=20 normals advanced through frame 2/10.
  Batch-5 texture/perceptual reached 13 rows, all with finite YUV-PSNR,
  projection-SSIM, and LPIPS and only intentional PCQM placeholders.
  Grad-PU batches 7 and 8 reached 2/10 and 1/10 immutable outputs;
  IterativePFN batch 13 reached 5/10. PathNet PCQM advanced to 175 finite
  TicTacToe and 301 finite TrumanShow emissions. No unexpected nonfinite value
  or pipeline error was found; all downstream dependencies remain intact.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 and batch-6 k=20 jobs
  advanced through frames 7/10 and 1/10, respectively, with clean logs.
  Batch-5 texture/perceptual job `20123243_26` emitted ten rows covering five
  baseline/method frame pairs; every YUV-PSNR, projection-SSIM, and LPIPS
  value is finite. All ten PCQM cells are intentional `--skip-pcqm`
  placeholders, while separate finite-checked PCQM remains dependency-gated.
  Grad-PU batches 7 and 8 each committed their first immutable PLY.
  IterativePFN batch 13 reached 5/10 outputs. PathNet PCQM advanced to 172
  finite TicTacToe and 294 finite TrumanShow emissions, with zero unexpected
  NaN/infinite values. No traceback, OOM, CUDA-visibility, dependency, or
  metric-pipeline error was found; no repair was required.
- Scheduled continuation 2026-07-22: Grad-PU batch-5 k=20 normals advanced
  through frame 4/10, and batch-6 k=20 normals started with clean logs. Its
  batch-5 texture/perceptual job `20123243_26` also released and emitted seven
  current rows; explicit parsing confirmed all YUV-PSNR, projection-SSIM, and
  LPIPS values finite. Its seven PCQM fields are the intentional `--skip-pcqm`
  placeholders; the separate finite-checked PCQM job `20123244_26` remains
  correctly chained after texture. Grad-PU batch 7 committed its first PLY,
  batch 8 passed startup, and batch 9 remains queued under `%2`. IterativePFN
  batch 13 reached 4/10 outputs. PathNet PCQM advanced to 169 finite
  TicTacToe and 288 finite TrumanShow emissions, with zero unexpected
  NaN/infinite values. No repair was required.
- Scheduled continuation 2026-07-22: Grad-PU BlueVolley task 26 completed with
  exit code 0 and ten immutable batch-5 PLYs; task 28 started automatically as
  the `%2` slot released. Flash-side validation job `20124506` proved the
  committed batch-5 geometry CSV has 20 rows, 20 unique
  method/sequence/frame keys, zero duplicates, all 280 geometry/F-score values
  finite, and exactly 60 expected deferred normal placeholders with no
  unexpected finite or nonfinite value. The reusable CSV validator was
  extended to distinguish required nonfinite placeholder columns from metric
  failures. Task 27 then completed with exit code 0 and ten immutable batch-6
  PLYs; validator `20124510` proved its geometry CSV has the same valid shape:
  20 rows, 20 unique keys, 280 finite geometry/F-score values, 60 expected
  deferred normal placeholders, and no duplicate or unexpected nonfinite
  value. Tasks 28 and 29 started automatically. Matching k=20 tasks 26 and 27
  also released; task 26 reached frame 3/10 and task 27 began with clean logs.
  PathNet PCQM advanced to 162 finite TicTacToe and 271 finite TrumanShow
  emissions, with zero NaN/infinite values. No execution or dependency repair
  was required; all full-objective stages remain intact.
- Scheduled continuation 2026-07-22: Grad-PU BlueVolley task 26 committed all
  ten immutable batch-5 PLYs and remains active while finishing its geometry
  evaluation; it is therefore not yet recorded as a completed batch. Task 27
  remains healthy at 9/10 outputs and is processing its final frame. Their
  corresponding k=20 tasks `20123242_26` and `_27` remain correctly gated.
  IterativePFN batch 13 remains healthy at 3/10 outputs. PathNet PCQM advanced
  to 159 TicTacToe and 266 TrumanShow emissions; every parsed value is finite
  with zero NaN/infinite values. No traceback, OOM, CUDA-visibility,
  dependency failure, or objective-pipeline error was found, so no repair or
  resubmission was required.
- Scheduled continuation 2026-07-22: IterativePFN batch 13 advanced to 3/10
  immutable BlueSpeech outputs. Grad-PU BlueVolley tasks 26 and 27 remain
  healthy at 9/10 outputs while processing their final frames. PathNet PCQM
  advanced to 157 TicTacToe and 261 TrumanShow emissions; explicit parsing
  confirmed every value finite and zero NaN/infinite values. No traceback,
  OOM, CUDA-visibility, dependency failure, or objective-metric pipeline error
  was found. All immutable k=20, texture/perceptual, PCQM, paper-table, and
  continuation dependencies remain intact, so no repair or duplicate
  resubmission was needed.
- Scheduled continuation 2026-07-22: Grad-PU BlueVolley tasks 26 and 27 each
  reached 9/10 immutable outputs and remain healthy on the A40 CUDA node.
  IterativePFN batch 13 remains healthy at 2/10 outputs. PathNet PCQM advanced
  to 155 TicTacToe and 257 TrumanShow emissions; explicit parsing confirmed
  every value finite and zero NaN/infinite values. No traceback, OOM,
  CUDA-visibility, dependency failure, or objective-metric pipeline error was
  found. All method-specific immutable k=20, texture/perceptual, PCQM,
  paper-table, and continuation dependencies remain intact; no resubmission or
  repair was required.
- Scheduled continuation 2026-07-22: no new benchmark or dependency failure
  was found. Grad-PU BlueVolley task 26 advanced to 9/10 immutable outputs;
  task 27 remains healthy at 8/10 while processing its next frame.
  IterativePFN batch 13 remains healthy at 2/10 outputs. PathNet PCQM advanced
  to 153 TicTacToe and 252 TrumanShow emissions; explicit parsing confirmed
  every current value finite and zero NaN/infinite values. The previously
  validated IterativePFN batch-11 and batch-12 k=20 CSVs remain complete and
  all downstream immutable normal, texture/perceptual, PCQM, paper-table, and
  continuation dependencies remain intact. No resubmission was required.
- Scheduled continuation 2026-07-22: IterativePFN batch-12 k=20 normal job
  `20122039_12` completed with exit code 0. Flash-side validator `20124258`
  proved its immutable CSV has 20 rows, 20 unique method/sequence/frame keys,
  zero duplicates, 60 finite `N_Acc`/`N_Comp`/combined-normal values, and zero
  NaN/infinite values. Initial dependency-gated validator `20124253` briefly
  appeared stuck behind Slurm array-parent bookkeeping; cancellation and an
  ungated read-only retry were requested after the metric task was already
  complete. The original validator released concurrently and also completed,
  so both read-only checks succeeded and no benchmark output was modified.
  IterativePFN batch 13 reached 2/10 immutable outputs. PathNet PCQM advanced
  to 149 finite TicTacToe and 244 finite TrumanShow emissions. Grad-PU tasks
  26-27 remain healthy at 8/10 outputs each; all downstream dependencies are
  intact.
- Scheduled continuation 2026-07-22: all active benchmark stages remain
  healthy. IterativePFN batch 13 committed its first immutable BlueSpeech PLY,
  and batch-12 k=20 normals advanced through frame 6/10 with empty stderr.
  Grad-PU BlueVolley tasks 26 and 27 each reached 8/10 immutable outputs.
  PathNet PCQM advanced to 147 TicTacToe and 239 TrumanShow emissions;
  explicit parsing found every value finite and zero NaN/infinite values.
  No traceback, OOM, CUDA-visibility, dependency failure, or metric-pipeline
  error was found. The visible stderr contains only the already recorded LUMI
  stack notice and trusted-checkpoint `FutureWarning`; all immutable k=20,
  texture/perceptual, PCQM, table, and continuation dependencies remain
  intact.
- Scheduled continuation 2026-07-22: IterativePFN batch-11 k=20 normal job
  `20122039_11` completed with exit code 0. Read-only validation job `20124122`
  proved its committed immutable CSV has 20 rows, 20 unique
  method/sequence/frame keys, zero duplicates, 60 finite
  `N_Acc`/`N_Comp`/combined-normal values, and zero NaN/infinite values.
  Validator attempt `20124116` first failed before reading data because the
  default compute-node Python did not support `from __future__ import
  annotations`; the unnecessary import was removed and the validator was
  resubmitted successfully. Batch-12 k=20 normals are running and reached
  frame 2/10. IterativePFN batch 13 and Grad-PU tasks 26-27 remain healthy.
  PathNet PCQM reached 139 finite TicTacToe and 222 finite TrumanShow
  emissions, with zero NaN/infinite values. All downstream immutable
  dependencies remain intact.
- Scheduled continuation 2026-07-22: IterativePFN batch 12 completed all ten
  immutable BlueSpeech outputs with Slurm exit code 0; its k=20 task
  `20122039_12` remains correctly dependency-gated during parent cleanup.
  Batch-11 k=20 evaluation advanced through frame 5/10 with clean stderr.
  Batch 13 is running on A40 `nid000016`. Grad-PU BlueVolley tasks 26 and 27
  reached 7/10 and 6/10 outputs, respectively. PathNet PCQM advanced to 137
  TicTacToe and 217 TrumanShow emissions; explicit parsing found all values
  finite and zero NaN/infinite values. No traceback, OOM, CUDA-visibility,
  dependency failure, or objective-pipeline error was found, so the existing
  immutable continuation and full-metric dependencies were preserved without
  resubmission.
- Scheduled continuation 2026-07-22: the completed IterativePFN batch-11
  inference dependency released correctly. Its normal task `20122039_11`
  started the required k=20 evaluation at BlueSpeech frame 0110 with clean
  stderr, while batch 13 began CUDA inference on A40 `nid000016`. Batch 12
  remains healthy at 9/10 immutable outputs. Grad-PU BlueVolley tasks 26 and
  27 remain healthy at 6/10 outputs each. PathNet PCQM advanced to 134
  TicTacToe and 212 TrumanShow emissions; all parsed values are finite with
  zero NaN/infinite values. No traceback, OOM, CUDA-visibility, dependency
  failure, or objective-pipeline error was found. Downstream method-specific
  immutable metric and continuation jobs remain correctly gated.
- Scheduled continuation 2026-07-22: IterativePFN batch 11 completed all ten
  immutable BlueSpeech outputs with Slurm exit code 0; its k=20 normal job
  `20122039_11` remains correctly gated until array-parent cleanup finishes.
  Batch 12 reached 9/10 outputs and batch 13 remains queued under the `%2`
  GPU limit. Grad-PU BlueVolley tasks 26 and 27 each reached 6/10 outputs on
  the A40 CUDA partition, with tasks 28-30 queued. PathNet PCQM advanced to
  132 TicTacToe and 208 TrumanShow emissions; explicit parsing found every
  value finite and no NaN/infinite result. No traceback, OOM, CUDA-visibility,
  dependency failure, or objective-pipeline error was found, so no repair or
  duplicate resubmission was needed. All immutable k=20, texture/perceptual,
  PCQM, paper-table, and continuation dependencies remain intact.
- Scheduled continuation 2026-07-22: non-paper benchmark execution remains
  healthy. IterativePFN batches 11 and 12 each reached 9/10 immutable PLYs.
  Grad-PU BlueVolley tasks 26 and 27 reached 6/10 and 5/10 PLYs,
  respectively, on the LUMI A40 CUDA partition. Explicit parsing confirmed
  all 127 current TicTacToe and all 198 current TrumanShow PCQM emissions are
  finite, with zero NaN/infinite values. No traceback, OOM, CUDA-visibility,
  dependency, or objective-metric pipeline error was found. All downstream
  k=20 normal, texture/perceptual, PCQM, paper-table, and continuation jobs
  remain correctly dependency-gated. The login node does not mount `/flash`,
  so live immutable-output progress was verified from each compute-job log.
- Scheduled continuation 2026-07-22: all monitored jobs remain healthy and no
  repair was required. IterativePFN batches 11 and 12 each reached 8/10
  immutable PLYs. Grad-PU BlueVolley tasks 26 and 27 each reached 5/10 PLYs.
  Explicit parsing confirmed all 123 current TicTacToe and all 188 current
  TrumanShow PCQM values finite, with zero NaN/infinite values. No traceback,
  OOM, CUDA-visibility, dependency, or objective-metric pipeline error was
  found. All queued k=20 and full-objective stages remain intact; flash usage
  remains about 382 GiB of 2 TiB.
- Scheduled continuation 2026-07-22: no new benchmark failure was found.
  IterativePFN batches 11 and 12 retain 8/10 and 7/10 immutable PLYs and remain
  active. Grad-PU BlueVolley task 26 advanced to 5/10 PLYs, while task 27
  retains 4/10 and is processing its fifth frame. Explicit PathNet parsing
  confirmed all 120 current TicTacToe and all 182 current TrumanShow PCQM
  values finite, with zero NaN/infinite values. No traceback, OOM,
  CUDA-visibility, dependency, or metric-pipeline error appeared. All k=20 and
  full-objective stages remain correctly dependency-gated; flash usage is
  about 382 GiB of 2 TiB.
- Scheduled continuation 2026-07-22: no new execution, CUDA, memory,
  dependency, or metric error was found. IterativePFN batches 11 and 12
  advanced to 8/10 and 7/10 immutable PLYs. Grad-PU BlueVolley tasks 26 and 27
  retain 4/10 PLYs each while processing their next frames. Explicit PathNet
  parsing confirmed all 118 current TicTacToe and all 177 current TrumanShow
  PCQM values finite, with zero NaN/infinite values. All k=20,
  texture/perceptual, PCQM, paper-table, and continuation dependencies remain
  intact; flash usage remains about 382 GiB of 2 TiB.
- Scheduled continuation 2026-07-22: no new error or unexpected nonfinite
  value was found. IterativePFN batches 11 and 12 each reached 7/10 immutable
  PLYs. Grad-PU BlueVolley tasks 26 and 27 each reached 4/10 PLYs. Explicit
  PathNet parsing confirmed all 115 current TicTacToe and all 171 current
  TrumanShow PCQM values finite, with zero NaN/infinite values. All downstream
  k=20, texture/perceptual, PCQM, table, and continuation dependencies remain
  healthy. Flash usage is about 382 GiB of 2 TiB.
- Scheduled continuation 2026-07-22: all active jobs and dependencies remain
  healthy. IterativePFN batches 11 and 12 advanced to 7/10 and 6/10 immutable
  PLYs. Grad-PU BlueVolley task 26 advanced to 4/10 PLYs and task 27 retains
  3/10 while processing its fourth frame. Explicit parsing confirmed all 113
  current TicTacToe and all 166 current TrumanShow PCQM values finite, with
  zero NaN/infinite values. No traceback, OOM, CUDA-visibility, dependency, or
  metric-pipeline error was found. Flash usage remains about 381 GiB of 2 TiB.
- Scheduled continuation 2026-07-22: no new execution, dependency, CUDA, OOM,
  or metric-pipeline failure was found. IterativePFN batches 11 and 12 retain
  6/10 immutable PLYs each and remain active. Grad-PU BlueVolley tasks 26 and
  27 retain 3/10 PLYs each while processing their next frames. Explicit log
  parsing confirmed all 110 current TicTacToe and all 160 current TrumanShow
  PCQM values finite, with zero NaN/infinite values. Flash usage remains about
  381 GiB of 2 TiB. All normal, texture/perceptual, PCQM, paper-table, and
  continuation dependencies remain intact.
- Scheduled continuation 2026-07-22: all active work remains healthy and no
  repair was required. IterativePFN batches 11 and 12 each reached 6/10
  immutable PLYs. Grad-PU BlueVolley tasks 26 and 27 each reached 3/10 PLYs.
  Explicit parsing confirmed all 107 current TicTacToe and all 154 current
  TrumanShow PCQM emissions are finite, with zero NaN/infinite values. No OOM,
  CUDA-visibility, traceback, dependency, or metric-pipeline error was found.
  All k=20, texture/perceptual, PCQM, table, and continuation jobs remain
  correctly dependency-gated.
- Scheduled continuation 2026-07-22: all monitored jobs remain healthy.
  IterativePFN batches 11 and 12 advanced to 6/10 and 5/10 immutable PLYs.
  Grad-PU BlueVolley tasks 26 and 27 retain 2/10 PLYs each and are actively
  processing their third frames. Explicit PathNet parsing found all 104
  TicTacToe and all 149 TrumanShow PCQM values currently emitted to be finite,
  with zero NaN/infinite values. No traceback, OOM, CUDA-visibility,
  dependency, or pipeline error appeared. Flash usage remains about 381 GiB
  of 2 TiB, and all k=20, full-metric, table, and continuation dependencies are
  intact.
- Scheduled continuation 2026-07-22: no new failure or unexpected nonfinite
  value was found. Grad-PU BlueVolley batch-4 k=20 normal job `20120474_25`
  completed with exit code 0 and was validated at 20 rows, 10 frames, 20
  unique method/sequence/frame keys, all 60 `N_Acc`/`N_Comp`/combined normal
  values finite, zero duplicates, and zero NaN/infinite values. New-wave tasks 26 and 27
  each reached 2/10 immutable PLYs. IterativePFN batches 11 and 12 each reached
  5/10 PLYs. Explicit PathNet log parsing found all 101 emitted TicTacToe PCQM
  values and all 143 emitted TrumanShow PCQM values finite, with zero NaN or
  infinite value. All normal, texture/perceptual, PCQM, paper-table, and
  continuation dependencies remain intact.
- Scheduled continuation 2026-07-22: no new job or pipeline failure was found.
  Grad-PU BlueVolley batch-4 k=20 normals reached frame 5/10. New-wave tasks
  26 and 27 each retain one nonempty immutable PLY (136,466,730 and
  127,464,613 bytes) and are processing their second frames. IterativePFN
  batches 11 and 12 remain healthy at 4/10 PLYs each. PathNet PCQM logs were
  explicitly parsed: all 95 TicTacToe and all 130 TrumanShow values emitted at
  this check are finite, with zero NaN/infinite values. All downstream normal,
  texture/perceptual, PCQM, table, and continuation dependencies remain intact.
- Scheduled continuation 2026-07-22: no new execution error, dependency
  failure, or unexpected NaN/infinite value was found. Grad-PU BlueVolley
  batch-4 k=20 normals reached frame 4/10. New-wave tasks `20123239_26` and
  `_27` each committed their first immutable PLY, while tasks 28-30 remain
  correctly queued under `%2`; all matching k=20, texture/perceptual, PCQM,
  and next-wave dependencies remain intact. IterativePFN batches 11 and 12
  each reached 4/10 immutable PLYs. PathNet PCQM emitted at least 92 finite
  values for TicTacToe and 124 for TrumanShow. The only warning match remains
  the already recorded trusted-checkpoint `FutureWarning`.
- Scheduled continuation 2026-07-22: no new execution failure or unexpected
  nonfinite value was found. Grad-PU BlueVolley batch-4 k=20 normals reached
  frame 3/10. New-wave tasks `20123239_26` and `_27` remain healthy on A40
  `nid000018`; task 27 wrote its first immutable PLY and task 26 completed all
  chunks for its first frame. Their matching
  normal, texture/perceptual, and PCQM arrays retain their corresponding-task
  dependencies. IterativePFN batches 11 and 12 advanced to 4/10 and 3/10
  immutable PLYs. PathNet PCQM emitted at least 90 finite values for TicTacToe
  and 120 for TrumanShow. Flash usage is about 381 GiB of 2 TiB. The only
  matched stderr text is the already recorded
  PyTorch trusted-checkpoint `FutureWarning`, not a benchmark error.
- Scheduled continuation 2026-07-22: Grad-PU batch-4 k=20 normal task
  `20120474_25` released normally and began frame 1/10. Patched controller
  `20122910` completed successfully and submitted Grad-PU BlueVolley global
  inference array `20123239` for tasks 26-30, matching k=20 array `20123242`,
  texture/perceptual array `20123243`, and chained finite-checked PCQM array
  `20123244`. Tasks 26 and 27 passed CUDA/checkpoint startup on A40
  `nid000018` and are actively processing their first frames; stderr contains
  only the known LUMI-stack and trusted-checkpoint warnings. Next controller
  `20123261` is dependency-gated on this wave and will continue BlueVolley
  global tasks 31-35 through the same four-stage pipeline. IterativePFN batches
  11 and 12 each reached 3/10 immutable PLYs. PathNet PCQM reached at least 87
  finite values for TicTacToe and 113 for TrumanShow. No new traceback, OOM,
  fatal, dependency-failure, or nonfinite signature was found.
- Scheduled continuation 2026-07-22: no new traceback, OOM, fatal,
  dependency-failure, or nonfinite signature was found. Grad-PU BlueVolley
  batch 4 completed with exit code 0 and ten immutable PLYs. Its committed
  geometry CSV was validated at 20 rows, 10 frames, 20 unique
  method/sequence/frame keys, 280 finite geometry/F-score values, exactly 60
  expected deferred normal placeholders, zero duplicates, and zero unexpected
  nonfinite values. Slurm is finishing node cleanup; therefore
  its k=20 normal task `20120474_25`, patched next-wave controller `20122910`,
  and full-metric backfill remain correctly success-gated. IterativePFN
  batches 11 and 12 reached 3/10 and 2/10 immutable PLYs, with batch 13 queued under
  the `%2` GPU limit. PathNet PCQM advanced to at least 82 finite values for
  TicTacToe and 102 for TrumanShow; tasks 10-11 and the paper-table job remain
  dependency-gated. The complete batch-9 and batch-10 IterativePFN k=20 CSVs
  remain independently validated with zero NaN/infinite values.
- Scheduled continuation 2026-07-22: IterativePFN retry5 batch-9 k=20 normal
  job `20122039_9` completed with exit code 0 and was validated at 20 rows,
  10 frames, 20 unique method/sequence/frame keys, 60 finite
  `N_Acc`/`N_Comp`/combined normal values, zero duplicates, and zero
  NaN/infinite values. Batches 11 and 12 each produced their first immutable
  PLY without an error. Grad-PU BlueVolley batch 4 completed nine PLYs and is
  processing its tenth frame; matching k=20 normals and patched continuation
  controller `20122910` remain correctly dependency-gated. PathNet PCQM
  advanced to at least 76 finite values for TicTacToe and 90 for TrumanShow.
  IterativePFN batch 11 reached 2/10 PLYs while batch 12 remains at 1/10. No new
  traceback, OOM, fatal, dependency-failure, or nonfinite signature was found.
  Because the active inference waves were spooled before full-metric controller
  automation was added, their missing stages were explicitly backfilled with
  immutable run IDs. IterativePFN batches 9-13 use texture/perceptual array
  `20123096` and chained PCQM array `20123097`; Grad-PU BlueVolley batches 0-4
  (global tasks 21-25) use `20123098` and `20123099`. Both texture arrays are
  gated on successful completion of PathNet PCQM and their complete inference
  parent arrays, and each PCQM array is corresponding-task gated on texture.
  `scontrol` confirmed all four compound dependencies are intact.
- Scheduled continuation 2026-07-22: no new benchmark failure was detected.
  IterativePFN retry5 batch-10 k=20 normal job `20122039_10` completed with
  exit code 0 and was independently validated at 20 rows, 10 frames, 20
  unique method/sequence/frame keys, 60 finite `N_Acc`/`N_Comp`/combined
  normal values, zero duplicates, and zero NaN/infinite values. Batch-9 k=20
  normals reached frame 4/10 and are running normally. IterativePFN batch 11 produced its first
  immutable PLY and batch 12 passed CUDA startup; batch 13 remains queued under
  the `%2` GPU limit. Grad-PU BlueVolley batch 4 advanced to 9/10 PLYs.
  PathNet PCQM advanced to 71 finite values for TicTacToe and 78 for
  TrumanShow. Patched full-metric continuation controllers `20122910` and
  `20122911` remain correctly dependency-gated.
- Scheduled continuation 2026-07-22: IterativePFN retry5 batch 9 completed
  with exit code 0 and ten immutable PLYs. Its committed geometry CSV passed
  the same validation as batch 10: 20 rows, 10 frames, 20 unique keys, 280
  finite geometry/F-score values, 60 expected deferred normal placeholders,
  no duplicates, and no unexpected nonfinite values. Batch-10 k=20 normal
  task `20122039_10` released and reached frame 6/10; batch-9 normal task
  `20122039_9` also released normally after Slurm finished parent cleanup.
  IterativePFN batches 11 and 12 started on A40 `nid000018`; Grad-PU
  BlueVolley batch 4 reached 8/10
  PLYs. PathNet PCQM reached 65 finite values for TicTacToe and 64 for
  TrumanShow. No new traceback, OOM, fatal, dependency-failure, or nonfinite
  signature was found.
  Both method-wave controllers were extended and passed `bash -n`: every
  future wave now submits immutable inference, matching k=20 normals,
  texture/YUV-PSNR/projection-SSIM/LPIPS, and separate finite-checked PCQM.
  Texture is `aftercorr`-gated on inference and PCQM is `aftercorr`-gated on
  texture, with one-task metric concurrency. The pre-patch pending Grad-PU
  controller `20120475` was cancelled before it ran and replaced by patched
  controller `20122910` for global batches 26-30. Patched IterativePFN
  controller `20122911` is dependency-gated on the active wave and will submit
  batches 14-16 while continuing to exclude `nid000017`.
- Scheduled continuation 2026-07-22: IterativePFN retry5 batch 10 completed
  with exit code 0 and ten immutable PLYs. Its geometry CSV was validated at
  20 rows, 10 frames, 20 unique method/sequence/frame keys, 280 finite
  geometry/F-score values, exactly 60 expected deferred normal placeholders,
  zero duplicates, and zero unexpected nonfinite values. Matching k=20 normal
  task `20122039_10` remains dependency-gated. Retry5 batch 9 reached 9/10
  PLYs and task 11 started automatically on a second A40 allocation. Grad-PU
  BlueVolley batch 4 reached 7/10 PLYs. PathNet PCQM reached at least 63 finite
  values for TicTacToe and 60 for TrumanShow, with no new error signature.
  Full missing objective stages were queued immutably behind PathNet PCQM for
  17 already completed method batches. IterativePFN texture/PCQM pairs are
  `20122707`/`20122708` (base batches 0,1,3),
  `20122709`/`20122710` (retry1 batch 2),
  `20122711`/`20122712` (retry5 batches 5,6,8), and
  `20122713`/`20122714` (retry6 batch 7). Grad-PU pairs are
  `20122715`/`20122716` (retry1 batches 1-7,9-11) and
  `20122717`/`20122817` (retry2 batches 0,8). Each texture array uses a new
  Slurm-job run name and intentional PCQM placeholders; its paired separate
  PCQM array is `afterok`-gated and rejects nonfinite values. Explicit Slurm
  inspection confirmed all dependencies are intact, so these arrays cannot
  read incomplete inputs or overwrite earlier metric runs.
- Scheduled check 2026-07-22: non-paper benchmark execution remains active.
  PathNet PCQM tasks 8 and 9 are running on TicTacToe and TrumanShow; they have
  emitted 56 and 45 finite PCQM values, respectively, with tasks 10-11 queued
  behind the `%2` limit. Grad-PU BlueSpeech batch-15 PCQM job `20120335_15`
  completed with exit code 0 and was validated at 20 rows, 10 frames, 20
  unique method/sequence/frame keys, zero duplicates, and zero NaN/infinite
  PCQM values. Grad-PU BlueVolley immutable batch 4 (`20120468_25`) is running
  on an A40 and has written 7/10 PLYs. IterativePFN immutable batches 9 and 10
  (`20122038_9`, `20122038_10`) are running on A40 GPUs and have written 8/10
  and 9/10 PLYs; batches 11-13 and matching k=20 normals remain correctly
  dependency/array-limit queued. No new traceback, OOM, fatal, nonfinite, or
  dependency-failure signature was found. Flash usage is 380 GiB of 2 TiB.
- LUMI audit on 2026-07-22 verified exactly 2,152 PathNet PLY outputs and
  2,152 finite geometry rows. The active normal (`k=20`) array has produced
  all expected 4,304 finite rows in 50 validated immutable batch CSVs, with
  zero nonfinite cells and zero duplicate method/sequence/frame keys. This is
  exactly 2 methods x 2,152 paired frames across all 12 sequences. The normal
  array completed successfully. Texture/perceptual and PCQM arrays remain active; they write their
  final CSVs only after each sequence finishes. The paper-table job is
  dependency-gated. BlueSpeech texture job `20113401` completed with exactly
  336 per-frame rows (168 baseline + 168 PathNet), 2,016 per-view rows, and
  finite YUV-PSNR, projection-SSIM, and LPIPS values. Its 336 PCQM cells are
  intentionally NaN because that array uses `--skip-pcqm`; the separate PCQM
  array is producing those values. BlueVolley texture job `20113402` also
  completed with exactly 342 per-frame rows (171 + 171), 2,052 per-view rows,
  and finite YUV-PSNR, projection-SSIM, and LPIPS values. Its 342 PCQM cells
  are the same intentional `--skip-pcqm` placeholders. FitFluencer texture job
  `20114808` completed with exactly 402 per-frame rows (201 + 201), 2,412
  per-view rows, and finite YUV-PSNR, projection-SSIM, and LPIPS values. Its
  402 PCQM cells are intentional `--skip-pcqm` placeholders; task 4 started
  automatically on GoodVision. BouncingBlue texture job `20114699` completed
  with exactly 314 per-frame rows (157 + 157), 1,884 per-view rows, and finite
  YUV-PSNR, projection-SSIM, and LPIPS values. Its 314 PCQM cells are the
  expected `--skip-pcqm` placeholders; task 5 started automatically on
  Mannequin. Mannequin texture job `20116057` completed with exactly 376
  per-frame rows (188 + 188), 2,256 per-view rows, and finite YUV-PSNR,
  projection-SSIM, and LPIPS values. Its 376 PCQM cells are intentional
  `--skip-pcqm` placeholders; task 6 started automatically on OrangeKettlebell.
  The separate BlueSpeech
  PCQM job `20113403` completed with exit code 0 and exactly 336 rows (168
  baseline + 168 PathNet), covering 168 unique frames with no missing or
  nonfinite PCQM values. BlueVolley PCQM job `20113404` also completed with
  exit code 0 and exactly 342 rows (171 baseline + 171 PathNet), covering 171
  unique frames with no missing or nonfinite cells. BouncingBlue PCQM job
  `20115588` completed with exit code 0 and was validated at exactly 314 rows
  (157 baseline + 157 PathNet), 157 unique frames, 314 unique method/frame
  keys, and zero nonfinite PCQM values. GoodVision PCQM task 4 started
  automatically and completed with exit code 0. Its committed CSV was validated
  at exactly 334 rows (167 baseline + 167 PathNet), 167 frames, 334 unique
  method/sequence/frame keys, and zero nonfinite values. FitFluencer PCQM
  completed with exit code 0 and was validated at exactly 402 rows (201
  baseline + 201 PathNet), 201 unique frames, 402 unique method/sequence/frame
  keys, and zero nonfinite PCQM values. Task 5 started automatically on
  Mannequin and completed with exit code 0. Its committed CSV was validated at
  exactly 376 rows (188 baseline + 188 PathNet), 188 frames, 376 unique
  method/sequence/frame keys, and zero nonfinite PCQM values. Task 7 then
  started automatically on PinkNoir. GoodVision texture
  job `20115899` completed with exit code 0 and was validated at exactly 334
  per-frame rows (167 baseline + 167 PathNet), 2,004 per-view rows, 167 unique
  frames, and no duplicate method/frame keys. Point counts, YUV-PSNR,
  projection-SSIM, and LPIPS are all finite. Its 334 PCQM cells are intentional
  `--skip-pcqm` placeholders, not computation failures. PinkNoir texture task
  7 completed with exit code 0 and was validated at exactly 402 per-frame rows
  (201 baseline + 201 PathNet), 2,412 per-view rows, 201 unique frames, no
  duplicate keys, and finite point-count, YUV-PSNR, projection-SSIM, and LPIPS
  values. Its 402 PCQM cells are intentional `--skip-pcqm` placeholders. Task
  8 completed successfully on TicTacToe and was validated at exactly 330
  per-frame rows (165 baseline + 165 PathNet), 1,980 per-view rows, 165 unique
  frames, no duplicate method/frame/view keys, 2,310 finite point-count/YUV/
  projection summary values, and 3,960 finite per-view SSIM/LPIPS values. Its
  330 PCQM cells are intentional `--skip-pcqm` placeholders. Task 10 started
  automatically on VictoryHeart. OrangeKettlebell
  texture task 6 completed with exit code 0 and was validated at exactly 340
  per-frame rows (170 baseline + 170 PathNet), 2,040 per-view rows, 170 frames,
  and no duplicate method/frame/view keys. Point counts, YUV-PSNR,
  projection-SSIM, and LPIPS are all finite; its 340 PCQM cells are intentional
  `--skip-pcqm` placeholders. Task 9 started automatically on TrumanShow.
  VictoryHeart texture task 10 completed with exit code 0 and was validated at
  exactly 394 per-frame rows (197 baseline + 197 PathNet), 2,364 per-view rows,
  197 frames, no duplicate method/frame/view keys, and no unexpected nonfinite
  point-count/YUV-PSNR/projection-SSIM/LPIPS values. Its 394 PCQM cells are the
  intentional `--skip-pcqm` placeholders.
  TrumanShow texture task 9 completed with exit code 0 and was validated at
  342 per-frame rows, 171 frames, 342 unique method/frame keys, 2,052 per-view
  rows, zero duplicates, finite point-count/YUV/projection-SSIM/LPIPS values,
  and exactly 342 intentional `--skip-pcqm` placeholders. VirtualLife texture
  task 11 completed with exit code 0 and was validated at exactly 392 per-frame
  rows (196 baseline + 196 PathNet), 2,352 per-view rows, 196 frames, no
  duplicate method/frame/view keys, and no unexpected nonfinite point-count,
  YUV-PSNR, projection-SSIM, or LPIPS values. Its 392 PCQM cells are intentional
  `--skip-pcqm` placeholders. The full PathNet texture array is now complete.
  PCQM task 6 completed with exit code 0
  and was validated at exactly 340 rows, 170 frames, 340 unique
  method/sequence/frame keys, 340 finite PCQM values, zero duplicates, and zero
  nonfinite values. PinkNoir PCQM task 7 completed with exit code 0 and was
  validated at exactly 402 rows, 201 frames, 402 unique method/sequence/frame
  keys, 402 finite PCQM values, zero duplicates, and zero nonfinite values.
  TicTacToe PCQM task 8 completed its first twenty-six frame pairs plus the next
  baseline value; all fifty-three emitted values are finite. TrumanShow PCQM
  task 9 started automatically under the `%2` array limit and completed its
  first eighteen frame pairs plus the next baseline value; all thirty-seven
  emitted values are finite.
  Flash storage is healthy at approximately 377 GiB
  used of 2 TiB. The live
  cache currently contains
  50 normal CSVs; all are included in the full validation above.
- IterativePFN LUMI smoke `20114142` completed after installing the missing
  ABI-matched `torch-cluster 1.6.3+pt24cu121` dependency. Geometry values are
  finite; only the deliberately unset normal fields are NaN and will be
  replaced by the separate `k=20` normal pipeline. Immutable full-data batches
  `20114232` were released after the current Grad-PU wave completed. Tasks 0
  and 1 launched with the repaired CUDA environment; tasks 2 and 3 are held
  only by the intended `%2` array concurrency limit. Both active tasks loaded
  the legacy Lightning checkpoint successfully; its automatic in-memory
  checkpoint upgrade and `torch.load` messages are warnings, not errors. No
  startup exception or CUDA-visibility failure is present in their logs. Their
  allocated `srun` steps remain active on one A40 each. Both produced their
  first immutable PLYs: task 0 wrote frame `0000` (528,129 input to 528,109
  output points), and task 1 wrote frame `0010` (571,939 input/output points).
  Tasks 0 and 1 reached nine and eight immutable PLYs respectively with no
  runtime exception; their
  second outputs preserve separate batch-specific paths. This
  confirms the repaired `torch_cluster`/CUDA stack in full-data execution.
  Task 0 subsequently completed and was validated at 10 PLYs, 20 rows, 10
  frames, 20 unique method/frame keys, and zero nonfinite geometry cells. Its
  60 nonfinite normal cells are intentional placeholders for the shared
  `k=20` evaluation. Task 2 (`20118404`) then failed before inference on
  `nid000017`: Slurm allocated an A40 but CUDA was unavailable inside the job.
  It wrote zero PLYs. The failed empty result directory is preserved, and
  immutable retry `20118413` uses method prefix
  `iterativepfn_full_20260722_retry1`; it successfully passed CUDA startup and
  loaded the checkpoint on `nid000018`. Both the retry and task 3 retain live
  A40 allocations and active inference steps with no new traceback. Task 3
  produced all ten PLYs (`0030-0039`) and completed with exit code 0. Its CSV was
  validated at 20 rows, 10 frames, 20 unique method/frame keys, zero nonfinite
  geometry cells, and exactly 60 expected deferred normal placeholders. Normal
  array `20119260_3` then started for this batch with `k=20` and the corrected
  eight-core request (16 allocated CPUs versus 32 for the earlier 16-core
  request due LUMI's allocation granularity). It completed with exit code 0;
  its `k=20` CSV was validated at 20 rows, 10 frames, 20 unique method/frame
  keys, and zero nonfinite `N_Acc`, `N_Comp`, or combined normal values. Retry
  task 2 completed with exit code 0 and was validated at ten
  PLYs, 20 rows, 10 frames, 20 unique method/frame keys, zero nonfinite geometry
  cells, and exactly 60 expected deferred normal placeholders. Its separate
  `k=20` normal job `20119280_2` completed with exit code 0 under the immutable
  retry method prefix. Its CSV was validated at 20 rows, 10 frames, 20 unique
  method/frame keys, and zero nonfinite `N_Acc`, `N_Comp`, or combined normal
  values. Task 1
  completed and was validated
  at 10 PLYs, 20 metric rows, 10
  frames, 20 unique method/frame keys, and zero nonfinite geometry cells. Its
  60 nonfinite normal cells are the expected deferred `k=20` placeholders.
  Obsolete controller `20118388`, whose
  `afterok` dependency could no longer succeed, was cancelled. Replacement
  controller `20118414` completed after task 1, task 3, and retry `20118413`, and
  submitted immutable array `20119277` for batches `4-8`. Tasks 4 and 5 started
  on A40 GPUs with CUDA commands active and no startup traceback; batch 4
  completed with exit code 0. Its committed output was validated at 10 PLYs,
  20 metric rows, 10 frames, 20 unique method/frame keys, zero nonfinite values
  across 280 geometry cells, and exactly 60 expected deferred normal
  placeholders.
  Task 5 failed after six PLYs with a CUDA OOM while requesting 8.44 GiB;
  PyTorch reported 10.87 GiB reserved but unallocated. The partial immutable
  batch remains preserved. First retry `20119802_5` attempted patch size 500
  plus `expandable_segments`, but container `--cleanenv` stripped those ordinary
  host variables and the command still showed patch size 1000; it was cancelled
  before writing any PLY, and its empty retry2 directory remains preserved.
  `run_full_dataset_method.py` now supports
  `PCE_ITERATIVEPFN_PATCH_SIZE`. Corrected immutable retry `20119810_5` passed
  the override and allocator through `SINGULARITYENV_`; its logged command
  confirmed patch size 500, but it failed before writing a PLY with another
  OOM while requesting 13.41 GiB. Inspection showed that reducing patch size
  increases the number of patches and therefore enlarged the original dense
  `num_patches x N` stitching matrix. The implementation now computes the same
  winning patch for each point by incrementally retaining the minimum normalized
  distance, avoiding that quadratic dense allocation while preserving patch
  size 1000 and the original `argmax(exp(-distance))` decision. Immutable retry4
  `20119829_5` then exposed uncovered points: the dense implementation silently
  defaults these to patch 0 and subsequently omits them, whereas the first sparse
  version raised an explicit error. The sparse version now preserves that exact
  fallback behavior. Original batch 6 produced one PLY and then failed at the
  separate network patch-minibatch stage while requesting 3.72 GiB. Following
  the official error guidance, `run_full_dataset_method.py` now also supports
  `PCE_ITERATIVEPFN_SEED_K_ALPHA`; increasing it from 10 to 20 halves the patch
  minibatch without changing patch size or stitching selection. The original
  tasks 7-8 were cancelled before further default-configuration failures.
  Immutable retry5 array `20119952` covers batches 5-8 with patch size 1000,
  `seed_k_alpha=20`, the sparse-equivalent stitching fix, and `%2` concurrency;
  logged commands for running tasks 5 and 6 confirm these settings. Matching
  normal array `20119953` uses `aftercorr` so each k=20 task depends only on its
  corresponding successful inference task. Obsolete combined normal
  dependency `20119595`, retry2 normal job `20119804`, and retry3 normal job
  `20119811` and retry4 normal job `20119830` were cancelled. Replacement batch-4
  normal job `20119803_4` released after successful batch 4 and completed with
  exit code 0. Its immutable k=20 CSV was validated at 20 rows, 10 frames,
  20 unique method/frame keys, 60 normal values, and zero nonfinite values.
  Retry5 tasks 5 and 6 completed all ten valid PLYs without an OOM. Each
  geometry result was validated at 20 rows, 10 frames, 20 unique
  method/sequence/frame keys, 300 finite numeric values, and exactly 60
  expected deferred `normals`/`N_Acc`/`N_Comp` placeholders. Their matching
  k=20 normal tasks `20119953_5` and `20119953_6` released and are running,
  confirming that the combined sparse-stitching and smaller-minibatch repair
  crossed the failure point seen by the earlier attempts.
  Normal task 5 completed with exit code 0 and was validated at 20 rows, 10
  frames, 20 unique method/sequence/frame keys, 60 finite normal metric values,
  zero duplicate keys, and zero nonfinite values. Normal task 6 completed with
  exit code 0 and was validated at 20 rows, 10 frames, 20 unique
  method/sequence/frame keys, 60 finite normal metric values, zero duplicate
  keys, and zero nonfinite values. Retry5 batch 8 completed all ten immutable
  PLYs with exit code 0. Its geometry CSV was validated at 20 rows, 10 frames,
  20 unique method/sequence/frame keys, 300 finite numeric values, zero
  duplicates, zero unexpected nonfinite values, and exactly 60 expected
  deferred normal placeholders.
  Repaired retry6 batch 7 completed all ten PLYs with exit code 0, confirming
  the CUDA-node-exclusion repair beyond container startup. Its geometry CSV
  was validated at 20 rows, 10 frames, 20 unique method/sequence/frame keys,
  300 finite numeric values, zero duplicates, zero unexpected nonfinite values,
  and exactly 60 expected deferred normal placeholders. Dependency-linked k=20
  normal job `20121056_7` released normally after Slurm finished the node
  cleanup transition, completed with exit code 0, and was validated at 20
  rows, 10 frames, 20 unique method/sequence/frame keys, 60 finite normal
  metric values, zero duplicates, and zero nonfinite values. Retry5 batch-8
  normal job `20119953_8` completed with exit code 0 and was validated at 20
  rows, 10 frames, 20 unique method/sequence/frame keys, 60 finite normal
  metric values, zero duplicates, and zero nonfinite values.
  Continuation controller
  `20121057` released and submitted immutable inference wave `20122038` for
  batches 9-13 with `%2` GPU concurrency and matching k=20 normal array
  `20122039`; inference tasks 9 and 10 started on `nid000016`, loaded the
  checkpoint successfully, and produced eight and nine immutable PLYs,
  respectively. Their stderr
  contains only the known LUMI-stack and PyTorch checkpoint warnings, not an
  execution error.
  Original continuation controller `20120100` was cancelled while still
  dependency-pending after discovering that Slurm had spooled its pre-patch
  script at submission time; allowing it to release would have restored
  `seed_k_alpha=10` and omitted automatic normals. First replacement controller
  `20120480` contained the corrected OOM-safe settings and matching k=20
  normals, but was later cancelled after retry5 task 7 failed as recorded below.
  When retry5 task 7 first released, Slurm allocated an A40 on `nid000017` but
  CUDA was not visible inside the container; the task failed during the
  explicit CUDA preflight after nine seconds and wrote no output. This failed
  attempt remains in job/log history as `20119952_7`. Its obsolete normal task
  and whole-array continuation `20120480` were cancelled. Immutable replacement
  `20121055_7` uses prefix `iterativepfn_full_20260722_retry6`, excludes
  `nid000017`, passed CUDA startup on `nid000016`, and is running. Matching k=20
  normal job `20121056_7` depends on the replacement. Replacement continuation
  `20121057` depends on retry5 tasks 6 and 8 plus retry6 task 7, then will submit
  batches 9-13; future inference from that controller also excludes the node
  with failed CUDA visibility.
  Before that controller ran, `submit_iterativepfn_wave.slurm` was corrected to
  propagate the validated patch-size 1000, `seed_k_alpha=20`, and expandable
  allocator settings through Singularity `--cleanenv`. It now also captures the
  submitted inference-array ID and creates a matching per-task `aftercorr` k=20
  normal array, preventing both the former OOM configuration from returning and
  silent omission of normal metrics in subsequent waves. `bash -n` passes.
- Grad-PU LUMI smoke `20112889` completed. Full-data wave `20114168` uses
  immutable ten-frame batches on flash. Task `0` failed before inference when
  CUDA was temporarily invisible; tasks `1` onward are progressing and have
  written their first valid BlueSpeech PLY files. A new-name
  batch-0 retry `20114187` is dependency-gated. The earlier incorrectly routed
  attempt `20114159` was cancelled before any PLY was written and its empty
  result directories are retained rather than overwritten. A request to queue
  the remaining BlueSpeech wave was rejected by `AssocMaxSubmitJobLimit`; retry
  controller `20114369` was therefore queued behind batch-0 retry `20114187` to
  submit BlueSpeech batches `12-16` after scheduler slots became available.
  This avoided repeated submission-limit failures while keeping the next wave
  dependency-gated and immutable.
  BlueSpeech batches `01-04` (jobs `20114170-20114173`) subsequently completed.
  Each has 10 PLYs and exactly 20 geometry rows (10 baseline + 10 Grad-PU).
  Together they cover 40 unique frames (`0010-0049`) without duplicates; CD,
  Chamfer, F10, and F20 values are all finite. Batch `05` (job `20114179`) also
  completed with 10 PLYs, exactly 20 geometry rows, and finite geometry values.
  Batch `06` has written all 10 PLYs and its geometry CSV has exactly 20 rows
  (10 baseline + 10 Grad-PU); all geometry and F-score values are finite. Its
  60 normal-field cells are intentional placeholders and will be replaced by
  the separate `k=20` normal pipeline. The Slurm step remains active while
  finalization completes. Batch `07` has written all 10 PLYs and its geometry
  CSV has exactly 20 rows across 10 unique frames; all geometry and F-score
  values are finite, with only the expected 60 normal placeholders. Tasks `09`
  and `10` completed with exit code 0. Each has 10 PLYs and exactly 20 metric
  rows across 10 unique frames; all geometry and F-score values are finite,
  with only the expected 60 normal placeholders per batch. Task `11` started
  with CUDA visible and has written all 10 PLYs (frames `0110-0119`). Final
  frame `0119` completed inference with 576,582 input and 2,306,328 output
  points. Task 11 completed with exit code 0 and was validated at 10 PLYs,
  exactly 20 metric rows, 10 unique frames, no duplicate method/frame keys,
  and finite CD, Chamfer, precision, recall, and F-score fields. Its 60 normal
  cells are intentional placeholders for the separate `k=20` evaluation. Task
  `08` job `20115668`
  failed before inference because CUDA was temporarily invisible, writing no
  PLY. Immutable retry `20115684` uses prefix
  `gradpu_chunked_4x_full_20260722_retry2` and is dependency-gated after the
  current array; the empty failed retry1 directory remains untouched. The
  array completed, and immutable retries `20114187` (batch 0) and `20115684`
  (batch 8) both launched successfully and are actively performing CUDA
  inference. Batch 0 produced its first immutable PLY (`0000`, 528,129 input
  to 2,112,516 output points); batch 8 produced its first immutable PLY
  (`0080`, 614,214 input to 2,456,856 output points). Both immutable retries
  use separate result names. Batch 0 produced all 10 PLYs and completed
  successfully. Its committed geometry CSV was validated at exactly 20 rows,
  10 unique frames, 20 unique method/frame keys, and zero nonfinite geometry
  values. Its 60 nonfinite normal cells are the expected three deferred normal
  fields across 20 rows, not metric failures.
  Batch 8 produced all 10 PLYs and completed with exit code 0 and no
  CUDA-visibility regression. Its
  committed CSV was validated at 20 rows, 10 frames, 20 unique method/frame
  keys, and zero nonfinite geometry cells. Its 60 nonfinite normal cells are
  exactly the three deferred normal fields across 20 rows for later `k=20`
  evaluation.
  Controller `20114369` then completed successfully and submitted immutable
  Grad-PU wave `20118366` for batches `12-16`. Tasks 12 and 13 completed their
  inference/finalization phase on one A40 each. Each was validated at 10 PLYs,
  20 metric rows, 10 frames, 20 unique method/frame keys, and zero nonfinite
  geometry cells. Each has exactly 60 expected deferred normal placeholders.
  Tasks 14 and 15 completed with exit code 0 and wrote all ten PLYs each. Both committed
  geometry CSVs were validated at 20 rows, 10 frames, 20 unique method/frame
  keys, zero nonfinite geometry values, and exactly 60 expected deferred normal
  placeholders each. Task 16 started automatically with CUDA inference active
  on its eight remaining BlueSpeech frames and completed with exit code 0. Its
  committed geometry was validated at 8 PLYs, 16 metric rows, 8 frames, 16
  unique method/frame keys, 224 finite geometry values, and exactly 48 expected
  deferred normal placeholders. Its normal, texture, PCQM, and next-wave jobs
  were still waiting for Slurm to finish the short COMPLETING transition at
  the time of validation.
  Controller `20118387` completed and submitted inference wave `20120468` for
  the first five BlueVolley batches (`21-25`) under the same immutable prefix.
  Because Slurm had spooled that controller before its automatic-normal patch,
  matching k=20 array `20120474` was added manually with
  `aftercorr:20120468`. Patched controller `20120475` is dependency-gated on
  that wave and will submit batches `26-30` plus their matching normal array.
  Wave tasks 21 and 22 completed with exit code 0 and were each validated at
  10 PLYs, 20 metric rows, 10 frames, 20 unique method/sequence/frame keys,
  300 finite numeric geometry values, and exactly 60 expected deferred normal
  placeholders. K=20 normal task 21 completed with exit code 0 and was
  validated at 20 rows, 10 frames, 20 unique method/sequence/frame keys, 60
  finite normal metric values, zero duplicates, and zero nonfinite values.
  Normal task 22 completed with exit code 0 and was validated at 20 rows, 10 frames, 20
  unique method/sequence/frame keys, 60 finite normal metric values, zero
  duplicates, and zero nonfinite values.
  Tasks 23 and 24 started automatically under the array's `%2` limit and
  each produced all ten immutable PLYs. Batch 2 completed with exit code 0 and
  was validated at 20 rows, 10 frames, 20 unique method/sequence/frame keys,
  300 finite numeric values, zero duplicates, zero unexpected nonfinite values,
  and exactly 60 expected deferred normal placeholders. Batch 3 also completed
  with exit code 0 and its CSV passed the same shape, uniqueness, placeholder,
  and finiteness checks. Their matching k=20 normal tasks `20120474_23` and
  `20120474_24` completed with exit code 0. Each was independently validated at
  20 rows, 10 frames, 20 unique method/sequence/frame keys, 60 finite normal
  metric values, zero duplicates, and zero nonfinite values. Global
  task 25 (local immutable batch 4) started automatically on `nid000016` and
  produced its first six immutable PLYs. Global task IDs 21/22 correctly
  map to sequence-local immutable method
  names `batch_00`/`batch_01`; zero counts under nonexistent BlueVolley
  `batch_21`/`batch_22` paths are therefore not missing outputs.
  Grad-PU batch-16 k=20 job `20120093_16` completed with exit code 0 and was
  validated at 16 rows, 8 frames, 16 unique method/frame keys, 48 normal
  values, and zero nonfinite values. Its separate PCQM job `20120356_16`
  completed with exit code 0 and was validated at 16 rows, 8 frames, 16 unique
  method/sequence/frame keys, zero duplicates, and zero nonfinite values.
  Texture job `20120355_16` completed with exit code 0 and
  was validated at 16 per-frame rows, 8 frames, 16 unique method/frame keys,
  112 finite point-count/YUV/projection values, 96 per-view rows with 192
  finite SSIM/LPIPS values, and exactly 16 intentional PCQM placeholders.
  Dependency-linked normal job `20120093_16` was submitted for Grad-PU batch 16;
  it will run the same k=20 evaluator only after inference exits successfully.
  The queued Grad-PU continuation controller was likewise updated to capture its
  inference-array ID and automatically submit a corresponding per-task
  `aftercorr` k=20 normal array. `bash -n` passes.
  Normal
  values remain outside this adapter and
  will be computed consistently with the separate `k=20` pipeline after
  consolidation.
- Generic immutable-batch normal job
  `jobs/batched_method_normal_k20_array.slurm` was added and passed `bash -n`.
  It reuses the retained HE `pca_knn20` cache, estimates each enhanced output's
  normals in memory with `k=20`, writes a separate method/run CSV, and refuses
  to overwrite an existing CSV. Array `20119211` was submitted for the fully
  validated Grad-PU batches 12 and 13; both tasks completed with exit code 0.
  Each immutable `k=20` CSV was validated at 20 rows, 10 frames, 20 unique
  method/frame keys, and zero nonfinite `N_Acc`, `N_Comp`, or combined normal
  values. Dependency-gated normal array `20119594` for Grad-PU batches 14-15
  released successfully after both inference tasks completed and both normal
  tasks completed with exit code 0. Each CSV was validated at 20 rows, 10 frames,
  20 unique method/frame keys, 60 normal values, and zero nonfinite values.
  IterativePFN uses replacement normal job `20119803_4` for original batch 4
  and `aftercorr` array `20119953` for retry5 batches 5-8 after its OOM repair.
  The evaluator's
  two cKDTree operations
  explicitly use eight workers, so future jobs now request eight rather than
  sixteen CPU cores; `bash -n` remains clean. This prevents wasting half of the
  reserved CPU allocation while preserving exactly the same `k=20` algorithm
  and numerical path. This closes the
  missing normal-metric stage for immutable
  ten-frame method batches without retaining enhanced-normal cache files.
  Geometry and k=20 normals are dependency-linked for the active Grad-PU and
  IterativePFN waves. Their full-data texture/perceptual and separate PCQM
  stages still need to be scheduled against the immutable batch method names
  after outputs are complete; do not treat geometry plus normals alone as the
  complete objective-metric suite. This should be staged carefully because the
  project CPU allocation is substantially more constrained than GPU hours.
  Reusable jobs `jobs/batched_method_texture_array.slurm` and
  `jobs/batched_method_pcqm_array.slurm` now implement those missing stages for
  any immutable `METHOD_PREFIX`: the first computes YUV-PSNR,
  projection-SSIM, and LPIPS with intentional `--skip-pcqm` placeholders; the
  second computes PCQM separately and rejects nonfinite values. Both derive
  frames only from the corresponding immutable batch directory, write under
  method-specific run paths, refuse missing inputs, and pass `bash -n`. They
  were first submitted only for already completed and validated batches with
  one-task concurrency: Grad-PU batches 12-15 use texture job `20120334` and
  PCQM job `20120335`; IterativePFN batch 4 uses texture job `20120336_4` and
  PCQM job `20120337_4`. All four first tasks entered RUNNING state. Incomplete
  inference batches remain unscheduled for these metric stages. Startup was
  validated: both texture jobs produced finite Y-PSNR, projection-SSIM, and
  LPIPS values, while their PCQM fields are the intentional `--skip-pcqm`
  placeholders; the separate Grad-PU PCQM job produced finite value `0.146932`
  and neither PCQM job reported a nonfinite-value exception.
  IterativePFN batch-4 texture job `20120336_4` completed with exit code 0 and
  was validated at 20 per-frame rows, 10 frames, 20 unique method/frame keys,
  140 finite point-count/YUV/projection values, 120 per-view rows with 240
  finite SSIM/LPIPS values, and exactly 20 intentional PCQM placeholders.
  Its separate PCQM job `20120337_4` completed with exit code 0 and was
  validated at 20 rows, 10 frames, 20 unique method/sequence/frame keys, and
  zero nonfinite PCQM values.
  Grad-PU batch-12 texture job `20120334_12` completed with exit code 0 and was
  validated at 20 per-frame rows, 10 frames, 20 unique method/frame keys, 140
  finite point-count/YUV/projection values, 120 per-view rows with 240 finite
  SSIM/LPIPS values, and exactly 20 intentional PCQM placeholders. Array task
  13 then started automatically.
  Grad-PU batch-13 texture task `20120334_13` subsequently completed with exit
  code 0 and the same validated shape: 20 per-frame rows, 10 frames, 20 unique
  method/frame keys, 140 finite point-count/YUV/projection values, 120 per-view
  rows with 240 finite SSIM/LPIPS values, and exactly 20 intentional PCQM
  placeholders. Task 14 started automatically.
  Grad-PU batch-14 texture task `20120334_14` also completed and was validated
  at 20 per-frame rows, 10 frames, 20 unique method/frame keys, 120 per-view
  rows, zero duplicate keys, finite YUV-PSNR/projection-SSIM/LPIPS values, and
  exactly 20 intentional `--skip-pcqm` placeholders. Task 15 started
  automatically and completed with exit code 0. Batch 15 was validated at 20
  per-frame rows, 10 frames, 20 unique method/frame keys, 120 per-view rows,
  zero duplicate keys, finite YUV-PSNR/projection-SSIM/LPIPS values, and exactly
  20 intentional `--skip-pcqm` placeholders. The separate batch-12 PCQM task
  completed with exit code 0 and was validated at 20 rows, 10 frames, 20
  unique method/sequence/frame keys, zero duplicates, and zero nonfinite
  values. PCQM task 13 completed with exit code 0 and was validated at 20 rows,
  10 frames, 20 unique method/frame keys, zero duplicates, and zero nonfinite
  values. Task 14 completed with exit code 0 and was validated at 20 rows, 10
  frames, 20 unique method/sequence/frame keys, zero duplicates, and zero
  nonfinite values. Task 15 started automatically and emitted its first finite
  first nine PCQM frame pairs plus the next baseline value; all nineteen emitted values are
  finite. Grad-PU
  BlueVolley inference batches
  2 and 3 have produced three and four final immutable frame PLYs,
  respectively; batch 4 remains held by the two-task GPU array limit.
  Grad-PU batch-16 texture job `20120355_16` and PCQM job `20120356_16`
  were additionally submitted with `afterok:20118366_16`; texture is complete
  and validated, while PCQM remains active as recorded above.
- IterativePFN continuation controller `20118414` completed and submitted
  immutable array `20119277` for batches `4-8` with `%2` GPU concurrency. The
  controller script is `jobs/submit_iterativepfn_wave.slurm`; `bash -n`
  validation passed before submission.
- Count SAPCU and its extension, *Self-Supervised Arbitrary-Scale Implicit
  Point Clouds Upsampling*, once as one method/paper family. They share one
  benchmark implementation and must not create duplicate result rows.
- Grad-PU was read by the user and confirmed included as original Final-20
  method #20 on 2026-07-22. Its new LUMI run must retain the explicit
  chunked/non-default large-frame adaptation label.
- Full-dataset expansion has been approved by the user for methods included in
  the survey/special-issue paper. Included methods should be run on the full
  UVG-CWI-DQPC dataset even if smoke or selected-10 evidence is negative; record
  negative results as benchmark findings, not skip reasons.
- Full-dataset input data must use `HE_15` as reference and `CGv2_15` as
  consumer-grade input.
- Full-dataset aggregate metrics must first average all frames inside each
  sequence, then average the 12 sequence means for the dataset-level result.
- Full-dataset method arrays should run up to two sequences in parallel per
  method when GPU availability allows it. This only changes scheduling
  throughput; it does not change method inference settings.
- If a full-dataset method/sequence fails, record the method, sequence, job id,
  log path, error message, and likely cause here, then continue with the
  remaining sequences and methods instead of blocking the full benchmark.
- The accidentally submitted full170 jobs `24030797`, `24030799`, `24030800`,
  and `24030801` were cancelled on 2026-06-19.
- Exception: PD-Flow full170 was explicitly requested after the user read the
  paper. Job `24076038` was submitted on 2026-06-21 as `pdflow_full170` and
  failed after frame `0000` due workspace disk quota while writing output PLY;
  rerun job `24076375` wrote frames `0000` through `0013` under project storage
  before failing inside PD-Flow KNN on frame `0014`. Keep it marked as a broader
  ablation rather than the default selected-10 protocol.
- Publication-year screening rule: prioritize point-cloud-only enhancement
  methods published in 2021 or later. Methods published before 2021 should be
  marked as needing paper reading and should not receive new benchmark runs
  unless they already show positive UVG selected-10 performance or the user
  explicitly selects them as historical baselines.
- Storage/quota note: large selected-10 PLY outputs should live in project
  storage under `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/`, with
  symlinks kept at the original repo paths. Do not write full-sequence outputs
  into quota-limited home storage.
- Aggregate selected-10 summary: `skills/methods/SELECTED10_RESULTS_SUMMARY.md`
  and `results/selected10_method_summary.csv`.
- User has read `Efficient Point Clouds Upsampling via Flow Matching` / `PUFM`
  and marked it suitable for the survey scope; it has already been benchmarked
  on selected-10 as `pufm_pugan_4x_selected10`.

Full-dataset UVG-CWI-DQPC run audit:

- Run ids: download/layout `24518695`/`24518798`; method arrays
  `24518799` through `24518821`; texture metrics `24518822`; final summary
  `24518823`.
- Dataset: all 12 sequences downloaded from the official UVG-CWI-DQPC source
  using `HE_15` as reference and `CGv2_15` as input. Manifest contains 2152
  paired frames. `BlueSpeech`, `GoodVision`, and `TrumanShow` each have one CG
  frame without HE reference, so only paired frames are evaluated.
- Scheduling: method arrays were updated to run up to two sequences in parallel
  per method (`0-11%2`). This is a throughput-only scheduling change.
- As of 2026-07-11 status check, completed all 12 sequence summaries:
  `apuldi_local_pu1k_4x_2048_full`, `crcir_aftercomp_4x_full`,
  `neuralpoints_16x_2048_full`, `pucrn_pu1k_4x_full`,
  `puflow_discrete_full`, `pufm_pugan_4x_full`, `spu_pointnet_4x_full`,
  and `upsample_clean_ounet_full`.
- Interim geometry-only completed-12 summary was written locally to
  `results/full_dataset/summary_completed12_interim/` using per-frame metric
  CSVs. The first interim texture/perceptual attempt `24544711` was cancelled
  immediately by the scheduler with `AdminComment=reason=budget`; reducing the
  GPU request to one-GPU billing and resubmitting as `24544729` was also
  cancelled for budget. A CPU-staging fallback attempt `24544741` was likewise
  cancelled for budget before metric code ran. Treat this as an allocation
  blocker, not a texture/perceptual implementation failure.
- Corrected texture/perceptual and summary job files now use `per_frame_metrics.csv`
  for geometry summarization and one-GPU/CPU-aware resource shapes. The current
  final all-method texture job is `24544743` on CPU staging, dependency-gated
  behind the method arrays, with final summary `24544746`. It will run only if
  budget is available when the dependency is released.
- Resource/time/domain failures recorded during this pass:
  - `pdflow_full_dataset`: all sequence tasks failed, usually inside PD-Flow
    patch KNN with `RuntimeError: min(): Expected reduction dim to be
    specified for input.numel() == 0`; `VirtualLife` also hit A100 CUDA OOM.
    This matches the earlier dense-frame instability seen on
    `OrangeKettlebell` frame `0014`.
  - `pc2pu_4x_chunks256_full`: all sequence tasks failed with
    `RuntimeError: selected index k out of range` after writing only partial
    frame outputs. Treat this as a dense full-sequence wrapper/fallback failure
    for the first pass.
  - `pu_gaussian_pu1k_4x_full`: partial sequence completion only. Failed
    sequences hit the official PU-Gaussian KNN patch extractor with
    `ValueError: Expected n_neighbors <= n_samples` for patches smaller than
    256 neighbors; one early task also hit a symlink race before rerun-like
    partial outputs existed.
  - `pdlts_light_fbm_full`: `BouncingBlue` and `TicTacToe` failed on A100
    CUDA OOM; other sequence summaries completed.
  - `p2p_bridge_pvds_punet_full`: early sequences timed out at the 24h walltime
    after partial outputs; `BouncingBlue` failed with A100 CUDA OOM. Later
    sequence tasks are still running/pending in the first pass.
  - `pudm_pu1k_4x_full`, `spupmd_pu1k_4x_full`, and `pointcleannet_full`:
    early sequence tasks timed out at the 24h walltime with partial frame
    outputs, reflecting very slow full-frame processing.
  - `mag_full` and `score_denoise_full`: `BouncingBlue` failed with the same
    empty-neighborhood `min()` reduction error seen in score/MAG large-frame
    patch processing; later sequence tasks continue.
- Current first-pass policy: do not block the benchmark for these failures.
  Keep the successful per-sequence summaries and partial outputs, let remaining
  arrays continue, then decide targeted retries or exclusions after the first
  full pass and texture/perceptual jobs finish.

## SCUTSurface / SUSTech-First Methods

| Method | Source/provenance | Reading paper | Publication year | Paper title | Paper / repo link | Status | Why / observed result | Potential adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAL | SCUTSurface/SUSTech | yes | 2020 | Sign Agnostic Learning of Shapes From Raw Data | Paper/repo: `https://github.com/matanatz/SAL`; SCUTSurface list: `https://github.com/Gorilla-Lab-SCUT/SCUTSurface-code/tree/main/reconstruction`<br>Publication: CVPR 2020 | Tried, selected-10 completed | Worse than CG baseline on selected-10. Reconstruction smooths/hallucinates geometry and loses capture sampling structure. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/sal_selected10/summary_metrics.csv` | Only rerun as a non-default ablation if changing training length, point cap, resolution, or sampling strategy is explicitly allowed. |
| IGR | SCUTSurface/SUSTech | yes | 2020 | Implicit Geometric Regularization for Learning Shapes | `https://github.com/amosgropp/IGR`<br>Publication: ICML 2020 | Tried on frame `0000`; parameter sweep completed; not expanded | All tested variants were worse than CG on frame `0000`, so expanding to selected-10 is not justified under the current benchmark rule. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/igr_sweep/frame_0000_baseline_vs_igr_sweep.csv` | Could be revisited only after reading the paper/code for normalization and sampling assumptions; selected-10 expansion would be expensive and likely not useful. |
| Points2Surf | SCUTSurface/SUSTech | yes | 2020 | Points2Surf: Learning Implicit Surfaces from Point Clouds | `https://github.com/ErlerPhilipp/points2surf`<br>Publication: ECCV 2020 | Tried, selected-10 completed | Worse than CG baseline on selected-10. Mesh reconstruction plus resampling degrades this dynamic capture benchmark. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/points2surf/summary_metrics.csv` | Only rerun if a paper-faithful setting for real dynamic captures is identified. |
| DSE-Meshing | SCUTSurface/SUSTech | no | 2021 | Learning Delaunay Surface Elements for Mesh Reconstruction | `https://github.com/mrakotosaon/dse-meshing`<br>Publication: CVPR 2021 | Held / skipped | Requires TensorFlow 1.15/Python 3.6 and C++ triangle-selection build; not efficient for current pass. | Run only if a maintained PyTorch implementation or ready environment is found. |
| DeepMLS | SCUTSurface/SUSTech | yes | 2019 | Deep Moving Least Squares | `https://github.com/Andy97/DeepMLS`<br>Publication: CVPR 2019 | Held / skipped | TensorFlow/custom CUDA/O-CNN setup; not quota-efficient. | Run only if dependency setup is justified by the paper review. |
| DeepSDF | SCUTSurface/SUSTech | yes | 2019 | DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation | `https://github.com/facebookresearch/DeepSDF`<br>Publication: CVPR 2019 | Held / skipped | Needs watertight mesh SDF samples and trained experiment checkpoints; not a direct raw point-cloud enhancement path. | Treat as domain-transfer reconstruction only, not direct enhancement. |
| Occupancy Networks | SCUTSurface/SUSTech | yes | 2019 | Occupancy Networks: Learning 3D Reconstruction in Function Space | `https://github.com/autonomousvision/occupancy_networks`<br>Publication: CVPR 2019 | Held / skipped | Old PyTorch/Cython extension and ShapeNet-style input/checkpoint assumptions. | Run only as domain-transfer reconstruction after stronger direct methods. |

## External Enhancement Methods

| Method | Source/provenance | Reading paper | Publication year | Paper title | Paper / repo link | Status | Why / observed result | Potential adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PointCleanNet | LilydotEE survey repo | yes | 2019 | PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds | `https://github.com/mrakotosaon/pointcleannet`<br>Publication: Computer Graphics Forum 2019 | Selected-10 completed as SLURM job `23980358`; selected-10 texture/perceptual metrics completed as `24007175`; subjective quality accepted by user | Geometry selected-10 improved mean Chamfer and F-scores against CG baseline. Texture/perceptual selected-10 is mixed: improves Y-PSNR and YUV mean, but worsens projection SSIM, LPIPS, and PCQM. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pointcleannet_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10/summary_texture_perceptual_metrics.csv` | Candidate for expansion to more sequences. |
| Pointfilter | LilydotEE survey repo | yes | 2020 | Pointfilter: Point Cloud Filtering via Encoder-Decoder Modeling | `https://github.com/dongbo-BUAA-VR/Pointfilter`<br>Publication: IEEE TVCG 2020 | Selected-10 completed via first job `24046276` plus resume job `24057181`; selected-10 texture/perceptual completed as `24062115` | Smoke improved accuracy slightly but worsened completeness, recall, Chamfer, and F-scores. First selected-10 pass wrote frames `0000` through `0060` before the 12h limit; resume job reused existing outputs and computed `0070`, `0080`, and `0090` without changing Pointfilter inference settings. Selected-10 is negative overall: `CD_Acc` improves `25.5177 -> 24.7878`, but `CD_Comp` worsens `27.6065 -> 35.6982`, `chamfer-L1` worsens `53.1241 -> 60.4860`, `chamfer-L2` worsens `2293.13 -> 2915.04`, `F_10` worsens `0.2459 -> 0.2119`, and `F_20` worsens `0.4552 -> 0.3952`. Texture/perceptual is mixed but mostly negative: `dYUV-PSNR +0.1639`, `dProj-SSIM -0.0113`, `dLPIPS -0.0205`, `dPCQM -0.0022`. RGB is transferred from CG by `k=1`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pointfilter_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pointfilter_crcir/summary_texture_perceptual_metrics.csv`. | Excluded from current execution because its 2020 publication is outside the 2021+ cutoff. Preserve historical selected-10 outputs; do not submit new Pointfilter jobs. |
| DMRDenoise | LilydotEE survey repo | yes | 2020 | Differentiable Manifold Reconstruction for Point Cloud Denoising | `https://github.com/luost26/DMRDenoise`<br>Publication: ACM MM 2020 | Failed under official/default inference | Official model crashed before writing denoised `.xyz`: tensor neighborhood size mismatch, expected 23 got 24. Log: `logs/dmrdenoise_ok_0000_smoke_23527266.out` | Could rerun only as a clearly labeled non-default compatibility ablation changing patch/cluster/KNN behavior. |
| GPDNet | LilydotEE survey repo | yes | 2020 | Learning Graph-Convolutional Representations for Point Cloud Denoising | `https://github.com/diegovalsesia/GPDNet`<br>Publication: ECCV 2020 | Held / skipped | Repo was cloned for inspection. It requires Python 2.7, TensorFlow 1.12, CUDA 9.0, and old dependencies, so it violates the current TF-skip rule. | Run only if a maintained PyTorch/pretrained implementation is found or the user approves a TensorFlow exception. |
| Deep-RS | Official author release; also listed by the LilydotEE survey repo | yes, included by user decision | 2022 | Deep Point Set Resampling via Gradient Fields | Paper: `https://arxiv.org/abs/2111.02045`; official code: `https://github.com/ChenhLiwnl/deep-rs`; author page: `https://luost.me/publications`<br>Publication: IEEE TPAMI 2022 (print 2023) | Smoke completed as `24030907`; included as smoke-only/negative evidence | The public repository identifies itself as the official TPAMI implementation and includes denoising and upsampling checkpoints. It was verified again on 2026-07-21 and is restored locally under `third_party/enhancement/deep-rs`; therefore code status is **available**, not `NA`. Shared `torch_env` loads the denoising checkpoint with zero missing/unexpected weights after a lazy-import compatibility patch for the absent MCCNN module. Adapter uses the repo's own large-point-cloud clustering helper, then transfers RGB from CG by `k=1`. Frame `0000` completed on UVG-CWI-DQPC and is clearly negative overall: tiny `CD_Acc` gain (`23.6178 -> 23.6153`) and tiny `P_5` gain, but worse `CD_Comp`, Chamfer, recall, and all F-scores (`F_10` `0.2570 -> 0.2471`). Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/deeprs_denoise/summary_metrics.csv`; output: `results/method_outputs/deeprs_denoise/OrangeKettlebell/15fps/frame_0000.ply`. | Include in the survey/benchmark record as a theory-relevant gradient-field resampling method. Clearly label the available evidence as smoke-only and negative; broader expansion requires an explicit override of the smoke gate. |
| TotalDenoising | LilydotEE survey repo | yes | 2019 | Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning | `https://github.com/phermosilla/TotalDenoising`<br>Publication: ICCV 2019 | Held / skipped | Repo requires TensorFlow custom ops for KNN, neighbor selection, point-to-mesh distance, and spatial convolution. | Skip under the current TF/custom-op rule unless a PyTorch/pretrained replacement is found. |
| Score-Denoise | LilydotEE survey repo | no | 2021 | Score-Based Point Cloud Denoising | `https://github.com/luost26/score-denoise`<br>Publication: ICCV 2021 | Smoke completed as `24001558`; selected-10 completed as `24003261`; selected-10 texture/perceptual metrics completed as `24007175`; subjective quality accepted by user | Geometry selected-10 slightly improves `CD_Acc`, `chamfer-L1`, `chamfer-L2`, `F_10`, and `F_20`, but worsens `CD_Comp`. Texture/perceptual selected-10 is near-neutral/mixed: slightly improves Y-PSNR and YUV mean, slightly worsens projection SSIM and PCQM, and is almost unchanged on LPIPS. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/score_denoise_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10/summary_texture_perceptual_metrics.csv` | Candidate for more sequences. |
| RePCD-Net | LilydotEE survey repo | no | 2022 | RePCD-Net: Feature-Aware Recurrent Point Cloud Denoising Network | Paper/repo: `https://github.com/chenhonghua/RePCD-Net`; paper: `https://link.springer.com/article/10.1007/s11263-021-01564-7`<br>Publication: IJCV 2022 | Blocked / not runnable under current rule | Repo cloned under `third_party/enhancement/RePCD-Net`. The public repo releases denoised example results and synthetic test data, but no visible pretrained model checkpoint. README usage requires compiling TensorFlow operators and running `codes/main.py --phase test`; it also notes large inputs should be split into patches. Under the current skip-TF/custom-op and no-training rules, no smoke job was submitted. | Keep as a survey-listed denoising method, but do not benchmark unless a maintained PyTorch/pretrained implementation or a complete TF checkpoint/env exception is approved. |
| PD-Flow | Independent paper/repo search | yes, user read and included in final 20 | 2022 | PD-Flow: A Point Cloud Denoising Framework with Normalizing Flows | Paper: `https://arxiv.org/abs/2203.05940`; repo: `https://github.com/unknownue/pdflow`<br>Publication: ECCV 2022 | Paper read by user; included in final 20; smoke completed as `24004024`; smoke texture/perceptual completed as `24056897`; full170 broader ablation failed as `24076038` due disk quota; project-storage rerun `24076375` failed at frame `0014` | Pretrained checkpoint exists locally. Compatibility fixes applied for lazy `kaolin` import and removed NumPy alias `np.long`. Adapter `scripts/run_pdflow_selected_frames.py` reads UVG `xyzrgb`, writes only XYZ to PD-Flow's `.xyz` input because PD-Flow expects `(N,3)`, then transfers RGB from the original CG frame to the denoised geometry using nearest-neighbor `k=1` with no averaging before writing UVG-compatible XYZRGB PLY. Frame `0000` geometry improves `CD_Acc` and precision slightly, but worsens completeness, Chamfer, recall, and F-scores. Smoke texture/perceptual improves YUV-PSNR and PCQM, but worsens projection SSIM and LPIPS. Full170 is a user-requested ablation after paper reading, not a selected-10 gate pass; job `24076038` finished denoising frame `0000` but failed while writing PLY with `OSError: [Errno 122] Disk quota exceeded`. Rerun job `24076375` used the same PD-Flow settings and wrote work files, method outputs, and metrics under `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results` via `--results-root`; it produced frames `0000` through `0013`, then failed on frame `0014` inside `pytorch3d.ops.knn_gather` with `RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0`, indicating a dense-frame/cluster instability in PD-Flow's patch KNN path. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pdflow/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/smoke_0000_pdflow/summary_texture_perceptual_metrics.csv`; logs: `logs/pdflow_ok_full170_24076038.out`, `logs/pdflow_ok_full170_24076375.out`; rerun job file: `jobs/pdflow_orangekettlebell_full170.slurm`. | Included in final 20 as a theory-relevant normalizing-flow denoising method. Treat performance as mixed/negative and do not retry full170 without a paper/code-level decision, because avoiding the frame `0014` failure may require non-default clustering or patch handling. |
| P2P-Bridge | Independent paper/repo search | yes, user read and included in survey | 2024 | P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising | Paper/project: `https://arxiv.org/abs/2408.16325`; repo: `https://github.com/matvogel/P2P-Bridge`; checkpoint folder: `https://drive.google.com/drive/folders/1hkd_gTU2EAMFJmgUzHmifviKDVunb6aK?usp=sharing`<br>Publication: ECCV 2024 | Repo cloned; object denoising checkpoint downloaded; smoke completed as `24080593`; selected-10 completed as `24093763`; texture/perceptual completed as `24126456` after cancelling short job `24126400` | Official PyTorch repo cloned under `third_party/enhancement/P2P-Bridge`. Object denoising checkpoint is local at `third_party/enhancement/P2P-Bridge/pretrained/PVDS_PUNet/latest.pth` with `opt.yaml`. Broad scene-checkpoint folders accidentally downloaded during inspection were removed to save space. Method-local Python deps are under `third_party/enhancement/P2P-Bridge/python_deps`. Upstream inference requires a compiled `pointnet2_batch_cuda` extension from OpenPoints; job `jobs/p2p_bridge_build_and_smoke.slurm` builds only that extension on a compute node using the `2023` + `CUDA/12.1.1` module stack, then runs `scripts/run_p2p_bridge_selected_frames.py`. Initial retries exposed three setup issues: `CUDA/12.1.1` must be loaded from stack `2023`; Chamfer/EMD metric imports in `models/evaluation.py` must be optional for inference-only denoising; and the compiled `pointnet2_batch` directory must be on `PYTHONPATH`. The adapter writes only XYZ to P2P-Bridge, runs the released `denoise_object.py`, then transfers RGB from CG by `k=1`. Selected-10 is mixed/positive: `CD_Acc 25.5177 -> 23.7832`, `chamfer-L1 53.1241 -> 51.6558`, `chamfer-L2 2293.13 -> 2202.25`, `F_10 0.2459 -> 0.2566`, and `F_20 0.4552 -> 0.4733` improve, while `CD_Comp 27.6065 -> 27.8727` and recall metrics worsen slightly. Geometry metrics are symlinked at `results/uvg_cwi_dqpc/OrangeKettlebell/p2p_bridge_pvds_punet_selected10/summary_metrics.csv` and stored physically under `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/p2p_bridge_pvds_punet_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_p2p_bridge/summary_texture_perceptual_metrics.csv`; texture job file: `jobs/texture_metrics_orangekettlebell_selected10_p2p_bridge.slurm`. Texture/perceptual is mixed: `YUV-PSNR 21.1708 -> 21.6352` improves, but projection SSIM `0.6190 -> 0.6150`, LPIPS `0.4684 -> 0.4751`, and PCQM `0.1147 -> 0.1014` worsen. The paper also supports optional image/color/DINOv2 conditioning, but our benchmark uses the point-cloud-only XYZ path to keep the method comparable. | Included in survey and final-20 candidate list. Inspect subjectively before broader runs; objective result is geometry-positive/mixed and texture-positive only on YUV-PSNR. |
| PD-LTS | Independent paper/repo search | yes, user included in survey | 2024 | Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network | Paper: `https://openaccess.thecvf.com/content/CVPR2024/html/Mao_Denoising_Point_Clouds_in_Latent_Space_via_Graph_Convolution_and_CVPR_2024_paper.html`; repo: `https://github.com/yanbiao1/PD-LTS`<br>Publication: CVPR 2024 | Repo cloned; pretrained checkpoints included; light smoke completed as `24107462`; heavy smoke completed as `24108284`; included by user decision; not expanded to selected-10 | Official PyTorch repo cloned under `third_party/enhancement/PD-LTS`. The repo includes light and heavy pretrained FBM checkpoints at `product/ckpt/Denoiseflow-light-FBM.ckpt` and `product/ckpt/Denoiseflow-heavy-FBM.ckpt`. Adapter `scripts/run_pdlts_selected_frames.py` writes only XYZ to PD-LTS because the method expects `(N,3)`, runs the official denoising script, then transfers RGB from the original CG frame to the denoised geometry by nearest-neighbor `k=1` with no averaging. First smoke job `24097325` failed before inference because the custom Pila CUDA extension used conda `nvcc` and could not find `thrust/complex.h`; the SLURM job now loads Snellius modules `2023` and `CUDA/12.1.1`, sets `TORCH_CUDA_ARCH_LIST=8.0`, and builds JIT extensions under project storage. Second job `24097381` compiled Pila successfully but failed on an eager `kaolin.metrics.pointcloud` import from `metric/loss.py`; this loss is not used by FBM denoising inference, so local patch `skills/methods/patches/pdlts_inference_compat.patch` makes the Kaolin import lazy while preserving an explicit error if `ChamferCUDA` is called. Light-checkpoint smoke job `24107462` completed but is mostly negative: improves `CD_Acc 23.6178 -> 23.2732` and precision, but worsens `CD_Comp 26.4884 -> 27.1074`, `chamfer-L1 50.1062 -> 50.3806`, `chamfer-L2 2098.37 -> 2122.80`, `F_10 0.2570 -> 0.2545`, and `F_20 0.4866 -> 0.4843`. Heavy-checkpoint smoke job `24108284` is negative on all listed objective geometry metrics: `CD_Acc 23.6178 -> 23.6419`, `CD_Comp 26.4884 -> 27.7493`, `chamfer-L1 50.1062 -> 51.3913`, `chamfer-L2 2098.37 -> 2192.21`, `F_10 0.2570 -> 0.2464`, and `F_20 0.4866 -> 0.4726`. Light metrics: `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/pdlts_light_fbm/summary_metrics.csv`; heavy metrics: `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/pdlts_heavy_fbm/summary_metrics.csv`. | Included in survey by user decision as a recent CVPR 2024 latent-space denoising method. Keep benchmark evidence labeled smoke-only/negative; do not expand to selected-10 under the current objective-metric gate unless explicitly requested. |
| PathNet | Independent paper/repo search | yes, user included in survey | 2024 | PathNet: Path-Selective Point Cloud Denoising | Paper/repo: `https://github.com/ZeyongWei/PathNet`; pretrained folder: `https://drive.google.com/drive/folders/1qaxpcqBGVK59HBfTTS68AoaqSWLcp9si?usp=sharing`<br>Publication: IEEE TPAMI 2024 | Repo cloned; pretrained checkpoint downloaded; smoke completed as `24110023`; selected-10 completed as `24112993`; texture/perceptual first run `24161176` produced baseline-only rows due missing symlink; rerun `24163722` completed | Official PyTorch repo cloned under `third_party/enhancement/PathNet`. The Drive folder was listed before download; only `best_model.pth` was downloaded to `third_party/enhancement/PathNet/log/path-denoise/model/checkpoints/best_model.pth`, while `test_dataset.zip` and `train _data.hdf5` were not downloaded. The checkpoint loads with all denoiser/analyser keys matched. Adapter `scripts/run_pathnet_selected_frames.py` writes only XYZ to PathNet because the method is geometry denoising, runs the official `get_model(6, 2)` and analyser with the released checkpoint, then transfers RGB from CG to denoised geometry by nearest-neighbor `k=1` with no averaging. Because PathNet's official inference is designed for roughly 10K-50K points and performs PCA normalization for every 128-neighbor patch, the UVG run uses a clearly labeled Morton-chunked large-frame wrapper with `--chunk-size 50000`, preserving the original point count. Selected-10 is weak/mixed: `F_10 0.2459 -> 0.2465` and `F_20 0.4552 -> 0.4564` improve slightly, but `CD_Acc 25.5177 -> 25.5261`, `CD_Comp 27.6065 -> 27.6568`, `chamfer-L1 53.1241 -> 53.1830`, and `chamfer-L2 2293.13 -> 2304.99` worsen. Geometry metrics are symlinked at `results/uvg_cwi_dqpc/OrangeKettlebell/pathnet_chunked_selected10/summary_metrics.csv` and stored physically under `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/pathnet_chunked_selected10/summary_metrics.csv`; selected-10 job: `jobs/pathnet_orangekettlebell_selected10.slurm`; texture/perceptual job: `jobs/texture_metrics_orangekettlebell_selected10_pathnet.slurm`; rerun job `24163722` used repo-local symlink `results/method_outputs/pathnet_chunked_selected10 -> /gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/method_outputs/pathnet_chunked_selected10`. Texture/perceptual is mixed: `dYUV-PSNR +0.0069`, `dProj-SSIM -0.0027`, `dLPIPS -0.0096`, `dPCQM +0.0003`. | Included in survey by user decision. Keep objective selected-10 evidence labeled weak/mixed and note the 50K Morton-chunked large-frame wrapper in the benchmark discussion. |
| IterativePFN | Independent paper/repo search | yes; user read and included on 2026-07-22 | 2023 | IterativePFN: True Iterative Point Cloud Filtering | Repo: `https://github.com/ddsediri/IterativePFN`<br>Publication: CVPR 2023 | Included; earlier smoke `24004995` completed and was negative; immutable LUMI retry `20114142` is running | The official repository was restored on LUMI at commit `79efe40`, and the official `denoisenet-ep-99.ckpt` was restored from the authors' `data_and_ckpt.zip` with SHA-256 `bb06759bf6a970b540e6b67dad02f31583a0417bc0c0ecafb86699738ae56183`. First LUMI job `20114108` failed before inference because `torch_cluster` was missing. The fix installed ABI-matched `torch-cluster 1.6.3+pt24cu121`, `torch-geometric 2.8.0.post1`, PyTorch Lightning 1.9.5, and TorchMetrics 0.11.4 into the CUDA-container overlay; all imports passed. Retry `20114142` uses immutable method name `iterativepfn_lumid_smoke_20260722_retry1`. The older frame `0000` evidence worsened Chamfer, completeness, and F-scores vs CG baseline and remains preserved. | Apply it to UVG-CWI-DQPC under a new immutable method name despite the negative earlier smoke, retaining a clear fixed-size/chunking and cross-domain caveat. Validate retry `20114142`, then use resumable batches before objective geometry, normal (`k=20`), texture/perceptual, and PCQM metrics. |
| MAG | Independent paper/repo search | no | 2023 | Point Cloud Denoising via Momentum Ascent in Gradient Fields | Paper: `https://ieeexplore.ieee.org/abstract/document/10222122`; repo: `https://github.com/IndigoPurple/MAG`<br>Publication: ICIP 2023 | Smoke completed as `24000030`; selected-10 completed as `24001693`; selected-10 texture/perceptual metrics completed as `24007175`; subjective quality accepted by user | Geometry selected-10 slightly improves `CD_Acc`, `chamfer-L1`, `chamfer-L2`, `F_10`, and `F_20`, but worsens `CD_Comp`. Texture/perceptual selected-10 is near-neutral/mixed: slightly improves Y-PSNR and YUV mean, slightly worsens projection SSIM, LPIPS, and PCQM. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/mag_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10/summary_texture_perceptual_metrics.csv` | Candidate for more sequences. |
| GQE-Net | Independent paper/repo search | yes; excluded from main benchmark by user decision on 2026-07-23 | 2023 | GQE-Net: A Graph-based Quality Enhancement Network for Point Cloud Color Attribute | Paper: `https://arxiv.org/abs/2303.13764`; repo: `https://github.com/xjr998/GQE-Net`<br>Publication: IEEE TIP 2023 | Smoke `24000230`, selected-10 `24001694`, and selected-10 texture/perceptual job `24007175` completed before the scope decision; partial full-dataset summaries also exist | GQE-Net is a post-compression color-artifact enhancement method. The paper motivates quality loss from point-cloud compression and reports G-PCC BD-PSNR/BD-rate results for Y, Cb, and Cr. It is therefore outside the main captured CG-to-HE restoration scope even though the existing cross-domain selected-10 run improved color metrics. Preserve those outputs as compression-derived evidence. | Do not submit additional selected-10 or full-dataset runs. Remove it from the main enhancement ranking and mention it only in a compression-artifact or excluded-scope discussion. |
| SuperPC | Independent paper/repo search | yes; user read, included, and added to Overleaf on 2026-07-23 | 2025 | SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization | Paper: `https://openaccess.thecvf.com/content/CVPR2025/html/Du_SuperPC_A_Single_Diffusion_Model_for_Point_Cloud_Completion_Upsampling_CVPR_2025_paper.html`; repo: `https://github.com/sair-lab/SuperPC`; project: `https://sairlab.org/superpc/`<br>Publication: CVPR 2025 | Included in the survey as an explicitly multimodal method; excluded from execution by user decision; no-vision smoke completed earlier as `24066610` | The paper-faithful method is not point-cloud-only. The official ShapeNet, TartanAir, and KITTI-360 evaluation commands set `--use_vision_conditioning true`, and official dataset preparation requires ground-truth point clouds and images. The earlier UVG adapter used `--use_vision_conditioning false` with the TartanAir checkpoint (`11520 -> 46080` points); this is an ablation, not the published configuration. That frame-0000 ablation was negative: `CD_Acc 23.6178 -> 24.6330`, `CD_Comp 26.4884 -> 62.1257`, `chamfer-L1 50.1062 -> 86.7586`, `chamfer-L2 2098.37 -> 6848.23`, `F_10 0.2570 -> 0.1433`, and `F_20 0.4866 -> 0.2849`. | Do not submit selected-10 or full-dataset SuperPC jobs. Keep the paper in the survey/Overleaf as a multimodal unified-restoration reference and preserve the prior no-vision smoke only as an explicitly non-faithful ablation. |

LUMI continuation note (2026-07-13): PathNet compatibility smoke job `19843269`
completed on an NVIDIA A40 in 5m28s for OrangeKettlebell frame `0000`, preserving
`502941 -> 502941` points and transferring CG color with nearest-neighbor `k=1`.
The result is mixed: `CD_Comp 17.43543 -> 17.43266`, `chamfer-L2 1011.44635 ->
1007.85707`, and `F_10 0.45926 -> 0.46043` improve, while `CD_Acc 13.58263 ->
13.58778`, `chamfer-L1 31.01806 -> 31.02044`, and `F_20 0.75228 -> 0.75048`
worsen. The method remains included in the survey, needs no further paper
reading, and still needs subjective inspection. Outputs and per-frame/summary
geometry CSVs are under
`/scratch/project_465003117/PointCloudEnhancement/full_dataset/results`; no PLY
output is committed. Full-dataset status remains 10/12 because this distinct
smoke run does not identify or replace either missing sequence summary.

## External Upsampling Methods

| Method | Source/provenance | Reading paper | Publication year | Paper title | Paper / repo link | Status | Why / observed result | Potential adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meta-PU | LilydotEE survey repo | no | 2021 | Meta-PU: An Arbitrary-Scale Upsampling Network for Point Cloud | `https://github.com/pleaseconnectwifi/Meta-PU`<br>Publication: IEEE TVCG 2021 | Held / waiting for pretrained checkpoint | PyTorch repo cloned under `third_party/enhancement/Meta-PU`. README provides training and test-data links, but no pretrained `.pt`/`.pth`/checkpoint file is included in the repo. It also requires building a local extension with `setup.py`. Under the current benchmark rule, training from scratch is not part of this pass. | Use only if a compatible pretrained Meta-PU checkpoint is found/downloaded; then run with the same selected-10 large-frame adapter policy as other upsampling methods. |
| PUGeo-Net | LilydotEE survey repo | yes | 2020 | PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling | `https://github.com/ninaqy/PUGeo`; PyTorch port: `https://github.com/guochengqian/PUGeoNet_pytorch`<br>Publication: ECCV 2020 | Held / skipped for now | Official repo is TensorFlow; local PyTorch port has code but no downloaded pretrained checkpoint found. | Use only if a compatible pretrained PyTorch checkpoint is found or training is explicitly allowed. |
| PU-GCN / PU-Net / MPU TF checkpoints | LilydotEE survey repo | yes | 2021 / 2018 | PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks; PU-Net: Point Cloud Upsampling Network | PU-GCN: `https://github.com/guochengqian/PU-GCN`; PU-Net paper/repo family: `https://arxiv.org/abs/1801.06761`<br>Publication: CVPR 2021 / CVPR 2018 | Skipped for now; checkpoint ZIPs and extracted TF checkpoints archived/symlinked | User-provided archives remain visible via symlinks under `third_party/enhancement/PU-GCN/`: `pretrained_PU1K.zip` extracted to `pu1k-mpu/`, `pu1k-pugcn/`, and `pu1k-punet/`; `pugan-pugcn.zip` extracted to `pugan-pugcn/`. The physical ZIP files and extracted checkpoint directories were moved to `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/archives/PU-GCN/` to reduce home quota usage. These are TensorFlow checkpoint shards (`model-*.meta`, `.index`, `.data-*`) from the official PU-GCN code path, not PyTorch `.pth` weights for standalone PyTorch ports. Official implementation is TensorFlow 1.x with custom PointNet++ ops, so it remains skipped under the current TF/custom-op rule. Marked for reading because the row includes older PU-Net/MPU-style baselines and a fragile legacy TF setup. | Do not build a TF1 environment now. Reopen only if the user explicitly approves a dedicated TF1/custom-op exception environment or if a maintained PyTorch implementation with compatible pretrained `.pth` checkpoints is found. |
| Flexible-PU / MAPU-Net | LilydotEE survey repo | no | 2021 | Deep Magnification-Flexible Upsampling over 3D Point Clouds | `https://github.com/ninaqy/Flexible-PU`; paper: `https://arxiv.org/abs/2011.12745`<br>Publication: IEEE TIP 2021 | Held / skipped | Official implementation is CUDA 10.0, TensorFlow 1.14, Python 2.7, and custom TF ops. The README has Google Drive links for data and `MAFU_model.tar`, but this violates the current TF/Python2 skip rule. | Revisit only if a maintained PyTorch pretrained implementation is found or user approves a TF/Python2 exception. |
| Dis-PU | LilydotEE survey repo | no | 2021 | Point Cloud Upsampling via Disentangled Refinement | `https://github.com/liruihui/Dis-PU`<br>Publication: CVPR 2021 | Held / skipped | Repo was cloned for inspection. It requires TensorFlow 1.11.1 and custom TF ops; pretrained model is external Google Drive. | Skip under the current TF/custom-op rule unless a maintained PyTorch/pretrained implementation is found. |
| SAPCU | LilydotEE survey repo | **yes — included by user decision** | 2022 | Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation | `https://github.com/xnowbzhao/sapcu`<br>Publication: CVPR 2022 | LUMI repository restored at official commit `84a1249154e3430ea906ad28b8ad8d18f71ab4e2`; checkpoint restoration and LUMI GPU rerun in progress | Official checkpoint release is available through the repository README. The dense-seed implementation limits input point clouds to about 5000 points; adapter `scripts/run_sapcu_selected_frames.py` therefore uses a clearly labeled fixed-size protocol: FPS-sample UVG CG to 2048 points, apply official SAPCU 4x, retain 8192 output points, then transfer RGB from full CG by `k=1`. Earlier Snellius retry `24044830` produced `502,941 -> sampled 2,048 -> 8,192` points. Frame `0000` was negative overall: `CD_Acc 23.6178 -> 32.2786`, `CD_Comp 26.4884 -> 26.6987`, `chamfer-L1 50.1062 -> 58.9774`, `chamfer-L2 2098.37 -> 4471.45`, `F_10 0.2570 -> 0.1971`, `F_20 0.4866 -> 0.4806`; only `R_20` improved due sparse coverage. | Run on LUMI under an immutable `sapcu_4x_2048_*` method name. Preserve the fixed-size/domain-transfer caveat in paper tables; do not represent the 8192-point output as a full-resolution enhancement. |
| ZSPU | LilydotEE survey repo | no | 2022 | "Zero-Shot" Point Cloud Upsampling | `https://github.com/ky-zhou/ZSPU`<br>Publication: ICME 2022 | Held / skipped for now | TensorFlow/custom-op implementation. | Run only as a TF exception. |
| Neural Points | LilydotEE survey repo | yes, user read and included in final 20 | 2022 | Neural Points: Point Cloud Representation with Neural Fields for Arbitrary Upsampling | Paper: `https://arxiv.org/abs/2112.04148`; repo: `https://github.com/WanquanF/NeuralPoints`<br>Publication: CVPR 2022 | Repo cloned; pretrained `v3.pt` included; included in final 20; smoke completed as `24067505`; selected-10 completed as `24076269`; selected-10 texture/perceptual completed as `24079520` | Official repo cloned under `third_party/enhancement/NeuralPoints`. README says the method uses 256-point patches and the released test command runs `testing_up_ratio=16` with `over_sampling_scale=4`. The adapter `scripts/run_neuralpoints_selected_frames.py` keeps that patch shape: FPS-sample UVG CG to 2048 points, split into eight Morton-ordered 256-point patches, run released `Net_conpu_v7` with `pre_trained/v3.pt`, concatenate 32,768 output points, and transfer RGB from CG by `k=1`. Compatibility patches make unused `igl`/`torch_scatter` imports optional and use PyTorch3D fallbacks for PointNet++ FPS/gather instead of compiling the old CUDA extension. Frame `0000` is mixed but not a clear pass: `CD_Comp` improves `26.4884 -> 25.8369`, `chamfer-L1` improves `50.1062 -> 49.9662`, and `F_20` improves `0.4866 -> 0.5014`, but `CD_Acc` worsens `23.6178 -> 24.1292`, `chamfer-L2` worsens `2098.37 -> 2197.85`, and `F_10` worsens `0.2570 -> 0.2472`. Selected-10 is mixed/positive on mean geometry: `CD_Comp 27.6065 -> 27.1829`, `chamfer-L1 53.1241 -> 52.9137`, `F_10 0.2459 -> 0.2503`, and `F_20 0.4552 -> 0.4810` improve, but `CD_Acc 25.5177 -> 25.7307` and `chamfer-L2 2293.13 -> 2617.23` worsen. Selected-10 texture/perceptual is mostly negative except PCQM: Y-PSNR worsens `18.1194 -> 17.4715`, YUV mean worsens `21.1708 -> 20.8853`, projection SSIM worsens `0.6190 -> 0.5803`, LPIPS worsens `0.4684 -> 0.6882`, while PCQM improves `0.1147 -> 0.1196`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/neuralpoints_16x_2048_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_neuralpoints/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/neuralpoints_16x_2048_selected10/OrangeKettlebell/15fps/`; selected-10 job: `jobs/neuralpoints_orangekettlebell_selected10.slurm`; texture job: `jobs/texture_metrics_orangekettlebell_selected10_neuralpoints.slurm`. | Included in final 20. Keep clearly labeled as fixed-size patch/object-domain upsampling because each frame is reduced to 2048 input points before 16x output; projection/perceptual texture metrics drop despite mixed/positive geometry and PCQM. |
| PU-Transformer | Independent paper/repo search | no | 2022 | PU-Transformer: Point Cloud Upsampling Transformer | Official paper/repo: `https://github.com/ShiQiu0419/PU-Transformer`; unofficial PyTorch repo inspected: `https://github.com/rhtm02/PU-Transformer`; paper: `https://arxiv.org/abs/2111.12242`<br>Publication: ACCV 2022 | Blocked / no pretrained checkpoint in inspected code | Official repo states code is to be updated. The unofficial PyTorch repo was cloned under `third_party/enhancement/PU-Transformer` and includes model/test/train code, but no `.pth`/`.pt` checkpoint and no model-download link in the README. Its `test.py` expects `./k-20_putransformer_6900/best_model.pth`, which is not present. Under the current no-training rule, no smoke run was submitted. | Revisit only if a compatible pretrained `best_model.pth` is found or user approves training. |
| PUCRN | Independent paper/repo search | yes, read and suitable | 2022 | Point Cloud Upsampling via Cascaded Refinement Network | Paper: `https://arxiv.org/abs/2210.03942`; repo: `https://github.com/hikvision-research/3DVision/tree/main/PointUpsampling/PUCRN`<br>Publication: ACCV 2022 | Smoke completed as `24069996`; selected-10 completed as `24070009`; selected-10 texture/perceptual completed as `24071151` | Official PyTorch implementation cloned under `third_party/enhancement/3DVision/PointUpsampling/PUCRN`. The release checkpoint `model/release/model.pth` is included and loads as `net_state_dict`. Adapter `scripts/run_pucrn_selected_frames.py` follows the official 4x inference shape with 2048-point Morton chunks, normalizes each chunk, runs `CRNet(up_ratio=4)`, denormalizes, writes UVG-compatible PLY, and transfers RGB from CG by `k=1`. This allows dense UVG frames with about 0.5-0.6M points to be processed as many local 2048-point patches rather than reducing the full frame to 1024/2048 total points. Compatibility patch replaces old PointNet++ extension calls with PyTorch3D/PyTorch inference fallbacks for FPS, gather, and grouping, and prevents an unused extension JIT path in `network/operations.py`. Selected-10 is positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.4838`, `CD_Comp 27.6065 -> 26.5389`, `chamfer-L1 53.1241 -> 52.0227`, `chamfer-L2 2293.13 -> 2217.32`, `F_10 0.2459 -> 0.2543`, and `F_20 0.4552 -> 0.4646`. Texture/perceptual selected-10 is also positive: `dYUV-PSNR +0.0036`, `dProj-SSIM +0.0189`, `dLPIPS +0.0431` where lower LPIPS is better, and `dPCQM +0.0001`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pucrn_pu1k_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pucrn/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/pucrn_pu1k_4x_selected10/OrangeKettlebell/15fps/`. | Strong candidate upsampling method for the survey benchmark. Inspect subjectively and consider for more sequences / full-dataset run. |
| PC2-PU | Independent paper/repo search | yes, user included in final 20 | 2022 | PC2-PU: Patch Correlation and Position Correction for Effective Point Cloud Upsampling | Paper: `https://arxiv.org/abs/2109.09337`; repo: `https://github.com/chenlongwhu/PC2-PU`; checkpoint: `https://drive.google.com/file/d/1CebnBUtX2OsoPnBNtquUVfZmgqRQPfhm/view?usp=sharing`<br>Publication: ACM MM 2022 | Repo cloned; checkpoint downloaded; first smoke `24077001` failed at final chunk shape; smoke retry completed as `24077563`; selected-10 completed as `24079519`; texture/perceptual completed as `24081137` | Official PyTorch implementation cloned under `third_party/enhancement/PC2-PU`. The checkpoint is local at `third_party/enhancement/PC2-PU/log/PC2-PU/model_best.pth.tar` and loads cleanly with zero missing/unexpected state-dict keys. The official code expects `KNN_CUDA` and `pointnet2_ops`; local inference-only fallbacks were added for KNN, FPS, and grouping. Adapter `scripts/run_pc2pu_selected_frames.py` runs the pretrained model on duplicated 256-point Morton chunks so the paper's pair/correlation path remains shape-compatible, merges 4x geometry outputs, writes UVG-compatible PLY, and transfers RGB from CG by nearest-neighbor `k=1`. First smoke finished almost all chunks but failed because the final merged chunk had a larger shape; the adapter now pads only the final short chunk by repeating existing indices to keep 256-point input shape. Smoke retry on frame `0000` is mixed/positive: `CD_Acc 23.6178 -> 23.5545`, `chamfer-L1 50.1062 -> 50.0587`, `F_10 0.2570 -> 0.2576`, and `F_20 0.4866 -> 0.4880` improve, while `CD_Comp 26.4884 -> 26.5042` and `chamfer-L2 2098.37 -> 2114.17` worsen. Selected-10 geometry is mixed/slightly positive: `CD_Acc 25.5177 -> 25.4566`, `chamfer-L1 53.1241 -> 53.1004`, `F_10 0.2459 -> 0.2465`, and `F_20 0.4552 -> 0.4556` improve, while `CD_Comp 27.6065 -> 27.6437` and `chamfer-L2 2293.13 -> 2300.14` worsen. Selected-10 texture/perceptual is positive overall: `YUV-PSNR 21.1708 -> 21.1743`, projection SSIM `0.6190 -> 0.6324`, LPIPS `0.4684 -> 0.4183` where lower is better, and PCQM `0.1147357 -> 0.1147413` improve. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pc2pu_4x_chunks256_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pc2pu/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/pc2pu_4x_chunks256_selected10/OrangeKettlebell/15fps/`. | Included in final 20. Objective result is weak-positive/mixed geometry and positive texture/perceptual; inspect subjectively before broader full-dataset execution. |
| SPU-Net | Independent paper/repo search | no | 2022 | SPU-Net: Self-Supervised Point Cloud Upsampling by Coarse-to-Fine Reconstruction with Self-Projection Optimization | Repo: `https://github.com/liuxinhai/SPU-Net`<br>Publication: IEEE TIP 2022 | Skipped under current TF rule | Official repo cloned under `third_party/enhancement/SPU-Net`. It is TensorFlow 1.x with custom TF ops and does not include a ready pretrained checkpoint in the repo. | Revisit only if a maintained PyTorch implementation with compatible pretrained weights is found or the user approves a dedicated TensorFlow exception. |
| PU-EVA | Independent paper/repo search | no | 2021 | PU-EVA: An Edge-Vector Based Approximation Solution for Flexible-Scale Point Cloud Upsampling | Repo: `https://github.com/GabrielleTse/PU-EVA`; checkpoint folder: `https://drive.google.com/drive/folders/1UEVm1SHUi-3O9x-T004jimEH6U__BfkZ`<br>Publication: ICCV 2021 | Skipped under current TF rule | Official repo cloned under `third_party/enhancement/PU-EVA`. It is TensorFlow/custom-op based and pretrained models are external. | Revisit only if a maintained PyTorch implementation with compatible pretrained weights is found or the user approves a dedicated TensorFlow exception. |
| PUFA-GAN | Independent paper/repo search | no | 2022 | PUFA-GAN: A Frequency-Aware Generative Adversarial Network for 3D Point Cloud Upsampling | Repo: `https://github.com/yuanhui0325/PUFA-GAN`<br>Publication: IEEE TIP 2022 | Skipped under current TF rule | Official repo cloned under `third_party/enhancement/PUFA-GAN` and includes pretrained TensorFlow checkpoint shards under `model/`, but the README targets TensorFlow 1.11, Python 3.6, and compiled PointNet++ TF operators. | Revisit only if the user approves a TF1/custom-op exception or a maintained PyTorch checkpoint path is found. |
| SSPU-Net | LilydotEE survey repo | no | 2021 | Self-Supervised Point Cloud Upsampling via Differentiable Rendering | Official listed repo: `https://github.com/fpthink/SSPU-Net`; paper: `https://arxiv.org/abs/2108.00454`<br>Publication: ACM MM 2021 | Blocked / no released code | Official GitHub repo only has a README saying code would be released later; no implementation or pretrained checkpoint is available. A similarly named mirror found online also contains only a README. | Skip unless a complete implementation with pretrained weights appears. |
| BIMS-PU | LilydotEE survey repo | no | 2022 | BIMS-PU: Bi-Directional and Multi-Scale Point Cloud Upsampling | Paper: `https://arxiv.org/abs/2206.12648`<br>Publication: IEEE RA-L 2022 | Held / no usable open-source implementation found | Survey-listed method. Web search found the paper but no official runnable GitHub repository with pretrained weights. | Revisit only if code/checkpoint is located. |
| SPU | LilydotEE survey repo | no | 2022 | Semantic Point Cloud Upsampling | `https://github.com/lizhuangzi/SPU`<br>Publication: IEEE TMM 2022 | 4x smoke completed; 4x selected-10 completed as `24032965`; texture/perceptual completed; 2x/8x scale smoke completed incidentally; 16x scale smoke cancelled | PyTorch repo cloned under `third_party/enhancement/SPU`; pretrained 2x/4x/8x/16x upsampling `.parm` weights are included in `savedModel/`, with classification priors in `PretrainModel/`. Official environment is Python 3.6, PyTorch 1.2.0, and `KNN_CUDA`. Adapter uses the included PointNet classifier prior, PyTorch3D/PyTorch fallbacks for unavailable `knn_cuda` and old PointNet++ grouping ops, Morton-ordered chunks, native 4x geometry output, and CG-to-output RGB transfer by `k=1`. Selected-10 is mixed/positive: improves `CD_Comp`, Chamfer-L1/L2, recall, and F-scores, but slightly worsens `CD_Acc` and precision. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0029`, `dProj-SSIM +0.0155`, `dLPIPS +0.0471`, `dPCQM +0.0002`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/spu_pointnet_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_upsampling_candidates_retry/summary_texture_perceptual_metrics.csv`. 2x/8x/16x are not counted as separate methods; they are scale variants only. | Candidate upsampling method. Inspect subjectively and decide whether SPU should be represented by native 4x or a same-count adapter. |
| PU-Flow | LilydotEE survey repo | no | 2022 | PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows | Paper: `https://arxiv.org/abs/2107.05893`; repo verified from paper/README: `https://github.com/unknownue/puflow`<br>Publication: IEEE TVCG 2022 | Smoke completed as retry `24032579`; selected-10 completed as `24032682`; selected-10 texture/perceptual completed as `24067001` | Official PyTorch repo cloned under `third_party/enhancement/puflow`; pretrained discrete x4 checkpoints are included in `pretrain/`. The LilydotEE survey README lists PU-Flow but appears to point its code link to the SPU repo, so this row records the verified PU-Flow repo separately. Current adapter `scripts/run_puflow_selected_frames.py` uses the official discrete `puflow-x4-pugeo.pt` checkpoint, re-enables the repo's own patch-batching path, replaces unavailable `knn_cuda` and `pointnet2_ops` with PyTorch3D fallbacks, fixes removed NumPy `np.long`, splits dense UVG frames into Morton-ordered chunks, returns the same total point count as CG via the official `num_out` behavior, and transfers RGB from CG by `k=1`. Selected-10 is mostly positive in geometry: `CD_Acc 25.5177 -> 25.1997`, `CD_Comp 27.6065 -> 25.4816`, `chamfer-L1 53.1241 -> 50.6813`, `F_10 0.2459 -> 0.2591`, and `F_20 0.4552 -> 0.4789`; `chamfer-L2` worsened `2293.13 -> 2876.23`. Texture/perceptual is mixed: `dYUV-PSNR -0.1000`, `dProj-SSIM +0.2441`, `dLPIPS +0.2752`, and `dPCQM -0.0142`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/puflow_discrete_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_puflow/summary_texture_perceptual_metrics.csv`; aggregate summary: `skills/methods/SELECTED10_RESULTS_SUMMARY.md`. | Candidate upsampling method by geometry and projection perceptual metrics, but inspect subjectively because YUV-PSNR and PCQM drop. |
| Grad-PU | Independent paper/repo search | no | 2023 | Arbitrary-Scale Point Cloud Upsampling by Voxel-Based Network With Latent Geometric-Consistent Learning / Grad-PU repo candidate | `https://github.com/yunhe20/Grad-PU`<br>Publication: CVPR 2023 | Official full-frame inference failed/timed out; non-default chunked smoke `24046203` completed; selected-10 completed as `24046992`; texture/perceptual completed | Official PyTorch repo cloned locally with pretrained `.pth` files. Compatibility fallback uses PyTorch3D for KNN/FPS because the original custom `pointops` extension uses the removed PyTorch `THC` API. A100 full-frame UVG smoke `24006869` ran for about 3 hours and failed with CUDA OOM while trying to allocate ~23 GiB inside feature extraction; no enhanced PLY was written. First H100 retry `24013338` ran 01:58:37 and failed because the benchmark wrapper incorrectly called the official gradient-based inference inside `torch.no_grad()`, causing `loss.backward()` to fail. Wrapper fix: `scripts/run_gradpu_selected_frames.py` now allows gradients inside `pcd_upsample`. Corrected H100 retry `24028380` timed out at 04:00:27 with no output PLY or metric files, so the official full-frame path is not practical for UVG dense frames. A clearly labeled non-default large-scene wrapper was added to `scripts/run_gradpu_selected_frames.py` with `--chunk-size`; smoke job `jobs/gradpu_orangekettlebell_0000_chunked_h100.slurm` runs 2048-point chunks and transfers RGB from CG by `k=1`. Chunked smoke is positive on most frame `0000` metrics: `CD_Acc 23.6178 -> 23.5218`, `CD_Comp 26.4884 -> 26.1939`, `chamfer-L1 50.1062 -> 49.7157`, `chamfer-L2 2098.37 -> 2076.06`, `F_10 0.2570 -> 0.2594`, and `F_20 0.4866 -> 0.4902`; only `P_5` slightly worsens. Selected-10 is positive on mean geometry metrics: `CD_Acc 25.5177 -> 25.4108`, `CD_Comp 27.6065 -> 27.3596`, `chamfer-L1 53.1241 -> 52.7703`, `chamfer-L2 2293.13 -> 2278.19`, `F_10 0.2459 -> 0.2485`, and `F_20 0.4552 -> 0.4588`. Texture/perceptual is mixed/positive: `dYUV-PSNR +0.0078`, `dProj-SSIM +0.0151`, `dLPIPS +0.0482`, `dPCQM -0.0000`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/gradpu_chunked_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pufm_gradpu/summary_texture_perceptual_metrics.csv`; job: `jobs/gradpu_orangekettlebell_selected10_chunked_h100.slurm`. Logs: `logs/gradpu_ok_0000_smoke_24006869.out`, `logs/gradpu_h100_0000_smoke_24013338.out`, `logs/gradpu_h100_0000_smoke_24028380.out`. | Candidate only if clearly labeled as a non-default large-frame adaptation; keep official full-frame Grad-PU recorded as failed. |
| Joint upsampling and cleaning with Octree CNNs | Independent paper/repo search | no | 2026 | Joint Point Cloud Upsampling and Cleaning with Octree-based CNNs | `https://github.com/octree-nn/upsample-clean`<br>Publication: Computational Visual Media 2026 | Smoke completed as retry `24030983`; selected-10 completed as `24031002`; texture/perceptual completed as `24031073` | Modern PyTorch repo; method is relevant because it jointly upsamples and cleans. Official checkpoint downloaded via Google Drive to `third_party/enhancement/upsample-clean/logs/puc/checkpoints/ounet.pth`. Method-scoped local deps installed under `third_party/enhancement/upsample-clean/python_deps` for `ocnn==2.3.2`, `thsolver==1.2.1`, and `yacs==0.1.8` without mutating `torch_env`. The public checkpoint does not match the checked-in YAML; exact load requires the inferred checkpoint architecture `depth=8`, `full_depth=2`, `channels=[0,0,256,256,128,128,64,64,32]`. Adapter normalizes UVG frame to fit the official octree input range, runs the model, denormalizes, and transfers RGB by `k=1`. First smoke `24030942` failed before inference because the manual adapter did not call `octree.construct_all_neigh()` after `build_octree`; second `24030964` failed because neighbor construction mixed CUDA points with CPU OCNN lookup tables. Third smoke fixed those wrapper issues. Selected-10 is positive on all geometry metrics despite reducing each dense frame from about 0.5-0.6M points to about 47k-60k points: `dCD_Acc +2.2532`, `dCD_Comp +0.4222`, `dChamfer-L1 +2.6754`, `dF_10 +0.0386`, `dF_20 +0.0385`. Texture/perceptual is mixed/negative from sparsity: `dYUV-PSNR +0.0830`, but `dProj-SSIM -0.0844`, `dLPIPS -0.2048`, `dPCQM -0.0093`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/upsample_clean_ounet_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10/summary_texture_perceptual_metrics.csv`; aggregate summary updated in `skills/methods/SELECTED10_RESULTS_SUMMARY.md`. | Strong geometry candidate but requires subjective inspection because the output is much sparser and perceptual projection metrics drop. Decide whether to keep as geometry-focused joint upsampling/cleaning baseline or paper-review hold. |
| PU-Refiner | LilydotEE survey repo | no | 2022 | PU-Refiner: A Geometry Refiner with Adversarial Learning for Point Cloud Upsampling | `https://github.com/liuhaoyun/PU-Refiner`<br>Publication: ICASSP 2022 | Unavailable / no runnable code | Repo cloned under `third_party/enhancement/PU-Refiner`, but it contains only a README saying the source code will be released soon. No implementation or checkpoint is available in that repo. | Skip unless another complete open-source implementation with pretrained weights is found. |
| PU-Gaussian | Independent paper/repo search | no | 2025 | PU-Gaussian: Point Cloud Upsampling using 3D Gaussian Representation | `https://github.com/mvg-inatech/PU-Gaussian`<br>Publication: ICCV 2025 e2e3D Workshop | Smoke completed as `24033365`; selected-10 completed as `24033428`; texture/perceptual completed | Official PyTorch repo cloned under `third_party/enhancement/PU-Gaussian`; pretrained PU1K and PUGAN checkpoints are included. Adapter `scripts/run_pu_gaussian_selected_frames.py` calls the official `infer.py`, uses the PU1K checkpoint first, writes UVG-compatible PLY, and transfers RGB from CG by `k=1`. Compatibility fallbacks were added for unavailable `pointops_cuda` and Chamfer3D extensions using PyTorch3D/PyTorch. Frame `0000` smoke is positive on all reported geometry metrics. Selected-10 is also positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.1519`, `CD_Comp 27.6065 -> 27.4463`, `chamfer-L1 53.1241 -> 52.5982`, `chamfer-L2 2293.13 -> 2272.05`, `F_10 0.2459 -> 0.2506`, `F_20 0.4552 -> 0.4612`. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0164`, `dProj-SSIM +0.0199`, `dLPIPS +0.0660`, `dPCQM +0.0003`. Smoke metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pu_gaussian_pu1k_4x/summary_metrics.csv`; selected-10 metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pu_gaussian_pu1k_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_upsampling_candidates_retry/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/pu_gaussian_pu1k_4x_selected10/OrangeKettlebell/15fps/`. | Candidate upsampling method; inspect subjective output. |
| SnowflakeNet-PU | Independent paper/repo search | no | 2021 / 2023 | SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer | `https://github.com/AllenXiangX/SnowflakeNet`<br>Publication: ICCV 2021; TPAMI 2023 extension | First smoke `24033372` failed before inference; retry completed as `24033399`; selected-10 completed as `24034309`; texture/perceptual completed | This is the official SnowflakeNet point-cloud upsampling branch, separate from the completion-domain-transfer SnowflakeNet row. Local pretrained PU checkpoint `third_party/enhancement/SnowflakeNet/pretrained/pu/ckpt-pu.pth` loads with its `model` state dict. First adapter tried to import `PU.utils`, which unnecessarily imported the Chamfer CUDA training loss and failed to JIT because `thrust/complex.h` is unavailable. Safer compatibility patch now makes Chamfer import lazy in `PU/utils.py`, so the adapter uses the official `PU.utils.patch_extraction()`. Selected-10 is positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.2377`, `CD_Comp 27.6065 -> 27.1780`, `chamfer-L1 53.1241 -> 52.4157`, `chamfer-L2 2293.13 -> 2248.38`, `F_10 0.2459 -> 0.2509`, `F_20 0.4552 -> 0.4616`. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0175`, `dProj-SSIM +0.0215`, `dLPIPS +0.0645`, `dPCQM +0.0001`. Smoke metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/snowflakenet_pu_4x/summary_metrics.csv`; selected-10 metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/snowflakenet_pu_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_upsampling_candidates_retry/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/snowflakenet_pu_4x_selected10/OrangeKettlebell/15fps/`. | Candidate upsampling method; inspect subjective output. |
| PU-Dense | Independent paper/repo search | no | 2022 | PU-Dense: Sparse Tensor-based Point Cloud Geometry Upsampling | `https://github.com/aniqueakhtar/PointCloudUpsampling`; checkpoint link from README: `https://umkc.box.com/s/pohfuxkojai8yqc236nw6ncqzw1qyqm8`<br>Publication: IEEE TIP 2022 | Checkpoint available; CPU-only import/model-load passes; GPU benchmark blocked | Repo cloned under `third_party/enhancement/PU-Dense`. User-provided checkpoint archive `third_party/enhancement/PU-Dense/ckpts/pretrained_model-selected.zip` was extracted on 2026-06-22. Expected official paths now exist: `third_party/enhancement/PU-Dense/ckpts/4x_0x_ks5/iter64000.pth` and `third_party/enhancement/PU-Dense/ckpts/8x_0x_ks5/iter64000.pth`. The first home env `/home/xzhou/.conda/envs/pudense` was removed after `MinkowskiEngine==0.5.4` failed against PyTorch 1.13/CUDA 11.7. A README-matched env was created on project storage at `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/envs/pudense_mink` with Python 3.7.9, PyTorch 1.6.0/CUDA 10.2 runtime packages, Open3D 0.13.0, and `MinkowskiEngine==0.4.3` built CPU-only from patched source. With `LD_PRELOAD=/usr/lib64/libstdc++.so.6` and Torch lib path in `LD_LIBRARY_PATH`, `import MinkowskiEngine` succeeds and `MyNet(last_kernel_size=5)` loads the 4x checkpoint with zero missing/unexpected keys. The official Box checkpoint link was checked on 2026-06-21 and returned HTTP 404, but the manual checkpoint is now present. | GPU selected-10 remains blocked on Snellius because current CUDA modules are 12.x while this official stack needs CUDA 10.x-compatible nvcc/MinkowskiEngine. Run at most a tiny CPU smoke unless a compatible GPU sparse-conv build path is provided. |
| RepKPU | Independent paper/repo search | no | 2024 | RepKPU: Point Cloud Upsampling with Kernel Point Representation and Deformation | Paper: `https://openaccess.thecvf.com/content/CVPR2024/papers/Rong_RepKPU_Point_Cloud_Upsampling_with_Kernel_Point_Representation_and_Deformation_CVPR_2024_paper.pdf`; repo: `https://github.com/EasyRy/RepKPU`<br>Publication: CVPR 2024 | First smoke `24034082` failed before output; retry completed as `24034092`; selected-10 completed as `24034308`; texture/perceptual completed | Official PyTorch repo cloned under `third_party/enhancement/RepKPU`. PU1K pretrained weights were downloaded from the official Google Drive folder listed in the README to `third_party/enhancement/RepKPU/pretrain/`; current smoke uses `pretrain/exp1/ckpt-best.pth` because its training log reports the lowest best CD among the downloaded PU1K runs. Adapter `scripts/run_repkpu_selected_frames.py` follows the official test path: normalize chunk, FPS seeds, KNN patches, model forward, FPS to 4x, denormalize, and transfer RGB by `k=1`. First smoke failed because the missing `pointops_cuda` fallback did not rebind the old `Function.apply` symbols; retry includes explicit PyTorch3D/PyTorch fallback rebindings. Chamfer3D is optional at inference import. Selected-10 is positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.3282`, `CD_Comp 27.6065 -> 27.2268`, `chamfer-L1 53.1241 -> 52.5550`, `chamfer-L2 2293.13 -> 2264.64`, `F_10 0.2459 -> 0.2495`, `F_20 0.4552 -> 0.4603`. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0236`, `dProj-SSIM +0.0196`, `dLPIPS +0.0615`, `dPCQM +0.0003`. Smoke metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/repkpu_pu1k_4x/summary_metrics.csv`; selected-10 metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/repkpu_pu1k_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_upsampling_candidates_retry/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/repkpu_pu1k_4x_selected10/OrangeKettlebell/15fps/`. | Candidate upsampling method; inspect subjective output. If a future checkpoint cannot be downloaded automatically, list the URL and expected local path and ask the user to place it manually. |
| PUDM | Independent paper/repo search | yes, user read and included in survey | 2024 | A Conditional Denoising Diffusion Probabilistic Model for Point Cloud Upsampling | Paper/repo: `https://github.com/QWTforGithub/PUDM`; checkpoint folder: `https://drive.google.com/drive/folders/1XIgLSpAPmt_Zjn9SSBF4EWSCyiHF6ByZ?usp=sharing`<br>Publication: CVPR 2024 | Smoke completed as `24035011` on `gpu_h100`; selected-10 completed as `24046202`; selected-10 texture/perceptual completed as `24056847` | Official PyTorch repo cloned under `third_party/enhancement/PUDM`. PU1K checkpoint is local at `third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl`. Adapter `scripts/run_pudm_selected_frames.py` follows the official PU1K config, uses 2048-point Morton chunks because PU1K training uses 2048 sparse input points, runs 30-step DDIM with `R=4`, denormalizes each chunk, writes UVG-compatible PLY, and transfers RGB from CG by `k=1`. Compatibility patches in `pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py` and `pointops/functions/pointops.py` add PyTorch3D/PyTorch inference fallbacks and `PCE_FORCE_PUDM_FALLBACKS=1` to avoid old CUDA/THC extension builds. Smoke runtime was 00:09:29 on H100, producing `502,941 -> 2,011,764` points. Frame `0000` is mixed/mostly negative. Selected-10 is also mixed/mostly negative in geometry: `CD_Acc` slightly improves `25.5177 -> 25.4931`, but `CD_Comp` worsens `27.6065 -> 28.1901`, `chamfer-L1` worsens `53.1241 -> 53.6832`, `chamfer-L2` worsens `2293.13 -> 2349.26`, `F_10` worsens `0.2459 -> 0.2438`, and `F_20` worsens `0.4552 -> 0.4508`. Texture/perceptual selected-10 is mixed/positive: `dYUV-PSNR +0.0186`, `dProj-SSIM +0.0110`, `dLPIPS +0.0433`, but `dPCQM -0.0001`. Smoke metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pudm_pu1k_4x/summary_metrics.csv`; selected-10 geometry: `results/uvg_cwi_dqpc/OrangeKettlebell/pudm_pu1k_4x_selected10/summary_metrics.csv`; texture: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pudm_spupmd/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/pudm_pu1k_4x_selected10/OrangeKettlebell/15fps/`. | Included in survey as the conditional diffusion upsampling comparison. Keep the benchmark result as mixed/mostly negative on geometry but mixed-positive on texture/perceptual. |
| PUFM | Independent paper/repo search | yes, user read and included in survey | 2025 | Efficient Point Clouds Upsampling via Flow Matching | Paper/repo: `https://github.com/Holmes-Alan/PUFM`; paper: `https://arxiv.org/abs/2501.15286`<br>Publication: arXiv 2025 / AAAI 2026 listing found online | Repo cloned; pretrained checkpoints included; first smoke `24045273` failed before output; retry completed as `24045320`; selected-10 completed as `24045541`; texture/perceptual completed | Official PyTorch repo cloned under `third_party/enhancement/PUFM`. Included checkpoints `pretrained_model/pufm.pth` and `pretrained_model/pufm_w_attn.pth` load with zero missing/unexpected keys. Adapter `scripts/run_pufm_selected_frames.py` uses the official midpoint interpolation and five-step flow update with the `pufm` checkpoint, chunks UVG dense frames into 2048-point patches, writes UVG-compatible PLY, and transfers RGB from CG by `k=1`. Compatibility patch in `models/pointops/functions/pointops.py` adds PyTorch3D/PyTorch inference fallbacks and `PCE_FORCE_PUFM_FALLBACKS=1` for the old `pointops_cuda` extension. First smoke failed because fallback `ballquery` returned `int32` indices to `torch.gather`; fixed fallback to return `int64`. Frame `0000` retry is mixed/positive: `CD_Acc` worsens `23.6178 -> 24.1800`, but `CD_Comp` improves `26.4884 -> 25.5348`, `chamfer-L1` improves `50.1062 -> 49.7148`, `chamfer-L2` improves `2098.37 -> 2082.66`, `F_10` improves `0.2570 -> 0.2606`, and `F_20` improves `0.4866 -> 0.4942`. Selected-10 is mixed/positive: worsens `CD_Acc 25.5177 -> 26.1718` and precision, but improves `CD_Comp 27.6065 -> 26.5879`, `chamfer-L1 53.1241 -> 52.7597`, `chamfer-L2 2293.13 -> 2272.52`, `F_10 0.2459 -> 0.2476`, and `F_20 0.4552 -> 0.4586`. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0369`, `dProj-SSIM +0.0270`, `dLPIPS +0.0418`, `dPCQM +0.0016`. Smoke metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pufm_pugan_4x/summary_metrics.csv`; selected-10 metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pufm_pugan_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pufm_gradpu/summary_texture_perceptual_metrics.csv`; Slurm: `jobs/pufm_orangekettlebell_selected10.slurm`. | Included in survey as the flow-matching upsampling comparison; inspect subjectively and consider for more sequences. |
| PUFM++ / Enhanced_PUFM | Independent paper/repo search | no | 2025 | PUFM++: Point Cloud Upsampling via Enhanced Flow Matching | Repo: `https://github.com/Holmes-Alan/Enhanced_PUFM`; paper: `https://arxiv.org/abs/2501.15286`; checkpoint link from README: `https://drive.google.com/file/d/1vjqawPpqCArqulmBK2kcdeeMQf_P-Bnk/view?usp=sharing`<br>Publication: arXiv 2025 | Blocked / official repo incomplete for inference | Official PyTorch repo cloned under `third_party/enhancement/Enhanced_PUFM`. The README checkpoint was downloaded successfully to `third_party/enhancement/Enhanced_PUFM/pretrained_model/enhanced_pufm_pretrained.zip`, and `torch.load(..., map_location='cpu')` confirms it is a valid Lightning checkpoint with a `state_dict`. However, the official inference scripts import `models.utils_ed.Batched_ED`, but `models/utils_ed.py` is absent from the cloned repo and from the checkpoint package. The official scripts also load `t_sched_cdf.pt`, which is absent from the repo and checkpoint package. No smoke run submitted because replacing the missing earth-mover-distance/schedule pieces would change the method rather than reproduce it. | Ask authors/check issues or locate a complete release containing `models/utils_ed.py` and `t_sched_cdf.pt`; otherwise keep this as blocked despite having the checkpoint. |
| SPU-PMD | Independent paper/repo search | no | 2024 | SPU-PMD: Self-Supervised Point Cloud Upsampling via Progressive Mesh Deformation | Paper/repo: `https://github.com/lyz21/SPU-PMD`; PU1K checkpoint: `https://drive.google.com/file/d/1v26YqHQ3CKZjSFOS3zW9F_SooAg5iyI9/view?usp=drive_link`<br>Publication: CVPR 2024 | Repo cloned; PU1K checkpoint downloaded; first smoke `24045723` failed before model load; second smoke `24045790` failed before model load; third smoke `24045823` failed before model load; fourth smoke `24045863` completed; selected-10 completed as `24046991`; selected-10 texture/perceptual completed as `24056847` | Official PyTorch repo cloned under `third_party/enhancement/SPU-PMD`. PU1K checkpoint is local at `third_party/enhancement/SPU-PMD/pretrained/spupmd_pu1k_pretrained` and loads as `net_state_dict`. Adapter `scripts/run_spupmd_selected_frames.py` follows the official test path: normalize chunk, FPS seeds, KNN patches of 256 points, model forward, concatenate patches, FPS to 4x, denormalize, write UVG-compatible PLY, and transfer RGB from CG by `k=1`. Compatibility patch in `pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py` adds PyTorch inference fallbacks and `PCE_FORCE_SPUPMD_FALLBACKS=1` to avoid old PointNet++ CUDA extension builds. First smoke failed because `SPUPMD.py` imported the unused `up_mesh` path, which triggered missing `pyvista`; second smoke showed `Upsampling_unit` does use `up_mesh`. `MeshUtil.py` now treats `pyvista` as optional because the official default `up_mesh.Upsampling(mesh_method_name='ball')` path uses Open3D ball-pivoting, not PyVista alpha meshing. Third smoke failed because `network/operations.py` still tried the old PointNet++ JIT path; patched that file to respect `PCE_FORCE_SPUPMD_FALLBACKS=1`. Fourth smoke is positive on all reported frame `0000` geometry metrics. Selected-10 is positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.1333`, `CD_Comp 27.6065 -> 27.4807`, `chamfer-L1 53.1241 -> 52.6140`, `chamfer-L2 2293.13 -> 2264.00`, `F_10 0.2459 -> 0.2493`, and `F_20 0.4552 -> 0.4601`. Texture/perceptual selected-10 is also mostly positive: `dYUV-PSNR +0.0132`, `dProj-SSIM +0.0169`, `dLPIPS +0.0566`, but `dPCQM -0.0003`. Smoke job: `jobs/spupmd_orangekettlebell_0000_smoke.slurm`; selected-10 job: `jobs/spupmd_orangekettlebell_selected10.slurm`; selected-10 geometry: `results/uvg_cwi_dqpc/OrangeKettlebell/spupmd_pu1k_4x_selected10/summary_metrics.csv`; texture: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pudm_spupmd/summary_texture_perceptual_metrics.csv`; output: `results/method_outputs/spupmd_pu1k_4x_selected10/OrangeKettlebell/15fps/`. | Candidate upsampling method; inspect subjectively and consider for more sequences. |
| SPU-IMR | Independent paper/repo search | no | 2025 | SPU-IMR: Self-supervised Arbitrary-scale Point Cloud Upsampling via Iterative Mask-recovery Network | Paper/repo: `https://github.com/hapifuzi/spu-imr`; paper: `https://arxiv.org/abs/2502.19452`; **Checkpoint: NA — official link unavailable**<br>Publication: AAAI 2025 | Skipped / checkpoint **NA** | Official PyTorch code is available and cloned under `third_party/enhancement/spu-imr`, but the pretrained checkpoint is not available. The user rechecked the authors' Baidu Pan URL on 2026-07-21 and confirmed that it is unavailable. No checkpoint is present locally, so the method cannot be run under the current no-training benchmark rule. Do not present this as code unavailable: code is available; checkpoint status is **NA**. | Pass. Revisit only if the authors restore an official checkpoint or release another compatible pretrained model. |
| APU-LDI | Independent paper/repo search | yes, user included in survey | 2024 | Learning Continuous Implicit Field with Local Distance Indicator for Arbitrary-Scale Point Cloud Upsampling | Paper: `https://arxiv.org/abs/2312.15133`; repo: `https://github.com/lisj575/APU-LDI`; pretrained folder: `https://drive.google.com/drive/folders/1SgebUP_9JupIGsRpemHer8koTTucgSQh?usp=drive_link`<br>Publication: AAAI 2024 | Included by user decision; local-LDI checkpoint isolated; smoke completed as `24161095`; not expanded | Official PyTorch repo cloned under `third_party/enhancement/APU-LDI`. README targets Python 3.7.12, PyTorch 1.7.1+cu110, Open3D 0.17, and compiled Chamfer3D/pointops. The Drive folder is public but contains full datasets, test point clouds, per-shape global-field checkpoints, local LDI checkpoints, and test results in one large recursive tree. A recursive pull was avoided; folder metadata was parsed one level at a time and only the PU1K local-LDI checkpoint was downloaded from file ID `1lWm7TxCR_5fDWlk7sqTDrts-ssm_KBQW` to `third_party/enhancement/APU-LDI/local_distance_indicator/pretrained_local/pu1k_local/ckpt/ckpt-epoch-60.pth`. Compatibility patch `skills/methods/patches/apuldi_inference_compat.patch` makes Chamfer3D optional for inference and adds `PCE_FORCE_APULDI_FALLBACKS=1` torch FPS/KNN fallbacks for the old pointops extension. Adapter `scripts/run_apuldi_selected_frames.py` uses only the pretrained local LDI path, not the per-shape global-field optimization. It samples UVG CG to 2048 input points, runs 4x local upsampling to 8192 output points, and transfers RGB from CG by nearest-neighbor `k=1`. Smoke frame `0000` is negative overall: precision improves slightly (`P_10 0.2418 -> 0.2461`, `P_20 0.4734 -> 0.4885`), but `CD_Acc 23.6178 -> 23.8509`, `CD_Comp 26.4884 -> 29.0848`, `chamfer-L1 50.1062 -> 52.9357`, `chamfer-L2 2098.37 -> 2347.66`, recall, and F-scores all worsen. Metrics: `/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/results/uvg_cwi_dqpc/OrangeKettlebell/apuldi_local_pu1k_4x_2048/summary_metrics.csv`; job: `jobs/apuldi_orangekettlebell_0000_smoke.slurm`. | Keep in the survey as a recent arbitrary-scale upsampling method, but label current UVG evidence as smoke-only/negative and fixed-size local-LDI domain transfer. Expanding to selected-10 or full dataset would require an explicit decision because full APU-LDI global field uses per-frame optimization and is not directly comparable to pretrained point-cloud-only inference. |
| TULIP | Independent paper/repo search | no | 2024 | TULIP: Transformer for Upsampling of LiDAR Point Clouds | Paper/repo: `https://github.com/ethz-asl/TULIP`; paper: `https://arxiv.org/abs/2312.06733`; checkpoint package from README: `https://drive.google.com/file/d/15Ty7sKOrFHhB94vLBJOKasXaz1_DCa8o/view?usp=drive_link`<br>Publication: CVPR 2024 | Repo cloned; pretrained package downloaded; adapter not run yet | Official PyTorch repo cloned under `third_party/enhancement/TULIP`. Pretrained package downloaded to `third_party/enhancement/TULIP/pretrained/trained.zip`; extracted base checkpoints: `trained/tulip_kitti.pth`, `trained/tulip_durlar.pth`, and `trained/tulip_carla.pth`. This method is range-image LiDAR upsampling, not direct point-set upsampling: official evaluation expects KITTI/DurLAR/CARLA `.npy` range-image grids such as `16x1024 -> 64x1024`, then converts predictions back to LiDAR scans. UVG human captures are dense object/scene point clouds with RGB and no LiDAR sensor origin/ring structure. | Treat only as a domain-transfer experiment if we build a spherical/range-image UVG adapter. Do not mix its result with direct point-set upsampling unless clearly labeled. |
| CRCIR upsampling mode | Independent paper/repo search | no | 2024 | Fast Point Cloud Geometry Compression with Context-based Residual Coding and INR-based Refinement | Repo: `https://github.com/hxu160/CRCIR_for_PCGC`; paper: `https://arxiv.org/pdf/2408.02966`<br>Publication: ECCV 2024 | Smoke completed as `24056860`; selected-10 completed as `24057175`; selected-10 texture/perceptual completed as `24062115` | Official PyTorch repo cloned under `third_party/enhancement/CRCIR_for_PCGC`. The repo includes pretrained checkpoints at `result/ex0_hyper_5e_3/checkpoint_best.pth` and `configs/8D_lr3/checkpoint_best.pth`. This is primarily a point-cloud geometry compression method, but the README explicitly documents direct upsampling during decompression and post-decompression cascaded upsampling on PU-GAN. It is therefore relevant as an upsampling-through-compression baseline, not a pure upsampling network. Current `torch_env` already has PyTorch3D and Open3D. Method-local dependencies are installed under `third_party/enhancement/CRCIR_for_PCGC/python_deps`: `pybind11==2.13.6`, `compressai==1.2.6`, and `pytorch-msssim==1.0.0`; import check passes without duplicating Torch/CUDA packages. Adapter `scripts/run_crcir_selected_frames.py` uses the repo's upsample-after-compression idea without Draco: for each 2048-point UVG chunk it normalizes the chunk, FPS downsamples by `K_e=2`, computes residual features against the full chunk, compresses/decompresses those features through CRCIR, decodes 4x geometry via `K_e * decoder_multiplier = 8`, writes UVG-compatible PLY, and transfers RGB from CG by `k=1`. Selected-10 is positive on all listed geometry metrics: `CD_Acc 25.5177 -> 25.1211`, `CD_Comp 27.6065 -> 27.2121`, `chamfer-L1 53.1241 -> 52.3332`, `chamfer-L2 2293.13 -> 2256.60`, `F_10 0.2459 -> 0.2525`, and `F_20 0.4552 -> 0.4631`. Texture/perceptual is mixed/positive: `dYUV-PSNR -0.0482`, `dProj-SSIM +0.0139`, `dLPIPS +0.0454`, `dPCQM +0.0008`. Geometry metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/crcir_aftercomp_4x_selected10/summary_metrics.csv`; texture metrics: `results/texture_perceptual_metrics/OrangeKettlebell/selected10_pointfilter_crcir/summary_texture_perceptual_metrics.csv`. | Candidate compression-derived enhancement; inspect subjectively and keep separate from pure upsampling methods. |

## External Completion Methods

Completion methods are tracked as domain-transfer experiments because most
survey-listed completion models assume normalized ShapeNet/MVP-style partial
object inputs and fixed-size complete object outputs, not dense dynamic
UVG-CWI-DQPC frames. For UVG, adapters may sample a fixed-size partial input,
run official/default completion inference, denormalize, and transfer RGB from
CG with nearest neighbor `k=1`.

| Method | Source/provenance | Reading paper | Publication year | Paper title | Paper / repo link | Status | Why / observed result | Potential adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PoinTr | LilydotEE survey repo | no | 2021 | PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers | Paper: `https://arxiv.org/abs/2108.08839`; repo: `https://github.com/yuxumin/PoinTr`<br>Publication: ICCV 2021 | Smoke completed as `24009123`; selected-10 completed as `24009166` | First completion-domain-transfer candidate from the survey. Adapter samples 2048 UVG CG points, applies official ShapeNet normalization, runs PoinTr, denormalizes, transfers RGB by `k=1`, and evaluates against HE. Selected-10 is worse overall than CG baseline: `CD_Acc`, `CD_Comp`, Chamfer, recall, and F-scores worsen; output is fixed at 8192 points versus ~560k input points, causing severe completeness/recall loss. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pointr_shapenet55/summary_metrics.csv` | Keep as a domain-transfer failure/completion baseline; do not expand unless the paper specifically needs a completion-method negative example. |
| AdaPoinTr | LilydotEE survey repo | no | 2023 | AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers | Paper: `https://arxiv.org/abs/2301.04545`; repo: `https://github.com/yuxumin/PoinTr`<br>Publication: IEEE TPAMI 2023 | Included in the survey and Overleaf. Projected-ShapeNet55 checkpoint was CPU-load verified on Snellius; smoke job `24009484` completed. On 2026-07-23 the user explicitly promoted AdaPoinTr to execution on UVG-CWI-DQPC despite the negative direct CG-to-HE smoke. The official repository was restored on LUMI and the 372 MiB Projected-ShapeNet55 checkpoint was downloaded from the official model link with SHA-256 `807ef42b3649f22dfc4cbc7ff7632d2ba593dcaa5bd93433c6137555fe3280a7`. | Primary protocol: create deterministic projected/occlusion masks from HE, sample 2,048 visible input points, run the unmodified pretrained model, and compare its 8,192 output points with an independently sampled 8,192-point HE target. Use multiple fixed mask severities and seeds. Secondary protocol: reproduce the existing direct CG-to-HE 2,048-to-8,192 zero-shot stress test. Run OrangeKettlebell selected-10 first, then expand the validated protocol to all 2,152 pairs under immutable run names. Keep results in a separate completion table. Fine-tune only after zero-shot reporting, using sequence-level splits with no frame leakage. |
| AtlasNet | LilydotEE survey repo | no | 2018 | A Papier-Mache Approach to Learning 3D Surface Generation | Paper: `https://arxiv.org/abs/1802.05384`; repo: `https://github.com/ThibaultGROUEIX/AtlasNet`; trained/data package: `https://drive.google.com/drive/folders/1If_-t0Aw9Zps-gj5ttgaMSTqRwYms9Ag?usp=sharing`<br>Publication: CVPR 2018 | Held / no local trained model and not direct dense completion | Repo cloned under `third_party/enhancement/AtlasNet`. README provides downloadable trained models and ShapeNet assets, but no checkpoint is local. The method is an autoencoder/single-view surface generation baseline rather than direct dense point-cloud enhancement; running it would require choosing a trained ShapeNet model and a mesh/point sampling path. No smoke job submitted. | Revisit only if a ShapeNet-trained model is downloaded and the survey needs an older generation/completion baseline; otherwise lower priority than direct point-completion methods already tested. |
| MSN | LilydotEE survey repo | yes | 2020 | Morphing and Sampling Network for Dense Point Cloud Completion | Paper: `http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf`; repo: `https://github.com/Colin97/MSN-Point-Cloud-Completion`; trained model/data folder: `https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7?usp=sharing`<br>Publication: AAAI 2020 | Blocked / waiting for trained model package and old extension decision | Repo cloned under `third_party/enhancement/MSN-Point-Cloud-Completion`. The README says to download data and trained models from the Google Drive folder; no `trained_model/network.pth` or other checkpoint is present locally. Official validation expects PyTorch 1.2/CUDA 10, Open3D, Visdom, and compiled custom CUDA modules for EMD, expansion penalty, and minimum-density sampling. No smoke job was submitted because there is no local checkpoint and building/replacing those old extensions would be a dedicated compatibility task. | Ask user to download the trained-model package if MSN is needed. Expected checkpoint path from `val.py`: `third_party/enhancement/MSN-Point-Cloud-Completion/trained_model/network.pth`. If provided, decide whether to create an old dedicated env or implement inference-only fallbacks before running a smoke test. |
| Disp3D | LilydotEE survey repo | no | 2022 | Learning Local Displacements for Point Cloud Completion | Paper: `https://arxiv.org/abs/2203.16600`; repo: `https://github.com/wangyida/disp3d`<br>Publication: CVPR 2022 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/disp3d`. README validation expects a trained model such as `log/exp_shapenet/network.pth`, but no `.pth`/`.pt` checkpoint is present locally. Setup targets a dedicated CUDA 10.2 PyTorch environment and builds custom Chamfer/EMD extensions. No smoke job was submitted because training is outside the current benchmark rule and there is no local pretrained model. | Revisit only if a compatible pretrained `network.pth` is found/downloaded; then run as a fixed-size completion domain-transfer smoke, not a dense-scene direct enhancement method. |
| SoftPoolNet | LilydotEE survey repo | yes | 2020 | SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification | Paper: `https://arxiv.org/abs/2008.07358`; repo: `https://github.com/wangyida/softpool`<br>Publication: ECCV 2020 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/softpool`. README validation expects checkpoints such as `log/ijcv_shapenet_softpool/network.pth`, but no `.pth`/`.pt` checkpoint is present locally. The repo also recommends compiling old custom CUDA packages for Chamfer, EMD, expansion penalty, and GRNet extensions. No smoke job was submitted because training is outside the current benchmark rule and no compatible pretrained model is local. | Revisit only if a compatible `network.pth` is found/downloaded; then treat as fixed-size completion domain-transfer, not dense-scene direct enhancement. |
| ECG | LilydotEE survey repo | yes | 2020 | ECG: Edge-aware Point Cloud Completion with Graph Convolution | Paper: `https://ieeexplore.ieee.org/document/9093117`; repo: `https://github.com/paul007pl/ECG`<br>Publication: IEEE RA-L 2020 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/ECG`. It is a PyTorch implementation evaluated with Python 3.5 and PyTorch 1.2, and includes old custom CUDA modules for MDS, EMD, expansion penalty, and PointNet++ utilities. No `.pth`/`.pt` checkpoint is present locally and the README does not provide a direct pretrained model link. No smoke job was submitted under the no-training rule. | Revisit only if a compatible pretrained checkpoint is found; then run as fixed-size completion domain-transfer in a dedicated old-PyTorch or compatibility env. |
| SnowflakeNet | LilydotEE survey repo | no | 2021 / 2023 | SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer | Paper: `https://arxiv.org/abs/2108.04444`; repo: `https://github.com/AllenXiangX/SnowflakeNet`<br>Publication: ICCV 2021; extended TPAMI 2023 | Smoke completed as `24013376`; not expanded | Survey-listed completion method. Completion checkpoints are local under `third_party/enhancement/SnowflakeNet/pretrained/completion/`. Adapter `scripts/run_snowflakenet_selected_frames.py` samples 2048 UVG CG points, applies unit-object normalization, runs the official PCN CD-L1 checkpoint, denormalizes, transfers RGB by `k=1`, and evaluates against HE. Compatibility fallback replaces old PointNet++ ops with PyTorch/PyTorch3D equivalents. Smoke frame `0000` is clearly worse than CG baseline: `CD_Acc` 23.62 -> 33.52, `CD_Comp` 26.49 -> 33.56, `F_10` 0.257 -> 0.166, `F_20` 0.487 -> 0.382. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/snowflakenet_pcn_cd1/summary_metrics.csv` | Do not expand to selected-10 under current rules because smoke is clearly negative; keep as completion domain-transfer failure. |
| SeedFormer | LilydotEE survey repo | no | 2022 | SeedFormer: Patch Seeds Based Point Cloud Completion with Upsample Transformer | Paper: `https://arxiv.org/abs/2207.10315`; repo: `https://github.com/hrzhou2/seedformer`<br>Publication: ECCV 2022 | Smoke completed as `24013389`; not expanded | Strong survey-listed completion method with PyTorch repo. Adapter `scripts/run_seedformer_selected_frames.py` uses the official PCN checkpoint, samples 2048 UVG CG points, applies unit-object normalization, denormalizes the 16,384-point output, transfers RGB by `k=1`, and evaluates against HE. Compatibility fallback replaces old PointNet++ ops with PyTorch/PyTorch3D equivalents. Smoke frame `0000` is clearly worse than CG baseline: `CD_Acc` 23.62 -> 45.71, `CD_Comp` 26.49 -> 36.69, `F_10` 0.257 -> 0.098, `F_20` 0.487 -> 0.272. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/seedformer_pcn/summary_metrics.csv` | Do not expand to selected-10 under current rules because smoke is clearly negative; keep as completion domain-transfer failure. |
| VRCNet | LilydotEE survey repo | no | 2021 | Variational Relational Point Completion Network | Paper: `https://arxiv.org/abs/2104.10154`; repo: `https://github.com/paul007pl/VRCNet`<br>Publication: CVPR 2021 | Smoke completed as `24013421`; not expanded | Survey-listed probabilistic completion method. Adapter `scripts/run_vrcnet_selected_frames.py` samples 2048 UVG CG points, applies unit-object normalization, runs the official VRCNet forward components with fixed seed, denormalizes the 2048-point output, transfers RGB by `k=1`, and evaluates against HE. Compatibility patches make PointNet++/EMD/Chamfer imports optional or PyTorch/PyTorch3D-backed; the adapter avoids the repo's internal EMD/CD test path and uses shared UVG metrics. Smoke frame `0000` is clearly worse than CG baseline: `CD_Acc` 23.62 -> 34.72, `CD_Comp` 26.49 -> 47.24, `F_10` 0.257 -> 0.065, `F_20` 0.487 -> 0.285. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/vrcnet_mvp_2048/summary_metrics.csv` | Do not expand to selected-10 under current rules because smoke is clearly negative; keep as completion domain-transfer failure. |
| ASFM-Net | LilydotEE survey repo | no | 2021 | ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion | Paper: `https://dl.acm.org/doi/abs/10.1145/3474085.3475348`; repo: `https://github.com/Yan-Xia/ASFM-Net`; trained model folder: `https://drive.google.com/drive/folders/1r8x6jq1QCWJ9fvep604nMkexykqQGpT0?usp=sharing`<br>Publication: ACM MM 2021 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/ASFM-Net`. README uses a Docker/TensorFlow workflow and `shapenet_test.sh`; code includes TensorFlow distance, grouping, and sampling custom CUDA ops. No trained checkpoint is present locally; README points to an external Google Drive folder. No smoke job was submitted because this violates the current skip-TF/custom-op rule unless a maintained PyTorch/pretrained implementation appears. | Keep as survey-listed completion method. Revisit only if user approves a TensorFlow exception or finds a PyTorch implementation with pretrained weights. |
| LAKe-Net | LilydotEE survey repo / independent repo search | no | 2022 | LAKe-Net: Topology-Aware Point Cloud Completion by Localizing Aligned Keypoints | Paper: `https://arxiv.org/abs/2203.16771`; repo: `https://github.com/junshutang/LAKe-Net`<br>Publication: CVPR 2022 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/LAKe-Net`. It is PyTorch code with fixed ShapeNet-style loaders and many custom CUDA dependencies: PointNet++ FPS/gather, EMD, expansion penalty, MDS, gridding, cubic feature sampling, and Chamfer. The official README has no setup/test instructions beyond citation/description, and no `.pth`/`.pt` checkpoint is present locally. No smoke job was submitted under the missing-checkpoint and no-training rules. | Revisit only if a compatible `args.load_model` checkpoint is found; then treat as completion domain-transfer with the same 2048-input normalization protocol. |
| RFNet | LilydotEE survey paper-only entry / independent repo search | no | 2021 | RFNet: Recurrent Forward Network for Dense Point Cloud Completion | Paper: `https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_RFNet_Recurrent_Forward_Network_for_Dense_Point_Cloud_Completion_ICCV_2021_paper.pdf`; repo: `https://github.com/Tianxinhuang/RFNet`<br>Publication: ICCV 2021 | Held / skipped under TF/custom-op rule despite local checkpoint | Repo cloned under `third_party/enhancement/RFNet`. It contains a TensorFlow checkpoint under `bestrecord/model-229999.*`, but README targets TensorFlow 1.13.1, CUDA 10.0, Python 3.6.9, LMDB/TensorPack, and custom TensorFlow ops. No smoke job was submitted because the current benchmark rule skips TF/custom-op methods unless a maintained PyTorch version exists. | Revisit only if user approves a TF exception. If revisited, adapt `recon_test.py` to a UVG single-frame LMDB/PCD wrapper and transfer RGB by `k=1`. |
| ARFNet | Independent paper/repo search | no | 2022 | Adaptive Recurrent Forward Network for Dense Point Cloud Completion | Paper: `https://ieeexplore.ieee.org/document/9845478`; repo: `https://github.com/Tianxinhuang/ARFNet`<br>Publication: IEEE TMM 2022 | Held / skipped under TF/custom-op rule despite local checkpoint | Repo cloned under `third_party/enhancement/ARFNet`. It is the TMM extension of RFNet and contains a TensorFlow checkpoint under `bestrecon/modelvv_recon/model-209999.*`, but README targets TensorFlow 1.13.1, CUDA 10.0, Python 3.6.9, LMDB/TensorPack, and custom TensorFlow ops. No smoke job was submitted under the TF/custom-op skip rule. | Revisit only if user approves a TF exception. If included, classify as completion, not dense colored enhancement; geometry output would need RGB transfer from CG by `k=1`. |
| RL-GAN-Net | LilydotEE survey repo | yes | 2019 | RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion | Paper: `https://arxiv.org/abs/1904.12304`; repo: `https://github.com/iSarmad/RL-GAN-Net`<br>Publication: CVPR 2019 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/RL-GAN-Net`. README describes a multi-stage training workflow: train an autoencoder, generate global feature vectors, train GAN, then train RL. No `.pth`/`.pt`/checkpoint file is present locally and no direct pretrained model link is documented. No smoke job was submitted under the no-training rule. | Revisit only if compatible pretrained AE/GAN/RL checkpoints are found; otherwise keep as older completion survey method. |
| Pcl2Pcl-GAN | LilydotEE survey repo | yes | 2020 | Unpaired Point Cloud Completion on Real Scans using Adversarial Training | Paper: `https://arxiv.org/abs/1904.00069`; repo: `https://github.com/xuelin-chen/pcl2pcl-gan-pub`<br>Publication: ICLR 2020 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/pcl2pcl-gan-pub`. README targets Python 3.5, TensorFlow 1.5, CUDA 9.0, PointNet++ TensorFlow ops, and CUDA structural losses. It provides a Baidu data link but no pretrained checkpoint; workflow is train AE then train GAN. No smoke job was submitted under TF/custom-op and no-training rules. | Revisit only if a maintained PyTorch/pretrained implementation appears or user approves a TF exception and provides trained checkpoints. |
| Point Completion Shape Prior | LilydotEE survey repo | yes | 2020 | Point Cloud Completion by Learning Shape Priors | Paper: `https://ieeexplore.ieee.org/document/9341206`; repo: `https://github.com/xiaogangw/point-cloud-completion-shape-prior`; pretrained models: `https://drive.google.com/file/d/1JIMgKtlWPxP30nb1BnbPKUbY6mO6uaBt/view?usp=sharing`<br>Publication: IROS 2020 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/point-cloud-completion-shape-prior`. README requires Python 3.5, CUDA 10, TensorFlow 1.13, and custom TensorFlow PointNet++/distance/sampling ops. A pretrained model link is provided, but no model is local. No smoke job was submitted under the TF/custom-op skip rule. | Revisit only if a TensorFlow exception is approved or a maintained PyTorch/pretrained implementation is found. |
| Cascaded Refinement Network | LilydotEE survey repo | yes | 2020 | Cascaded Refinement Network for Point Cloud Completion | Paper: `https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cascaded_Refinement_Network_for_Point_Cloud_Completion_CVPR_2020_paper.html`; repo: `https://github.com/xiaogangw/cascaded-point-completion`; pretrained models: `https://drive.google.com/file/d/1egNorG-u98SWUueBsZquw02l4cHU8xBD/view?usp=sharing`<br>Publication: CVPR 2020 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/cascaded-point-completion`. It requires Python 3.5, CUDA 10, TensorFlow 1.13, and custom TensorFlow PointNet++/distance/grouping/sampling ops. README provides a pretrained model link, but no model is local. No smoke job was submitted under the TF/custom-op skip rule. | Revisit only if a TF exception is approved or a maintained PyTorch/pretrained implementation is found. |
| SAUM | LilydotEE survey repo | yes | 2020 | SAUM: Symmetry-Aware Upsampling Module for Consistent Point Cloud Completion | Paper: `https://openaccess.thecvf.com/content/ACCV2020/papers/Son_SAUM_Symmetry-Aware_Upsampling_Module_for_Consistent_Point_Cloud_Completion_ACCV_2020_paper.pdf`; repo: `https://github.com/countywest/SAUM`; pretrained folder: `https://drive.google.com/drive/folders/1DMNY7Q3mnkz3UpYptXAH97iT9ysVqQLc?usp=sharing`<br>Publication: ACCV 2020 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/SAUM`. It is TensorFlow code based on PCN and requires custom TensorFlow FPS and point-cloud-distance extensions. README provides a Google Drive pretrained-model folder for PCN and TopNet decoder variants, but no checkpoint is local. No smoke job was submitted because it violates the current TF/custom-op skip rule. | Revisit only if user approves a TensorFlow exception or a maintained PyTorch/pretrained implementation is found. |
| Cycle4Completion | LilydotEE survey repo | no | 2021 | Cycle4Completion: Unpaired Point Cloud Completion using Cycle Transformation with Missing Region Coding | Paper: `https://arxiv.org/abs/2103.07838`; repo: `https://github.com/diviswen/Cycle4Completion`<br>Publication: CVPR 2021 | Held / skipped under TF/Python2 rule | Repo cloned under `third_party/enhancement/Cycle4Completion`. README requires Python 2.7, TensorFlow 1.14, and PointNet++-style setup. The repo provides a dataset link but no pretrained checkpoint or inference-ready model. No smoke job was submitted because it violates the TF/Python2 skip rule and training is outside the current benchmark pass. | Keep as survey-listed unpaired completion method; revisit only if a maintained PyTorch/pretrained implementation is found or user approves a TensorFlow exception. |
| SpareNet | LilydotEE survey repo | no | 2021 | Style-based Point Generator with Adversarial Rendering for Point Cloud Completion | Paper: `https://arxiv.org/abs/2103.02535`; repo: `https://github.com/microsoft/SpareNet`; ShapeNet checkpoint: `https://drive.google.com/file/d/15PiH-bRlSlK4AUUnVwREzuAlMVJ9TfQG`<br>Publication: CVPR 2021 | Blocked / waiting for pretrained completion checkpoint and extension decision | Repo cloned under `third_party/enhancement/SpareNet`. README provides external pretrained model links, but no completion checkpoint is local; the only `.pth` found in the clone is `Frechet/cls_model_39.pth`, used for FPD evaluation, not completion inference. The official setup targets Python 3.7, PyTorch with CUDA 10.1, and several CUDA extensions including MDS, Chamfer, EMD, gridding, gridding loss, cubic feature sampling, and differentiable rendering. No smoke job was submitted because the completion checkpoint is missing and extension compatibility would need a dedicated env or fallback decision. | Ask user to download the SpareNet ShapeNet checkpoint if this method is needed. Expected local path can be `third_party/enhancement/SpareNet/pretrained/sparenet_shapenet.pth`; then decide whether to build a dedicated env or adapt inference-only fallbacks. |
| PF-Net | LilydotEE survey repo | yes | 2020 | PF-Net: Point Fractal Network for 3D Point Cloud Completion | Paper: `https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf`; repo: `https://github.com/zztianzz/PF-Net-Point-Fractal-Network`<br>Publication: CVPR 2020 | Blocked / no pretrained checkpoint found | Repo cloned under `third_party/enhancement/PF-Net-Point-Fractal-Network`. README says the project is under construction and documents training/evaluation scripts, but no `.pth`/`.pt` checkpoint is present locally. It targets PyTorch 1.0.1 and Python 3.7.4. No smoke job was submitted under the no-training rule. | Revisit only if a compatible pretrained checkpoint is found; otherwise keep as a survey-listed older completion method. |
| PMP-Net / PMP-Net++ | LilydotEE survey repo | no | 2021 / 2022 | Point Cloud Completion by Learning Multi-step Point Moving Paths / Transformer-Enhanced Multi-step Point Moving Paths | Paper: `https://arxiv.org/abs/2012.03408`; repo: `https://github.com/diviswen/PMP-Net`<br>Publication: CVPR 2021; TPAMI 2022 for PMP-Net++ | Smoke completed as retry `24013575`; not expanded | Survey-listed point-moving-path completion family. Both PCN PyTorch checkpoints load with zero missing weights after a PointNet++ PyTorch3D fallback patch. Adapter `scripts/run_pmpnet_selected_frames.py` samples normalized UVG CG points, runs eight 2048-point PMP-Net++ completion/deformation passes with the official PCN checkpoint, concatenates to 16,384 points, denormalizes, transfers RGB by `k=1`, and evaluates against HE. First smoke `24013572` failed in fallback `three_nn`; retry fixed fallback interpolation when a group-all layer has fewer than three source points. Smoke frame `0000` is clearly worse than CG baseline: `CD_Acc` 23.62 -> 45.10, `CD_Comp` 26.49 -> 116.96, `F_10` 0.257 -> 0.107, `F_20` 0.487 -> 0.241. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/pmpnetplus_pcn/summary_metrics.csv`. Environment: shared `torch_env`, recorded in `skills/methods/ENVIRONMENT_REPRODUCIBILITY.md`. | Do not expand to selected-10 under current rules because smoke is clearly negative; keep as completion domain-transfer failure. |
| Detail-Preserved SFA | LilydotEE survey repo | yes | 2020 | Detail Preserved Point Cloud Completion via Separated Feature Aggregation | Paper: `https://arxiv.org/abs/2007.02374`; repo: `https://github.com/XLechter/Detail-Preserved-Point-Cloud-Completion-via-SFA`; pretrained folder: `https://drive.google.com/drive/folders/1BjQxULQuAKrFaPSZbNMuacqPQp0rDSa4?usp=sharing`<br>Publication: ECCV 2020 | Held / skipped under TF/custom-op rule | Repo cloned under `third_party/enhancement/Detail-Preserved-Point-Cloud-Completion-via-SFA`. README targets TensorFlow 1.14/CUDA 10.0/Python 3.6 and requires custom TensorFlow operators under `tf_ops` plus `pc_distance`. The repo has placeholder folders only; no pretrained RFA/GLFA model files are local. No smoke job was submitted under the TF/custom-op and missing-checkpoint rules. | Revisit only if user approves a TF exception and downloads the trained models into `third_party/enhancement/Detail-Preserved-Point-Cloud-Completion-via-SFA/data/trained_models/`; otherwise keep as survey-listed completion method. |
| Vaccine-style-net | LilydotEE survey repo | yes | 2020 | Vaccine-style-net: Point Cloud Completion in Implicit Continuous Function Space | Paper: `https://dl.acm.org/doi/10.1145/3394171.3413648`; repo: `https://github.com/YanWei123/Vaccine-style-net-Point-Cloud-Completion-in-Implicit-Continuous-Function-Space`<br>Publication: ACM MM 2020 | Not runnable / code not released in cloned repo | Repo cloned under `third_party/enhancement/Vaccine-style-net`, but it contains only a README placeholder saying code will be released later. There is no model code, inference script, environment file, or checkpoint. No smoke job can be submitted. | User should read the paper if deciding survey inclusion; benchmark cannot run this official repo unless code/checkpoints are released or an independent implementation is found. |
| Shape Inversion | LilydotEE survey repo | yes | 2021 | Unsupervised 3D Shape Completion through GAN Inversion | Paper: `https://arxiv.org/abs/2104.13366`; repo: `https://github.com/junzhezhang/shape-inversion`; pretrained folder: `https://drive.google.com/drive/folders/1FA29POuh5hlg50ulCxSMjCImbgI6wxvR`<br>Publication: CVPR 2021 | Blocked / waiting for class-specific Tree-GAN checkpoints | Repo cloned under `third_party/enhancement/shape-inversion`. README targets Python 3.7, PyTorch 1.2, and external Chamfer/EMD packages. It provides only evaluation assets locally (`evaluation/cls_model_39.pth`, `pre_statistics_chair.npz`), not the required generative checkpoints such as `pretrained_models/chair.pt`. Inference is iterative optimization, not a direct feed-forward dense-scene enhancement pass, and would require choosing a ShapeNet class prior for UVG. No smoke job was submitted. | Revisit only after class-specific Tree-GAN checkpoints are downloaded under `third_party/enhancement/shape-inversion/pretrained_models/` and after deciding whether a category prior is valid for OrangeKettlebell. |
| FBNet | Independent paper/repo search | yes, read and excluded from main benchmark | 2022 | FBNet: Feedback Network for Point Cloud Completion | Paper: `https://arxiv.org/abs/2210.03974`; repo: `https://github.com/hikvision-research/3DVision/tree/main/PointCompletion/FBNet`<br>Publication: ECCV 2022 | Excluded from survey benchmark after paper reading; smoke completed as `24070222` before exclusion decision | Official PyTorch code and MVP 2048 checkpoint are included in the Hikvision 3DVision repo under `third_party/enhancement/3DVision/PointCompletion/FBNet`. The paper has been read and is excluded because it is designed for partial point cloud completion, while the current survey benchmark focuses on degraded dense point cloud enhancement. Adapter `scripts/run_fbnet_selected_frames.py` followed the same domain-transfer protocol as other completion baselines before exclusion: sample 2048 UVG CG points, unit-object normalize, run official checkpoint, denormalize, write UVG-compatible PLY, and transfer RGB from CG by `k=1`. A small compatibility patch provides inference-only PyTorch/PyTorch3D fallbacks for the missing MVP_Benchmark PointNet++ utilities. Frame `0000` is clearly negative on all non-NaN objective geometry metrics: `CD_Acc 23.6178 -> 28.5483`, `CD_Comp 26.4884 -> 40.6107`, `chamfer-L1 50.1062 -> 69.1590`, `chamfer-L2 2098.37 -> 3831.58`, `F_10 0.2570 -> 0.0658`, and `F_20 0.4866 -> 0.3033`. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/fbnet_mvp2k/summary_metrics.csv`; output: `results/method_outputs/fbnet_mvp2k/OrangeKettlebell/15fps/frame_0000.ply`. | Do not expand to selected-10 or count as an included benchmark method. Keep only as an excluded partial-completion/domain-transfer reference if the survey discusses scope boundaries. |
| GRNet / PCN / TopNet | LilydotEE survey repo | yes | 2020 / 2018 / 2019 | GRNet: Gridding Residual Network for Dense Point Cloud Completion / PCN: Point Completion Network / TopNet: Structural Point Cloud Decoder | GRNet repo: `https://github.com/hzxie/GRNet`; PCN repo: `https://github.com/wentaoyuan/pcn`; TopNet code: `https://github.com/lynetcha/completion3d`<br>Publication: ECCV 2020 / 3DV 2018 / CVPR 2019 | Repos cloned; held / not run; PyTorch PCN checkpoint check completed | Classic completion baselines, but lower priority after PoinTr, AdaPoinTr, SnowflakeNet, SeedFormer, VRCNet, and PMP-Net++ all failed domain-transfer smoke tests. PCN official code is TensorFlow 1.12/CUDA 9, so it violates the current TF-skip rule. The PoinTr repo includes PyTorch model/config files for the PCN baseline (`models/PCN.py`, `cfgs/PCN_models/PCN.yaml`), but no standalone pretrained PCN baseline checkpoint is local. The PoinTr README checkpoint links named `PoinTr_PCN.pth` / `AdaPoinTr_PCN.pth` are for PoinTr/AdaPoinTr trained on the PCN dataset, not the original PCN baseline model. TopNet/completion3d has PyTorch scaffolding but no obvious pretrained PyTorch checkpoint in the repo. GRNet has a listed ShapeNet checkpoint, but the current link returns an HTML gateway page rather than a direct `.pth`, and the model requires four custom CUDA extensions: Chamfer, cubic feature sampling, gridding, and gridding loss. | Hold unless a classic completion baseline is explicitly needed. If needed, first obtain a direct standalone PCN/GRNet/TopNet checkpoint and build/patch extensions in a dedicated env; otherwise do not spend full-dataset time on these. |

## Traditional Baselines

| Method | Source/provenance | Reading paper | Publication year | Source | Status | Why / observed result | Potential adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SOR direct filtering | Traditional baseline | no | N/A | Open3D statistical outlier removal | Selected-10 completed | Worsened completeness/F-scores. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/geometry_filter_sor/summary_metrics.csv` | Keep as traditional failure/ablation baseline. |
| SPSR | Traditional baseline | no | 2013 | Screened Poisson Surface Reconstruction, Kazhdan and Hoppe 2013 | Selected-10 completed | Clean face visually but separated blobs near edges; slight recall/F20 gain but worse Chamfer. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/traditional_spsr/summary_metrics.csv` | Density filtering could reduce blobs but would be a tuned ablation. |
| BPA | Traditional baseline | no | 1999 | Ball-Pivoting Algorithm, Bernardini et al. 1999 | Selected-10 completed | Reasonable traditional baseline; small mean Chamfer/F-score improvement, lower completeness/recall. Metrics: `results/uvg_cwi_dqpc/OrangeKettlebell/traditional_bpa/summary_metrics.csv` | Keep as a traditional baseline. |
