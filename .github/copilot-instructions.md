## Repository: GrandChallenge — Copilot Instructions

This repository is a small Python tool to compute point-cloud metrics (Chamfer, Hausdorff,
Precision/Recall/F-score) between matched pairs of PLY files. The primary entrypoint is
`pointcloud_metrics.py` at the repo root.

Key facts you should know when making code changes or suggesting edits:

- Entrypoint: `pointcloud_metrics.py` — a single-file CLI script. Changes to behavior should
  reference this file and update its CLI help (`parse_args`). Example usage is included
  in the file header.

- File matching: pairs are matched by PLY filename basename (extension removed). The helper
  `find_matching_pairs(gt_dir, rec_dir)` lists matches and reports missing files. When adding
  tests or examples, create matching basenames in both `gt_dir` and `rec_dir`.

- PLY reading: `read_ply_points(filepath)` supports common PLY conventions:
  - vertex positions: `x,y,z` (case-insensitive variants are accepted)
  - color: multiple conventions handled (`red/green/blue`, `r/g/b`, packed `rgb` integers)
  - If colors are present and `--color_weight > 0`, colors are normalized to [0,1] and
    concatenated to XYZ scaled by the `color_weight` to form feature vectors.

- Distance computation uses SciPy's `cKDTree` for nearest-neighbor queries. When changing
  distance semantics, update all metric functions: `chamfer_distance`, `hausdorff_distance`,
  and `precision_recall_fscore` to keep consistency.

- Output: CSV written using column names in `fieldnames` inside `process_all`. If you add
  new per-sample metrics, append them to `fieldnames` and write them to the same CSV.

Developer workflows and quick commands

- Install minimal runtime dependencies used in the header:

  ```powershell
  python -m pip install numpy scipy plyfile
  ```

- Run the script (example):

  ```powershell
  python pointcloud_metrics.py --gt_dir <GT_DIR> --rec_dir <REC_DIR> --out_csv results.csv --threshold 0.5
  ```

- There is no test harness in the repo. When adding tests, prefer plain pytest files that
  create small temporary PLY files (use `plyfile` APIs) and validate outputs for a few
  synthetic cases (e.g., identical point clouds => zero distances, translated clouds =>
  predictable Chamfer/Hausdorff values).

Project-specific conventions and patterns

- Single-file CLI: prefer keeping the logic centralized in `pointcloud_metrics.py`. If you
  split into modules, keep public CLI behavior unchanged and add unit tests.

- Robust PLY parsing: changes should preserve tolerant handling of color encodings and
  case-insensitive attribute names (see `_get_attr`). New readers must not break existing
  packed-RGB handling.

- Deterministic matching: file pairing is based solely on basename matching in the two
  directories. Avoid introducing other heuristics unless you also update `find_matching_pairs`.

Integration points and external dependencies

- External packages used: `numpy`, `scipy` (for cKDTree), `plyfile` (PLY parsing). Keep
  compatibility with reasonably recent versions (e.g., numpy >= 1.20, scipy >= 1.5).

- No network or external services are used. Input is local PLY files and output is a CSV.

Editing guidance for AI agents

- When editing code, provide small, focused diffs; preserve CLI compatibility and the CSV
  schema unless the change is explicit and reviewed.
- Include small example inputs in PR descriptions (e.g., two 3-point PLYs) to make metrics
  behavior easy to verify in CI or by reviewers.
- When suggested changes affect numeric outputs, include updated example output or unit
  tests that assert the expected numeric result with a tolerance (e.g., abs tol 1e-6).

Files to reference when working here

- `pointcloud_metrics.py` — main script and the canonical source for behavior.
- Any new tests you add should live next to the script or in a `tests/` directory and use
  `pytest`.

If anything above is unclear or you need more repository-specific rules (naming, CI,
or target package versions), tell me what to look for and I'll update this file accordingly.
