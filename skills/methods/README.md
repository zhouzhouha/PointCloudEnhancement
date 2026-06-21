# Method Notes Index

This folder stores per-method notes and status files only. The canonical
benchmark workflow is the root `README.md`; do not duplicate execution protocol
or decision rules here.

Core tracking files:

- `BENCHMARK_METHOD_STATUS.md`: canonical status table for enhancement and
  reconstruction methods, including source, publication year, paper-reading
  flag, outcome, and next action.
- `EVALUATION_METRICS_STATUS.md`: canonical status table for evaluation metrics.
- `ENHANCEMENT_SURVEY_APPLICABILITY.md`: survey-derived applicability notes.
- `METHOD_TEMPLATE.md`: template for new per-method notes.

Per-method notes should explain only method-specific information: paper/repo
source, category, input/output assumptions, environment, adapter command, and
observed issues. Shared rules such as selected-10 frames, mixed-smoke handling,
color transfer, TensorFlow skipping, and final decision workflow belong in the
root `README.md`.
