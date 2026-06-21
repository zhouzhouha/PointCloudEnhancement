# Metrics Implementation Plan

Use this file to track metric implementation one by one. The final benchmark should produce a table where each row is a method/sequence result and each metric below is a column.

## Primary Source

- Main metric repo: `https://github.com/UVG-CWI/Metric`
- Main implementation target: `metrics.py` from UVG-CWI/Metric
- Local SCUTSurface metrics: reference and cross-check only

## Metric Checklist

| Metric | Source | Status | Output Column | Notes |
| --- | --- | --- | --- | --- |
| Accuracy | UVG-CWI/Metric | todo | `CD_Acc` | Mean nearest-neighbor distance from enhanced/predicted point cloud to HE reference. |
| Completeness | UVG-CWI/Metric | todo | `CD_Comp` | Mean nearest-neighbor distance from HE reference to enhanced/predicted point cloud. |
| Chamfer L1 | UVG-CWI/Metric | todo | `chamfer-L1` | Sum of bidirectional mean nearest-neighbor distances. |
| Chamfer L2 | UVG-CWI/Metric | todo | `chamfer-L2` | Sum of bidirectional squared nearest-neighbor distances. |
| Legacy Chamfer | UVG-CWI/Metric | todo | `chamferL2_old` | Keep for compatibility with the repo output. |
| Precision at threshold | UVG-CWI/Metric | todo | `P_5`, `P_10`, `P_20` | Thresholds must be checked against UVG-CWI-DQPC coordinate units. |
| Recall at threshold | UVG-CWI/Metric | todo | `R_5`, `R_10`, `R_20` | Same thresholds as precision. |
| F-score at threshold | UVG-CWI/Metric | todo | `F_5`, `F_10`, `F_20` | Must use per-point distances, not scalar means. |
| Normal accuracy | UVG-CWI/Metric | todo | `N_Acc` | Predicted/enhanced normals matched to HE reference. |
| Normal completeness | UVG-CWI/Metric | todo | `N_Comp` | HE reference normals matched to predicted/enhanced cloud. |
| Normal correctness | UVG-CWI/Metric | todo | `normals` | Combined normal score. |
| Neural Feature Similarity | SCUTSurface | optional | `NFS` | Add only if the pretrained model and input format are validated. |
| Runtime | This benchmark | todo | `runtime_sec` | Wall-clock inference time per frame or sequence. |
| Temporal stability | This benchmark | todo | `temporal_metric_*` | Define after checking available dynamic point-cloud quality metrics. |
| Color / texture quality | This benchmark | todo | `color_metric_*` | Define after checking method outputs and UVG-CWI-DQPC color format. |

## Required Output Tables

- `per_frame_metrics.csv`: one row per sequence, method, and frame.
- `summary_metrics.csv`: mean and standard deviation per sequence and method.
- `all_sequences_summary.csv`: aggregate table for the paper.
- `metric_implementation_status.csv`: implementation and validation status for each metric.

## Validation Order

1. Implement or vendor one UVG-CWI/Metric metric module.
2. Run one `OrangeKettlebell` CG/HE frame pair.
3. Check identical-cloud behavior.
4. Check shifted-cloud behavior.
5. Run 5 `OrangeKettlebell` frames.
6. Run all 170 `OrangeKettlebell` frames.
7. Freeze output schema before running all methods.
