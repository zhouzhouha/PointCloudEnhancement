# Method Name

## Source

- Repository:
- Paper:
- Category:
- Selected for benchmark: yes/no

## Purpose

Briefly state what the method is designed to do: denoising, completion, up-sampling / super-resolution, geometry reconstruction, color enhancement, temporal enhancement, or hybrid enhancement.

## Environment

- Python / CUDA / compiler requirements:
- Required pretrained model:
- Snellius module or conda environment:
- Expected hardware:

Read the Snellius skill reference before writing environment commands or SLURM scripts.

## Input

- Expected input file type:
- Expected point attributes:
- Expected point count:
- Expected coordinate scale:
- Does it use color:
- Does it use normals:
- Does it process single frames or full sequences:

## UVG-CWI-DQPC Adaptation

- Dataset root: `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC`
- Toy sequence: `OrangeKettlebell`
- Input path pattern: `<dataset_root>/<sequence>/cg/15fps/*.ply`
- Reference path pattern: `<dataset_root>/<sequence>/he/15fps/*.ply`
- Frame pairing rule:
- Preprocessing required:
- Postprocessing required:

## Output

- Output file type:
- Output path pattern:
- Preserved attributes:
- Generated attributes:
- Failure cases:

## Run Command

```bash
# Fill in after integration.
```

## Metrics

Evaluate output against the HE reference with UVG-CWI/Metric.

- Per-frame output:
- Per-sequence summary:
- Runtime logging:

## Status

- Integration status: todo
- Toy frame tested: no
- 5-frame test passed: no
- Full `OrangeKettlebell` sequence passed: no
- All sequences passed: no

## Notes

Record assumptions, incompatible defaults, required patches, and any paper-table caveats here.
