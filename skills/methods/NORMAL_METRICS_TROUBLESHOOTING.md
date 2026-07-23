# Normal Metrics Troubleshooting

This file records each investigation or remediation attempted for `N_Acc`,
`N_Comp`, and combined normal correctness. A `NaN` result must not be silently
accepted or removed from the final tables.

## 2026-07-17: Initial `NaN` Investigation

1. **Inspected UVG-CWI metric behavior.**
   - Result: `distance_p2p` deliberately returns an all-`NaN` normal array when
     either input has no normals; taking its mean propagates `NaN`.
2. **Inspected representative HE and enhanced PLY headers.**
   - Result: files contain `x`, `y`, `z`, and RGB properties but no `nx`, `ny`,
     or `nz` properties. The missing values are therefore expected and are not
     evidence of corrupted metric CSV files.
3. **Checked the existing Trimesh loading fallback.**
   - Result: raw point-cloud PLY inputs do not provide reliable stored vertex
     normals for this evaluator. Mesh-only `fix_normals`/face-normal sampling
     is not applicable because the UVG inputs and method outputs have no faces.
4. **Implemented batched k-NN PCA normal estimation.**
   - Method: query `k=20` local neighbors with SciPy `cKDTree`, fit the local
     covariance matrix, and use the smallest-eigenvalue eigenvector as the
     unoriented normal. Compare with absolute dot products, matching the
     original metric's sign-invariant convention.
   - Safety: reject malformed inputs, fewer than three points, invalid `k`, and
     non-finite/zero eigenvectors. Queries and eigendecompositions are batched
     to bound memory on dense UVG frames.
   - Integration: opt-in post-processing only
     (`estimate_missing_normals=True`) so active GPU inference jobs are not
     slowed by CPU normal estimation.
5. **Added synthetic validation tests.**
   - Plane test: estimated normals must align with the known plane normal.
   - Sphere test: estimated normals must align with radial ground truth up to
     sign.
   - Missing/zero-normal test: invalid arrays must be rejected.
   - Result: all three tests passed in the LUMI PyTorch container on
     2026-07-17. The container does not include `pytest`, so the three plain
     assertion-based test functions were executed directly after the module
     passed `py_compile`.

## Required Follow-up

- Use the fixed decision `k=20` for every CG, HE, and enhanced cloud; never
  select `k` per method or content based on its score.
- Run normal estimation as CPU post-processing and cache or stage results only
  if repeated computation becomes the dominant cost.
