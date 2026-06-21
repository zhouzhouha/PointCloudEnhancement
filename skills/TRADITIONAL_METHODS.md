# Traditional Surface Reconstruction Methods

This document explains the **traditional (non-deep-learning) reconstruction methods** used in the surface reconstruction benchmark paper.

## Traditional Methods from the Paper

According to the paper "Surface Reconstruction From Point Clouds: A Survey and a Benchmark", the traditional methods evaluated are:

### 1. Triangulation-Based Methods
- **GD (Greedy Delaunay)** [41] - Delaunay triangulation-based reconstruction
- **BPA (Ball Pivoting Algorithm)** [42] - Pivots a ball over the surface to create triangles

### 2. Surface Smoothness Prior Methods
- **SPSR (Screened Poisson Surface Reconstruction)** [39] - Uses Poisson equation with screening
- **RIMLS (Robust Implicit Moving Least Squares)** [63] - Local implicit surface fitting

### 3. Deep Learning Methods (for comparison)
- **SALD** [83] - Can reconstruct from un-oriented point clouds
- **IGR** [37] - Reconstructs from oriented point clouds

## Implementation Status

✅ **Currently Implemented:**
- **SPSR (Screened Poisson Surface Reconstruction)** - via Open3D
- **BPA (Ball Pivoting Algorithm)** - via Open3D
- **Direct point-cloud statistical outlier filtering (SOR)** - via Open3D, already benchmarked as `geometry_filter_sor`
- **Alpha shapes** - available via Open3D in `scripts/run_traditional_reconstruction_selected_frames.py`, not yet submitted

⏳ **Not Yet Implemented:**
- GD (Greedy Delaunay) - requires specific library
- RIMLS - requires specific implementation

## Current UVG-CWI-DQPC Run Status

Traditional methods are valuable benchmark baselines because they are reproducible, non-learning-based, and do not depend on pretrained synthetic-object priors.

Current selected-10 OrangeKettlebell jobs:

- SPSR / Screened Poisson: SLURM job `23580389`, method name `traditional_spsr`
- BPA / Ball Pivoting: SLURM job `23580391`, method name `traditional_bpa`
- SOR direct filtering: completed earlier as `geometry_filter_sor`; it worsened completeness and should be reported as a traditional geometry-filter baseline.
- SPSR visual note: although the face can look cleaner because Poisson fits a smooth implicit surface through dense, locally consistent samples, it may create separated blobs around human/object boundaries. Those blobs typically come from open surfaces, noisy silhouettes, missing regions, unoriented/ambiguous normals, and low-density extrapolation near edges.

Output convention:

- `results/method_outputs/<method>/OrangeKettlebell/15fps/frame_*.ply`
- `results/uvg_cwi_dqpc/OrangeKettlebell/<method>/summary_metrics.csv`

Color policy:

- Traditional geometry methods are geometry-first. For every reconstructed/sampled point, RGB is transferred from the original CG frame with nearest neighbor `k=1`.

TF policy:

- TensorFlow 1.x/custom-op methods should be skipped unless there is a maintained PyTorch version with usable pretrained weights. For survey coverage, record them as skipped due reproducibility/setup limitations rather than spending benchmark time on obsolete environments.

## Usage

### 1. Run Single Traditional Method

**SPSR (Screened Poisson):**
```bash
python scripts/poisson_reconstruction.py --method poisson --frames 10 --depth 9
```

**BPA (Ball Pivoting):**
```bash
python scripts/poisson_reconstruction.py --method bpa --frames 10
```

### 2. Compare All Methods

Run all traditional methods and compare against baseline:
```bash
python scripts/traditional_reconstruction_comparison.py --frames 10
```

This will:
1. Run baseline evaluation (if not done)
2. Run SPSR reconstruction
3. Run BPA reconstruction
4. Evaluate all results
5. Create comparison table

### 3. Advanced Parameters

**For SPSR:**
```bash
python scripts/poisson_reconstruction.py \
    --method poisson \
    --frames 10 \
    --depth 9 \              # Octree depth (8-10 recommended, higher=more detail)
    --knn 30 \               # KNN for normal estimation
    --density_quantile 0.01  # Filter low-density vertices
```

**For BPA:**
```bash
python scripts/poisson_reconstruction.py \
    --method bpa \
    --frames 10 \
    --knn 30 \                          # KNN for normal estimation
    --bpa_radii 5.0 10.0 20.0          # Ball radii (optional, auto-computed if omitted)
```

## Method Details

### SPSR (Screened Poisson Surface Reconstruction)

**Reference:** Kazhdan & Hoppe, "Screened Poisson Surface Reconstruction", 2013

**How it works:**
- Solves a modified Poisson equation to find an implicit function whose gradient best matches the input normals
- Screening term allows handling of noisy and incomplete data
- Produces watertight meshes

**Advantages:**
- Robust to noise and outliers
- Produces smooth, watertight surfaces
- Works well with incomplete data

**Disadvantages:**
- Can over-smooth fine details
- Requires oriented normals
- Computational cost increases with octree depth

**Best for:** Smooth organic shapes, noisy scans with missing data

### BPA (Ball Pivoting Algorithm)

**Reference:** Bernardini et al., "The Ball-Pivoting Algorithm for Surface Reconstruction", 1999

**How it works:**
- Virtually "rolls" a ball of given radius over the point cloud
- Creates triangles when the ball touches three points
- Multiple ball radii handle different levels of detail

**Advantages:**
- Fast and simple
- Preserves sharp features better than Poisson
- No over-smoothing

**Disadvantages:**
- Sensitive to noise and outliers
- May produce non-watertight meshes (holes)
- Requires good normal estimation
- Radius selection is critical

**Best for:** Clean scans with uniform sampling, objects with sharp features

## Expected Results

Based on the paper and baseline evaluation:

**Baseline (CG low-quality):**
- Chamfer Distance: ~25.8 mm
- F-score@10mm: ~0.26
- F-score@20mm: ~0.48

**Expected Traditional Methods Performance:**
- **SPSR**: Should improve Chamfer distance by 10-30%, better smoothness
- **BPA**: May preserve more details but sensitive to noise, variable performance

**Deep Learning (SAL):**
- Expected best performance with proper training (~30+ min per frame)
- Significant improvement over traditional methods

## Comparison with Deep Learning

### Traditional Methods (SPSR, BPA)
✅ **Pros:**
- Fast (seconds per frame)
- No training required
- Deterministic results
- Well-understood behavior
- Lower memory requirements

❌ **Cons:**
- Lower reconstruction quality
- More sensitive to noise
- Limited ability to fill large holes
- Require good normal estimation

### Deep Learning (SAL, IGR)
✅ **Pros:**
- Better reconstruction quality
- More robust to noise
- Can learn from data
- Better at handling incomplete data

❌ **Cons:**
- Slow (30-40 min per frame)
- Requires training
- High memory requirements
- GPU recommended
- Less interpretable

## File Organization

```
GrandChallenge/
├── scripts/
│   ├── poisson_reconstruction.py              # SPSR & BPA implementation
│   ├── traditional_reconstruction_comparison.py # Compare all methods
│   ├── evaluate_orangekettlebell.py           # Evaluation script
│   └── train_sal_single_frame.py              # SAL (deep learning)
├── dataset/
│   └── UVG-CWI-DQPC/
│       └── OrangeKettlebell/
│           ├── CG/15fps/                       # Input (low-quality)
│           ├── HE/15fps/                       # Ground truth
│           └── Reconstructed/                  # Output
└── results/
    ├── orangekettlebell_baseline_metrics.xlsx
    ├── orangekettlebell_spsr_metrics.xlsx
    ├── orangekettlebell_bpa_metrics.xlsx
    ├── orangekettlebell_sal_metrics.xlsx      # If SAL completed
    └── orangekettlebell_methods_comparison.xlsx
```

## Workflow

### Quick Start (Traditional Methods Only)

1. **Run SPSR reconstruction:**
   ```bash
   python scripts/poisson_reconstruction.py --method poisson --frames 10
   ```

2. **Evaluate:**
   ```bash
   python scripts/evaluate_orangekettlebell.py --mode reconstructed --max_frames 10
   ```

3. **Save results:**
   Results are automatically saved in `results/` directory

### Complete Comparison

Run everything in one command:
```bash
python scripts/traditional_reconstruction_comparison.py --frames 10
```

This will produce `orangekettlebell_methods_comparison.xlsx` with all methods compared.

## Troubleshooting

### Issue: "No module named 'open3d'"
**Solution:**
```bash
pip install open3d
```

### Issue: BPA produces empty mesh
**Causes:**
- Normals not properly estimated
- Ball radii too large or too small

**Solution:**
- Adjust `--knn` for better normal estimation
- Try different `--bpa_radii` values (e.g., `5.0 10.0 20.0`)

### Issue: SPSR over-smooths details
**Solution:**
- Increase octree depth: `--depth 10` (slower but more detail)
- Adjust density filtering: `--density_quantile 0.0` (no filtering)

## References

[39] M. Kazhdan and H. Hoppe, "Screened poisson surface reconstruction," ACM TOG, 2013.

[42] F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, and G. Taubin, "The ball-pivoting algorithm for surface reconstruction," IEEE TVCG, 1999.

[63] Y. Öztireli, G. Guennebaud, and M. Gross, "Feature preserving point set surfaces based on non-linear kernel regression," Computer Graphics Forum, 2009.

---

**Note:** The paper evaluated more methods (GD, RIMLS), but SPSR and BPA are the most commonly used traditional methods and are readily available in Open3D.
