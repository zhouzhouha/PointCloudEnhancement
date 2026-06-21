# Points2Surf

## Source

- Repository: `https://github.com/ErlerPhilipp/points2surf`
- Local checkout: `third_party/SCUTSurface/reconstruction/Points2Surf`
- Paper/project page: `https://www.cg.tuwien.ac.at/research/publications/2020/erler-2020-p2s/`
- Selected version: `p2s_max`, because the README recommends the Max model as the best single pretrained version.
- Official pretrained Max model zip: `https://www.cg.tuwien.ac.at/research/publications/2020/erler-2020-p2s/erler-2020-p2s-max_model.zip`

## Category

Surface reconstruction / learning-based implicit SDF reconstruction.

## Input

Points2Surf expects normalized NumPy point clouds:

- `datasets/<dataset>/04_pts/<frame>.xyz.npy`
- `datasets/<dataset>/testset.txt`, with one frame stem per line

The UVG-CWI-DQPC CG input must be converted from binary PLY to `float32` NumPy arrays and normalized to the `[-1, 1]` model-space range. Store one normalization JSON per frame so the reconstructed mesh or sampled PLY can be denormalized back to UVG coordinates.

Toy benchmark frames:

- `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`

Input sequence path:

- `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC/OrangeKettlebell/cg/15fps`

Reference sequence path:

- `/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC/OrangeKettlebell/he/15fps`

## Output

The method reconstructs meshes from predicted SDF values. For this benchmark, keep both:

- Mesh PLY: `results/method_outputs/points2surf/OrangeKettlebell/15fps/frame_<id>_mesh.ply`
- Sampled point cloud PLY for UVG-CWI metrics: `results/method_outputs/points2surf/OrangeKettlebell/15fps/frame_<id>.ply`

The sampled PLY is compared against HE reference with the UVG-CWI metric runner. The CG input is evaluated with the same runner as the "doing nothing" baseline.

## Current Status

- Repository cloned.
- Max pretrained model download failed from the Snellius restricted shell due DNS/network access.
- Waiting for the official model zip to be downloaded manually or made available locally.

Expected placement after manual download:

- Put the zip or extracted model files in `third_party/SCUTSurface/reconstruction/Points2Surf/models/`.
- If using the zip, run `python models/download_models_max.py` only after network works, or unzip it manually into the same `models/` folder.

## Next Command Shape

After the pretrained model is available:

1. Prepare the 10 UVG frames as normalized `.xyz.npy`.
2. Run `full_eval.py` with `--models p2s_max`, `--dataset testset.txt`, `--reconstruction True`, and a reduced `--query_grid_resolution` for the first smoke test.
3. Denormalize the reconstructed meshes and sample point clouds.
4. Run `scripts/evaluate_method_vs_baseline_uvg_metric.py` for each of the 10 frames.
