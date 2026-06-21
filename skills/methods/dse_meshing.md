# DSE Meshing

## Source

- Repository: `https://github.com/mrakotosaon/dse-meshing`
- Local checkout: `third_party/SCUTSurface/reconstruction/DSE-Meshing`

## Category

Surface reconstruction / learning-based mesh reconstruction.

## Input

The original pipeline expects `.xyz` point clouds in:

- `data/test_data`

For UVG-CWI-DQPC this would require converting each CG frame to `.xyz`, then denormalizing/sampling the output mesh for UVG-CWI metrics.

Toy benchmark frames:

- `0000`, `0010`, `0020`, `0030`, `0040`, `0050`, `0060`, `0070`, `0080`, `0090`

## Output

The original pipeline writes final meshes to:

- `data/test_data/select/final_mesh_<shape>.ply`

For this benchmark the expected output would be:

- Mesh PLY: `results/method_outputs/dse_meshing/OrangeKettlebell/15fps/frame_<id>_mesh.ply`
- Sampled point cloud PLY: `results/method_outputs/dse_meshing/OrangeKettlebell/15fps/frame_<id>.ply`

## Current Status

Held for this pass.

Reason:

- Requires Python 3.6 and TensorFlow 1.15.
- Existing checked environments `torch_env` and `py37` do not have TensorFlow installed.
- The method also needs pretrained models from the authors and a C++ triangle-selection submodule build.
- Under the current rule to use one runnable version and prefer PyTorch when possible, this is not a good quota-efficient next method.

Next action if we later need it:

1. Create a dedicated TF1 environment.
2. Download `dse_meshing_pretrained_models.tar.xz`.
3. Initialize/build `triangle_selection/postprocess`.
4. Run a single-frame smoke test before attempting the 10-frame protocol.
