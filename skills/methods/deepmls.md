# DeepMLS

## Source

- Repository: `https://github.com/Andy97/DeepMLS`
- Local checkout: `third_party/SCUTSurface/reconstruction/DeepMLS`

## Category

Surface reconstruction / learning-based implicit moving least-squares reconstruction.

## Current Status

Held for this pass.

Reason:

- Requires TensorFlow-era custom CUDA/TensorFlow ops.
- Requires O-CNN TensorFlow module build.
- Requires custom neighbor-search ops and modified PyMCubes.
- Existing `torch_env` and `py37` do not provide the required TensorFlow setup.

Under the current rule to skip difficult setup and prefer PyTorch when possible, DeepMLS is not a good next method for the 10-frame UVG-CWI-DQPC benchmark.

Expected benchmark output if revisited:

- Mesh PLY: `results/method_outputs/deepmls/OrangeKettlebell/15fps/frame_<id>_mesh.ply`
- Sampled point cloud PLY: `results/method_outputs/deepmls/OrangeKettlebell/15fps/frame_<id>.ply`
