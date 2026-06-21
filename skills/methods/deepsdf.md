# DeepSDF

## Source

- Repository: `https://github.com/facebookresearch/DeepSDF`
- Local checkout: `third_party/SCUTSurface/reconstruction/DeepSDF`

## Category

Learning-based implicit SDF reconstruction.

## Current Status

Held for this pass.

Reason:

- The released pipeline is not direct point-cloud enhancement from raw UVG CG frames.
- It requires SDF samples generated from watertight meshes.
- It requires trained experiment checkpoints and latent-code optimization.
- The preprocessing path uses C++ mesh sampling and OpenGL/Pangolin setup.

This makes DeepSDF unsuitable as an immediate 10-frame UVG-CWI-DQPC method under the current rule to skip difficult/non-direct methods.

Expected benchmark output if revisited:

- Mesh PLY: `results/method_outputs/deepsdf/OrangeKettlebell/15fps/frame_<id>_mesh.ply`
- Sampled point cloud PLY: `results/method_outputs/deepsdf/OrangeKettlebell/15fps/frame_<id>.ply`
