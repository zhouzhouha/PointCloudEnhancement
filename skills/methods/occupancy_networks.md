# Occupancy Networks

## Source

- Repository: `https://github.com/autonomousvision/occupancy_networks`
- Local checkout: `third_party/SCUTSurface/reconstruction/OccupancyNetworks`

## Category

Learning-based implicit occupancy reconstruction.

## Current Status

Held for this pass.

Reason:

- The direct point-cloud pretrained config expects ShapeNet-style dataset folders.
- Point-cloud inputs are `.npz` files with `points`, `normals`, and optional `loc/scale`; UVG CG frames do not provide this directly.
- The repository targets Python 3.6 / PyTorch 1.0-era dependencies.
- Mesh generation depends on compiled Cython/C++ extensions (`libmcubes`, `libmise`, `libsimplify`) and the setup script also includes DMC CUDA extensions.
- The pretrained point-cloud model is ShapeNet category-oriented, so using it on UVG-CWI-DQPC real dynamic point clouds would be a domain-transfer experiment, not a clean enhancement baseline.

Expected benchmark output if revisited:

- Mesh PLY: `results/method_outputs/occupancy_networks/OrangeKettlebell/15fps/frame_<id>_mesh.ply`
- Sampled point cloud PLY: `results/method_outputs/occupancy_networks/OrangeKettlebell/15fps/frame_<id>.ply`
