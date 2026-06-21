# Total Denoising

## Source

- Paper: Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning
- Local checkout: `third_party/enhancement/TotalDenoising`
- Survey category: point cloud denoising

## Category

- External enhancement method
- Unsupervised denoising
- Geometry-only

## Current Status

- Integration status: cloned, hold
- Reason: the official code requires TensorFlow 1.13 GPU, MCCNN, and custom TensorFlow GPU ops. No CPU implementation is provided.
- It does not ship an obvious ready-to-run pretrained checkpoint in the cloned repository.

## Applicability

Scientifically, Total Denoising is relevant because it is unsupervised and could reduce dependence on synthetic object training data. Practically, it is not a good next smoke job until the TensorFlow 1.x/MCCNN environment and model checkpoint situation are clarified.

## Next Action If Resumed

1. Verify whether pretrained weights exist from the authors.
2. Check whether a TensorFlow 1.13-compatible GPU environment exists on Snellius.
3. Compile MCCNN and the repository's `tf_ops`.
4. Only then write the UVG adapter.
