# Selected-10 Benchmark Summary

Scope: `OrangeKettlebell`, frames `0000 0010 0020 0030 0040 0050 0060 0070 0080 0090`.

Positive `delta` means the method improved over the CG baseline for that metric. Lower-is-better metrics are inverted before computing delta.

| Method | Category | Decision | dCD-Acc | dCD-Comp | dChamfer-L1 | dF10 | dF20 | dYUV-PSNR | dProj-SSIM | dLPIPS | dPCQM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointCleanNet | Denoising | Candidate | 0.0817 | 0.5318 | 0.6135 | 0.0104 | 0.0048 | 0.0493 | -0.0035 | -0.0298 | -0.0009 |
| Pointfilter | Denoising | Negative baseline | 0.7299 | -8.0918 | -7.3619 | -0.0339 | -0.0600 | 0.1639 | -0.0113 | -0.0205 | -0.0022 |
| MAG | Denoising | Candidate | 0.4305 | -0.2012 | 0.2293 | 0.0025 | 0.0042 | 0.0071 | -0.0001 | -0.0004 | -0.0006 |
| Score-Denoise | Denoising | Candidate | 0.4313 | -0.2029 | 0.2284 | 0.0025 | 0.0041 | 0.0076 | -0.0001 | -0.0001 | -0.0006 |
| P2P-Bridge | Denoising | Candidate | 1.7345 | -0.2662 | 1.4683 | 0.0108 | 0.0181 | 0.4644 | -0.0041 | -0.0067 | -0.0133 |
| PathNet | Denoising | Mixed / recent denoising candidate | -0.0085 | -0.0504 | -0.0589 | 0.0006 | 0.0012 |  |  |  |  |
| GQE-Net | Color / texture enhancement | Candidate for texture | 0.0000 | -0.0000 | -0.0000 | -0.0000 | 0.0000 | 0.0603 | 0.0010 | 0.0010 | 0.0008 |
| Octree upsample-clean | Joint upsampling / cleaning | Candidate | 2.2532 | 0.4222 | 2.6754 | 0.0386 | 0.0385 | 0.0830 | -0.0844 | -0.2048 | -0.0093 |
| PU-Flow | Upsampling | Candidate | 0.3180 | 2.1248 | 2.4428 | 0.0132 | 0.0237 | -0.1000 | 0.2441 | 0.2752 | -0.0142 |
| PC2-PU | Upsampling | Candidate / weak-positive geometry | 0.0610 | -0.0372 | 0.0238 | 0.0007 | 0.0004 | 0.0035 | 0.0134 | 0.0501 | 0.0000 |
| Neural Points | Arbitrary upsampling | Mixed / fixed-size domain transfer | -0.2131 | 0.4235 | 0.2104 | 0.0044 | 0.0258 | -0.2855 | -0.0387 | -0.2198 | 0.0049 |
| SPU | Upsampling | Candidate | -0.0441 | 0.6883 | 0.6443 | 0.0059 | 0.0058 | -0.0029 | 0.0155 | 0.0471 | 0.0002 |
| RepKPU | Upsampling | Candidate | 0.1894 | 0.3797 | 0.5691 | 0.0036 | 0.0051 | -0.0236 | 0.0196 | 0.0615 | 0.0003 |
| SnowflakeNet-PU | Upsampling | Candidate | 0.2800 | 0.4284 | 0.7084 | 0.0050 | 0.0064 | -0.0175 | 0.0215 | 0.0645 | 0.0001 |
| PU-Gaussian | Upsampling | Candidate | 0.3658 | 0.1601 | 0.5259 | 0.0047 | 0.0060 | -0.0164 | 0.0199 | 0.0660 | 0.0003 |
| Grad-PU chunked | Upsampling | Candidate / non-default chunked | 0.1069 | 0.2469 | 0.3538 | 0.0027 | 0.0036 | 0.0078 | 0.0151 | 0.0482 | -0.0000 |
| PUFM | Upsampling | Candidate | -0.6542 | 1.0185 | 0.3644 | 0.0017 | 0.0034 | -0.0369 | 0.0270 | 0.0418 | 0.0016 |
| PUDM | Upsampling | Mixed / likely negative | 0.0246 | -0.5836 | -0.5590 | -0.0021 | -0.0044 | 0.0186 | 0.0110 | 0.0433 | -0.0001 |
| SPU-PMD | Upsampling | Candidate | 0.3844 | 0.1257 | 0.5101 | 0.0035 | 0.0049 | 0.0132 | 0.0169 | 0.0566 | -0.0003 |
| PUCRN | Upsampling | Candidate | 0.0339 | 1.0676 | 1.1015 | 0.0084 | 0.0094 | 0.0036 | 0.0189 | 0.0431 | 0.0001 |
| CRCIR after-compression 4x | Compression-derived enhancement | Candidate | 0.3966 | 0.3944 | 0.7910 | 0.0066 | 0.0079 | -0.0482 | 0.0139 | 0.0454 | 0.0008 |
| BPA | Traditional reconstruction | Optional baseline | 1.2524 | -1.2107 | 0.0417 | 0.0042 | 0.0065 |  |  |  |  |
| SPSR | Traditional reconstruction | Optional baseline | -5.2539 | 1.7090 | -3.5449 | 0.0026 | 0.0081 |  |  |  |  |
| SOR | Traditional filtering | Negative baseline | 0.0135 | -2.4019 | -2.3884 | -0.0106 | -0.0172 |  |  |  |  |
| SAL | Implicit reconstruction | Negative | -23.7566 | -118.66 | -142.42 | -0.1360 | -0.2385 |  |  |  |  |
| Points2Surf | Implicit reconstruction | Negative | -5.7831 | -19.5703 | -25.3534 | -0.0809 | -0.1365 |  |  |  |  |
| PoinTr | Completion | Negative domain transfer | -0.7008 | -10.9463 | -11.6471 | -0.1462 | -0.1152 |  |  |  |  |

Detailed machine-readable table: `results/selected10_method_summary.csv`.
Texture/perceptual metrics are available only for methods included in the texture metric pass.
