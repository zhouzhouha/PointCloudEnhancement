# Enhancement Survey Applicability to UVG-CWI-DQPC

This note summarizes enhancement methods from three survey/source tracks and
judges whether each family can be applied to the UVG-CWI-DQPC benchmark.

Sources read:

- Siwen Quan et al., "Deep Learning for 3D Point Cloud Enhancement: A Survey", arXiv:2411.00857.
- Chen Jianwen et al., "Deep learning-based quality enhancement for 3D point clouds: a survey", Journal of Image and Graphics, 2023, DOI `10.11834/jig.221076`.
- Wei Gao and Ge Li, "Deep-Learning-Based Point Cloud Enhancement II", in *Deep Learning for 3D Point Clouds*, Springer, 2024. The full chapter is subscription-limited from the current environment, so this note uses the visible abstract and reference list only.
- Local survey repo: `third_party/Point_cloud_quality_enhancement`.

## Dataset Assumptions

UVG-CWI-DQPC provides paired dynamic point cloud frames:

- Input: low captured/CG XYZRGB PLY.
- Reference: high-quality/HE XYZRGB PLY.
- Current toy sequence: `OrangeKettlebell`, frames `0000`, `0010`, ..., `0090`.

For fair benchmarking, method core settings should remain official/default. Adapters may only convert file format, split/merge large frames when the method already supports patch processing, transfer color by nearest original CG point with `k=1` for geometry-only outputs, and evaluate with the same UVG-CWI metric runner.

This survey benchmark focuses on point-cloud-only enhancement: degraded point cloud in, enhanced point cloud out. Multimodal methods that additionally require RGB images, depth images, or RGB-D inputs are outside the main benchmarking scope because they introduce extra requirements for image availability, calibration, foreground alignment, and modality-specific preprocessing. SuperPC has been read and is excluded from the main benchmark under this rule: it formulates restoration as an image-and-point-cloud conditioned diffusion problem using image-point-cloud fusion to construct raw, local, and global conditions. It can be mentioned only as an excluded multimodal/background method if the survey discusses scope boundaries.

## Important Findings From The Surveys

- The 2024 arXiv survey defines point cloud enhancement as producing clean, complete, dense points from low-quality raw point clouds, with three main tasks: denoising, completion, and upsampling.
- The 2024 arXiv survey states that it primarily studies geometry-only point clouds, not RGB/intensity attributes.
- The 2023 Journal of Image and Graphics survey explicitly warns that most methods focus on single-frame geometry, ignore temporal correlation, and ignore attributes such as color or intensity.
- The Springer chapter is closer to compression/restoration and dynamic point cloud enhancement. Its visible references include geometry post-processing, compression artifact removal, dynamic attribute enhancement, and attribute compression/restoration methods, which are more relevant to our texture concern but may require codec-specific inputs.

## Applicability Scale

- `High`: method input/output naturally matches UVG frames or can be adapted with format conversion and `k=1` color transfer.
- `Medium`: can be run, but assumptions differ strongly from UVG-CWI-DQPC; results should be labeled as domain transfer.
- `Low`: not a clean benchmark method for this dataset without changing the problem setting, retraining, or generating extra supervision.
- `Hold`: source/checkpoint/setup is unavailable or not worth running before stronger candidates.

## Denoising Methods

These are the best first external enhancement candidates because UVG-CWI-DQPC CG frames are noisy/incomplete but already contain many points. Output is usually the same or similar point count with improved geometry. Most are geometry-only.

| Method | Source category | Typical input | Typical output | Code/pretrained status | UVG applicability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| PointCleanNet | Denoising/outlier removal | Dense noisy XYZ point cloud, local patches | Denoised XYZ, may move/remove points | Code and pretrained model available locally; smoke running | High as external baseline | Already running on frame `0000`; record as external, not SCUTSurface. Needs `k=1` color transfer. |
| Pointfilter | Denoising | Noisy XYZ point cloud and neighborhoods | Filtered/denoised XYZ | Code listed in both survey tracks | High | Good next denoising candidate after PointCleanNet if checkpoint/setup is usable. Geometry-only. |
| Total Denoising | Unsupervised denoising | Noisy XYZ point cloud | Cleaned XYZ | Code listed | Medium-High | Useful because unsupervised; may avoid domain mismatch from pretrained object datasets. Need inspect scalability to 500k-point frames. |
| DMRDenoise / DMR | Denoising | Noisy XYZ point cloud | Denoised/downsampled XYZ | Code and checkpoint local | Medium; current default failed | Official default failed on UVG frame due internal KNN/patch tensor mismatch. Do not tune unless separately labeled. |
| Score-Based Point Cloud Denoising | Denoising | Noisy XYZ point cloud; large-cloud script exists | Denoised XYZ | Code and checkpoint local; environment not built | High if environment can be built | Strong candidate, but needs old PyTorch3D/torch-cluster environment. Hold until external phase. |
| PD-Flow | Denoising | Noisy XYZ point cloud | Denoised XYZ | Code listed by 2024 survey | Medium-High | Flow-based denoising. Need inspect checkpoint and large-cloud support. |
| IterativePFN | Denoising | Noisy XYZ point cloud | Denoised XYZ | Code listed by 2024 survey | Medium-High | Iterative filtering is conceptually aligned with not over-reconstructing. Need checkpoint check. |
| StraightPCF | Denoising | Noisy XYZ point cloud | Denoised XYZ | Code listed by 2024 survey | Medium | Recent method; inspect availability and inference script. |
| PD-LTS | Denoising | Noisy XYZ point cloud | Denoised XYZ | Code listed by 2024 survey | Medium | Likely research-code setup; inspect after stronger candidates. |
| Neural Projection | Local surface denoising | Noisy XYZ point cloud | Projected/denoised XYZ | Survey code link failed locally | Hold | The listed GitHub repo was unavailable when tried. |
| GPDNet | Graph denoising | Noisy XYZ point cloud | Denoised XYZ | Code listed in 2023 survey; 2024 table has no URL | Hold/Medium | Need verify active repo and checkpoint. |
| RePCD-Net, NoiseTrans, SVCNet, MODNet, LPCDNet, 3DMambaIPF, Noise4Denoise | Denoising | Usually synthetic noisy XYZ patches | Denoised XYZ | Mixed or unavailable code | Hold | Include in literature review, but do not prioritize without code/checkpoints. |

## Upsampling / Super-Resolution Methods

These increase point density. They are relevant because UVG-CWI-DQPC CG can be sparse, but many methods are trained on normalized synthetic object patches and may alter density distribution in ways that hurt Chamfer/completeness metrics.

| Method | Source category | Typical input | Typical output | Code/pretrained status | UVG applicability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| ZSPU | Zero-shot upsampling | Single sparse XYZ point cloud | Denser XYZ | Code listed | High-Medium | Attractive because no supervised pretrained domain is required. Need inspect runtime/scalability. |
| PU-GCN | GCN upsampling | Sparse object/patch XYZ | Upsampled XYZ at fixed ratio | Code listed | Medium | Strong benchmark method, but synthetic object assumptions. Use patch adapter and `k=1` color transfer. |
| PUGeo-Net | Geometry-centric upsampling | Sparse XYZ patches | Dense XYZ | Code listed | Medium | Geometry-aware; may preserve surfaces better than older PU-Net. |
| MPU / 3PU | Patch-based progressive upsampling | Sparse XYZ patches | Dense XYZ | Code listed | Medium | Patch-based nature may fit large frames, but old dependencies likely. |
| PU-Net | CNN upsampling | Sparse fixed-size object patches | Dense XYZ | Code listed | Medium-Low | Classic baseline; old TensorFlow setup and synthetic object bias. |
| PU-GAN | GAN upsampling | Sparse object patches | Dense XYZ | Code listed | Medium-Low | Can hallucinate/detail-shift; use carefully as external baseline. |
| Dis-PU | Disentangled refinement upsampling | Sparse XYZ patches | Dense XYZ | Code listed | Medium | Good candidate if pretrained inference is available. |
| Flexible-PU / Meta-PU | Arbitrary-scale upsampling | Sparse XYZ, desired scale | Dense XYZ | Code listed | Medium | Scale flexibility is useful for matching CG/HE point counts. |
| SAPCU / NePs / NeuralPoints | Self-supervised/implicit upsampling | Sparse XYZ | Dense XYZ | Code listed | Medium-High | Self-supervised/adaptive methods may generalize better; inspect code and runtime. |
| Grad-PU | Gradient-driven upsampling | Sparse XYZ | Dense XYZ | Code listed | Medium | Recent; inspect checkpoint and large-cloud handling. |
| PUDM | Diffusion upsampling | Sparse XYZ | Dense XYZ | Code listed | Medium-Low | May be expensive and trained on synthetic objects. |
| PU-Mask, APU-LDI, TP-NoDe, PU-CRN, PU-CycGAN, PU-GACNet, PUFA-GAN, PU-Dense, PC2-PU | Upsampling variants | Usually sparse XYZ patches | Dense XYZ | Code listed | Hold | Later candidates after one representative upsampling family works. |
| VPU / Sequential temporal point cloud upsampling | Video/temporal upsampling | Point cloud sequence | Dense sequence | Paper listed; code not obvious | High conceptually, Hold practically | Most relevant to dynamic UVG because it uses temporal correlation, but needs source/checkpoint verification. |
| LiUpNet / density-imbalance LiDAR upsampling | LiDAR upsampling | Sparse LiDAR-style scans | Dense LiDAR-style points | Paper listed in 2023 JIG refs | Medium | Scene/LiDAR oriented, but UVG dense object/scene geometry differs from LiDAR. |

## Completion Methods

Completion fills missing regions. It is less clean for our current benchmark because most methods assume partial object input and complete object output from ShapeNet/MVP-style categories. UVG-CWI-DQPC frames are real dynamic captures with color and arbitrary sampling, not normalized CAD partial views.

| Method | Source category | Typical input | Typical output | Code/pretrained status | UVG applicability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| PoinTr | Point-based completion | Partial object XYZ, often fixed point count | Complete object XYZ | Code listed | Medium-Low | Can be tested as domain transfer if we normalize/sample frames. It may hallucinate shape priors and worsen metric. |
| SnowflakeNet | Completion | Partial object XYZ | Complete object XYZ | Code listed | Medium-Low | Same issue as PoinTr; object-category prior may not match UVG. |
| SeedFormer | Completion | Partial object XYZ | Complete dense object XYZ | Code listed in JIG refs | Medium-Low | Strong completion baseline but domain assumptions are large. |
| GRNet | Voxel/point completion | Partial object, voxel grid | Complete object | Code listed | Low | Voxelization/fixed shape assumptions; likely poor for full UVG frames. |
| PCN | Completion | Partial ShapeNet object | Coarse/fine complete object | Code listed | Low-Medium | Classic baseline; only run if we explicitly want completion domain-transfer comparison. |
| TopNet, MSN, Folding/Atlas-based, PMP-Net/PMP-Net++, VRCNet, ASFM-Net, RFNet, SoftPoolNet, Cycle4Completion, SpareNet, LAKe-Net | Completion variants | Partial object point set | Complete object point set | Mixed code availability | Hold | Too many shape-completion assumptions; not first-line for UVG. |
| QiNet, Flow-based completion with adversarial refinement | Completion/enhancement | Corrupted point clouds | Completed XYZ | Springer/JIG references | Medium | More enhancement-framed than ShapeNet-only completion; inspect if code/checkpoint exists. |

## Compression / Attribute / Texture-Oriented Enhancement

These are important because our current geometry-only methods ignore texture. They may not be directly comparable to denoising/completion because many assume decoded G-PCC/V-PCC compressed input rather than low-captured CG input.

| Method | Source category | Typical input | Typical output | Code/pretrained status | UVG applicability | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Fast inter-frame motion prediction for compressed dynamic point cloud attribute enhancement | Dynamic attribute enhancement | Compressed dynamic point cloud sequence with attributes | Enhanced attributes/color | Springer reference visible | High conceptually | Strong match to texture and temporal concerns, but likely codec-specific. Need paper/code search. |
| DANet | Geometry-based point cloud compression artifact removal | Decompressed geometry | Restored geometry | Springer reference visible | Medium-High if code exists | Good match if CG behaves like degraded/decompressed geometry. Need source/checkpoint. |
| Deep geometry post-processing for decompressed point clouds | Compression artifact removal | Decompressed geometry | Post-processed geometry | Springer reference visible | Medium | Similar to DANet; likely codec-specific. |
| TDRnet | Transformer dual-branch restoration for G-PCC artifacts | Geometry-compressed point cloud | Restored geometry | Springer reference visible | Medium | Relevant for compression artifact benchmarks, not necessarily capture noise. |
| PCAC-GAN | Attribute compression enhancement | Compressed color attributes | Enhanced color attributes | Springer reference visible | Medium-High for color | Need code/checkpoint. Could be more useful than geometry-only methods if input assumptions fit. |
| GQE-Net | Graph quality enhancement for point cloud color attribute | Distorted/compressed color point cloud patches | Enhanced color attributes | Related arXiv search result | Medium-High for texture | Candidate for texture-only enhancement; geometry may be fixed. |
| Texture-guided graph transform / diffusion-based texture-aware intra prediction / attribute-guided GFT | Attribute compression | Codec blocks/attributes | Better attribute coding/restoration | Springer references visible | Low-Medium | Mostly coding tools rather than post-enhancement methods. |

## Current Recommended Order

Keep two tracks separate.

### Track A: Finish SCUTSurface/SUSTech-First Methods

1. SAL: already tested, worse/aside.
2. IGR: already tested, worse/aside.
3. Points2Surf: already tested, worse/aside.
4. DSE-Meshing / DeepMLS / DeepSDF / Occupancy Networks: only attempt if setup is feasible and the method has a fair raw-point-cloud inference path.

### Track B: External Enhancement Methods From Surveys

1. Finish current PointCleanNet smoke only; record as external.
2. Denoising next: Pointfilter, Total Denoising, Score-Denoise, PD-Flow, IterativePFN.
3. Upsampling next: ZSPU, PU-GCN, PUGeo-Net, MPU/3PU, Flexible-PU.
4. Temporal/texture next: VPU or dynamic attribute enhancement methods if code/checkpoints exist.
5. Completion last: PoinTr, SnowflakeNet, SeedFormer, GRNet, PCN, only as domain-transfer experiments.

## Adapter Rules For All External Methods

- Do not change method internals for metric improvement.
- Preserve official/default inference settings where possible.
- Use selected 10-frame OrangeKettlebell protocol first.
- For geometry-only outputs, transfer RGB from CG input using nearest neighbor `k=1`, no averaging.
- Record whether point count changes.
- Record whether the method is single-frame, temporal, geometry-only, attribute-only, or joint geometry+attribute.
- Label completion methods as domain-transfer if they require normalized object categories or fixed ShapeNet-style point counts.
