"""
Apply basic_enhancement.py pipeline to all BouncingBlue CG frames.
Saves enhanced point clouds as PLY files for later metric computation.
"""
import os
import glob
import open3d as o3d

# Paths
CG_DIR = r"C:\Xuemei\2024Tampare_Intership\GrandChallenge\BouncingBlue_UVG-CWI-DQPC_v2-0_CG_15fps\UVG-CWI-DQPC\BouncingBlue\consumer-grade_capture_system\CGv2\15fps"
OUTPUT_DIR = r"C:\Xuemei\2024Tampare_Intership\GrandChallenge\BouncingBlue_enhanced_basic"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get all CG ply files sorted
cg_files = sorted(glob.glob(os.path.join(CG_DIR, "*.ply")))
print(f"Found {len(cg_files)} CG frames to process")

for i, cg_path in enumerate(cg_files):
    fname = os.path.basename(cg_path)
    out_path = os.path.join(OUTPUT_DIR, fname)

    # 1. Load point cloud (XYZRGB)
    pcd = o3d.io.read_point_cloud(cg_path)
    n_before = len(pcd.points)

    # 2. Statistical Outlier Removal
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 3. Radius outlier removal (radius in mm, matching point cloud scale)
    pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=20.0)

    # 4. Voxel resampling (stabilize density, voxel_size in mm)
    pcd = pcd.voxel_down_sample(voxel_size=3.0)

    n_after = len(pcd.points)

    # 5. Save output (still XYZRGB)
    o3d.io.write_point_cloud(out_path, pcd)

    print(f"[{i+1}/{len(cg_files)}] {fname}: {n_before} -> {n_after} points")

print(f"\nDone! Enhanced point clouds saved to: {OUTPUT_DIR}")
