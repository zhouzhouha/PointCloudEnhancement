import open3d as o3d

# 1. Load point cloud (XYZRGB)
pcd = o3d.io.read_point_cloud("input.ply")

# 2. Statistical Outlier Removal (core baseline)
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=20,     # standard choice
    std_ratio=2.0        # widely used threshold
)

# 3. Radius outlier removal (extra robustness)
pcd, ind = pcd.remove_radius_outlier(
    nb_points=16,
    radius=0.05
)

# 4. Optional: voxel resampling (stabilize density)
pcd = pcd.voxel_down_sample(voxel_size=0.005)

# 5. Save output (still XYZRGB)
o3d.io.write_point_cloud("output.ply", pcd)