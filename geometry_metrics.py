import os
import re
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# ------------------------------------------------------
# 1. Utility: Load point cloud (works for any PLY format)
# ------------------------------------------------------
def load_pointcloud(filepath):
    """Load PLY file with Open3D (supports color)."""
    pcd = o3d.io.read_point_cloud(filepath)
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load points from {filepath}")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors


# ------------------------------------------------------
# 2. Distance Metrics
# ------------------------------------------------------
def chamfer_distance(pc1, pc2):
    """Compute Chamfer Distance between two point sets."""
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)
    return np.mean(dist1 ** 2) + np.mean(dist2 ** 2)


def hausdorff_distance(pc1, pc2):
    """Compute Hausdorff distance between two point sets."""
    dists = cdist(pc1, pc2)
    return max(dists.min(axis=1).max(), dists.min(axis=0).max())


# ------------------------------------------------------
# 3. Color Distance (optional)
# ------------------------------------------------------
def color_distance(c1, c2):
    """Compute average RGB difference between two point sets."""
    if c1 is None or c2 is None:
        return np.nan
    min_len = min(len(c1), len(c2))
    return np.mean(np.linalg.norm(c1[:min_len] - c2[:min_len], axis=1))


# ------------------------------------------------------
# 4. Precision, Recall, Completeness, F-score
# ------------------------------------------------------
def precision_recall_fscore(pc_ref, pc_test, threshold=0.5):
    """
    Compute precision, recall, completeness, and F-score.
    threshold: distance threshold for 'match'
    """
    tree_ref = cKDTree(pc_ref)
    tree_test = cKDTree(pc_test)

    dist_ref, _ = tree_ref.query(pc_test)
    dist_test, _ = tree_test.query(pc_ref)

    precision = np.mean(dist_ref < threshold)
    recall = np.mean(dist_test < threshold)
    completeness = recall
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return precision, recall, completeness, fscore


# ------------------------------------------------------
# 5. Extract frame number (last 4 digits)
# ------------------------------------------------------
def extract_frame_number(filename):
    """Extract the last 4 consecutive digits from the filename."""
    match = re.search(r'(\d{4})(?!.*\d)', filename)
    return int(match.group(1)) if match else -1


# ------------------------------------------------------
# 6. Main Evaluation Loop
# ------------------------------------------------------
def evaluate_pointclouds(high_dir, low_dir, output_csv="metrics_results.csv", threshold=0.5):
    results = []

    high_files = sorted([f for f in os.listdir(high_dir) if f.endswith(".ply")])
    low_files = sorted([f for f in os.listdir(low_dir) if f.endswith(".ply")])

    if len(high_files) != len(low_files):
        print("⚠️ Warning: the number of files in both folders is different.")

    for fname in high_files:
        high_path = os.path.join(high_dir, fname)
        low_path = os.path.join(low_dir, fname)
        if not os.path.exists(low_path):
            print(f"Skipping {fname}, not found in low_quality folder.")
            continue

        print(f"Processing {fname} ...")

        # Load point clouds
        pc_high, col_high = load_pointcloud(high_path)
        pc_low, col_low = load_pointcloud(low_path)

        # Compute metrics
        cd = chamfer_distance(pc_high, pc_low)
        hd = hausdorff_distance(pc_high, pc_low)
        color_diff = color_distance(col_high, col_low)
        precision, recall, completeness, fscore = precision_recall_fscore(pc_high, pc_low, threshold)
        frame_id = extract_frame_number(fname)

        results.append({
            "name": fname,
            "frame_id": frame_id,
            "Chamfer": cd,
            "Hausdorff": hd,
            "ColorDiff": color_diff,
            "Precision": precision,
            "Recall": recall,
            "Completeness": completeness,
            "F-score": fscore
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Metrics saved to {output_csv}")
    return df


# ------------------------------------------------------
# 7. CLI Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute metrics between two sets of PLY point clouds.")
    parser.add_argument("--root", type=str, default="GrandChallenge/Data", help="Root directory containing 'high_quality' and 'low_quality' folders")
    parser.add_argument("--threshold", type=float, default=0.5, help="Distance threshold for precision/recall")
    parser.add_argument("--output", type=str, default="metrics_results.csv", help="Output CSV file")
    args = parser.parse_args()

    high_dir = os.path.join(args.root, "high_quality")
    low_dir = os.path.join(args.root, "low_quality")

    evaluate_pointclouds(high_dir, low_dir, args.output, args.threshold)
