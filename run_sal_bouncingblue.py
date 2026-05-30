"""
SAL reconstruction pipeline for BouncingBlue.
Uses basic-enhanced CG point clouds as input, HE as ground truth reference.
Outputs reconstructed point clouds (PLY) for metric evaluation.

Usage:
  python run_sal_bouncingblue.py --frame 0 --epochs 2000
  python run_sal_bouncingblue.py --frame 0 --epochs 500 --resolution 128  # quick test
"""
import sys
import os
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

# Paths
BASE_DIR = Path(r"C:\Xuemei\2024Tampare_Intership\GrandChallenge")
ENH_DIR = BASE_DIR / "BouncingBlue_enhanced_basic"
HE_DIR = BASE_DIR / "BouncingBlue_UVG-CWI-DQPC_v1-0_HE_15fps" / "UVG-CWI-DQPC" / "BouncingBlue" / "high-end_capture_system" / "HE" / "15fps"
SAL_DATA_DIR = BASE_DIR / "third_party" / "SCUTSurface" / "reconstruction" / "SAL" / "data" / "bouncingblue"
SAL_CODE_DIR = BASE_DIR / "third_party" / "SCUTSurface" / "reconstruction" / "SAL" / "code"
OUTPUT_DIR = BASE_DIR / "BouncingBlue_SAL_reconstructed"


def prepare_frame(frame_idx, enh_files, he_files):
    """Prepare enhanced CG frame for SAL: normalize to unit sphere."""
    points_dir = SAL_DATA_DIR / "points" / "BouncingBlue"
    points_iou_dir = SAL_DATA_DIR / "points_iou" / "BouncingBlue"
    points_dir.mkdir(parents=True, exist_ok=True)
    points_iou_dir.mkdir(parents=True, exist_ok=True)

    enh_file = enh_files[frame_idx]
    he_file = he_files[frame_idx]

    print(f"Preparing frame {frame_idx}...")
    print(f"  Enhanced CG: {enh_file}")
    print(f"  HE ref:      {he_file}")

    # Load point clouds
    enh_pcd = o3d.io.read_point_cloud(str(enh_file))
    he_pcd = o3d.io.read_point_cloud(str(he_file))

    enh_pts = np.asarray(enh_pcd.points)
    he_pts = np.asarray(he_pcd.points)

    print(f"  Enhanced points: {len(enh_pts)}")
    print(f"  HE points:       {len(he_pts)}")

    # Center and normalize using both sets for consistent scale
    all_pts = np.vstack([enh_pts, he_pts])
    center = all_pts.mean(axis=0)
    scale = np.abs(all_pts - center).max()

    enh_norm = (enh_pts - center) / scale
    he_norm = (he_pts - center) / scale

    # Save as .xyz
    enh_output = points_dir / f"frame_{frame_idx:04d}.xyz"
    he_output = points_iou_dir / f"frame_{frame_idx:04d}.xyz"

    np.savetxt(enh_output, enh_norm, fmt="%.8f")
    np.savetxt(he_output, he_norm, fmt="%.8f")

    # Save normalization params for later rescaling
    norm_file = SAL_DATA_DIR / f"norm_params_{frame_idx:04d}.npz"
    np.savez(norm_file, center=center, scale=scale)

    print(f"  Center: {center}")
    print(f"  Scale:  {scale:.2f}")
    return center, scale


def create_config(frame_idx, epochs, resolution):
    """Create SAL config for this frame."""
    config_content = f"""train{{
    plot_frequency = 100
    preprocess = True
    auto_decoder = False
    latent_size = 0
    expname = BouncingBlue_frame_{frame_idx:04d}
    dataset_path = ../data/bouncingblue/points/BouncingBlue/frame_{frame_idx:04d}.xyz
    adjust_lr = False
    dataset = datasets.recon_dataset.ReconDataSet
    data_split = none

    learning_rate_schedule = [{{ "Type" : "Step",
                              "Initial" : 0.0005,
                               "Interval" : 500,
                                "Factor" : 0.5
                            }},
                            {{
                                "Type" : "Step",
                                "Initial" : 0.001,
                                "Interval" : 500,
                                "Factor" : 0.5
                            }}]
    network_class = model.network.SALNetwork
}}

plot{{
    resolution = {resolution}
    mc_value = 0.0
    is_uniform_grid = True
    verbose = False
    save_html = True
    save_ply = True
    overwrite = True
}}

network{{
    decode_mnfld_pnts = False
    encoder{{
    }}
    decoder
    {{
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ]
        dropout = []
        dropout_prob = 0.2
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        latent_in = []
        xyz_in_all = False
        activation = None
        latent_dropout = False
        weight_norm = True
    }}

    loss{{
        loss_type = model.loss.SALLoss
        properties{{
            manifold_pnts_weight = 0
            unsigned = True
        }}
    }}
}}
"""
    config_path = SAL_CODE_DIR / "confs" / f"bouncingblue_frame_{frame_idx:04d}.conf"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Config: {config_path}")
    return config_path


def train_sal(config_path, epochs):
    """Train SAL."""
    import subprocess

    print(f"\n{'='*70}")
    print(f"Training SAL for {epochs} epochs (GPU mode)...")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable,
        "training/exp_runner.py",
        "--batch_size", "1",
        "--nepoch", str(epochs),
        "--conf", str(Path(config_path).absolute()),
        "--workers", "1",
        "--gpu", "0",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SAL_CODE_DIR))
    return result.returncode == 0


def evaluate_sal(config_path, frame_idx, epochs):
    """Evaluate SAL and extract mesh."""
    import subprocess

    print(f"\n{'='*70}")
    print(f"Evaluating SAL at checkpoint {epochs}...")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable,
        "evaluate/evaluate.py",
        "--conf", str(Path(config_path).absolute()),
        "--checkpoint", str(epochs),
        "--split", "none",
    ]

    result = subprocess.run(cmd, cwd=str(SAL_CODE_DIR))
    return result.returncode == 0


def rescale_and_save(frame_idx, center, scale, num_sample_points=500000):
    """Find SAL output mesh, rescale to original coords, save as PLY point cloud."""
    import trimesh

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find the mesh output
    exp_base = SAL_CODE_DIR / "exps" / f"BouncingBlue_frame_{frame_idx:04d}"
    ply_files = list(exp_base.rglob("*.ply"))

    if not ply_files:
        print(f"ERROR: No PLY mesh found in {exp_base}")
        return None

    mesh_file = ply_files[0]
    print(f"Found mesh: {mesh_file}")

    mesh = trimesh.load(mesh_file)
    vertices = np.array(mesh.vertices)

    # Sample points from mesh surface if it has faces
    if hasattr(mesh, "faces") and len(mesh.faces) > 0 and num_sample_points:
        points, face_idx = trimesh.sample.sample_surface(mesh, num_sample_points)
    else:
        points = vertices

    # Denormalize back to original mm coordinates
    points_rescaled = points * scale + center

    # Save as PLY with Open3D (compatible with metric scripts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_rescaled)

    out_path = OUTPUT_DIR / f"BouncingBlue_UVG-CWI-DQPC_CG_15_0_156_{frame_idx:04d}.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved reconstructed point cloud ({len(points_rescaled)} pts): {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SAL reconstruction for BouncingBlue")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (0-156)")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--resolution", type=int, default=256, help="Marching cubes resolution")
    parser.add_argument("--skip_train", action="store_true", help="Skip training, use existing checkpoint")
    parser.add_argument("--sample_points", type=int, default=500000, help="Points to sample from mesh")
    args = parser.parse_args()

    # Get file lists
    enh_files = sorted(ENH_DIR.glob("*.ply"))
    he_files = sorted(HE_DIR.glob("*.ply"))

    if not enh_files:
        print("ERROR: No enhanced files found. Run run_basic_enhancement_bouncingblue.py first.")
        return 1
    if args.frame >= len(enh_files):
        print(f"ERROR: Frame {args.frame} not found. Available: 0-{len(enh_files)-1}")
        return 1

    print(f"{'#'*70}")
    print(f"SAL Reconstruction - BouncingBlue Frame {args.frame}")
    print(f"{'#'*70}\n")

    # Step 1: Prepare data
    center, scale = prepare_frame(args.frame, enh_files, he_files)

    # Step 2: Create config
    config_path = create_config(args.frame, args.epochs, args.resolution)

    # Step 3: Train
    if not args.skip_train:
        if not train_sal(config_path, args.epochs):
            print("Training failed!")
            return 1

    # Step 4: Evaluate (extract mesh)
    if not evaluate_sal(config_path, args.frame, args.epochs):
        print("Evaluation failed!")
        return 1

    # Step 5: Rescale and save
    out_path = rescale_and_save(args.frame, center, scale, args.sample_points)
    if out_path:
        print(f"\nDone! Reconstructed point cloud: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
