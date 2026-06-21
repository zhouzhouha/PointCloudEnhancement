"""
Run vanilla_metric evaluation from SCUTSurface on a UVG-CWI-DQPC sequence.
Uses the metrics.py with eval_type='ply' for point cloud files.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the vanilla_metric directory to path
base_dir = Path(__file__).resolve().parents[1]
vanilla_metric_dir = base_dir / "third_party" / "SCUTSurface" / "metrics" / "vanilla_metric"
sys.path.insert(0, str(vanilla_metric_dir))

# Import the eval_pointcloud function
from metrics import eval_pointcloud


DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")


def frame_id(path):
    """Return the zero-padded frame id at the end of a UVG-CWI-DQPC filename."""
    return path.stem.rsplit("_", 1)[-1]


def run_evaluation(max_frames=10, sequence="OrangeKettlebell", dataset_root=DATASET_ROOT):
    """
    Run vanilla metric evaluation on one UVG-CWI-DQPC sequence.
    """
    # Directories
    cg_dir = Path(dataset_root) / sequence / "cg" / "15fps"
    he_dir = Path(dataset_root) / sequence / "he" / "15fps"
    output_dir = base_dir / "results" / "uvg_cwi_dqpc" / sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    cg_by_frame = {frame_id(path): path for path in cg_dir.glob("*.ply")}
    he_by_frame = {frame_id(path): path for path in he_dir.glob("*.ply")}
    common_frames = sorted(set(cg_by_frame) & set(he_by_frame))
    if max_frames is not None:
        common_frames = common_frames[:max_frames]
    
    if not common_frames:
        print("ERROR: No files found!")
        print(f"CG directory: {cg_dir}")
        print(f"HE directory: {he_dir}")
        return
    
    print("="*80)
    print("VANILLA METRIC EVALUATION (from SCUTSurface)")
    print("="*80)
    print(f"\nDataset root: {dataset_root}")
    print(f"Sequence: {sequence}")
    print(f"Frames to evaluate: {len(common_frames)}")
    print(f"Ground Truth (HE): {he_dir}")
    print(f"Low Quality (CG):  {cg_dir}")
    print()
    print(f"Using thresholds: [5, 10, 20] mm")
    print(f"Sample points: Will use actual point count from files")
    print()
    
    # Process each frame
    results = []
    
    for i, frame in enumerate(common_frames):
        cg_file = cg_by_frame[frame]
        he_file = he_by_frame[frame]

        print(f"[{i+1}/{len(common_frames)}] Processing frame {frame}...")
        print(f"  CG: {cg_file.name}")
        print(f"  HE: {he_file.name}")
        
        try:
            # Run evaluation using metrics.py with eval_type='ply'
            out_dict = eval_pointcloud(
                pre_mesh_ply=str(cg_file),
                gt_mesh_ply=str(he_file),
                eval_type='ply',     # Use 'ply' for point cloud files
                thresholds=[5, 10, 20]
            )
            
            # Add frame info
            out_dict['frame'] = frame
            out_dict['cg_file'] = cg_file.name
            out_dict['he_file'] = he_file.name
            
            results.append(out_dict)
            
            # Print results for this frame
            print(f"    Chamfer-L1: {out_dict['chamfer-L1']:.4f} mm")
            print(f"    Chamfer-L2: {out_dict['chamfer-L2']:.4f} mm")
            print(f"    F-score @ 5mm:  {out_dict['F_5']:.4f}")
            print(f"    F-score @ 10mm: {out_dict['F_10']:.4f}")
            print(f"    F-score @ 20mm: {out_dict['F_20']:.4f}")
            print()
            
        except Exception as e:
            print(f"    ERROR: {e}")
            print()
            continue
    
    if not results:
        print("No results generated!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    cols_order = ['frame', 'cg_file', 'he_file', 
                  'chamfer-L1', 'chamfer-L2', 'chamferL2_old',
                  'CD_Acc', 'CD_Comp',
                  'P_5', 'R_5', 'F_5',
                  'P_10', 'R_10', 'F_10',
                  'P_20', 'R_20', 'F_20',
                  'N_Acc', 'N_Comp', 'normals']
    
    # Only keep columns that exist
    cols_order = [col for col in cols_order if col in df.columns]
    df = df[cols_order]
    
    # Save to CSV
    output_csv = output_dir / "per_frame_metrics.csv"
    df.to_csv(output_csv, index=False)
    
    print("="*80)
    print("SUMMARY (Average over all frames)")
    print("="*80)
    print()
    
    # Print averages
    metrics_to_show = [
        ('Chamfer-L1 (mm)', 'chamfer-L1'),
        ('Chamfer-L2 (mm²)', 'chamfer-L2'),
        ('Chamfer-L2-old (mm)', 'chamferL2_old'),
        ('Accuracy (mm)', 'CD_Acc'),
        ('Completeness (mm)', 'CD_Comp'),
        ('Precision @ 5mm', 'P_5'),
        ('Recall @ 5mm', 'R_5'),
        ('F-score @ 5mm', 'F_5'),
        ('Precision @ 10mm', 'P_10'),
        ('Recall @ 10mm', 'R_10'),
        ('F-score @ 10mm', 'F_10'),
        ('Precision @ 20mm', 'P_20'),
        ('Recall @ 20mm', 'R_20'),
        ('F-score @ 20mm', 'F_20'),
        ('Normal Accuracy', 'N_Acc'),
        ('Normal Completeness', 'N_Comp'),
        ('Normal Correctness', 'normals'),
    ]
    
    for label, col in metrics_to_show:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{label:<30} {mean_val:>10.4f} ± {std_val:.4f}")
    
    print()
    print("="*80)
    print(f"✓ Results saved to: {output_csv}")
    print("="*80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run vanilla metric evaluation")
    parser.add_argument('--frames', type=int, default=10,
                       help='Number of frames to evaluate (default: 10)')
    parser.add_argument('--sequence', default='OrangeKettlebell',
                       help='UVG-CWI-DQPC sequence name (default: OrangeKettlebell)')
    parser.add_argument('--dataset-root', default=str(DATASET_ROOT),
                       help='Path to UVG-CWI-DQPC dataset root')
    args = parser.parse_args()
    
    run_evaluation(
        max_frames=args.frames,
        sequence=args.sequence,
        dataset_root=Path(args.dataset_root),
    )


if __name__ == "__main__":
    main()
