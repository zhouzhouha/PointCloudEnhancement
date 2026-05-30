# SAL Reconstruction Setup Complete

## Summary

✓ **SAL cloned** to `third_party/SCUTSurface/reconstruction/SAL`  
✓ **Dependencies installed**: PyTorch, pyhocon, trimesh, plotly, scikit-image, GPUtil, point_cloud_utils, plyfile  
✓ **Data prepared**: 20 object pairs aligned and ready in SAL's data structure  
✓ **Scripts created**: alignment checker, data preprocessor, training script  

## Your Data

**Location**: `third_party/SCUTSurface/reconstruction/SAL/data/real_objects/`

- **Scans**: `points/<object_name>/<object_name>_scan.xyz`
- **Ground Truth**: `points_iou/<object_name>/<object_name>_gt.xyz`

**Available objects** (20 total):
- bottle_shampoo
- bowl_chinese
- cloth_duck
- coffe_bottle_metal
- coffe_bottle_plastic
- cup1
- flower_pot
- flower_pot_2
- gift_box
- lock_pengfen
- marker
- mouse_two
- rabbit
- romoter
- screwnew
- tap2
- toy_cat
- toy_duck
- wrench
- xiaojiejie2

## Alignment Status

All scan-GT pairs have been **center-aligned**. Scale ratios are within 10% (good alignment).

Sample alignment results:
- bottle_shampoo: offset was 9.76 → aligned ✓
- toy_cat: offset was 8.44 → aligned ✓
- cup1: offset was 1.48 → aligned ✓

## Quick Start - Train Your First Reconstruction

### Option 1: Using the training script (recommended)

```powershell
# Train on bottle_shampoo (500 epochs for quick test)
python scripts/train_sal_object.py bottle_shampoo --epochs 500

# Or train with default 2000 epochs
python scripts/train_sal_object.py bottle_shampoo
```

### Option 2: Manual training

```powershell
# Navigate to SAL code directory
cd third_party/SCUTSurface/reconstruction/SAL/code

# Train (replace bottle_shampoo with your object name)
python training/exp_runner.py --batch_size 1 --nepoch 2000 --conf confs/real_objects.conf --workers 1

# Note: Edit confs/real_objects.conf first to set the correct dataset_path
```

### After training - Extract mesh

```powershell
cd third_party/SCUTSurface/reconstruction/SAL/code

# Evaluate at checkpoint 2000 (or whatever epoch you trained to)
python evaluate/evaluate.py --conf confs/bottle_shampoo.conf --checkpoint 2000 --split none
```

The reconstructed mesh will be saved in:
`third_party/SCUTSurface/reconstruction/SAL/code/exps/real_objects_<object_name>/evaluation/`

## Useful Scripts

### Check alignment between scan and GT

```powershell
python scripts/check_alignment.py dataset/real_object_scan/real_object_scan/bottle_shampoo_pcd.ply dataset/real_object_GT/real_gt/bottle_shampoo.xyz
```

### Re-prepare data (if needed)

```powershell
python scripts/prepare_sal_data.py
```

## Training Notes

- **CPU Training**: Your environment uses PyTorch CPU version. Training will be slower than GPU but will work.
- **Epochs**: SAL paper uses 2000 epochs. You can start with 500-1000 for testing.
- **Batch size**: Set to 1 for single object reconstruction.
- **Output**: Checkpoints and plots saved in `SAL/code/exps/real_objects_<name>/`

## Troubleshooting

**If training fails with import errors:**
```powershell
# Ensure you're using the virtual environment
& C:/Xuemei/2024Tampare_Intership/GrandChallenge/.venv/Scripts/Activate.ps1

# Check dependencies
pip list | Select-String -Pattern "torch|pyhocon|trimesh"
```

**If CUDA errors occur:**
SAL will use CPU automatically if CUDA is not available. Training will be slower but functional.

**If file not found errors:**
Make sure you're running from the repository root directory:
```powershell
cd c:\Xuemei\2024Tampare_Intership\GrandChallenge
```

## Next Steps

1. **Start with one object** to test the pipeline (e.g., `cup1` which had good alignment)
2. **Monitor training** in the terminal output - loss should decrease
3. **Generate mesh** after training completes
4. **Compare reconstruction** to ground truth using your existing `pointcloud_metrics.py`
5. **Batch process** remaining objects once the pipeline is validated

## File Structure

```
GrandChallenge/
├── scripts/
│   ├── check_alignment.py       # Verify scan-GT alignment
│   ├── prepare_sal_data.py      # Preprocess and align data
│   └── train_sal_object.py      # Train SAL on one object
├── third_party/
│   └── SCUTSurface/
│       └── reconstruction/
│           └── SAL/
│               ├── code/           # SAL source code
│               │   ├── confs/      # Configuration files
│               │   ├── training/   # Training scripts
│               │   ├── evaluate/   # Evaluation scripts
│               │   └── exps/       # Output directory (created during training)
│               └── data/
│                   └── real_objects/
│                       ├── points/        # Aligned scans
│                       └── points_iou/    # Ground truth
```

Good luck with your reconstructions! Start with a quick test on one object to validate the setup.
