#!/usr/bin/env python3
"""
Reorganize dataset from camera-based folders to frame-based folders.

Source structure:
  Z:\\3DGS\\unzipped\\{camera}\\{index}_{suffix}.png

Target structure:
  Z:\\3DGS\\unzipped\\{index}\\{camera}_{index}_{suffix}.png
"""

import os
import shutil
from pathlib import Path
from typing import Set, List

# Configuration
BASE_PATH = r"Z:\3DGS\unzipped"
CAMERA_FOLDERS = list(range(1, 33))  # Camera folders 1-32
FILE_SUFFIXES = ["RGB_01.png", "RGB_02.png", "DepthMap16bit.png"]


def get_all_frame_indices(base_path: Path, camera_folders: List[int]) -> Set[str]:
    """
    Scan all camera folders to find all unique frame indices.
    
    Args:
        base_path: Root directory
        camera_folders: List of camera folder numbers
    
    Returns:
        Set of frame indices (e.g., {"00000", "00001", ...})
    """
    indices = set()
    
    for camera in camera_folders:
        camera_dir = base_path / str(camera)
        if not camera_dir.exists():
            continue
        
        # Find all PNG files
        for file in camera_dir.glob("*_RGB_01.png"):
            # Extract the 5-digit index from filename
            # e.g., "00000_RGB_01.png" -> "00000"
            index = file.stem.split("_")[0]
            if len(index) == 5 and index.isdigit():
                indices.add(index)
    
    return indices


def reorganize_single_frame(base_path: Path, 
                            frame_index: str, 
                            camera_folders: List[int],
                            suffixes: List[str],
                            mode: str = "copy") -> dict:
    """
    Reorganize files for a single frame index.
    
    Args:
        base_path: Root directory
        frame_index: 5-digit frame index (e.g., "00000")
        camera_folders: List of camera folder numbers
        suffixes: List of file suffixes to process
        mode: "copy" or "move"
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "processed": 0,
        "missing": 0,
        "errors": 0,
        "frame_index": frame_index
    }
    
    # Create target frame folder
    frame_folder = base_path / frame_index
    frame_folder.mkdir(exist_ok=True)
    print(f"📁 Created/verified frame folder: {frame_folder}")
    
    # Process each camera
    for camera in camera_folders:
        camera_dir = base_path / str(camera)
        
        if not camera_dir.exists():
            continue
        
        # Process each file suffix for this camera and frame
        for suffix in suffixes:
            source_filename = f"{frame_index}_{suffix}"
            source_path = camera_dir / source_filename
            
            # Target filename includes camera prefix
            target_filename = f"{camera}_{source_filename}"
            target_path = frame_folder / target_filename
            
            # Check if source file exists
            if not source_path.exists():
                stats["missing"] += 1
                print(f"  ⚠ Missing: {source_path}")
                continue
            
            # Check if target already exists
            if target_path.exists():
                print(f"  ⚠ Target already exists: {target_filename}, skipping")
                continue
            
            # Copy or move the file
            try:
                if mode == "copy":
                    shutil.copy2(source_path, target_path)
                    action = "Copied"
                elif mode == "move":
                    shutil.move(str(source_path), str(target_path))
                    action = "Moved"
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                
                stats["processed"] += 1
                print(f"  ✓ {action}: {source_filename} -> {target_filename}")
                
            except Exception as e:
                stats["errors"] += 1
                print(f"  ✗ Error processing {source_filename}: {e}")
    
    return stats


def reorganize_all_frames(base_path: Path,
                          camera_folders: List[int],
                          suffixes: List[str],
                          mode: str = "copy",
                          test_index: str = None):
    """
    Reorganize dataset from camera-based to frame-based folders.
    
    Args:
        base_path: Root directory
        camera_folders: List of camera folder numbers
        suffixes: List of file suffixes to process
        mode: "copy" or "move"
        test_index: If specified, only process this single index
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist!")
        return
    
    print("=" * 70)
    print("Dataset Reorganization Script")
    print("=" * 70)
    print(f"Base path: {base_path}")
    print(f"Mode: {mode.upper()}")
    print(f"Camera folders: {len(camera_folders)} cameras")
    print(f"File suffixes: {suffixes}")
    
    # Determine which indices to process
    if test_index:
        print(f"TEST MODE: Processing only index '{test_index}'")
        indices_to_process = [test_index]
    else:
        print("\nScanning for all frame indices...")
        indices_to_process = sorted(get_all_frame_indices(base_path, camera_folders))
        print(f"Found {len(indices_to_process)} unique frame indices")
    
    print("-" * 70)
    
    # Process each frame
    total_stats = {
        "frames": 0,
        "processed": 0,
        "missing": 0,
        "errors": 0
    }
    
    for i, frame_index in enumerate(indices_to_process, 1):
        print(f"\n[{i}/{len(indices_to_process)}] Processing frame: {frame_index}")
        
        frame_stats = reorganize_single_frame(
            base_path, 
            frame_index, 
            camera_folders, 
            suffixes, 
            mode
        )
        
        total_stats["frames"] += 1
        total_stats["processed"] += frame_stats["processed"]
        total_stats["missing"] += frame_stats["missing"]
        total_stats["errors"] += frame_stats["errors"]
        
        print(f"  Frame summary: {frame_stats['processed']} files processed, "
              f"{frame_stats['missing']} missing, {frame_stats['errors']} errors")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Frames processed:    {total_stats['frames']}")
    print(f"Files processed:     {total_stats['processed']}")
    print(f"Files missing:       {total_stats['missing']}")
    print(f"Errors:              {total_stats['errors']}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reorganize dataset from camera-based to frame-based folders"
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="Operation mode: 'copy' (default) or 'move'"
    )
    parser.add_argument(
        "--test-index",
        type=str,
        default=None,
        help="Test mode: process only this frame index (e.g., '00000')"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=BASE_PATH,
        help=f"Base directory path (default: {BASE_PATH})"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    # Show configuration and get confirmation
    print("\nConfiguration:")
    print(f"  Base path: {args.base_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Test index: {args.test_index if args.test_index else 'None (process all)'}")
    print()
    
    if not args.yes:
        if args.mode == "move":
            print("⚠ WARNING: MOVE mode will remove files from camera folders!")
        
        response = input("Do you want to proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nOperation cancelled.")
            exit(0)
    
    print("\nStarting reorganization...\n")
    
    reorganize_all_frames(
        base_path=args.base_path,
        camera_folders=CAMERA_FOLDERS,
        suffixes=FILE_SUFFIXES,
        mode=args.mode,
        test_index=args.test_index
    )
