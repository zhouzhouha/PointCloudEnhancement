#!/usr/bin/env python3
"""
Script to flatten nested directory structure.
Moves files from Z:\\3DGS\\unzipped\\X\\X to Z:\\3DGS\\unzipped\\X
"""

import os
import shutil
from pathlib import Path

def flatten_directories(base_path):
    """
    Flatten nested directory structure by moving contents up one level.
    
    Args:
        base_path: Base directory path (e.g., Z:\\3DGS\\unzipped)
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist!")
        return
    
    print(f"Scanning base directory: {base_path}")
    print("-" * 60)
    
    # Counter for processed directories
    total_processed = 0
    total_moved_files = 0
    
    # List all subdirectories in the base path
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    subdirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(subdirs)} top-level directories\n")
    
    for top_dir in subdirs:
        top_dir_name = top_dir.name
        
        # Check if nested directory exists (e.g., Z:\3DGS\unzipped\2\2)
        nested_dir = top_dir / top_dir_name
        
        if not nested_dir.exists():
            print(f"⚠ Skipping '{top_dir_name}' - nested directory '{nested_dir}' not found")
            continue
        
        if not nested_dir.is_dir():
            print(f"⚠ Skipping '{top_dir_name}' - '{nested_dir}' is not a directory")
            continue
        
        print(f"📁 Processing: {top_dir_name}")
        print(f"   Moving from: {nested_dir}")
        print(f"   Moving to:   {top_dir}")
        
        moved_count = 0
        error_count = 0
        
        # Move all items from nested directory to parent
        try:
            items = list(nested_dir.iterdir())
            for item in items:
                destination = top_dir / item.name
                
                # Check if destination already exists
                if destination.exists():
                    print(f"  ⚠ Warning: '{item.name}' already exists in destination, skipping")
                    continue
                
                try:
                    shutil.move(str(item), str(destination))
                    moved_count += 1
                    total_moved_files += 1
                except Exception as e:
                    print(f"  ✗ Error moving '{item.name}': {e}")
                    error_count += 1
            
            # Remove the now-empty nested directory
            if moved_count > 0:
                try:
                    nested_dir.rmdir()
                    print(f"  ✓ Moved {moved_count} item(s), removed empty nested directory")
                    total_processed += 1
                except Exception as e:
                    print(f"  ⚠ Could not remove nested directory: {e}")
            else:
                print(f"  No items moved")
                
            if error_count > 0:
                print(f"  ✗ {error_count} error(s) occurred")
                
        except Exception as e:
            print(f"  ✗ Error accessing nested directory: {e}")
        
        print()
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Directories processed: {total_processed}")
    print(f"  Total items moved: {total_moved_files}")
    print("Done!")


if __name__ == "__main__":
    # Define the base path
    BASE_PATH = r"Z:\3DGS\unzipped"
    
    # Confirm before moving
    print("=" * 60)
    print("Directory Flattening Script")
    print("=" * 60)
    print(f"\nTarget directory: {BASE_PATH}")
    print("\nThis script will move all contents from nested directories")
    print("up one level (e.g., Z:\\3DGS\\unzipped\\2\\2 -> Z:\\3DGS\\unzipped\\2)")
    print("and remove the empty nested directories.\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\nStarting flattening process...\n")
        flatten_directories(BASE_PATH)
    else:
        print("\nOperation cancelled.")
