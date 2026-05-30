#!/usr/bin/env python3
"""
Script to remove Mac system files (starting with ._) from nested directory structure.
Structure: Z:\\3DGS\\unzipped\\{1-32}\\{1-32}\\*._*
"""

import os
from pathlib import Path

def delete_mac_system_files(base_path):
    """
    Delete all files starting with ._ in the nested directory structure.
    
    Args:
        base_path: Base directory path (e.g., Z:\\3DGS\\unzipped)
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist!")
        return
    
    print(f"Scanning base directory: {base_path}")
    print("-" * 60)
    
    # Counter for deleted files
    total_deleted = 0
    total_scanned_dirs = 0
    
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
        
        total_scanned_dirs += 1
        print(f"📁 Processing: {nested_dir}")
        
        # Find all files starting with ._
        deleted_count = 0
        for file_path in nested_dir.rglob("._*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    print(f"  ✓ Deleted: {file_path.name}")
                    deleted_count += 1
                    total_deleted += 1
                except Exception as e:
                    print(f"  ✗ Error deleting {file_path.name}: {e}")
        
        if deleted_count == 0:
            print(f"  No ._ files found")
        else:
            print(f"  Deleted {deleted_count} file(s)")
        print()
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Directories scanned: {total_scanned_dirs}")
    print(f"  Total files deleted: {total_deleted}")
    print("Done!")


if __name__ == "__main__":
    # Define the base path
    BASE_PATH = r"Z:\\3DGS\\unzipped"
    
    # Confirm before deletion
    print("=" * 60)
    print("Mac System Files Cleanup Script")
    print("=" * 60)
    print(f"\nTarget directory: {BASE_PATH}")
    print("\nThis script will delete all files starting with '._'")
    print("from the nested directory structure.\n")
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\nStarting cleanup...\n")
        delete_mac_system_files(BASE_PATH)
    else:
        print("\nOperation cancelled.")
