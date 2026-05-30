#!/usr/bin/env python3
"""onepair_runner.py

Compute metrics for a single pair of PLY files using functions from pointcloud_metrics.py
and write a single-row CSV output.
"""
import sys
import os
import csv
import argparse

import pointcloud_metrics as pcm


def main():
    p = argparse.ArgumentParser(description='Compute metrics for one GT/REC PLY pair')
    p.add_argument('gt_path')
    p.add_argument('rec_path')
    p.add_argument('out_csv')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--color_weight', type=float, default=0.0)
    p.add_argument('--append', action='store_true', help='Append results to out_csv instead of overwriting')
    args = p.parse_args()

    gt_pts, gt_colors = pcm.read_ply_points(args.gt_path)
    rec_pts, rec_colors = pcm.read_ply_points(args.rec_path)

    gt_feat = pcm.build_feature_points(gt_pts, gt_colors, color_weight=args.color_weight)
    rec_feat = pcm.build_feature_points(rec_pts, rec_colors, color_weight=args.color_weight)

    mean_ab, mean_ba, chamfer = pcm.chamfer_distance(gt_feat, rec_feat)
    max_ab, max_ba, hausdorff = pcm.hausdorff_distance(gt_feat, rec_feat)
    precision, recall, fscore, mean_rec_to_gt, mean_gt_to_rec = pcm.precision_recall_fscore(gt_feat, rec_feat, threshold=args.threshold)

    fieldnames = [
        'name', 'frame', 'gt_path', 'rec_path',
        'mean_d_gt_to_rec', 'mean_d_rec_to_gt', 'chamfer',
        'max_d_gt_to_rec', 'max_d_rec_to_gt', 'hausdorff',
        'precision', 'recall', 'fscore',
        'mean_rec_to_gt', 'mean_gt_to_rec'
    ]

    # compute short object name and frame (last 4 digits)
    gt_base = pcm.os.path.splitext(pcm.os.path.basename(args.gt_path))[0]
    obj_name = gt_base.split('_')[0] if '_' in gt_base else gt_base
    m = __import__('re').search(r'(\d{4})$', gt_base)
    frame = m.group(1) if m else ''
    name = obj_name

    row = {
        'name': name,
        'frame': frame,
        'gt_path': args.gt_path,
        'rec_path': args.rec_path,
        'mean_d_gt_to_rec': float(mean_ab),
        'mean_d_rec_to_gt': float(mean_ba),
        'chamfer': float(chamfer),
        'max_d_gt_to_rec': float(max_ab),
        'max_d_rec_to_gt': float(max_ba),
        'hausdorff': float(hausdorff),
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'mean_rec_to_gt': float(mean_rec_to_gt),
        'mean_gt_to_rec': float(mean_gt_to_rec),
    }

    mode = 'a' if args.append else 'w'
    write_header = True
    if args.append and os.path.exists(args.out_csv):
        write_header = False

    with open(args.out_csv, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Wrote results to {args.out_csv}")


if __name__ == '__main__':
    main()
