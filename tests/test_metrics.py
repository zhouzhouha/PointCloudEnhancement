import os
import tempfile
import numpy as np
from plyfile import PlyData, PlyElement

import pointcloud_metrics as pcm


def write_ply(path, points, colors=None):
    # build vertex dtype
    if colors is None:
        verts = np.array([tuple(p) for p in points], dtype=[('x','f8'),('y','f8'),('z','f8')])
    else:
        verts = np.array([tuple(p)+tuple(c) for p,c in zip(points,colors)], dtype=[('x','f8'),('y','f8'),('z','f8'),('red','u1'),('green','u1'),('blue','u1')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(path)


def test_identical_pointclouds():
    with tempfile.TemporaryDirectory() as d:
        p = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
        p1 = os.path.join(d,'a.ply')
        p2 = os.path.join(d,'b.ply')
        write_ply(p1,p)
        write_ply(p2,p)

        pts1, cols1 = pcm.read_ply_points(p1)
        pts2, cols2 = pcm.read_ply_points(p2)
        mean_ab, mean_ba, chamfer = pcm.chamfer_distance(pts1, pts2)
        max_ab, max_ba, haus = pcm.hausdorff_distance(pts1, pts2)

        assert abs(chamfer) < 1e-9
        assert abs(haus) < 1e-9


def test_translated_pointclouds():
    with tempfile.TemporaryDirectory() as d:
        p = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0]])
        q = p + np.array([0.5,0.0,0.0])
        p1 = os.path.join(d,'a.ply')
        p2 = os.path.join(d,'b.ply')
        write_ply(p1,p)
        write_ply(p2,q)

        pts1, _ = pcm.read_ply_points(p1)
        pts2, _ = pcm.read_ply_points(p2)
        mean_ab, mean_ba, chamfer = pcm.chamfer_distance(pts1, pts2)

        # one-sided means should be ~0.5 and 0.5
        assert abs(mean_ab - 0.5) < 1e-6
        assert abs(mean_ba - 0.5) < 1e-6
