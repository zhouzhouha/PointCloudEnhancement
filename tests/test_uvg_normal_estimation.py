import sys
from pathlib import Path

import numpy as np


METRIC_DIR = Path(__file__).resolve().parents[1] / "third_party" / "UVG-CWI-Metric"
sys.path.insert(0, str(METRIC_DIR))
from metrics import estimate_normals_knn, valid_normals  # noqa: E402


def test_knn_normals_on_plane():
    grid = np.linspace(-1.0, 1.0, 15)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    normals = estimate_normals_knn(points, k=16, batch_size=40)

    assert valid_normals(normals, len(points))
    alignment = np.abs(normals @ np.array([0.0, 0.0, 1.0]))
    assert np.min(alignment) > 0.999


def test_knn_normals_on_sphere():
    phi = np.linspace(0.15, np.pi - 0.15, 20)
    theta = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    pp, tt = np.meshgrid(phi, theta, indexing="ij")
    points = np.column_stack([
        (np.sin(pp) * np.cos(tt)).ravel(),
        (np.sin(pp) * np.sin(tt)).ravel(),
        np.cos(pp).ravel(),
    ])

    normals = estimate_normals_knn(points, k=20, batch_size=100)
    alignment = np.abs(np.sum(normals * points, axis=1))

    assert np.mean(alignment) > 0.99


def test_missing_normals_are_invalid():
    assert not valid_normals(None, 10)
    assert not valid_normals(np.zeros((10, 3)), 10)
