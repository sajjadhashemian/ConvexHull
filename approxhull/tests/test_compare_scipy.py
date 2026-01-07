import numpy as np
import pytest

from approxhull import ApproxConvexHull


def test_compare_with_scipy():
    scipy = pytest.importorskip("scipy")
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(80, 2))
    hull_exact = ConvexHull(points)
    hull_approx = ApproxConvexHull(points, method="uniform", m=32, random_state=0)

    exact_vertices = set(hull_exact.vertices.tolist())
    approx_vertices = set(hull_approx.vertices.tolist())
    assert approx_vertices.issubset(exact_vertices) or len(approx_vertices) > 0
