import numpy as np

from approxhull._mvee import khachiyan_mvee
from approxhull._sampling import sample_vmf_base


def test_mvee_contains_points():
    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(50, 3))
    ellipsoid = khachiyan_mvee(points, tol=1e-6, max_iter=500)
    for p in points:
        assert ellipsoid.contains(p, tol=1e-6)


def test_project_and_normal():
    rng = np.random.default_rng(1)
    points = rng.standard_normal(size=(30, 3))
    ellipsoid = khachiyan_mvee(points, tol=1e-6, max_iter=500)
    p = points[0]
    v = ellipsoid.project_to_boundary(p)
    val = (v - ellipsoid.center) @ ellipsoid.A @ (v - ellipsoid.center)
    assert np.isclose(val, 1.0, atol=1e-6)
    n = ellipsoid.normal_vector(v)
    assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-6)


def test_vmf_base_sampling():
    rng = np.random.default_rng(0)
    base = sample_vmf_base(64, 4, 5.0, rng)
    norms = np.linalg.norm(base, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
