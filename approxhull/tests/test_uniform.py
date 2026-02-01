import numpy as np

from approxhull._sampling import sample_uniform_sphere


def test_uniform_sphere_norms():
    rng = np.random.default_rng(0)
    samples = sample_uniform_sphere(128, 5, rng)
    norms = np.linalg.norm(samples, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
