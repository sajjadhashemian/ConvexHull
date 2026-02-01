import numpy as np

from approxhull import ApproxConvexHull


def test_reproducibility_uniform():
    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(30, 2))
    hull1 = ApproxConvexHull(points, method="uniform", m=10, random_state=123)
    hull2 = ApproxConvexHull(points, method="uniform", m=10, random_state=123)
    assert np.array_equal(hull1.vertices, hull2.vertices)
