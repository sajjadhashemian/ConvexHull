import numpy as np

from approxhull import ApproxConvexHull

rng = np.random.default_rng(42)
points = rng.standard_normal(size=(300, 4))

configs = [
    ("python", "uniform"),
    ("numba", "uniform"),
    ("cpp", "uniform"),
]

for backend, method in configs:
    try:
        hull = ApproxConvexHull(points, method=method, epsilon=0.15, delta=1e-2, backend=backend)
        print(backend, method, "vertices", len(hull.vertices))
    except Exception as exc:
        print(backend, "failed:", exc)
