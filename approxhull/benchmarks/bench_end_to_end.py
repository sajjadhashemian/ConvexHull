import time
import numpy as np

from approxhull import ApproxConvexHull

rng = np.random.default_rng(0)
points = rng.standard_normal(size=(2000, 6))

start = time.perf_counter()
ApproxConvexHull(points, method="uniform", epsilon=0.2, delta=1e-2, backend="python")
print("elapsed", time.perf_counter() - start)
