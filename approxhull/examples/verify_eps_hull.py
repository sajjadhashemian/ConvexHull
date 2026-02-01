import numpy as np

from approxhull import ApproxConvexHull

rng = np.random.default_rng(123)
points = rng.standard_normal(size=(200, 3))

hull = ApproxConvexHull(points, method="uniform", epsilon=0.2, delta=1e-2, backend="python")

stats = hull.support_error(num_probe=256, random_state=rng)
print("Max gap:", stats.max_gap)
print("Mean gap:", stats.mean_gap)
print("Frac > eps:", stats.frac_gap_gt_eps)
