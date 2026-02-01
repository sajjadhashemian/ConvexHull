import numpy as np

from approxhull import ApproxConvexHull

rng = np.random.default_rng(0)
points = rng.standard_normal(size=(200, 3))

hull = ApproxConvexHull(points, method="uniform", epsilon=0.2, delta=1e-2, backend="python")
print("Uniform vertices:", hull.vertices)

hull_mvee = ApproxConvexHull(points, method="mvee_vmf", m=8, kappa=15.0, backend="python")
print("MVEE-vMF vertices:", hull_mvee.vertices)
