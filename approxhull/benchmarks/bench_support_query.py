import time
import numpy as np

from approxhull._support import argmax_dot_blocked

rng = np.random.default_rng(0)
points = rng.standard_normal(size=(2000, 8))
dirs = rng.standard_normal(size=(500, 8))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

start = time.perf_counter()
argmax_dot_blocked(points, dirs)
print("elapsed", time.perf_counter() - start)
