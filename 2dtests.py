import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from common import sample_input
from plots import plots

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

import numpy as np


def sample_sphere_gaussian_approx(m, d, sigma):
    # mean vector e1
    mu = np.zeros(d)
    mu[0] = 1.0

    # sample m x d Gaussians
    X = np.random.normal(loc=mu, scale=sigma, size=(10, 2))

    # normalize each row to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    V = X / norms
    return V


if __name__ == "__main__":
    # m = 20
    # kappa = 10.0  # higher → more tightly around e1
    # S = convex_hull_via_mvee(_Z, m, kappa)

    """
    Create a set of random points in 2D.
    """
    # _Z = np.random.normal(loc=(0, 0), scale=0.5, size=(40, 2))

    n = 40
    A, B = np.random.normal(size=(n, 1)), np.random.normal(size=(n, 1))
    _Z = np.hstack((A, B))
    norms = np.linalg.norm(_Z, axis=1, keepdims=True)
    # _Z = _Z / norms

    """
    Plot the convex hull of the points.
    """
    hull = ConvexHull(_Z)
    plt = plots(_Z, hull)
    # x = plt.Plain()
    # x = plt.AllviaExtents()
    x = plt.AllviaMVEE()
    x.show()
