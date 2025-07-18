import numpy as np
from common import (
    vMF,
    householder_matrix,
    MVEE,
    project_onto_ellipsoid_surface,
    ellipsoid_normal,
)

from common import sample_input

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class ConvexHullviaMVEE:
    """
    Class to compute the convex hull using the Minimum Volume Enclosing Ellipsoid (MVEE).
    """

    def __init__(self, points):
        self.points = np.array(points)

    def extents_estimation(self, U, E, c, return_extents=True):
        """
        Implements the “Extents Estimation” subroutine.
        Inputs:
        P : (n,d) array of input points in R^d
        U : (m,d) array of sampled directions on S^{d-1}
        E : (k,d) array of extremal points from MVEE(P)
        Returns:
        S : an array of the selected subset of P (shape ≤ n×d)
        """
        P = self.points
        n, d = P.shape
        # unit basis e1 = [1,0,...,0]
        e1 = np.zeros(d)
        e1[0] = 1.0

        extents = dict([[tuple(x), []] for x in P])

        S_indices = set()

        # for each original point
        for i in range(n):
            p = P[i]

            # 1) find the closest MVEE point
            closest_ellipsoid_vector = project_onto_ellipsoid_surface(c, E, p)
            p_hat = ellipsoid_normal(c, E, p)

            # 2) build the rotation/reflection sending e1 → p_hat
            R = householder_matrix(e1, p_hat)

            # 3) rotate all directions in U
            U_rot = U @ R.T  # still shape (m,d)

            # 4) for each rotated direction, pick the supporting point in P
            #    (i.e. max dot with that direction)
            for u in U_rot:
                # compute dot products
                dots = P.dot(u)
                s_idx = int(np.argmax(dots))
                extents[tuple(P[s_idx])].append(u)
                S_indices.add(s_idx)

        # assemble S as the unique selected points
        S = P[list(S_indices), :]
        if return_extents == True:
            return S, extents
        return S

    def compute(self, m=20, kappa=10.0, return_extents=True):
        """
        Computes the convex hull using MVEE and returns the hull vertices.
        """
        n, d = self.points.shape
        U = vMF(d, kappa).sample(m)
        c, E = MVEE(self.points)
        S, extents = self.extents_estimation(U, E, c, return_extents)
        if return_extents == True:
            return S, extents, U
        return S


if __name__ == "__main__":
    Z = sample_input()
    conv = ConvexHullviaMVEE(Z)
    hull = conv.compute()
    from plots import plots
    from scipy.spatial import ConvexHull

    plots(Z, ConvexHull(Z)).all(hull)
