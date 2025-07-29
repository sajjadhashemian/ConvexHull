import numpy as np
import cvxpy as cp
import random
from typing import List, Tuple

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
random.seed(41)


class Ellipsoid:
    def __init__(self, points, center=None, shape_matrix=None, method="Khachiyan"):
        """
        Initialize the Ellipsoid with a set of points, center, and shape matrix.
        If center and shape_matrix are not provided, they will be computed.
        """
        self.points = np.asarray(points)
        self.n, self.d = self.points.shape

        if center is None or shape_matrix is None:
            self.center, self.shape_matrix = self.compute(method=method)
        else:
            self.center = np.asarray(center)
            self.shape_matrix = np.asarray(shape_matrix)

    def SDP_minimum_volume_enclosing_ellipsoid(self, tol: float = 1e-3):
        """
        Finds the minimum-volume enclosing ellipsoid of a set of points in R^d
        by solving the log‐det‐type SDP in CVXPY.

        Args:
        points : (n, d) array of input points.
        tol    : solver tolerance (passed to the SDP solver).

        Returns:
        c.value : center of the ellipsoid (d,)
        P.value : shape matrix P of the ellipsoid (d, d) PSD
        """
        n, d = self.points.shape

        # decision variables
        P = cp.Variable((d, d), PSD=True)  # P >> 0
        c = cp.Variable(d)  # center
        Z = cp.Variable(
            (d, d), symmetric=True
        )  # lower‐triangular part will be enforced
        v = cp.Variable(d)  # auxiliary log‐vars

        constraints = []

        # 1) force Z to be lower‐triangular
        for i in range(d):
            for j in range(i + 1, d):
                constraints.append(Z[i, j] == 0)

        # 2) block‐PSD constraint [ P  Z; Zᵀ  diag(Z) ] >> 0
        zdiag = cp.diag(Z)  # vector of diagonal entries of Z
        # build the 2d × 2d block
        upper = cp.hstack([P, Z])
        lower = cp.hstack([Z.T, cp.diag(zdiag)])
        M = cp.vstack([upper, lower])
        constraints.append(M >> 0)

        # 3) v <= log(diag(Z))
        #    ensures we are maximizing the sum of logs of diag(Z)
        constraints.append(v <= cp.log(zdiag))

        # 4) containment constraints ∥P x_i - c∥₂ ≤ 1 for all points
        for i in range(n):
            constraints.append(cp.norm(P @ self.points[i] - c, 2) <= 1)

        # objective: maximize sum(v) == maximize ∑ log(diag(Z))
        obj = cp.Maximize(cp.sum(v))

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, eps=tol)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver did not converge: {prob.status}")

        return c.value, P.value

    def khachiyan_minimum_volume_enclosing_ellipsoid(
        self, tol: float = 1e-5, max_iter: int = 1000
    ):
        """
        Computes the minimum‐volume enclosing ellipsoid of a set of points (Khachiyan’s algorithm).
        Returns center c and shape matrix A such that (x - c)^T A (x - c) ≤ 1.
        """
        P = np.asarray(self.points)
        N, d = P.shape

        # Build Q = [Pᵀ; 1ᵀ]
        Q = np.vstack([P.T, np.ones(N)])
        u = np.full(N, 1.0 / N)

        for _ in range(max_iter):
            V = Q @ np.diag(u) @ Q.T
            invV = np.linalg.inv(V)
            QT = Q.T  # shape (N, d+1)
            # M_j = q_j^T V^{-1} q_j for each column q_j of Q
            M = np.sum((QT @ invV) * QT, axis=1)
            j = np.argmax(M)
            max_M = M[j]

            step = (max_M - d - 1) / ((d + 1) * (max_M - 1))
            new_u = (1 - step) * u
            new_u[j] += step

            if np.linalg.norm(new_u - u) < tol:
                u = new_u
                break
            u = new_u

        # Center of ellipsoid
        c = P.T @ u

        # Covariance‐like matrix
        cov = (P.T * u) @ P - np.outer(c, c)
        # Shape matrix A
        A = np.linalg.inv(cov) / d

        return c, A

    def minimum_enclosing_sphere(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the minimum‐enclosing sphere of self.points via Welzl’s algorithm.
        Returns (c, A) where c is the center (shape (d,))
        and A = I / r^2 so that (x - c)^T A (x - c) ≤ 1 defines the sphere.
        """

        def _ball_from(R: List[np.ndarray]) -> Tuple[np.ndarray, float]:
            # Trivial sphere from up to d+1 points
            if not R:
                return np.zeros(self.d), 0.0
            if len(R) == 1:
                return R[0].copy(), 0.0
            if len(R) == 2:
                c = (R[0] + R[1]) / 2
                r = np.linalg.norm(R[0] - c)
                return c, r
            # Solve linear system for ≥3 points
            mat = []
            rhs = []
            for p in R[1:]:
                mat.append(2 * (p - R[0]))
                rhs.append(np.dot(p, p) - np.dot(R[0], R[0]))
            mat = np.vstack(mat)
            rhs = np.array(rhs)
            c = np.linalg.solve(mat, rhs)
            r = np.linalg.norm(R[0] - c)
            return c, r

        def _welzl(
            P: List[np.ndarray], R: List[np.ndarray]
        ) -> Tuple[np.ndarray, float]:
            if not P or len(R) == self.d + 1:
                return _ball_from(R)
            p = P.pop(random.randrange(len(P)))
            c, r = _welzl(P, R)
            if np.linalg.norm(p - c) <= r + 1e-12:
                P.append(p)
                return c, r
            R.append(p)
            c, r = _welzl(P, R)
            R.pop()
            P.append(p)
            return c, r

        # Prepare and shuffle
        P_list = [p.copy() for p in self.points]
        random.shuffle(P_list)
        center, radius = _welzl(P_list, [])

        if radius == 0:
            # Degenerate: infinite A
            A = np.eye(self.d) * np.inf
        else:
            A = np.eye(self.d) / (radius * radius)

        return center, A

    def compute(self, method="Khachiyan"):
        """
        Computes the minimum-volume enclosing ellipsoid (MVEE) of the points.
        Uses either Khachiyan's algorithm or a CVXPY-based approach.
        """
        if method == "Khachiyan":
            return self.khachiyan_minimum_volume_enclosing_ellipsoid()
        elif method == "SDP":
            return self.SDP_minimum_volume_enclosing_ellipsoid()
        elif method == "Sphere":
            return self.minimum_enclosing_sphere()
        else:
            raise ValueError(
                "Unknown method for computing MVEE. Use 'Khachiyan' or 'CVXPY'."
            )

    def project(self, x, tol=1e-6, max_iter=100):
        """
        Project point x onto the surface of the ellipsoid defined by
        (y - center)^T shape_matrix^-1 (y - center) = 1.
        """
        x = np.asarray(x)
        z = x - self.center

        # Eigen-decompose shape_matrix: M = U @ diag(e) @ U.T
        e, U = np.linalg.eigh(self.shape_matrix)
        y = U.T @ z

        # φ(λ) = Σ_i [e_i * y_i^2 / (1 + λ e_i)^2] − 1
        def phi(lam):
            return np.sum(e * y**2 / (1.0 + lam * e) ** 2) - 1.0

        # φ'(λ) = −2 Σ_i [e_i^2 * y_i^2 / (1 + λ e_i)^3]
        def phi_prime(lam):
            return -2.0 * np.sum(e**2 * y**2 / (1.0 + lam * e) ** 3)

        # Bracket a root for φ(λ)=0
        lam0 = 0.0
        f0 = phi(lam0)
        # Domain of λ is (-1/max(e), ∞)
        lam_min = -1.0 / np.max(e) + 1e-12

        if f0 > 0:
            lam_lo, lam_hi = lam0, 1.0
            while phi(lam_hi) > 0:
                lam_hi *= 2.0
        else:
            lam_lo, lam_hi = lam_min, lam0

        lam = 0.5 * (lam_lo + lam_hi)
        for _ in range(max_iter):
            f = phi(lam)
            if abs(f) < tol:
                break
            df = phi_prime(lam)
            lam_new = lam - f / df
            # Safeguard: keep within [lam_lo, lam_hi]
            if not (lam_lo < lam_new < lam_hi):
                lam_new = 0.5 * (lam_lo + lam_hi)
            # Update bracket
            if phi(lam_new) > 0:
                lam_lo = lam_new
            else:
                lam_hi = lam_new
            lam = lam_new

        # Compute projected point on surface
        w = y / (1.0 + lam * e)
        return U @ w + self.center

    def normal_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Given an ellipsoid E = { x : (x - c)^T A (x - c) = 1 } (with A ≻ 0),
        and a point v on its surface, returns the unit normal (perpendicular)
        vector at v on E.

        This u maximizes u ⋅ (v - c) over all unit u, and is proportional to the gradient:
                ∇[(x - c)^T A (x - c)]|_{x=v} = 2 A (v - c).
        The factor 2 cancels in normalization.

        Args:
                c: center of the ellipsoid, shape (d,)
                A: shape matrix of the ellipsoid, shape (d, d)
                v: point on the ellipsoid surface, shape (d,)

        Returns:
                u: unit normal vector at v, shape (d,)
        """

        def normalize(v):
            return v / np.linalg.norm(v)

        # Compute gradient direction
        w = self.shape_matrix @ (v - self.center)
        # L = np.linalg.cholesky(np.linalg.inv(A))  # A = (L^T L)^(-1)
        # w = np.linalg.inv(L) @ (v - c).T
        # Normalize to unit length
        return normalize(w)


# --- example usage ---
# suppose c,P come from MVEE and y was found by project_onto_ellipsoid_surface
# y = project_onto_ellipsoid_surface(c, P, x_query)
# n_unit = ellipsoid_normal(c, P, y, unit=True)
# print("Unit normal at y:", n_unit)
