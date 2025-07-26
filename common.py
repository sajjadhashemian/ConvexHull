import numpy as np
import cvxpy as cp
import numpy.linalg as la

np.random.seed(41)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


class vMF:
    """
    Class for sampling from the von Mises-Fisher distribution.
    """

    def __init__(self, d, kappa):
        self.kappa = kappa
        self.d = int(d)

    def _sample_weight(self, kappa, d):
        # Ulrich’s method constants
        b = (-2 * kappa + np.sqrt(4 * kappa**2 + (d - 1) ** 2)) / (d - 1)
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
        while True:
            # Beta-sample step
            Z = np.random.beta((d - 1) / 2, (d - 1) / 2)
            W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
            # Uniform for acceptance
            U = np.random.rand()
            if kappa * W + (d - 1) * np.log(1 - x0 * W) - c >= np.log(U):
                return W

    def sample(self, m):
        # Step 1: sample W = component along e1
        W = np.array([self._sample_weight(self.kappa, self.d) for _ in range(m)])
        # Step 2: sample (d-1)-dim isotropic Gaussian and normalize
        V = np.random.normal(size=(m, self.d - 1))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        # Step 3: build samples
        samples = np.zeros((m, self.d))
        samples[:, 0] = W
        # embed the (d-1)-dim orthonormal part
        samples[:, 1:] = np.sqrt(1 - W**2)[:, None] * V
        return samples


def householder_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the Householder reflection H (d×d) such that H @ a = b,
    where a and b are assumed to be unit vectors in R^d.
    """
    v = a - b
    v = v / np.linalg.norm(v)
    H = np.eye(a.shape[0]) - 2.0 * np.outer(v, v)
    return H


def old_MVEE(points, tol: float = 1e-3):
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
    n, d = points.shape

    # decision variables
    P = cp.Variable((d, d), PSD=True)  # P >> 0
    c = cp.Variable(d)  # center
    Z = cp.Variable((d, d), symmetric=True)  # lower‐triangular part will be enforced
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
        constraints.append(cp.norm(P @ points[i] - c, 2) <= 1)

    # objective: maximize sum(v) == maximize ∑ log(diag(Z))
    obj = cp.Maximize(cp.sum(v))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, eps=tol)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver did not converge: {prob.status}")

    print("MVEE:", c.value, "\n", P.value)
    return c.value, P.value


def MVEE(points: np.ndarray, tol: float = 1e-5, max_iter: int = 1000):
    """
    Computes the minimum‐volume enclosing ellipsoid of a set of points (Khachiyan’s algorithm).
    Returns center c and shape matrix A such that (x - c)^T A (x - c) ≤ 1.
    """
    P = np.asarray(points)
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


def project_onto_ellipsoid_surface(
    c,
    P,
    x,
    tol=1e-4,
    maxiter=100,
):
    """
    Returns the point y on the SURFACE of the ellipsoid { x : ||P x - c|| <= 1 }
    that is closest (in Euclidean norm) to the query point x.

    Args:
    c       : (d,) array, the 'c' returned by your MVEE solver
    P       : (d,d) array, the 'P' returned by your MVEE solver
    x       : (d,) array, the query point
    tol     : tolerance for the root finding
    maxiter : maximum Newton iterations

    Returns:
    y       : (d,) array, the projection of x onto the SURFACE of the ellipsoid
    """
    # build Q = P^T P and center of ellipsoid in original space
    Q = P.T @ P
    x0 = np.linalg.solve(P, c)  # center of ellipsoid
    # translate so center is at origin
    d = x - x0

    # spectral decomposition of Q = V diag(q) V^T
    q, V = np.linalg.eigh(Q)  # q = eigenvalues, V = orthonormal eigenvectors
    z = V.T.dot(d)  # coordinates of (x-x0) in the eigenbasis

    # helper: f(λ) = Σ_i [ q_i * z_i^2 / (1 + λ q_i)^2 ] - 1
    def f(lmbda):
        return np.sum((q * z**2) / (1.0 + lmbda * q) ** 2) - 1.0

    # if x is “outside” or “on” the ellipsoid f(0) ≥ 0, we find λ ≥ 0
    f0 = f(0.0)
    if f0 < 0:
        # x is strictly inside: the closest surface‐point is the radial intersection
        scale = 1.0 / np.sqrt((d @ (Q @ d)))
        return x0 + d * scale

    # otherwise x is outside or on boundary: we need the unique λ > 0 s.t. f(λ)=0
    # bracket [λ_lo=0, λ_hi] with f(λ_lo)≥0, f(λ_hi)<0
    l_lo, l_hi = 0.0, 1.0
    if f(l_hi) > 0:
        # expand until f(l_hi)<0
        while f(l_hi) > 0:
            l_hi *= 2.0

    # initial guess by secant
    lam = l_hi * f0 / (f0 - f(l_hi))

    # Newton–Raphson
    for _ in range(maxiter):
        val = f(lam)
        if abs(val) < tol:
            break
        # derivative: f'(λ) = -2 Σ_i [ q_i^2 * z_i^2 / (1 + λ q_i)^3 ]
        fp = -2.0 * np.sum((q**2 * z**2) / (1.0 + lam * q) ** 3)
        lam -= val / fp
        # keep in [0, l_hi]
        lam = max(0.0, min(lam, l_hi))

    # now form y - x0 = (I + λ Q)^{-1} (x - x0) = V diag(1/(1+λ q_i)) z
    y = x0 + V.dot(z / (1.0 + lam * q))
    return y


def ellipsoid_normal(c: np.ndarray, A: np.ndarray, v: np.ndarray) -> np.ndarray:
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
    w = A @ (v - c)
    # L = np.linalg.cholesky(np.linalg.inv(A))  # A = (L^T L)^(-1)
    # w = np.linalg.inv(L) @ (v - c).T
    # Normalize to unit length
    return normalize(w)


# --- example usage ---
# suppose c,P come from MVEE and y was found by project_onto_ellipsoid_surface
# y = project_onto_ellipsoid_surface(c, P, x_query)
# n_unit = ellipsoid_normal(c, P, y, unit=True)
# print("Unit normal at y:", n_unit)


def sample_input():
    X = [
        [-0.13536, 0.052424],
        [0.12526, -0.4626],
        [0.28357, -0.52009],
        [-0.07684, 0.39493],
        [-0.61311, -0.474],
        [-0.28483, -0.48858],
        [-0.38532, -0.016856],
        [-0.51643, 0.57121],
        [-0.30489, 0.73471],
        [0.74634, 0.35356],
        [-0.92925, -0.68531],
        [-0.16505, -0.75764],
        [0.60003, -0.91131],
        [0.13469, -0.22321],
        [0.55716, -0.6904],
        [0.50771, 0.11204],
        [-0.32228, 0.33077],
        [0.64648, -0.44766],
        [-0.28416, -1.0558],
        [-0.40915, -0.48119],
        [0.06225, 0.054254],
        [-0.21965, -0.35678],
        [0.46709, 0.029328],
        [0.80486, 0.42995],
        [-0.49260, -0.47918],
        [0.22455, -0.47123],
        [0.07945, 0.19404],
        [0.21867, 0.20911],
        [-0.36609, -0.71414],
        [-1.00461, -0.11672],
        [0.90198, -0.97434],
        [0.68393, -0.92937],
        [-0.61698, -0.25378],
        [0.70359, -0.47055],
        [0.43675, 0.56755],
        [0.58299, 0.024607],
        [0.25547, 0.31565],
        [0.44385, 0.028826],
        [-0.16477, -1.416],
        [-0.59128, -0.027423],
    ]
    return np.array(X)
