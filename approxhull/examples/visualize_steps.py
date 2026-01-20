import numpy as np

from approxhull import ApproxConvexHull
from approxhull._sampling import sample_uniform_sphere
from approxhull._support import argmax_dot_blocked


def _optional_imports():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, None, exc
    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except Exception:
        ConvexHull = None
    return plt, ConvexHull, None


def main() -> None:
    plt, convex_hull, err = _optional_imports()
    if err is not None:
        print("Missing optional dependency:", err)
        print("Install matplotlib to run this example.")
        return

    rng = np.random.default_rng(0)
    points = rng.standard_normal(size=(250, 2))

    m = 32
    directions = sample_uniform_sphere(m, 2, rng)
    winners = argmax_dot_blocked(points, directions)
    hull = ApproxConvexHull(points, method="uniform", m=m, random_state=0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].scatter(points[:, 0], points[:, 1], s=12, alpha=0.7, label="points")
    axes[0].set_title("Point cloud")
    axes[0].set_aspect("equal", "box")
    axes[0].legend()

    axes[1].scatter(points[:, 0], points[:, 1], s=10, alpha=0.4)
    axes[1].quiver(
        np.zeros(m),
        np.zeros(m),
        directions[:, 0],
        directions[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        color="tab:blue",
    )
    axes[1].scatter(
        points[winners, 0],
        points[winners, 1],
        s=40,
        color="tab:orange",
        label="support points",
    )
    axes[1].set_title("Uniform directions + support points")
    axes[1].set_aspect("equal", "box")
    axes[1].legend()

    axes[2].scatter(points[:, 0], points[:, 1], s=10, alpha=0.4, label="points")
    approx_pts = points[hull.vertices]
    axes[2].scatter(approx_pts[:, 0], approx_pts[:, 1], s=40, color="tab:green", label="approx vertices")
    if convex_hull is not None:
        exact = convex_hull(points)
        for simplex in exact.simplices:
            seg = points[simplex]
            axes[2].plot(seg[:, 0], seg[:, 1], color="tab:red", lw=1.0)
        axes[2].set_title("Approx vertices + exact hull")
    else:
        axes[2].set_title("Approx vertices")
    axes[2].set_aspect("equal", "box")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
