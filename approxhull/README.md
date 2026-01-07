# approxhull

`approxhull` computes an **approximate convex hull vertex set** (“ε-hull” proxy) for a point cloud in \(\mathbb{R}^d\). It implements two algorithms from the draft article:

1. **Uniform directional sampling** on \(S^{d-1}\)
2. **MVEE-guided vMF sampling** around ellipsoid normals

The API mirrors the feel of `scipy.spatial.ConvexHull`, while explicitly documenting the approximation semantics.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[numba]
pip install -e .[scipy]
```

## Quickstart

```python
import numpy as np
from approxhull import ApproxConvexHull

points = np.random.default_rng(0).standard_normal(size=(500, 3))

hull = ApproxConvexHull(points, method="uniform", epsilon=0.1, delta=1e-3, backend="python")
print(hull.vertices)

hull_mvee = ApproxConvexHull(points, method="mvee_vmf", m=16, kappa=25.0, backend="python")
print(hull_mvee.get_vertices_points())
```

## API overview

### Class

```python
class ApproxConvexHull(points, *, method="uniform", backend="python", m=None,
                       epsilon=None, delta=None, kappa=None,
                       random_state=None, return_diagnostics=False,
                       compute_facets=False, qhull_options=None,
                       check_input=True, include_ties=False):
    ...
```

**Key attributes** (SciPy-like):

- `points`: original input points
- `vertices`: indices into the original points
- `simplices`, `neighbors`, `equations`, `coplanar`, `good`, `area`, `volume`:
  available only if `compute_facets=True` and SciPy is installed

**Key methods**:

- `get_vertices_points()`
- `support_error(num_probe=..., random_state=...)`
- `to_scipy()`

### Functional API

```python
approx_convex_hull(points, **kwargs)
uniform_sample_hull(points, m=..., epsilon=..., delta=..., backend=...)
mvee_vmf_hull(points, m=..., kappa=..., backend=...)
```

## Parameter selection

### Uniform sampling

If you pass `epsilon` and `delta`, we compute the number of directions as:

\[
 m = C \cdot \frac{d}{\varepsilon^2} \log\left(\frac{1}{\delta}\right)
\]

with a conservative constant `C=4.0` (override via `uniform_constant`). Smaller `epsilon` or `delta` means more samples.

### MVEE + vMF sampling

`kappa` controls how concentrated the vMF samples are around each ellipsoid normal. Typical values grow with dimension (e.g., 10–100 in 3D). Larger `kappa` concentrates samples more tightly around the local normals.

## Reproducibility

Pass `random_state` as an `int` or `np.random.Generator` to guarantee reproducibility across backends. All random directions are generated in Python and passed to the backend kernels.

## Facets and SciPy interop

Set `compute_facets=True` and install SciPy to compute a *true* convex hull of the approximate vertices. You can also call `to_scipy()`.

## Performance tips

- Use `backend="numba"` or `backend="cpp"` for large inputs.
- The dominant cost is the support-function argmax; use batching or reduce sampling complexity when possible.
- For MVEE-guided sampling, large `m` and large `n` can be expensive.

## Troubleshooting

- **Degenerate inputs**: MVEE may be unstable for nearly singular data. Lower `mvee_tol`, or run uniform sampling instead.
- **High dimension**: use larger `m` or smaller `epsilon` to maintain quality.

## Technical notes

### Definitions

For a finite set \(P \subset \mathbb{R}^d\), the support function is

\[
\omega_u(P) = \max_{p \in P} \langle u, p \rangle.
\]

A subset \(S \subset P\) is an \((\varepsilon, \delta)\)-hull if

\[
\Pr_{u \sim \mathrm{Unif}(S^{d-1})}[\omega_u(P) - \omega_u(S) > \varepsilon] \le \delta.
\]

### Algorithms (summary)

**Algorithm 1** (uniform): sample directions \(u_i\) uniformly on \(S^{d-1}\); keep argmax points.

**Algorithm 2** (MVEE+vMF): compute the MVEE, project each point onto the ellipsoid boundary, compute outward normals, then sample directions from a vMF distribution around each normal.

### vMF sampling

We use a Wood (1994)-style sampler for the vMF distribution. Base samples around \(e_1\) are rotated to the target mean via a Householder reflection.

### Backend architecture

- **Python**: vectorized numpy kernels.
- **Numba**: `prange` kernels for argmax.
- **C++**: OpenMP-enabled kernels for argmax and Householder rotation, bound with pybind11.

### Complexity

The dominant cost is \(\mathcal{O}(n m d)\) for support queries, with additional MVEE overhead for Algorithm 2.

## Citation

> Draft manuscript reference placeholder.

## Contributing

Issues and PRs are welcome. Please include tests for new features and keep APIs consistent with SciPy conventions.

## License

MIT License. See [LICENSE](LICENSE).
