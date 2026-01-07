from __future__ import annotations

import numpy as np

try:
    import approxhull_cpp
except Exception as exc:  # pragma: no cover
    approxhull_cpp = None
    _import_error = exc
else:
    _import_error = None


class CppBackend:
    name = "cpp"

    def argmax_dot(self, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
        if approxhull_cpp is None:
            raise ImportError(f"approxhull_cpp not available: {_import_error}")
        return np.asarray(approxhull_cpp.argmax_dot(points, directions), dtype=int)

    def apply_householder_to_rows(self, base_dirs: np.ndarray, v_house: np.ndarray) -> np.ndarray:
        if approxhull_cpp is None:
            raise ImportError(f"approxhull_cpp not available: {_import_error}")
        return np.asarray(approxhull_cpp.apply_householder_to_rows(base_dirs, v_house), dtype=float)
