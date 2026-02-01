import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from approxhull import ApproxConvexHull


def _optional_imports():
    try:
        from sklearn import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        return None, None, exc
    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except Exception as exc:  # pragma: no cover
        return datasets, None, exc
    return datasets, ConvexHull, None


def _precision_recall(approx_vertices: np.ndarray, exact_vertices: np.ndarray) -> Tuple[float, float]:
    approx_set = set(approx_vertices.tolist())
    exact_set = set(exact_vertices.tolist())
    if not approx_set:
        return 0.0, 0.0
    intersection = approx_set.intersection(exact_set)
    precision = len(intersection) / len(approx_set)
    recall = len(intersection) / len(exact_set) if exact_set else 0.0
    return precision, recall


@dataclass
class DatasetSpec:
    name: str
    generator: Callable[[np.random.Generator], np.ndarray]


def _make_dataset_specs(datasets_module) -> List[DatasetSpec]:
    def blobs(rng: np.random.Generator) -> np.ndarray:
        points, _ = datasets_module.make_blobs(n_samples=400, n_features=3, centers=4, random_state=0)
        return points

    def moons(rng: np.random.Generator) -> np.ndarray:
        points, _ = datasets_module.make_moons(n_samples=300, noise=0.08, random_state=0)
        return points

    def swiss_roll(rng: np.random.Generator) -> np.ndarray:
        points, _ = datasets_module.make_swiss_roll(n_samples=400, noise=0.15, random_state=0)
        return points[:, :3]

    def gaussian_2d(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(size=(500, 2))

    def gaussian_5d(rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(size=(600, 5))

    def uniform_6d(rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(700, 6))

    return [
        DatasetSpec("make_blobs_3d", blobs),
        DatasetSpec("make_moons_2d", moons),
        DatasetSpec("make_swiss_roll_3d", swiss_roll),
        DatasetSpec("gaussian_2d", gaussian_2d),
        DatasetSpec("gaussian_5d", gaussian_5d),
        DatasetSpec("uniform_6d", uniform_6d),
    ]


def _try_exact_hull(convex_hull, points: np.ndarray) -> Optional[np.ndarray]:
    try:
        hull = convex_hull(points)
    except Exception:
        return None
    return hull.vertices


def _run_method(points: np.ndarray, method: str, backend: str, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    if method == "uniform":
        start = time.perf_counter()
        hull = ApproxConvexHull(
            points,
            method="uniform",
            epsilon=0.12,
            delta=1e-3,
            backend=backend,
            random_state=rng,
        )
        elapsed = time.perf_counter() - start
        return hull.vertices, elapsed
    start = time.perf_counter()
    hull = ApproxConvexHull(
        points,
        method="mvee_vmf",
        m=12,
        kappa=20.0,
        backend=backend,
        random_state=rng,
    )
    elapsed = time.perf_counter() - start
    return hull.vertices, elapsed


def main() -> None:
    datasets_module, convex_hull, err = _optional_imports()
    if err is not None:
        print("Missing optional dependency:", err)
        print("Install scikit-learn and scipy to run this benchmark.")
        return

    rng = np.random.default_rng(0)
    dataset_specs = _make_dataset_specs(datasets_module)
    backends = ["python", "numba", "cpp"]
    methods = ["uniform", "mvee_vmf"]

    results: List[Dict[str, object]] = []

    for spec in dataset_specs:
        points = np.asarray(spec.generator(rng), dtype=float)
        exact_vertices = _try_exact_hull(convex_hull, points)
        if exact_vertices is None:
            print(f"Skipping exact hull for {spec.name}: failed to compute.")
        for method in methods:
            for backend in backends:
                try:
                    vertices, elapsed = _run_method(points, method, backend, rng)
                except Exception as exc:
                    results.append(
                        {
                            "dataset": spec.name,
                            "method": method,
                            "backend": backend,
                            "precision": np.nan,
                            "recall": np.nan,
                            "time_s": np.nan,
                            "error": str(exc),
                        }
                    )
                    continue
                if exact_vertices is None:
                    precision = recall = np.nan
                else:
                    precision, recall = _precision_recall(vertices, exact_vertices)
                results.append(
                    {
                        "dataset": spec.name,
                        "method": method,
                        "backend": backend,
                        "precision": precision,
                        "recall": recall,
                        "time_s": elapsed,
                        "error": "",
                    }
                )

    header = f"{'dataset':<18} {'method':<9} {'backend':<6} {'precision':>9} {'recall':>7} {'time(s)':>8}"
    print(header)
    print("-" * len(header))
    for row in results:
        precision = row["precision"]
        recall = row["recall"]
        time_s = row["time_s"]
        print(
            f"{row['dataset']:<18} {row['method']:<9} {row['backend']:<6} "
            f"{precision:>9.3f} {recall:>7.3f} {time_s:>8.3f}"
        )
        if row["error"]:
            print(f"  error: {row['error']}")


if __name__ == "__main__":
    main()
