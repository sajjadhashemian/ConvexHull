from __future__ import annotations

from .python_impl import PythonBackend
from .numba_impl import NumbaBackend
from .cpp_impl import CppBackend


def get_backend(name: str):
    name = name.lower()
    if name == "python":
        return PythonBackend()
    if name == "numba":
        return NumbaBackend()
    if name == "cpp":
        return CppBackend()
    raise ValueError(f"Unknown backend: {name}")
