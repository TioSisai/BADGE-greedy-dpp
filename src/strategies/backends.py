"""Array backend for sample selection, preferring cupy/cuml on CUDA and falling back to numpy/sklearn."""

from __future__ import annotations

import numpy as np

# The GPU backend is installed as a pair (requirements-gpu.txt), so one flag covers cupy and cuml.
try:
    import cupy as cp
    from cuml.metrics import pairwise_distances as _cuml_pairwise_distances
    from cupyx.scipy.linalg import solve_triangular as _cupy_solve_triangular

    # The packages can be imported without a GPU, so visible devices must be checked.
    HAS_GPU_BACKEND = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    _cuml_pairwise_distances = None
    _cupy_solve_triangular = None
    HAS_GPU_BACKEND = False

if HAS_GPU_BACKEND:
    # Importing cuml swaps cupy's pooled allocator for RMM's synchronizing cudaMalloc, so the memory pool is put back.
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)


def backend_signature(device) -> str:
    """Return the selection backend serving a torch device, recorded in config.json to distinguish rounding paths."""
    return "cuml+cupy" if is_cupy(array_module(device)) else "sklearn+numpy"


def require_backend(signature: str, device) -> None:
    """Fail when this process cannot provide the selection backend the configuration was written with.

    Args:
        signature: Backend signature recorded in config.json.
        device: torch device of the experiment.

    Raises:
        RuntimeError: The current process resolves a different backend, which would change the selections.
    """
    current = backend_signature(device)
    if current != signature:
        raise RuntimeError(
            f"this process provides the {current} selection backend but the configuration requires "
            f"{signature}; results from the two backends are not comparable"
        )


def array_module(device) -> object:
    """Return cupy for a CUDA device when the GPU backend is usable, otherwise numpy."""
    if HAS_GPU_BACKEND and str(device).startswith("cuda"):
        return cp
    return np


def is_cupy(xp) -> bool:
    """Report whether ``xp`` is the cupy module."""
    return cp is not None and xp is cp


def _is_torch_tensor(array) -> bool:
    """Report whether an object is a torch tensor without importing torch."""
    return hasattr(array, "detach") and hasattr(array, "__dlpack__")


def to_backend(array, xp):
    """Move an array to ``xp``, avoiding host round trips for CUDA tensors."""
    if _is_torch_tensor(array):
        array = array.detach()
        if is_cupy(xp) and array.device.type == "cuda":
            array = cp.from_dlpack(array)
        else:
            array = array.cpu().numpy()
    if is_cupy(xp):
        return array if isinstance(array, cp.ndarray) else cp.asarray(np.ascontiguousarray(array))
    return np.asarray(array)


def to_numpy(array) -> np.ndarray:
    """Convert a cupy or torch array to numpy, returning numpy inputs unchanged."""
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    if _is_torch_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def pairwise_distances(X, Y):
    """Compute the Euclidean distance matrix with cuml on cupy inputs and sklearn otherwise.

    Args:
        X: Point set, ``[n_a, D]``.
        Y: Second point set, ``[n_b, D]``.

    Returns:
        Distance matrix in the backend of X, ``[n_a, n_b]``.
    """
    if cp is not None and isinstance(X, cp.ndarray):
        return _cuml_pairwise_distances(X, Y, metric="euclidean")
    from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances

    return sklearn_pairwise_distances(to_numpy(X), to_numpy(Y), metric="euclidean")


def solve_lower_triangular(lower, rhs, xp):
    """Solve ``lower @ out = rhs`` for a lower-triangular factor on the matching backend."""
    if is_cupy(xp):
        return _cupy_solve_triangular(lower, rhs, lower=True)
    from scipy.linalg import solve_triangular

    return solve_triangular(lower, rhs, lower=True, check_finite=False)


def seed_backend(seed: int) -> None:
    """Seed the global numpy and cupy random streams consumed by the selection kernels."""
    np.random.seed(seed)
    if HAS_GPU_BACKEND:
        cp.random.seed(seed)
