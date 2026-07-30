from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import scipy
from scipy.optimize import brentq

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import numpy.typing as npt

    NDArray: TypeAlias = npt.NDArray[Any]

logger = logging.getLogger(__name__)


def find_roots(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    atol: float = 1e-9,
    min_dx: float = 1e-2,
) -> list[float]:
    """Find all roots of func in [x_min, x_max].

    Uses a dense uniform grid to detect sign changes, then refines each bracket
    with Brent's method. Validates roots via |func(root)| < atol, which naturally
    filters out false sign changes caused by poles (discontinuities of tan).
    Finally, we also check if we missed roots that exactly hit zero at a grid point.

    Args:
        func: 1D scalar function to find roots of.
        x_min: Left endpoint of search interval.
        x_max: Right endpoint of search interval.
        atol: Absolute tolerance for root validation.
        min_dx: Minimum spacing between roots.
            We use this to determine a grid spacing that is fine enough to detect all roots.
            If roots are closer than min_dx, we might miss them.

    Returns:
        Sorted list of x values where func(x) ≈ 0.

    """
    if x_min > x_max:
        return []

    if x_min == x_max:
        if abs(func(x_min)) <= atol:
            return [x_min]
        return []

    n_grid = math.ceil((x_max - x_min) / min_dx) + 1

    xs = np.linspace(x_min, x_max, n_grid)
    fs = np.array([func(x) for x in xs])
    # TODO instead of grid search we also could try to implement a "recursive" kind of method,
    # but probably if this is fast enough we dont need it anyway ...

    # Vectorized bracket detection: sign changes between finite adjacent pairs
    finite = np.isfinite(fs)
    both_finite = finite[:-1] & finite[1:]
    sign_change = np.sign(fs[:-1]) * np.sign(fs[1:]) < 0
    bracket_indices = np.where(both_finite & sign_change)[0]

    roots: list[float] = []
    for i in bracket_indices:
        try:
            root = brentq(func, xs[i], xs[i + 1], xtol=1e-13, rtol=1e-13)
        except ValueError:
            logger.warning("Brent's method failed to find root in [%f, %f], skipping.", xs[i], xs[i + 1])
            continue

        val = func(root)
        if abs(val) > atol:
            if abs(val) > 1e5:
                pass
                # logger.debug("Root not close to zero (probably due to a pole): x=%f f(x)=%e. Skipping.", root, val)  # noqa: ERA001, E501
            else:
                logger.warning("Root not close to zero: x=%f f(x)=%e. Skipping.", root, val)
            continue

        roots.append(root)

    # Check if we missed roots, that exactly hit zero at a grid point
    sampled_roots = xs[finite & (np.abs(fs) <= 1e-13)]
    for root in sampled_roots:
        if len(roots) == 0 or np.min(np.abs(np.array(roots) - root)) > min_dx * (1 - 1e-10):
            roots.append(root)
    roots.sort()

    return roots


def calc_nullvector(
    matrix: NDArray,
    *,
    method: Literal["numpy_svd", "scipy_nullspace", "scipy_nullspace_gesvd"] = "scipy_nullspace",
    atol: float = 1e-8,
) -> NDArray | None:
    """Calculate the nullspace vector of a matrix.

    We use scipy.linalg.null_space.
    If the nullspace has more than one vector, we raise an error since this should not happen for the MQDT M-matrix.
    """
    if matrix.shape == (1, 1):
        if abs(matrix[0, 0]) > atol:
            raise RuntimeError(f"Matrix is 1x1 but not close to zero (value={matrix[0, 0]}), this should not happen.")
        return np.array([1.0])

    if method == "numpy_svd":
        _u, s, vt = np.linalg.svd(matrix)
        null_mask = s <= atol
        nullspace = vt.T[:, null_mask]
    elif method == "scipy_nullspace":
        nullspace = scipy.linalg.null_space(matrix, rcond=atol)
    elif method == "scipy_nullspace_gesvd":
        nullspace = scipy.linalg.null_space(matrix, rcond=atol, lapack_driver="gesvd")
    else:
        raise ValueError(f"Invalid method: {method}")

    if nullspace.shape[1] == 0:
        logger.error("Nullspace is empty, no solution found.")
        return None
    if nullspace.shape[1] > 1:
        logger.error("Nullspace has more than one vector (shape=%s), returning first vector.", nullspace.shape)

    return np.array(nullspace[:, 0])
