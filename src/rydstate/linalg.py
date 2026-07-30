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

    Uses a dense uniform grid to detect sign changes, then refines each bracket with Brent's method.
    A pair of roots lying between two adjacent grid points has no sign change on
    the grid, but shows up as a local minimum of |func|;
    We refine such dips by recursively calling find_approximate_roots on a smaller interval around the dip.

    Args:
        func: 1D scalar function to find roots of.
        x_min: Left endpoint of search interval.
        x_max: Right endpoint of search interval.
        atol: Absolute tolerance for root validation.
        min_dx: Grid spacing used to detect sign changes and dips.
            Isolated roots and pairs of roots are found even if they are closer than
            min_dx, as long as the grid resolves the dip of |func| around them.
            Three or more roots within one grid cell can still be missed.

    Returns:
        Sorted list of x values where func(x) ≈ 0.

    """
    if x_min > x_max:
        return []

    if x_min == x_max:
        if abs(func(x_min)) <= atol:
            return [x_min]
        return []

    approximate_roots = find_approximate_roots(func, x_min, x_max, min_dx=min_dx)

    roots: list[float] = []
    for x_left, x_right in approximate_roots.values():
        if x_left == x_right:
            root = x_left
        else:
            try:
                root = brentq(func, x_left, x_right, xtol=1e-13, rtol=1e-13)
            except ValueError:
                logger.warning("Brent's method failed to find root in [%f, %f], skipping.", x_left, x_right)
                continue

        val = func(root)
        if abs(val) > atol:
            logger.warning("Root not close to zero: x=%f f(x)=%e. Skipping.", root, val)
            continue

        roots.append(float(root))

    roots.sort()
    if np.any(np.diff(roots) < 1e-8):
        raise ValueError(f"Found roots that are very close together: {roots}, this should not happen.")

    return roots


def find_approximate_roots(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    min_dx: float = 1e-2,
) -> dict[float, tuple[float, float]]:
    assert x_min < x_max, "x_min must be less than x_max"
    approximate_roots: dict[float, tuple[float, float]] = {}

    n_grid = math.ceil((x_max - x_min) / min_dx) + 1

    xs = np.linspace(x_min, x_max, n_grid)
    dx = xs[1] - xs[0]

    fs = np.array([func(x) for x in xs])
    abs_fs = np.abs(fs)
    sign_fs = np.sign(fs)

    # find roots that are exactly zero at the grid point
    zeros = fs == 0
    approximate_roots.update({x: (x, x) for x in xs[zeros]})

    # find approximate roots that have a sign change between two adjacent (finite and non-zero) grid points
    finite = abs_fs < 1e10
    non_zeros = np.bitwise_not(zeros)
    sign_change = sign_fs[:-1] * sign_fs[1:] < 0
    conditions = sign_change & (finite[:-1] & finite[1:]) & (non_zeros[:-1] & non_zeros[1:])

    approximate_roots.update(
        {
            (x_left + x_right) / 2: (x_left, x_right)
            for x_left, x_right in zip(xs[:-1][conditions], xs[1:][conditions], strict=True)
        }
    )

    # check the endpoints for almost zero values (and no sign changes)
    if abs_fs[0] < 1e-13 and fs[0] != 0 and not sign_change[0]:
        approximate_roots[xs[0]] = (xs[0], xs[0])
    if abs_fs[-1] < 1e-13 and fs[-1] != 0 and not sign_change[-1]:
        approximate_roots[xs[-1]] = (xs[-1], xs[-1])

    # find dips in abs(func(x)) that are not detected by the sign change
    is_dip = (
        (abs_fs[1:-1] <= abs_fs[:-2])
        & (abs_fs[1:-1] <= abs_fs[2:])
        & (sign_fs[:-2] == sign_fs[1:-1])
        & (sign_fs[1:-1] == sign_fs[2:])
    )
    for i, x in zip(np.where(is_dip)[0] + 1, xs[1:-1][is_dip], strict=True):
        if abs_fs[i] / dx > 1e2:
            # if it is a local minimum of |func| but it is far away from zero (compared to the grid spacing),
            # we assume it is not a root and skip it
            continue
        if dx < 1e-8:
            logger.warning(
                "Found a dip which up to dx=%e does not cross zero, but is very close to zero: x=%f f(x)=%e. Skipping.",
                *(dx, x, fs[i]),
            )
            continue
        new_roots = find_approximate_roots(func, x - dx, x + dx, min_dx=dx * 1e-2)
        approximate_roots.update(new_roots)

    return approximate_roots


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
