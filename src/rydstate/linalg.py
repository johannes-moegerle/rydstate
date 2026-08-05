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
    min_dx: float = 1e-2,
    atol: float = 1e-9,
    xtol: float = 1e-13,
) -> list[float]:
    """Find all roots of func in [x_min, x_max].

    Uses a dense uniform grid to detect sign changes, then refines each bracket with Brent's method.
    A pair of roots lying between two adjacent grid points has no sign change on
    the grid, but shows up as a local minimum of |func|;
    We refine such dips by recursively calling _find_approximate_roots on a smaller interval around the dip.

    Args:
        func: 1D scalar function to find roots of.
        x_min: Left endpoint of search interval.
        x_max: Right endpoint of search interval.
        min_dx: Grid spacing used to detect sign changes and dips.
            Isolated roots and pairs of roots are found even if they are closer than
            min_dx, as long as the grid resolves the dip of |func| around them.
            Three or more roots within one grid cell can still be missed.
        atol: Absolute tolerance for root validation.
        xtol: Absolute tolerance for root refinement.

    Returns:
        Sorted list of x values where func(x) ≈ 0.

    """
    if x_min > x_max:
        return []

    approximate_roots = _find_approximate_roots(func, x_min, x_max, min_dx=min_dx, extend_grid=True)

    roots: list[float] = []
    for x_left, x_right in approximate_roots.values():
        if x_left == x_right:
            root = x_left
        else:
            try:
                root = brentq(func, x_left, x_right, xtol=xtol)
            except ValueError:
                logger.warning("Brent's method failed to find root in [%f, %f], skipping.", x_left, x_right)
                continue

        # the grid used in _find_approximate_roots is extended by one dx on each side, so it can also find roots
        # at the boundary (or slightly outside) of [x_min, x_max].
        # Points outside the interval (up to xtol) are discarded here.
        if not x_min - xtol <= root <= x_max + xtol:
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


def _find_approximate_roots(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    min_dx: float = 1e-2,
    extend_grid: bool = True,
) -> dict[float, tuple[float, float]]:
    assert x_min <= x_max, "x_min must be less than or equal to x_max"
    approximate_roots: dict[float, tuple[float, float]] = {}

    if x_min == x_max:
        xs = np.array([x_min])
        dx = min_dx
    else:
        n_grid = math.ceil((x_max - x_min) / min_dx) + 1
        xs = np.linspace(x_min, x_max, n_grid)
        dx = xs[1] - xs[0]

    # extend the grid by one dx on each side
    # the roots this finds outside of [x_min, x_max] are discarded again by find_roots
    if extend_grid:
        xs = np.concatenate(([xs[0] - dx], xs, [xs[-1] + dx]))

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

    # find dips in abs(func(x)) that are not detected by the sign change
    is_dip = (
        (abs_fs[1:-1] <= abs_fs[:-2])
        & (abs_fs[1:-1] <= abs_fs[2:])
        & (sign_fs[:-2] == sign_fs[1:-1])
        & (sign_fs[1:-1] == sign_fs[2:])
    )

    # How far (in units of the grid spacing) the roots of the local parabola may lie off the real axis,
    # before we consider a dip of |func| to be unrelated to any root
    max_dip_miss_in_dx = 2.0

    for i, x in zip(np.where(is_dip)[0] + 1, xs[1:-1][is_dip], strict=True):
        # Fit a parabola p(t) = a * t**2 + b * t + c through the dip and its two neighbours,
        # where t = (x - xs[i]) / dx is the distance from the dip in units of the grid spacing
        c = fs[i]
        b = (fs[i + 1] - fs[i - 1]) / 2
        a = (fs[i + 1] - 2 * fs[i] + fs[i - 1]) / 2
        # The parabola crosses zero if its discriminant is non-negative. If it is negative, the parabola has
        # a pair of complex roots t = -b / (2 * a) +- i * sqrt(-discriminant) / (2 * |a|), i.e. it misses the
        # real axis by sqrt(-discriminant) / (2 * |a|) grid spacings. Only if this miss is larger than
        # max_dip_miss_in_dx grid spacings, we assume the dip is not due to a pair of roots and skip it.
        # Note that this condition is invariant under rescaling func
        discriminant = b**2 - 4 * a * c
        if discriminant < -4 * (max_dip_miss_in_dx * a) ** 2:
            continue
        if dx < 1e-8:
            logger.warning(
                "Found a dip which up to dx=%e does not cross zero, but is very close to zero: x=%f f(x)=%e. Skipping.",
                *(dx, x, fs[i]),
            )
            continue
        new_roots = _find_approximate_roots(func, x - dx, x + dx, min_dx=dx * 1e-2, extend_grid=False)
        approximate_roots.update(new_roots)

    return approximate_roots


def calc_nullvector(
    matrix: NDArray,
    method: Literal["numpy_svd", "scipy_svd", "scipy_svd_gesvd"] = "scipy_svd",
) -> NDArray:
    """Calculate the nullvector of a matrix, which is singular by construction (like the MQDT M-matrix).

    We always return the right singular vector belonging to the smallest singular value,
    i.e. the best possible nullvector, even if the matrix is only approximately singular
    (e.g. because the root of det(M) was not located exactly).

    Args:
        matrix: The (by construction singular) matrix to calculate the nullvector of.
        method: Which routine to use for the singular value decomposition.

    Returns:
        The right singular vector belonging to the smallest singular value.

    """
    tol = 1e-6
    if matrix.shape == (1, 1):
        if abs(matrix[0, 0]) > tol:
            raise RuntimeError(f"Matrix is 1x1 but not close to zero (value={matrix[0, 0]}), this should not happen.")
        return np.array([1.0])
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"

    if method == "numpy_svd":
        _u, s, vt = np.linalg.svd(matrix)
    elif method == "scipy_svd":
        _u, s, vt = scipy.linalg.svd(matrix)
    elif method == "scipy_svd_gesvd":
        _u, s, vt = scipy.linalg.svd(matrix, lapack_driver="gesvd")
    else:
        raise ValueError(f"Invalid method: {method}")

    if s[0] == 0:
        raise RuntimeError("Matrix is entirely zero, this should not happen.")
    if len(s) > 1 and s[-1] / s[0] > tol:
        logger.warning("Matrix is not singular (s[-1]/s[0]=%.1e), the nullvector is only approximate.", s[-1] / s[0])
    elif len(s) > 2 and s[-2] <= 10 * s[-1]:
        logger.warning(
            "Nullspace has more than one vector (s[-1]/s[0]=%.1e, s[-2]/s[0]=%.1e), "
            "returning the one with the smallest singular value.",
            *(s[-1] / s[0], s[-2] / s[0]),
        )

    return np.array(vt[-1])
