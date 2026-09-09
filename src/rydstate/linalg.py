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
    """Find roots of func in [x_min, x_max] by sampling and adaptive refinement.

    Sample a uniform grid, then refine sign changes with Brent's method. To resolve roots
    sharing a grid cell, search again where |func| is small relative to nearby changes or
    has a local dip. Also refine unusually steep cells, which may contain a root and a pole.

    This is a heuristic search: roots can be missed if the samples give no indication of
    them, especially near poles. Decrease min_dx to check whether a spectrum is resolved.

    Args:
        func: 1D scalar function to find roots of.
        x_min: Left endpoint of search interval.
        x_max: Right endpoint of search interval.
        min_dx: Initial grid spacing. Selected cells are searched with finer grids.
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

        # _find_approximate_roots extends the grid by one dx, so it also finds roots outside of the interval
        if not x_min - xtol <= root <= x_max + xtol:
            continue

        val = func(root)
        if not np.isfinite(val) or abs(val) > 1e10:  # a singularity of func, not a root
            continue
        if abs(val) > atol:
            logger.warning("Root not close to zero: x=%f f(x)=%e. Skipping.", root, val)
            continue

        roots.append(float(root))

    roots.sort()
    # brentq is only accurate to xtol plus the relative tolerance 4 * eps it uses internally, so two
    # roots closer than the sum of these error bounds may be duplicate brackets.
    # Reject this ambiguity rather than silently merging potentially distinct roots.
    max_root_error = xtol + 4 * np.finfo(float).eps * np.abs(roots)
    if np.any(np.diff(roots) < max_root_error[:-1] + max_root_error[1:]):
        raise ValueError(f"Found roots closer than the refinement accuracy: {roots}.")

    return roots


def _find_approximate_roots(
    func: Callable[[float], float],
    x_min: float,
    x_max: float,
    min_dx: float = 1e-2,
    extend_grid: bool = True,
    *,
    pole_refinements_left: int = 2,
) -> dict[float, tuple[float, float]]:
    """Collect exact grid roots and disjoint brackets, refining selected cells recursively.

    Limit refinements prompted by poles, since a genuine singularity never becomes a
    root. Hints of close roots may still prompt further refinement down to dx < 1e-8.
    """
    assert x_min <= x_max, "x_min must be less than or equal to x_max"
    approximate_roots: dict[float, tuple[float, float]] = {}

    if x_min == x_max:
        xs = np.array([x_min])
        dx = min_dx
    else:
        n_grid = math.ceil((x_max - x_min) / min_dx) + 1
        xs = np.linspace(x_min, x_max, n_grid)
        dx = xs[1] - xs[0]

    # the roots this finds outside of [x_min, x_max] are discarded again by find_roots
    if extend_grid:
        xs = np.concatenate(([xs[0] - dx], xs, [xs[-1] + dx]))

    fs = np.array([func(x) for x in xs])
    sign_fs = np.sign(fs)

    zeros = fs == 0
    approximate_roots.update({x: (x, x) for x in xs[zeros]})

    # np.sign is 0 for a zero and nan for a nan, so a sign change implies nonzero, non-nan endpoints
    sign_change = sign_fs[:-1] * sign_fs[1:] < 0

    refine = np.zeros_like(sign_change)
    possible_poles = np.zeros_like(sign_change)
    if dx >= 1e-8 and len(fs) >= 4:
        close_roots, possible_poles = _root_refinement_masks(fs)
        refine = close_roots | (possible_poles & (pole_refinements_left > 0))

    # cells that are searched again below are skipped here, since searching them again also
    # brackets their sign change, and the same root must not be bracketed twice
    conditions = sign_change & np.bitwise_not(refine)

    approximate_roots.update(
        {
            (x_left + x_right) / 2: (x_left, x_right)
            for x_left, x_right in zip(xs[:-1][conditions], xs[1:][conditions], strict=True)
        }
    )

    # Search cells separately, without padding: their interiors do not overlap, and exact
    # endpoint roots are merged by the dictionary. Zero or infinite endpoints do not rule
    # out other roots inside the cell.
    for i in np.flatnonzero(refine):
        new_roots = _find_approximate_roots(
            func,
            xs[i],
            xs[i + 1],
            min_dx=dx / 10,
            extend_grid=False,
            pole_refinements_left=pole_refinements_left - int(possible_poles[i]),
        )
        if len(new_roots) == 0 and sign_change[i]:
            logger.warning(
                "The finer grid resolved nothing in [%f, %f], falling back to the sign change of the whole cell.",
                *(xs[i], xs[i + 1]),
            )
            new_roots = {(xs[i] + xs[i + 1]) / 2: (xs[i], xs[i + 1])}
        approximate_roots.update(new_roots)

    return approximate_roots


def _root_refinement_masks(fs: NDArray) -> tuple[NDArray, NDArray]:
    """Mark cells that may hide close roots and cells that may contain poles.

    Compare against nearby changes, so the decisions do not depend on the overall
    scale of func. These tests suggest where to sample again; they do not prove that
    a cell contains a root or a pole.
    """
    abs_fs = np.abs(fs)
    sign_fs = np.sign(fs)
    finite = np.isfinite(fs)
    # Non-finite samples need refinement too, but must not dominate the local scale.
    with np.errstate(invalid="ignore"):
        changes = np.abs(np.diff(fs))
    changes[~np.isfinite(changes)] = 0

    window = 7
    typical_change = np.median(_windows(changes, window), axis=-1)
    # Treat changes far above the local median as possible poles and exclude them
    # from the scale used to decide whether neighbouring values are near zero.
    possible_poles = (changes > 5 * typical_change) | ~finite[:-1] | ~finite[1:]
    local_scale = np.max(_windows(np.where(possible_poles, 0, changes), window), axis=-1)

    # Both endpoints near zero can signal several unresolved roots, even if the
    # samples are monotonic. A linear crossing has at least one endpoint >= 0.5
    # times its change per cell, so use a smaller threshold.
    small_ratio = 0.25
    small_endpoints = np.maximum(abs_fs[:-1], abs_fs[1:]) < small_ratio * local_scale

    # A pair near the edge of a cell may leave just one endpoint near zero.
    # Look for a dip without a sign change and refine both neighbouring cells.
    dips = np.zeros_like(fs, dtype=bool)
    dips[1:-1] = (
        (abs_fs[1:-1] <= abs_fs[:-2])
        & (abs_fs[1:-1] <= abs_fs[2:])
        & (sign_fs[:-2] == sign_fs[1:-1])
        & (sign_fs[1:-1] == sign_fs[2:])
        & (abs_fs[1:-1] < small_ratio * np.maximum(local_scale[:-1], local_scale[1:]))
    )
    return small_endpoints | dips[:-1] | dips[1:], possible_poles


def _windows(values: NDArray, window: int) -> NDArray:
    """Windows of `window` entries of values around each of its entries.

    Near the boundaries the windows are shifted inwards instead of being truncated, since a truncated
    window misjudges the scale of func exactly at the ends of the intervals searched again.
    """
    if len(values) <= window:
        return np.broadcast_to(values, (len(values), len(values)))
    starts = np.clip(np.arange(len(values)) - window // 2, 0, len(values) - window)
    return np.lib.stride_tricks.sliding_window_view(values, window)[starts]


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
