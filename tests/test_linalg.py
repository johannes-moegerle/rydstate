from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pytest
from rydstate.linalg import _windows, calc_nullvector, find_roots

dx_list = [0.5, 0.3, 0.1, 0.01, 0.015, 0.011]


@pytest.mark.parametrize("dx", dx_list)
def test_find_roots_detects_roots_on_grid_samples(dx: float) -> None:
    roots = find_roots(lambda x: x * (x - 0.5) * (x - 1), 0, 1, min_dx=dx)

    assert roots == pytest.approx([0, 0.5, 1])


@pytest.mark.parametrize("dx", dx_list)
def test_find_roots_detects_integer_endpoint_roots(dx: float) -> None:
    func = lambda nu: np.sin(np.pi * nu)  # noqa: E731
    reference_roots = [30, 31, 32, 33, 34, 35]

    roots = find_roots(func, 30, 35, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("x_null", [(26.9739, 26.9747), (26.8881, 26.8882)])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_nearly_degenerate_pair_within_one_grid_cell(
    x_null: tuple[float, float], dx: float, scale: float
) -> None:
    # Two roots much closer than min_dx produce no sign change on the grid,
    # only a dip of |func|, which is also detected by find_roots.
    # (e.g. the Yb174 6sng G J=4 MQDT model, whose two eigen quantum defects differ by less than 1e-3)
    # The overall scale of func must not matter here, since det(M) of the different MQDT models
    # ranges from order 1e-6 to order 1.
    func = lambda nu: scale * (nu - x_null[0]) * (nu - x_null[1]) * np.tan(nu * np.pi)  # noqa: E731

    reference_roots = [*x_null, 27]
    roots = find_roots(func, 26.5, 27.5, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_pair_whose_dip_is_masked_by_a_neighbouring_root(dx: float, scale: float) -> None:
    # A pair of roots within one grid cell is not always visible as a dip of |func|:
    # if a third root sits in a neighbouring cell, |func| decreases monotonically across all grid
    # points of the pair, so none of them is a local minimum. Such a cell is instead detected by
    # how flat func is in it compared to how much func changes per cell.
    # (e.g. the Sr87 D F=9/2 model with decoupled outer channels, whose roots these are)
    reference_roots = [46.18130, 46.18792, 46.19994]
    func = lambda nu: scale * np.prod([nu - root for root in reference_roots])  # noqa: E731

    roots = find_roots(func, 46.1, 46.3, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_a_pair_of_roots_at_the_edge_of_its_grid_cell(dx: float, scale: float) -> None:
    # A pair of roots close to one end of its grid cell leaves |func| large at the other end, so the
    # cell does not look flat, no matter how close the two roots are. Such a pair is instead detected
    # by the dip of |func| at the end of the cell it is close to.
    # (e.g. the Sr87 F F=9/2 model, whose two pairs of roots these are)
    reference_roots = [23.87882913, 23.87883084, 23.88918434, 23.88918605]
    func = lambda nu: scale * np.prod([nu - root for root in reference_roots])  # noqa: E731

    roots = find_roots(func, 23.85, 23.92, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-8, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_roots_next_to_a_pole(dx: float, scale: float) -> None:
    # A pole flips the sign of func just like a root does, so the cell containing a pole and a root
    # shows no sign change at all, and |func| is far too large at the pole for the cell to look flat.
    # Such a cell is instead detected by how much more func changes in it than in its neighbours.
    # This models the pole of an unscaled MQDT determinant.
    reference_roots = [26.48046499, 26.50495498]  # the second one is only 5e-3 away from the pole
    func = lambda nu: scale * (nu - reference_roots[0]) * (nu - reference_roots[1]) * np.tan(nu * np.pi)  # noqa: E731

    roots = find_roots(func, 25.5, 27.5, min_dx=dx)
    # 26 and 27 are the zeros of the tangent; the pole at 26.5 must not show up as a root
    np.testing.assert_allclose(roots, [26, reference_roots[0], reference_roots[1], 27], atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1])
@pytest.mark.parametrize("distance", [5e-3, 1e-3, 5e-4])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_a_root_next_to_a_pole_between_two_grid_points(
    dx: float, scale: float, distance: float
) -> None:
    # A pole that sits between two grid points raises |func| by a large but finite factor only, so
    # its cell is far less conspicuous than one whose grid point happens to fall onto the pole.
    # In these cases |func| dips on the far side of the pole, prompting finer sampling.
    # The interval is chosen such that no grid point falls onto the pole at 26.5.
    root = 26.5 + distance
    func = lambda nu: scale * (nu - root) * np.tan(nu * np.pi)  # noqa: E731

    roots = find_roots(func, 25.5321, 27.4137, min_dx=dx)
    # 26 and 27 are the zeros of the tangent; the pole at 26.5 must not show up as a root
    np.testing.assert_allclose(roots, [26, root, 27], atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("dx", dx_list)
def test_find_roots_detects_a_pair_of_roots_with_one_of_them_exactly_on_a_grid_point(dx: float, scale: float) -> None:
    # A root that happens to sit exactly on a grid point makes func return exactly 0.0 there, which is
    # common for roots at round decimals. Such a grid point is a root of its own, but neither of the two
    # cells next to it shows a sign change, so a second root in one of them is only found by searching
    # that cell again with a finer grid.
    reference_roots = [26.53, 26.5301]

    def func(nu: float) -> float:
        return scale * (nu - reference_roots[0]) * (nu - reference_roots[1])

    assert func(reference_roots[0]) == 0.0, "the first root must fall exactly onto a grid point"

    roots = find_roots(func, 26.5, 26.6, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1])
@pytest.mark.parametrize("side", [-1, 1])
def test_find_roots_detects_a_root_next_to_an_infinite_grid_sample(side: int, scale: float) -> None:
    # Binary-exact endpoints and spacing put the pole on the grid. Its value has
    # the sign of the right-hand limit, which hides a root on the left-hand side.
    pole, root = 26.5, 26.5 + side * 0.0031

    def func(nu: float) -> float:
        if nu == pole:
            return -side * np.inf
        return scale * (nu - root) / (nu - pole)

    roots = find_roots(func, 26, 27, min_dx=1 / 32)
    np.testing.assert_allclose(roots, [root], atol=1e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-6, 1])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_a_pair_of_roots_in_the_cell_next_to_a_pole_on_a_grid_point(dx: float, scale: float) -> None:
    # A pair of roots shows no sign change, so it is only found by searching its cell again with a
    # finer grid. This must also happen when that cell borders a pole that falls exactly onto a grid
    # point, even though the finer grid reproduces the infinite endpoint of the cell.
    pole, reference_roots = 26.5, [26.5076, 26.5087]

    def func(nu: float) -> float:
        if nu == pole:  # what the division by +0.0 returns, the numerator being positive here
            return np.inf
        return scale * (nu - reference_roots[0]) * (nu - reference_roots[1]) / (nu - pole)

    assert not np.isfinite(func(pole)), "the pole must fall exactly onto a grid point"

    roots = find_roots(func, 26.4, 26.6, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-10, rtol=0)


def test_windows_are_shifted_inwards_at_the_boundaries() -> None:
    # Each window must cover the full number of entries, also the ones at the boundaries. A truncated
    # window would under-estimate the maximum change of func per cell exactly at the ends of the
    # intervals that find_roots searches again, and thus not recognize them as flat.
    values = np.array([1.0, 2, 3, 4, 5, 6, 7])

    np.testing.assert_array_equal(np.max(_windows(values, 3), axis=-1), [3, 3, 4, 5, 6, 7, 7])
    np.testing.assert_array_equal(np.median(_windows(values, 3), axis=-1), [2, 2, 3, 4, 5, 6, 6])
    np.testing.assert_array_equal(np.max(_windows(values, 7), axis=-1), [7] * 7)
    # a window longer than the values simply covers all of them
    np.testing.assert_array_equal(np.max(_windows(values, 9), axis=-1), [7] * 7)


@pytest.mark.parametrize("sep", [1e-8, 1e-10, 1e-12])
def test_find_roots_returns_roots_that_are_barely_distinguishable(sep: float) -> None:
    # Brent's method still locates two roots this close together separately, so both are returned.
    # The duplicate check must use the refinement accuracy, not a fixed distance cutoff.
    reference_roots = [0.5015, 0.5015 + sep]
    func = lambda x: (x - reference_roots[0]) * (x - reference_roots[1])  # noqa: E731

    roots = find_roots(func, 0, 1, min_dx=0.01)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-13, rtol=0)


@pytest.mark.parametrize("dx", dx_list)
def test_find_roots_at_the_endpoints_of_the_interval(dx: float) -> None:
    # Since the grid is padded by one dx on each side, the endpoints are ordinary grid points,
    # i.e. whether a root close to an endpoint is returned only depends on the position of the root,
    # and not on the value of func at the endpoint (which is smaller than atol in all cases here).
    assert find_roots(lambda x: (x - 30 - 1e-10) * 1e-8, 30, 31, min_dx=dx) == pytest.approx([30 + 1e-10])
    assert find_roots(lambda x: (x - 31 + 1e-10) * 1e-8, 30, 31, min_dx=dx) == pytest.approx([31 - 1e-10])

    # roots slightly outside of the interval are not returned
    assert find_roots(lambda x: (x - 30 + 1e-10) * 1e-8, 30, 31, min_dx=dx) == []
    assert find_roots(lambda x: (x - 31 - 1e-10) * 1e-8, 30, 31, min_dx=dx) == []


@pytest.mark.parametrize("scale", [1, 1e3, 1e6])
@pytest.mark.parametrize("dx", dx_list)
def test_find_roots_ignores_dips_that_do_not_reach_zero(dx: float, scale: float) -> None:
    # A local minimum of |func| that stays clearly away from zero must not be reported as a root,
    # again independently of the overall scale of func.
    func = lambda x: scale * ((x - 0.5015) ** 2 + 1e-2)  # noqa: E731

    assert find_roots(func, 0, 1, min_dx=dx) == []


@pytest.mark.parametrize("method", ["numpy_svd", "scipy_svd", "scipy_svd_gesvd"])
def test_calc_nullvector_singular_matrix(
    method: Literal["numpy_svd", "scipy_svd", "scipy_svd_gesvd"], caplog: pytest.LogCaptureFixture
) -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])  # exactly singular

    with caplog.at_level(logging.WARNING):
        nullvector = calc_nullvector(matrix, method=method)

    assert np.linalg.norm(nullvector) == pytest.approx(1)
    assert np.linalg.norm(matrix @ nullvector) == pytest.approx(0, abs=1e-14)
    assert caplog.records == []


def test_calc_nullvector_almost_singular_matrix(caplog: pytest.LogCaptureFixture) -> None:
    # even if the matrix is not singular (up to rcond), we still return the best possible nullvector
    angle = 0.3
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = rot @ np.diag([1.0, 1e-4]) @ rot.T

    with caplog.at_level(logging.WARNING):
        nullvector = calc_nullvector(matrix)

    assert abs(nullvector @ rot[:, 1]) == pytest.approx(1)  # the singular vector of the smallest singular value
    assert "not singular" in caplog.text


def test_calc_nullvector_degenerate_nullspace(caplog: pytest.LogCaptureFixture) -> None:
    matrix = np.diag([1.0, 0.0, 0.0])

    with caplog.at_level(logging.WARNING):
        nullvector = calc_nullvector(matrix)

    assert np.linalg.norm(matrix @ nullvector) == pytest.approx(0, abs=1e-14)
    assert "more than one vector" in caplog.text


def test_calc_nullvector_one_by_one_matrix() -> None:
    assert calc_nullvector(np.array([[0.0]])) == pytest.approx([1.0])

    with pytest.raises(RuntimeError, match="Matrix is 1x1 but not close to zero"):
        calc_nullvector(np.array([[1.0]]))
