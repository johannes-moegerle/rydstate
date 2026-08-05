from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pytest
from rydstate.linalg import calc_nullvector, find_roots

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
