from __future__ import annotations

import numpy as np
import pytest
from rydstate.linalg import find_roots

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


@pytest.mark.parametrize("x_null", [(26.9739, 26.9747), (26.8881, 26.8882)])
@pytest.mark.parametrize("dx", [dx for dx in dx_list if dx <= 0.02])
def test_find_roots_detects_nearly_degenerate_pair_within_one_grid_cell(x_null: tuple[float, float], dx: float) -> None:
    # Two roots much closer than min_dx produce no sign change on the grid,
    # only a dip of |func|, which is also detected by find_roots.
    # (e.g. the Yb174 6sng G J=4 MQDT model, whose two eigen quantum defects differ by less than 1e-3)
    func = lambda nu: (nu - x_null[0]) * (nu - x_null[1]) * np.tan(nu * np.pi)  # noqa: E731

    reference_roots = [*x_null, 27]
    roots = find_roots(func, 26.5, 27.5, min_dx=dx)
    np.testing.assert_allclose(roots, reference_roots, atol=1e-13, rtol=1e-13)
