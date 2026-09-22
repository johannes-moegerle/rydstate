from __future__ import annotations

import numpy as np
import pytest
from rydstate.radial import Radial, RadialDummy


def _make_radial(*, dz: float = 0.01, zmax: float = 6.0, center: float = 3.0, width: float = 0.5) -> Radial:
    """Build a normalized gaussian-shaped radial wavefunction on the standard grid [0, dz, 2*dz, ...]."""
    z = np.arange(dz, zmax + dz / 2, dz)
    w = np.exp(-0.5 * ((z - center) / width) ** 2)
    radial = Radial(z, w)
    return radial / radial.norm


# --------------------------------------------------------------------------------------
# grid / wavefunction properties
# --------------------------------------------------------------------------------------


def test_grid_properties() -> None:
    dz = 0.1
    z = np.arange(dz, 2 + dz / 2, dz)
    radial = Radial(z, np.ones_like(z))

    assert radial.steps == len(z)
    assert np.isclose(radial.dz, dz)
    assert np.allclose(radial.x_list, z**2)
    assert np.allclose(radial.u_list, np.sqrt(z) * radial.w_list)


def test_norm_matches_manual_formula() -> None:
    dz = 0.1
    z = np.arange(dz, 3 + dz / 2, dz)
    w = np.exp(-0.5 * (z - 1.5) ** 2)
    radial = Radial(z, w)

    expected = np.sqrt(2 * np.sum(w * w * z * z) * dz)
    assert np.isclose(radial.norm, expected)


def test_normalization_yields_unit_norm() -> None:
    assert np.isclose(_make_radial().norm, 1.0)


# --------------------------------------------------------------------------------------
# arithmetic: addition, subtraction and negation
# --------------------------------------------------------------------------------------


def test_add_on_shared_grid() -> None:
    dz = 0.1
    z = np.arange(dz, 1 + dz / 2, dz)
    a = Radial(z, np.ones_like(z))
    b = Radial(z, 2 * np.ones_like(z))

    result = a + b
    assert np.allclose(result.z_list, z)
    assert np.allclose(result.w_list, 3.0)


def test_add_zero_pads_onto_common_grid() -> None:
    """Two states on offset (but lattice-aligned) grids are zero-padded onto a common grid."""
    dz = 0.1
    za = np.arange(dz, 1 + dz / 2, dz)  # covers [0.0, 1.0]
    zb = np.arange(0.5, 1.5 + dz / 2, dz)  # covers [0.5, 1.5]
    a = Radial(za, np.ones_like(za))
    b = Radial(zb, np.ones_like(zb))

    result = a + b
    # common grid spans the union of both grids
    assert np.isclose(result.z_list[0], dz)
    assert np.isclose(result.z_list[-1], 1.5)

    # where only `a` lives (z < 0.5) -> 1, in the overlap (0.5 <= z <= 1.0) -> 2, only `b` (z > 1.0) -> 1
    for z, w in zip(result.z_list, result.w_list, strict=True):
        if z < 0.5 - 1e-9:
            assert np.isclose(w, 1.0)
        elif z <= 1.0 + 1e-9:
            assert np.isclose(w, 2.0)
        else:
            assert np.isclose(w, 1.0)


def test_add_is_commutative() -> None:
    dz = 0.1
    za = np.arange(dz, 1 + dz / 2, dz)
    zb = np.arange(0.5, 1.5 + dz / 2, dz)
    a = Radial(za, np.ones_like(za))
    b = Radial(zb, 2 * np.ones_like(zb))

    assert np.allclose((a + b).w_list, (b + a).w_list)


def test_neg_and_sub() -> None:
    dz = 0.1
    z = np.arange(dz, 1 + dz / 2, dz)
    a = Radial(z, np.full_like(z, 3.0))
    b = Radial(z, np.full_like(z, 1.0))

    assert np.allclose((-a).w_list, -3.0)
    assert np.allclose((a - b).w_list, 2.0)
    # subtraction equals addition of the negation
    assert np.allclose((a - b).w_list, (a + (-b)).w_list)


def test_add_returns_not_implemented_for_non_radial() -> None:
    dz = 0.1
    z = np.arange(dz, 1 + dz / 2, dz)
    a = Radial(z, np.ones_like(z))
    with pytest.raises(TypeError):
        _ = a + 5  # type: ignore[operator]


# --------------------------------------------------------------------------------------
# arithmetic: scalar multiplication and division
# --------------------------------------------------------------------------------------


def test_scalar_multiplication() -> None:
    dz = 0.1
    z = np.arange(dz, 1 + dz / 2, dz)
    a = Radial(z, np.full_like(z, 2.0))

    assert np.allclose((a * 3).w_list, 6.0)
    assert np.allclose((3 * a).w_list, 6.0)  # __rmul__
    assert np.allclose((a / 2).w_list, 1.0)  # __truediv__
    # grid is untouched by scalar operations
    assert np.allclose((a * 3).z_list, z)


def test_mul_returns_not_implemented_for_non_number() -> None:
    dz = 0.1
    z = np.arange(dz, 1 + dz / 2, dz)
    a = Radial(z, np.ones_like(z))
    with pytest.raises(TypeError):
        _ = a * "x"  # type: ignore[operator]


def test_norm_scales_linearly_with_scalar() -> None:
    a = _make_radial()
    assert np.isclose((3 * a).norm, 3 * a.norm)
    assert np.isclose((a / 4).norm, a.norm / 4)


# --------------------------------------------------------------------------------------
# _align error handling
# --------------------------------------------------------------------------------------


def test_align_rejects_different_dz() -> None:
    a = Radial(np.arange(0.2, 1 + 0.05, 0.1), np.ones(9))
    b = Radial(np.arange(0.2, 1 + 0.1, 0.2), np.ones(5))
    with pytest.raises(ValueError, match="different dz"):
        _ = a + b


def test_rejects_grids_off_the_global_lattice() -> None:
    z = np.arange(0.03, 1 + 0.03, 0.1)
    with pytest.raises(ValueError, match="z_list must start at an integer multiple of dz"):
        Radial(z, np.ones_like(z))


# --------------------------------------------------------------------------------------
# calc_overlap / calc_matrix_element
# --------------------------------------------------------------------------------------


def test_self_overlap_of_normalized_state_is_one() -> None:
    a = _make_radial()
    assert np.isclose(a.calc_overlap(a), 1.0)


def test_overlap_is_symmetric() -> None:
    a = _make_radial(center=2.5)
    b = _make_radial(center=3.5)
    assert np.isclose(a.calc_overlap(b), b.calc_overlap(a))


def test_overlap_is_linear_in_scalar() -> None:
    a = _make_radial(center=2.5)
    b = _make_radial(center=3.5)
    assert np.isclose(a.calc_overlap(2 * b), 2 * a.calc_overlap(b))


def test_calc_matrix_element_k_radial_zero_equals_overlap() -> None:
    a = _make_radial(center=2.5)
    b = _make_radial(center=3.5)
    assert np.isclose(a.calc_matrix_element(b, k_radial=0, unit="a.u."), a.calc_overlap(b))


def test_calc_matrix_element_unit_handling() -> None:
    a = _make_radial(center=2.5)
    b = _make_radial(center=3.5)

    me_pint = a.calc_matrix_element(b, k_radial=1)  # default -> pint quantity in a0
    me_au = a.calc_matrix_element(b, k_radial=1, unit="a.u.")  # raw float in atomic units

    assert isinstance(me_au, float)
    # the pint quantity carries the atomic-unit magnitude in units of a0
    assert np.isclose(me_pint.magnitude, me_au)
    # explicitly requesting a unit returns the converted magnitude of the same quantity
    me_um = a.calc_matrix_element(b, k_radial=1, unit="micrometer")
    assert np.isclose(me_um, me_pint.to("micrometer").magnitude)


# --------------------------------------------------------------------------------------
# RadialDummy
# --------------------------------------------------------------------------------------


def test_radial_dummy_norm_is_abs_coefficient() -> None:
    assert RadialDummy(2.0, nu=50).norm == 2.0
    assert RadialDummy(-3.0, nu=50).norm == 3.0


def test_radial_dummy_multiplication_returns_dummy() -> None:
    d = RadialDummy(2.0, nu=50)

    scaled = d * 3
    assert isinstance(scaled, RadialDummy)
    assert scaled.norm == 6.0
    assert scaled.nu == 50

    # __rmul__, __truediv__ and __neg__ all route through RadialDummy.__mul__
    assert isinstance(3 * d, RadialDummy)
    assert isinstance(d / 2, RadialDummy)
    assert isinstance(-d, RadialDummy)
    assert (3 * d).norm == 6.0
    assert (d / 2).norm == 1.0
    assert (-d).norm == 2.0


def test_radial_dummy_mul_returns_not_implemented_for_non_number() -> None:
    d = RadialDummy(2.0, nu=50)
    with pytest.raises(TypeError):
        _ = d * "x"  # type: ignore[operator]
