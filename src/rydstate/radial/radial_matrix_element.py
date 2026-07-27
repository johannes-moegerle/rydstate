from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.integrate

if TYPE_CHECKING:
    from rydstate.units import NDArray

logger = logging.getLogger(__name__)

INTEGRATION_METHODS = Literal["sum", "trapezoid", "scipy_simpson", "scipy_trapezoid"]
RadialMatrixElementOperator = int | Literal["electric_dipole_closed_shell_core"]


def calc_radial_matrix_element(
    z1: NDArray, w1: NDArray, z2: NDArray, w2: NDArray, k_radial: int, integration_method: INTEGRATION_METHODS = "sum"
) -> float:
    r"""Calculate the radial matrix element of two wavefunctions w1(z1) and w2(z2).

    Computes the integral

    .. math::
        \int_{0}^{\infty} dz 2 z^{2 + 2 k_{radial}} w_1(z) w_2(z)
        = \int_{0}^{\infty} dx x^k_{radial} \tilde{u}_1(x) \tilde{u}_2(x)
        = a_0^{-k_{radial}} \int_{0}^{\infty} dr r^2 r^k_{radial} R_1(r) R_2(r)

    where R_1 and R_2 are the radial wavefunctions of the two states
    and w(z) = z^{-1/2} \tilde{u}(z^2) = (r/_a_0)^{1/4} \sqrt{a_0} r R(r).

    Args:
        z1: z coordinates of the first wavefunction
        w1: w(z) values of the first wavefunction
        z2: z coordinates of the second wavefunction
        w2: w(z) values of the second wavefunction
        k_radial: Power of r in the matrix element
        integration_method: Integration method to use, one of ["sum", "trapezoid", "scipy_simpson", "scipy_trapezoid"]
            (default="sum")

    Returns:
        float: The radial matrix element

    """
    z, w1, w2 = _get_overlapping_wavefunctions(z1, w1, z2, w2)

    integrand = 2 * w1 * w2
    integrand = _multiply_by_powers(integrand, z, 2 * k_radial + 2)

    return _integrate(integrand, z, integration_method)


def calc_radial_matrix_element_electric_dipole_closed_shell_core(
    z1: NDArray,
    w1: NDArray,
    z2: NDArray,
    w2: NDArray,
    alpha_closed_shell_core: float,
    r_c: float,
    integration_method: INTEGRATION_METHODS = "sum",
) -> float:
    z, w1, w2 = _get_overlapping_wavefunctions(z1, w1, z2, w2)

    # electric_dipole_closed_shell_core: 2 w1 w2 z^2 * (alpha_c / x^2) * (exp(-(x/r_c)^3) - 1) with x = z^2,
    # see also docstring of ElementProperties.alpha_closed_shell_core
    x = z * z
    x3 = x * x * x
    integrand = 2 * w1 * w2 * (alpha_closed_shell_core / x) * np.expm1(-(x3 / r_c**3))

    return _integrate(integrand, z, integration_method)


def _multiply_by_powers(result: NDArray, base: NDArray, exponent: int) -> NDArray:
    """Calculate result * base**(exponent) in an optimized way."""
    base_powers = {0: base}
    for i in range(exponent):
        if (exponent // 2**i) % 2 == 1:
            result *= base_powers[i]
            exponent -= 2**i
        if exponent == 0:
            break
        base_powers[i + 1] = np.square(base_powers[i])
    return result


def _get_overlapping_wavefunctions(
    z1: NDArray, w1: NDArray, z2: NDArray, w2: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Get the overlapping wavefunctions for integration.

    Args:
        z1: z coordinates of the first wavefunction
        w1: w(z) values of the first wavefunction
        z2: z coordinates of the second wavefunction
        w2: w(z) values of the second wavefunction

    Returns:
        tuple[NDArray, NDArray, NDArray]: (z, w1, w2) z coordinates and w(z) values of the overlapping wavefunctions

    """
    z_min = max(z1[0], z2[0])
    z_max = min(z1[-1], z2[-1])
    if z_max <= z_min:
        return np.array([]), np.array([]), np.array([])

    # Select overlapping points
    dz = z1[1] - z1[0]
    if z1[0] < z_min - dz / 2:
        ind = int((z_min - z1[0]) / dz + 0.5)
        z1 = z1[ind:]
        w1 = w1[ind:]
    elif z2[0] < z_min - dz / 2:
        ind = int((z_min - z2[0]) / dz + 0.5)
        z2 = z2[ind:]
        w2 = w2[ind:]

    if z1[-1] > z_max + dz / 2:
        ind = int((z1[-1] - z_max) / dz + 0.5)
        z1 = z1[:-ind]
        w1 = w1[:-ind]
    elif z2[-1] > z_max + dz / 2:
        ind = int((z2[-1] - z_max) / dz + 0.5)
        z2 = z2[:-ind]
        w2 = w2[:-ind]

    # do some sanity checks
    tol = 1e-10
    assert len(z1) == len(z2), f"Length mismatch: {len(z1)=} != {len(z2)=}"
    assert abs(z1[0] - z2[0]) < tol, f"First point mismatch: {z1[0]=} != {z2[0]=}"
    if len(z1) > 1:
        assert abs(z1[1] - z2[1]) < tol, f"Second point mismatch: {z1[1]=} != {z2[1]=}"
    if len(z1) > 2:
        assert abs(z1[2] - z2[2]) < tol, f"Third point mismatch: {z1[2]=} != {z2[2]=}"
    assert abs(z1[-1] - z2[-1]) < tol, f"Last point mismatch: {z1[-1]=} != {z2[-1]=}"

    return z1, w1, w2


def _integrate(integrand: NDArray, z: NDArray, method: INTEGRATION_METHODS) -> float:
    """Integrate the given integrand using the specified method."""
    if len(z) < 2:
        logger.debug("Less than two overlapping grid points between states, returning 0")
        return 0.0

    dz = z[1] - z[0]
    if method == "sum":
        value = np.sum(integrand) * dz
    elif method == "trapezoid":
        value = np.trapezoid(integrand, dx=dz)
    elif method == "scipy_trapezoid":
        value = scipy.integrate.trapezoid(integrand, dx=dz)
    elif method == "scipy_simpson":
        value = scipy.integrate.simpson(integrand, dx=dz)
    else:
        raise ValueError(f"Invalid integration method: {method}")

    return float(value)
