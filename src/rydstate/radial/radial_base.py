from __future__ import annotations

import logging
import math
from numbers import Number
from typing import TYPE_CHECKING, overload

import numpy as np

from rydstate.radial.radial_matrix_element import (
    calc_radial_matrix_element,
    calc_radial_matrix_element_electric_dipole_closed_shell_core,
)
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.radial.radial_matrix_element import INTEGRATION_METHODS, RadialMatrixElementOperator
    from rydstate.species.element_properties import ElementProperties
    from rydstate.units import NDArray, PintFloat

logger = logging.getLogger(__name__)


class Radial:
    _z_list: NDArray
    _w_list: NDArray
    _is_dummy: bool = False
    element_properties: ElementProperties | None = None
    nu: float | None = None
    """The effective principal quantum number of the wavefunction, or None if it is not well defined."""

    def __init__(
        self,
        z_list: NDArray,
        w_list: NDArray,
        *,
        element_properties: ElementProperties | None = None,
        nu: float | None = None,
    ) -> None:
        self._z_list = np.asarray(z_list)
        self._w_list = np.asarray(w_list)
        self.element_properties = element_properties
        self.nu = nu

        if len(z_list) < 2:
            raise ValueError("z_list must have at least 2 elements")
        if z_list[0] < 0 or self.dz < 0:
            raise ValueError("z_list must be non-negative and increasing")
        if len(w_list) != len(z_list):
            raise ValueError("w_list must have the same length as z_list")
        if not np.all(np.abs(np.diff(self.z_list) - self.dz) < 1e-10):
            raise ValueError("z_list must be equidistantly spaced")
        if not abs(round(self.z_list[0] / self.dz) - (self.z_list[0] / self.dz)) < 1e-8:
            raise ValueError("z_list must start at an integer multiple of dz")
        if np.isnan(self.w_list).any():
            raise ValueError("w_list must not contain NaN values")

    @property
    def z_list(self) -> NDArray:
        r"""The grid in the scaled dimensionless coordinate :math:`z = \sqrt{r/a_0}`.

        In this coordinate the grid points are chosen equidistant,
        because the nodes of the wavefunction are equally spaced in this coordinate.
        """
        return self._z_list

    @property
    def x_list(self) -> NDArray:
        r"""The grid in the dimensionless coordinate :math:`x = r/a_0`."""
        return self.z_list**2

    @property
    def dz(self) -> float:
        r"""The grid step size in the scaled dimensionless coordinate :math:`z = \sqrt{r/a_0}`."""
        return float(self.z_list[1] - self.z_list[0])

    @property
    def steps(self) -> int:
        """The number of grid points."""
        return len(self.z_list)

    @property
    def w_list(self) -> NDArray:
        r"""The dimensionless scaled wavefunction :math:`w(z)`.

        The scaled wavefunction is defined as

        .. math::
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \sqrt{a_0} r R(r)

        """
        return self._w_list

    @property
    def u_list(self) -> NDArray:
        r"""The dimensionless wavefunction :math:`\tilde{u}(x) = \sqrt{a_0} r R(r)`."""
        return np.sqrt(self.z_list) * self.w_list

    @property
    def r_list(self) -> NDArray:
        r"""The radial wavefunction in atomic units :math:`\tilde{R}(r) = a_0^{-3/2} R(r)`."""
        return self.u_list / self.x_list

    @property
    def norm(self) -> float:
        """The norm of the wavefunction."""
        return math.sqrt(2 * np.sum(np.abs(self.w_list) * np.abs(self.w_list) * self.z_list * self.z_list) * self.dz)

    @property
    def nodes(self) -> int:
        """The number of nodes (i.e. zero-crossings) of the wavefunction."""
        w_list_no_zeros = self.w_list[self.w_list != 0]
        return int(np.sum(np.abs(np.diff(np.sign(w_list_no_zeros)))) // 2)

    def _align(self, other: Radial) -> tuple[NDArray, NDArray, NDArray]:
        """Zero-pad ``self`` and ``other`` onto a common grid covering both.

        Returns ``(z_common, w_self, w_other)`` where all three arrays have the same length.
        """
        dz = self.dz
        if abs(dz - other.dz) > 1e-10:
            raise ValueError(f"Cannot combine radial states with different dz: {dz} != {other.dz}")

        # integer offsets on the global grid [0, dz, 2*dz, ...]
        i0_self = round(self.z_list[0] / dz)
        i0_other = round(other.z_list[0] / dz)
        lo = min(i0_self, i0_other)
        hi = max(i0_self + len(self.z_list), i0_other + len(other.z_list))

        # rebuild the common grid the same way RadialKet does, to keep identical float values
        z_common = np.arange(0, hi * dz + dz / 2, dz)[lo:hi]

        w_self = np.zeros(hi - lo, dtype=self.w_list.dtype)
        w_self[i0_self - lo : i0_self - lo + len(self.w_list)] = self.w_list
        w_other = np.zeros(hi - lo, dtype=other.w_list.dtype)
        w_other[i0_other - lo : i0_other - lo + len(other.w_list)] = other.w_list

        return z_common, w_self, w_other

    def _common_nu(self, other: Radial) -> float | None:
        """Return the nu shared by ``self`` and ``other``, or None if they differ."""
        if self.nu is None or other.nu is None:
            return None
        if abs(self.nu - other.nu) > 1e-10:
            return None
        return self.nu

    def __add__(self, other: Radial) -> Radial:
        if not isinstance(other, Radial):
            return NotImplemented
        if self.element_properties != other.element_properties:
            raise ValueError("Cannot add Radial states with different element_properties")

        z_common, w_self, w_other = self._align(other)
        return Radial(z_common, w_self + w_other, element_properties=self.element_properties, nu=self._common_nu(other))

    def __sub__(self, other: Radial) -> Radial:
        return self.__add__(-other)

    def __mul__(self, scalar: float) -> Radial:
        if not isinstance(scalar, Number):
            return NotImplemented
        return Radial(self.z_list, scalar * self.w_list, element_properties=self.element_properties, nu=self.nu)

    def __rmul__(self, scalar: float) -> Radial:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Radial:
        return self.__mul__(1 / scalar)

    def __neg__(self) -> Radial:
        return self.__mul__(-1)

    def get_outer_sign(self) -> int:
        """Get the sign of the outermost non-zero value of the wavefunction."""
        if self._is_dummy:
            return int(np.sign(self._coeff))  # type: ignore [attr-defined]
        for w in self.w_list[::-1]:
            if w != 0:
                return int(np.sign(w))
        return 0

    def calc_overlap(self, other: Radial, *, integration_method: INTEGRATION_METHODS = "sum") -> float:
        r"""Calculate the overlap <self|other> of two radial kets.

        Args:
            other: Other radial ket
            integration_method: Integration method to use

        Returns:
            The overlap integral between self and other.

        """
        return self.calc_matrix_element(other, k_radial=0, unit="a.u.", integration_method=integration_method)

    @overload
    def calc_matrix_element(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        *,
        unit: None = None,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        unit: str,
        *,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> float: ...

    def calc_matrix_element(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        unit: str | None = None,
        *,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> PintFloat | float:
        r"""Calculate the radial matrix element <self | r^k_radial | other>.

        Computes the integral

        .. math::
            \int_{0}^{\infty} dr r^2 r^k_{radial} R_1(r) R_2(r)
            = a_0^{k_{radial}} \int_{0}^{\infty} dx x^k_{radial} \tilde{u}_1(x) \tilde{u}_2(x)
            = a_0^{k_{radial}} \int_{0}^{\infty} dz 2 z^{2 + 2k_{radial}} w_1(z) w_2(z)

        where R_1 and R_2 are the radial wavefunctions of self and other,
        and w(z) = z^{-1/2} \tilde{u}(z^2) = (r/_a_0)^{1/4} \sqrt{a_0} r R(r).

        Args:
            other: Other radial ket
            k_radial: Power of r in the matrix element, or a string indicating a special operator
                (e.g. "electric_dipole_closed_shell_core")
            unit: Unit of the returned matrix element, default None returns a Pint quantity.
            integration_method: Integration method to use

        Returns:
            The radial matrix element in the desired unit.

        """
        radial_matrix_element_au = self._calc_matrix_element_au(other, k_radial, integration_method=integration_method)

        if unit == "a.u.":
            return radial_matrix_element_au
        radial_matrix_element: PintFloat
        if k_radial == "electric_dipole_closed_shell_core":
            radial_matrix_element = radial_matrix_element_au * ureg.Quantity(1, "a0") ** 1
        else:
            radial_matrix_element = radial_matrix_element_au * ureg.Quantity(1, "a0") ** k_radial
        if unit is None:
            return radial_matrix_element
        return radial_matrix_element.to(unit).magnitude

    def _calc_matrix_element_au(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        *,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> float:
        if self._is_dummy or other._is_dummy:
            if self._is_dummy is not other._is_dummy:
                return 0.0
            if k_radial == 0 and self._common_nu(other) is not None:
                return self._coeff.conjugate() * other._coeff  # type: ignore [attr-defined,no-any-return]
            # if not k_radial == 0 or nu are not the same we cant compute the matrix element and simply return 0
            return 0.0

        if k_radial == "electric_dipole_closed_shell_core":
            if self.element_properties is None or self.element_properties != other.element_properties:
                raise ValueError(f"Cannot compute {k_radial} for Radial with different or None element_properties")
            if (
                self.element_properties.alpha_closed_shell_core == 0.0
                or self.element_properties.r_c_dipole_operator is None
            ):
                return 0.0
            return calc_radial_matrix_element_electric_dipole_closed_shell_core(
                self.z_list,
                self.w_list,
                other.z_list,
                other.w_list,
                alpha_closed_shell_core=self.element_properties.alpha_closed_shell_core,
                r_c=self.element_properties.r_c_dipole_operator,
                integration_method=integration_method,
            )

        if isinstance(k_radial, int):
            return calc_radial_matrix_element(
                self.z_list, self.w_list, other.z_list, other.w_list, k_radial, integration_method
            )
        raise ValueError(f"Invalid k_radial: {k_radial}")
