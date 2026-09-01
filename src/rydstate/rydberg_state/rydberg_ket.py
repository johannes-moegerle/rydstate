from __future__ import annotations

import logging
import math
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, Literal, overload

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import (
    format_quantum_number,
    get_spectroscopic_letter,
    is_angular_momentum_quantum_number,
    is_angular_operator_type,
    is_not_set,
    is_unknown,
)
from rydstate.species.element_properties import get_element_properties
from rydstate.species.sqdt import get_sqdt
from rydstate.units import MatrixElementOperatorRanks, ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import AngularMomentumQuantumNumbers, AngularOperatorType
    from rydstate.radial.radial_base import Radial
    from rydstate.rydberg_state.rydberg_sqdt import RydbergStateSQDT
    from rydstate.units import MatrixElementOperator, MatrixElementPart, PintFloat


logger = logging.getLogger(__name__)


ELECTRIC_MULTIPOLE_PREFACTORS: dict[int, float] = {
    k_angular: math.sqrt(4 * math.pi / (2 * k_angular + 1)) for _, k_angular in MatrixElementOperatorRanks.values()
}
SQRT_2 = math.sqrt(2)


class RydbergKet:
    """Create a Rydberg ket, i.e. a tensor product of a radial ket and an angular ket."""

    def __init__(
        self,
        species: str,
        angular: AngularKetBase[Any],
        radial: Radial,
        *,
        n: int | None = None,
    ) -> None:
        r"""Initialize the Rydberg state."""
        self.species = species
        self.element_properties = get_element_properties(species)
        self.angular = angular
        self.radial = radial
        self.n = n

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.species}, {self.radial!r}, {self.angular!r})"

    def __str__(self) -> str:
        return self.get_label()

    def get_label(self, fmt: Literal["raw", "ket", "bra"] = "ket") -> str:  # noqa: C901, PLR0912
        """Return a label of the Rydberg ket in the common spectroscopic notation.

        Args:
            fmt: The format of the label, i.e. whether to return the raw label, or the label in ket or bra notation.

        Returns:
            The label of the Rydberg ket.

        """
        angular = self.angular

        def qn(name: AngularMomentumQuantumNumbers) -> str:
            return format_quantum_number(angular.get_qn(name, allow_unknown=False))

        label = f"{self.species}:"

        nu = self.radial.nu
        nu_str = f"{nu:.1f}" if nu is not None else "?"
        n_str = f"{self.n}" if self.n is not None else f"[{nu_str}]"

        if angular.contains_unknown:
            # channels with unknown quantum numbers can only be identified by their label
            angular_str = angular.label if angular.label is not None else "angular=?"
            label += f"nu={nu_str},{angular_str}"

        elif self.element_properties.number_valence_electrons == 1:
            if not (angular.s_c == 0 and angular.l_c == 0):
                raise RuntimeError(
                    f"Alkali RydbergKet with unexpected quantum numbers: s_c={angular.s_c}, l_c={angular.l_c}"
                )
            l_str = get_spectroscopic_letter(angular.l_r).upper()
            label += f"{n_str}{l_str}_{qn('j_tot')}"

        # divalent atoms
        elif isinstance(angular, AngularKetLS):
            l_tot_str = get_spectroscopic_letter(angular.l_tot).upper()
            if angular.l_c == 0:
                label += f"S={qn('s_tot')},{n_str}{l_tot_str}_{qn('j_tot')}"
            else:
                n_c = self.angular.n_c
                l_str = get_spectroscopic_letter(angular.l_r)
                l_c_str = get_spectroscopic_letter(angular.l_c)
                label += f"S={qn('s_tot')},({n_c}{l_c_str},{n_str}{l_str}){l_tot_str}_{qn('j_tot')}"
        elif isinstance(angular, (AngularKetJJ, AngularKetFJ)):
            n_c = self.angular.n_c
            l_str = get_spectroscopic_letter(angular.l_r)
            l_c_str = get_spectroscopic_letter(angular.l_c)
            label += f"({n_c}{l_c_str}_{qn('j_c')},{n_str}{l_str}_{qn('j_r')})"
            if isinstance(angular, AngularKetJJ) or angular.i_c == 0:
                label += f",J={qn('j_tot')}"
            else:
                label += f",f_c={qn('f_c')}"
        else:
            raise NotImplementedError(f"get_label is not implemented for angular kets of type {type(angular)}.")

        if angular.i_c != 0:
            label += f",F={qn('f_tot')}"
        if not is_not_set(angular.m):
            label += f",m={format_quantum_number(angular.m)}"

        if fmt == "raw":
            return label
        if fmt == "ket":
            return f"|{label}⟩"
        if fmt == "bra":
            return f"⟨{label}|"
        raise ValueError(f"Unknown fmt {fmt}")

    def calc_reduced_overlap(self, other: RydbergKet) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        angular_overlap = self.angular.calc_reduced_overlap(other.angular)
        if angular_overlap == 0:
            return 0.0
        radial_overlap = self.radial.calc_overlap(other.radial)
        return radial_overlap * angular_overlap

    @overload
    def calc_reduced_matrix_element(
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> float: ...

    def calc_reduced_matrix_element(  # noqa: C901
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str | None = None,
    ) -> PintFloat | float:
        r"""Calculate the reduced matrix element.

        Calculate the reduced matrix element between self and other (ignoring m quantum numbers)

        .. math::
            \left\langle self || r^k_radial \hat{O}^{(k_{angular})} || other \right\rangle

        where \hat{O}^{(k_{angular})} is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        For the ``"electric_..."`` operators, the matrix element of "rydberg", "inner_valence" and "closed_shell_core"
        are calculated separately and added together
        (currently "closed_shell_core" is only supported for the "electric_dipole" operator).
        In addition, each term is multiplied by the symmetry factor sqrt(2),
        if exactly one of self and other has both its valence electrons in the same shell,
        see also :attr:`valence_electrons_are_in_the_same_shell`.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            part: The part of the matrix element to calculate.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The reduced matrix element for the given operator.

        """
        if operator == "magnetic_dipole":
            if part != "all":
                raise ValueError(f"Part {part} is currently not supported for magnetic dipole matrix elements.")
            matrix_element_au = self._calc_magnetic_reduced_matrix_element_au(other, operator)
        elif operator.startswith("electric_"):
            matrix_element_au = self._calc_electric_reduced_matrix_element_au(other, operator, part=part)
        elif is_angular_operator_type(operator):
            if part != "all":
                raise ValueError(f"Part {part} is not valid for angular operators.")
            matrix_element_au = self._calc_angular_reduced_matrix_element_au(other, operator)
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if unit == "a.u.":
            return matrix_element_au

        k_radial, _k_angular = _get_ks(operator)
        radial_unit: PintFloat = ureg.Quantity(1, "bohr_radius") ** k_radial
        matrix_element_unit: PintFloat
        if operator == "magnetic_dipole":
            matrix_element_unit = radial_unit * ureg.Quantity(2, "bohr_magneton")
        elif operator.startswith("electric_"):
            matrix_element_unit = radial_unit * ureg.Quantity(1, "e")
        elif is_angular_operator_type(operator):
            matrix_element_unit = ureg.Quantity(1, "dimensionless")
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if unit is None:
            return matrix_element_au * matrix_element_unit.to_base_units()  # type: ignore [no-any-return]
        return matrix_element_au * matrix_element_unit.to(unit).magnitude

    def _calc_electric_reduced_matrix_element_au(  # noqa: C901
        self, other: RydbergKet, operator: MatrixElementOperator, part: MatrixElementPart
    ) -> float:
        if part == "all":
            if self.radial._is_dummy or other.radial._is_dummy:  # noqa: SLF001
                return 0.0
            matrix_element = self._calc_electric_reduced_matrix_element_au(other, operator, part="rydberg")
            if (
                self.element_properties.number_valence_electrons == 2
                and self.core_state is not None
                and other.core_state is not None
            ):
                matrix_element += self._calc_electric_reduced_matrix_element_au(other, operator, part="inner_valence")
            if operator == "electric_dipole" and self.element_properties.alpha_closed_shell_core != 0:
                matrix_element += self._calc_electric_reduced_matrix_element_au(
                    other, operator, part="closed_shell_core"
                )
            return matrix_element

        k_radial, k_angular = _get_ks(operator)

        angular_operator: AngularOperatorType
        angular_operator = "spherical_inner_valence" if part == "inner_valence" else "spherical"
        # Electric multipole operator: p_{k,q} = e r^k_radial * sqrt(4pi / (2k+1)) * Y_{k_angular,q}(\theta, phi)
        angular_matrix_element = self.angular.calc_reduced_matrix_element(other.angular, angular_operator, k_angular)
        if angular_matrix_element == 0:
            return 0.0

        # Prefactor sqrt(4 pi / (2 k_angular + 1)) for the electric multipole operators, precomputed for performance
        prefactor = ELECTRIC_MULTIPOLE_PREFACTORS[k_angular]

        if part == "rydberg":
            radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial, unit="a.u.")
        elif part == "inner_valence":
            radial_matrix_element = self._calc_core_radial_matrix_element_au(other, k_radial)
            rydberg_radial_overlap = self.radial.calc_overlap(other.radial)
            prefactor *= rydberg_radial_overlap
        elif part == "closed_shell_core":
            if operator != "electric_dipole":
                raise NotImplementedError(f"Operator {operator} not implemented for closed shell core matrix elements.")
            radial_matrix_element = self.radial.calc_matrix_element(
                other.radial, "electric_dipole_closed_shell_core", unit="a.u."
            )
        else:
            raise ValueError(f"Operator part {part} not implemented for electric multipole matrix elements.")

        if self.valence_electrons_are_in_the_same_shell != other.valence_electrons_are_in_the_same_shell:  # xor
            # we always assume that the two valence electrons are distinguishable,
            # which for one electron in the Rydberg state, and the other close to the core, is fine.
            # However, if for the initial state (or final) state, both electrons are in the same shell
            # then the matrix element is enhanced by a factor of sqrt(2) due to the antisymmetrization.
            # <5s5s|d_1 + d_2|5snp> actually means
            # <5s(1)5s(2)|d_1 + d_2 (|5s(1)np(2)> + |np(1)5s(2)>)/sqrt(2) = sqrt(2) <5s|d|np>
            prefactor *= SQRT_2

        return prefactor * angular_matrix_element * radial_matrix_element

    def _calc_magnetic_reduced_matrix_element_au(self, other: RydbergKet, operator: MatrixElementOperator) -> float:
        k_radial, k_angular = _get_ks(operator)

        # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
        g_s = 2.0023192
        value_s_tot = self.angular.calc_reduced_matrix_element(other.angular, "s_tot", k_angular)
        g_l = 1
        value_l_tot = self.angular.calc_reduced_matrix_element(other.angular, "l_tot", k_angular)
        angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot
        prefactor = -0.5
        # Note: we use the convention, that the magnetic dipole moments are given
        # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s_tot))
        # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy

        if angular_matrix_element == 0:
            return 0.0

        radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial, unit="a.u.")
        return prefactor * angular_matrix_element * radial_matrix_element

    def _calc_angular_reduced_matrix_element_au(self, other: RydbergKet, operator: AngularOperatorType) -> float:
        k_radial, k_angular = _get_ks(operator)
        angular_matrix_element = self.angular.calc_reduced_matrix_element(other.angular, operator, k_angular)

        if angular_matrix_element == 0:
            return 0.0

        radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial, unit="a.u.")
        return angular_matrix_element * radial_matrix_element

    def _calc_core_radial_matrix_element_au(self, other: RydbergKet, k_radial: int) -> float:
        r"""Calculate the radial matrix element :math:`\langle self_c | r^{k_{radial}} | other_c \rangle` in a.u.

        The core electron is treated as the low-lying valence electron of the corresponding singly charged SQDT ion
        (e.g. Yb174_ion for Yb174):
        the Rydberg electron is ignored and the radial matrix element is calculated between the two core states,
        where the principal quantum number of each core electron is given by the lowest allowed shell of the ion
        for the given l_c.
        """
        if self.core_state is None or other.core_state is None:
            raise RuntimeError("Cannot calculate core radial matrix element: core state is not available.")

        return self.core_state.radial.calc_matrix_element(other.core_state.radial, k_radial, unit="a.u.")

    @cached_property
    def core_state(self) -> RydbergStateSQDT[Any] | None:
        """Get the corresponding ion state of the Rydberg ket."""
        from rydstate.rydberg_state.rydberg_sqdt import RydbergStateSQDT  # noqa: PLC0415

        s_c = self.angular.s_c
        l_c = self.angular.l_c
        j_c = self.angular.get_qn("j_c", allow_unknown=True)
        f_c = self.angular.get_qn("f_c", allow_unknown=True)
        n_c = self.angular.n_c
        if s_c == 0:
            return None
        if is_unknown(l_c) or is_unknown(j_c) or is_unknown(f_c) or is_unknown(n_c):
            return None
        if n_c is None:
            raise RuntimeError("The principal quantum number n_c of the core electron is not set in the angular ket.")

        ion_species = f"{self.species}_ion"
        try:
            core_sqdt = get_sqdt(ion_species)
        except ValueError:
            logger.warning(
                "No SQDT data available for the ion species of %s, "
                "thus we cannot calculate the ion state and its matrix elements.",
                self.species,
            )
            return None

        # The core electron of the neutral atom is the valence electron of the ion, with total angular momentum j_c.
        core_angular_ket = AngularKetLS(l_r=l_c, j_tot=j_c, f_tot=f_c, species=ion_species)
        return RydbergStateSQDT(ion_species, n_c, angular=core_angular_ket, sqdt=core_sqdt)

    @cached_property
    def valence_electrons_are_in_the_same_shell(self) -> bool:
        r"""Whether the Rydberg electron occupies the same shell as the inner valence electron."""
        if self.radial.nu > self.element_properties.ground_state_shell[0]:  # type: ignore [operator]
            return False
        if self.angular.l_r != self.angular.l_c:
            return False
        j_r = self.angular.get_qn("j_r", allow_unknown=True)
        j_c = self.angular.get_qn("j_c", allow_unknown=True)
        if is_unknown(j_r) or is_unknown(j_c) or j_r != j_c:
            return False
        if self.core_state is None:
            return False
        overlap = abs(self.radial.calc_overlap(self.core_state.radial))
        if overlap < 0.5:
            return False
        if overlap < 0.8:
            logger.warning(
                "Overlap between Rydberg electron and inner valence electron is %.2f (between 0.5 and 0.8). "
                "Assuming they are in the same shell.",
                overlap,
            )
        return True

    @overload
    def calc_matrix_element(
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        q: int,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        q: int,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> float: ...

    def calc_matrix_element(
        self,
        other: RydbergKet,
        operator: MatrixElementOperator,
        q: int,
        *,
        part: MatrixElementPart = "all",
        unit: str | None = None,
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the full matrix element between self and other,
        also considering the magnetic quantum numbers m of self and other.

        .. math::
            \left\langle self | r^k_radial \hat{O}^{(k_{angular})}_q | other \right\rangle

        where \hat{O}^{(k_{angular})}_q is the operator of rank k_angular for which to calculate the matrix element.
        k_radial and k_angular are determined from the operator automatically.

        Args:
            other: The other Rydberg state for which to calculate the matrix element.
            operator: The operator for which to calculate the matrix element.
            q: The component of the operator.
            part: The part of the matrix element to calculate.
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        _k_radial, k_angular = _get_ks(operator)
        prefactor = self.angular._calc_wigner_eckart_prefactor(other.angular, k_angular, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, part=part, unit=unit)
        return prefactor * reduced_matrix_element


@cache
def _get_ks(operator: MatrixElementOperator) -> tuple[int, int]:
    """Get the k_radial and k_angular for the given operator."""
    if operator in MatrixElementOperatorRanks:
        return MatrixElementOperatorRanks[operator]
    if is_angular_operator_type(operator):
        k_radial = 0
        if operator.startswith("identity_"):
            return k_radial, 0
        if is_angular_momentum_quantum_number(operator):
            return k_radial, 1
    raise ValueError(f"Operator {operator} not supported.")
