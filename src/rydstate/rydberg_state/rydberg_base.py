from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from scipy.special import exprel

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.angular_state import AngularState
from rydstate.angular.utils import (
    get_coupling_scheme_for_quantum_number,
    is_angular_momentum_quantum_number,
    is_not_set,
    is_unknown,
)
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.species.element_properties import get_element_properties
from rydstate.units import BaseQuantities, ureg

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

    from rydstate.angular.utils import CouplingScheme
    from rydstate.radial import Radial
    from rydstate.units import MatrixElementOperator, MatrixElementPart, NDArray, PintArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergState:
    species: str
    """The species of the Rydberg state."""

    _coefficients: list[float]
    """The channel coefficients of the different RydbergKets that form the RydbergState stored as plain list."""
    rydberg_kets: list[RydbergKet]
    """The Rydberg kets that form the Rydberg state."""

    nu: float
    """The effective principal quantum number nu given in reference to the reference ionization threshold."""
    n: int
    """The principal quantum number n of the Rydberg state.

    For MQDT states, we define the corresponding principal quantum number n via the number of nodes
    in the radial wavefunction of the most dominant channel.
    """
    f_tot: float
    """The total angular momentum quantum number f_tot of the Rydberg state."""
    _energy_au: float
    """The energy of the Rydberg state in atomic units (Hartree)."""

    def __init__(
        self,
        species: str,
        coefficients: Sequence[float] | NDArray,
        rydberg_kets: Sequence[RydbergKet],
        nu: float,
        energy_au: float,
    ) -> None:
        self.species = species
        self.element_properties = get_element_properties(species)

        self._coefficients = np.asarray(coefficients).tolist()
        self.rydberg_kets = list(rydberg_kets)
        self.nu = float(nu)
        self._energy_au = float(energy_au)

        if len(rydberg_kets) == 0:
            raise ValueError("RydbergState must be initialized with at least one state.")
        if len(coefficients) != len(rydberg_kets):
            raise ValueError("Length of coefficients and rydberg_kets must be the same.")
        angular_kets = [rydberg_ket.angular for rydberg_ket in rydberg_kets]
        if len(set(angular_kets)) != len(angular_kets):
            raise ValueError("RydbergState initialized with duplicate angular kets.")

        if abs(self.norm - 1) > 1e-10:
            raise ValueError(
                "RydbergState initialized with non-normalized coefficients: "
                f"{self.norm=} {self.coefficients=}, {self.rydberg_kets=}"
            )
        self._coefficients = (self.coefficients / self.norm).tolist()

        self.f_tot = rydberg_kets[0].angular.f_tot
        if not all(rydberg_ket.angular.f_tot == self.f_tot for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have the same f_tot.")
        self.parity = rydberg_kets[0].angular.parity
        if not all(rydberg_ket.angular.parity == self.parity for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have the same parity.")
        self.m = rydberg_kets[0].angular.m
        if not all(rydberg_ket.angular.m == self.m for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have the same m.")
        self.coupling_scheme = rydberg_kets[0].angular.coupling_scheme
        if not all(rydberg_ket.angular.coupling_scheme == self.coupling_scheme for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have the same coupling_scheme.")

    def __repr__(self) -> str:
        terms = [f"{coeff}*{rydberg_ket!r}" for coeff, rydberg_ket in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        return self.get_label()

    def __iter__(self) -> Iterator[tuple[float, RydbergKet]]:
        return zip(self._coefficients, self.rydberg_kets, strict=True)

    def get_label(self, fmt: Literal["raw", "ket", "bra"] = "ket") -> str:
        """Label representing the state.

        Args:
            fmt: The format of the label, i.e. whether to return the raw label, or the label in ket or bra notation.

        Returns:
            The label of the state in the given format.

        """
        if len(self.rydberg_kets) == 1:
            sign = "" if self._coefficients[0] >= 0 else "-"
            return sign + self.rydberg_kets[0].get_label(fmt=fmt)
        label = " + ".join(f"{c:.2g}*{k.get_label(fmt=fmt)}" for c, k in self)
        return label.replace("+ -", "- ")

    def to_coupling_scheme(self, coupling_scheme: CouplingScheme) -> RydbergState:
        """Convert the Rydberg state to a different coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to which to convert the Rydberg state.

        Returns:
            The Rydberg state in the new coupling scheme.

        """
        if self.coupling_scheme == coupling_scheme:
            return RydbergState(
                self.species, self.coefficients, self.rydberg_kets, nu=self.nu, energy_au=self._energy_au
            )

        angular_ket: AngularKetBase[Any]
        angular_radial_wf: dict[AngularKetBase[Any], Radial] = {}
        for coeff_r, rydberg_ket in self:
            angular_state = rydberg_ket.angular.to_state(coupling_scheme)
            for coeff_a, angular_ket in angular_state:
                if angular_ket in angular_radial_wf:
                    angular_radial_wf[angular_ket] += (coeff_r * coeff_a) * rydberg_ket.radial
                else:
                    angular_radial_wf[angular_ket] = (coeff_r * coeff_a) * rydberg_ket.radial

        rydberg_kets: list[RydbergKet] = []
        coefficients: list[float] = []
        for angular_ket, radial_wf in angular_radial_wf.items():
            norm = radial_wf.norm
            if norm < 1e-12:
                # if channels cancel almost completely, we ignore them to avoid numerical issues
                continue
            outer_sign = radial_wf.get_outer_sign()
            rydberg_ket = RydbergKet(self.species, angular_ket, radial_wf / (outer_sign * norm))
            rydberg_kets.append(rydberg_ket)
            coefficients.append(outer_sign * norm)

        return RydbergState(self.species, coefficients, rydberg_kets, nu=self.nu, energy_au=self._energy_au)

    def _free_memory(self) -> None:
        """Release the cached radial, angular and core state data to reduce memory usage.

        This drops the references to the (potentially large) radial wavefunctions of the rydberg kets
        (including the radial wavefunctions of the cached core states).
        After calling this, matrix elements, overlaps and expectation values can no longer be calculated for this state.
        """
        for rydberg_ket in self.rydberg_kets:
            rydberg_ket.__dict__.pop("radial", None)
            rydberg_ket.__dict__.pop("angular", None)
            rydberg_ket.__dict__.pop("core_state", None)
        self.__dict__.pop("rydberg_kets", None)

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return float(np.linalg.norm(self._coefficients))

    @cached_property
    def coefficients(self) -> NDArray:
        """Return the channel coefficients as numpy array."""
        return np.array(self._coefficients)

    @cached_property
    def _coefficients_conjugate(self) -> list[float]:
        """Return the cached conjugate of the coefficients as a plain python list."""
        return np.conjugate(self._coefficients).tolist()  # type: ignore [no-any-return]

    @overload
    def get_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the Rydberg state.

        The energy is defined as

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2} + E_{ionization}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number,
        and `E_{ionization}` is the reference ionization threshold of the species.
        """
        if unit == "a.u.":
            return self._energy_au
        energy: PintFloat = self._energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_reduced_overlap(self, other: RydbergState) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        ov = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                ov += coeff1 * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    @overload
    def calc_reduced_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> float: ...

    def calc_reduced_matrix_element(
        self,
        other: RydbergState,
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
        if len(self.rydberg_kets) == 1 and len(other.rydberg_kets) == 1:
            # fast path for sqdt states
            me = self.rydberg_kets[0].calc_reduced_matrix_element(other.rydberg_kets[0], operator, part=part, unit=unit)
            return me * self._coefficients_conjugate[0] * other._coefficients[0]

        value = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                value += coeff1 * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, part=part, unit=unit)
        return value

    @overload
    def calc_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        q: int,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        q: int,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> float: ...

    def calc_matrix_element(
        self,
        other: RydbergState,
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
        value = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                value += coeff1 * coeff2 * ket1.calc_matrix_element(ket2, operator, q=q, part=part, unit=unit)
        return value

    def calc_exp_qn(self, qn: str) -> float:
        if qn == "nu":
            return self.nu
        if qn == "parity":
            return self.parity
        if qn == "n":
            return self.n

        if is_angular_momentum_quantum_number(qn):
            if qn not in self.rydberg_kets[0].angular.quantum_number_names:
                coupling_scheme = get_coupling_scheme_for_quantum_number(qn, [self.coupling_scheme])
                return self.to_coupling_scheme(coupling_scheme).calc_exp_qn(qn)
            angular_state = AngularState(self._coefficients, [ket.angular for ket in self.rydberg_kets])
            return angular_state.calc_exp_qn(qn)

        raise ValueError(f"Unknown quantum number {qn}")

    def calc_std_qn(self, qn: str) -> float:
        if qn in ("nu", "parity"):
            return 0

        if is_angular_momentum_quantum_number(qn):
            if qn not in self.rydberg_kets[0].angular.quantum_number_names:
                coupling_scheme = get_coupling_scheme_for_quantum_number(qn, [self.coupling_scheme])
                return self.to_coupling_scheme(coupling_scheme).calc_std_qn(qn)
            angular_state = AngularState(self._coefficients, [ket.angular for ket in self.rydberg_kets])
            return angular_state.calc_std_qn(qn)

        raise ValueError(f"Unknown quantum number {qn}")

    @overload
    def get_spontaneous_transition_rates(self: Self, unit: None = None) -> tuple[list[Self], PintArray]: ...

    @overload
    def get_spontaneous_transition_rates(self: Self, unit: str) -> tuple[list[Self], NDArray]: ...

    def get_spontaneous_transition_rates(self: Self, unit: str | None = None) -> tuple[list[Self], NDArray | PintArray]:
        """Calculate the spontaneous transition rates for the Rydberg state.

        The spontaneous transition rates are given by the Einstein A coefficients.

        Args:
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The relevant states and the transition rates.

        """
        relevant_states_masked, transition_rates_au = self._get_transition_rates_au(only_spontaneous=True)

        if unit == "a.u.":
            return relevant_states_masked, transition_rates_au
        transition_rates = ureg.Quantity(transition_rates_au, "1/atomic_unit_of_time")
        if unit is None:
            return relevant_states_masked, transition_rates
        return relevant_states_masked, transition_rates.to(unit).magnitude

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: None = None
    ) -> tuple[list[Self], PintArray]: ...

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: PintFloat, *, unit: str
    ) -> tuple[list[Self], NDArray]: ...

    @overload
    def get_black_body_transition_rates(
        self: Self, temperature: float, temperature_unit: str, unit: str
    ) -> tuple[list[Self], NDArray]: ...

    def get_black_body_transition_rates(
        self: Self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: str | None = None
    ) -> tuple[list[Self], NDArray | PintArray]:
        """Calculate the black body transition rates of the Rydberg state.

        The black body transition rates are given by the Einstein B coefficients,
        with a weight factor given by Planck's law.

        Args:
            temperature: The temperature, for which to calculate the black body transition rates.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as `pint.Quantity`.
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The relevant states and the transition rates.

        """
        temperature_au = ureg.Quantity(temperature, temperature_unit).to_base_units().magnitude
        relevant_states_masked, transition_rates_au = self._get_transition_rates_au(
            temperature_au, only_spontaneous=False
        )

        if unit == "a.u.":
            return relevant_states_masked, transition_rates_au
        transition_rates = ureg.Quantity(transition_rates_au, "1/atomic_unit_of_time")
        if unit is None:
            return relevant_states_masked, transition_rates
        return relevant_states_masked, transition_rates.to(unit).magnitude

    def _get_transition_rates_au(
        self: Self,
        temperature_au: float | None = None,
        *,
        only_spontaneous: bool = False,
    ) -> tuple[list[Self], NDArray]:
        r"""Calculate the transition rates in atomic units.

        The transition rates are given by the Einstein A coefficients for spontaneous transitions,
        and by the Einstein B coefficients with a weight factor given by Planck's law for black body transitions.

        Concretely the transition rates are calculated as

        .. math::
            \Gamma^{spontaneous}_{self \to other} = \frac{4}{3} \frac{\alpha}{c^2} |\omega|^3
                |\langle self || r^k_{radial} \hat{d}_q || other \rangle|^2

        where :math:`\alpha = 1/c` in atomic units and :math:`\omega = E_{self} - E_{other}` is the transition frequency
        (spontaneous transitions only occur for :math:`\omega > 0`).

        and

        .. math::
            \Gamma^{blackbody}_{self \to other} = \Gamma^{spontaneous}_{self \to other} \frac{1}{\exp(|\omega| / T) - 1}

        """
        from rydstate.basis import BasisMQDT, BasisSQDT  # noqa: PLC0415
        from rydstate.rydberg_state import RydbergStateMQDT, RydbergStateSQDT  # noqa: PLC0415

        m = self.m
        if is_not_set(m):
            raise RuntimeError("m quantum number must be defined to calculate transition rates.")

        basis: BasisMQDT | BasisSQDT
        if isinstance(self, RydbergStateSQDT):
            assert isinstance(self.angular, AngularKetBase)
            if self.angular.coupling_scheme != "LS":
                raise NotImplementedError("Transition rates are currently only implemented for LS coupled states.")
            l_r = int(self.angular.get_qn("l_r"))
            basis = BasisSQDT(
                self.species,
                n=(1, int(self.n + 35)),
                l_r=(l_r - 1, l_r + 1),
                m=(m - 1, m + 1),
                potential_class=self.potential_class,
                sqdt=self.sqdt,
            )
        elif isinstance(self, RydbergStateMQDT):
            l_r_list = [
                ket.angular.l_r for ket in self.to_coupling_scheme("LS").rydberg_kets if not is_unknown(ket.angular.l_r)
            ]
            basis = BasisMQDT(
                self.species,
                nu=(0, int(self.nu + 35)),
                l_r=(min(l_r_list) - 1, max(l_r_list) + 1),
                m=(m - 1, m + 1),
                potential_class=self.potential_class,
                mqdt=self.mqdt,
            )
        else:
            raise NotImplementedError(f"Transition rates are not implemented for {type(self)}.")

        if only_spontaneous:
            basis.filter_states("nu", (0, self.nu))

        relevant_states = basis.states
        energy_differences_au = np.abs(
            self.get_energy("a.u.") - np.array([s.get_energy("a.u.") for s in relevant_states])
        )
        electric_dipole_moments_au = np.zeros(len(relevant_states))
        for q in [-1, 0, 1]:
            el_di_m = np.array(
                [s.calc_matrix_element(self, "electric_dipole", q, unit="a.u.") for s in relevant_states]
            )
            electric_dipole_moments_au += np.abs(el_di_m) ** 2

        transition_rates_au = (
            (4 / 3) * electric_dipole_moments_au / ureg.Quantity(1, "speed_of_light").to_base_units().magnitude ** 3
        )

        if only_spontaneous:
            transition_rates_au *= energy_differences_au**3
        else:
            assert temperature_au is not None, "Temperature must be given for black body transitions."
            if temperature_au == 0:
                transition_rates_au *= 0
            else:  # for numerical stability we use 1 / exprel(x) = x / (exp(x) - 1)
                transition_rates_au *= (
                    energy_differences_au**2 * temperature_au / exprel(energy_differences_au / temperature_au)
                )

        mask = transition_rates_au != 0
        relevant_states_masked = [ket for ket, is_relevant in zip(relevant_states, mask, strict=True) if is_relevant]
        transition_rates_au = transition_rates_au[mask]

        if np.any(transition_rates_au < 0):
            raise RuntimeError("Got negative transition rates, which should not happen.")

        return relevant_states_masked, transition_rates_au  # type: ignore [return-value]

    @overload
    def get_lifetime(
        self,
        temperature: float | PintFloat | None = None,
        temperature_unit: str | None = None,
        unit: None = None,
    ) -> PintFloat: ...

    @overload
    def get_lifetime(self, *, unit: str) -> float: ...

    @overload
    def get_lifetime(self, temperature: PintFloat, *, unit: str) -> float: ...

    @overload
    def get_lifetime(self, temperature: float, temperature_unit: str, unit: str) -> float: ...

    def get_lifetime(
        self,
        temperature: float | PintFloat | None = None,
        temperature_unit: str | None = None,
        unit: str | None = None,
    ) -> float | PintFloat:
        """Calculate the lifetime of the Rydberg state.

        The lifetime is the inverse of the sum of all transition rates.

        Args:
            temperature: The temperature, for which to calculate the black body transition rates.
                Default None will not include black body transitions.
            temperature_unit: The unit of the temperature.
                Default None will assume the temperature is given as `pint.Quantity`.
            unit: The unit to which to convert the result.
                Default None will return a `pint.Quantity`.

        Returns:
            The lifetime of the state.

        """
        _, transition_rates = self.get_spontaneous_transition_rates()
        transition_rates_au = transition_rates.to_base_units().magnitude
        if temperature is not None:
            _, black_body_transition_rates = self.get_black_body_transition_rates(temperature, temperature_unit)
            transition_rates_au = np.append(transition_rates_au, black_body_transition_rates.to_base_units().magnitude)

        lifetime_au: float = 1 / np.sum(transition_rates_au)

        if unit == "a.u.":
            return lifetime_au
        lifetime = ureg.Quantity(lifetime_au, "atomic_unit_of_time")
        if unit is None:
            return lifetime
        return lifetime.to(unit).magnitude
