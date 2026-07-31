from __future__ import annotations

import logging
import math
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from scipy.special import exprel

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.angular_state import AngularState
from rydstate.angular.utils import is_angular_momentum_quantum_number, is_not_set, is_unknown
from rydstate.units import BaseQuantities, ureg

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.units import MatrixElementOperator, NDArray, PintArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    species: str
    """The species of the Rydberg state."""

    _coefficients: list[float]
    """The channel coefficients of the different RydbergKets that form the RydbergState stored as plain list."""
    rydberg_kets: list[RydbergKet]
    """The Rydberg kets that form the Rydberg state."""
    angular: AngularKetBase[Any] | AngularState[Any]
    """The angular part of the Rydberg state, i.e. the radial part is traced out."""

    nu: float
    """The effective principal quantum number nu.
    For SQDT states, this is also sometimes called n*.
    For MQDT nu is given in reference to the lowest ionization threshold.
    """
    _energy_au: float
    """The energy of the Rydberg state in atomic units (Hartree)."""

    n: int
    """The principal quantum number n of the Rydberg state.

    For MQDT states, we define the corresponding principal quantum number n via the number of nodes
    in the radial wavefunction of the most dominant channel.
    """

    def __init__(self) -> None:
        if abs(self.norm - 1) > 1e-10:
            raise ValueError(
                f"RydbergState initialized with non-normalized coefficients: {self._coefficients}, {self.rydberg_kets}"
            )

    def __iter__(self) -> Iterator[tuple[float, RydbergKet]]:
        return zip(self._coefficients, self.rydberg_kets, strict=True)

    def free_memory(self) -> None:
        """Release the cached radial and angular data to reduce memory usage.

        This drops the references to the (potentially large) radial wavefunctions of the rydberg kets.
        After calling this, matrix elements and overlaps can no longer be calculated for this state.
        """
        for rydberg_ket in self.rydberg_kets:
            rydberg_ket.__dict__.pop("radial", None)
            rydberg_ket.__dict__.pop("angular", None)

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return float(np.linalg.norm(self._coefficients))

    @property
    def nui(self) -> list[float]:
        """Return the effective principal quantum numbers nui of the different channels."""
        return [rydberg_ket.radial.nu for rydberg_ket in self.rydberg_kets]

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
        and `E_{ionization}` is the (reference) ionization energy of the species.
        """
        if unit == "a.u.":
            return self._energy_au
        energy: PintFloat = self._energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        ov = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                ov += coeff1 * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> float: ...

    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
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
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The reduced matrix element for the given operator.

        """
        if len(self.rydberg_kets) == 1 and len(other.rydberg_kets) == 1:
            # fast path for sqdt states
            me = self.rydberg_kets[0].calc_reduced_matrix_element(other.rydberg_kets[0], operator, unit=unit)
            return me * self._coefficients_conjugate[0] * other._coefficients[0]

        value = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                value += coeff1 * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, unit=unit)
        return value

    @overload
    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: str
    ) -> float: ...

    def calc_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, q: int, unit: str | None = None
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
            unit: The unit to which to convert the radial matrix element.
                Can be "a.u." for atomic units (so no conversion is done), or a specific unit.
                Default None will return a pint quantity.

        Returns:
            The matrix element for the given operator.

        """
        value = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.rydberg_kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.rydberg_kets, strict=True):
                value += coeff1 * coeff2 * ket1.calc_matrix_element(ket2, operator, q=q, unit=unit)
        return value

    def calc_exp_qn(self, qn: str) -> float:
        if is_angular_momentum_quantum_number(qn):
            return self.angular.calc_exp_qn(qn)
        if qn == "nu":
            return self.nu
        if qn == "n":
            return self.n
        if qn == "nui":
            return float(sum([abs(coeff) ** 2 * ket.radial.nu / self.norm**2 for coeff, ket in self]))
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_std_qn(self, qn: str) -> float:
        if is_angular_momentum_quantum_number(qn):
            return self.angular.calc_std_qn(qn)
        if qn in ("n", "nu"):
            return 0
        if qn == "nui":
            qns = np.array(self.nui)
            coefficients2 = np.conjugate(self.coefficients) * self.coefficients / self.norm**2
            exp_q = np.sum(coefficients2 * qns)
            exp_q2 = np.sum(coefficients2 * qns * qns)
            if abs(exp_q2 - exp_q**2) < 1e-10:
                return 0
            return math.sqrt(exp_q2 - exp_q**2)

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
            \Gamma^{spontaneous}_{self \to other} = \frac{4}{3} \frac{\alpha}{c^2} \omega^3
                |\langle self || r^k_{radial} \hat{d}_q || other \rangle|^2

        where :math:`\alpha = 1/c` in atomic units.

        and

        .. math::
            \Gamma^{blackbody}_{self \to other} = \Gamma^{spontaneous}_{self \to other} \frac{1}{\exp(\omega / T) - 1}

        """
        from rydstate.basis import BasisMQDT, BasisSQDT  # noqa: PLC0415
        from rydstate.rydberg_state import RydbergStateMQDT, RydbergStateSQDT  # noqa: PLC0415

        m = self.angular.m
        if is_not_set(m):
            raise RuntimeError("m quantum number must be defined to calculate transition rates.")

        basis: BasisMQDT | BasisSQDT[Any]
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
                coupling_scheme=self.angular.coupling_scheme,
                sqdt=self.sqdt,
                potential_class=type(self.potential),
            )
        elif isinstance(self, RydbergStateMQDT):
            assert isinstance(self.angular, AngularState)
            l_r_list = [
                angular_ket.l_r for angular_ket in self.angular.to("LS").kets if not is_unknown(angular_ket.l_r)
            ]
            basis = BasisMQDT(
                self.species,
                nu=(0, int(self.nu + 35)),
                l_r=(min(l_r_list) - 1, max(l_r_list) + 1),
                m=(m - 1, m + 1),
                mqdt=self.mqdt,
                potential_class=self.potential_class,
            )
        else:
            raise NotImplementedError(f"Transition rates are not implemented for {type(self)}.")

        if only_spontaneous:
            basis.filter_states("nu", (0, self.nu))

        relevant_states = basis.states
        energy_differences_au = self.get_energy("a.u.") - np.array([s.get_energy("a.u.") for s in relevant_states])
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
