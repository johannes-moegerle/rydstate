from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy.special import exprel

from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set, quantum_numbers_to_angular_ket
from rydstate.radial import RadialKet
from rydstate.rydberg.rydberg_base import RydbergStateBase
from rydstate.species import SpeciesObjectSQDT
from rydstate.species.utils import calc_energy_from_nu
from rydstate.units import BaseQuantities, MatrixElementOperatorRanks, ureg

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetJJ, AngularKetLS
    from rydstate.units import MatrixElementOperator, NDArray, PintArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateSQDT(RydbergStateBase):
    species: SpeciesObjectSQDT
    """The atomic species of the Rydberg state."""

    angular: AngularKetBase
    """The angular/spin part of the Rydberg electron."""

    def __init__(
        self,
        species: str | SpeciesObjectSQDT,
        n: int | None = None,
        nu: float | None = None,
        s_c: float | None = None,
        l_c: int = 0,
        j_c: float | None = None,
        f_c: float | None = None,
        s_r: float = 0.5,
        l_r: int | None = None,
        j_r: float | None = None,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            nu: Effective principal quantum number of the rydberg electron.
              Optional, if not given it will be calculated from n, l, j_tot, s_tot.
            s_c: Spin quantum number of the core electron (0 for Alkali, 0.5 for divalent atoms).
            l_c: Orbital angular momentum quantum number of the core electron.
            j_c: Total angular momentum quantum number of the core electron.
            f_c: Total angular momentum quantum number of the core (core electron + nucleus).
            s_r: Spin quantum number of the rydberg electron always 0.5)
            l_r: Orbital angular momentum quantum number of the rydberg electron.
            j_r: Total angular momentum quantum number of the rydberg electron.
            s_tot: Total spin quantum number of all electrons.
            l_tot: Total orbital angular momentum quantum number of all electrons.
            j_tot: Total angular momentum quantum number of all electrons.
            f_tot: Total angular momentum quantum number of the atom (rydberg electron + core)
            m: Total magnetic quantum number.
                Optional, only needed for concrete angular matrix elements.

        """
        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        self.species = species

        self.angular = quantum_numbers_to_angular_ket(
            species=self.species,
            s_c=s_c,
            l_c=l_c,
            j_c=j_c,
            f_c=f_c,
            s_r=s_r,
            l_r=l_r,
            j_r=j_r,
            s_tot=s_tot,
            l_tot=l_tot,
            j_tot=j_tot,
            f_tot=f_tot,
            m=m,  # type: ignore [arg-type]
        )

        self.n = n
        self._nu = nu
        if nu is None and n is None:
            raise ValueError("Either n or nu must be given to initialize the Rydberg state.")

        self._set_qn_as_attributes()

    def _set_qn_as_attributes(self) -> None:
        pass

    @classmethod
    def from_angular_ket(
        cls: type[Self],
        species: str | SpeciesObjectSQDT,
        angular_ket: AngularKetBase,
        n: int | None = None,
        nu: float | None = None,
    ) -> Self:
        """Initialize the Rydberg state from an angular ket."""
        obj = cls.__new__(cls)

        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        obj.species = species

        obj.n = n
        obj._nu = nu  # noqa: SLF001
        if nu is None and n is None:
            raise ValueError("Either n or nu must be given to initialize the Rydberg state.")

        obj.angular = angular_ket
        obj._set_qn_as_attributes()  # noqa: SLF001

        return obj

    def __repr__(self) -> str:
        species, n, nu = self.species.name, self.n, self.nu
        n_str = f", {n=}" if n is not None else ""
        return f"{self.__class__.__name__}({species}{n_str}, {nu=}, {self.angular})"

    def __str__(self) -> str:
        return self.__repr__()

    @cached_property
    def radial(self) -> RadialKet:
        """The radial part of the Rydberg electron."""
        if "l_r" not in self.angular.quantum_number_names:
            raise ValueError(
                f"l_r must be defined in the angular ket to access the radial ket, but angular={self.angular}."
            )

        radial_ket = RadialKet(self.species, nu=self.nu, l_r=self.angular.l_r)
        if self.n is not None:
            radial_ket.set_n_for_sanity_check(self.n)
            if isinstance(self.species, SpeciesObjectSQDT):
                s_tot_list = [self.angular.get_qn("s_tot")] if "s_tot" in self.angular.quantum_number_names else [0, 1]
                for s_tot in s_tot_list:
                    if not self.species.is_allowed_shell(self.n, self.angular.l_r, s_tot=s_tot):
                        raise ValueError(
                            f"The shell (n={self.n}, l_r={self.angular.l_r}, s_tot={s_tot}) "
                            f"is not allowed for the species {self.species}."
                        )
        return radial_ket

    @cached_property
    def nu(self) -> float:
        """The effective principal quantum number nu (for alkali atoms also known as n*) for the Rydberg state."""
        if self._nu is not None:
            return self._nu
        assert isinstance(self.species, SpeciesObjectSQDT), "nu must be given if not sqdt"
        assert self.n is not None, "either nu or n must be given"
        return self.species.calc_nu(self.n, self.angular)

    @property
    def nu_ref(self) -> float:
        return self.nu

    @overload
    def get_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_energy(self, unit: str) -> float: ...

    def get_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the Rydberg state.

        The energy is defined as

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number.
        """
        energy_au = calc_energy_from_nu(self.species.reduced_mass_au, self.nu)
        if unit == "a.u.":
            return energy_au
        energy: PintFloat = energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        if not isinstance(other, RydbergStateSQDT):
            raise NotImplementedError("Reduced overlap only implemented between RydbergStateSQDT states.")

        radial_overlap = self.radial.calc_overlap(other.radial)
        angular_overlap = self.angular.calc_reduced_overlap(other.angular)
        return radial_overlap * angular_overlap

    @overload  # type: ignore [override]
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
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
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
        if not isinstance(other, RydbergStateSQDT):
            raise NotImplementedError("Reduced matrix element only implemented between RydbergStateSQDT states.")

        if operator not in MatrixElementOperatorRanks:
            raise ValueError(
                f"Operator {operator} not supported, must be one of {list(MatrixElementOperatorRanks.keys())}."
            )

        k_radial, k_angular = MatrixElementOperatorRanks[operator]
        radial_matrix_element = self.radial.calc_matrix_element(other.radial, k_radial)

        matrix_element: PintFloat
        if operator == "magnetic_dipole":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = self.angular.calc_reduced_matrix_element(other.angular, "s_tot", k_angular)
            g_l = 1
            value_l_tot = self.angular.calc_reduced_matrix_element(other.angular, "l_tot", k_angular)
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot

            matrix_element = -ureg.Quantity(1, "bohr_magneton") * radial_matrix_element * angular_matrix_element
            # Note: we use the convention, that the magnetic dipole moments are given
            # as the same dimensionality as the Bohr magneton (mu = - mu_B (g_l l + g_s s_tot))
            # such that - mu * B (where the magnetic field B is given in dimension Tesla) is an energy

        elif operator in ["electric_dipole", "electric_quadrupole", "electric_octupole", "electric_quadrupole_zero"]:
            # Electric multipole operator: p_{k,q} = e r^k_radial * sqrt(4pi / (2k+1)) * Y_{k_angular,q}(\theta, phi)
            angular_matrix_element = self.angular.calc_reduced_matrix_element(other.angular, "spherical", k_angular)
            matrix_element = (
                ureg.Quantity(1, "e")
                * math.sqrt(4 * np.pi / (2 * k_angular + 1))
                * radial_matrix_element
                * angular_matrix_element
            )

        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if unit == "a.u.":
            return matrix_element.to_base_units().magnitude
        if unit is None:
            return matrix_element
        return matrix_element.to(unit).magnitude

    @overload
    def calc_matrix_element(self, other: RydbergStateSQDT, operator: MatrixElementOperator, q: int) -> PintFloat: ...

    @overload
    def calc_matrix_element(
        self, other: RydbergStateSQDT, operator: MatrixElementOperator, q: int, unit: str
    ) -> float: ...

    def calc_matrix_element(
        self, other: RydbergStateSQDT, operator: MatrixElementOperator, q: int, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the matrix element.

        Calculate the full matrix element between self and other,
        also considering the magnetic quantum numbers m of self and other.

        .. math::
            \left\langle self || r^k_radial \hat{O}_{k_angular} || other \right\rangle

        where \hat{O}_{k_angular} is the operator of rank k_angular for which to calculate the matrix element.
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
        _k_radial, k_angular = MatrixElementOperatorRanks[operator]
        prefactor = self.angular._calc_wigner_eckart_prefactor(other.angular, k_angular, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, unit)
        return prefactor * reduced_matrix_element

    @overload
    def get_spontaneous_transition_rates(self, unit: None = None) -> tuple[list[RydbergStateSQDT], PintArray]: ...

    @overload
    def get_spontaneous_transition_rates(self, unit: str) -> tuple[list[RydbergStateSQDT], NDArray]: ...

    def get_spontaneous_transition_rates(
        self, unit: str | None = None
    ) -> tuple[list[RydbergStateSQDT], NDArray | PintArray]:
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
        self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: None = None
    ) -> tuple[list[RydbergStateSQDT], PintArray]: ...

    @overload
    def get_black_body_transition_rates(
        self, temperature: PintFloat, *, unit: str
    ) -> tuple[list[RydbergStateSQDT], NDArray]: ...

    @overload
    def get_black_body_transition_rates(
        self, temperature: float, temperature_unit: str, unit: str
    ) -> tuple[list[RydbergStateSQDT], NDArray]: ...

    def get_black_body_transition_rates(
        self, temperature: float | PintFloat, temperature_unit: str | None = None, unit: str | None = None
    ) -> tuple[list[RydbergStateSQDT], NDArray | PintArray]:
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
        self,
        temperature_au: float | None = None,
        *,
        only_spontaneous: bool = False,
    ) -> tuple[list[RydbergStateSQDT], NDArray]:
        r"""Calculate the transition rates in atomic units.

        The transition rates are given by the Einstein A coefficients for spontaneous transitions,
        and by the Einstein B coefficients with a weight factor given by Planck's law for black body transitions.

        Concretely the transition rates are calculated as

        .. math::
            \Gamma^{spontaneous}_{self \to other} = \frac{4}{3} \frac{\alpha}{c^2} \omega^3
                |\langle self || r^k_radial \hat{O}_{k_angular} || other \rangle|^2

        and

        .. math::
            \Gamma^{blackbody}_{self \to other} = \Gamma^{spontaneous}_{self \to other} \frac{1}{\exp(\omega / T) - 1}

        """
        if self.species.number_valence_electrons == 2 and self.angular.coupling_scheme != "LS":
            raise NotImplementedError(
                "For alkaline earth atoms transition rates are only implemented for LS coupling scheme."
            )
        from rydstate.basis import BasisSQDTAlkali, BasisSQDTAlkalineLS  # noqa: PLC0415

        basis_class = BasisSQDTAlkali if self.species.number_valence_electrons == 1 else BasisSQDTAlkalineLS

        m = self.angular.m
        if is_not_set(m):
            raise RuntimeError("m quantum number must be defined to calculate transition rates.")

        basis = basis_class(self.species, n=(1, int(self.nu + 35)), m=(m - 1, m + 1))
        basis.filter_states("l_r", (self.angular.l_r - 1, self.angular.l_r + 1))

        if only_spontaneous:
            basis.filter_states("nu", (0, self.nu))

        relevant_states = basis.states
        energy_differences_au = self.get_energy("hartree") - np.array(
            [s.get_energy("hartree") for s in relevant_states]
        )
        electric_dipole_moments_au = np.zeros(len(relevant_states))
        for q in [-1, 0, 1]:
            # the different entries are only at most once nonzero -> we can just add the arrays
            el_di_m = np.array(
                [s.calc_matrix_element(self, "electric_dipole", q, unit="a.u.") for s in relevant_states]
            )
            electric_dipole_moments_au += el_di_m

        transition_rates_au = (
            (4 / 3)
            * np.abs(electric_dipole_moments_au) ** 2
            * energy_differences_au**2
            / ureg.Quantity(1, "speed_of_light").to_base_units().magnitude ** 3
        )

        if only_spontaneous:
            transition_rates_au *= energy_differences_au
        else:
            assert temperature_au is not None, "Temperature must be given for black body transitions."
            if temperature_au == 0:
                transition_rates_au *= 0
            else:  # for numerical stability we use 1 / exprel(x) = x / (exp(x) - 1)
                transition_rates_au *= temperature_au / exprel(energy_differences_au / temperature_au)

        mask = transition_rates_au != 0
        relevant_states_masked = [ket for ket, is_relevant in zip(relevant_states, mask, strict=True) if is_relevant]
        transition_rates_au = transition_rates_au[mask]

        if np.any(transition_rates_au < 0):
            raise RuntimeError("Got negative transition rates, which should not happen.")

        return relevant_states_masked, transition_rates_au

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


class RydbergStateSQDTAlkali(RydbergStateSQDT):
    """Create an Alkali Rydberg state, including the radial and angular states."""

    angular: AngularKetLS

    def __init__(
        self,
        species: str | SpeciesObjectSQDT,
        n: int,
        l: int,
        j: float | None = None,
        f: float | None = None,
        m: float | NotSet = NotSet,
        nu: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
            f: Total angular momentum quantum number of the atom (rydberg electron + core)
              Optional, only needed if the species supports hyperfine structure
              (i.e. species.i_c is not None and species.i_c != 0).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            nu: Effective principal quantum number of the rydberg electron.
              Optional, if not given it will be calculated from n, l, j.

        """
        super().__init__(species=species, n=n, nu=nu, l_r=l, j_tot=j, f_tot=f, m=m)

    def _set_qn_as_attributes(self) -> None:
        self.l = self.angular.l_r
        self.j = self.angular.j_tot
        self.f = self.angular.f_tot
        self.m = self.angular.m

    def __repr__(self) -> str:
        species, n, nu = self.species.name, self.n, self.nu
        l, j, f, m = self.l, self.j, self.f, self.m
        n_str = f", {n=}" if n is not None else ""
        f_string = f", {f=}" if self.species.i_c not in (None, 0) else ""
        return f"{self.__class__.__name__}({species}{n_str}, {nu=}, {l=}, {j=}{f_string}, {m=})"


class RydbergStateSQDTAlkalineLS(RydbergStateSQDT):
    """Create an Alkaline Rydberg state, including the radial and angular states."""

    angular: AngularKetLS

    def __init__(
        self,
        species: str | SpeciesObjectSQDT,
        n: int,
        l: int,
        s_tot: int,
        j_tot: int | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        nu: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            s_tot: Total spin quantum number of all electrons.
            j_tot: Total angular momentum quantum number of all electrons.
            f_tot: Total angular momentum quantum number of the atom (rydberg electron + core)
              Optional, only needed if the species supports hyperfine structure
              (i.e. species.i_c is not None and species.i_c != 0).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            nu: Effective principal quantum number of the rydberg electron.
              Optional, if not given it will be calculated from n, l, j_tot, s_tot.

        """
        super().__init__(species=species, n=n, nu=nu, l_r=l, s_tot=s_tot, j_tot=j_tot, f_tot=f_tot, m=m)

    def _set_qn_as_attributes(self) -> None:
        self.l = self.angular.l_r
        self.s_tot = self.angular.s_tot
        self.j_tot = self.angular.j_tot
        self.f_tot = self.angular.f_tot
        self.m = self.angular.m

    def __repr__(self) -> str:
        species, n, nu = self.species.name, self.n, self.nu
        l, s_tot, j_tot, f_tot, m = self.l, self.s_tot, self.j_tot, self.f_tot, self.m
        n_str = f", {n=}" if n is not None else ""
        return f"{self.__class__.__name__}({species}{n_str}, {nu=}, {l=}, {s_tot=}, {j_tot=}, {f_tot=}, {m=})"


class RydbergStateSQDTAlkalineJJ(RydbergStateSQDT):
    """Create an Alkaline Rydberg state, including the radial and angular states."""

    angular: AngularKetJJ

    def __init__(
        self,
        species: str | SpeciesObjectSQDT,
        n: int,
        l: int,
        j_r: float,
        j_tot: int | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        nu: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j_r: Total angular momentum quantum number of the Rydberg electron.
            j_tot: Total angular momentum quantum number of all electrons.
            f_tot: Total angular momentum quantum number of the atom (rydberg electron + core)
              Optional, only needed if the species supports hyperfine structure
              (i.e. species.i_c is not None and species.i_c != 0).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            nu: Effective principal quantum number of the rydberg electron.
              Optional, if not given it will be calculated from n, l, j_tot.

        """
        super().__init__(species=species, n=n, nu=nu, l_r=l, j_r=j_r, j_tot=j_tot, f_tot=f_tot, m=m)

    def _set_qn_as_attributes(self) -> None:
        self.l = self.angular.l_r
        self.j_r = self.angular.j_r
        self.j_tot = self.angular.j_tot
        self.f_tot = self.angular.f_tot
        self.m = self.angular.m

    def __repr__(self) -> str:
        species, n, nu = self.species.name, self.n, self.nu
        l, j_r, j_tot, f_tot, m = self.l, self.j_r, self.j_tot, self.f_tot, self.m
        n_str = f", {n=}" if n is not None else ""
        return f"{self.__class__.__name__}({species}{n_str}, {nu=}, {l=}, {j_r=}, {j_tot=}, {f_tot=}, {m=})"


class RydbergStateSQDTAlkalineFJ(RydbergStateSQDT):
    """Create an Alkaline Rydberg state, including the radial and angular states."""

    angular: AngularKetFJ

    def __init__(
        self,
        species: str | SpeciesObjectSQDT,
        n: int,
        l: int,
        j_r: float,
        f_c: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        nu: float | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j_r: Total angular momentum quantum number of the Rydberg electron.
            f_c: Total angular momentum quantum number of the core (core electron + nucleus).
            f_tot: Total angular momentum quantum number of the atom (rydberg electron + core)
              Optional, only needed if the species supports hyperfine structure
              (i.e. species.i_c is not None and species.i_c != 0).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            nu: Effective principal quantum number of the rydberg electron.
              Optional, if not given it will be calculated from n, l.

        """
        super().__init__(species=species, n=n, nu=nu, l_r=l, j_r=j_r, f_c=f_c, f_tot=f_tot, m=m)

    def _set_qn_as_attributes(self) -> None:
        self.l = self.angular.l_r
        self.j_r = self.angular.j_r
        self.f_c = self.angular.f_c
        self.f_tot = self.angular.f_tot
        self.m = self.angular.m

    def __repr__(self) -> str:
        species, n, nu = self.species.name, self.n, self.nu
        l, j_r, f_c, f_tot, m = self.l, self.j_r, self.f_c, self.f_tot, self.m
        l_c, j_c = self.angular.l_c, self.angular.j_c
        core_string = f", {l_c=}, {j_c=}" if l_c != 0 else ""
        n_str = f", {n=}" if n is not None else ""
        return f"{self.__class__.__name__}({species}{n_str}, {nu=}{core_string}, {l=}, {j_r=}, {f_c=}, {f_tot=}, {m=})"
