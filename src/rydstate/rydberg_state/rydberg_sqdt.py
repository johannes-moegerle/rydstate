from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetLS
from rydstate.angular.utils import AllKnown, is_not_set, quantum_numbers_to_angular_ket
from rydstate.radial import RadialKet
from rydstate.rydberg_state.rydberg_base import RydbergState
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.species import get_element_properties, get_sqdt
from rydstate.species.potential import Potential, get_potential_class
from rydstate.species.sqdt import SQDT
from rydstate.species.utils import calc_energy_from_nu
from rydstate.units import BaseQuantities

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ
    from rydstate.units import PintFloat

GenericT_AngularKet = TypeVar("GenericT_AngularKet", bound=AngularKetBase[AllKnown])
T_AngularKet = TypeVar("T_AngularKet", bound=AngularKetBase[AllKnown])

logger = logging.getLogger(__name__)


class RydbergStateSQDT(RydbergState, Generic[GenericT_AngularKet]):
    """Create a Rydberg SQDT state, including the radial and angular states."""

    species: str
    """The atomic species of the Rydberg state."""

    angular: GenericT_AngularKet
    """The angular/spin part of the Rydberg electron."""

    @overload
    def __init__(
        self: RydbergStateSQDT[T_AngularKet],
        species: str,
        n: int,
        *,
        angular_ket: T_AngularKet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetLS[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        s_r: float | None = None,
        l_r: int,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetJJ[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        s_r: float | None = None,
        l_r: int,
        j_r: float | None = None,
        j_tot: float,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: RydbergStateSQDT[AngularKetFJ[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        f_c: float,
        s_r: float | None = None,
        l_r: int,
        j_r: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None: ...

    def __init__(
        self: RydbergStateSQDT[AngularKetBase[AllKnown]],
        species: str,
        n: int,
        *,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        f_c: float | None = None,
        s_r: float | None = None,
        l_r: int | None = None,
        j_r: float | None = None,
        s_tot: float | None = None,
        l_tot: int | None = None,
        j_tot: float | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        angular_ket: AngularKetBase[AllKnown] | None = None,
        # potential and sqdt parameters
        sqdt: str | SQDT | None = None,
        potential: str | Potential | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
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
            angular_ket: The angular ket to use for the state.
                Either angular_ket or the quantum numbers for the angular ket must be given.
            sqdt: The SQDT to use for the state.
                Either a string representing the tag of the SQDT class to use,
                or an instance of an SQDT class.
            potential: The potential to use for the radial ket.
                Either a string representing the tag of the potential to use,
                or an instance of a potential class.

        """
        self.species = species
        self.element_properties = get_element_properties(species)

        if angular_ket is not None:
            if any(
                q is not None for q in [s_c, l_c, j_c, f_c, s_r, l_r, j_r, s_tot, l_tot, j_tot, f_tot]
            ) or not is_not_set(m):
                raise ValueError("Specify either angular_ket or the quantum numbers for the angular ket, not both.")
            self.angular = angular_ket
        else:
            l_c = 0 if l_c is None else l_c

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
                m=m,
            )

        self.n = n
        self.sqdt = sqdt if isinstance(sqdt, SQDT) else get_sqdt(species, tag=sqdt)
        _s_tot = self.angular.get_qn("s_tot", allow_unknown=True)
        if not self.sqdt.is_allowed_shell(self.n, self.angular.l_r, _s_tot):
            raise ValueError(f"The Rydberg state {self} is not allowed due to forbidden shell configurations.")

        if isinstance(potential, Potential):
            if potential.l_r != self.angular.l_r:
                raise ValueError("The potential must have the same l_r as the angular ket.")
            self.potential = potential
        else:
            self.potential = get_potential_class(species, tag=potential)(self.angular.l_r)

        if abs(self.norm - 1) > 1e-10:
            raise ValueError(
                f"RydbergState initialized with non-normalized coefficients: {self._coefficients}, {self.rydberg_kets}"
            )

        self.f_tot = self.angular.f_tot
        self.parity = self.angular.parity
        self.m = self.angular.m
        self.coupling_scheme = self.angular.coupling_scheme

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.species}, n={self.n}, {self.angular!r})"

    def __str__(self) -> str:
        return f"|{self.species}:n={self.n}, {self.angular!s}⟩"

    @cached_property
    def nu(self) -> float:  # type: ignore [override]
        return self.sqdt.calc_nu(self.n, self.angular)

    @cached_property
    def _energy_au(self) -> float:  # type: ignore [override]
        return (
            calc_energy_from_nu(self.element_properties.reduced_mass_au, self.nu, self.element_properties.net_charge)
            + self.sqdt.ionization_energy_au
        )

    @cached_property
    def radial(self) -> RadialKet:
        """The radial part of the Rydberg electron."""
        return RadialKet(self.nu, self.potential, n_expected=self.n, sign_convention="n_l_1")

    @cached_property
    def _coefficients(self) -> list[float]:  # type: ignore [override]
        return [1.0]

    @cached_property
    def rydberg_kets(self) -> list[RydbergKet]:  # type: ignore [override]
        return [RydbergKet(self.species, self.angular, self.radial)]

    def _free_memory(self) -> None:
        super()._free_memory()
        self.__dict__.pop("radial", None)
        self.__dict__.pop("angular", None)

    @overload
    def get_radial_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_radial_energy(self, unit: str) -> float: ...

    def get_radial_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the energy of the radial part of the Rydberg state.

        The radial part of the energy is given by

        .. math::
            E = - \frac{1}{2} \frac{\mu}{\nu^2}

        where `\mu = R_M/R_\infty` is the reduced mass and `\nu` the effective principal quantum number.
        """
        _energy_au = calc_energy_from_nu(
            self.element_properties.reduced_mass_au, self.nu, self.element_properties.net_charge
        )
        if unit == "a.u.":
            return _energy_au
        energy: PintFloat = _energy_au * BaseQuantities["energy"]
        if unit is None:
            return energy
        return energy.to(unit, "spectroscopy").magnitude

    def calc_exp_qn(self, qn: str) -> float:
        if qn == "nui":
            return self.nu

        return super().calc_exp_qn(qn)

    def calc_std_qn(self, qn: str) -> float:
        if qn == "nui":
            return 0

        return super().calc_std_qn(qn)


class RydbergStateSQDTAlkali(RydbergStateSQDT[AngularKetLS[AllKnown]]):
    """Create an Alkali Rydberg state, including the radial and angular states."""

    def __init__(
        self,
        species: str,
        n: int,
        *,
        l: int | None = None,
        j: float | None = None,
        m: float | NotSet = NotSet,
        angular_ket: AngularKetLS[AllKnown] | None = None,
        # potential and sqdt parameters
        sqdt: SQDT | str | None = None,
        potential: Potential | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            angular_ket: The angular ket to use for the state.
              Either angular_ket or the quantum numbers for the angular ket must be given.
            sqdt: The SQDT to use for the state.
              Either a string representing the tag of the SQDT class to use,
              or an instance of an SQDT class.
            potential: The potential to use for the radial ket.
              Either a string representing the tag of the potential to use,
              or an instance of a potential class.

        """
        super().__init__(  # type: ignore [call-overload,misc]
            species=species, n=n, l_r=l, j_tot=j, m=m, angular_ket=angular_ket, sqdt=sqdt, potential=potential
        )
