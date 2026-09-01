from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetLS
from rydstate.angular.utils import AllKnown
from rydstate.radial import RadialKet
from rydstate.rydberg_state.rydberg_base import RydbergState
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.species import get_element_properties, get_sqdt
from rydstate.species.potential import Potential, get_potential_class
from rydstate.species.sqdt import SQDT
from rydstate.species.utils import calc_energy_from_nu
from rydstate.units import BaseQuantities

if TYPE_CHECKING:
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

    def __init__(
        self: RydbergStateSQDT[T_AngularKet],
        species: str,
        n: int,
        angular: T_AngularKet,
        *,
        sqdt: str | SQDT | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            angular: The angular ket of the state, which can be given in any coupling scheme,
                i.e. as :class:`~rydstate.angular.AngularKetLS`, :class:`~rydstate.angular.AngularKetJJ`
                or :class:`~rydstate.angular.AngularKetFJ`.
            sqdt: The SQDT to use for the state.
                Either a string representing the tag of the SQDT class to use,
                or an instance of an SQDT class.
            potential_class: The potential class to use for the radial ket.
                Either a string representing the tag of the potential class to use, or a potential class.
                If None, the default potential class for the species is used.

        """
        self.species = species
        self.element_properties = get_element_properties(species)

        if angular.i_c != self.element_properties.i_c or angular.s_c != self.element_properties.s_c:
            raise ValueError(
                f"The angular ket {angular!r} does not belong to the species {species} "
                f"with i_c={self.element_properties.i_c} and s_c={self.element_properties.s_c}."
            )
        self.angular = angular

        self.n = n
        self.sqdt = sqdt if isinstance(sqdt, SQDT) else get_sqdt(species, tag=sqdt)
        _s_tot = self.angular.get_qn("s_tot", allow_unknown=True)
        if not self.element_properties.is_allowed_shell(self.n, self.angular.l_r, _s_tot):
            raise ValueError(f"The Rydberg state {self!r} is not allowed due to forbidden shell configurations.")

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        elif potential_class is None or isinstance(potential_class, str):
            self.potential_class = get_potential_class(species, tag=potential_class)
        else:
            raise TypeError(
                f"potential_class must be a subclass of Potential, a string tag or None, got {potential_class!r}."
            )

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
        return RadialKet(self.nu, self.potential_class(self.angular.l_r), n_expected=self.n, sign_convention="n_l_1")

    @cached_property
    def _coefficients(self) -> list[float]:  # type: ignore [override]
        return [1.0]

    @cached_property
    def rydberg_kets(self) -> list[RydbergKet]:  # type: ignore [override]
        return [RydbergKet(self.species, self.angular, self.radial, n=self.n)]

    def _free_memory(self) -> None:
        super()._free_memory()
        self.__dict__.pop("radial", None)
        self.__dict__.pop("angular", None)

    @overload
    def get_binding_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_binding_energy(self, unit: str) -> float: ...

    def get_binding_energy(self, unit: str | None = None) -> PintFloat | float:
        r"""Get the binding energy of the Rydberg state, relative to its ionization energy.

        The binding energy is negative by convention, so it can be added directly to the
        ionization energy to obtain the total state energy.

        The binding energy is given by

        .. math::
            E = - \frac{Z^2 R_M}{\nu^2}
              = - \frac{1}{2} \frac{Z^2 \mu/m_e}{\nu^2} E_H

        where :math:`R_M = R_\infty \mu/m_e` is the mass corrected Rydberg constant,
        :math:`Z` is the net charge of the ionic core seen by the Rydberg electron
        (note :math:`E_H = 2 R_\infty`), :math:`\mu/m_e` the reduced mass in atomic units
        and :math:`\nu` the effective principal quantum number.
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
    """Create an Alkali Rydberg state, including the radial and angular states.

    This is a convenience wrapper around :class:`RydbergStateSQDT`,
    which constructs the LS coupled angular ket from the alkali quantum numbers l and j.
    """

    def __init__(
        self,
        species: str,
        n: int,
        *,
        l: int,
        j: float | None = None,
        m: float | NotSet = NotSet,
        # potential and sqdt parameters
        sqdt: SQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            j: Angular momentum quantum number of the rydberg electron.
              Optional, if it is uniquely determined by l (i.e. for l = 0).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            sqdt: The SQDT to use for the state.
              Either a string representing the tag of the SQDT class to use,
              or an instance of an SQDT class.
            potential_class: The potential class to use for the radial ket.
                Either a string representing the tag of the potential class to use, or a potential class.
                If None, the default potential class for the species is used.

        """
        angular = AngularKetLS(l_c=0, l_r=l, j_tot=j, m=m, species=species)
        super().__init__(species, n, angular, sqdt=sqdt, potential_class=potential_class)
