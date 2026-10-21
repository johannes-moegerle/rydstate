from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from rydstate import units
from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetLS
from rydstate.angular.utils import AllKnown, is_unknown
from rydstate.element_properties import get_element_properties
from rydstate.qdt import get_sqdt
from rydstate.radial import RadialKet
from rydstate.radial.potential import Potential, get_potential_class
from rydstate.radial.radial_ket import RadialDummy
from rydstate.rydberg_state.rydberg_base import RydbergState
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.utils import calc_energy_from_nu, calc_nu_from_energy

if TYPE_CHECKING:
    from rydstate.qdt import ModelOSQDT, ModelSQDTBase
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
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            angular: The angular ket of the state, which can be given in any coupling scheme,
                i.e. as :class:`~rydstate.angular.AngularKetLS`, :class:`~rydstate.angular.AngularKetJJ`
                or :class:`~rydstate.angular.AngularKetFJ`.
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
    def model(self) -> ModelSQDTBase | ModelOSQDT:
        """The model describing this state.

        This is the model of the SQDT model set of the species, see :meth:`~rydstate.qdt.ModelSetSQDT.get_model`.
        """
        return get_sqdt(self.species).get_model(self.n, self.angular)

    @cached_property
    def nui(self) -> float:
        r"""The effective principal quantum number nui, defined with reference to the ionization threshold.

        nui is defined with reference to the ionization threshold of the channel
        (see :attr:`~rydstate.qdt.ModelSQDT.ionization_threshold_au` of the :attr:`model`), i.e. via the binding energy

        .. math::
            E = I - \frac{R_M}{\nu_i^2} = I_{\text{ref}} - \frac{R_M}{\nu^2}

        where :math:`R_M = R_\infty \mu/m_e` is the mass corrected Rydberg constant.

        Therefore, nui is the quantum number that determines the radial wavefunction, see :attr:`radial`.
        For low lying states, nui is taken from the NIST energy levels if available,
        see :meth:`~rydstate.qdt.ModelSetSQDT.calc_nui`.
        """
        return get_sqdt(self.species).calc_nui(self.n, self.angular)

    @cached_property
    def nu(self) -> float:  # type: ignore [override]
        r"""The effective principal quantum number nu, defined with reference to the reference ionization threshold.

        In contrast to :attr:`nui`, nu is defined with reference to the reference ionization threshold of the species
        (see :attr:`~rydstate.qdt.Model.reference_ionization_threshold_au` of the :attr:`model`).

        For states without hyperfine splitting, nui and nu are equal.
        """
        reference_threshold_au = self.model.reference_ionization_threshold_au
        reference_shift = self.model.ionization_threshold_au - reference_threshold_au
        if reference_shift == 0:
            return self.nui
        reduced_mass_au = self.element_properties.reduced_mass_au
        net_charge = self.element_properties.net_charge
        energy_wrt_reference_au = self.get_binding_energy("a.u.") + reference_shift
        if energy_wrt_reference_au >= 0:
            raise ValueError(
                f"The energy of the state {self!r} lies above the reference ionization threshold "
                f"({reference_threshold_au} hartree), thus nu is not defined."
            )
        return calc_nu_from_energy(reduced_mass_au, energy_wrt_reference_au, net_charge)

    @cached_property
    def _energy_au(self) -> float:  # type: ignore [override]
        return self.get_binding_energy("a.u.") + self.model.ionization_threshold_au

    @cached_property
    def radial(self) -> RadialKet:
        """The radial part of the Rydberg electron."""
        return RadialKet(self.nui, self.potential_class(self.angular.l_r), n_expected=self.n, sign_convention="n_l_1")

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
            E = - \frac{Z^2 R_M}{\nu_i^2}
              = - \frac{1}{2} \frac{Z^2 \mu/m_e}{\nu_i^2} E_H

        where :math:`R_M = R_\infty \mu/m_e` is the mass corrected Rydberg constant,
        :math:`Z` is the net charge of the ionic core seen by the Rydberg electron
        (note :math:`E_H = 2 R_\infty`), :math:`\mu/m_e` the reduced mass in atomic units
        and :math:`\nu_i` the effective principal quantum number
        with reference to the ionization energy, see :attr:`nui`.
        """
        energy_au = calc_energy_from_nu(
            self.element_properties.reduced_mass_au, self.nui, self.element_properties.net_charge
        )
        return units.au_to_user(energy_au, "energy", unit)

    def calc_exp_qn(self, qn: str) -> float:
        if qn == "nui":
            return self.nui

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
        # potential parameters
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
            potential_class: The potential class to use for the radial ket.
                Either a string representing the tag of the potential class to use, or a potential class.
                If None, the default potential class for the species is used.

        """
        angular = AngularKetLS(l_c=0, l_r=l, j_tot=j, m=m, species=species)
        super().__init__(species, n, angular, potential_class=potential_class)


class RydbergStateSQDTDivalent(RydbergStateSQDT[AngularKetLS[AllKnown]]):
    """Create a divalent Rydberg state, including the radial and angular states.

    This is a convenience wrapper around :class:`RydbergStateSQDT`,
    which constructs the LS coupled angular ket from the divalent quantum numbers s, l, j (and f).
    """

    def __init__(
        self,
        species: str,
        n: int,
        *,
        l: int,
        s: int,
        j: int | None = None,
        f: float | None = None,
        m: float | NotSet = NotSet,
        # potential parameters
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        r"""Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron.
            l: Orbital angular momentum quantum number of the rydberg electron.
            s: Total electron spin quantum number, i.e. 0 for singlet and 1 for triplet states.
            j: Total electron angular momentum quantum number.
              Optional, if it is uniquely determined by l and s (i.e. for l = 0).
            f: Total angular momentum quantum number, including the nuclear spin.
              Optional, if it is uniquely determined by j (i.e. for species without nuclear spin).
            m: Total magnetic quantum number.
              Optional, only needed for concrete angular matrix elements.
            potential_class: The potential class to use for the radial ket.
                Either a string representing the tag of the potential class to use, or a potential class.
                If None, the default potential class for the species is used.

        """
        angular = AngularKetLS(l_c=0, l_r=l, s_tot=s, j_tot=j, f_tot=f, m=m, species=species)
        super().__init__(species, n, angular, potential_class=potential_class)


class RydbergStateOSQDT(RydbergStateSQDT[AngularKetFJ[AllKnown]]):
    """Create a Rydberg state of a single outer channel of the MQDT models.

    The states of this class are not meant to be used directly.
    They are rather used for comparison and didactic purposes.

    This is a :class:`RydbergStateSQDT`, whose quantum defects are given by an
    :class:`~rydstate.qdt.ModelOSQDT` model, i.e. by the diagonal elements of the K-matrix
    of an MQDT model in the outer channel frame.
    """

    model: ModelOSQDT
    """The OSQDT model describing this state."""

    def __init__(
        self,
        species: str,
        n: int,
        angular: AngularKetFJ[AllKnown],
        model: ModelOSQDT,
        *,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        """Initialize the Rydberg state.

        Args:
            species: Atomic species.
            n: Principal quantum number of the rydberg electron, see :meth:`~rydstate.qdt.ModelSetOSQDT.calc_nui`.
            angular: The angular ket of the state, i.e. the channel of the OSQDT model.
            model: The OSQDT model describing this state, which must have a state with the principal quantum number n.
            potential_class: The potential class to use for the radial ket.
                Either a string representing the tag of the potential class to use, or a potential class.
                If None, the default potential class for the species is used.

        """
        if angular.replace_m(NotSet) != model.channel:
            raise ValueError(f"The angular ket {angular!r} is not the channel {model.channel!r} of the OSQDT model.")
        if n not in model.solutions:
            raise ValueError(f"The OSQDT model {model} has no state with n={n}.")
        self.model = model
        super().__init__(species, n, angular, potential_class=potential_class)

    @cached_property
    def nui(self) -> float:
        """The effective principal quantum number nui of the channel, given by the OSQDT model."""
        return self.model.solutions[self.n]

    @cached_property
    def radial(self) -> RadialKet | RadialDummy:  # type: ignore [override]
        """The radial part of the Rydberg electron."""
        if is_unknown(self.angular.l_r):
            return RadialDummy(1.0, self.nui)
        return RadialKet(
            self.nui,
            self.potential_class(self.angular.l_r),
            sign_convention="positive_at_outer_bound",
            n_expected=self.n,
        )
