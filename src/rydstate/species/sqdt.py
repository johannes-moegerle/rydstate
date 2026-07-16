from __future__ import annotations

import logging
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.angular.utils import check_spin_addition_rule, get_possible_quantum_number_values, is_unknown
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.species.element_properties import get_element_properties
from rydstate.species.nist import parse_nist_energy_levels, resolve_species_data_file
from rydstate.species.utils import (
    calc_modified_ritz_formula,
    calc_nu_from_energy,
    get_all_subclasses,
)
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import Unknown
    from rydstate.species.nist import NistEnergyLevels
    from rydstate.species.utils import RydbergRitzParameters
    from rydstate.units import PintFloat


logger = logging.getLogger(__name__)


class SQDT(ABC, metaclass=CachedABCMeta):
    """Base class for all SQDT classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str | None] = None
    """The tag for these SQDT parameters."""
    is_default: ClassVar[bool] = False
    """Whether this SQDT is the default SQDT for the species."""
    nist_data_file: ClassVar[str | None] = None
    """Name of the file (in the species directory) with the low-lying NIST energy levels,
    usually ``"nist_data.txt"``. Left as None for species without a NIST data file."""

    ionization_energy: ClassVar[tuple[float, str]]
    """Ionization energy and unit: (value, unit)."""

    quantum_defects: ClassVar[dict[tuple[int, float, float], RydbergRitzParameters] | None] = None
    """Dictionary containing the quantum defects for each (l, j_tot, s_tot) combination, i.e.
    quantum_defects[(l,j_tot,s_tot)] = (d0, d2, d4, d6, d8)
    """

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)

        self._setup_nist_energy_levels()

    def __repr__(self) -> str:
        return f"SQDT({self.species}, {self.tag})"

    def _setup_nist_energy_levels(self) -> None:
        """Set up NIST energy levels.

        This method is called in the constructor to load the NIST energy levels.
        It reads the file given by ``nist_data_file`` and prepares the data for further use
        (see :func:`~rydstate.species.nist.parse_nist_energy_levels`).
        """
        self._nist_energy_levels: NistEnergyLevels = {}

        if self.nist_data_file is None:
            return

        file = resolve_species_data_file(type(self), self.nist_data_file)
        self._nist_energy_levels = parse_nist_energy_levels(
            file, self.element_properties.core_electron_configuration, species=self.species
        )

    def is_allowed_shell(self, n: int, l: int, s_tot: float | Unknown) -> bool:
        """Check if the quantum numbers describe an allowed shell.

        I.e. whether the shell is above the ground state shell.

        Args:
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            s_tot: Total spin quantum number

        Returns:
            True if the quantum numbers specify a shell equal to or above the ground state shell, False otherwise.

        """
        if is_unknown(s_tot):
            allowed_s_tot = get_possible_quantum_number_values(
                self.element_properties.s_c, self.element_properties.s_r, s_tot
            )
            return all(self.is_allowed_shell(n, l, _s_tot) for _s_tot in allowed_s_tot)

        if not check_spin_addition_rule(self.element_properties.s_c, self.element_properties.s_r, s_tot):
            raise ValueError(f"Invalid spin {s_tot=} for {self.species}.")

        if (n, l) == self.element_properties.ground_state_shell:
            return s_tot != 1  # For alkaline earth atoms, the triplet state of the ground state shell is not allowed
        if n < 1 or l < 0 or l >= n:
            raise ValueError(f"Invalid shell: (n={n}, l={l}). Must be n >= 1 and 0 <= l <= n-1.")
        if (n, l) >= self.element_properties.ground_state_shell:
            return True
        return (n, l) in self.element_properties.additional_allowed_shells

    @overload
    def get_ionization_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_energy(self, unit: str) -> float: ...

    def get_ionization_energy(self, unit: str | None = None) -> PintFloat | float:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is None (returns a Pint quantity).

        Returns:
            Ionization energy in the desired unit.

        """
        ionization_energy: PintFloat = ureg.Quantity(self.ionization_energy[0], self.ionization_energy[1])
        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    @cached_property
    def ionization_energy_au(self) -> float:
        """Ionization energy in atomic units (Hartree)."""
        return self.get_ionization_energy("hartree")

    def calc_nu(
        self,
        n: int,
        angular_ket: AngularKetBase[Any],
        *,
        use_nist_data: bool = True,
        nist_n_max: int = 15,
    ) -> float:
        r"""Calculate the effective principal quantum number nu of a Rydberg state with the given n, l, j_tot and s_tot.

        I.e. either look up the energy for low lying states in the nist data (if use_nist_data is True),
        and calculate nu from the energy via (see also `calc_nu_from_energy`):

        .. math::
            \nu = \sqrt{\frac{1}{2} \frac{\mu/m_e}{-E/E_H}}

        Or calculate nu via the quantum defect theory,
        where nu is defined as series expansion :math:`\nu = n^* = n - \delta_{lj}(n)`
        with the quantum defect

        .. math::
            \delta_{lj}(n) = d0_{lj} + d2_{lj} / [n - d0_{lj}(n)]^2 + d4_{lj} / [n - \delta_{lj}(n)]^4 + ...

        References:
            - On a New Law of Series Spectra, Ritz; DOI: 10.1086/141591, https://ui.adsabs.harvard.edu/abs/1908ApJ....28..237R/abstract
            - Rydberg atoms, Gallagher; DOI: 10.1088/0034-4885/51/2/001, (Eq. 16.19)

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket specifying l, j_tot, and s_tot of the Rydberg state.
            use_nist_data: Whether to use NIST energy data.
                Default is True.
            nist_n_max: Maximum principal quantum number for which to use the NIST energy data.
                Default is 15.

        """
        if angular_ket.coupling_scheme != "LS":
            angular_state = angular_ket.to_state("LS")
            if len(angular_state.kets) == 1:
                angular_ket = angular_state.kets[0]
            else:
                raise NotImplementedError("calc_nu is only implemented for AngularKetLS.")

        l_r = angular_ket.l_r
        j_tot = angular_ket.get_qn("j_tot", allow_unknown=True)
        s_tot = angular_ket.get_qn("s_tot", allow_unknown=True)

        if is_unknown(j_tot) or is_unknown(s_tot):
            raise ValueError(f"Cannot calculate nu for unknown j_tot or s_tot of {angular_ket!r}.")

        if n <= nist_n_max and use_nist_data:  # try to use NIST data
            if (n, l_r, j_tot, s_tot) in self._nist_energy_levels:
                energy_au = self._nist_energy_levels[(n, l_r, j_tot, s_tot)]
                energy_au -= self.ionization_energy_au  # use the cached ionization energy for better performance
                return calc_nu_from_energy(
                    self.element_properties.reduced_mass_au, energy_au, self.element_properties.net_charge
                )
            logger.debug(
                "NIST energy levels for (n=%d, l_r=%d, j_tot=%s, s_tot=%s) not found, using quantum defect theory.",
                *(n, l_r, j_tot, s_tot),
            )

        if self.quantum_defects is None:
            raise ValueError(f"No quantum defect data available for species {self.species}.")

        quantum_defects = self.quantum_defects.get((l_r, j_tot, s_tot), 0)
        delta_nlj = calc_modified_ritz_formula(n, quantum_defects)

        return n - delta_nlj


def get_sqdt(species: str, tag: str | None = None) -> SQDT:
    """Get an instance of the subclass of SQDT for the given species and tag."""
    subclasses = get_all_subclasses(SQDT, species, tag)

    if tag is None:
        subclasses = [cls for cls in subclasses if getattr(cls, "is_default", False)]

    if len(subclasses) == 0:
        raise ValueError(f"No subclass of SQDT found for {species=} and {tag=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of SQDT found for {species=} and {tag=}: {subclasses}.")
