from __future__ import annotations

import logging
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.angular.utils import is_unknown
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
    """Ionization energy: (value, unit)."""
    _reference_ionization_energy: ClassVar[tuple[float, str] | None] = None
    """Reference ionization energy: (value, unit), in reference to which nu is defined.
    If None, the ionization_energy is used as the reference and nu = nui."""

    quantum_defects: ClassVar[dict[tuple[int, float, float], RydbergRitzParameters] | None] = None
    """Dictionary containing the quantum defects for each (l, j_tot, s_tot) combination, i.e.
    quantum_defects[(l,j_tot,s_tot)] = (d0, d2, d4, d6, d8)
    """

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)

        self._nist_energy_levels: NistEnergyLevels = {}
        if self.nist_data_file is not None:
            # Load the NIST energy levels if a NIST data file is specified
            file = resolve_species_data_file(type(self), self.nist_data_file)
            self._nist_energy_levels = parse_nist_energy_levels(file, self.element_properties)

    def __repr__(self) -> str:
        return f"SQDT({self.species}, {self.tag})"

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

    @cached_property
    def reference_ionization_energy_au(self) -> float:
        r"""Reference ionization energy in atomic units (Hartree).

        The reference ionization energy defines nu via

        .. math::
            E = I_i - \frac{Z^2 R_M}{\nu_i^2}
              = I_{\text{ref}} - \frac{Z^2 R_M}{\nu^2}

        If no :attr:`_reference_ionization_energy` is defined (as it is the case e.g. for all alkali atoms),
        the :attr:`ionization_energy` is used as the reference and nu = nui.
        """
        if self._reference_ionization_energy is None:
            return self.ionization_energy_au
        value, unit = self._reference_ionization_energy
        reference: PintFloat = ureg.Quantity(value, unit)
        return float(reference.to("hartree", "spectroscopy").magnitude)

    def calc_nui(
        self,
        n: int,
        angular_ket: AngularKetBase[Any],
        *,
        use_nist_data: bool = True,
        nist_n_max: int = 15,
    ) -> float:
        r"""Calculate the effective principal quantum number nui of a Rydberg state with the given n and angular ket.

        The effective principal quantum number nui is defined with reference to the
        :attr:`ionization_energy`, i.e. it describes the binding energy of the Rydberg electron
        :math:`E = I_i - \frac{Z^2 R_M}{\nu_i^2}`
        with the mass corrected Rydberg constant :math:`R_M = R_\infty \mu/m_e`.

        I.e. either look up the energy for low lying states in the nist data (if use_nist_data is True),
        and calculate nui from the energy via (see also `calc_nu_from_energy`):

        .. math::
            \nu_i = Z \sqrt{\frac{1}{2} \frac{\mu/m_e}{-(E - I)/E_H}}

        Or calculate nui via the quantum defect theory,
        where nui is defined as series expansion :math:`\nu_i = n^* = n - \delta_{lj}(n)`
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
                raise NotImplementedError("calc_nui is only implemented for AngularKetLS.")

        l_r = angular_ket.l_r
        j_tot = angular_ket.get_qn("j_tot", allow_unknown=True)
        s_tot = angular_ket.get_qn("s_tot", allow_unknown=True)

        if is_unknown(j_tot) or is_unknown(s_tot):
            raise ValueError(f"Cannot calculate nui for unknown j_tot or s_tot of {angular_ket!r}.")

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
