from __future__ import annotations

import logging
import re
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np

from rydstate.angular.angular_ket import AngularKetLS
from rydstate.species.species_object import SpeciesObject
from rydstate.species.utils import calc_nu_from_energy, convert_electron_configuration
from rydstate.units import ureg

if TYPE_CHECKING:
    from pathlib import Path

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.units import PintFloat

logger = logging.getLogger(__name__)


class SpeciesObjectSQDT(SpeciesObject):
    """Abstract base class for all species objects.

    For the electronic ground state configurations and sorted shells,
    see e.g. https://www.webelements.com/atoms.html

    """

    ground_state_shell: ClassVar[tuple[int, int]]
    """Shell (n, l) describing the electronic ground state configuration."""
    _additional_allowed_shells: ClassVar[list[tuple[int, int]]] = []
    """Additional allowed shells (n, l), which (n, l) is smaller than the ground state shell."""

    _core_electron_configuration: ClassVar[str]
    """Electron configuration of the core electrons, e.g. 4p6 for Rb or 5s for Sr."""
    _ionization_energy: tuple[float, float | None, str]
    """Ionization energy with uncertainty and unit: (value, uncertainty, unit)."""

    # Parameters for the extended Rydberg Ritz formula, see calc_nu
    _quantum_defects: ClassVar[dict[tuple[int, float, float], tuple[float, float, float, float, float]] | None] = None
    """Dictionary containing the quantum defects for each (l, j_tot, s_tot) combination, i.e.
    _quantum_defects[(l,j_tot,s_tot)] = (d0, d2, d4, d6, d8)
    """

    _nist_energy_levels_file: Path | None = None
    """Path to the NIST energy levels file for this species.
    The file should be directly downloaded from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    in the 'Tab-delimited' format and in units of Hartree.
    """

    def __init__(self) -> None:
        """Initialize an species instance.

        Use this init method to set up additional properties and data for the species,
        like loading NIST energy levels from a file.

        """
        self._nist_energy_levels: dict[tuple[int, int, float, float], float] = {}
        if self._nist_energy_levels_file is not None:
            self._setup_nist_energy_levels(self._nist_energy_levels_file)

    def _setup_nist_energy_levels(self, file: Path) -> None:  # noqa: C901, PLR0912
        """Set up NIST energy levels from a file.

        This method should be called in the constructor to load the NIST energy levels
        from the specified file. It reads the file and prepares the data for further use.

        Args:
            file: Path to the NIST energy levels file.

        """
        if not file.exists():
            raise ValueError(f"NIST energy data file {file} does not exist.")

        header = file.read_text().splitlines()[0]
        if "Level (Hartree)" not in header:
            raise ValueError(
                f"NIST energy data file {file} not given in Hartree, please download the data in units of Hartree."
            )

        data = np.loadtxt(file, skiprows=1, dtype=str, quotechar='"', delimiter="\t")
        # data[i] := (Configuration, Term, J, Prefix, Energy, Suffix, Uncertainty, Reference)
        core_config_parts = convert_electron_configuration(self._core_electron_configuration)

        for row in data:
            if re.match(r"^([A-Z])", row[0]):
                # Skip rows, where the first column starts with an element symbol
                continue

            try:
                config_parts = convert_electron_configuration(row[0])
            except ValueError:
                # Skip rows with invalid electron configuration format
                # (they usually correspond to core configurations, that are not the ground state configuration)
                # e.g. strontium "4d.(2D<3/2>).4f"
                continue
            if sum(part[2] for part in config_parts) != sum(part[2] for part in core_config_parts) + 1:
                # Skip configurations, where the number of electrons does not match the core configuration + 1
                continue

            for part in core_config_parts:
                if part in config_parts:
                    config_parts.remove(part)
                elif (part[0], part[1], part[2] + 1) in config_parts:
                    config_parts.remove((part[0], part[1], part[2] + 1))
                    config_parts.append((part[0], part[1], 1))
                else:
                    break
            if sum(part[2] for part in config_parts) != 1:
                # Skip configurations, where the inner electrons are not in the ground state configuration
                continue
            n, l = config_parts[0][:2]

            multiplicity = int(row[1][0])
            s_tot = (multiplicity - 1) / 2

            j_tot_list = [float(Fraction(j_str)) for j_str in row[2].split(",")]
            for j_tot in j_tot_list:
                energy = float(row[4])
                self._nist_energy_levels[(n, l, j_tot, s_tot)] = energy

        if len(self._nist_energy_levels) == 0:
            raise ValueError(f"No NIST energy levels found for species {self.name} in file {file}.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def is_allowed_shell(self, n: int, l: int, s_tot: float | None = None) -> bool:
        """Check if the quantum numbers describe an allowed shell.

        I.e. whether the shell is above the ground state shell.

        Args:
            n: Principal quantum number
            l: Orbital angular momentum quantum number
            s_tot: Total spin quantum number

        Returns:
            True if the quantum numbers specify a shell equal to or above the ground state shell, False otherwise.

        """
        if s_tot is None:
            if self.number_valence_electrons > 1:
                raise ValueError("s_tot must be specified for species with more than one valence electron.")
            s_tot = self.number_valence_electrons / 2
        if (self.number_valence_electrons / 2) % 1 != s_tot % 1 or s_tot > self.number_valence_electrons / 2:
            raise ValueError(f"Invalid spin {s_tot=} for {self.name}.")

        if (n, l) == self.ground_state_shell:
            return s_tot != 1  # For alkaline earth atoms, the triplet state of the ground state shell is not allowed
        if n < 1 or l < 0 or l >= n:
            raise ValueError(f"Invalid shell: (n={n}, l={l}). Must be n >= 1 and 0 <= l <= n-1.")
        if (n, l) >= self.ground_state_shell:
            return True
        return (n, l) in self._additional_allowed_shells

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
        ionization_energy: PintFloat = ureg.Quantity(self._ionization_energy[0], self._ionization_energy[2])
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
        angular_ket: AngularKetBase,
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
        if not isinstance(angular_ket, AngularKetLS):
            raise NotImplementedError("calc_nu is only implemented for AngularKetLS.")

        l, j_tot, s_tot = angular_ket.l_r, angular_ket.j_tot, angular_ket.s_tot

        if n <= nist_n_max and use_nist_data:  # try to use NIST data
            if (n, l, j_tot, s_tot) in self._nist_energy_levels:
                energy_au = self._nist_energy_levels[(n, l, j_tot, s_tot)]
                energy_au -= self.ionization_energy_au  # use the cached ionization energy for better performance
                return calc_nu_from_energy(self.reduced_mass_au, energy_au)
            logger.debug(
                "NIST energy levels for (n=%d, l=%d, j_tot=%s, s_tot=%s) not found, using quantum defect theory.",
                *(n, l, j_tot, s_tot),
            )

        if self._quantum_defects is None:
            raise ValueError(f"No quantum defect data available for species {self.name}.")
        quantum_defects = list(self._quantum_defects.get((l, j_tot, s_tot), []))
        if len(quantum_defects) > 5:
            raise ValueError(f"Quantum defects for {self.name} must be a list with up to 5 elements.")

        d0, d2, d4, d6, d8 = quantum_defects + [0] * (5 - len(quantum_defects))
        delta_nlj = d0 + d2 / (n - d0) ** 2 + d4 / (n - d0) ** 4 + d6 / (n - d0) ** 6 + d8 / (n - d0) ** 8
        return n - delta_nlj
