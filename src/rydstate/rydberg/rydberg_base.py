from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.species.utils import calc_nu_from_energy

if TYPE_CHECKING:
    from rydstate.angular import AngularState
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.species import SpeciesObject
    from rydstate.units import MatrixElementOperator, PintFloat

logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    angular: AngularState[Any] | AngularKetBase
    species: SpeciesObject

    @abstractmethod
    def get_energy(self, unit: str | None = None) -> PintFloat | float: ...

    @cached_property
    def nu_reference(self) -> float:
        """Return the reference effective principal quantum number for the state."""
        energy_au = self.get_energy("hartree")
        ref_ionization_energy_au = self.species.get_reference_ionization_energy("hartree")
        if ref_ionization_energy_au <= energy_au:
            return np.inf

        return calc_nu_from_energy(self.species.reduced_mass_au, energy_au - ref_ionization_energy_au)

    @abstractmethod
    def calc_reduced_overlap(self, other: RydbergStateBase) -> float: ...

    @abstractmethod
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintFloat | float: ...
