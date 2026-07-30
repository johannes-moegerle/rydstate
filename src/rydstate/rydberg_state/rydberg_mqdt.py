from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ, AngularState
from rydstate.rydberg_state.rydberg_base import RydbergStateBase

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.species import MQDT, Potential
    from rydstate.species.fmodel import FModel
    from rydstate.units import NDArray


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergStateBase):
    angular: AngularState[AngularKetFJ[Any]]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        species: str,
        coefficients: Sequence[float] | NDArray,
        rydberg_kets: Sequence[RydbergKet],
        nu: float,
        energy_au: float,
        model: FModel,
        potential_class: type[Potential],
    ) -> None:
        self.species = species
        self._coefficients = np.asarray(coefficients).tolist()
        self.rydberg_kets = list(rydberg_kets)
        self.nu = float(nu)
        self._energy_au = float(energy_au)
        self.model = model
        self.potential_class = potential_class

        if len(rydberg_kets) == 0:
            raise ValueError("RydbergStateMQDT must be initialized with at least one state.")
        if len(coefficients) != len(rydberg_kets):
            raise ValueError("Length of coefficients and rydberg_kets must be the same.")
        if not all(isinstance(rydberg_ket.angular, AngularKetFJ) for rydberg_ket in rydberg_kets):
            raise ValueError("All rydberg_kets must have an angular part of type AngularKetFJ.")
        if len(set(rydberg_kets)) != len(rydberg_kets):
            raise ValueError("RydbergStateMQDT initialized with duplicate rydberg_kets.")

        self.angular = AngularState(self._coefficients, [ket.angular for ket in rydberg_kets])  # type: ignore [misc]

        super().__init__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{rydberg_ket!r}" for coeff, rydberg_ket in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{rydberg_ket!s}" for coeff, rydberg_ket in self]
        return f"{', '.join(terms)}"

    @property
    def mqdt(self) -> MQDT:
        """Return the MQDT object used to calculate this state."""
        return self.model.mqdt
