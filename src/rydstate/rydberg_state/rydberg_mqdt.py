from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ, AngularState
from rydstate.angular.utils import is_unknown
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
        self.f_tot = self.angular.f_tot

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

    @cached_property
    def n(self) -> int:  # type: ignore [override]
        """Return the corresponding principal quantum number n of the state.

        We define the corresponding principal quantum number n for MQDT states via the nodes of
        the main contributing rydberg ket (nodes = n - l_r - 1).
        For FModelSQDT states, the quantum defect is zero, so the channel dependent effective quantum number nui
        is already an integer and we simply round it to the nearest integer.
        """
        defects = self.model.eigen_quantum_defects
        if (
            len(defects) == 1 and np.isscalar(defects[0]) and abs(defects[0]) < 1e-10  # type: ignore [arg-type]
        ):
            return round(self.nui[0])

        main_ket = max(
            [(coeff, ket) for coeff, ket in self if not is_unknown(ket.angular.l_r)], key=lambda x: abs(x[0])
        )[1]
        return main_ket.radial.nodes + main_ket.angular.l_r + 1
