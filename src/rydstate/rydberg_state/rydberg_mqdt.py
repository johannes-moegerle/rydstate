from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular.utils import is_unknown
from rydstate.rydberg_state.rydberg_base import RydbergState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.rydberg_state.rydberg_ket import RydbergKet
    from rydstate.species import MQDT, Potential
    from rydstate.species.fmodel import FModel
    from rydstate.units import NDArray


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergState):
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
        self.model = model
        self.potential_class = potential_class

        super().__init__(species, coefficients, rydberg_kets, nu, energy_au)

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

    @property
    def nui(self) -> NDArray:
        """Return the effective principal quantum numbers nui of the different channels."""
        return np.array([rydberg_ket.radial.nu for rydberg_ket in self.rydberg_kets])  # type: ignore [attr-defined]

    def calc_exp_qn(self, qn: str) -> float:
        if qn == "nui":
            coefficients2 = np.conjugate(self.coefficients) * self.coefficients / self.norm**2
            return float(np.sum(coefficients2 * self.nui))

        return super().calc_exp_qn(qn)

    def calc_std_qn(self, qn: str) -> float:
        if qn == "nui":
            coefficients2 = np.conjugate(self.coefficients) * self.coefficients / self.norm**2
            exp_q = np.sum(coefficients2 * self.nui)
            exp_q2 = np.sum(coefficients2 * self.nui * self.nui)
            if abs(exp_q2 - exp_q**2) < 1e-10:
                return 0
            return math.sqrt(exp_q2 - exp_q**2)

        return super().calc_std_qn(qn)
