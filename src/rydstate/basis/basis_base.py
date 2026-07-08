from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from rydstate.angular.utils import is_angular_momentum_quantum_number, is_unknown
from rydstate.rydberg_state.rydberg_base import RydbergState
from rydstate.species import get_element_properties
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.units import MatrixElementOperator, NDArray, PintArray, PintFloat

_RydbergState = TypeVar("_RydbergState", bound=RydbergState)


class BasisBase(ABC, Generic[_RydbergState]):
    states: list[_RydbergState]

    def __init__(self, species: str) -> None:
        self.species = species
        self.element_properties = get_element_properties(species)

    def __len__(self) -> int:
        return len(self.states)

    @overload
    def filter_states(
        self, qn: str, value: tuple[float, float], *, delta: float = 1e-10, keep_unknown: bool = False
    ) -> Self: ...

    @overload
    def filter_states(self, qn: str, value: float, *, delta: float = 1e-10, keep_unknown: bool = False) -> Self: ...

    def filter_states(
        self, qn: str, value: float | tuple[float, float], *, delta: float = 1e-10, keep_unknown: bool = False
    ) -> Self:
        if isinstance(value, Sequence) and len(value) == 2:
            qn_min = value[0] - delta
            qn_max = value[1] + delta
        else:
            qn_min = value - delta
            qn_max = value + delta

        if is_angular_momentum_quantum_number(qn):
            new_states: list[_RydbergState] = []
            for state in self.states:
                qn_value = state.calc_exp_qn(qn)
                if is_unknown(qn_value):
                    if keep_unknown:
                        new_states.append(state)
                elif qn_min <= qn_value <= qn_max:
                    new_states.append(state)
            self.states = new_states
        elif qn in ["n", "nu"]:
            self.states = [state for state in self.states if qn_min <= getattr(state, qn) <= qn_max]
        else:
            raise ValueError(f"Unknown quantum number {qn}")

        return self

    def sort_states(self, *qns: str) -> Self:
        """Sort the basis states according to the given quantum numbers.

        The first quantum number given is the primary sorting key, the second quantum number
        is the secondary sorting key, and so on.
        """
        values = np.array([self.calc_exp_qn(qn) for qn in qns])
        sorted_indices = np.lexsort(values[::-1])
        self.states = [self.states[i] for i in sorted_indices]
        return self

    def calc_exp_qn(self, qn: str) -> NDArray:
        return np.array([state.calc_exp_qn(qn) for state in self.states])

    def calc_std_qn(self, qn: str) -> NDArray:
        return np.array([state.calc_std_qn(qn) for state in self.states])

    def calc_short_range_exp_qn(self, qn: str, r_max: float) -> NDArray:
        return np.array([state.calc_short_range_exp_qn(qn, r_max) for state in self.states])

    def calc_reduced_overlap(self, other: RydbergState) -> NDArray:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        return np.array([bra.calc_reduced_overlap(other) for bra in self.states])

    def calc_reduced_overlaps(self, other: BasisBase[Any]) -> NDArray:
        """Calculate the reduced overlap <bra|ket> for all states in the bases self and other.

        Returns a numpy array overlaps, where overlaps[i,j] corresponds to the overlap of the
        i-th state of self and the j-th state of other.
        """
        return np.array([[bra.calc_reduced_overlap(ket) for ket in other.states] for bra in self.states])

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergState, operator: MatrixElementOperator, unit: None = None
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergState, operator: MatrixElementOperator, unit: str
    ) -> NDArray: ...

    def calc_reduced_matrix_element(
        self, other: RydbergState, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element :math:`\langle bra || O || other \rangle` for all states of self.

        Each state of the basis self is used as the bra and the single state other is used as the ket.

        Returns a 1D array values, where values[i] corresponds to the reduced matrix element
        :math:`\langle self.states[i] || O || other \rangle`.
        """
        values_list = [bra.calc_reduced_matrix_element(other, operator, unit=unit) for bra in self.states]
        if unit is not None:
            return np.array(values_list)

        values: list[PintFloat] = values_list  # type: ignore[assignment]
        _unit = values[0].units
        _values = np.array([v.magnitude for v in values])
        return ureg.Quantity(_values, _unit)

    @overload
    def calc_reduced_matrix_elements(
        self, other: BasisBase[Any], operator: MatrixElementOperator, unit: None = None
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_elements(
        self, other: BasisBase[Any], operator: MatrixElementOperator, unit: str
    ) -> NDArray: ...

    def calc_reduced_matrix_elements(
        self, other: BasisBase[Any], operator: MatrixElementOperator, unit: str | None = None
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element for all states in self and other.

        The states of the basis self are used as the bra and the states of the basis other are used as the ket.

        Returns a 2D array values, where values[i, j] corresponds to the reduced matrix element
        :math:`\langle self.states[i] || O || other.states[j] \rangle`.
        """
        values_list = [
            [bra.calc_reduced_matrix_element(ket, operator, unit=unit) for ket in other.states] for bra in self.states
        ]
        if unit is not None:
            return np.array(values_list)

        values: list[list[PintFloat]] = values_list  # type: ignore[assignment]
        _unit = values[0][0].units
        _values = np.array([[v.magnitude for v in vs] for vs in values])
        return ureg.Quantity(_values, _unit)
