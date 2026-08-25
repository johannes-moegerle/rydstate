from __future__ import annotations

import copy
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from rydstate.rydberg_state.rydberg_base import RydbergState
from rydstate.species import get_element_properties
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.utils import Unknown
    from rydstate.units import MatrixElementOperator, MatrixElementPart, NDArray, PintArray, PintFloat

_RydbergState = TypeVar("_RydbergState", bound=RydbergState)


class BasisBase(ABC, Generic[_RydbergState]):
    states: list[_RydbergState]

    def __init__(self, species: str) -> None:
        self.species = species
        self.element_properties = get_element_properties(species)

    def __len__(self) -> int:
        return len(self.states)

    def shallow_copy(self) -> Self:
        """Return a shallow copy of the basis (with its own independent list of states)."""
        new_basis = copy.copy(self)
        new_basis.states = self.states.copy()
        return new_basis

    def filter_states(self, qn: str, value: float | Unknown | tuple[float, float], *, delta: float = 1e-10) -> Self:
        if isinstance(value, Sequence) and len(value) == 2:
            qn_min = value[0] - delta
            qn_max = value[1] + delta
        else:
            qn_min = value - delta  # type: ignore [operator]
            qn_max = value + delta  # type: ignore [operator]

        self.states = [state for state in self.states if qn_min <= state.calc_exp_qn(qn) <= qn_max]
        return self

    def filter_states_label(self, substring: str) -> Self:
        """Filter the basis states by a substring in their label."""
        self.states = [
            state
            for state in self.states
            if any(substring in ket.angular.label for ket in state.rydberg_kets if ket.angular.label is not None)
        ]
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
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> NDArray: ...

    def calc_reduced_matrix_element(
        self,
        other: RydbergState,
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str | None = None,
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element :math:`\langle bra || O || other \rangle` for all states of self.

        Each state of the basis self is used as the bra and the single state other is used as the ket.

        Returns a 1D array values, where values[i] corresponds to the reduced matrix element
        :math:`\langle self.states[i] || O || other \rangle`.
        """
        values_list = [bra.calc_reduced_matrix_element(other, operator, part=part, unit=unit) for bra in self.states]
        if unit is not None:
            return np.array(values_list)

        values: list[PintFloat] = values_list  # type: ignore[assignment]
        _unit = values[0].units
        _values = np.array([v.magnitude for v in values])
        return ureg.Quantity(_values, _unit)

    @overload
    def calc_reduced_matrix_elements(
        self,
        other: BasisBase[Any],
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: None = None,
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_elements(
        self,
        other: BasisBase[Any],
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str,
    ) -> NDArray: ...

    def calc_reduced_matrix_elements(
        self,
        other: BasisBase[Any],
        operator: MatrixElementOperator,
        *,
        part: MatrixElementPart = "all",
        unit: str | None = None,
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element for all states in self and other.

        The states of the basis self are used as the bra and the states of the basis other are used as the ket.

        Returns a 2D array values, where values[i, j] corresponds to the reduced matrix element
        :math:`\langle self.states[i] || O || other.states[j] \rangle`.
        """
        values_list = [
            [bra.calc_reduced_matrix_element(ket, operator, part=part, unit=unit) for ket in other.states]
            for bra in self.states
        ]
        if unit is not None:
            return np.array(values_list)

        values: list[list[PintFloat]] = values_list  # type: ignore[assignment]
        _unit = values[0][0].units
        _values = np.array([[v.magnitude for v in vs] for vs in values])
        return ureg.Quantity(_values, _unit)
