from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from rydstate.angular.angular_matrix_element import is_angular_momentum_quantum_number
from rydstate.rydberg.rydberg_base import RydbergStateBase
from rydstate.species.species_object import SpeciesObject
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.units import MatrixElementOperator, NDArray, PintArray, PintFloat

_RydbergState = TypeVar("_RydbergState", bound=RydbergStateBase)


class BasisBase(ABC, Generic[_RydbergState]):
    states: list[_RydbergState]

    def __init__(self, species: str | SpeciesObject) -> None:
        if isinstance(species, str):
            species = SpeciesObject.from_name(species)
        self.species = species

    def __len__(self) -> int:
        return len(self.states)

    def filter_states(self, qn: str, qn_min: float, qn_max: float) -> Self:
        if is_angular_momentum_quantum_number(qn):
            self.states = [state for state in self.states if qn_min <= state.angular.calc_exp_qn(qn) <= qn_max]
        elif qn in ["n", "nu", "nu_energy"]:
            self.states = [state for state in self.states if qn_min <= getattr(state, qn) <= qn_max]
        else:
            raise ValueError(f"Unknown quantum number {qn}")

        return self

    def calc_exp_qn(self, qn: str) -> list[float]:
        if is_angular_momentum_quantum_number(qn):
            return [state.angular.calc_exp_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [getattr(state, qn) for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_std_qn(self, qn: str) -> list[float]:
        if is_angular_momentum_quantum_number(qn):
            return [state.angular.calc_std_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [0 for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_reduced_overlap(self, other: RydbergStateBase) -> NDArray:
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
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintArray: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> NDArray: ...

    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintArray | NDArray:
        r"""Calculate the reduced matrix element."""
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
        r"""Calculate the reduced matrix element."""
        values_list = [
            [bra.calc_reduced_matrix_element(ket, operator, unit=unit) for ket in other.states] for bra in self.states
        ]
        if unit is not None:
            return np.array(values_list)

        values: list[list[PintFloat]] = values_list  # type: ignore[assignment]
        _unit = values[0][0].units
        _values = np.array([[v.magnitude for v in vs] for vs in values])
        return ureg.Quantity(_values, _unit)
