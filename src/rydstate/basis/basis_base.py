from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, get_args

from typing_extensions import Self

from rydstate.angular.angular_matrix_element import AngularMomentumQuantumNumbers
from rydstate.species.species_object import SpeciesObject

if TYPE_CHECKING:
    from rydstate.rydberg import RydbergStateSQDT


class BasisBase(ABC):
    states: list[RydbergStateSQDT]

    def __init__(self, species: str | SpeciesObject) -> None:
        if isinstance(species, str):
            species = SpeciesObject.from_name(species)
        self.species = species

    def __len__(self) -> int:
        return len(self.states)

    def filter_states(self, qn: str, qn_min: float, qn_max: float) -> Self:
        if qn in get_args(AngularMomentumQuantumNumbers):
            self.states = [state for state in self.states if qn_min <= state.angular.calc_exp_qn(qn) <= qn_max]
        elif qn in ["n", "nu", "nu_energy"]:
            self.states = [state for state in self.states if qn_min <= getattr(state, qn) <= qn_max]
        else:
            raise ValueError(f"Unknown quantum number {qn}")

        return self

    def calc_exp_qn(self, qn: str) -> list[float]:
        if qn in get_args(AngularMomentumQuantumNumbers):
            return [state.angular.calc_exp_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [getattr(state, qn) for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")

    def calc_std_qn(self, qn: str) -> list[float]:
        if qn in get_args(AngularMomentumQuantumNumbers):
            return [state.angular.calc_std_qn(qn) for state in self.states]
        if qn in ["n", "nu", "nu_energy"]:
            return [0 for state in self.states]
        raise ValueError(f"Unknown quantum number {qn}")
