from __future__ import annotations

from abc import ABC
from typing import get_args

import numpy as np
from typing_extensions import Self

from rydstate.angular.angular_matrix_element import AngularMomentumQuantumNumbers
from rydstate.rydberg import RydbergStateMQDT, RydbergStateSQDT, RydbergStateSQDTAlkali, RydbergStateSQDTAlkalineLS
from rydstate.species.species_object import SpeciesObject


class BasisBase(ABC):
    states: list[RydbergStateSQDT | RydbergStateMQDT]

    def __init__(self, species: str | SpeciesObject) -> None:
        if isinstance(species, str):
            species = SpeciesObject.from_name(species)
        self.species = species

    def __len__(self) -> int:
        return len(self.states)

    def filter_states(self, qn: str, qn_min: float, qn_max: float) -> Self:
        if qn in get_args(AngularMomentumQuantumNumbers):
            self.states = [state for state in self.states if qn_min <= state.angular.calc_exp_qn(qn) <= qn_max]
        elif qn in ["n", "nu"]:
            self.states = [state for state in self.states if qn_min <= getattr(state, qn) <= qn_max]
        else:
            raise ValueError(f"Unknown quantum number {qn}")

        return self


class BasisSQDTAlkali(BasisBase):
    def __init__(self, species: str, n_min: int = 1, n_max: int | None = None):
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        s = 1 / 2
        i_c = self.species.i_c if self.species.i_c is not None else 0
        states: list[RydbergStateSQDTAlkali] = []
        for n in range(n_min, n_max + 1):
            for l in range(n):
                for j in np.arange(abs(l - s), l + s + 1):
                    for f in np.arange(abs(j - i_c), j + i_c + 1):
                        state = RydbergStateSQDTAlkali(species, n=n, l=l, j=j, f=f)
                        states.append(state)

        self.states = states


class BasisSQDTAlkalineLS(BasisBase):
    def __init__(self, species: str, n_min: int = 1, n_max: int | None = None):
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        i_c = self.species.i_c if self.species.i_c is not None else 0
        states: list[RydbergStateSQDTAlkalineLS] = []
        for s_tot in [0, 1]:
            for n in range(n_min, n_max + 1):
                for l in range(n):
                    for j_tot in range(abs(l - s_tot), l + s_tot + 1):
                        for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                            state = RydbergStateSQDTAlkalineLS(species, n=n, l=l, s_tot=s_tot, j_tot=j_tot, f_tot=f_tot)
                            states.append(state)

        self.states = states
