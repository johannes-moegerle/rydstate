from __future__ import annotations

import numpy as np

from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg import RydbergStateSQDTAlkali, RydbergStateSQDTAlkalineLS


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
