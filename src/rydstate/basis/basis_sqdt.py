from __future__ import annotations

import logging

import numpy as np

from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg import (
    RydbergStateSQDTAlkali,
    RydbergStateSQDTAlkalineFJ,
    RydbergStateSQDTAlkalineJJ,
    RydbergStateSQDTAlkalineLS,
)

logger = logging.getLogger(__name__)


class BasisSQDTAlkali(BasisBase[RydbergStateSQDTAlkali]):
    def __init__(self, species: str, n_min: int = 1, n_max: int | None = None) -> None:
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        s = 1 / 2
        i_c = self.species.i_c if self.species.i_c is not None else 0

        self.states = []
        for n in range(n_min, n_max + 1):
            for l in range(n):
                if not self.species.is_allowed_shell(n, l, s):
                    continue
                for j in np.arange(abs(l - s), l + s + 1):
                    for f in np.arange(abs(j - i_c), j + i_c + 1):
                        state = RydbergStateSQDTAlkali(species, n=n, l=l, j=float(j), f=float(f))
                        self.states.append(state)


class BasisSQDTAlkalineLS(BasisBase[RydbergStateSQDTAlkalineLS]):
    def __init__(self, species: str, n_min: int = 1, n_max: int | None = None) -> None:
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        i_c = self.species.i_c if self.species.i_c is not None else 0

        self.states = []
        for n in range(n_min, n_max + 1):
            for l in range(n):
                for s_tot in [0, 1]:
                    if not self.species.is_allowed_shell(n, l, s_tot):
                        continue
                    for j_tot in range(abs(l - s_tot), l + s_tot + 1):
                        for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                            state = RydbergStateSQDTAlkalineLS(
                                species, n=n, l=l, s_tot=s_tot, j_tot=j_tot, f_tot=float(f_tot)
                            )
                            self.states.append(state)


class BasisSQDTAlkalineJJ(BasisBase[RydbergStateSQDTAlkalineJJ]):
    def __init__(self, species: str, n_min: int = 0, n_max: int | None = None) -> None:
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        i_c = self.species.i_c if self.species.i_c is not None else 0
        j_c = 0.5
        s_r = 0.5
        self.states = []
        for n in range(n_min, n_max + 1):
            for l_r in range(n):
                if self.species.is_allowed_shell(n, l_r, 0) != self.species.is_allowed_shell(n, l_r, 1):
                    logger.warning(
                        "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                        "In JJ coupling the state does not exist, thus skipping this shell",
                        *(l_r, n),
                    )
                if not all(self.species.is_allowed_shell(n, l_r, s_tot) for s_tot in [0, 1]):
                    continue
                for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                    for j_tot in range(int(abs(j_r - j_c)), int(j_r + j_c + 1)):
                        for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                            state = RydbergStateSQDTAlkalineJJ(
                                species, n=n, l=l_r, j_r=float(j_r), j_tot=j_tot, f_tot=float(f_tot)
                            )
                            self.states.append(state)


class BasisSQDTAlkalineFJ(BasisBase[RydbergStateSQDTAlkalineFJ]):
    def __init__(self, species: str, n_min: int = 0, n_max: int | None = None) -> None:
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        i_c = self.species.i_c if self.species.i_c is not None else 0
        j_c = 0.5
        s_r = 0.5
        self.states = []
        for n in range(n_min, n_max + 1):
            for l_r in range(n):
                if self.species.is_allowed_shell(n, l_r, 0) != self.species.is_allowed_shell(n, l_r, 1):
                    logger.warning(
                        "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                        "In FJ coupling the state does not exist, thus skipping this shell",
                        *(l_r, n),
                    )
                if not all(self.species.is_allowed_shell(n, l_r, s_tot) for s_tot in [0, 1]):
                    continue
                for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                    for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                        for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                            state = RydbergStateSQDTAlkalineFJ(
                                species, n=n, l=l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(f_tot)
                            )
                            self.states.append(state)
