from __future__ import annotations

import logging
from typing import TypeVar

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg import (
    RydbergStateSQDT,
    RydbergStateSQDTAlkali,
    RydbergStateSQDTAlkalineFJ,
    RydbergStateSQDTAlkalineJJ,
    RydbergStateSQDTAlkalineLS,
)
from rydstate.species import SpeciesObjectSQDT

logger = logging.getLogger(__name__)

_RydbergStateSQDT = TypeVar("_RydbergStateSQDT", bound=RydbergStateSQDT)


class BasisSQDT(BasisBase[_RydbergStateSQDT]):
    def _get_m_range(self, m: tuple[float, float] | None | NotSet, f_tot: float | np.floating) -> list[NotSet | float]:
        if is_not_set(m):
            return [NotSet]
        if m is None:
            m = (-np.inf, np.inf)

        m_min = max(-f_tot, m[0])
        m_max = min(f_tot, m[1])
        return [float(_m) for _m in np.arange(m_min, m_max + 1)]


class BasisSQDTAlkali(BasisSQDT[RydbergStateSQDTAlkali]):
    def __init__(
        self, species: str | SpeciesObjectSQDT, n: tuple[int, int], m: tuple[float, float] | None | NotSet = NotSet
    ) -> None:
        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        self.species = species

        i_c = self.species.i_c_number
        self.states = []

        _s = 1 / 2
        for _n in range(n[0], n[1] + 1):
            for _l in range(_n):
                if not self.species.is_allowed_shell(_n, _l, _s):
                    continue
                for _j in np.arange(abs(_l - _s), _l + _s + 1):
                    for _f in np.arange(abs(_j - i_c), _j + i_c + 1):
                        for _m in self._get_m_range(m, _f):
                            state = RydbergStateSQDTAlkali(species, n=_n, l=_l, j=float(_j), f=float(_f), m=_m)
                            self.states.append(state)


class BasisSQDTAlkalineLS(BasisSQDT[RydbergStateSQDTAlkalineLS]):
    def __init__(
        self, species: str | SpeciesObjectSQDT, n: tuple[int, int], m: tuple[float, float] | None | NotSet = NotSet
    ) -> None:
        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        self.species = species

        i_c = self.species.i_c_number
        self.states = []

        for _n in range(n[0], n[1] + 1):
            for _l in range(_n):
                for _s_tot in [0, 1]:
                    if not self.species.is_allowed_shell(_n, _l, _s_tot):
                        continue
                    for _j_tot in range(abs(_l - _s_tot), _l + _s_tot + 1):
                        for _f_tot in np.arange(abs(_j_tot - i_c), _j_tot + i_c + 1):
                            for _m in self._get_m_range(m, _f_tot):
                                state = RydbergStateSQDTAlkalineLS(
                                    species, n=_n, l=_l, s_tot=_s_tot, j_tot=_j_tot, f_tot=float(_f_tot), m=_m
                                )
                                self.states.append(state)


class BasisSQDTAlkalineJJ(BasisSQDT[RydbergStateSQDTAlkalineJJ]):
    def __init__(
        self, species: str | SpeciesObjectSQDT, n: tuple[int, int], m: tuple[float, float] | None | NotSet = NotSet
    ) -> None:
        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        self.species = species

        i_c = self.species.i_c_number
        self.states = []

        _j_c = 0.5
        _s_r = 0.5
        for _n in range(n[0], n[1] + 1):
            for _l_r in range(_n):
                allowed = [self.species.is_allowed_shell(_n, _l_r, s) for s in [0, 1]]
                if not all(allowed):
                    if any(allowed):
                        logger.warning(
                            "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                            "In JJ coupling the state does not exist, thus skipping this shell",
                            *(_l_r, _n),
                        )
                    continue
                for _j_r in np.arange(abs(_l_r - _s_r), _l_r + _s_r + 1):
                    for _j_tot in range(int(abs(_j_r - _j_c)), int(_j_r + _j_c + 1)):
                        for _f_tot in np.arange(abs(_j_tot - i_c), _j_tot + i_c + 1):
                            for _m in self._get_m_range(m, _f_tot):
                                state = RydbergStateSQDTAlkalineJJ(
                                    species, n=_n, l=_l_r, j_r=float(_j_r), j_tot=_j_tot, f_tot=float(_f_tot), m=_m
                                )
                                self.states.append(state)


class BasisSQDTAlkalineFJ(BasisSQDT[RydbergStateSQDTAlkalineFJ]):
    def __init__(
        self, species: str | SpeciesObjectSQDT, n: tuple[int, int], m: tuple[float, float] | None | NotSet = NotSet
    ) -> None:
        if isinstance(species, str):
            species = SpeciesObjectSQDT.from_name(species)
        self.species = species

        i_c = self.species.i_c_number
        self.states = []

        _j_c = 0.5
        _s_r = 0.5
        for _n in range(n[0], n[1] + 1):
            for _l_r in range(_n):
                allowed = [self.species.is_allowed_shell(_n, _l_r, s) for s in [0, 1]]
                if not all(allowed):
                    if any(allowed):
                        logger.warning(
                            "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                            "In FJ coupling the state does not exist, thus skipping this shell",
                            *(_l_r, _n),
                        )
                    continue
                for _j_r in np.arange(abs(_l_r - _s_r), _l_r + _s_r + 1):
                    for _f_c in np.arange(abs(_j_c - i_c), _j_c + i_c + 1):
                        for _f_tot in np.arange(abs(_f_c - _j_r), _f_c + _j_r + 1):
                            for _m in self._get_m_range(m, _f_tot):
                                state = RydbergStateSQDTAlkalineFJ(
                                    species, n=_n, l=_l_r, j_r=float(_j_r), f_c=float(_f_c), f_tot=float(_f_tot), m=_m
                                )
                                self.states.append(state)
