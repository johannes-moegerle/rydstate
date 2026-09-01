from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetLS
from rydstate.angular.utils import AllKnown
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.rydberg_state import RydbergStateSQDT
from rydstate.species import get_sqdt
from rydstate.species.sqdt import SQDT

if TYPE_CHECKING:
    from rydstate.species.potential import Potential


logger = logging.getLogger(__name__)


class BasisSQDT(BasisBase[RydbergStateSQDT[AngularKetLS[AllKnown]]]):
    states: list[RydbergStateSQDT[AngularKetLS[AllKnown]]]
    _channels: list[AngularKetLS[AllKnown]]

    def __init__(
        self,
        species: str,
        n: tuple[int, int],
        *,
        l_r: tuple[int, int] | None = None,
        f_tot: tuple[float, float] | None = None,
        m: tuple[float, float] | NotSet | None = NotSet,
        # potential and sqdt parameters
        potential_class: type[Potential] | str | None = None,
        sqdt: SQDT | str | None = None,
    ) -> None:
        """Initialize the SQDT basis.

        Args:
            species: Atomic species.
            n: Tuple of (n_min, n_max) for the principal quantum number.
            l_r: Optional tuple of (l_r_min, l_r_max) for the Rydberg electron orbital angular momentum.
                Default None, include all l_r values.
            f_tot: Optional tuple of (f_tot_min, f_tot_max) for the total angular momentum.
                Default None, include all f_tot values.
            m: Optional tuple of (m_min, m_max) for the magnetic quantum number.
                If None, all m values are included.
                Default NotSet, m is not specified and will be set to NotSet for all states.
            potential_class: The potential class to use for the radial ket.
                Either a a potential class
                or a string representing the tag of the potential class to use.
            sqdt: The SQDT data to use for the states.
                Either an instance of an SQDT class
                or a string representing the tag of the SQDT class to use.

        """
        super().__init__(species, potential_class)
        self.sqdt = sqdt if isinstance(sqdt, SQDT) else get_sqdt(species, tag=sqdt)

        if l_r is None:
            l_r = (0, n[1] - 1)
        elif isinstance(l_r, Sequence) and len(l_r) == 2:
            l_r = (max(l_r[0], 0), min(l_r[1], n[1] - 1))
        else:
            raise ValueError("Invalid qn_range: l_r. Must be None or a tuple of two numbers.")

        self._init_channels(l_r, f_tot)
        self._init_states(n, m)

    def _init_channels(self, l_r_range: tuple[int, int], f_tot_range: tuple[float, float] | None) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        channels = []

        for l_r in range(l_r_range[0], l_r_range[1] + 1):
            for s_tot in s_tot_list:
                for j_tot in np.arange(abs(l_r - s_tot), l_r + s_tot + 1):
                    for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                        if not is_allowed_qn(f_tot_range, f_tot):
                            continue
                        angular = AngularKetLS(
                            l_r=l_r, s_tot=s_tot, j_tot=j_tot, f_tot=f_tot, m=NotSet, species=self.species
                        )
                        channels.append(angular)

        self._channels = channels

    def _init_states(
        self,
        n_range: tuple[int, int],
        m_range: tuple[float, float] | NotSet | None,
    ) -> None:
        states = []

        for angular in self._channels:
            angular_m_list = [angular.replace_m(m) for m in get_m_range(angular.f_tot, m_range)]

            for n in range(max(n_range[0], angular.l_r + 1), n_range[1] + 1):
                if not self.element_properties.is_allowed_shell(n, angular.l_r, angular.s_tot):
                    continue
                for angular_m in angular_m_list:
                    state = RydbergStateSQDT(
                        self.species,
                        n=n,
                        angular=angular_m,
                        potential_class=self.potential_class,
                        sqdt=self.sqdt,
                    )
                    states.append(state)

        # sort by energy (and not by nu, since nu is not always defined, see RydbergStateSQDT.nu)
        states.sort(key=lambda state: state.get_energy("a.u."))
        self.states = states
