from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetFJ
from rydstate.angular.utils import is_unknown
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.basis_sqdt import BasisSQDT
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.rydberg_state.rydberg_sqdt import RydbergStateOSQDT
from rydstate.species.mqdt import MQDT, get_mqdt
from rydstate.species.osqdt import OSQDT, NoOSQDTStateError

if TYPE_CHECKING:
    from rydstate.angular.utils import AllKnown
    from rydstate.species.potential import Potential


logger = logging.getLogger(__name__)


class BasisOSQDT(BasisSQDT):
    r"""Basis of single channel states, using Outer channel Single Quantum Defect Theory (OSQDT).

    The states of this class are not meant to be used directly.
    They are rather used for comparison and didactic purposes.

    Like :class:`~rydstate.basis.BasisSQDT`, this basis is constructed by deterministically looping over
    all possible quantum number configurations and a range of principal quantum numbers n,
    resulting in exactly one state per channel and n.
    Since the states are requested in a range of nu, the n of each channel are looped over generously
    and the states outside the requested nu range are discarded afterwards.
    In contrast to :class:`~rydstate.basis.BasisSQDT`, the quantum defects are not taken from the
    :class:`~rydstate.species.SQDT` data, but from the diagonal elements of the K-matrix of the
    :class:`~rydstate.species.MQDT` models in the collision (outer) channel frame.

    This is the same condition one obtains from :class:`~rydstate.basis.BasisMQDT` when the coupling
    between the outer channels is switched off, i.e. when the off-diagonal elements of the K-matrix are neglected.
    """

    states: list[RydbergStateOSQDT]  # type: ignore [assignment]
    _channels: list[AngularKetFJ[AllKnown]]  # type: ignore [assignment]

    def __init__(
        self,
        species: str,
        nu: tuple[float, float],
        *,
        l_r: tuple[int, int] | None = None,
        f_tot: tuple[float, float] | None = None,
        m: tuple[float, float] | NotSet | None = NotSet,
        # potential and mqdt parameters
        potential_class: type[Potential] | str | None = None,
        mqdt: MQDT | str | None = None,
    ) -> None:
        """Initialize the OSQDT basis.

        Args:
            species: Atomic species.
            nu: Tuple of (nu_min, nu_max) for the effective principal quantum number nu,
                which is defined with reference to the reference ionization threshold of the MQDT models
                (i.e. the same nu as for :class:`~rydstate.basis.BasisMQDT`).
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
            mqdt: The MQDT data to use for the states.
                Either an instance of an MQDT class
                or a string representing the tag of the MQDT class to use.

        """
        # skip BasisSQDT.__init__, since this basis uses MQDT models instead of SQDT data
        BasisBase.__init__(self, species, potential_class)
        self.mqdt = mqdt if isinstance(mqdt, MQDT) else get_mqdt(species, tag=mqdt)

        if not 0 <= nu[0] < nu[1] < math.inf:
            raise ValueError(f"nu must be a tuple (nu_min, nu_max) with 0 <= nu_min < nu_max < inf, but got {nu}.")

        if l_r is None:
            # the maximum l_r is limited by the maximum nu, because l_r < n for bound states
            # and for high l_r the quantum defects are 0, so n = nu
            l_r = (0, int(nu[1]))

        self._init_channels(l_r, f_tot)
        self._init_states(nu, m)

    def _init_channels(self, l_r_range: tuple[int, int], f_tot_range: tuple[float, float] | None) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = self.element_properties.s_c
        j_c = s_c

        channels = []

        for l_r in range(l_r_range[0], l_r_range[1] + 1):
            for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                    for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                        if not is_allowed_qn(f_tot_range, f_tot):
                            continue

                        angular = AngularKetFJ(l_r=l_r, j_r=j_r, f_c=f_c, f_tot=f_tot, m=NotSet, species=self.species)
                        channels.append(angular)

        # also add channels l_c > 0 or with unknown quantum numbers given in the MQDT models
        for model in self.mqdt.models:
            for channel in model.fj_channels:
                if (
                    channel in channels
                    or not is_allowed_qn(f_tot_range, channel.f_tot)
                    or (not is_unknown(channel.l_r) and not is_allowed_qn(l_r_range, channel.l_r))
                ):
                    continue
                channels.append(channel)

        self._channels = channels

    def _init_states(
        self,
        nu_range: tuple[float, float],
        m_range: tuple[float, float] | NotSet | None,
    ) -> None:
        states = []

        for angular in self._channels:
            s_tot = angular.get_qn("s_tot", allow_unknown=True)
            osqdt = OSQDT(tuple(self.mqdt.get_mqdt_models(angular)), angular, nu_range=nu_range)

            n_min, n_max = osqdt.n_min, osqdt.n_max
            for n in range(n_min, n_max + 1):
                if not self.element_properties.is_allowed_shell(n, angular.l_r, s_tot):
                    continue

                try:
                    osqdt.calc_nui(n, angular)
                except NoOSQDTStateError as err:
                    logger.warning(
                        "No OSQDT state found for intermediate n=%d in channel %s. "
                        "This might be due to a missing model for this channel and nu range. Error: %s",
                        *(n, angular, err),
                    )
                    continue

                for m in get_m_range(angular.f_tot, m_range):
                    state = RydbergStateOSQDT(
                        self.species,
                        n=n,
                        angular=angular.replace_m(m),
                        sqdt=osqdt,
                        potential_class=self.potential_class,
                    )
                    states.append(state)

        states.sort(key=lambda state: state.get_energy("a.u."))
        # discard the states outside the requested nu range, which were only needed for the labeling with n
        self.states = [state for state in states if nu_range[0] <= state.nu <= nu_range[1]]
