from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetFJ
from rydstate.angular.utils import (
    InvalidQuantumNumbersError,
    Unknown,
    get_possible_quantum_number_values,
    is_unknown,
    try_trivial_spin_addition,
)
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.radial.radial_ket import RadialKet
from rydstate.rydberg_state.rydberg_ket import RydbergKet
from rydstate.rydberg_state.rydberg_mqdt import RydbergStateMQDT
from rydstate.species.mqdt import MQDT, get_mqdt
from rydstate.species.potential import Potential, PotentialDummy, get_potential_class

logger = logging.getLogger(__name__)


class BasisMSQDT(BasisBase):
    states: list[RydbergStateMQDT]

    def __init__(
        self,
        species: str,
        n: tuple[int, int],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        # potential and sqdt parameters
        mqdt: MQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        super().__init__(species)
        self.mqdt = mqdt if isinstance(mqdt, MQDT) else get_mqdt(species, tag=mqdt)

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        else:
            self.potential_class = get_potential_class(species, tag=potential_class)

        self._init_states(n, f_tot, l_r, m)

    def _init_states(
        self,
        n_range: tuple[int, int],
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
        m_range: tuple[float, float] | None | NotSet,
    ) -> None:
        self.coupling_scheme = "FJ"
        self.states = []

        for core_ket in self.mqdt.ionization_threshold_dict:
            if core_ket.contains_unknown:
                # we will handle these below
                continue
            logger.info("Generating states for core ket: %s", core_ket)

            for n in range(n_range[0], n_range[1] + 1):
                for l_r in range(n):
                    if not is_allowed_qn(l_r_range, l_r):
                        continue
                    self._add_states_fj(
                        n,
                        l_r,
                        f_tot_range,
                        m_range,
                        l_c=core_ket.l_c,
                        j_c=core_ket.j_c,
                        f_c=core_ket.f_c,
                        allow_unknown=True,
                    )

        # add all addition series, which are defined in the mqdt parameters but have dummy core kets
        for fmodel in self.mqdt.models:
            for angular_ket in fmodel.outer_channels:
                if not angular_ket.contains_unknown:
                    # handled above
                    continue
                for n in range(n_range[0], n_range[1] + 1):
                    self._add_states_fj(
                        n,
                        angular_ket.l_r,
                        f_tot_range,
                        m_range,
                        l_c=angular_ket.l_c,
                        j_c=angular_ket.j_c,
                        f_c=angular_ket.f_c,
                        f_tot=angular_ket.f_tot,
                        allow_unknown=True,
                    )

    def _add_states_fj(
        self,
        n: int,
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
        l_c: int | Unknown = 0,
        j_c: float | Unknown = Unknown,
        f_c: float | Unknown = Unknown,
        f_tot: float | Unknown = Unknown,
        allow_unknown: bool = False,
    ) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = 0 if self.element_properties.number_valence_electrons == 1 else 0.5
        j_c = try_trivial_spin_addition(l_c, s_c, j_c)
        f_c = try_trivial_spin_addition(j_c, i_c, f_c)
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        allowed = [self.mqdt.is_allowed_shell(n, l_r, s) for s in s_tot_list]
        if not all(allowed):
            if any(allowed):
                logger.warning(
                    "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                    "In FJ coupling the state does not exist, thus skipping this shell",
                    *(l_r, n),
                )
            return

        for j_r in get_possible_quantum_number_values(l_r, s_r, Unknown):
            for _f_c in get_possible_quantum_number_values(j_c, i_c, f_c):
                for _f_tot in get_possible_quantum_number_values(_f_c, j_r, f_tot):
                    if is_unknown(_f_tot):
                        raise ValueError(
                            "Cannot determine f_tot for BasisMSQDT. Please provide more specific quantum numbers."
                        )
                    if not is_allowed_qn(f_tot_range, _f_tot):
                        continue

                    for m in get_m_range(_f_tot, m_range):
                        try:
                            angular_ket = AngularKetFJ(  # type: ignore [call-overload]
                                l_r=l_r,
                                j_r=j_r,
                                f_c=_f_c,
                                f_tot=_f_tot,
                                m=m,
                                species=self.species,
                                allow_unknown=allow_unknown,
                            )
                        except InvalidQuantumNumbersError:
                            continue
                        self._create_state(n, angular_ket, m)

    def _create_state(self, n: int, angular_ket: AngularKetFJ[Any], m: float | NotSet) -> None:
        if not is_unknown(angular_ket.l_r):
            potential = self.potential_class(angular_ket.l_r)
        else:
            potential = PotentialDummy(self.species, angular_ket.l_r)

        models = self.mqdt.get_mqdt_models(angular_ket)
        possible_states: list[RydbergStateMQDT] = []
        for model in models:
            nu = model.calc_nu(n, angular_ket)

            if not (model.nu_range[0] <= nu <= model.nu_range[1]):
                continue

            nuis = model.calc_channel_nuis(nu)
            nui = next(nui for nui, ket in zip(nuis, model.outer_channels, strict=True) if ket == angular_ket)
            energy_au = model.calc_energy_au(nu)

            radial_ket = RadialKet(float(nui), potential)
            rydberg_ket = RydbergKet(angular_ket.replace_m(m), radial_ket)

            state = RydbergStateMQDT(
                self.species,
                [1],
                [rydberg_ket],
                nu=nu,
                energy_au=energy_au,
                mqdt=self.mqdt,
                potential_class=self.potential_class,
            )
            state.n = n
            possible_states.append(state)

        if len(possible_states) == 0:
            logger.warning(
                "No MQDT states found for n=%s, angular_ket=%s, m=%s. "
                "This can happen if the nu value is outside the range of the MQDT model.",
                *(n, angular_ket, m),
            )
            return
        if len(possible_states) > 1:
            logger.warning(
                "Multiple MQDT states found for n=%s, angular_ket=%s, m=%s. "
                "This can happen if the nu value is close to a root of det(M). "
                "Keeping only the first state, but you should treat them with caution.",
                *(n, angular_ket, m),
            )
        self.states.append(possible_states[0])
