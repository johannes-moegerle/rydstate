from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular import AngularKetFJ
from rydstate.angular.utils import NotSet, is_unknown
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.linalg import calc_nullvector, find_roots
from rydstate.radial.radial_ket import RadialKet
from rydstate.rydberg_state import RydbergKet, RydbergStateMQDT
from rydstate.species import MQDT, FModelSQDT, Potential, PotentialDummy, get_mqdt, get_potential_class

if TYPE_CHECKING:
    from rydstate.species import FModel


logger = logging.getLogger(__name__)


class BasisMQDT(BasisBase[RydbergStateMQDT]):
    states: list[RydbergStateMQDT]

    def __init__(
        self,
        species: str,
        nu: tuple[float, float],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        include_sqdt_fallback_models: bool = True,
        # potential and mqdt parameters
        mqdt: MQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        """Initialize the MQDT basis.

        Args:
            species: Atomic species.
            nu: Tuple of (nu_min, nu_max) for the effective principal quantum number.
            f_tot: Optional tuple of (f_tot_min, f_tot_max) for the total angular momentum.
                Default None, include all f_tot values.
            l_r: Optional tuple of (l_r_min, l_r_max) for the Rydberg electron orbital angular momentum.
                This is used to filter models, which include at least one channel with
                l_c=0 and l_r in the specified range.
                Default None, include all models.
            m: Optional tuple of (m_min, m_max) for the magnetic quantum number range.
                Default NotSet, only include states with m=NotSet.
                If m is given as None, include all allowed m values.
            include_sqdt_fallback_models: Whether to include simple SQDT models (with zero quantum defects) as fallback
                for states, for which no MQDT models are available.
            mqdt: The MQDT data to use for the states.
                Either an instance of an MQDT class
                or a string representing the tag of the MQDT class to use.
            potential_class: The potential class to use for the radial ket.
                Either a a potential class
                or a string representing the tag of the potential class to use.

        """
        super().__init__(species)
        self.mqdt = mqdt if isinstance(mqdt, MQDT) else get_mqdt(species, tag=mqdt)

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        else:
            self.potential_class = get_potential_class(species, tag=potential_class)

        # the maximum l_r is limited by the maximum nu, because l_r < n for bound states
        # and for high l_r the quantum defects are 0, so n = nu
        max_l_r = int(nu[1])
        self._init_models(max_l_r, f_tot, l_r, include_sqdt_fallback_models=include_sqdt_fallback_models)
        self._init_states(nu, m)

    def _init_models(
        self,
        max_l_r: int,
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
        *,
        include_sqdt_fallback_models: bool,
    ) -> None:
        self.models: list[FModel] = []
        s_r = 0.5
        i_c = self.element_properties.i_c
        s_c = self.element_properties.s_c
        j_c = s_c
        for l_r in range(max_l_r + 1):
            if not is_allowed_qn(l_r_range, l_r):
                continue
            for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                    for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                        if not is_allowed_qn(f_tot_range, f_tot):
                            continue
                        channel = AngularKetFJ(
                            l_r=l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(f_tot), species=self.species
                        )
                        for model in self.mqdt.get_mqdt_models(channel):
                            if model in self.models:
                                continue
                            if isinstance(model, FModelSQDT) and not include_sqdt_fallback_models:
                                continue
                            self.models.append(model)

    def _init_states(
        self,
        nu_range: tuple[float, float],
        m_range: tuple[float, float] | None | NotSet,
    ) -> None:
        logger.debug("Calculating MQDT states...")
        self.states = []
        for model in self.models:
            logger.debug("  calculating states for model %s with nu_range=%s", model.name, nu_range)
            _states = get_mqdt_states_from_fmodel(model, nu_range, m_range, self.potential_class)
            if len(_states) == 0:
                logger.debug("  no states found for model %s", model.name)
            else:
                logger.debug(
                    "  model %s: nu_min=%s, nu_max=%s, total states=%d",
                    model.name,
                    _states[0].nu,
                    _states[-1].nu,
                    len(_states),
                )
            self.states.extend(_states)

        self.states.sort(key=lambda state: state.nu)


def get_mqdt_states_from_fmodel(
    model: FModel,
    nu_range: tuple[float, float],
    m_range: tuple[float, float] | None | NotSet,
    potential_class: type[Potential],
    *,
    overwrite_model_limits: bool = False,
) -> list[RydbergStateMQDT]:
    """Calculate MQDT states from an FModel by finding zeros of det(M-matrix).

    Args:
        model: The MQDT model to compute states for.
        nu_range: Tuple of (nu_min, nu_max) for the search range.
        m_range: Tuple of (m_min, m_max) for the magnetic quantum number range.
            NotSet will only include states with m=NotSet.
        potential_class: The potential class to use for the radial ket.
        overwrite_model_limits: If True, use nu_min/nu_max directly without clamping to
            the model's validity range.

    Returns:
        List of :class:`RydbergStateMQDT` objects, one per root of det(M).

    """
    nu_min, nu_max = nu_range
    if overwrite_model_limits:
        if nu_min is None or nu_max is None:
            raise ValueError("nu_min and nu_max must be given if overwrite_model_limits is True.")
    else:
        nu_min = model.nu_min if nu_min is None else max(nu_min, model.nu_min)
        nu_max = model.nu_max if nu_max is None else min(nu_max, model.nu_max)
    if np.isinf(nu_max):
        raise ValueError("nu_max must be finite to calculate MQDT states.")

    nu_list = find_roots(model.calc_det_scaled_m_matrix, nu_min, nu_max)
    if len(nu_list) == 0:
        logger.warning(
            "No MQDT states found in the range nu_min=%s, nu_max=%s for model %s", nu_min, nu_max, model.name
        )
        return []

    states: list[RydbergStateMQDT] = []
    for nu in nu_list:
        det_mmat = model.calc_det_m_matrix(nu)
        if abs(det_mmat) > 1e-6:
            # this can happen, because we use the scaled M-matrix to find roots ...
            logger.warning(
                "%s: Found a root of det(M) that is not actually a root (nu=%s, det(M)=%s). "
                "Keeping this state, but you should treat it with caution.",
                *(model.full_name, nu, det_mmat),
            )

        nuis = model.calc_channel_nuis(nu)
        coefficients = calc_nullvector(model.calc_m_matrix(nu))
        coefficients = np.array(
            [coeff * (nui ** (3 / 2)) / np.cos(np.pi * nui) for coeff, nui in zip(coefficients, nuis, strict=True)]
        )
        coefficients /= np.linalg.norm(coefficients)
        arg_max = np.argmax(np.abs(coefficients))
        coefficients *= np.sign(coefficients[arg_max])

        radial_kets: list[RadialKet] = []
        for nui, angular_ket in zip(nuis, model.outer_channels, strict=True):
            if not is_unknown(angular_ket.l_r):
                potential = potential_class(angular_ket.l_r)
            else:
                potential = PotentialDummy(model.species, angular_ket.l_r)
            radial_kets.append(RadialKet(float(nui), potential, sign_convention="positive_at_outer_bound"))

        energy_au = model.calc_energy_au(nu)
        for m in get_m_range(model.f_tot, m_range):
            rydberg_kets = [
                RydbergKet(angular_ket.replace_m(m), radial_ket)
                for angular_ket, radial_ket in zip(model.outer_channels, radial_kets, strict=True)
            ]
            states.append(
                RydbergStateMQDT(
                    model.species,
                    coefficients,
                    rydberg_kets,
                    nu=nu,
                    energy_au=energy_au,
                    model=model,
                    potential_class=potential_class,
                )
            )

    return states
