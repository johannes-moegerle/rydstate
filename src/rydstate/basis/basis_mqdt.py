from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg import RydbergStateMQDT
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT
from rydstate.species import FModel, FModelSQDT, SpeciesObjectMQDT
from rydstate.utils.linalg import calc_nullvector, find_roots

if TYPE_CHECKING:
    from rydstate.species import FModel

logger = logging.getLogger(__name__)


class BasisMQDT(BasisBase[RydbergStateMQDT]):
    """Basis set of MQDT Rydberg states for a given species over a range of effective principal quantum numbers."""

    def __init__(  # noqa: C901
        self,
        species: str | SpeciesObjectMQDT,
        nu: tuple[float, float],
        f_tot: float | tuple[float, float] | None = None,
        l_r: int | tuple[int, int] | None = None,
        *,
        skip_high_l: bool = True,
        n_min_high_l: int = 0,
    ) -> None:
        self.species = SpeciesObjectMQDT.from_name(species) if isinstance(species, str) else species

        self.models: list[FModel] = []
        i_c, j_c, s_r = self.species.i_c, 0.5, 0.5
        for _l_r in range(int(nu[1]) + 1):
            if not is_allowed_qn(l_r, _l_r):
                continue
            for j_r in np.arange(abs(_l_r - s_r), _l_r + s_r + 1):
                for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                    for _f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                        if not is_allowed_qn(f_tot, _f_tot):
                            continue
                        channel = AngularKetFJ(
                            l_r=_l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(_f_tot), species=self.species
                        )
                        self.models.extend(self.species.get_mqdt_models(channel))
        self.models = list(set(self.models))  # remove duplicates

        logger.debug("Calculating MQDT states...")
        self.states: list[RydbergStateMQDT] = []
        for model in self.models:
            _nu_min = nu[0]
            if isinstance(model, FModelSQDT):
                if skip_high_l:
                    continue
                _nu_min = max(_nu_min, n_min_high_l)
            logger.debug("  calculating states for model %s with nu_min=%s, nu_max=%s", model.name, _nu_min, nu[1])
            _states = get_mqdt_states_from_fmodel(model, _nu_min, nu[1])
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


def is_allowed_qn(qn_range: float | tuple[float, float] | None, qn: float, delta: float = 1e-6) -> bool:
    if qn_range is None:
        return True
    if isinstance(qn_range, (int, float)) or np.isscalar(qn_range):
        return abs(qn - qn_range) <= delta  # type: ignore [operator]
    if isinstance(qn_range, tuple) or (isinstance(qn_range, list) and len(qn_range) == 2):
        return qn_range[0] - delta <= qn <= qn_range[1] + delta
    raise ValueError(f"Invalid qn_range: {qn_range}. Must be a float or a tuple of two floats.")


def get_mqdt_states_from_fmodel(
    model: FModel,
    nu_min: float | None = None,
    nu_max: float | None = None,
    *,
    overwrite_model_limits: bool = False,
) -> list[RydbergStateMQDT]:
    """Calculate MQDT states from an FModel by finding zeros of det(M-matrix).

    Args:
        model: The MQDT model to compute states for.
        nu_min: Lower bound of the search range.  Defaults to ``model.nu_min``.
        nu_max: Upper bound of the search range.  Defaults to ``model.nu_max``.
        overwrite_model_limits: If True, use nu_min/nu_max directly without clamping to
            the model's validity range.  Both nu_min and nu_max must be provided.

    Returns:
        List of :class:`RydbergStateMQDT` objects, one per root of det(M).

    """
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
        if coefficients is None:
            logger.warning("Failed to calculate nullvector for nu=%s, skipping this state.", nu)
            continue
        coefficients = np.array(
            [coeff * (nui ** (3 / 2)) / np.cos(np.pi * nui) for coeff, nui in zip(coefficients, nuis, strict=True)]
        )
        coefficients /= np.linalg.norm(coefficients)
        arg_max = np.argmax(np.abs(coefficients))
        coefficients *= np.sign(coefficients[arg_max])

        sqdt_states: list[RydbergStateSQDT[AngularKetFJ[Any]]] = []
        for nui, angular_ket in zip(nuis, model.outer_channels, strict=True):
            sqdt_species = model.species_name.replace("_mqdt", "")
            sqdt_states.append(RydbergStateSQDT.from_angular_ket(sqdt_species, angular_ket, nu=float(nui)))

        states.append(RydbergStateMQDT(coefficients, sqdt_states, nu=nu))

    return states
