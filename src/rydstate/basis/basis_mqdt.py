from __future__ import annotations

import logging

import numpy as np

from rydstate.angular import AngularKetFJ
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg import RydbergStateMQDT
from rydstate.rydberg.rydberg_mqdt import get_mqdt_states_from_fmodel
from rydstate.species import FModelSQDT, SpeciesObjectMQDT

logger = logging.getLogger(__name__)


class BasisMQDT(BasisBase[RydbergStateMQDT]):
    """Basis set of MQDT Rydberg states for a given species over a range of effective principal quantum numbers."""

    def __init__(  # noqa: C901, PLR0912
        self,
        species: str | SpeciesObjectMQDT,
        nu_min: float = 0,
        nu_max: int | None = None,
        *,
        f_tot: float | tuple[float, float] | None = None,
        skip_high_l: bool = True,
        n_min_high_l: int = 0,
    ) -> None:
        if isinstance(species, str):
            species = SpeciesObjectMQDT.from_name(species)
        self.species = species

        if nu_max is None:
            raise ValueError("nu_max must be given")

        self.models = []
        i_c, j_c, s_r = self.species.i_c, 0.5, 0.5
        for l_r in range(int(nu_max) + 1):
            for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                    for _f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                        if f_tot is not None:
                            if isinstance(f_tot, tuple):
                                if not (f_tot[0] <= _f_tot <= f_tot[1]):
                                    continue
                            elif _f_tot != f_tot:
                                continue
                        channel = AngularKetFJ(
                            l_r=l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(_f_tot), species=species
                        )
                        self.models.extend(species.get_mqdt_models(channel))
        self.models = list(set(self.models))  # remove duplicates

        logger.debug("Calculating MQDT states...")
        self.states: list[RydbergStateMQDT] = []
        for model in self.models:
            _nu_min = nu_min
            if isinstance(model, FModelSQDT):
                if skip_high_l:
                    continue
                _nu_min = n_min_high_l
            logger.debug("  calculating states for model %s with nu_min=%s, nu_max=%s", model.name, _nu_min, nu_max)
            _states = get_mqdt_states_from_fmodel(model, _nu_min, nu_max)
            if len(_states) == 0:
                logger.debug("  no states found for model %s", model.name)
            else:
                logger.debug(
                    "  model %s: nu_min=%s, nu_max=%s, total states=%d",
                    model.name,
                    _states[0].nu_ref,
                    _states[-1].nu_ref,
                    len(_states),
                )
            self.states.extend(_states)
