from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular.angular_ket import julia_qn_to_dict
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg.rydberg_mqdt import RydbergStateMQDT
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
    from juliacall import (
        JuliaError,
        Main as jl,  # noqa: N813
        convert,
    )

    from rydstate.species import SpeciesObject

logger = logging.getLogger(__name__)

try:
    USE_JULIACALL = True
    from juliacall import (
        JuliaError,
        Main as jl,  # noqa: N813
        convert,
    )
except ImportError:
    USE_JULIACALL = False


if USE_JULIACALL:
    try:
        jl.seval("using MQDT")
        jl.seval("using CGcoefficient")
    except JuliaError:
        logger.exception("Failed to load Julia MQDT or CGcoefficient package")
        USE_JULIACALL = False


class BasisMQDT(BasisBase[RydbergStateMQDT[Any]]):
    def __init__(
        self,
        species: str | SpeciesObject,
        n_min: int = 0,
        n_max: int | None = None,
        *,
        skip_high_l: bool = True,
    ) -> None:
        super().__init__(species)

        if n_max is None:
            raise ValueError("n_max must be given")

        if not USE_JULIACALL:
            raise ImportError("JuliaCall or the MQDT Julia package is not available.")

        # initialize Wigner symbol calculation
        if skip_high_l:
            jl.CGcoefficient.wigner_init_float(10, "Jmax", 9)
        else:
            jl.CGcoefficient.wigner_init_float(n_max - 1, "Jmax", 9)

        jl_species = jl.Symbol(self.species.name)
        parameters = jl.MQDT.get_parameters(jl_species)

        self.models = []
        i_c = self.species.i_c if self.species.i_c is not None else 0
        for l in range(n_max):
            jtot_min = min(l, abs(l - 1))
            jtot_max = l + 1
            for f_tot in np.arange(abs(jtot_min - i_c), jtot_max + i_c + 1):
                models = jl.MQDT.get_fmodels(jl_species, l, f_tot)
                self.models.extend(models)

        n_min_high_l = 25

        logger.debug("Calculating MQDT states...")
        jl_states = []
        for model in self.models:
            _n_min = n_min
            if model.name.startswith("SQDT"):
                if skip_high_l:
                    continue
                _n_min = n_min_high_l

            logger.debug(f"{model.name}:")
            states = jl.MQDT.eigenstates(_n_min, n_max, model, parameters)
            jl_states.append(states)
            logger.debug(f"  found nu_min={min(states.n)}, nu_max={max(states.n)}, total states={len(states.n)}")

        jl_basis = jl.basisarray(convert(jl.Vector, jl_states), convert(jl.Vector, self.models))

        logger.debug("Generated state table with %d states", len(jl_basis.states))

        self.states = []
        for jl_state in jl_basis.states:
            coeffs = jl_state.coeff
            nus = jl_state.nu
            nu_energy = jl_state.energy
            qns = jl_state.channels.i
            qns = [julia_qn_to_dict(qn) for qn in qns]

            sqdt_states = [RydbergStateSQDT(species, nu=nu, **qn) for nu, qn in zip(nus, qns)]
            # check angular and radial are created correctly
            [(s.angular, s.radial) for s in sqdt_states]

            mqdt_state = RydbergStateMQDT(coeffs, sqdt_states, nu_energy=nu_energy, warn_if_not_normalized=False)
            self.states.append(mqdt_state)
