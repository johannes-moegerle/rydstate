from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rydstate.angular.angular_ket import julia_qn_to_dict
from rydstate.basis.basis_base import BasisBase
from rydstate.rydberg.rydberg_mqdt import RydbergStateMQDT
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
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

FMODEL_MAX_L = {"Sr87": 2, "Sr88": 2, "Yb171": 4, "Yb173": 1, "Yb174": 4}


class BasisMQDT(BasisBase[RydbergStateMQDT[Any]]):
    def __init__(
        self,
        species: str | SpeciesObject,
        n_min: int = 0,
        n_max: int | None = None,
        *,
        skip_high_l: bool = True,
        model_names: list[str] | None = None,
    ) -> None:
        super().__init__(species)

        if not USE_JULIACALL:
            raise ImportError("JuliaCall or the MQDT Julia package is not available.")

        try:
            self.jl_species = getattr(jl.MQDT, self.species.name)
            parameters = self.jl_species.PARA
        except AttributeError as e:
            raise ValueError(f"Species '{species}' is not supported in the MQDT Julia package.") from e

        # TODO use n_min and n_max of the different models

        if n_max is None:
            raise ValueError("n_max must be given")

        # initialize Wigner symbol calculation
        if skip_high_l:
            jl.CGcoefficient.wigner_init_float(5, "Jmax", 9)
        else:
            jl.CGcoefficient.wigner_init_float(n_max - 1, "Jmax", 9)

        logger.debug("Calculating low l MQDT states...")

        jl_species_attr_names = [str(name) for name in jl.seval(f"names(MQDT.{self.species.name}, all=true)")]
        self.models = {name: getattr(self.jl_species, name) for name in jl_species_attr_names}
        self.models = {k: v for k, v in self.models.items() if str(v).startswith("fModel")}
        if model_names is not None:
            self.models = {k: v for k, v in self.models.items() if k in model_names}

        if skip_high_l:
            logger.debug("Skipping high l states.")
        else:
            logger.debug("Calculating high l SQDT states...")
            l_start = FMODEL_MAX_L[self.species.name] + 1
            high_l_models = {
                f"high_l_{l_ryd}": jl.single_channel_models(species, l_ryd, parameters)
                for l_ryd in range(l_start, n_max)
            }
            self.models.update(high_l_models)

        model_names = list(self.models.keys())
        jl_states = {name: jl.eigenstates(n_min, n_max, model, parameters) for name, model in self.models.items()}
        _models_vector = convert(jl.Vector, [self.models[name] for name in model_names])
        _jl_states_vector = convert(jl.Vector, [jl_states[name] for name in model_names])
        jl_basis = jl.basisarray(_jl_states_vector, _models_vector)

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
