from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import numpy as np

from rydstate.angular import AngularState
from rydstate.angular.utils import is_dummy_ket
from rydstate.rydberg.rydberg_base import RydbergStateBase
from rydstate.rydberg.rydberg_dummy import RydbergStateSQDTDummy, is_dummy_rydberg_state
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT, RydbergStateSQDTAlkalineFJ
from rydstate.species import FModel
from rydstate.utils.linalg import calc_nullvector

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rydstate.angular import AngularKetFJ
    from rydstate.angular.angular_ket_dummy import AngularKetDummy
    from rydstate.species import FModel
    from rydstate.units import MatrixElementOperator, NDArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergStateBase):
    angular: AngularState[AngularKetFJ | AngularKetDummy]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        coefficients: Sequence[float] | NDArray,
        sqdt_states: Sequence[RydbergStateSQDTAlkalineFJ | RydbergStateSQDTDummy],
        nu_ref: float,
        *,
        warn_if_not_normalized: bool = True,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.sqdt_states = sqdt_states
        self.species = sqdt_states[0].species
        self._nu_ref = nu_ref
        self.angular = AngularState(self.coefficients.tolist(), [ket.angular for ket in sqdt_states])

        if len(coefficients) != len(sqdt_states):
            raise ValueError("Length of coefficients and sqdt_states must be the same.")
        if not all(
            (type(sqdt_state) is type(sqdt_states[0])) or is_dummy_rydberg_state(sqdt_state)
            for sqdt_state in sqdt_states
        ):
            raise ValueError("All sqdt_states must be of the same type.")
        if not all((sqdt_state.species is sqdt_states[0].species) for sqdt_state in sqdt_states):
            raise ValueError("All sqdt_states must be of the same species.")
        if len(set(sqdt_states)) != len(sqdt_states):
            raise ValueError("RydbergStateMQDT initialized with duplicate sqdt_states.")
        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning(
                "RydbergStateMQDT initialized with non-normalized coefficients "
                "(norm=%s, coefficients=%s, sqdt_states=%s)",
                self.norm,
                coefficients,
                sqdt_states,
            )
        if self.norm > 1:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, RydbergStateSQDTAlkalineFJ | RydbergStateSQDTDummy]]:
        return zip(self.coefficients, self.sqdt_states, strict=True).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!r}" for coeff, sqdt_state in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!s}" for coeff, sqdt_state in self]
        return f"{', '.join(terms)}"

    @property
    def nu_ref(self) -> float:
        return self._nu_ref

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return float(np.linalg.norm(self.coefficients))

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        other_iter: list[tuple[float, RydbergStateSQDT]]
        if isinstance(other, RydbergStateSQDT):
            other_iter = [(1.0, other)]
        elif isinstance(other, RydbergStateMQDT):
            other_iter = [(coeff, sqdt) for coeff, sqdt in other]
        else:
            raise NotImplementedError(f"calc_reduced_overlap not implemented for {type(self)=}, {type(other)=}")

        ov = 0
        for coeff1, sqdt1 in self:
            for coeff2, sqdt2 in other_iter:
                ov += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_overlap(sqdt2)
        return ov

    @overload  # type: ignore [override]
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: None = None
    ) -> PintFloat: ...

    @overload
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str
    ) -> float: ...

    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintFloat | float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        other_iter: list[tuple[float, RydbergStateSQDT]]
        if isinstance(other, RydbergStateSQDT):
            other_iter = [(1.0, other)]
        elif isinstance(other, RydbergStateMQDT):
            other_iter = [(coeff, sqdt) for coeff, sqdt in other]
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element not implemented for {type(self)=}, {type(other)=}")

        value = 0
        for coeff1, sqdt1 in self:
            for coeff2, sqdt2 in other_iter:
                value += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_matrix_element(sqdt2, operator, unit=unit)
        return value


def get_mqdt_states_from_fmodel(
    model: FModel,
    nu_min: float | None = None,
    nu_max: float | None = None,
    *,
    overwrite_model_limits: bool = False,
) -> list[RydbergStateMQDT]:
    if overwrite_model_limits:
        if nu_min is None or nu_max is None:
            raise ValueError("nu_min and nu_max must be given if overwrite_model_limits is True.")
    else:
        nu_min = model.nu_min if nu_min is None else max(nu_min, model.nu_min)
        nu_max = model.nu_max if nu_max is None else min(nu_max, model.nu_max)
    if np.isinf(nu_max):
        raise ValueError("nu_max must be finite to calculate MQDT states.")

    nu_ref_list = model.calc_detm_roots(nu_min, nu_max)
    if len(nu_ref_list) == 0:
        logger.warning(
            "No MQDT states found in the range nu_min=%s, nu_max=%s for model %s", nu_min, nu_max, model.name
        )
        return []

    states: list[RydbergStateMQDT] = []
    for nu_ref in nu_ref_list:
        nuis = model.calc_channel_nuis(nu_ref)
        coefficients = calc_nullvector(model.calc_m_matrix(nu_ref))
        if coefficients is None:
            logger.warning("Failed to calculate nullvector for nu_ref=%s, skipping this state.", nu_ref)
            continue
        coefficients = np.array(
            [coeff * (nui ** (3 / 2)) / np.cos(np.pi * nui) for coeff, nui in zip(coefficients, nuis, strict=True)]
        )
        coefficients /= np.linalg.norm(coefficients)
        arg_max = np.argmax(np.abs(coefficients))
        coefficients *= np.sign(coefficients[arg_max])

        sqdt_states: list[RydbergStateSQDTAlkalineFJ | RydbergStateSQDTDummy] = []
        for nui, angular_ket in zip(nuis, model.outer_channels, strict=True):
            rydberg_class = RydbergStateSQDTDummy if is_dummy_ket(angular_ket) else RydbergStateSQDTAlkalineFJ
            sqdt_states.append(rydberg_class.from_angular_ket(model.species, angular_ket, nu=float(nui)))

        states.append(RydbergStateMQDT(coefficients, sqdt_states, nu_ref=nu_ref))

    return states
