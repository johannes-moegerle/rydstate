from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from rydstate.angular import AngularKetFJ, AngularState
from rydstate.rydberg.rydberg_base import RydbergStateBase
from rydstate.rydberg.rydberg_sqdt import RydbergStateBaseSQDT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rydstate.angular.angular_ket_base import AngularKetBaseFJ
    from rydstate.units import MatrixElementOperator, NDArray, PintFloat


logger = logging.getLogger(__name__)


class RydbergStateMQDT(RydbergStateBase):
    angular: AngularState[AngularKetFJ | AngularKetBaseFJ]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        coefficients: Sequence[float] | NDArray,
        sqdt_states: Sequence[RydbergStateBaseSQDT[AngularKetFJ | AngularKetBaseFJ]],
        nu: float,
        *,
        warn_if_not_normalized: bool = True,
        normalize: bool = True,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.sqdt_states = sqdt_states
        self.species = sqdt_states[0].species
        self._nu = nu
        self.angular = AngularState(self.coefficients.tolist(), [ket.angular for ket in sqdt_states])

        if len(coefficients) != len(sqdt_states):
            raise ValueError("Length of coefficients and sqdt_states must be the same.")
        if not all(isinstance(sqdt_state.angular, AngularKetFJ) for sqdt_state in sqdt_states):
            raise ValueError("All sqdt_states must have an angular part of type AngularKetFJ.")
        if not all((sqdt_state.species is sqdt_states[0].species) for sqdt_state in sqdt_states):
            raise ValueError("All sqdt_states must be of the same species.")
        if len(set(sqdt_states)) != len(sqdt_states):
            raise ValueError("RydbergStateMQDT initialized with duplicate sqdt_states.")

        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning(
                "RydbergStateMQDT initialized with non-normalized coefficients: %s, %s", coefficients, sqdt_states
            )
        if normalize:
            self.coefficients /= self.norm

    def __iter__(self) -> Iterator[tuple[float, RydbergStateBaseSQDT[AngularKetFJ | AngularKetBaseFJ]]]:
        return zip(self.coefficients, self.sqdt_states, strict=True).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!r}" for coeff, sqdt_state in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!s}" for coeff, sqdt_state in self]
        return f"{', '.join(terms)}"

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return float(np.linalg.norm(self.coefficients))

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        other_iter: list[tuple[float, RydbergStateBaseSQDT[Any]]]
        if isinstance(other, RydbergStateBaseSQDT):
            other_iter = [(1.0, other)]
        elif isinstance(other, RydbergStateMQDT):
            other_iter = [(coeff, sqdt) for coeff, sqdt in other]
        else:
            raise NotImplementedError(f"calc_reduced_overlap not implemented for {type(self)=}, {type(other)=}")

        ov = 0.0
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
        other_iter: list[tuple[float, RydbergStateBaseSQDT[Any]]]
        if isinstance(other, RydbergStateBaseSQDT):
            other_iter = [(1.0, other)]
        elif isinstance(other, RydbergStateMQDT):
            other_iter = [(coeff, sqdt) for coeff, sqdt in other]
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element not implemented for {type(self)=}, {type(other)=}")

        value = 0.0
        for coeff1, sqdt1 in self:
            for coeff2, sqdt2 in other_iter:
                value += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_matrix_element(sqdt2, operator, unit=unit)
        return value
