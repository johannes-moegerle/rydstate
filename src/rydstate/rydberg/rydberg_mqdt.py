from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np

from rydstate.angular import AngularState
from rydstate.rydberg.rydberg_base import RydbergStateBase
from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from rydstate.units import MatrixElementOperator, PintFloat


logger = logging.getLogger(__name__)


_RydbergState = TypeVar("_RydbergState", bound=RydbergStateSQDT)


class RydbergStateMQDT(RydbergStateBase, Generic[_RydbergState]):
    angular: AngularState[Any]
    """Return the angular part of the MQDT state as an AngularState."""

    def __init__(
        self,
        coefficients: Sequence[float],
        sqdt_states: Sequence[_RydbergState],
        *,
        nu_energy: float | None = None,
        warn_if_not_normalized: bool = True,
    ) -> None:
        self.coefficients = np.array(coefficients)
        self.sqdt_states = sqdt_states
        self.nu_energy = nu_energy
        self.angular = AngularState(self.coefficients.tolist(), [ket.angular for ket in sqdt_states])

        if len(coefficients) != len(sqdt_states):
            raise ValueError("Length of coefficients and sqdt_states must be the same.")
        if not all(type(sqdt_state) is type(sqdt_states[0]) for sqdt_state in sqdt_states):
            raise ValueError("All sqdt_states must be of the same type.")
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

    def __iter__(self) -> Iterator[tuple[float, _RydbergState]]:
        return zip(self.coefficients, self.sqdt_states).__iter__()

    def __repr__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!r}" for coeff, sqdt_state in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{sqdt_state!s}" for coeff, sqdt_state in self]
        return f"{', '.join(terms)}"

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    def calc_reduced_overlap(self, other: RydbergStateBase) -> float:
        """Calculate the reduced overlap <self|other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, RydbergStateSQDT):
            other = other.to_mqdt()

        if isinstance(other, RydbergStateMQDT):
            ov = 0
            for coeff1, sqdt1 in self:
                for coeff2, sqdt2 in other:
                    ov += np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_overlap(sqdt2)
            return ov

        raise NotImplementedError(f"calc_reduced_overlap not implemented for {type(self)=}, {type(other)=}")

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
        if isinstance(other, RydbergStateSQDT):
            other = other.to_mqdt()

        if isinstance(other, RydbergStateMQDT):
            value = 0
            for coeff1, sqdt1 in self:
                for coeff2, sqdt2 in other:
                    value += (
                        np.conjugate(coeff1) * coeff2 * sqdt1.calc_reduced_matrix_element(sqdt2, operator, unit=unit)
                    )
            return value

        raise NotImplementedError(f"calc_reduced_overlap not implemented for {type(self)=}, {type(other)=}")
