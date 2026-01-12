from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import numpy as np

from rydstate.angular.angular_ket import (
    AngularKetBase,
    AngularKetFJ,
    AngularKetJJ,
    AngularKetLS,
)
from rydstate.angular.angular_matrix_element import is_angular_momentum_quantum_number

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Never, Self

    from rydstate.angular.angular_matrix_element import AngularMomentumQuantumNumbers, AngularOperatorType
    from rydstate.angular.utils import CouplingScheme

logger = logging.getLogger(__name__)


_AngularKet = TypeVar("_AngularKet", bound=AngularKetBase)


class AngularState(Generic[_AngularKet]):
    is_dummy: bool = False

    def __init__(
        self, coefficients: Sequence[float], kets: Sequence[_AngularKet], *, warn_if_not_normalized: bool = True
    ) -> None:
        """Initialize an angular state as a linear combination of angular kets.

        All kets must be of the same type (coupling scheme), and no duplicate kets are allowed.
        Dummy kets (AngularKetDummy) are ignored in the state representation,
        however adding them is recommended for normalization purposes.
        """
        self.coefficients = np.array(coefficients)
        self.kets = list(kets)
        self._warn_if_not_normalized = warn_if_not_normalized

        if len(coefficients) != len(kets):
            raise ValueError("Length of coefficients and kets must be the same.")
        if len(kets) == 0:
            raise ValueError("At least one ket must be provided.")
        if not all(type(ket) is type(self.nondummy_kets[0]) for ket in self.nondummy_kets):
            raise ValueError("All kets must be of the same type.")
        if len(set(self.kets)) != len(self.kets):
            raise ValueError("AngularState initialized with duplicate kets: %s", self.kets)
        if abs(self.norm - 1) > 1e-10 and warn_if_not_normalized:
            logger.warning("AngularState initialized with non-normalized coefficients: %s, %s", coefficients, kets)
        if self.norm > 1:
            self.coefficients /= self.norm

    def __repr__(self) -> str:
        terms = [f"{coeff}*{ket!r}" for coeff, ket in zip(self.coefficients, self.kets)]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{ket!s}" for coeff, ket in zip(self.coefficients, self.kets)]
        return f"{', '.join(terms)}"

    @property
    def nondummy_kets(self) -> list[_AngularKet]:
        return [ket for ket in self.kets if not ket.is_dummy]

    @property
    def nondummy_coefficients(self) -> np.ndarray:
        return np.array([coeff for coeff, ket in zip(self.coefficients, self.kets) if not ket.is_dummy])

    @property
    def coupling_scheme(self) -> CouplingScheme:
        """Return the coupling scheme of the state."""
        return self.nondummy_kets[0].coupling_scheme

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self.coefficients)  # type: ignore [return-value]

    @property
    def f_tot(self) -> float:
        """Return the total angular momentum f_tot of the state."""
        if not all(ket.f_tot == self.kets[0].f_tot for ket in self.kets):
            raise ValueError("Cannot return f_tot for AngularState with different f_tot values.")
        return self.kets[0].f_tot

    @overload
    def to(self, coupling_scheme: Literal["LS"]) -> AngularState[AngularKetLS]: ...

    @overload
    def to(self, coupling_scheme: Literal["JJ"]) -> AngularState[AngularKetJJ]: ...

    @overload
    def to(self, coupling_scheme: Literal["FJ"]) -> AngularState[AngularKetFJ]: ...

    @overload
    def to(self, coupling_scheme: Literal["Dummy"]) -> Never: ...

    def to(self, coupling_scheme: CouplingScheme) -> AngularState[Any]:
        """Convert to specified coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to convert to (e.g. "LS", "JJ", "FJ").

        Returns:
            The angular state in the specified coupling scheme.

        """
        newkets: list[AngularKetBase] = []
        new_coefficients: list[float] = []
        for coeff, ket in zip(self.coefficients, self.kets):
            state = ket.to_state() if ket.is_dummy else ket.to_state(coupling_scheme)
            for coeff2, ket2 in zip(state.coefficients, state.kets):
                if ket2 in newkets:
                    index = newkets.index(ket2)  # type: ignore [arg-type]
                    new_coefficients[index] += coeff * coeff2
                else:
                    newkets.append(ket2)  # type: ignore [arg-type]
                    new_coefficients.append(coeff * coeff2)
        warn_if_not_normalized = self._warn_if_not_normalized and (abs(self.norm - 1) < 1e-10)
        return AngularState(new_coefficients, newkets, warn_if_not_normalized=warn_if_not_normalized)

    def calc_exp_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the expectation value of a quantum number q.

        Args:
            q: The quantum number to calculate the expectation value for.

        """
        if len(self.nondummy_kets) == 0:
            raise ValueError("Cannot calculate expectation value for AngularState with only dummy kets.")

        if q not in self.nondummy_kets[0].quantum_number_names:
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if q in ket_class.quantum_number_names:
                    return self.to(ket_class.coupling_scheme).calc_exp_qn(q)

        qs = np.array([ket.get_qn(q) for ket in self.nondummy_kets])
        if all(q_val == qs[0] for q_val in qs):
            return qs[0]  # type: ignore [no-any-return]

        return np.sum(np.conjugate(self.nondummy_coefficients) * self.nondummy_coefficients * qs)  # type: ignore [no-any-return]

    def calc_std_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the standard deviation of a quantum number q.

        Args:
            q: The quantum number to calculate the standard deviation for.

        """
        if len(self.nondummy_kets) == 0:
            raise ValueError("Cannot calculate std quantum number for AngularState with only dummy kets.")

        if q not in self.nondummy_kets[0].quantum_number_names:
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if q in ket_class.quantum_number_names:
                    return self.to(ket_class.coupling_scheme).calc_std_qn(q)

        qs = np.array([ket.get_qn(q) for ket in self.nondummy_kets])
        if all(q_val == qs[0] for q_val in qs):
            return 0

        coefficients2 = np.conjugate(self.nondummy_coefficients) * self.nondummy_coefficients
        exp_q = np.sum(coefficients2 * qs)
        exp_q2 = np.sum(coefficients2 * qs * qs)

        if abs(exp_q2 - exp_q**2) < 1e-10:
            return 0
        return math.sqrt(exp_q2 - exp_q**2)

    def calc_reduced_overlap(self, other: AngularState[Any] | AngularKetBase) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        ov = 0
        for coeff1, ket1 in zip(self.coefficients, self.kets):
            for coeff2, ket2 in zip(other.coefficients, other.kets):
                ov += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    def calc_reduced_matrix_element(  # noqa: C901
        self: Self, other: AngularState[Any] | AngularKetBase, operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if len(self.nondummy_kets) == 0:
            raise ValueError("Cannot calculate reduced matrix element for AngularState with only dummy kets.")

        if isinstance(other, AngularKetBase):
            other = other.to_state()
        if is_angular_momentum_quantum_number(operator) and operator not in self.nondummy_kets[0].quantum_number_names:
            for ket_class in [AngularKetLS, AngularKetJJ, AngularKetFJ]:
                if operator in ket_class.quantum_number_names:
                    return self.to(ket_class.coupling_scheme).calc_reduced_matrix_element(other, operator, kappa)

        if self.coupling_scheme != other.coupling_scheme:
            other = other.to(self.coupling_scheme)

        value = 0
        for coeff1, ket1 in zip(self.coefficients, self.kets):
            if ket1.is_dummy:
                continue
            for coeff2, ket2 in zip(other.coefficients, other.kets):
                if ket2.is_dummy:
                    continue
                value += np.conjugate(coeff1) * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, kappa)

        return value

    def calc_matrix_element(
        self: Self, other: AngularState[Any] | AngularKetBase, operator: AngularOperatorType, kappa: int, q: int
    ) -> float:
        r"""Calculate the dimensionless angular matrix element.

        This means, calculate the following matrix element:

        .. math::
            \left\langle self | \hat{O}^{(\kappa)}_q | other \right\rangle
        """
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        states: list[AngularState[Any]] = [self, other]
        for state in states:
            if not all(ket.f_tot == state.kets[0].f_tot for ket in state.kets):
                raise NotImplementedError(
                    "Different f_tot values are not supported yet for AngularState.calc_matrix_element."
                )
            if not all(ket.m == state.kets[0].m for ket in state.kets):
                raise NotImplementedError(
                    "Different m values are not supported yet for AngularState.calc_matrix_element."
                )

        if self.kets[0].m is None or other.kets[0].m is None:
            raise ValueError("m must be set for all kets to calculate the matrix element.")

        prefactor = self.nondummy_kets[0]._calc_wigner_eckart_prefactor(other.nondummy_kets[0], kappa, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        return prefactor * reduced_matrix_element
