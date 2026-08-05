from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import numpy as np

from rydstate.angular.angular_ket import (
    AngularKetBase,
    AngularKetFJ,
    AngularKetJJ,
    AngularKetLS,
)
from rydstate.angular.utils import is_angular_momentum_quantum_number, is_not_set, is_unknown

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

    from rydstate.angular.utils import AngularMomentumQuantumNumbers, AngularOperatorType, CouplingScheme, NotSet
    from rydstate.units import NDArray

logger = logging.getLogger(__name__)


GenericT_AngularKet = TypeVar("GenericT_AngularKet", bound=AngularKetBase[Any])


class AngularState(Generic[GenericT_AngularKet]):
    def __init__(self, coefficients: Sequence[float] | NDArray, kets: Sequence[GenericT_AngularKet]) -> None:
        if np.isreal(coefficients).all():
            coefficients = np.array(coefficients, dtype=float)
        else:
            coefficients = np.array(coefficients, dtype=complex)
        self._coefficients: list[float] = coefficients.tolist()
        self.kets = kets

        if len(coefficients) != len(kets):
            raise ValueError("Length of coefficients and kets must be the same.")
        if len(kets) == 0:
            raise ValueError("At least one ket must be provided.")
        if not all(ket.coupling_scheme == self.coupling_scheme for ket in kets):
            raise ValueError("All kets must have the same coupling scheme.")
        if len(set(kets)) != len(kets):
            raise ValueError("AngularState initialized with duplicate kets.")

        if abs(self.norm - 1) > 1e-10:
            raise ValueError(
                f"AngularState initialized with non-normalized coefficients: {self._coefficients}, {self.kets}"
            )

    def __repr__(self) -> str:
        terms = [f"{coeff}*{ket!r}" for coeff, ket in self]
        return f"{self.__class__.__name__}({', '.join(terms)})"

    def __str__(self) -> str:
        terms = [f"{coeff}*{ket!s}" for coeff, ket in self]
        return f"{', '.join(terms)}"

    def __iter__(self) -> Iterator[tuple[float, GenericT_AngularKet]]:
        return zip(self._coefficients, self.kets, strict=True)

    @property
    def coupling_scheme(self) -> CouplingScheme:
        """Return the coupling scheme of the state."""
        return self.kets[0].coupling_scheme

    @property
    def norm(self) -> float:
        """Return the norm of the state (should be 1)."""
        return np.linalg.norm(self._coefficients)  # type: ignore [return-value]

    @cached_property
    def coefficients(self) -> NDArray:
        """Return the coefficients as numpy array."""
        return np.array(self._coefficients)

    @cached_property
    def _coefficients_conjugate(self) -> list[float]:
        """Return the conjugate of the coefficients as a plain python list."""
        return np.conjugate(self._coefficients).tolist()  # type: ignore [no-any-return]

    @property
    def i_c(self) -> float:
        """Return the i_c quantum number of the state."""
        i_c_list = [ket.i_c for ket in self.kets]
        if not all(f == i_c_list[0] for f in i_c_list):
            raise ValueError(f"Cannot calculate i_c for {self!r} because the kets have different i_c values.")
        return i_c_list[0]

    @property
    def f_tot(self) -> float:
        """Return the total f quantum number of the state."""
        f_tot_list = [ket.f_tot for ket in self.kets]
        if not all(f == f_tot_list[0] for f in f_tot_list):
            raise ValueError(f"Cannot calculate f_tot for {self!r} because the kets have different f_tot values.")
        return f_tot_list[0]

    @property
    def parity(self) -> int:
        """Return the parity of the state."""
        parities = [ket.parity for ket in self.kets]
        if not all(parity == parities[0] for parity in parities):
            raise ValueError(f"Cannot calculate parity for {self!r} because the kets have different parities.")
        return parities[0]

    @property
    def m(self) -> float | NotSet:
        """Return the m quantum number of the state."""
        m_list = [ket.m for ket in self.kets]
        if not all(m == m_list[0] for m in m_list):
            raise ValueError(f"Cannot calculate m for {self!r} because the kets have different m values.")
        return m_list[0]

    @overload
    def to(self, coupling_scheme: Literal["LS"]) -> AngularState[AngularKetLS[Any]]: ...

    @overload
    def to(self, coupling_scheme: Literal["JJ"]) -> AngularState[AngularKetJJ[Any]]: ...

    @overload
    def to(self, coupling_scheme: Literal["FJ"]) -> AngularState[AngularKetFJ[Any]]: ...

    def to(self: AngularState[Any], coupling_scheme: CouplingScheme) -> AngularState[Any]:
        """Convert to specified coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to convert to (e.g. "LS", "JJ", "FJ").

        Returns:
            The angular state in the specified coupling scheme.

        """
        kets: list[AngularKetBase[Any]] = []
        coefficients: list[float] = []
        for coeff, ket in self:
            for scheme_coeff, scheme_ket in ket.to_state(coupling_scheme):
                if scheme_ket in kets:
                    index = kets.index(scheme_ket)
                    coefficients[index] += coeff * scheme_coeff
                else:
                    kets.append(scheme_ket)
                    coefficients.append(coeff * scheme_coeff)
        return AngularState(coefficients, kets)

    def calc_exp_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the expectation value of a quantum number q.

        Args:
            q: The quantum number to calculate the expectation value for.

        """
        if q not in self.kets[0].quantum_number_names:
            for ket_class in (AngularKetLS, AngularKetJJ, AngularKetFJ):
                if q in ket_class.quantum_number_names:
                    return self.to(ket_class.coupling_scheme).calc_exp_qn(q)

        qns = [ket.get_qn(q) for ket in self.kets]
        if all(q_val == qns[0] for q_val in qns):
            return qns[0]

        coeffs = np.array([coeff for coeff, qn in zip(self.coefficients, qns, strict=True) if not is_unknown(qn)])
        qns = [qn for qn in qns if not is_unknown(qn)]
        norm = np.linalg.norm(coeffs)
        if 1 - norm / self.norm > 5e-2:
            logger.warning(
                "Expectation value of quantum number %s calculated from kets with unknown values. "
                "The contribution of the unknown kets is significant but neglected: %.1f%%",
                *(q, (1 - norm / self.norm) * 100),
            )

        coefficients2 = np.conjugate(coeffs) * coeffs / norm**2
        return float(np.sum(coefficients2 * np.array(qns)))

    def calc_std_qn(self, q: AngularMomentumQuantumNumbers) -> float:
        """Calculate the standard deviation of a quantum number q.

        Args:
            q: The quantum number to calculate the standard deviation for.

        """
        if q not in self.kets[0].quantum_number_names:
            for ket_class in (AngularKetLS, AngularKetJJ, AngularKetFJ):
                if q in ket_class.quantum_number_names:
                    return self.to(ket_class.coupling_scheme).calc_std_qn(q)

        qns = np.array([ket.get_qn(q) for ket in self.kets])
        if all(qn == qns[0] for qn in qns):
            return 0

        coeffs = np.array([coeff for coeff, qn in zip(self.coefficients, qns, strict=True) if not is_unknown(qn)])
        qns = np.array([qn for qn in qns if not is_unknown(qn)])
        norm = np.linalg.norm(coeffs)
        if 1 - norm / self.norm > 5e-2:
            logger.warning(
                "Standard deviation of quantum number %s calculated from kets with unknown values. "
                "The contribution of the unknown kets is significant but neglected: %.1f%%",
                *(q, (1 - norm / self.norm) * 100),
            )

        coefficients2 = np.conjugate(coeffs) * coeffs / norm**2
        exp_q = np.sum(coefficients2 * qns)
        exp_q2 = np.sum(coefficients2 * qns * qns)

        if abs(exp_q2 - exp_q**2) < 1e-10:
            return 0
        return math.sqrt(exp_q2 - exp_q**2)

    def calc_reduced_overlap(self, other: AngularState[Any] | AngularKetBase[Any]) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m)."""
        if isinstance(other, AngularKetBase):
            other = other.to_state()

        ov = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.kets, strict=True):  # noqa: SLF001
                ov += coeff1 * coeff2 * ket1.calc_reduced_overlap(ket2)
        return ov

    def calc_reduced_matrix_element(
        self: Self, other: AngularState[Any] | AngularKetBase[Any], operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        This means, calculate the following matrix element (self is the bra, other is the ket):

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        Args:
            other: The other AngularState (or AngularKet) :math:`|other>` (used as the ket).
            operator: The operator type :math:`\hat{O}^{(\kappa)}` for which to calculate the matrix element.
                E.g. 'spherical', 's_tot', 'l_r', etc.
            kappa: The rank :math:`\kappa` of the angular momentum operator.

        Returns:
            The reduced dimensionless angular matrix element.

        """
        if isinstance(other, AngularKetBase):
            other = other.to_state()
        if is_angular_momentum_quantum_number(operator) and operator not in self.kets[0].quantum_number_names:
            for ket_class in (AngularKetLS, AngularKetJJ, AngularKetFJ):
                if operator in ket_class.quantum_number_names:
                    state = self.to(ket_class.coupling_scheme)
                    return state.calc_reduced_matrix_element(other, operator, kappa)

        if self.coupling_scheme != other.coupling_scheme:
            other = other.to(self.coupling_scheme)

        value = 0.0
        for coeff1, ket1 in zip(self._coefficients_conjugate, self.kets, strict=True):
            for coeff2, ket2 in zip(other._coefficients, other.kets, strict=True):  # noqa: SLF001
                value += coeff1 * coeff2 * ket1.calc_reduced_matrix_element(ket2, operator, kappa)
        return value

    def calc_matrix_element(
        self: Self, other: AngularState[Any] | AngularKetBase[Any], operator: AngularOperatorType, kappa: int, q: int
    ) -> float:
        r"""Calculate the dimensionless angular matrix element.

        This means, calculate the following matrix element (self is the bra, other is the ket):

        .. math::
            \left\langle self | \hat{O}^{(\kappa)}_q | other \right\rangle

        Args:
            other: The other AngularState (or AngularKet) :math:`|other>` (used as the ket).
            operator: The operator type :math:`\hat{O}^{(\kappa)}_q` for which to calculate the matrix element.
                E.g. 'spherical', 's_tot', 'l_r', etc.
            kappa: The rank :math:`\kappa` of the angular momentum operator.
            q: The component :math:`q` of the angular momentum operator.

        Returns:
            The dimensionless angular matrix element.

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

        if is_not_set(self.kets[0].m) or is_not_set(other.kets[0].m):
            raise RuntimeError("m must be set for all kets to calculate the matrix element.")

        prefactor = self.kets[0]._calc_wigner_eckart_prefactor(other.kets[0], kappa, q)  # noqa: SLF001
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        return prefactor * reduced_matrix_element
