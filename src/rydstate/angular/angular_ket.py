from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Literal

from rydstate.angular.angular_ket_base import AngularKetBase, AngularKetBaseFJ, AngularKetBaseJJ, AngularKetBaseLS
from rydstate.angular.angular_matrix_element import (
    calc_prefactor_of_operator_in_coupled_scheme,
    calc_reduced_identity_matrix_element,
    calc_reduced_spherical_matrix_element,
    calc_reduced_spin_matrix_element,
)
from rydstate.angular.utils import (
    is_angular_momentum_quantum_number,
    is_angular_operator_type,
    is_not_set,
    is_unknown,
    minus_one_pow,
)
from rydstate.angular.wigner_symbols import calc_wigner_3j

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.angular.utils import AngularMomentumQuantumNumbers, AngularOperatorType

logger = logging.getLogger(__name__)


class AngularKet(AngularKetBase, ABC):
    """Base class for a angular ket where no quantum numbers are unknown (i.e. a simple canonical spin ketstate)."""

    i_c: float
    s_c: float
    l_c: int
    s_r: float
    l_r: int

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if any(is_unknown(qn) for qn in self.quantum_numbers):
            cls_name = type(self).__name__
            cls_base_name = cls_name.replace("AngularKet", "AngularKetBase")
            raise ValueError(f"Unknown quantum numbers are not allowed for {cls_name}, use {cls_base_name} instead.")

        super().sanity_check(msgs)

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        """Get the value of a quantum number by name."""
        qn_value = super().get_qn(qn)
        if is_unknown(qn_value):
            raise ValueError(f"Quantum number {qn} is unknown for {self!r}.")
        return qn_value

    def calc_exp_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        """Calculate the expectation value of a quantum number qn.

        If the quantum number is a good quantum number simply return it,
        otherwise calculate it, see also AngularState.calc_exp_qn for more details.

        Args:
            qn: The quantum number to calculate the expectation value for.

        """
        if qn in self.quantum_number_names:
            return self.get_qn(qn)
        return self.to_state().calc_exp_qn(qn)

    def calc_std_qn(self, qn: AngularMomentumQuantumNumbers) -> float:
        """Calculate the standard deviation of a quantum number qn.

        If the quantum number is a good quantum number return 0,
        otherwise calculate the std, see also AngularState.calc_std_qn for more details.

        Args:
            qn: The quantum number to calculate the standard deviation for.

        """
        if qn in self.quantum_number_names:
            return 0
        return self.to_state().calc_std_qn(qn)

    def calc_reduced_matrix_element(  # noqa: C901
        self: Self, other: AngularKetBase, operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        We follow equation (7.1.7) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if not is_angular_operator_type(operator):
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if type(self) is not type(other):
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)
        if is_angular_momentum_quantum_number(operator) and operator not in self.quantum_number_names:
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)

        qn_name: AngularMomentumQuantumNumbers
        if operator == "spherical":
            qn_name = "l_r"
            complete_reduced_matrix_element = calc_reduced_spherical_matrix_element(self.l_r, other.l_r, kappa)
        elif operator in self.quantum_number_names:
            if not kappa == 1:
                raise ValueError("Only kappa=1 is supported for spin operators.")
            qn_name = operator
            complete_reduced_matrix_element = calc_reduced_spin_matrix_element(
                self.get_qn(qn_name), other.get_qn(qn_name)
            )
        elif operator.startswith("identity_"):
            if not kappa == 0:
                raise ValueError("Only kappa=0 is supported for identity operator.")
            qn_name = operator.replace("identity_", "")  # type: ignore [assignment]
            complete_reduced_matrix_element = calc_reduced_identity_matrix_element(
                self.get_qn(qn_name), other.get_qn(qn_name)
            )
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if complete_reduced_matrix_element == 0:
            return 0
        if self._kronecker_delta_non_involved_spins(other, qn_name) == 0:
            return 0
        prefactor = self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_name, kappa)
        return prefactor * complete_reduced_matrix_element

    def calc_matrix_element(self, other: AngularKetBase, operator: AngularOperatorType, kappa: int, q: int) -> float:
        r"""Calculate the dimensionless angular matrix element.

        Use the Wigner-Eckart theorem to calculate the angular matrix element from the reduced matrix element.
        We stick to the convention from Edmonds 1985 "Angular Momentum in Quantum Mechanics", see equation (5.4.1).
        This means, calculate the following matrix element:

        .. math::
            \left\langle self | \hat{O}^{(\kappa)}_q | other \right\rangle
            = <\alpha',f_{tot}',m'| \hat{O}^{(\kappa)}_q |\alpha,f_{tot},m>
            = (-1)^{(f_{tot} - m)} \cdot \mathrm{Wigner3j}(f_{tot}', \kappa, f_{tot}, -m', q, m)
                \cdot <\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>

        where alpha denotes all other quantum numbers
        and :math:`<\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>` is the reduced matrix element
        (see `calc_reduced_matrix_element`).

        Args:
            other: The other AngularKet :math:`|other>`.
            operator: The operator type :math:`\hat{O}_{kq}` for which to calculate the matrix element.
                E.g. 'spherical', 's_tot', 'l_r', etc.
            kappa: The rank :math:`\kappa` of the angular momentum operator.
            q: The component :math:`q` of the angular momentum operator.

        Returns:
            The dimensionless angular matrix element.

        """
        if is_not_set(self.m) or is_not_set(other.m):
            raise RuntimeError("m must be set to calculate the matrix element.")

        prefactor = self._calc_wigner_eckart_prefactor(other, kappa, q)
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        return prefactor * reduced_matrix_element

    def _calc_wigner_eckart_prefactor(self, other: AngularKetBase, kappa: int, q: int) -> float:
        if is_not_set(self.m) or is_not_set(other.m):
            raise RuntimeError("m must be set to calculate the Wigner-Eckart prefactor.")
        return minus_one_pow(self.f_tot - self.m) * calc_wigner_3j(self.f_tot, kappa, other.f_tot, -self.m, q, other.m)

    def _kronecker_delta_non_involved_spins(self, other: AngularKetBase, qn: AngularMomentumQuantumNumbers) -> int:
        """Calculate the Kronecker delta for non involved angular momentum quantum numbers.

        This means return 0 if any of the quantum numbers,
        that are not qn or a coupled quantum number resulting from qn differ between self and other.
        """
        if qn not in self.quantum_number_names:
            raise ValueError(f"Quantum number {qn} is not a valid angular momentum quantum number for {self!r}.")

        resulting_qns = {qn}
        last_qn = qn
        while last_qn != "f_tot":
            for key, qs in self.coupled_quantum_numbers.items():
                if last_qn in qs:
                    resulting_qns.add(key)
                    last_qn = key
                    break
            else:
                raise ValueError(
                    f"_kronecker_delta_non_involved_spins: {last_qn} not found in coupled_quantum_numbers."
                )

        non_involved_qns = set(self.quantum_number_names) - resulting_qns
        for _qn in non_involved_qns:
            if self.get_qn(_qn) != other.get_qn(_qn):
                return 0
        return 1

    def _calc_prefactor_of_operator_in_coupled_scheme(
        self, other: AngularKetBase, qn: AngularMomentumQuantumNumbers, kappa: int
    ) -> float:
        """Calculate the prefactor for the complete reduced matrix element.

        This approach is only valid if the operator acts only on one of the well defined quantum numbers.
        """
        if type(self) is not type(other):
            raise ValueError(
                "Both kets must be of the same type to calculate the prefactor of the operator in the coupled scheme."
            )

        if qn == "f_tot":
            return 1

        for key, qs in self.coupled_quantum_numbers.items():
            if qn not in qs:
                continue
            qn_combined = key
            # NOTE: the order does actually matter for the sign of some matrix elements
            # we use this to convention to stay consistent with the old pairinteraction database signs
            qn2, qn1 = qs
            operator_acts_on: Literal["first", "second"] = "first" if qn == qn1 else "second"
            break
        else:  # no break
            raise ValueError(f"Quantum number {qn} not found in coupled_quantum_numbers.")

        f1, f2, f_tot = (self.get_qn(qn1), self.get_qn(qn2), self.get_qn(qn_combined))
        i1, i2, i_tot = (other.get_qn(qn1), other.get_qn(qn2), other.get_qn(qn_combined))

        if (operator_acts_on == "first" and f2 != i2) or (operator_acts_on == "second" and f1 != i1):
            return 0
        prefactor = calc_prefactor_of_operator_in_coupled_scheme(f1, f2, f_tot, i1, i2, i_tot, kappa, operator_acts_on)
        return prefactor * self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_combined, kappa)


class AngularKetLS(AngularKet, AngularKetBaseLS):
    """Spin ket in LS coupling."""

    s_tot: float
    l_tot: int
    j_tot: float


class AngularKetJJ(AngularKet, AngularKetBaseJJ):
    """Spin ket in JJ coupling."""

    j_c: float
    j_r: float
    j_tot: float


class AngularKetFJ(AngularKet, AngularKetBaseFJ):
    """Spin ket in FJ coupling."""

    j_c: float
    f_c: float
    j_r: float
