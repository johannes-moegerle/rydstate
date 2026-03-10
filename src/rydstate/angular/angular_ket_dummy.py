from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.angular_matrix_element import is_angular_operator_type
from rydstate.angular.core_ket_base import CoreKetDummy
from rydstate.angular.utils import InvalidQuantumNumbersError, NotSet, Unknown, is_not_set

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.angular.angular_matrix_element import AngularMomentumQuantumNumbers, AngularOperatorType
    from rydstate.angular.angular_state import AngularState
    from rydstate.angular.utils import CouplingScheme, UnknownType

logger = logging.getLogger(__name__)


class AngularKetDummy(AngularKetBase):
    """Dummy spin ket for unknown quantum numbers."""

    __slots__ = ("name",)
    quantum_number_names: ClassVar = ("f_tot",)
    coupled_quantum_numbers: ClassVar = {}
    coupling_scheme = "Dummy"  # type: ignore [assignment]

    name: str
    """Name of the dummy ket."""

    def __init__(
        self,
        name: str,
        f_tot: float,
        m: float | NotSet = NotSet,
    ) -> None:
        """Initialize the Spin ket."""
        self.name = name

        self.f_tot = f_tot
        self.m = m

        super()._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not is_not_set(self.m) and not -self.f_tot <= self.m <= self.f_tot:
            msgs.append(f"m must be between -f_tot and f_tot, but {self.f_tot=}, {self.m=}")

        if msgs:
            msg = "\n  ".join(msgs)
            raise InvalidQuantumNumbersError(self, msg)

    def __repr__(self) -> str:
        args = f"{self.name}, f_tot={self.f_tot}"
        if not is_not_set(self.m):
            args += f", m={self.m}"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return self.__repr__().replace("AngularKet", "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularKetBase):
            return NotImplemented
        if not isinstance(other, AngularKetDummy):
            return False
        return self.name == other.name and self.f_tot == other.f_tot and self.m == other.m

    def __hash__(self) -> int:
        return hash((self.name, self.f_tot, self.m))

    def calc_reduced_matrix_element(
        self: Self,
        other: AngularKetBase,  # noqa: ARG002
        operator: AngularOperatorType,
        kappa: int,  # noqa: ARG002
    ) -> float:
        if not is_angular_operator_type(operator):
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        # ignore contributions from dummy kets
        return 0

    def get_core_ket(self) -> CoreKetDummy:
        """Get the dummy core ket corresponding to this Dummy ket."""
        core_name = self.name.split("nl")[0]
        return CoreKetDummy(name=core_name)

    def calc_exp_qn(self, qn: AngularMomentumQuantumNumbers) -> UnknownType:
        if qn == "f_tot":
            return self.f_tot
        return Unknown

    def calc_std_qn(self, qn: AngularMomentumQuantumNumbers) -> UnknownType:
        if qn == "f_tot":
            return 0
        return Unknown

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> UnknownType:
        if qn == "f_tot":
            return self.f_tot
        return Unknown

    def to_state(self, coupling_scheme: CouplingScheme | None = None) -> AngularState[AngularKetDummy]:  # type: ignore [override]
        """Convert to state in the specified coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to convert to (e.g. "LS", "JJ", "FJ").
                If None, the state will be a trivial state (one component) in the current coupling scheme.

        Returns:
            The angular state in the specified coupling scheme.

        """
        coupling_scheme  # noqa: B018
        return self._create_angular_state([1], [self])
