from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.angular_matrix_element import is_angular_operator_type
from rydstate.angular.utils import InvalidQuantumNumbersError

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.angular.angular_matrix_element import AngularOperatorType

logger = logging.getLogger(__name__)


class AngularKetDummy(AngularKetBase):
    """Dummy spin ket for unknown quantum numbers."""

    __slots__ = ("name",)
    quantum_number_names: ClassVar = ("f_tot",)
    coupled_quantum_numbers: ClassVar = {}
    coupling_scheme = "Dummy"

    name: str
    """Name of the dummy ket."""

    def __init__(
        self,
        name: str,
        f_tot: float,
        m: float | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        self.name = name

        self.f_tot = f_tot
        self.m = m

        super()._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if self.m is not None and not -self.f_tot <= self.m <= self.f_tot:
            msgs.append(f"m must be between -f_tot and f_tot, but {self.f_tot=}, {self.m=}")

        if msgs:
            msg = "\n  ".join(msgs)
            raise InvalidQuantumNumbersError(self, msg)

    def __repr__(self) -> str:
        args = f"{self.name}, f_tot={self.f_tot}"
        if self.m is not None:
            args += f", m={self.m}"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return self.__repr__().replace("AngularKet", "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularKetBase):
            raise NotImplementedError(f"Cannot compare {self!r} with {other!r}.")
        if not isinstance(other, AngularKetDummy):
            return False
        return self.name == other.name and self.f_tot == other.f_tot and self.m == other.m

    def __hash__(self) -> int:
        return hash((self.name, self.f_tot, self.m))

    def calc_reduced_overlap(self, other: AngularKetBase) -> float:
        return int(self == other)

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
