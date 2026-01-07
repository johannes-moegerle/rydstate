from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rydstate.angular import AngularState
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.units import MatrixElementOperator, PintFloat

logger = logging.getLogger(__name__)


class RydbergStateBase(ABC):
    angular: AngularState[Any] | AngularKetBase

    @abstractmethod
    def calc_reduced_overlap(self, other: RydbergStateBase) -> float: ...

    @abstractmethod
    def calc_reduced_matrix_element(
        self, other: RydbergStateBase, operator: MatrixElementOperator, unit: str | None = None
    ) -> PintFloat | float: ...
