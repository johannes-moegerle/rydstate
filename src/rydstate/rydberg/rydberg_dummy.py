from __future__ import annotations

from typing import TYPE_CHECKING, TypeGuard

from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT

if TYPE_CHECKING:
    from rydstate.angular.angular_ket_dummy import AngularKetDummy


class RydbergStateSQDTDummy(RydbergStateSQDT):
    """Create an Alkaline Rydberg state, including the radial and angular states."""

    angular: AngularKetDummy

    def __init__(self) -> None:
        raise RuntimeError("RydbergStateSQDTDummy ican only be created via the from_angular_ket method.")

    def __repr__(self) -> str:
        species, nu = self.species, self.nu
        return f"{self.__class__.__name__}({species.name}, {nu=}, angular_ket={self.angular})"


def is_dummy_rydberg_state(state: RydbergStateSQDT) -> TypeGuard[RydbergStateSQDTDummy]:
    """Check if state is a RydbergStateSQDTDummy."""
    return isinstance(state, RydbergStateSQDTDummy)
