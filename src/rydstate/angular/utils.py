from __future__ import annotations

import contextlib
import typing as t
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, TypeVar, get_args

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec, TypeIs

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.species.species_object import SpeciesObject

    P = ParamSpec("P")
    R = TypeVar("R")

    def lru_cache(maxsize: int) -> Callable[[Callable[P, R]], Callable[P, R]]: ...  # type: ignore [no-redef]


CouplingScheme = Literal["LS", "JJ", "FJ"]


AngularMomentumQuantumNumbers = Literal[
    "i_c", "s_c", "l_c", "s_r", "l_r", "s_tot", "l_tot", "j_c", "j_r", "j_tot", "f_c", "f_tot"
]

IdentityOperators = Literal[
    "identity_i_c",
    "identity_s_c",
    "identity_l_c",
    "identity_s_r",
    "identity_l_r",
    "identity_s_tot",
    "identity_l_tot",
    "identity_j_c",
    "identity_j_r",
    "identity_j_tot",
    "identity_f_c",
    "identity_f_tot",
]

AngularOperatorType = Literal[
    "spherical",
    AngularMomentumQuantumNumbers,
    IdentityOperators,
]


class _Meta(type(t.Protocol)):  # type: ignore [misc]
    def __repr__(cls) -> str:
        return str(cls.__name__)


@t.runtime_checkable
class NotSet(t.Protocol, metaclass=_Meta):
    """Singleton for a not set value and type at the same time.

    See Also:
    https://stackoverflow.com/questions/77571796/how-to-create-singleton-object-which-could-be-used-both-as-type-and-value-simi

    """

    @staticmethod
    def __not_set() -> None: ...


@t.runtime_checkable
class Unknown(t.Protocol, metaclass=_Meta):
    """Singleton for a unknown quantum number and type at the same time.

    See Also:
    https://stackoverflow.com/questions/77571796/how-to-create-singleton-object-which-could-be-used-both-as-type-and-value-simi

    """

    @staticmethod
    def __unknown() -> None: ...


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase, msg: str = "") -> None:
        _msg = f"Invalid quantum numbers for {ket!r}"
        if len(msg) > 0:
            _msg += f"\n  {msg}"
        super().__init__(_msg)


def is_angular_momentum_quantum_number(qn: str) -> TypeGuard[AngularMomentumQuantumNumbers]:
    """Check if the given string is an AngularMomentumQuantumNumbers."""
    return qn in get_args(AngularMomentumQuantumNumbers)


def is_angular_operator_type(qn: str) -> TypeGuard[AngularOperatorType]:
    """Check if the given string is an AngularOperatorType."""
    return qn in get_args(AngularOperatorType)


def is_unknown(obj: float | Unknown | None) -> TypeIs[Unknown]:
    """Check if obj is Unknown."""
    return obj is Unknown


def is_not_set(obj: Any) -> TypeIs[NotSet]:  # noqa: ANN401
    """Check if the obj is the NotSet singleton."""
    return obj is NotSet


def minus_one_pow(n: float) -> int:
    """Calculate (-1)^n for an integer n and raise an error if n is not an integer."""
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"minus_one_pow: Invalid input {n=} is not an integer.")


def try_trivial_spin_addition(
    s_1: float | Unknown, s_2: float | Unknown, s_tot: float | Unknown | None
) -> float | Unknown:
    """Try to determine s_tot from s_1 and s_2 if it is not given.

    If s_tot is None and cannot be uniquely determined from s_1 and s_2, raise an error.
    Otherwise return s_tot or the trivial sum s_1 + s_2.
    """
    if s_tot is not None and not is_unknown(s_tot):
        return float(s_tot)
    if s_1 == 0 and not is_unknown(s_2):
        return float(s_2)
    if s_2 == 0 and not is_unknown(s_1):
        return float(s_1)
    return Unknown


def check_spin_addition_rule(s_1: float | Unknown, s_2: float | Unknown, s_tot: float | Unknown) -> bool:
    r"""Check if the spin addition rule is satisfied.

    If any of the quantum numbers is Unknown, return True.
    Else check the following conditions:
    :math:`|s_1 - s_2| \leq s_{tot} \leq s_1 + s_2`
    and
    :math:`s_1 + s_2 + s_{tot}` is an integer
    """
    if is_unknown(s_1) or is_unknown(s_2) or is_unknown(s_tot):
        return True
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


def get_possible_quantum_number_values(
    s_1: float | Unknown, s_2: float | Unknown, s_tot: float | Unknown | None
) -> list[float] | list[Unknown]:
    """Determine a list of possible s_tot values from s_1 and s_2 if s_tot is not given, else return [s_tot]."""
    if s_tot is not None and not is_unknown(s_tot):
        return [s_tot]
    if is_unknown(s_1) or is_unknown(s_2):
        return [Unknown]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]


@lru_cache(maxsize=1_000)
def quantum_numbers_to_angular_ket(
    species: str | SpeciesObject,
    s_c: float | None = None,
    l_c: int = 0,
    j_c: float | None = None,
    f_c: float | None = None,
    s_r: float = 0.5,
    l_r: int | None = None,
    j_r: float | None = None,
    s_tot: float | None = None,
    l_tot: int | None = None,
    j_tot: float | None = None,
    f_tot: float | None = None,
    m: float | NotSet = NotSet,
) -> AngularKetBase:
    r"""Return an AngularKet object in the corresponding coupling scheme from the given quantum numbers.

    Args:
        species: Atomic species.
        s_c: Spin quantum number of the core electron (0 for Alkali, 0.5 for divalent atoms).
        l_c: Orbital angular momentum quantum number of the core electron.
        j_c: Total angular momentum quantum number of the core electron.
        f_c: Total angular momentum quantum number of the core (core electron + nucleus).
        s_r: Spin quantum number of the rydberg electron (always 0.5).
        l_r: Orbital angular momentum quantum number of the rydberg electron.
        j_r: Total angular momentum quantum number of the rydberg electron.
        s_tot: Total spin quantum number of all electrons.
        l_tot: Total orbital angular momentum quantum number of all electrons.
        j_tot: Total angular momentum quantum number of all electrons.
        f_tot: Total angular momentum quantum number of the atom (rydberg electron + core).
        m: Total magnetic quantum number.
          Optional, only needed for concrete angular matrix elements.

    """
    from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS  # noqa: PLC0415

    with contextlib.suppress(InvalidQuantumNumbersError, ValueError):
        return AngularKetLS(
            s_c=s_c, l_c=l_c, s_r=s_r, l_r=l_r, s_tot=s_tot, l_tot=l_tot, j_tot=j_tot, f_tot=f_tot, m=m, species=species
        )

    with contextlib.suppress(InvalidQuantumNumbersError, ValueError):
        return AngularKetJJ(
            s_c=s_c, l_c=l_c, j_c=j_c, s_r=s_r, l_r=l_r, j_r=j_r, j_tot=j_tot, f_tot=f_tot, m=m, species=species
        )

    with contextlib.suppress(InvalidQuantumNumbersError, ValueError):
        return AngularKetFJ(
            s_c=s_c, l_c=l_c, j_c=j_c, f_c=f_c, s_r=s_r, l_r=l_r, j_r=j_r, f_tot=f_tot, m=m, species=species
        )

    raise ValueError("Invalid combination of angular quantum numbers provided.")
