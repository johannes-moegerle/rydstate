from __future__ import annotations

import contextlib
import logging
import typing as t
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.angular_ket_dummy import AngularKetDummy
    from rydstate.species.species_object import SpeciesObject

logger = logging.getLogger(__name__)


CouplingScheme = Literal["LS", "JJ", "FJ"]


@t.runtime_checkable
class NotSet(t.Protocol):
    """Singleton for a not set value and type at the same time.

    See Also:
    https://stackoverflow.com/questions/77571796/how-to-create-singleton-object-which-could-be-used-both-as-type-and-value-simi

    """

    @staticmethod
    def __not_set() -> None: ...


UnknownType: TypeAlias = float  # type(np.nan) == float
Unknown = np.nan


def is_not_set(obj: Any) -> TypeIs[NotSet]:  # noqa: ANN401
    """Check if the obj is the NotSet singleton."""
    return obj is NotSet


def is_unknown(x: float | UnknownType | None) -> TypeIs[UnknownType]:
    """Check if x is Unknown."""
    if x is None:
        return False
    return bool(np.isnan(x))


def is_dummy_ket(ket: AngularKetBase) -> TypeIs[AngularKetDummy]:
    """Check if ket is a AngularKetDummy."""
    from rydstate.angular.angular_ket_dummy import AngularKetDummy  # noqa: PLC0415

    return isinstance(ket, AngularKetDummy)


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase | None, msg: str = "") -> None:
        _msg = f"Invalid quantum numbers for {ket!r}"
        if len(msg) > 0:
            _msg += f"\n  {msg}"
        super().__init__(_msg)


def minus_one_pow(n: float) -> int:
    """Calculate (-1)^n for an integer n and raise an error if n is not an integer."""
    if n % 2 == 0:
        return 1
    if n % 2 == 1:
        return -1
    raise ValueError(f"minus_one_pow: Invalid input {n=} is not an integer.")


def try_trivial_spin_addition(
    s_1: float | UnknownType, s_2: float | UnknownType, s_tot: float | UnknownType | None, name: str
) -> float | UnknownType:
    """Try to determine s_tot from s_1 and s_2 if it is not given.

    If s_tot is Unknown, return Unknown.
    If s_tot is None and any part is Unknown, return Unknown.
    If s_tot is None and cannot be uniquely determined from s_1 and s_2, raise an error.
    Otherwise return s_tot or the trivial sum s_1 + s_2.
    """
    if is_unknown(s_1) or is_unknown(s_2):
        if s_tot is not None:
            return s_tot
        return Unknown

    if s_1 != 0 and s_2 != 0:
        if s_tot is not None:
            return s_tot
        msg = f"{name} must be set if both parts ({s_1} and {s_2}) are non-zero."
        raise ValueError(msg)

    if s_1 == 0 or s_2 == 0:
        calculated_s_tot = s_1 + s_2
        if is_unknown(s_tot):
            logger.warning(
                "%s is Unknown but could be uniquely determined as %s, using this value now", name, calculated_s_tot
            )
        elif s_tot is None:
            pass
        elif s_tot != calculated_s_tot:
            msg = f"{name} is {s_tot} but should be {calculated_s_tot} for the given parts ({s_1} and {s_2})."
            raise InvalidQuantumNumbersError(None, msg)
        return calculated_s_tot

    raise RuntimeError("This should never happen, all cases should be covered by the above conditions.")


def check_spin_addition_rule(s_1: float | UnknownType, s_2: float | UnknownType, s_tot: float | UnknownType) -> bool:
    r"""Check if the spin addition rule is satisfied.

    This means check the following conditions:
    :math:`|s_1 - s_2| \leq s_{tot} \leq s_1 + s_2`
    and
    :math:`s_1 + s_2 + s_{tot}` is an integer

    If any of the quantum numbers is Unknown, return True (cannot verify).
    """
    if is_unknown(s_1) or is_unknown(s_2) or is_unknown(s_tot):
        return True
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


def get_possible_quantum_number_values(s_1: float, s_2: float, s_tot: float | None) -> list[float]:
    """Determine a list of possible s_tot values from s_1 and s_2 if s_tot is not given, else return [s_tot]."""
    if s_tot is not None:
        return [s_tot]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]


def is_equal(qn1: float | UnknownType, qn2: float | UnknownType) -> bool:
    """Check if two quantum numbers are equal, treating Unknown equal to Unknown."""
    if is_unknown(qn1) and is_unknown(qn2):
        return True
    return qn1 == qn2


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
