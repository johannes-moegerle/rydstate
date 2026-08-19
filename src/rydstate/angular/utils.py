from __future__ import annotations

import contextlib
import functools
import numbers
import typing as t
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, get_args

import numpy as np
from typing_extensions import Never

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec, TypeIs

    from rydstate.angular.angular_ket import AngularKetBase

    P = ParamSpec("P")
    R = TypeVar("R")


def lru_cache(maxsize: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap functools.lru_cache for correct type annotations."""
    return functools.lru_cache(maxsize=maxsize)  # type: ignore [return-value]


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
    "spherical_inner_valence",
    AngularMomentumQuantumNumbers,
    IdentityOperators,
]


@functools.total_ordering
class _Meta(type(t.Protocol)):  # type: ignore [misc]
    def __repr__(cls) -> str:
        return str(cls.__name__)

    def __lt__(cls, other: object) -> bool:
        if other is cls:
            return False
        if isinstance(other, numbers.Real):
            return False  # objects derived from this metaclass always sort above every real number
        return NotImplemented

    def _binary_operation(cls, other: object) -> Any:  # noqa: ANN401
        if other is cls or isinstance(other, numbers.Real):
            return cls
        return NotImplemented

    __add__ = __radd__ = _binary_operation
    __sub__ = __rsub__ = _binary_operation
    __mul__ = __rmul__ = _binary_operation
    __truediv__ = __rtruediv__ = _binary_operation

    def _unary_operation(cls) -> Any:  # noqa: ANN401
        return cls

    __pos__ = __neg__ = _unary_operation


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


AllKnown: TypeAlias = Never


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase[Any], msg: str = "") -> None:
        _msg = f"Invalid quantum numbers for {ket!r}"
        if len(msg) > 0:
            _msg += f"\n  {msg}"
        super().__init__(_msg)


def is_angular_momentum_quantum_number(qn: str) -> TypeIs[AngularMomentumQuantumNumbers]:
    """Check if the given string is an AngularMomentumQuantumNumbers."""
    return qn in get_args(AngularMomentumQuantumNumbers)


def is_angular_operator_type(qn: str) -> TypeIs[AngularOperatorType]:
    """Check if the given string is an AngularOperatorType."""
    return qn in get_args(AngularOperatorType)


def is_unknown(obj: Any) -> TypeIs[Unknown]:  # noqa: ANN401
    """Check if obj is Unknown."""
    return obj is Unknown


def is_not_set(obj: Any) -> TypeIs[NotSet]:  # noqa: ANN401
    """Check if the obj is the NotSet singleton."""
    return obj is NotSet


ORBITAL_TO_STRING: dict[int, str] = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}


def get_spectroscopic_letter(l: int | Unknown) -> str:
    """Return the spectroscopic letter (s, p, d, f, ...) for the orbital angular momentum quantum number l.

    For l values without a spectroscopic letter in :data:`ORBITAL_TO_STRING` the fallback "(l={l})" is returned,
    and for an Unknown l the placeholder "(l=?)".
    """
    if is_unknown(l):
        return "(l=?)"
    if l != int(l) or l < 0:
        raise ValueError(f"get_spectroscopic_letter: Invalid orbital angular momentum quantum number {l=}.")
    return ORBITAL_TO_STRING.get(l, f"(l={int(l)})")


def format_quantum_number(value: float | Unknown) -> str:
    """Format a quantum number as string.

    Half-integers are formatted as fractions as long as the numerator is a single digit,
    i.e. ``abs(value) < 5`` (e.g. 1.5 -> '3/2'), and as decimals otherwise (e.g. 5.5 -> '5.5').
    """
    if is_unknown(value):
        return "?"
    if value % 1 == 0:
        return str(int(value))
    if (2 * value) % 1 == 0:
        if abs(value) < 5:
            return f"{int(2 * value)}/2"
        return f"{value:.1f}"
    raise ValueError(f"format_quantum_number: Invalid quantum number {value=}. Must be integer or half-integer.")


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

    If s_tot is None and cannot be uniquely determined from s_1 and s_2, return Unknown.
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
        return [Unknown]  # type: ignore [return-value]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]


def get_coupling_scheme_for_quantum_number(
    qn: AngularMomentumQuantumNumbers, preferred_schemes: list[CouplingScheme] | None = None
) -> CouplingScheme:
    """Return the coupling scheme, in which the given quantum number is a good quantum number."""
    preferred_schemes = preferred_schemes if preferred_schemes is not None else []
    preferred_schemes = [*preferred_schemes, "LS", "JJ", "FJ"]
    preferred_schemes = list(dict.fromkeys(preferred_schemes))  # remove duplicates while preserving order
    for scheme in preferred_schemes:
        if qn in get_quantum_number_names_for_coupling_scheme(scheme):
            return scheme
    raise ValueError(f"Quantum number {qn} is not a good quantum number in any of the coupling schemes.")


def get_quantum_number_names_for_coupling_scheme(
    coupling_scheme: CouplingScheme,
) -> tuple[AngularMomentumQuantumNumbers, ...]:
    """Return a list of quantum number names that are good quantum numbers in the given coupling scheme."""
    if coupling_scheme == "LS":
        from rydstate.angular.angular_ket import AngularKetLS  # noqa: PLC0415

        return AngularKetLS.quantum_number_names
    if coupling_scheme == "JJ":
        from rydstate.angular.angular_ket import AngularKetJJ  # noqa: PLC0415

        return AngularKetJJ.quantum_number_names
    if coupling_scheme == "FJ":
        from rydstate.angular.angular_ket import AngularKetFJ  # noqa: PLC0415

        return AngularKetFJ.quantum_number_names
    raise ValueError(f"Invalid coupling scheme {coupling_scheme}.")


def get_qn_name_from_operator(operator: AngularOperatorType) -> AngularMomentumQuantumNumbers:
    """Return the quantum number name corresponding to the given operator."""
    qn: str = operator
    if operator == "spherical":
        qn = "l_r"
    elif operator == "spherical_inner_valence":
        qn = "l_c"
    elif operator.startswith("identity_"):
        qn = operator.removeprefix("identity_")

    if not is_angular_momentum_quantum_number(qn):
        raise ValueError(f"Invalid operator {operator}.")
    return qn


@lru_cache(maxsize=1_000)
def quantum_numbers_to_angular_ket(
    species: str,
    s_c: float | None,
    l_c: int | None,
    j_c: float | None,
    f_c: float | None,
    s_r: float | None,
    l_r: int | None,
    j_r: float | None,
    s_tot: float | None,
    l_tot: int | None,
    j_tot: float | None,
    f_tot: float | None,
    m: float | NotSet = NotSet,
) -> AngularKetBase[Any]:
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
            s_c=s_c,
            l_c=l_c,
            s_r=s_r,
            l_r=l_r,
            s_tot=s_tot,
            l_tot=l_tot,
            j_tot=j_tot,
            f_tot=f_tot,
            m=m,
            species=species,
            allow_unknown=True,
        )

    with contextlib.suppress(InvalidQuantumNumbersError, ValueError):
        return AngularKetJJ(
            s_c=s_c,
            l_c=l_c,
            j_c=j_c,
            s_r=s_r,
            l_r=l_r,
            j_r=j_r,
            j_tot=j_tot,
            f_tot=f_tot,
            m=m,
            species=species,
            allow_unknown=True,
        )

    with contextlib.suppress(InvalidQuantumNumbersError, ValueError):
        return AngularKetFJ(
            s_c=s_c,
            l_c=l_c,
            j_c=j_c,
            f_c=f_c,
            s_r=s_r,
            l_r=l_r,
            j_r=j_r,
            f_tot=f_tot,
            m=m,
            species=species,
            allow_unknown=True,
        )

    raise ValueError("Invalid combination of angular quantum numbers provided.")
