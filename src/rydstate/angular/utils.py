from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.species.species_object import SpeciesObject

CouplingScheme = Literal["LS", "JJ", "FJ"]


class InvalidQuantumNumbersError(ValueError):
    def __init__(self, ket: AngularKetBase, msg: str = "") -> None:
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


def try_trivial_spin_addition(s_1: float, s_2: float, s_tot: float | None, name: str) -> float:
    """Try to determine s_tot from s_1 and s_2 if it is not given.

    If s_tot is None and cannot be uniquely determined from s_1 and s_2, raise an error.
    Otherwise return s_tot or the trivial sum s_1 + s_2.
    """
    if s_tot is None:
        if s_1 != 0 and s_2 != 0:
            msg = f"{name} must be set if both parts ({s_1} and {s_2}) are non-zero."
            raise ValueError(msg)
        s_tot = s_1 + s_2
    return float(s_tot)


def check_spin_addition_rule(s_1: float, s_2: float, s_tot: float) -> bool:
    r"""Check if the spin addition rule is satisfied.

    This means check the following conditions:
    :math:`|s_1 - s_2| \leq s_{tot} \leq s_1 + s_2`
    and
    :math:`s_1 + s_2 + s_{tot}` is an integer
    """
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


def get_possible_quantum_number_values(s_1: float, s_2: float, s_tot: float | None) -> list[float]:
    """Determine a list of possible s_tot values from s_1 and s_2 if s_tot is not given, else return [s_tot]."""
    if s_tot is not None:
        return [s_tot]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]


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
    m: float | None = None,
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
