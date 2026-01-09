from __future__ import annotations

import numpy as np


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
    """Check if the spin addition rule is satisfied.

    This means check the following conditions:
    - |s_1 - s_2| <= s_tot <= s_1 + s_2
    - s_1 + s_2 + s_tot is an integer
    """
    return abs(s_1 - s_2) <= s_tot <= s_1 + s_2 and (s_1 + s_2 + s_tot) % 1 == 0


def get_possible_quantum_number_values(s_1: float, s_2: float, s_tot: float | None) -> list[float]:
    """Determine a list of possible s_tot values from s_1 and s_2 if s_tot is not given, else return [s_tot]."""
    if s_tot is not None:
        return [s_tot]
    return [float(s) for s in np.arange(abs(s_1 - s_2), s_1 + s_2 + 1, 1)]
