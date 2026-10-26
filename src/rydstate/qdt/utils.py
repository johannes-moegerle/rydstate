from __future__ import annotations

from typing import TypeAlias

ExpansionCoefficients: TypeAlias = list[float] | tuple[float, ...]


def check_expansion_coefficients(coefficients: ExpansionCoefficients, name: str = "Expansion coefficients") -> None:
    """Check that the expansion coefficients are a non-empty list (or tuple) of floats.

    This is meant to be called once when a model is constructed, so that the functions evaluating
    the expansions can simply assume a non-empty sequence of coefficients.

    Args:
        coefficients: The expansion coefficients to check.
        name: How to refer to the checked coefficients in the error message.

    Raises:
        TypeError: If the coefficients are not a list or tuple.
        ValueError: If the coefficients are an empty list.

    """
    if not isinstance(coefficients, (list, tuple)):
        raise TypeError(
            f"{name} must be a list (or tuple) of floats, got {type(coefficients).__name__}. "
            "A constant value must be given as a single element list [p₀]."
        )
    if len(coefficients) == 0:
        raise ValueError(f"{name} must not be an empty list.")


def calc_polynomial(x: float, coefficients: ExpansionCoefficients) -> float:
    """Calculate the polynomial p₀ + p₁ x + p₂ x² + ... for the given expansion coefficients.

    The polynomial is evaluated via Horner's method, i.e. rewritten as p₀ + x (p₁ + x (p₂ + ...)),
    which needs only one multiplication and one addition per coefficient and avoids computing any powers.

    Args:
        x: The value at which to evaluate the polynomial.
        coefficients: The expansion coefficients [p₀, p₁, p₂, ...].

    Returns:
        The value of the polynomial at x.

    """
    result = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        result = result * x + coefficient
    return result


def calc_modified_ritz_formula(n: int, coefficients: ExpansionCoefficients) -> float:
    """Calculate the modified Ritz formula: p₀ + p₁/(n - p₀)² + p₂/(n - p₀)⁴ + ...

    This is the polynomial p₀ + p₁ x + p₂ x² + ... evaluated at x = 1/(n - p₀)².
    The coefficients are given as a non-empty list [p₀, p₁, p₂, ...]; a constant value is a single element
    list [p₀]. They are not validated here, see :func:`check_expansion_coefficients`.
    Note usually, for quantum defects, the formula is written with p₀ = δ₀, p₁ = δ₂, p₂ = δ₄, ...

    Args:
        n: The principal quantum number.
        coefficients: The expansion coefficients [p₀, p₁, p₂, ...].

    Returns:
        The value of the quantity at the given n.

    """
    return calc_polynomial(1.0 / (n - coefficients[0]) ** 2, coefficients)


def calc_modified_ritz_formula_in_nu(nui: float, coefficients: ExpansionCoefficients) -> float:
    """Calculate the modified Ritz formula for the effective principal quantum number nu: p₀ + p₁/ν² + p₂/ν⁴ + ...

    This is the polynomial p₀ + p₁ x + p₂ x² + ... evaluated at x = 1/ν².
    The coefficients are given as a non-empty list [p₀, p₁, p₂, ...]; a constant value is a single element
    list [p₀]. They are not validated here, see :func:`check_expansion_coefficients`.

    Args:
        nui: Channel-dependent effective principal quantum number.
        coefficients: The expansion coefficients [p₀, p₁, p₂, ...].

    Returns:
        The value of the quantity at the given nui.

    """
    return calc_polynomial(1.0 / nui**2, coefficients)
