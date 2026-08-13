from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, TypeAlias, TypeVar

import numpy as np

RydbergRitzParameters: TypeAlias = tuple[float, ...] | list[float] | float


if TYPE_CHECKING:
    T = TypeVar("T", bound=type)
    F = TypeVar("F")

    def cache(func: F) -> F: ...


def calc_nu_from_energy(reduced_mass_au: float, energy_au: float, charge: int = 1) -> float:
    r"""Calculate the effective principal quantum number nu from a given energy.

    The effective principal quantum number is given by

    .. math::
        \nu
        = Z \sqrt{\frac{1}{2} \frac{R_M/R_\infty}{-E/E_H}}
        = Z \sqrt{\frac{1}{2} \frac{\mu/m_e}{-E/E_H}}

    where :math:`\mu/m_e` is the reduced mass in atomic units, :math:`E/E_H` the energy in atomic units,
    and :math:`Z` the net charge of the ionic core seen by the Rydberg electron.

    Args:
        reduced_mass_au: The reduced mass in atomic units (electron mass).
        energy_au: The energy in atomic units (hartree).
        charge: The net charge of the ionic core seen by the Rydberg electron. Default 1 (neutral atom).

    Returns:
        The effective principal quantum number nu.

    """
    return charge * math.sqrt(0.5 * reduced_mass_au / -energy_au)


def calc_energy_from_nu(reduced_mass_au: float, nu: float, charge: int = 1) -> float:
    r"""Calculate the energy from a given effective principal quantum number nu.

    The energy is given by

    .. math::
        E/E_H
        = -\frac{1}{2} \frac{Z^2 R_M/R_\infty}{\nu^2}
        = -\frac{1}{2} \frac{Z^2 \mu/m_e}{\nu^2}

    where :math:`\mu/m_e` is the reduced mass in atomic units, :math:`\nu` the effective principal
    quantum number, and :math:`Z` the net charge of the ionic core seen by the Rydberg electron.

    Args:
        reduced_mass_au: The reduced mass in atomic units :math:`\mu/m_e = \frac{m_{Core}}{m_{Core} + m_e}`.
        nu: The effective principal quantum number :math:`\nu`.
        charge: The net charge of the ionic core seen by the Rydberg electron. Default 1 (neutral atom).

    Returns:
        The energy E in atomic units (hartree).

    """
    return -0.5 * charge**2 * reduced_mass_au / nu**2


def calc_modified_ritz_formula(n: int, parameters: RydbergRitzParameters) -> float:
    """Calculate the modified Ritz formula: p₀ + p₁/(n - p₀)² + p₂/(n - p₀)⁴ + ...

    The parameters are given as a list [p₀, p₁, p₂, ...] or a single float (= p₀ and all other coefficients are zero).
    Note usually, for quantum defects, the formula is written with p₀ = δ₀, p₁ = δ₂, p₂ = δ₄, ...

    Args:
        n: The principal quantum number.
        parameters: Rydberg-Ritz parameters.
            A single float is a constant value; a list gives polynomial coefficients [p₀, p₁, p₂, ...].

    Returns:
        The value of the quantity at the given n.

    """
    if isinstance(parameters, (int, float)) or np.isscalar(parameters):
        return float(parameters)  # type: ignore [arg-type]
    if len(parameters) == 0:
        raise ValueError("Rydberg-Ritz parameters list cannot be empty.")

    p0 = parameters[0]
    result = p0
    for i, param in enumerate(parameters[1:], 1):
        result += param * 1.0 / (n - p0) ** (2 * i)
    return result


def calc_modified_ritz_formula_in_nu(nui: float, parameters: RydbergRitzParameters) -> float:
    """Calculate the modified Ritz formula for the effective principal quantum number nu: p₀ + p₁/ν² + p₂/ν⁴ + ...

    The parameters are given as a list [p₀, p₁, p₂, ...] or a single float (= p₀ and all other coefficients are zero).

    Args:
        nui: Channel-dependent effective principal quantum number.
        parameters: Rydberg-Ritz parameters.
            A single float is a constant value; a list gives polynomial coefficients [p₀, p₁, p₂, ...].

    Returns:
        The value of the quantity at the given nui.

    """
    if isinstance(parameters, (int, float)) or np.isscalar(parameters):
        return float(parameters)  # type: ignore [arg-type]
    if len(parameters) == 0:
        raise ValueError("Rydberg-Ritz parameters list cannot be empty.")

    result = 0.0
    for i, param in enumerate(parameters):
        result += param * 1.0 / nui ** (2 * i)
    return result


def get_all_subclasses(cls: T, species: str | None = None, tag: str | None = None) -> list[T]:
    """Get all subclasses of cls for the given species (and tag)."""
    subclasses: list[T] = []
    for subclass in cls.__subclasses__():
        subclasses.extend(get_all_subclasses(subclass, species, tag))
        if inspect.isabstract(subclass) or getattr(subclass, "species", None) is None:
            continue
        if species is not None and getattr(subclass, "species", None) != species:
            continue
        if tag is not None and getattr(subclass, "tag", None) != tag:
            continue
        subclasses.append(subclass)

    return subclasses
