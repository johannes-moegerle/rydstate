from __future__ import annotations

import inspect
import math
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    T = TypeVar("T", bound=type)


def get_all_subclasses(cls: T, species: str | None = None, tag: str | None = None) -> list[T]:
    """Get all subclasses of cls for the given species (and tag)."""
    subclasses: list[T] = []
    direct_subclasses: list[Any] = cls.__subclasses__()
    for subclass in direct_subclasses:
        subclasses.extend(get_all_subclasses(subclass, species, tag))
        if inspect.isabstract(subclass) or getattr(subclass, "species", None) is None:
            continue
        if species is not None and getattr(subclass, "species", None) != species:
            continue
        if tag is not None and getattr(subclass, "tag", None) != tag:
            continue
        subclasses.append(subclass)

    return subclasses


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
        energy_au: The energy in atomic units (Hartree).
        charge: The net charge of the ionic core seen by the Rydberg electron. Default 1 (neutral atom).

    Returns:
        The effective principal quantum number nu.

    """
    if energy_au >= 0:
        return np.inf
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
        The energy E in atomic units (Hartree).

    """
    if np.isinf(nu):
        raise ValueError("Cannot calculate energy from nu=inf.")
    if nu == 0:
        return -np.inf
    return -0.5 * charge**2 * reduced_mass_au / nu**2
