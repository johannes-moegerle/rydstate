from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args, overload

from pint import UnitRegistry
from pint.facets.plain import PlainQuantity

from rydstate.angular.utils import AngularOperatorType

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy.typing as npt
    from pint.facets.plain import PlainUnit

    NDArray: TypeAlias = npt.NDArray[Any]
    PintFloat: TypeAlias = PlainQuantity[float]
    PintArray: TypeAlias = PlainQuantity[NDArray]
    PintComplex: TypeAlias = PlainQuantity[complex]

ureg: UnitRegistry[float] = UnitRegistry(system="atomic")


MatrixElementOperator = Literal[
    "magnetic_dipole",
    "electric_monopole",
    "electric_dipole",
    "electric_quadrupole",
    "electric_octupole",
    "electric_quadrupole_zero",
    AngularOperatorType,
]
MatrixElementPart = Literal["all", "rydberg", "inner_valence", "closed_shell_core"]
MatrixElementOperatorRanks: dict[MatrixElementOperator, tuple[int, int]] = {
    # "operator": (k_radial, k_angular)
    "magnetic_dipole": (0, 1),
    "electric_monopole": (0, 0),
    "electric_dipole": (1, 1),
    "electric_quadrupole": (2, 2),
    "electric_octupole": (3, 3),
    "electric_quadrupole_zero": (2, 0),
}

Dimension = Literal[
    MatrixElementOperator,
    "electric_field",
    "magnetic_field",
    "distance",
    "energy",
    "mass",
    "transition_rate",
    "charge",
    "velocity",
    "temperature",
    "time",
    "radial_matrix_element",
    "angular_matrix_element",
    "arbitrary",
    "zero",
]
DimensionLike = Dimension | tuple[Dimension, Dimension]

# some abbreviations: au_time: atomic_unit_of_time; au_current: atomic_unit_of_current; m_e: electron_mass
_CommonUnits: dict[Dimension, str] = {
    "electric_field": "V/cm",  # 1 V/cm ~ 1.94469e-10 bohr * m_e / au_current / au_time ** 3
    "magnetic_field": "T",  # 1 T ~ 4.25438e-06 m_e / au_current / au_time ** 2
    "distance": "micrometer",  # 1 mum ~ 1.88973e+04 bohr
    "energy": "hartree",  # 1 hartree = 1 bohr ** 2 * m_e / au_time ** 2
    "mass": "m_e",  # 1 m_e
    "charge": "e",  # 1 e = 1 au_current * au_time
    "velocity": "speed_of_light",  # 1 c ~ 1.37036e+02 bohr / au_time
    "temperature": "K",  # 1 K ~ 3.16681e-06 atomic_unit_of_temperature
    "time": "s",  # 1 s ~ 4.13414e+16 au_time
    "transition_rate": "1/s",  # 1 /s ~ 2.41888e-17 / au_time
    "radial_matrix_element": "bohr",  # 1 bohr
    "angular_matrix_element": "",  # 1 dimensionless
    "electric_monopole": "e",  # 1 e = 1 au_current * au_time
    "electric_dipole": "e * a0",  # 1 e * a0 = 1 au_current * au_time * bohr
    "electric_quadrupole": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "electric_quadrupole_zero": "e * a0^2",  # 1 e * a0^2 = 1 au_current * au_time * bohr ** 2
    "electric_octupole": "e * a0^3",  # 1 e * a0^3 = 1 au_current * au_time * bohr ** 3
    "magnetic_dipole": "bohr_magneton",  # 1 bohr_magneton = 0.5 au_current * bohr ** 2'
    "arbitrary": "",  # 1 dimensionless
    "zero": "",  # 1 dimensionless
}
# all angular operators are dimensionless
_CommonUnits.update(dict.fromkeys(get_args(AngularOperatorType), ""))

BaseUnits: dict[Dimension, PlainUnit] = {
    k: ureg.Quantity(1, unit).to_base_units().units for k, unit in _CommonUnits.items()
}
BaseQuantities: dict[Dimension, PintFloat] = {k: ureg.Quantity(1, unit) for k, unit in BaseUnits.items()}

Context = Literal["spectroscopy", "Gaussian"]
BaseContexts: dict[Dimension, Context] = {
    "magnetic_field": "Gaussian",
    "energy": "spectroscopy",
}


rydberg_constant_au = ureg.Quantity(1, "rydberg_constant").to(BaseUnits["energy"], "spectroscopy").m
electron_mass_u = ureg.Quantity(1, "electron_mass").to("u").m


def _contexts(dimension: Dimension | None) -> tuple[Context, ...]:
    """Return the pint contexts needed to convert the given dimension (e.g. "spectroscopy" for energies)."""
    if dimension is None:
        return ()
    context = BaseContexts.get(dimension)
    return () if context is None else (context,)


def user_to_au(value: PintFloat | float, unit: str | None, dimension: Dimension | None = None) -> float:
    """Convert a user-defined value + unit to a value in atomic units.

    Args:
        value: The value to convert. If unit is None, this must be a `pint.Quantity`.
        unit: The unit of the value. The special value "a.u." means the value is already given in atomic units.
        dimension: The physical dimension of the value. Only needed for dimensions that require a
            pint context to be converted (e.g. "energy", which may be given as a frequency or wavenumber).

    Returns:
        The value in atomic units.

    """
    if unit is None:
        if not isinstance(value, PlainQuantity) or value._REGISTRY is not ureg:  # noqa: SLF001
            raise ValueError("If unit is None, value must be a Pint Quantity created from `rydstate.units.ureg`.")
        quantity = value
    elif isinstance(value, PlainQuantity):
        raise ValueError("If unit is not None, value must be a float (not a Pint Quantity).")
    elif unit == "a.u.":
        return value
    else:
        quantity = ureg.Quantity(value, unit)

    if dimension is None:
        return float(quantity.to_base_units().magnitude)
    return float(quantity.to(BaseUnits[dimension], *_contexts(dimension)).magnitude)


@overload
def au_to_user(value_au: float, dimension: Dimension, unit: str) -> float: ...


@overload
def au_to_user(value_au: float, dimension: Dimension, unit: None) -> PintFloat: ...


@overload
def au_to_user(value_au: NDArray, dimension: Dimension, unit: str) -> NDArray: ...


@overload
def au_to_user(value_au: NDArray, dimension: Dimension, unit: None) -> PintArray: ...


def au_to_user(
    value_au: float | NDArray, dimension: Dimension, unit: str | None
) -> PintFloat | PintArray | float | NDArray:
    """Convert a value in atomic units to a user-defined unit.

    Args:
        value_au: The value in atomic units.
        dimension: The physical dimension of the value, used to look up the corresponding atomic unit
            (and the pint context needed to convert it, if any).
        unit: The unit to convert to. The special value "a.u." will return the value unchanged.

    Returns:
        The value in the desired unit.

    """
    if unit == "a.u.":
        return value_au

    quantity: PintFloat | PintArray = value_au * BaseQuantities[dimension]
    if unit is None:
        return quantity
    return quantity.to(unit, *_contexts(dimension)).magnitude
