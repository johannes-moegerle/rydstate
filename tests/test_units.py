import numpy as np
from rydstate import units
from rydstate.units import rydberg_constant_au, ureg


def test_constants() -> None:
    assert np.isclose(units.au_to_user(rydberg_constant_au, "energy", "1/cm"), 109737.31568157, rtol=1e-10, atol=1e-10)
    assert np.isclose(
        ureg.Quantity(1, "fine_structure_constant").to_base_units().magnitude,
        0.0072973525643394025,
        rtol=1e-10,
        atol=1e-10,
    )


def test_user_to_au_zero() -> None:
    for unit in ("1/cm", "GHz", "eV", "hartree"):
        assert units.user_to_au(0, unit, "energy") == 0
    assert units.user_to_au(0, "G", "magnetic_field") == 0
    assert np.isclose(units.user_to_au(0, "degC", "temperature"), units.user_to_au(273.15, "K", "temperature"))
    assert units.user_to_au(0, "nm", "energy") == np.inf  # a zero wavelength is an infinite energy


def test_au_to_user_zero() -> None:
    for unit in ("1/cm", "GHz", "eV", "hartree"):
        assert units.au_to_user(0.0, "energy", unit) == 0
    assert units.au_to_user(0.0, "magnetic_field", "G") == 0
    assert units.au_to_user(0.0, "energy", "nm") == np.inf  # a zero energy is an infinite wavelength
    assert type(units.au_to_user(0.0, "energy", "1/cm")) is float
    values = units.au_to_user(np.array([0.0, 1e-6]), "energy", "1/cm")
    assert values[0] == 0
    assert values[1] == units.au_to_user(1e-6, "energy", "1/cm")
