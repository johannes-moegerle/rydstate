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
