from abc import ABC
from typing import ClassVar

from rydstate.species.element_properties import ElementProperties
from rydstate.units import electron_mass, rydberg_constant


class _ElementPropertiesYtterbiumAbstract(ElementProperties, ABC):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "4f14.6s"


class ElementPropertiesYtterbium171(_ElementPropertiesYtterbiumAbstract):
    species = "Yb171"
    i_c = 1 / 2

    _isotope_mass_u = 170.9363258
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=171&Z=70
    nuclear_dipole = 0.4923


class ElementPropertiesYtterbium173(_ElementPropertiesYtterbiumAbstract):
    species = "Yb173"
    i_c = 5 / 2

    _isotope_mass_u = 172.938216212
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=173&Z=70
    nuclear_dipole = -0.6780


class ElementPropertiesYtterbium174(_ElementPropertiesYtterbiumAbstract):
    species = "Yb174"
    i_c = 0

    _isotope_mass_u = 173.938859
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )
    nuclear_dipole = 0.0
