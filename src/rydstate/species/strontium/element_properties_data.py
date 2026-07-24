from abc import ABC
from typing import ClassVar

from rydstate.species.element_properties import ElementProperties
from rydstate.units import electron_mass, rydberg_constant


class _ElementPropertiesStrontiumAbstract(ElementProperties, ABC):
    Z = 38
    number_valence_electrons = 2
    ground_state_shell = (5, 0)
    additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    core_electron_configuration = "5s"


class ElementPropertiesStrontium87(_ElementPropertiesStrontiumAbstract):
    species = "Sr87"
    i_c = 9 / 2

    _isotope_mass_u = 86.9088774970
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=87&Z=38
    nuclear_dipole = -1.09316


class ElementPropertiesStrontium88(_ElementPropertiesStrontiumAbstract):
    species = "Sr88"
    i_c = 0

    _isotope_mass_u = 87.9056122571
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )
    nuclear_dipole = 0.0
