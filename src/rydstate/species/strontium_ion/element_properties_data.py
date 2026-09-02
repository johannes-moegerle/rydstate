from abc import ABC
from typing import ClassVar

from rydstate.species.element_properties import ElementProperties
from rydstate.units import electron_mass, rydberg_constant


class _ElementPropertiesStrontiumAbstractIon(ElementProperties, ABC):
    Z = 38
    net_charge = 2
    number_valence_electrons = 1
    ground_state_shell = (5, 0)
    additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    core_electron_configuration = "4p6"

    # Greene, Aymar Phys. Rev. A 44, 1773, 1991 (https://doi.org/10.1103/PhysRevA.44.1773)
    alpha_closed_shell_core = 7.5
    # fitted to the NIST ASD line strengths of Sr II with accuracy AA / A+ (5s1/2-5p1/2, 5s1/2-5p3/2, 4d3/2-5p1/2),
    # which are reproduced within 1% (the bare dipole operator overestimates them by 7-8%)
    r_c_dipole_operator = 3.46


class ElementPropertiesStrontium87Ion(_ElementPropertiesStrontiumAbstractIon):
    species = "Sr87_ion"
    i_c = 9 / 2

    _isotope_mass_u = 86.9088774970
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=87&Z=38
    nuclear_dipole = -1.09316


class ElementPropertiesStrontium88Ion(_ElementPropertiesStrontiumAbstractIon):
    species = "Sr88_ion"
    i_c = 0

    _isotope_mass_u = 87.9056122571
    corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass_u),
        str(rydberg_constant.u),
    )
