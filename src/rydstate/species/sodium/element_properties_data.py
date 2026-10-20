from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties


class ElementPropertiesSodium(ElementProperties):
    species = "Na"

    Z = 11
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (3, 0)
    core_electron_configuration = "2p6"

    # Phys. Rev. A 45, 4720 (1992) (https://doi.org/10.1103/PhysRevA.45.4720)
    # Precision Stark spectroscopy of sodium: improved values for the ionization limit and bound states
    # I = 41449.451(2) 1/cm (the same reference as the quantum defects of the SQDT models)
    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): (41449.451, "1/cm")}

    mass_number = 23
    atomic_mass_u = 22.98976928195

    alpha_closed_shell_core = 0.9448  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 3.18  # fitted to NIST matrix elements
