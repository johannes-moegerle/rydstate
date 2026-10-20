from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties


class ElementPropertiesCesium(ElementProperties):
    species = "Cs"

    Z = 55
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(4, 3), (5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "5p6"

    # Phys. Rev. A 93, 013424 (2016) (https://doi.org/10.1103/PhysRevA.93.013424),
    # as corrected by the Erratum Phys. Rev. A 112, 049902 (2025) (https://doi.org/10.1103/5v5y-8m53)
    # I = 31406.4677482(10)stat(34)syst 1/cm.
    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): (31406.4677482, "1/cm")}

    mass_number = 133
    atomic_mass_u = 132.905451958

    alpha_closed_shell_core = 15.6440  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 3.42  # fitted to NIST matrix elements
