from typing import ClassVar

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesPotassium(ElementProperties):
    species = "K"

    Z = 19
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (4, 0)
    additional_allowed_shells: ClassVar = [(3, 2)]
    core_electron_configuration = "3p6"

    corrected_rydberg_constant = (109735.774, "1/cm")

    alpha_closed_shell_core = 5.331  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 2.3  # fitted to NIST matrix elements
