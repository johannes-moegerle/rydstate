from rydstate.species.element_properties import ElementProperties


class ElementPropertiesSodium(ElementProperties):
    species = "Na"

    Z = 11
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (3, 0)
    core_electron_configuration = "2p6"

    corrected_rydberg_constant = (109734.69, "1/cm")

    alpha_closed_shell_core = 0.9448  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 3.18  # fitted to NIST matrix elements
