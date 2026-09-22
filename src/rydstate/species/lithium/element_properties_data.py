from rydstate.species.element_properties import ElementProperties


class ElementPropertiesLithium(ElementProperties):
    species = "Li"

    Z = 3
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (2, 0)
    core_electron_configuration = "1s2"

    mass_number = 7
    atomic_mass_u = 7.01600343426

    alpha_closed_shell_core = 0.1923  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 3.04  # fitted to NIST matrix elements
