from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties


class ElementPropertiesLithium(ElementProperties):
    species = "Li"

    Z = 3
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (2, 0)
    core_electron_configuration = "1s2"

    # Bushaw 2007, Phys. Rev. A 75, 052503 (https://doi.org/10.1103/PhysRevA.75.052503)
    # Ionization energy of 7Li determined by triple-resonance laser spectroscopy
    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): (43487.15940, "1/cm")}

    mass_number = 7
    atomic_mass_u = 7.01600343426

    alpha_closed_shell_core = 0.1923  # M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    r_c_dipole_operator = 3.04  # fitted to NIST matrix elements
