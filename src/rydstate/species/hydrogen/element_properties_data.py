import math

from rydstate.species.element_properties import ElementProperties


class ElementPropertiesHydrogen(ElementProperties):
    species = "H"

    Z = 1
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (1, 0)
    core_electron_configuration = "1s0"

    mass_number = 1
    atomic_mass_u = 1.007825031898


class ElementPropertiesHydrogenTextBook(ElementProperties):
    species = "H_textbook"

    Z = 1
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (1, 0)
    core_electron_configuration = "1s0"

    mass_number = 1
    # infinite nuclear mass, i.e. the textbook Rydberg constant R_infinity
    atomic_mass_u = math.inf
