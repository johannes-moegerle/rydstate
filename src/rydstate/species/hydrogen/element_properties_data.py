import math
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties
from rydstate.units import rydberg_constant_au


class ElementPropertiesHydrogen(ElementProperties):
    species = "H"

    Z = 1
    i_c = 0
    number_valence_electrons = 1
    ground_state_shell = (1, 0)
    core_electron_configuration = "1s0"

    mass_number = 1
    atomic_mass_u = 1.007825031898

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): (13.598_434_599_702, "eV")}


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

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): (rydberg_constant_au, "a.u.")}
