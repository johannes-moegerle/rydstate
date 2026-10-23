from abc import ABC
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties

# https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Yb&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
_IONIZATION_THRESHOLD = (12.179185, "eV")


class _ElementPropertiesYtterbiumAbstractIon(ElementProperties, ABC):
    Z = 70
    net_charge = 2
    number_valence_electrons = 1
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "4f14"


class ElementPropertiesYtterbium174Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb174_ion"
    i_c = 0

    mass_number = 174
    atomic_mass_u = 173.938867545

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): _IONIZATION_THRESHOLD}
