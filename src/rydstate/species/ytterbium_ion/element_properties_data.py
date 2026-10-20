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


class ElementPropertiesYtterbium171Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb171_ion"
    i_c = 1 / 2

    mass_number = 171
    atomic_mass_u = 170.936331515

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=171&Z=70
    nuclear_dipole = 0.4923

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=1 / 2, s_c=0, l_c=0): _IONIZATION_THRESHOLD}


class ElementPropertiesYtterbium173Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb173_ion"
    i_c = 5 / 2

    mass_number = 173
    atomic_mass_u = 172.938216211

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=173&Z=70
    nuclear_dipole = -0.6780

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=5 / 2, s_c=0, l_c=0): _IONIZATION_THRESHOLD}


class ElementPropertiesYtterbium174Ion(_ElementPropertiesYtterbiumAbstractIon):
    species = "Yb174_ion"
    i_c = 0

    mass_number = 174
    atomic_mass_u = 173.938867545

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): _IONIZATION_THRESHOLD}
