from abc import ABC
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import Unknown
from rydstate.element_properties import ElementProperties


class _ElementPropertiesYtterbiumAbstract(ElementProperties, ABC):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]
    core_electron_configuration = "4f14.6s"


class ElementPropertiesYtterbium171(_ElementPropertiesYtterbiumAbstract):
    species = "Yb171"
    core_radial_species = "Yb174_ion"
    i_c = 1 / 2

    mass_number = 171
    atomic_mass_u = 170.936331515

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=171&Z=70
    nuclear_dipole = 0.4923

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0.5, n_c=6, l_c=0, j_c=0.5, f_c=0): (50442.795744, "1/cm"),
        CoreKet(i_c=0.5, n_c=6, l_c=0, j_c=0.5, f_c=1): (50443.217463, "1/cm"),
        CoreKet(i_c=0.5, n_c=6, l_c=1, j_c=0.5, f_c=Unknown, label=Unknown): (77504.98, "1/cm"),
        CoreKet(i_c=0.5, n_c=6, l_c=1, j_c=Unknown, f_c=Unknown, label=Unknown): (79725.35, "1/cm"),
        CoreKet(i_c=0.5, n_c=6, l_c=1, j_c=1.5, f_c=Unknown, label=Unknown): (80835.39, "1/cm"),
        CoreKet(i_c=0.5, n_c=Unknown, l_c=Unknown, j_c=Unknown, f_c=Unknown, label="4f13"): (83967.7, "1/cm"),
    }
    # ionization threshold of the F=1 core state (the upper of the two hyperfine thresholds)
    reference_ionization_threshold_tuple = (50443.217463, "1/cm")


class ElementPropertiesYtterbium173(_ElementPropertiesYtterbiumAbstract):
    species = "Yb173"
    core_radial_species = "Yb174_ion"
    i_c = 5 / 2

    mass_number = 173
    atomic_mass_u = 172.938216211

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=173&Z=70
    nuclear_dipole = -0.6780

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=2.5, n_c=6, l_c=0, j_c=0.5, f_c=3): (50442.941262, "1/cm"),
        CoreKet(i_c=2.5, n_c=6, l_c=0, j_c=0.5, f_c=2): (50443.291203, "1/cm"),
        CoreKet(i_c=2.5, n_c=6, l_c=1, j_c=0.5, f_c=Unknown, label=Unknown): (77504.98, "1/cm"),
        CoreKet(i_c=2.5, n_c=6, l_c=1, j_c=1.5, f_c=Unknown, label=Unknown): (80835.39, "1/cm"),
        CoreKet(i_c=2.5, n_c=Unknown, l_c=Unknown, j_c=Unknown, f_c=Unknown, label="4f13"): (83967.7, "1/cm"),
    }
    # ionization threshold of the F=2 core state (the upper of the two hyperfine thresholds)
    reference_ionization_threshold_tuple = (50443.291203, "1/cm")


class ElementPropertiesYtterbium174(_ElementPropertiesYtterbiumAbstract):
    species = "Yb174"
    core_radial_species = "Yb174_ion"
    i_c = 0

    mass_number = 174
    atomic_mass_u = 173.938867545
    nuclear_dipole = 0.0

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=6, l_c=0, j_c=0.5): (50443.070393, "1/cm"),
        CoreKet(i_c=0, n_c=6, l_c=1, j_c=0.5): (77504.98, "1/cm"),
        CoreKet(i_c=0, n_c=6, l_c=1, j_c=1.5): (80835.39, "1/cm"),
        CoreKet(i_c=0, n_c=6, l_c=1, j_c=Unknown, label=Unknown): (79725.35, "1/cm"),
        CoreKet(i_c=0, n_c=Unknown, l_c=Unknown, j_c=Unknown, label="4f13"): (83967.7, "1/cm"),
    }
