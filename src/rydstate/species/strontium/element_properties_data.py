from abc import ABC
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import Unknown
from rydstate.element_properties import ElementProperties


class _ElementPropertiesStrontiumAbstract(ElementProperties, ABC):
    Z = 38
    number_valence_electrons = 2
    ground_state_shell = (5, 0)
    additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    core_electron_configuration = "5s"


class ElementPropertiesStrontium87(_ElementPropertiesStrontiumAbstract):
    species = "Sr87"
    i_c = 9 / 2

    mass_number = 87
    atomic_mass_u = 86.90887749454

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=87&Z=38
    nuclear_dipole = -1.09316

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=4): (45932.287373577, "1/cm"),
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=5): (45932.120512528, "1/cm"),
    }
    # hyperfine centroid of the two F thresholds above, i.e. their (2F+1)-weighted mean
    # (9 * 45932.287373577 + 11 * 45932.120512528) / 20, which is the reference used for the Sr87 quantum defects
    # ("weighted threshold" of F. Robicheaux, J. Phys. B 52, 244001 (2019)).
    reference_ionization_threshold_tuple = (45932.1956, "1/cm")


class ElementPropertiesStrontium88(_ElementPropertiesStrontiumAbstract):
    species = "Sr88"
    i_c = 0

    mass_number = 88
    atomic_mass_u = 87.905612253
    nuclear_dipole = 0.0

    # The 5s threshold is taken from Couturier et al., Phys. Rev. A 99, 022503 (2019)
    # (https://doi.org/10.1103/PhysRevA.99.022503): I = 1_377_012_721(10) MHz (= 45932.20024 1/cm).
    # The Addendum of Vaillant et al. 2024 (https://doi.org/10.1088/1361-6455/ad76f0) uses the same value
    # rounded to 45932.2002 1/cm.
    # The 4d and 5p thresholds are the ones used in the Addendum (Table I), i.e. the 5s threshold plus the
    # Sr+ 4d_3/2, 4d_5/2, 5p_1/2 and 5p_3/2 excitation energies (Sansonetti 2012).
    # For LS-coupled channels (4dnl, 5pnp), the unweighted average of the fine structure thresholds is used.
    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=5, l_c=0, j_c=0.5): (1_377_012_721, "MHz"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=1.5): (60488.09, "1/cm"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=2.5): (60768.43, "1/cm"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=Unknown, label=Unknown): (60628.26, "1/cm"),
        CoreKet(i_c=0, n_c=5, l_c=1, j_c=Unknown, label=Unknown): (70048.11, "1/cm"),
    }
