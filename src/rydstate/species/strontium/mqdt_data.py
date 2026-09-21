from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.species.mqdt import MQDT
from rydstate.species.mqdt_model import get_model_classes
from rydstate.species.strontium import sr87_eigen_channel_model_data, sr88_eigen_channel_model_data


class MQDTStrontium87(MQDT):
    species = "Sr87"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=4): (45932.287373577, "1/cm"),
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=5): (45932.120512528, "1/cm"),
    }
    # hyperfine centroid of the two F thresholds above, i.e. their (2F+1)-weighted mean
    # (9 * 45932.287373577 + 11 * 45932.120512528) / 20, which is the reference used for the Sr87 quantum defects
    # ("weighted threshold" of F. Robicheaux, J. Phys. B 52, 244001 (2019)).
    reference_ionization_threshold_tuple = (45932.1956, "1/cm")
    model_classes = get_model_classes(sr87_eigen_channel_model_data, species)


class MQDTStrontium88(MQDT):
    species = "Sr88"
    is_default = True

    # Couturier et al., Phys. Rev. A 99, 022503 (2019) (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.022503)
    # I = 1_377_012_721(10) MHz (= 45932.2002 1/cm), same as is used in the Addendum of
    # Vaillant et al. 2024 (https://doi.org/10.1088/1361-6455/ad76f0)
    # Since in sr88_eigen_channel_model_data only this single threshold is used,
    # changing the value here (compared to F. Robicheaux 2019) does not affect the models.
    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=5, l_c=0, j_c=0.5): (1_377_012_721, "MHz"),
    }
    model_classes = get_model_classes(sr88_eigen_channel_model_data, species)
