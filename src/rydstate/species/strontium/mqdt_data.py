from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.species.eigen_channel_model import get_model_classes
from rydstate.species.mqdt import MQDT
from rydstate.species.strontium import sr87_eigen_channel_model_data, sr88_eigen_channel_model_data


class MQDTStrontium87(MQDT):
    species = "Sr87"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=4): (45932.287373577, "1/cm"),
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=5): (45932.120512528, "1/cm"),
    }
    # hyperfine centroid of the two F thresholds above, i.e. their (2F+1)-weighted mean
    # (9 * 45932.287373577 + 11 * 45932.120512528) / 20, which coincides with the Sr88
    # ionization threshold and is the reference used for the Sr87 quantum defects.
    reference_ionization_threshold_tuple = (45932.1956, "1/cm")
    model_classes = get_model_classes(sr87_eigen_channel_model_data, species)


class MQDTStrontium88(MQDT):
    species = "Sr88"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=5, l_c=0, j_c=0.5): (45932.1956, "1/cm"),
    }
    model_classes = get_model_classes(sr88_eigen_channel_model_data, species)
