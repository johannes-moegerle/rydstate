from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.species.fmodel import get_fmodels
from rydstate.species.mqdt import MQDT
from rydstate.species.strontium import sr87_mqdt_fmodel_data, sr88_mqdt_fmodel_data


class MQDTStrontium87(MQDT):
    species = "Sr87"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(4.5, 0.5, 0, 0.5, 4): (45932.287373577, "1/cm"),
        CoreKet(4.5, 0.5, 0, 0.5, 5): (45932.120512528, "1/cm"),
    }
    reference_ionization_threshold_tuple = (45932.287373577, "1/cm")
    model_classes = get_fmodels(sr87_mqdt_fmodel_data, species)


class MQDTStrontium88(MQDT):
    species = "Sr88"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(0, 0.5, 0, 0.5): (45932.1956, "1/cm"),
    }
    model_classes = get_fmodels(sr88_mqdt_fmodel_data, species)
