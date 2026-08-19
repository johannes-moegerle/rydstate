from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import Unknown
from rydstate.species.fmodel import get_fmodels
from rydstate.species.mqdt import MQDT
from rydstate.species.ytterbium import (
    yb171_mqdt_fmodel_data,
    yb173_mqdt_fmodel_data,
    yb174_mqdt_fmodel_data,
)


class MQDTYtterbium171(MQDT):
    species = "Yb171"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0.5, l_c=0, j_c=0.5, f_c=0): (50442.795744, "1/cm"),
        CoreKet(i_c=0.5, l_c=0, j_c=0.5, f_c=1): (50443.217463, "1/cm"),
        CoreKet(i_c=0.5, l_c=1, j_c=0.5, f_c=Unknown, label=Unknown): (77504.98, "1/cm"),
        CoreKet(i_c=0.5, l_c=1, j_c=Unknown, f_c=Unknown, label=Unknown): (79725.35, "1/cm"),
        CoreKet(i_c=0.5, l_c=1, j_c=1.5, f_c=Unknown, label=Unknown): (80835.39, "1/cm"),
        CoreKet(i_c=0.5, l_c=Unknown, j_c=Unknown, f_c=Unknown, label="4f13 5d 6s"): (
            83967.7,
            "1/cm",
        ),
    }
    # ionization threshold of the F=1 core state (the upper of the two hyperfine thresholds)
    reference_ionization_threshold_tuple = (50443.217463, "1/cm")
    model_classes = get_fmodels(yb171_mqdt_fmodel_data, species)


class MQDTYtterbium173(MQDT):
    species = "Yb173"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=2.5, l_c=0, j_c=0.5, f_c=3): (50442.941262, "1/cm"),
        CoreKet(i_c=2.5, l_c=0, j_c=0.5, f_c=2): (50443.291203, "1/cm"),
        CoreKet(i_c=2.5, l_c=1, j_c=0.5, f_c=Unknown, label=Unknown): (77504.98, "1/cm"),
        CoreKet(i_c=2.5, l_c=1, j_c=1.5, f_c=Unknown, label=Unknown): (80835.39, "1/cm"),
        CoreKet(i_c=2.5, l_c=Unknown, j_c=Unknown, f_c=Unknown, label="4f13 5d 6s"): (
            83967.7,
            "1/cm",
        ),
    }
    # ionization threshold of the F=2 core state (the upper of the two hyperfine thresholds)
    reference_ionization_threshold_tuple = (50443.291203, "1/cm")
    model_classes = get_fmodels(yb173_mqdt_fmodel_data, species)


class MQDTYtterbium174(MQDT):
    species = "Yb174"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, l_c=0, j_c=0.5): (50443.070393, "1/cm"),
        CoreKet(i_c=0, l_c=1, j_c=0.5): (77504.98, "1/cm"),
        CoreKet(i_c=0, l_c=1, j_c=1.5): (80835.39, "1/cm"),
        CoreKet(i_c=0, l_c=1, j_c=Unknown, label=Unknown): (79725.35, "1/cm"),
        CoreKet(i_c=0, l_c=Unknown, j_c=Unknown, label="4f13 5d 6s"): (83967.7, "1/cm"),
    }
    model_classes = get_fmodels(yb174_mqdt_fmodel_data, species)
