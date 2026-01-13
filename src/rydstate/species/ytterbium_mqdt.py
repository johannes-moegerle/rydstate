from __future__ import annotations

from typing import ClassVar

from rydstate.species.species_mqdt_object import SpeciesMQDTObject


class _YtterbiumMQDTAbstract(SpeciesMQDTObject):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    _additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]

    _core_electron_configuration = "4f14.6s"

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)


class YtterbiumMQDT171(_YtterbiumMQDTAbstract):
    name = "Yb171_mqdt"
    i_c = 1 / 2


class YtterbiumMQDT173(_YtterbiumMQDTAbstract):
    name = "Yb173_mqdt"
    i_c = 5 / 2


class YtterbiumMQDT174(_YtterbiumMQDTAbstract):
    name = "Yb174_mqdt"
    i_c = 0
