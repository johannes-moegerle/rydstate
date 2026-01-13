from __future__ import annotations

from typing import ClassVar

from rydstate.species.species_mqdt_object import SpeciesMQDTObject


class _StrontiumMQDTAbstract(SpeciesMQDTObject):
    Z = 38
    number_valence_electrons = 2
    ground_state_shell = (5, 0)
    _additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]

    _core_electron_configuration = "5s"

    potential_type_default = "model_potential_fei_2009"

    # Phys. Rev. A 89, 023426 (2014)
    alpha_c_marinescu_1993 = 7.5
    r_c_dict_marinescu_1993: ClassVar = {0: 1.59, 1: 1.58, 2: 1.57, 3: 1.56}
    model_potential_parameter_marinescu_1993: ClassVar = {
        0: (3.36124, 1.3337, 0, 5.94337),
        1: (3.28205, 1.24035, 0, 3.78861),
        2: (2.155, 1.4545, 0, 4.5111),
        3: (2.1547, 1.14099, 0, 2.1987),
    }

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.9959, 16.9567, 0.2648, 0.1439)


class StrontiumMQDT87(_StrontiumMQDTAbstract):
    name = "Sr87_mqdt"
    i_c = 9 / 2


class StrontiumMQDT88(_StrontiumMQDTAbstract):
    name = "Sr88_mqdt"
    i_c = 0
