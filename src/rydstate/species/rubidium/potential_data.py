from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialFei2009, PotentialMarinescu1994


class PotentialCoulombRubidium(PotentialCoulomb):
    species = "Rb"


class PotentialMarinescu1994Rubidium(PotentialMarinescu1994):
    species = "Rb"
    is_default = True

    alpha_core_marinescu_1994 = 9.076
    r_c_dict_marinescu_1994: ClassVar = {0: 1.66242117, 1: 1.50195124, 2: 4.86851938, 3: 4.79831327}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (3.69628474, 1.64915255, -9.86069196, 0.19579987),
        1: (4.44088978, 1.92828831, -16.79597770, -0.81633314),
        2: (3.78717363, 1.57027864, -11.6558897, 0.52942835),
        3: (2.39848933, 1.76810544, -12.0710678, 0.77256589),
    }


class PotentialFei2009Rubidium(PotentialFei2009):
    species = "Rb"

    model_potential_parameter_fei_2009 = (0.9708, 13.9706, 0.2909, 0.2215)
