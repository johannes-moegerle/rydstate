from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialFei2009, PotentialMarinescu1994


class PotentialCoulombSodium(PotentialCoulomb):
    species = "Na"


class PotentialMarinescu1994Sodium(PotentialMarinescu1994):
    species = "Na"
    is_default = True

    alpha_core_marinescu_1994 = 0.9448
    r_c_dict_marinescu_1994: ClassVar = {0: 0.45489422, 1: 0.45798739, 2: 0.71875312, 3: 28.6735059}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (4.82223117, 2.45449865, -1.12255048, -1.42631393),
        1: (5.08382502, 2.18226881, -1.19534623, -1.03142861),
        2: (3.53324124, 2.48697936, -0.75688448, -1.27852357),
        3: (1.11056646, 1.05458759, 1.73203428, -0.09265696),
    }


class PotentialFei2009Sodium(PotentialFei2009):
    species = "Na"

    model_potential_parameter_fei_2009 = (0.9729, 2.5434, 1.0406, 0.4685)
