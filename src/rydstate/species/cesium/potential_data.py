from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialFei2009, PotentialMarinescu1994


class PotentialCoulombCesium(PotentialCoulomb):
    species = "Cs"


class PotentialMarinescu1994Cesium(PotentialMarinescu1994):
    species = "Cs"
    is_default = True

    alpha_core_marinescu_1994 = 15.6440
    r_c_dict_marinescu_1994: ClassVar = {0: 1.92046930, 1: 2.13383095, 2: 0.93007296, 3: 1.99969677}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (3.49546309, 1.47533800, -9.72143084, 0.02629242),
        1: (4.69366096, 1.71398344, -24.65624280, -0.09543125),
        2: (4.32466196, 1.61365288, -6.70128850, -0.74095193),
        3: (3.01048361, 1.40000001, -3.20036138, 0.00034538),
    }


class PotentialFei2009Cesium(PotentialFei2009):
    species = "Cs"

    model_potential_parameter_fei_2009 = (0.9447, 14.7149, 0.2944, 0.1934)
