from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialFei2009, PotentialMarinescu1994


class PotentialCoulombPotassium(PotentialCoulomb):
    species = "K"


class PotentialMarinescu1994Potassium(PotentialMarinescu1994):
    species = "K"
    is_default = True

    alpha_core_marinescu_1994 = 5.3310
    r_c_dict_marinescu_1994: ClassVar = {0: 0.83167545, 1: 0.85235381, 2: 0.83216907, 3: 6.50294371}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (3.56079437, 1.83909642, -1.74701102, -1.03237313),
        1: (3.65670429, 1.67520788, -2.07416615, -0.89030421),
        2: (4.12713694, 1.79837462, -1.69935174, -0.98913582),
        3: (1.42310446, 1.27861156, 4.77441476, -0.94829262),
    }


class PotentialFei2009Potassium(PotentialFei2009):
    species = "K"

    model_potential_parameter_fei_2009 = (0.9172, 4.1728, 0.6845, 0.2280)
