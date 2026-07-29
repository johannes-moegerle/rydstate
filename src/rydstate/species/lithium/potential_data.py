from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialFei2009, PotentialMarinescu1994


class PotentialCoulombLithium(PotentialCoulomb):
    species = "Li"


class PotentialMarinescu1994Lithium(PotentialMarinescu1994):
    species = "Li"
    is_default = True

    alpha_core_marinescu_1994 = 0.1923
    r_c_dict_marinescu_1994: ClassVar = {0: 0.61340824, 1: 0.61566441, 2: 2.34126273}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (2.47718079, 1.84150932, -0.02169712, -0.11988362),
        1: (3.45414648, 2.55151080, -0.21646561, -0.06990078),
        2: (2.51909839, 2.43712450, 0.32505524, 0.10602430),
    }


class PotentialFei2009Lithium(PotentialFei2009):
    species = "Li"

    model_potential_parameter_fei_2009 = (1.0255, 1.7402, 1.0543, 0.7165)
