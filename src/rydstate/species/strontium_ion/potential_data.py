from typing import ClassVar

from rydstate.species.potential import PotentialCoulomb, PotentialMarinescu1994


class PotentialCoulombStrontium87Ion(PotentialCoulomb):
    species = "Sr87_ion"


class PotentialCoulombStrontium88Ion(PotentialCoulomb):
    species = "Sr88_ion"


class _PotentialMarinescu1994StrontiumIonAbstract(PotentialMarinescu1994):
    is_default = True
    # these values are taken from
    # Greene, Aymar (1991), https://doi.org/10.1103/PhysRevA.44.1773
    # Note that the potential there is defined with Marinescu a_j = Greene \alpha_i as follows:
    # a_1 = \alpha_1
    # a_2 = \alpha_3
    # a_3 = -\alpha_2 (Note the minus sign!)
    # and a_4 = 0
    #
    # Exception: the l = 3 parameters (a_1, a_2) were refitted here, see below.
    # The original Greene, Aymar l = 3 values (5.3540, 5.6624, -7.9517, 0)
    # did not produce nice wavefunctions for the given quantum defects,
    # so a_1 and a_2 were refitted (a_3 has no measurable influence on delta_f)
    # to reproduce the quantum defects that the database uses.
    # This changes the stored radial matrix elements by at most 0.8 %,
    # and only for elements that are themselves ~5 orders of magnitude below the median of their table.
    alpha_core_marinescu_1994 = 7.5
    r_c_dict_marinescu_1994: ClassVar = {0: 1.7965, 1: 1.3960, 2: 1.6820, 3: 1.0057}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (3.4187, 1.5915, -4.7332, 0),
        1: (3.3235, 1.5712, -2.2539, 0),
        2: (3.2533, 1.5996, -3.2330, 0),
        3: (3.8700, 2.6800, -7.9517, 0),  # a_1, a_2 refitted (see above), a_3 as in Greene, Aymar (1991)
    }


class PotentialMarinescu1994Strontium87Ion(_PotentialMarinescu1994StrontiumIonAbstract):
    species = "Sr87_ion"


class PotentialMarinescu1994Strontium88Ion(_PotentialMarinescu1994StrontiumIonAbstract):
    species = "Sr88_ion"
