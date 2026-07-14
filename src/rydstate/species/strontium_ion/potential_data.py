from typing import ClassVar

from rydstate.species.potential import (
    PotentialCorePolarizabilityWithCutoff,
    PotentialCoulomb,
    PotentialMarinescu1994,
)


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
    alpha_c_marinescu_1994 = 7.5
    r_c_dict_marinescu_1994: ClassVar = {0: 1.7965, 1: 1.3960, 2: 1.6820, 3: 1.0057}
    model_potential_parameter_marinescu_1994: ClassVar = {
        0: (3.4187, 1.5915, -4.7332, 0),
        1: (3.3235, 1.5712, -2.2539, 0),
        2: (3.2533, 1.5996, -3.2330, 0),
        3: (5.3540, 5.6624, -7.9517, 0),
    }


class PotentialMarinescu1994Strontium87Ion(_PotentialMarinescu1994StrontiumIonAbstract):
    species = "Sr87_ion"


class PotentialMarinescu1994Strontium88Ion(_PotentialMarinescu1994StrontiumIonAbstract):
    species = "Sr88_ion"


class _PotentialCorePolarizabilityWithCutoffStrontiumIonAbstract(PotentialCorePolarizabilityWithCutoff):
    # these values are taken from
    # Jiang, Mitroy, Cheng, Bromley (2016), https://arxiv.org/abs/1605.05040
    # alpha_c is the first-order (dipole) static polarizability of the Sr2+ core (their alpha_core^1),
    # and the cutoff radii rho_{l,j} (Table III) were tuned to reproduce the Sr+ energies including the
    # spin-orbit splittings. For l > 3 the cutoff falls back to the largest tabulated l of the same j-branch.
    reference: ClassVar[str] = "J. Jiang et al. (2016), Phys. Rev. A 94, 062514, https://arxiv.org/abs/1605.05040"
    alpha_c_core_polarizability_with_cutoff = 5.813
    rho_dict_core_polarizability_with_cutoff: ClassVar = {
        (0, 0.5): 2.04960,  # s_1/2
        (1, 0.5): 1.97169,  # p_1/2
        (1, 1.5): 1.97600,  # p_3/2
        (2, 1.5): 2.35353,  # d_3/2
        (2, 2.5): 2.36534,  # d_5/2
        (3, 2.5): 2.15023,  # f_5/2
        (3, 3.5): 2.19469,  # f_7/2
    }


class PotentialCorePolarizabilityWithCutoffStrontium87Ion(_PotentialCorePolarizabilityWithCutoffStrontiumIonAbstract):
    species = "Sr87_ion"


class PotentialCorePolarizabilityWithCutoffStrontium88Ion(_PotentialCorePolarizabilityWithCutoffStrontiumIonAbstract):
    species = "Sr88_ion"
