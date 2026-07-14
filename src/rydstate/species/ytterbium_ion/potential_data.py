from typing import ClassVar

from rydstate.species.potential import PotentialCorePolarizabilityWithCutoff, PotentialCoulomb


class PotentialCoulombYtterbium171Ion(PotentialCoulomb):
    species = "Yb171_ion"


class PotentialCoulombYtterbium173Ion(PotentialCoulomb):
    species = "Yb173_ion"


class PotentialCoulombYtterbium174Ion(PotentialCoulomb):
    species = "Yb174_ion"


class _PotentialCorePolarizabilityWithCutoffYtterbiumIonAbstract(PotentialCorePolarizabilityWithCutoff):
    is_default = True
    # these values are taken from
    # Chen, Wu, Zhang, Tang, Jiang, Dong (2023), https://doi.org/10.1088/1674-1056/acbc6c
    # alpha_core^1 is the first-order (dipole) static polarizability of the Yb2+ core electrons (only the
    # dipole term is used, see Eq. (3)), and the cutoff radii rho_{l,j} (Table 1) were tuned to reproduce
    # the binding energies of Yb+.
    alpha_dict_core_polarizability_with_cutoff: ClassVar = {1: 7.72}
    rho_dict_core_polarizability_with_cutoff: ClassVar = {
        (0, 0.5): 2.4687,  # s_1/2
        (1, 0.5): 2.1196,  # p_1/2
        (1, 1.5): 1.9624,  # p_3/2
        (2, 1.5): 2.1842,  # d_3/2
        (2, 2.5): 2.1951,  # d_5/2
        (3, 2.5): 2.2400,  # f_5/2
        (3, 3.5): 1.7080,  # f_7/2
    }


class PotentialCorePolarizabilityWithCutoffYtterbium171Ion(_PotentialCorePolarizabilityWithCutoffYtterbiumIonAbstract):
    species = "Yb171_ion"


class PotentialCorePolarizabilityWithCutoffYtterbium173Ion(_PotentialCorePolarizabilityWithCutoffYtterbiumIonAbstract):
    species = "Yb173_ion"


class PotentialCorePolarizabilityWithCutoffYtterbium174Ion(_PotentialCorePolarizabilityWithCutoffYtterbiumIonAbstract):
    species = "Yb174_ion"
