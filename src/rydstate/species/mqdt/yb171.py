from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.angular_ket_dummy import AngularKetDummy
from rydstate.angular.core_ket_base import CoreKet, CoreKetDummy
from rydstate.angular.utils import Unknown
from rydstate.species.mqdt.fmodel import FModel
from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT
from rydstate.units import electron_mass, rydberg_constant


class Ytterbium171MQDT(SpeciesObjectMQDT):
    name = "Yb171_mqdt"
    Z = 70
    i_c = 0.5
    number_valence_electrons = 2

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    _isotope_mass = 170.9363258  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    _ionization_threshold_dict = {
        CoreKet(i_c, 0.5, 0, 0.5, 0): (50442.795744, None, "1/cm"),
        CoreKet(i_c, 0.5, 0, 0.5, 1): (50443.217463, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 0.5, Unknown): (77504.98, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, Unknown, Unknown): (79725.35, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 1.5, Unknown): (80835.39, None, "1/cm"),
        CoreKetDummy("4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5, 1)
    nuclear_dipole = 0.49367


# --------------------------------------------------------
# MQDT models valid at large n
# --------------------------------------------------------


class Yb171_S05_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "S F=0.5, nu > 26"
    f_tot = 0.5
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817, 10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, f_tot=0.5, species="Yb171"),  # "6sns 1S0"
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, f_tot=0.5, species="Yb171"),  # "6pnp 1S0"
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=0.5, species="Yb171"),  # "6pnp 3P0"
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=0, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=1.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=0.5, f_c=Unknown, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=1, j_r=0.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.357488757, 0.165981371, 0, 0, 0],
        [0.203918644, 0, 0, 0, 0],
        [0.116819032, 0, 0, 0, 0],
        [0.287350241, 0, 0, 0, 0],
        [0.247621114, 0, 0, 0, 0],
        [0.148681324, 0, 0, 0, 0],
        [0.438542187, 3.78366407, -10709.7378, 8054542.58, -2523011670],
    ]
    mixing_angles = [
        (0, 1, 0.131755467),
        (0, 2, 0.297504211),
        (0, 3, 0.055421439),
        (2, 3, 0.100871756),
        (2, 4, 0.103123032),
        (0, 5, 0.137753117),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, np.sqrt(2 / 3), 0, -np.sqrt(1 / 3), 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, np.sqrt(1 / 3), 0, np.sqrt(2 / 3), 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -1 / 2],
        ]
    )


class Yb171_S15_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "S F=1.5, nu > 26"
    f_tot = 1.5
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=1, j_r=0.5, f_tot=1.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]
    mixing_angles = []


class Yb171_P05_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=0.5, nu > 5.7"
    f_tot = 0.5
    nu_range = (5.7, np.inf)
    reference = "10.1103/PhysRevA.112.042817, 10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=0.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=0.5, species="Yb171"),  # "6snp 3P0"
        AngularKetDummy("4f13 5d 6snl e", f_tot=0.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=0.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=0.5),
    ]

    eigen_quantum_defects = [
        [0.922094502, 2.12370136],
        [0.981191543, -4.54209175],
        [0.229094016, 0],
        [0.206073107, 0],
        [0.193527627, 0],
        [0.181165673, 0],
        [0.953185132, 0.0277444042],
        [0.198448494, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.102285383, 153.521338, -15393.2283]),
        (1, 6, -0.00168607392),
        (0, 2, -0.0719467433),
        (0, 3, -0.0673315968),
        (0, 4, -0.0221077377),
        (0, 5, -0.107638329),
        (1, 2, 0.0416653549),
        (1, 3, 0.0590660991),
        (1, 4, 0.0861585559),
        (1, 5, 0.0566417469),
        (6, 7, 0.163113423),
    ]


class Yb171_P15_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=1.5, nu > 10"
    f_tot = 1.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snf 3F2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=2.5, f_tot=1.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.922094502, 2.12370136, 0],
        [0.981191543, -4.54209175, 0],
        [0.229094016, 0, 0],
        [0.206073107, 0, 0],
        [0.193527627, 0, 0],
        [0.181165673, 0, 0],
        [0.925345494, -3.23594086, 80.2535181],
        [0.232649227, 0, 0],
        [0.210070444, 0, 0],
        [0.185699031, 0, 0],
        [0.0718955585, -1.0913707, -38.4618954],
    ]
    mixing_angles = [
        (0, 1, [-0.102285383, 153.251338, -15393.2283]),
        (0, 2, -0.0719467433),
        (0, 3, -0.0673315968),
        (0, 4, -0.0221077377),
        (0, 5, -0.107638329),
        (1, 2, 0.0416653549),
        (1, 3, 0.0590660991),
        (1, 4, 0.0861585559),
        (1, 5, 0.0566417469),
        (6, 7, 0.0703574701),
        (6, 8, 0.0235308506),
        (6, 9, -0.0295876723),
        (6, 10, 0.018377516),
    ]


class Yb171_D05_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=0.5, nu > 30"
    f_tot = 0.5
    nu_range = (30.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.75258093, 0.382628525, -483.120633],
    ]
    mixing_angles = []


class Yb171_D15_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=1.5, nu > 30"
    f_tot = 1.5
    nu_range = (30.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817, 10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snd 3D2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, f_tot=1.5, species="Yb171"),  # "6pnp 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=Unknown, f_c=Unknown, j_r=Unknown, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=0, j_r=1.5, f_tot=1.5, species="Yb171"),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [-np.sqrt(3 / 5), -np.sqrt(2 / 5), 0, 0, 0, 0],
            [np.sqrt(3 / 5) / 2, -3 / (2 * np.sqrt(10)), 0, 0, 0, np.sqrt(5 / 2) / 2],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [-1 / 2, np.sqrt(3 / 2) / 2, 0, 0, 0, np.sqrt(3 / 2) / 2],
        ]
    )

    eigen_quantum_defects = [
        [0.73056016, -0.108286264, 0],
        [0.75155852, 0.000367204397, 0],
        [0.195831577, 0, 0],
        [0.236133225, 0, 0],
        [0.147506921, 0, 0],
        [0.75336354, -1.84349555, 994.210321],
    ]
    mixing_angles = [
        (0, 1, [0.22146327, -16.2798928]),
        (0, 2, 0.00431695191),
        (0, 3, 0.0381576181),
        (1, 3, -0.00708200703),
        (0, 4, 0.109346659),
        (1, 4, 0.0636016813),
    ]


class Yb171_D25_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=2.5, nu > 30"
    f_tot = 2.5
    nu_range = (30.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817, 10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snd 3D2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, f_tot=2.5, species="Yb171"),  # "6pnp 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=2.5, species="Yb171"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=2.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=2.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=Unknown, f_c=Unknown, j_r=Unknown, f_tot=2.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=0, j_r=2.5, f_tot=2.5, species="Yb171"),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [np.sqrt(7 / 5) / 2, np.sqrt(7 / 30), 0, 0, 0, -np.sqrt(5 / 3) / 2],
            [-np.sqrt(2 / 5), np.sqrt(3 / 5), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1 / 2, np.sqrt(1 / 6), 0, 0, 0, np.sqrt(7 / 3) / 2],
        ]
    )

    eigen_quantum_defects = [
        [0.73056016, -0.108286264, 0],
        [0.75155852, 0.000367204397, 0],
        [0.195831577, 0, 0],
        [0.236133225, 0, 0],
        [0.147506921, 0, 0],
        [0.72861481, 0.79979111, -484.236631],
    ]
    mixing_angles = [
        (0, 1, [0.22146327, -16.2798928]),
        (0, 2, 0.00431695191),
        (0, 3, 0.0381576181),
        (1, 3, -0.00708200703),
        (0, 4, 0.109346659),
        (1, 4, 0.0636016813),
    ]


class Yb171_D35_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=3.5, nu > 14"
    f_tot = 3.5
    nu_range = (14.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=3.5, species="Yb171"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=3.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.72895315, -0.20653489, 220.484722],
    ]
    mixing_angles = []


class Yb171_F25_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "F F=2.5, nu > 20"
    f_tot = 2.5
    nu_range = (20.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, f_tot=2.5, species="Yb171"),  # "6snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, f_tot=2.5, species="Yb171"),  # "6snf 3F3"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl e", f_tot=2.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snf 3P2"
        AngularKetDummy("4f13 5d 6snl f", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl h", f_tot=2.5),
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snf 3F2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=3.5, f_tot=2.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=2.5, f_tot=2.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl e", f_tot=2.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=2.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl f", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl h", f_tot=2.5),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=0, j_r=2.5, f_tot=2.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.277086649, -13.2829133, 0],
        [0.0719837014, -0.741492064, 0],
        [0.251457795, 0, 0],
        [0.227434828, 0, 0],
        [0.175780645, 0, 0],
        [0.196547521, 0, 0],
        [0.21440857, 0, 0],
        [0.925345494, -3.23594086, 80.2535181],
        [0.232649227, 0, 0],
        [0.210070444, 0, 0],
        [0.185699031, 0, 0],
        [0.0718955585, -1.0913707, -38.4618954],
    ]
    mixing_angles = [
        (0, 1, [-0.0209955122, 0.251041249]),
        (0, 2, -0.0585753224),
        (0, 3, -0.0750574327),
        (0, 4, 0.122671919),
        (0, 5, -0.0401036164),
        (0, 6, 0.0654271994),
        (1, 2, -0.0683007974),
        (1, 3, 0.035415976),
        (1, 4, -0.0327625807),
        (1, 5, -0.050225071),
        (1, 6, 0.0455759316),
        (7, 8, 0.0703574701),
        (7, 9, 0.0235308506),
        (7, 10, -0.0295876723),
        (7, 11, 0.018377516),
    ]


class Yb171_F35_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "F F=3.5, nu > 20"
    f_tot = 3.5
    nu_range = (20.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, f_tot=3.5, species="Yb171"),  # "6snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, f_tot=3.5, species="Yb171"),  # "6snf 3F3"
        AngularKetDummy("4f13 5d 6snl a", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl e", f_tot=3.5),
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, f_tot=3.5, species="Yb171"),  # "6snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=3.5, f_tot=3.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=2.5, f_tot=3.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl e", f_tot=3.5),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=0, j_r=3.5, f_tot=3.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.277086649, -13.290196, 0],
        [0.0719837014, -0.754736076, 0],
        [0.251457795, 0, 0],
        [0.227434828, 0, 0],
        [0.175780645, 0, 0],
        [0.196547521, 0, 0],
        [0.21440857, 0, 0],
        [0.0834193873, -1.11453386, -1545.71844],
    ]
    mixing_angles = [
        (0, 1, [0.0209955122, 0.251041249]),
        (0, 2, -0.0585753224),
        (0, 3, -0.0750574327),
        (0, 4, 0.122671919),
        (0, 5, -0.0401036164),
        (0, 6, 0.0654271994),
        (1, 2, -0.0683007974),
        (1, 3, 0.035415976),
        (1, 4, -0.0327625807),
        (1, 5, -0.050225071),
        (1, 6, 0.0455759316),
    ]


class Yb171_F45_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "F F=4.5, nu > 20"
    f_tot = 4.5
    nu_range = (20.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, f_tot=4.5, species="Yb171"),  # "6snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=1, j_r=3.5, f_tot=4.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.0834193873, -1.11453386, -1545.71844],
    ]
    mixing_angles = []


class Yb171_G25_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "G F=2.5, nu > 25"
    f_tot = 2.5
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=3, f_tot=2.5, species="Yb171"),  # "6sng 3G3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=3.5, f_tot=2.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.02613255, -0.14203905],
    ]
    mixing_angles = []


class Yb171_G35_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "G F=3.5, nu > 25"
    f_tot = 3.5
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, j_tot=4, f_tot=3.5, species="Yb171"),  # "6sng +G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, j_tot=4, f_tot=3.5, species="Yb171"),  # "6sng -G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, j_tot=3, f_tot=3.5, species="Yb171"),  # "6sng 3G3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=4.5, f_tot=3.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=3.5, f_tot=3.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=0, j_r=3.5, f_tot=3.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.02628545, -0.13182564],
        [0.02548145, -0.12028462],
        [0.02613255, -0.14203905],
    ]
    mixing_angles = [
        (0, 1, -0.089123698),
    ]


class Yb171_G45_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "G F=4.5, nu > 25"
    f_tot = 4.5
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, j_tot=4, f_tot=4.5, species="Yb171"),  # "6sng +G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, j_tot=4, f_tot=4.5, species="Yb171"),  # "6sng -G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, j_tot=5, f_tot=4.5, species="Yb171"),  # "6sng 3G5"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=4.5, f_tot=4.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=3.5, f_tot=4.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=0, j_r=4.5, f_tot=4.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.02628545, -0.13182564],
        [0.02548145, -0.12028462],
        [0.02536571, -0.18507079],
    ]
    mixing_angles = [
        (0, 1, -0.089123698),
    ]


class Yb171_G55_HighN(FModel):
    species_name = "Yb171_mqdt"
    name = "G F=5.5, nu > 25"
    f_tot = 5.5
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=5, f_tot=5.5, species="Yb171"),  # "6sng 3G5"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, f_c=1, j_r=4.5, f_tot=5.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.02536571, -0.18507079],
    ]
    mixing_angles = []


# --------------------------------------------------------
# MQDT models valid at small n
# --------------------------------------------------------


class Yb171_S05_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "S F=0.5, 2 < nu < 26"
    f_tot = 0.5
    nu_range = (2.0, 26.0)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, f_tot=0.5, species="Yb171"),  # "6sns 1S0"
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, f_tot=0.5, species="Yb171"),  # "6pnp 1S0"
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=0.5, species="Yb171"),  # "6pnp 3P0"
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=0, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=1.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=0.5, f_c=Unknown, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=1, j_r=0.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.357488757, 0.163255076, 0],
        [0.203917828, 0, 0],
        [0.116813499, 0, 0],
        [0.287210377, 0, 0],
        [0.247550262, 0, 0],
        [0.148686263, 0, 0],
        [0.432841, 0.724559, -1.95424],
    ]
    mixing_angles = [
        (0, 1, 0.13179534),
        (0, 2, 0.29748039),
        (0, 3, 0.0553920359),
        (2, 3, 0.100843905),
        (2, 4, 0.10317753),
        (0, 5, 0.137709223),
    ]

    manual_frame_transformation_outer_inner = np.array(
        [
            [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, np.sqrt(2 / 3), 0, -np.sqrt(1 / 3), 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, np.sqrt(1 / 3), 0, np.sqrt(2 / 3), 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -1 / 2],
        ]
    )


class Yb171_S15_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "S F=1.5, 2 < nu < 26"
    f_tot = 1.5
    nu_range = (2.0, 26.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=1, j_r=0.5, f_tot=1.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.432841, 0.724559, -1.95424],
    ]
    mixing_angles = []


class Yb171_P05_Lowest(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=0.5, 1.5 < nu < 2.5"
    f_tot = 0.5
    nu_range = (1.5, 2.5)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=0.5, species="Yb171"),  # "6snp 3P0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=0.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.161083, 0],
        [0.920424, 0],
        [0.180701, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.426128, 6.272986]),
    ]


class Yb171_P05_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=0.5, 2.9 < nu < 5.9"
    f_tot = 0.5
    nu_range = (2.9, 5.9)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=0.5, species="Yb171"),  # "6snp 3P0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=0.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=0.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.967223, -3.03997, 0.569205],
        [0.967918, 0.25116, 0.868505],
        [0.969279, 0.288219, 1.36228],
    ]
    mixing_angles = []


class Yb171_P15_Lowest(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=1.5, 1.5 < nu < 2.5"
    f_tot = 1.5
    nu_range = (1.5, 2.5)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=1.5, f_tot=1.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.161083, 0],
        [0.920424, 0],
        [0.110501, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.426128, 6.272986]),
    ]


class Yb171_P15_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=1.5, 3 < nu < 10"
    f_tot = 1.5
    nu_range = (3.0, 10.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=0.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=0, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
    ]

    eigen_quantum_defects = [
        [0.967223, -3.03997, 0.569205],
        [0.967918, 0.25116, 0.868505],
        [0.228828720, 0, 0],
        [0.205484818, 0, 0],
        [0.193528629, 0, 0],
        [0.181385000, 0, 0],
        [0.906105, 0.383471, 1.23512],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.08410871, 120.37555, -9314.23]),
        (0, 2, -0.073904060),
        (0, 3, -0.063632668),
        (0, 4, -0.021924569),
        (0, 5, -0.106678810),
        (1, 2, 0.032556999),
        (1, 3, 0.054105142),
        (1, 4, 0.086127672),
        (1, 5, 0.053804487),
        (6, 7, 0.071426685),
        (6, 8, 0.027464110),
        (6, 9, -0.029741862),
    ]


class Yb171_P25_Lowest(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=2.5, 1.5 < nu < 4.5"
    f_tot = 2.5
    nu_range = (1.5, 4.5)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=2.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.906105, 0.383471, 1.23512],
    ]
    mixing_angles = []


class Yb171_P25_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "P F=2.5, 5 < nu < 20"
    f_tot = 2.5
    nu_range = (5.0, 20.0)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=2.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
    ]

    eigen_quantum_defects = [
        [0.925121305, -2.73247165, 74.664989],
        [0.230133261, 0, 0],
        [0.209638118, 0, 0],
        [0.186228192, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.0706666127),
        (0, 2, 0.0232711158),
        (0, 3, -0.0292153659),
    ]


class Yb171_D05_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=0.5, 2 < nu < 30"
    f_tot = 0.5
    nu_range = (2.0, 30.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=0.5, species="Yb171"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=0.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.758222, -0.017906, 3.392161],
    ]
    mixing_angles = []


class Yb171_D15_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=1.5, 2 < nu < 30"
    f_tot = 1.5
    nu_range = (2.0, 30.0)
    reference = "arXiv:2507.11487v1"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=1.5, species="Yb171"),  # "6snd 3D2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, f_tot=1.5, species="Yb171"),  # "6pnp 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=1.5, species="Yb171"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=Unknown, f_c=Unknown, j_r=Unknown, f_tot=1.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=0, j_r=1.5, f_tot=1.5, species="Yb171"),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [-np.sqrt(3 / 5), -np.sqrt(2 / 5), 0, 0, 0, 0],
            [np.sqrt(3 / 5) / 2, -3 / (2 * np.sqrt(10)), 0, 0, 0, np.sqrt(5 / 2) / 2],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [-1 / 2, np.sqrt(3 / 2) / 2, 0, 0, 0, np.sqrt(3 / 2) / 2],
        ]
    )

    eigen_quantum_defects = [
        [0.730541589, -0.0967938662, 0],
        [0.751542685, 0.00038836127, 0],
        [0.195864083, 0, 0],
        [0.235944408, 0, 0],
        [0.147483609, 0, 0],
        [0.758222, -0.017906, 3.392161],
    ]
    mixing_angles = [
        (0, 1, 0.220048245),
        (0, 2, 0.00427599),
        (0, 3, 0.0381563093),
        (1, 3, -0.00700797918),
        (0, 4, 0.109380331),
        (1, 4, 0.0635544456),
    ]


class Yb171_D25_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=2.5, 2 < nu < 30"
    f_tot = 2.5
    nu_range = (2.0, 30.0)
    reference = "arXiv:2507.11487v1"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=2.5, species="Yb171"),  # "6snd 3D2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, f_tot=2.5, species="Yb171"),  # "6pnp 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=2.5, species="Yb171"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=2.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=1.5, f_tot=2.5, species="Yb171"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=Unknown, f_c=Unknown, j_r=Unknown, f_tot=2.5, species="Yb171"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=0, j_r=2.5, f_tot=2.5, species="Yb171"),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [np.sqrt(7 / 5) / 2, np.sqrt(7 / 30), 0, 0, 0, -np.sqrt(5 / 3) / 2],
            [-np.sqrt(2 / 5), np.sqrt(3 / 5), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1 / 2, np.sqrt(1 / 6), 0, 0, 0, np.sqrt(7 / 3) / 2],
        ]
    )

    eigen_quantum_defects = [
        [0.730541589, -0.0967938662, 0],
        [0.751542685, 0.00038836127, 0],
        [0.195864083, 0, 0],
        [0.235944408, 0, 0],
        [0.147483609, 0, 0],
        [0.734512, -0.019501, 3.459114],
    ]
    mixing_angles = [
        (0, 1, [0.220048245, -14.9486]),
        (0, 2, 0.00427599),
        (0, 3, 0.0381563093),
        (1, 3, -0.00700797918),
        (0, 4, 0.109380331),
        (1, 4, 0.0635544456),
    ]


class Yb171_D35_LowN(FModel):
    species_name = "Yb171_mqdt"
    name = "D F=3.5, 2 < nu < 14"
    f_tot = 3.5
    nu_range = (2.0, 14.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=3.5, species="Yb171"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=1, j_r=2.5, f_tot=3.5, species="Yb171"),
    ]

    eigen_quantum_defects = [
        [0.734512, -0.019501, 3.459114],
    ]
    mixing_angles = []
