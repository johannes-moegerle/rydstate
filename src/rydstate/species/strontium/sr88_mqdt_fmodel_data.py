# ruff: noqa: RUF012, N801

from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetLS
from rydstate.species.fmodel import FModel

REFERENCE_ROBICHEAUX_2019 = (
    "F. Robicheaux, J. Phys. B: At. Mol. Opt. Phys. 52 244001 (2019), https://doi.org/10.1088/1361-6455/ab4c22"
)


class Sr88_S0_HighN(FModel):
    species = "Sr88"
    name = "S J=0, nu > 10"
    f_tot = 0
    nu_range = (10.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Sr88"),  # "5sns 1S0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=0, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [3.26896, -0.138, 0.9],
    ]


class Sr88_S1_HighN(FModel):
    species = "Sr88"
    name = "S J=1, nu > 11"
    f_tot = 1
    nu_range = (11.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Sr88"),  # "5sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=1, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [3.370778, 0.418, -0.3],
    ]


# --------------------------------------------------------
# MQDT models valid at small n
# --------------------------------------------------------


class Sr88_P1_LowN(FModel):
    species = "Sr88"
    name = "P J=1 (recombination), 1.8 < nu < 2.2"
    f_tot = 1
    nu_range = (1.8, 2.2)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Sr88"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Sr88"),  # "5snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Sr88"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.87199081, 0],
        [3.13140955, 0],
    ]
    mixing_angles = [
        (0, 1, [1.31169947, -4.48280597]),
    ]


# --------------------------------------------------------
# MQDT models valid at large n (continued)
# --------------------------------------------------------


class Sr88_P0_HighN(FModel):
    species = "Sr88"
    name = "P J=0, nu > 7"
    f_tot = 0
    nu_range = (7.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Sr88"),  # "5snp 3P0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.8867, 0.44, -1.9],
    ]


class Sr88_P1_HighN(FModel):
    species = "Sr88"
    name = "P J=1, nu > 5"
    f_tot = 1
    nu_range = (5.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Sr88"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Sr88"),  # "5snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Sr88"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.724, -4.67, -157],
        [2.8826, 0.407, -1.3],
    ]


class Sr88_P2_HighN(FModel):
    species = "Sr88"
    name = "P J=2, nu > 5"
    f_tot = 2
    nu_range = (5.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Sr88"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=2, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.882, 0.446, -1.9],
    ]


class Sr88_D1_HighN(FModel):
    species = "Sr88"
    name = "D J=1, nu > 17"
    f_tot = 1
    nu_range = (17.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Sr88"),  # "5snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=1, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.67524, -13.15, -4444],
    ]


class Sr88_D2_HighN(FModel):
    species = "Sr88"
    name = "D J=2, nu > 25"
    f_tot = 2
    nu_range = (25.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Sr88"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Sr88"),  # "5snd 3D2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=2, species="Sr88"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=2, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [3.3847, -39.41, -1090],
        [2.66149, -16.77, -6656],
    ]
    mixing_angles = [
        (0, 1, -0.14),
    ]


class Sr88_D3_HighN(FModel):
    species = "Sr88"
    name = "D J=3, nu > 25"
    f_tot = 3
    nu_range = (25.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Sr88"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=3, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [2.655, -41.4, -15363],
    ]


class Sr88_F2_HighN(FModel):
    species = "Sr88"
    name = "F J=2, nu > 9"
    f_tot = 2
    nu_range = (9.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, species="Sr88"),  # "5snf 3F2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=2, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [0.12, -2.2, 120],
    ]


class Sr88_F3_HighN(FModel):
    species = "Sr88"
    name = "F J=3, nu > 9"
    f_tot = 3
    nu_range = (9.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, species="Sr88"),  # "5snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, species="Sr88"),  # "5snf 3F3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=3, species="Sr88"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=3, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [0.089, -2, 30],
        [0.12, -2.2, 120],
    ]


class Sr88_F4_HighN(FModel):
    species = "Sr88"
    name = "F J=4, nu > 9"
    f_tot = 4
    nu_range = (9.0, np.inf)
    reference = REFERENCE_ROBICHEAUX_2019

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, species="Sr88"),  # "5snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=4, species="Sr88"),
    ]

    eigen_quantum_defects = [
        [0.12, -2.2, 120],
    ]
