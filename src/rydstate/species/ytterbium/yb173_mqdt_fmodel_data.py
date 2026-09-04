# ruff: noqa: RUF012, N801

from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import Unknown
from rydstate.species.fmodel import FModel


class Yb173_S15_HighN(FModel):
    species = "Yb173"
    name = "S F=3/2, nu > 26"
    f_tot = 1.5
    nu_range = (26.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=1.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=2, j_r=0.5, f_tot=1.5, species="Yb173"),
    ]

    eigen_quantum_defects = [
        [4.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]


class Yb173_S25_HighN(FModel):
    species = "Yb173"
    name = "S F=5/2, nu > 26"
    f_tot = 2.5
    nu_range = (26.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, f_tot=2.5, species="Yb173"),  # "6sns 1S0"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, f_tot=2.5, species="Yb173"),  # "6pnp 1S0"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=2.5, species="Yb173"),  # "6pnp 3P0"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=2.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=2, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetJJ(l_c=1, l_r=1, j_c=1.5, j_r=1.5, j_tot=0, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetJJ(l_c=1, l_r=1, j_c=0.5, j_r=0.5, j_tot=0, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=3, j_r=0.5, f_tot=2.5, species="Yb173"),
    ]

    eigen_quantum_defects = [
        [4.357519763, 0.298712849, 0, 0, 0],
        [4.203907536, 0, 0, 0, 0],
        [4.116803536, 0, 0, 0, 0],
        [4.286731074, 0, 0, 0, 0],
        [4.248113946, 0, 0, 0, 0],
        [4.148678953, 0, 0, 0, 0],
        [4.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]
    mixing_angles = [
        (0, 1, 0.131810463),
        (0, 2, 0.297612147),
        (0, 3, 0.055508821),
        (2, 3, 0.101030515),
        (2, 4, 0.102911159),
        (0, 5, 0.137723736),
    ]


class Yb173_S35_HighN(FModel):
    species = "Yb173"
    name = "S F=7/2, nu > 26"
    f_tot = 3.5
    nu_range = (26.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=3.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=3, j_r=0.5, f_tot=3.5, species="Yb173"),
    ]

    eigen_quantum_defects = [
        [4.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]


class Yb173_P05_HighN(FModel):
    species = "Yb173"
    name = "P F=1/2, nu > 10"
    f_tot = 0.5
    nu_range = (10.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=0.5, species="Yb173"),  # "6snp 3P2"
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=0.5, species="Yb173"),
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=0.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
    ]

    eigen_quantum_defects = [
        [3.924825736, -3.542481644, 81.5334687],
        [4.236866903, 0, 0],
        [4.221055883, 0, 0],
        [4.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.071426685),
        (0, 2, 0.027464110),
        (0, 3, -0.029741862),
    ]


class Yb173_P15_HighN(FModel):
    species = "Yb173"
    name = "P F=3/2, nu > 10"
    f_tot = 1.5
    nu_range = (10.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=1.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=1.5, species="Yb173"),  # "6snp 3P1"
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=1.5, species="Yb173"),  # "6snp 3P2"
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=0.5, f_tot=1.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=1.5, species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=1.5, species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=1.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
    ]

    eigen_quantum_defects = [
        [3.921706585, 2.56569459, 0],
        [3.97963858, -5.239904224, 0],
        [4.228828720, 0, 0],
        [4.205484818, 0, 0],
        [4.193528629, 0, 0],
        [4.181385000, 0, 0],
        [3.924825736, -3.542481644, 81.5334687],
        [4.236866903, 0, 0],
        [4.221055883, 0, 0],
        [4.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.087127227, 135.400009, -12985.0162]),
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


class Yb173_P25_HighN(FModel):
    species = "Yb173"
    name = "P F=5/2, nu > 10"
    f_tot = 2.5
    nu_range = (10.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=2.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=2.5, species="Yb173"),  # "6snp 3P1"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=2.5, species="Yb173"),  # "6snp 3P0"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Yb173"),  # "6snp 3P2"
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl h", species="Yb173"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
        AngularKetFJ(f_tot=2.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl h", species="Yb173"),
    ]

    eigen_quantum_defects = [
        [3.921706585, 2.56569459, 0],
        [3.97963858, -5.239904224, 0],
        [4.228828720, 0, 0],
        [4.205484818, 0, 0],
        [4.193528629, 0, 0],
        [4.181385000, 0, 0],
        [3.95356884, -0.28602498, 0],
        [4.19845903, 0, 0],
        [3.924825736, -3.542481644, 81.5334687],
        [4.236866903, 0, 0],
        [4.221055883, 0, 0],
        [4.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.087127227, 135.400009, -12985.0162]),
        (0, 2, -0.073904060),
        (0, 3, -0.063632668),
        (0, 4, -0.021924569),
        (0, 5, -0.106678810),
        (1, 2, 0.032556999),
        (1, 3, 0.054105142),
        (1, 4, 0.086127672),
        (1, 5, 0.053804487),
        (6, 7, 0.16328854),
        (9, 8, -0.071426685),
        (10, 8, -0.027464110),
        (11, 8, 0.029741862),
    ]


class Yb173_P35_HighN(FModel):
    species = "Yb173"
    name = "P F=7/2, nu > 10"
    f_tot = 3.5
    nu_range = (10.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=3.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=3.5, species="Yb173"),  # "6snp 3P1"
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=3.5, species="Yb173"),  # "6snp 3P2"
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=3.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=0.5, f_tot=3.5, species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=3.5, species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl f", species="Yb173"),
        AngularKetFJ(f_tot=3.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl g", species="Yb173"),
    ]

    eigen_quantum_defects = [
        [3.921706585, 2.56569459, 0],
        [3.97963858, -5.239904224, 0],
        [4.228828720, 0, 0],
        [4.205484818, 0, 0],
        [4.193528629, 0, 0],
        [4.181385000, 0, 0],
        [3.924825736, -3.542481644, 81.5334687],
        [4.236866903, 0, 0],
        [4.221055883, 0, 0],
        [4.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.087127227, 135.400009, -12985.0162]),
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


class Yb173_P45_HighN(FModel):
    species = "Yb173"
    name = "P F=9/2, nu > 10"
    f_tot = 4.5
    nu_range = (10.0, np.inf)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=4.5, species="Yb173"),  # "6snp 3P2"
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=4.5, species="Yb173"),
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb173"),
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb173"),
        AngularKetFJ(f_tot=4.5, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb173"),
    ]

    eigen_quantum_defects = [
        [3.924825736, -3.542481644, 81.5334687],
        [4.236866903, 0, 0],
        [4.221055883, 0, 0],
        [4.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.071426685),
        (0, 2, 0.027464110),
        (0, 3, -0.029741862),
    ]
