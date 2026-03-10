from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetLS
from rydstate.angular.angular_ket_dummy import AngularKetDummy
from rydstate.angular.core_ket_base import CoreKet, CoreKetDummy
from rydstate.angular.utils import Unknown
from rydstate.species.mqdt.fmodel import FModel
from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT
from rydstate.units import electron_mass, rydberg_constant


class Ytterbium173MQDT(SpeciesObjectMQDT):
    name = "Yb173_mqdt"
    Z = 70
    i_c = 2.5
    number_valence_electrons = 2

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)

    _isotope_mass = 172.938216212  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    _ionization_threshold_dict = {
        CoreKet(i_c, 0.5, 0, 0.5, 2): (50443.291203, None, "1/cm"),
        CoreKet(i_c, 0.5, 0, 0.5, 3): (50442.941262, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 0.5, Unknown): (77504.98, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 1.5, Unknown): (80835.39, None, "1/cm"),
        CoreKetDummy("4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5, 2)
    nuclear_dipole = -0.68


# --------------------------------------------------------
# MQDT models valid at large n
# --------------------------------------------------------


class Yb173_S15_HighN(FModel):
    species_name = "Yb173_mqdt"
    name = "S F=3/2, nu > 26"
    f_tot = 1.5
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=1.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=2, j_r=0.5, f_tot=1.5, species="Yb173"),
    ]

    eigen_quantum_defects = [
        [0.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]
    mixing_angles = []


class Yb173_S25_HighN(FModel):
    species_name = "Yb173_mqdt"
    name = "S F=5/2, nu > 26"
    f_tot = 2.5
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, f_tot=2.5, species="Yb173"),  # "6sns 1S0"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, f_tot=2.5, species="Yb173"),  # "6pnp 1S0"
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=2.5, species="Yb173"),  # "6pnp 3P0"
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=2.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=2, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=1.5, f_c=3, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetFJ(l_c=1, l_r=1, j_c=0.5, f_c=Unknown, j_r=0.5, f_tot=2.5, species="Yb173"),  # Fc could be 2 or 3
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=3, j_r=0.5, f_tot=2.5, species="Yb173"),
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [np.sqrt(5) / 2 / np.sqrt(3), 0, 0, 0, 0, 0, np.sqrt(7) / 2 / np.sqrt(3)],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, -np.sqrt(2 / 3), 0, np.sqrt(1 / 3), 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, np.sqrt(1 / 3), 0, np.sqrt(2 / 3), 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [np.sqrt(7) / 2 / np.sqrt(3), 0, 0, 0, 0, 0, -np.sqrt(5) / 2 / np.sqrt(3)],
        ]
    )

    eigen_quantum_defects = [
        [0.357519763, 0.298712849, 0, 0, 0],
        [0.203907536, 0, 0, 0, 0],
        [0.116803536, 0, 0, 0, 0],
        [0.286731074, 0, 0, 0, 0],
        [0.248113946, 0, 0, 0, 0],
        [0.148678953, 0, 0, 0, 0],
        [0.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
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
    species_name = "Yb173_mqdt"
    name = "S F=7/2, nu > 26"
    f_tot = 3.5
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=3.5, species="Yb173"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=3, j_r=0.5, f_tot=3.5, species="Yb173"),
    ]

    eigen_quantum_defects = [
        [0.438426851, 3.91762642, -10612.6828, 8017432.38, -2582622910.0],
    ]
    mixing_angles = []


class Yb173_P05_HighN(FModel):
    species_name = "Yb173_mqdt"
    name = "P F=1/2, nu > 10"
    f_tot = 0.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=0.5, species="Yb173"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=0.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0.5),
    ]

    eigen_quantum_defects = [
        [0.924825736, -3.542481644, 81.5334687],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.071426685),
        (0, 2, 0.027464110),
        (0, 3, -0.029741862),
    ]


class Yb173_P15_HighN(FModel):
    species_name = "Yb173_mqdt"
    name = "P F=3/2, nu > 10"
    f_tot = 1.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=1.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=1.5, species="Yb173"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=1.5, species="Yb173"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=0.5, f_tot=1.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=1.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=1.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=1.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=1.5),
    ]

    eigen_quantum_defects = [
        [0.921706585, 2.56569459, 0],
        [0.979638580, -5.239904224, 0],
        [0.228828720, 0, 0],
        [0.205484818, 0, 0],
        [0.193528629, 0, 0],
        [0.181385000, 0, 0],
        [0.924825736, -3.542481644, 81.5334687],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
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
    species_name = "Yb173_mqdt"
    name = "P F=5/2, nu > 10"
    f_tot = 2.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=2.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=2.5, species="Yb173"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=2.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=2.5, species="Yb173"),  # "6snp 3P0"
        AngularKetDummy("4f13 5d 6snl e", f_tot=2.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Yb173"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl f", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl h", f_tot=2.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=2.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=0.5, f_tot=2.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=2.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=2.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl f", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=2.5),
        AngularKetDummy("4f13 5d 6snl h", f_tot=2.5),
    ]

    eigen_quantum_defects = [
        [0.921706585, 2.56569459, 0],
        [0.979638580, -5.239904224, 0],
        [0.228828720, 0, 0],
        [0.205484818, 0, 0],
        [0.193528629, 0, 0],
        [0.181385000, 0, 0],
        [0.95356884, -0.28602498, 0],
        [0.19845903, 0, 0],
        [0.924825736, -3.542481644, 81.5334687],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
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
    species_name = "Yb173_mqdt"
    name = "P F=7/2, nu > 10"
    f_tot = 3.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=3.5, species="Yb173"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=3.5, species="Yb173"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3.5),
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=3.5, species="Yb173"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl e", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=3.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=2, j_r=1.5, f_tot=3.5, species="Yb173"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=0.5, f_tot=3.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3.5),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=3.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl e", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl f", f_tot=3.5),
        AngularKetDummy("4f13 5d 6snl g", f_tot=3.5),
    ]

    eigen_quantum_defects = [
        [0.921706585, 2.56569459, 0],
        [0.979638580, -5.239904224, 0],
        [0.228828720, 0, 0],
        [0.205484818, 0, 0],
        [0.193528629, 0, 0],
        [0.181385000, 0, 0],
        [0.924825736, -3.542481644, 81.5334687],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
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
    species_name = "Yb173_mqdt"
    name = "P F=9/2, nu > 10"
    f_tot = 4.5
    nu_range = (10.0, np.inf)
    reference = "10.1103/PhysRevA.110.042821"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=4.5, species="Yb173"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=4.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=4.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=4.5),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=3, j_r=1.5, f_tot=4.5, species="Yb173"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=4.5),
        AngularKetDummy("4f13 5d 6snl b", f_tot=4.5),
        AngularKetDummy("4f13 5d 6snl c", f_tot=4.5),
    ]

    eigen_quantum_defects = [
        [0.924825736, -3.542481644, 81.5334687],
        [0.236866903, 0, 0],
        [0.221055883, 0, 0],
        [0.185599376, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.071426685),
        (0, 2, 0.027464110),
        (0, 3, -0.029741862),
    ]
