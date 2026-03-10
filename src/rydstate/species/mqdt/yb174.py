from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.angular_ket_dummy import AngularKetDummy
from rydstate.angular.core_ket_base import CoreKet, CoreKetDummy
from rydstate.angular.utils import Unknown
from rydstate.species.mqdt.fmodel import FModel
from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT
from rydstate.units import electron_mass, rydberg_constant


class Ytterbium174MQDT(SpeciesObjectMQDT):
    name = "Yb174_mqdt"
    Z = 70
    i_c = 0
    number_valence_electrons = 2

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    _isotope_mass = 173.938859  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    _ionization_threshold_dict = {
        CoreKet(i_c, 0.5, 0, 0.5): (50443.070393, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 0.5): (77504.98, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, 1.5): (80835.39, None, "1/cm"),
        CoreKet(i_c, 0.5, 1, Unknown): (79725.35, None, "1/cm"),
        CoreKetDummy("4f13 5d 6s"): (83967.7, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5)
    nuclear_dipole = 2.1


# --------------------------------------------------------
# MQDT models valid at large n
# --------------------------------------------------------


class Yb174_S0_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "S J=0, nu > 2"
    f_tot = 0
    nu_range = (2.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6sns 1S0"
        AngularKetDummy("4f13 5d 6snl a", f_tot=0),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6pnp 1S0"
        AngularKetDummy("4f13 5d 6snl b", f_tot=0),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6pnp 3P0"
        AngularKetDummy("4f13 5d 6snl c", f_tot=0),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=0),
        AngularKetFJ(l_c=1, l_r=1, j_c=1.5, j_r=1.5, f_tot=0, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl b", f_tot=0),
        AngularKetFJ(l_c=1, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl c", f_tot=0),
    ]

    eigen_quantum_defects = [
        [0.355101645, 0.277673956],
        [0.204537535, 0],
        [0.116393648, 0],
        [0.295439966, 0],
        [0.257664798, 0],
        [0.155797119, 0],
    ]
    mixing_angles = [
        (0, 1, 0.126557575),
        (0, 2, 0.300103593),
        (0, 3, 0.056987912),
        (2, 3, 0.114312578),
        (2, 4, 0.0986363362),
        (0, 5, 0.142498543),
    ]


class Yb174_S1_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "S J=1, nu > 26"
    f_tot = 1
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevLett.128.033201"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Yb174"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.4382, 4, -1e4, 8e6, -3e9],
    ]
    mixing_angles = []


class Yb174_P0_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=0, nu > 6"
    f_tot = 0
    nu_range = (6.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6snp 3P0"
        AngularKetDummy("4f13 5d 6snd", f_tot=0),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetDummy("4f13 5d 6snd", f_tot=0),
    ]

    eigen_quantum_defects = [
        [0.953661478, -0.287531374],
        [0.198460766, 0],
    ]
    mixing_angles = [
        (0, 1, 0.163343232),
    ]


class Yb174_P1_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=1, nu > 6"
    f_tot = 1
    nu_range = (6.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
        AngularKetDummy("4f13 5d 6snl a", f_tot=1),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=1),
        AngularKetDummy("4f13 5d 6snl b", f_tot=1),
        AngularKetDummy("4f13 5d 6snl c", f_tot=1),
        AngularKetDummy("4f13 5d 6snl d", f_tot=1),
    ]

    eigen_quantum_defects = [
        [0.92271098, 2.6036257],
        [0.98208719, -5.4562725],
        [0.22851720, 0],
        [0.20607759, 0],
        [0.19352751, 0],
        [0.18153094, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.08410871, 120.37555, -9314.23]),
        (0, 2, -0.07317986),
        (0, 3, -0.06651879),
        (0, 4, -0.02212194),
        (0, 5, -0.10452109),
        (1, 2, 0.02477464),
        (1, 3, 0.05763934),
        (1, 4, 0.0860644),
        (1, 5, 0.04993818),
    ]


class Yb174_P2_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=2, nu > 5"
    f_tot = 2
    nu_range = (5.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Yb174"),  # "6snp 3P2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2),
        AngularKetDummy("4f13 5d 6snl c", f_tot=2),
    ]

    eigen_quantum_defects = [
        [0.925150932, -2.69197178, 66.7159709],
        [0.230028034, 0, 0],
        [0.209224174, 0, 0],
        [0.186236574, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.0706189664),
        (0, 2, 0.0231221428),
        (0, 3, -0.0291730345),
    ]


class Yb174_D1_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=1, nu > 26"
    f_tot = 1
    nu_range = (26.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Yb174"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.75258093, 0.3826, -483.1],
    ]
    mixing_angles = []


class Yb174_D2_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=2, nu > 5"
    f_tot = 2
    nu_range = (5.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Yb174"),  # "6snd 3D2"
        AngularKetDummy("4f13 5d 6snl a", f_tot=2),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6pnp 1D2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=2),
        AngularKetDummy("4f13 5d 6snl b", f_tot=2),
        AngularKetFJ(l_c=1, l_r=1, j_c=Unknown, j_r=Unknown, f_tot=2, species="Yb174"),  # Jc could be either 1/2 or 3/2
    ]
    manual_frame_transformation_outer_inner = np.array(
        [
            [np.sqrt(3 / 5), np.sqrt(2 / 5), 0, 0, 0],
            [-np.sqrt(2 / 5), np.sqrt(3 / 5), 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )

    eigen_quantum_defects = [
        [0.729513646, -0.0377841183],
        [0.752292223, 0.104072325],
        [0.19612036, 0],
        [0.233752026, 0],
        [0.152911249, 0],
    ]
    mixing_angles = [
        (0, 1, [0.21157531, -15.3844]),
        (0, 2, 0.00521559431),
        (0, 3, 0.0398131577),
        (1, 3, -0.0071658109),
        (0, 4, 0.10481227),
        (1, 4, 0.0721660042),
    ]


class Yb174_D3_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=3, nu > 18"
    f_tot = 3
    nu_range = (18.0, np.inf)
    reference = "10.1103/PhysRevX.15.011009"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Yb174"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.72902016, -0.705328923, 829.238844],
    ]
    mixing_angles = []


class Yb174_F2_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "F J=2, nu > 25"
    f_tot = 2
    nu_range = (25.0, np.inf)
    reference = "arXiv:2507.11487v1"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, species="Yb174"),  # "6snf 3F2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0718252326, -1.00091963, -106.291066],
    ]
    mixing_angles = []


class Yb174_F3_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "F J=3, nu > 7"
    f_tot = 3
    nu_range = (7.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, species="Yb174"),  # "6snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, species="Yb174"),  # "6snf 3F3"
        AngularKetDummy("4f13 5d 6snl a", f_tot=3),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3),
        AngularKetDummy("4f13 5d 6snl e", f_tot=3),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=3, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
        AngularKetDummy("4f13 5d 6snl a", f_tot=3),
        AngularKetDummy("4f13 5d 6snl b", f_tot=3),
        AngularKetDummy("4f13 5d 6snl c", f_tot=3),
        AngularKetDummy("4f13 5d 6snl d", f_tot=3),
        AngularKetDummy("4f13 5d 6snl e", f_tot=3),
    ]

    eigen_quantum_defects = [
        [0.276158949, -12.7258012],
        [0.0715123712, -0.768462937],
        [0.239015576, 0],
        [0.226770354, 0],
        [0.175354845, 0],
        [0.196660618, 0],
        [0.21069642, 0],
    ]
    mixing_angles = [
        (0, 1, [0.0209955122, 0.251041249]),
        (0, 2, -0.00411835457),
        (0, 3, -0.0962784945),
        (0, 4, 0.132826901),
        (0, 5, -0.0439244317),
        (0, 6, 0.0508460294),
        (1, 2, -0.0376574252),
        (1, 3, 0.026944623),
        (1, 4, -0.0148474857),
        (1, 5, -0.0521244126),
        (1, 6, 0.0349516329),
    ]


class Yb174_F4_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "F J=4, nu > 25"
    f_tot = 4
    nu_range = (25.0, np.inf)
    reference = "arXiv:2507.11487v1"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, species="Yb174"),  # "6snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=4, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0839027969, -2.91009023],
    ]
    mixing_angles = []


class Yb174_G3_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "G J=3, nu > 25"
    f_tot = 3
    nu_range = (25.0, np.inf)
    reference = "arXiv:2507.11487v1"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=3, species="Yb174"),  # "6sng 3G3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0260964574, -0.14139526],
    ]
    mixing_angles = []


class Yb174_G4_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "G J=4, nu > 25"
    f_tot = 4
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, j_tot=4, species="Yb174"),  # "6sng +G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, j_tot=4, species="Yb174"),  # "6sng -G4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, f_tot=4, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, f_tot=4, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0262659964, -0.148808463],
        [0.0254568575, -0.134219071],
    ]
    mixing_angles = [
        (0, 1, -0.089123698),
    ]


class Yb174_G5_HighN(FModel):
    species_name = "Yb174_mqdt"
    name = "G J=5, nu > 25"
    f_tot = 5
    nu_range = (25.0, np.inf)
    reference = "10.1103/PhysRevA.112.042817"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=5, species="Yb174"),  # "6sng 3G5"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, f_tot=5, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.02536571, -0.18507079],
    ]
    mixing_angles = []


# --------------------------------------------------------
# MQDT models valid at small n
# --------------------------------------------------------


class Yb174_S0_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "S J=0, 1 < nu < 2"
    f_tot = 0
    nu_range = (1.0, 2.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6sns 1S0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.525055, 0],
    ]
    mixing_angles = []


class Yb174_S1_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "S J=1, 2 < nu < 26"
    f_tot = 1
    nu_range = (2.0, 26.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Yb174"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.432841, 0.724559, -1.95424],
    ]
    mixing_angles = []


class Yb174_P0_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=0, 1.5 < nu < 5.5"
    f_tot = 0
    nu_range = (1.5, 5.5)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6snp 3P0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.969279, 0.288219, 1.36228],
    ]
    mixing_angles = []


class Yb174_P1_Lowest(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=1, 1.7 < nu < 2.7"
    f_tot = 1
    nu_range = (1.7, 2.7)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.161083, 0],
        [0.920424, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.426128, 6.272986]),
    ]


class Yb174_P1_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=1, 2.7 < nu < 5.7"
    f_tot = 1
    nu_range = (2.7, 5.7)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.967223, -3.03997, 0.569205],
        [0.967918, 0.25116, 0.868505],
    ]
    mixing_angles = []


class Yb174_P2_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "P J=2, 1.5 < nu < 4.5"
    f_tot = 2
    nu_range = (1.5, 4.5)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Yb174"),  # "6snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.906105, 0.383471, 1.23512],
    ]
    mixing_angles = []


class Yb174_D1_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=1, 2 < nu < 26"
    f_tot = 1
    nu_range = (2.0, 26.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Yb174"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.758222, -0.017906, 3.392161],
    ]
    mixing_angles = []


class Yb174_D2_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=2, 2 < nu < 5"
    f_tot = 2
    nu_range = (2.0, 5.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Yb174"),  # "6snd 3D2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.703156, 0.973192],
        [0.724546, 0.372621],
    ]
    mixing_angles = [
        (0, 1, [0.948409, 2.121270]),
    ]


class Yb174_D3_LowN(FModel):
    species_name = "Yb174_mqdt"
    name = "D J=3, 2 < nu < 18"
    f_tot = 3
    nu_range = (2.0, 18.0)
    reference = None

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Yb174"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.734512, -0.019501, 3.459114],
    ]
    mixing_angles = []
