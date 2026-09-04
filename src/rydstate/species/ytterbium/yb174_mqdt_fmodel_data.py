# ruff: noqa: RUF012, N801

from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import Unknown
from rydstate.species.fmodel import FModel

REFERENCE_PEPER_2025 = "M. Peper et al., Phys. Rev. X 15, 011009 (2025), https://doi.org/10.1103/PhysRevX.15.011009"
REFERENCE_KURODA_2025 = "R. Kuroda et al., Phys. Rev. A 112, 042817 (2025), https://doi.org/10.1103/mzsv-rckx"
REFERENCE_WILSON_2022 = (
    "J. T. Wilson et al., Phys. Rev. Lett. 128, 033201 (2022), https://doi.org/10.1103/PhysRevLett.128.033201"
)


class Yb174_S0_HighN(FModel):
    species = "Yb174"
    name = "S J=0, nu > 2"
    f_tot = 0
    nu_range = (2.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6sns 1S0"
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetLS(l_c=1, l_r=1, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6pnp 1S0"
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6pnp 3P0"
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(l_c=1, l_r=1, j_c=1.5, j_r=1.5, f_tot=0, species="Yb174"),
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(l_c=1, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
    ]

    eigen_quantum_defects = [
        [4.355101645, 0.277673956],
        [4.204537535, 0],
        [4.116393648, 0],
        [4.295439966, 0],
        [4.257664798, 0],
        [4.155797119, 0],
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
    species = "Yb174"
    name = "S J=1, nu > 26"
    f_tot = 1
    nu_range = (26.0, np.inf)
    reference = REFERENCE_WILSON_2022

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Yb174"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [4.4382, 4, -1e4, 8e6, -3e9],
    ]


class Yb174_P0_HighN(FModel):
    species = "Yb174"
    name = "P J=0, nu > 6"
    f_tot = 0
    nu_range = (6.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6snp 3P0"
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snd", species="Yb174"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
        AngularKetFJ(f_tot=0, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snd", species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.953661478, -0.287531374],
        [4.198460766, 0],
    ]
    mixing_angles = [
        (0, 1, 0.163343232),
    ]


class Yb174_P1_HighN(FModel):
    species = "Yb174"
    name = "P J=1, nu > 6"
    f_tot = 1
    nu_range = (6.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb174"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
        AngularKetFJ(f_tot=1, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.922709076, 2.60055203],
        [3.982084772, -5.45063476],
        [4.228518316, 0],
        [4.206081775, 0],
        [4.193527605, 0],
        [4.181533031, 0],
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
    species = "Yb174"
    name = "P J=2, nu > 5"
    f_tot = 2
    nu_range = (5.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Yb174"),  # "6snp 3P2"
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.925150932, -2.69197178, 66.7159709],
        [4.230028034, 0, 0],
        [4.209224174, 0, 0],
        [4.186236574, 0, 0],
    ]
    mixing_angles = [
        (0, 1, 0.0706189664),
        (0, 2, 0.0231221428),
        (0, 3, -0.0291730345),
    ]


class Yb174_D1_HighN(FModel):
    species = "Yb174"
    name = "D J=1, nu > 26"
    f_tot = 1
    nu_range = (26.0, np.inf)
    reference = REFERENCE_PEPER_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Yb174"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [  # TABLE I main text
        [2.75258093, 0.3826, -483.1],
    ]


class Yb174_D2_HighN(FModel):
    species = "Yb174"
    name = "D J=2, nu > 5"
    f_tot = 2
    nu_range = (5.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Yb174"),  # "6snd 3D2"
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6pnp 1D2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=2, l_c=Unknown, parity=1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6pnp 1D2"
    ]

    eigen_quantum_defects = [
        [2.729513646, -0.0377841183],
        [2.752292223, 0.104072325],
        [4.19612036, 0],
        [4.233752026, 0],
        [4.152911249, 0],
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
    species = "Yb174"
    name = "D J=3, nu > 18"
    f_tot = 3
    nu_range = (18.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Yb174"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [2.72902016, -0.705328923, 829.238844],
    ]


class Yb174_F2_HighN(FModel):
    species = "Yb174"
    name = "F J=2, nu > 25"
    f_tot = 2
    nu_range = (25.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, species="Yb174"),  # "6snf 3F2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0718252326, -1.00091963, -106.291066],
    ]


class Yb174_F3_HighN(FModel):
    species = "Yb174"
    name = "F J=3, nu > 7"
    f_tot = 3
    nu_range = (7.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, species="Yb174"),  # "6snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, species="Yb174"),  # "6snf 3F3"
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb174"),
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=3, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl a", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl b", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl c", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl d", species="Yb174"),
        AngularKetFJ(f_tot=3, l_c=Unknown, parity=-1, allow_unknown=True, label="4f13 5d 6snl e", species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.276158949, -12.7258012],
        [0.0715123712, -0.768462937],
        [4.239015576, 0],
        [4.226770354, 0],
        [4.175354845, 0],
        [4.196660618, 0],
        [4.21069642, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.0209955122, 0.251041249]),
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
    species = "Yb174"
    name = "F J=4, nu > 25"
    f_tot = 4
    nu_range = (25.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, species="Yb174"),  # "6snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, j_r=3.5, f_tot=4, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0839027969, -2.91009023],
    ]


class Yb174_G3_HighN(FModel):
    species = "Yb174"
    name = "G J=3, nu > 25"
    f_tot = 3
    nu_range = (25.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=3, species="Yb174"),  # "6sng 3G3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.0260964574, -0.14139526],
    ]


class Yb174_G4_HighN(FModel):
    species = "Yb174"
    name = "G J=4, nu > 25"
    f_tot = 4
    nu_range = (25.0, np.inf)
    reference = REFERENCE_KURODA_2025

    inner_channels = [
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, j_tot=4, species="Yb174"),  # "6sng +G4"
        AngularKetJJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, j_tot=4, species="Yb174"),  # "6sng -G4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, f_tot=4, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=3.5, f_tot=4, species="Yb174"),
    ]

    eigen_quantum_defects = [
        # TODO is this correct? In the paper the "matrix" is transposed
        # but Fig 7 suggests that this interpretation is correct
        [0.0262659964, -0.148808463],
        [0.0254568575, -0.134219071],
    ]
    mixing_angles = [
        (0, 1, -0.089123698),
    ]


class Yb174_G5_HighN(FModel):
    species = "Yb174"
    name = "G J=5, nu > 25"
    f_tot = 5
    nu_range = (25.0, np.inf)
    reference = (REFERENCE_KURODA_2025, "obtained from a fit to the 171Yb 6sng (F = 9/2) spectroscopic data")

    inner_channels = [
        AngularKetLS(l_c=0, l_r=4, l_tot=4, s_tot=1, j_tot=5, species="Yb174"),  # "6sng 3G5"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=4, j_c=0.5, j_r=4.5, f_tot=5, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [0.02536571, -0.18507079],
    ]


# --------------------------------------------------------
# MQDT models valid at small n
# --------------------------------------------------------


class Yb174_S0_LowN(FModel):
    species = "Yb174"
    name = "S J=0, 1 < nu < 2"
    f_tot = 0
    nu_range = (1.0, 2.0)
    reference = "fit to the 6s^2 ground state"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Yb174"),  # "6sns 1S0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [4.525055, 0],
    ]


class Yb174_S1_LowN(FModel):
    species = "Yb174"
    name = "S J=1, 2 < nu < 26"
    f_tot = 1
    nu_range = (2.0, 26.0)
    reference = "fit to NIST data between 7s and 13s, extrapolation seems good up to 30s"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Yb174"),  # "6sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [4.432841, 0.724559, -1.95424],
    ]


class Yb174_P0_LowN(FModel):
    species = "Yb174"
    name = "P J=0, 1.5 < nu < 5.5"
    f_tot = 0
    nu_range = (1.5, 5.5)
    reference = "fit to NIST data between 6p and 9p"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Yb174"),  # "6snp 3P0"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=0, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.969279, 0.288219, 1.36228],
    ]


class Yb174_P1_Lowest(FModel):
    species = "Yb174"
    name = "P J=1, 1.7 < nu < 2.7"
    f_tot = 1
    nu_range = (1.7, 2.7)
    reference = "fit to NIST data for the 6p states"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [4.161083, 0],
        [3.920424, 0],
    ]
    mixing_angles = [
        (0, 1, [-0.426128, 6.272986]),
    ]


class Yb174_P1_LowN(FModel):
    species = "Yb174"
    name = "P J=1, 2.7 < nu < 5.7"
    f_tot = 1
    nu_range = (2.7, 5.7)
    reference = "fit to NIST data between 7p and 9p"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Yb174"),  # "6snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Yb174"),  # "6snp 3P1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=0.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.967223, -3.03997, 0.569205],
        [3.967918, 0.25116, 0.868505],
    ]


class Yb174_P2_LowN(FModel):
    species = "Yb174"
    name = "P J=2, 1.5 < nu < 4.5"
    f_tot = 2
    nu_range = (1.5, 4.5)
    reference = "fit to NIST data between 6p and 8p"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Yb174"),  # "6snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [3.906105, 0.383471, 1.23512],
    ]


class Yb174_D1_LowN(FModel):
    species = "Yb174"
    name = "D J=1, 2 < nu < 26"
    f_tot = 1
    nu_range = (2.0, 26.0)
    reference = "fit to NIST data between 5d and 8d, causes a µ=0.005 difference at the 30d state"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Yb174"),  # "6snd 3D1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=1, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [2.758222, -0.017906, 3.392161],
    ]


class Yb174_D2_LowN(FModel):
    species = "Yb174"
    name = "D J=2, 2 < nu < 5"
    f_tot = 2
    nu_range = (2.0, 5.0)
    reference = "fit to NIST data between 5d and 7d"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Yb174"),  # "6snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Yb174"),  # "6snd 3D2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=2, species="Yb174"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, f_tot=2, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [2.703156, 0.973192],
        [2.724546, 0.372621],
    ]
    mixing_angles = [
        (0, 1, [0.948409, 2.121270]),
    ]


class Yb174_D3_LowN(FModel):
    species = "Yb174"
    name = "D J=3, 2 < nu < 18"
    f_tot = 3
    nu_range = (2.0, 18.0)
    reference = "fit to NIST data between 5d and 8d, provides good match around 21d"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Yb174"),  # "6snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, f_tot=3, species="Yb174"),
    ]

    eigen_quantum_defects = [
        [2.734512, -0.019501, 3.459114],
    ]
