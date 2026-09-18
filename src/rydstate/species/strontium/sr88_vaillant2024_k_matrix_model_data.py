# ruff: noqa: RUF012, N801
"""MQDT models for Sr88 in the reactance matrix (K-matrix) formulation of Vaillant, Jones and Potvliege.

The models were originally published in C. L. Vaillant, M. P. A. Jones and R. M. Potvliege,
J. Phys. B: At. Mol. Opt. Phys. 47, 155001 (2014). The parameters of that publication are incorrect for most
series (sign error in the fitting program), all K-matrix parameters below are taken from the Addendum
J. Phys. B: At. Mol. Opt. Phys. 57, 199401 (2024), https://doi.org/10.1088/1361-6455/ad76f0 (arXiv:2403.08742).
Table III of the Addendum quotes the parameters with 7 significant digits, we use the values with 9 significant
digits from the mqdtfit driver scripts of the authors (https://github.com/durham-qlm/mqdtfit, folder strontium),
which reproduce the theoretical energies of the supplementary material of the Addendum to better than 1e-3 1/cm
(the rounded values of Table III deviate by up to 0.01 1/cm for the lowest 3S1 states, due to cancellations
between the large K-matrix elements of this model).

The K-matrix elements are given in the frame of the dissociation channels of the paper (outer channels here):
jj-coupled channels where the fine structure of the ionic core is resolved (5s_1/2 nl_j, 4d_5/2 nl_j, 4d_3/2 nl_j)
and LS-coupled channels where an averaged ionization threshold is used (4dnl, 5pnp), see Table I of the Addendum.
The diagonal elements depend linearly on the energy, K_ii = K_ii^(0) + K_ii^(1) (I_s - E) / I_s,
with the first ionization threshold I_s = 45932.2002 1/cm (see :class:`~rydstate.species.k_matrix_model.KMatrixModel`).

Sign convention of the off-diagonal elements
--------------------------------------------
The relative phases of the jj-coupled channel kets in rydstate differ from the ones used in the paper
for the pairs (5s_1/2 nd_5/2, 5s_1/2 nd_3/2) and (4d_5/2 ns_1/2, 4d_3/2 ns_1/2):
the jj->LS recoupling matrices of the paper (U_{i alphabar} in the mqdtfit drivers) are reproduced by the
overlaps of the rydstate kets up to a sign flip of one ket of each pair.
The corresponding off-diagonal K-matrix elements of the D J=2 model are therefore multiplied by -1
compared to Table III of the Addendum (marked in the comments below).
This does not change any energy, but is necessary to get the correct singlet/triplet character of the states.
For the pair (4d_5/2 nd_5/2, 4d_3/2 nd_3/2) of the S J=0 model the phases agree and no sign is changed.
The relative phases between channels of different configurations (e.g. 5snd and 4dns) are not fixed by
angular momentum algebra and thus taken as in the paper.

Validity ranges
---------------
The lower bound of nu_range of each model is chosen such that all states used in the fit are included,
but no lower lying states, which the models do not reproduce (e.g. 5s6s 1S0 or 5s5d 1D2).

Triplet F series
----------------
The Addendum does not cover the 5snf 3FJ series. They are described here by single channel models using the
Rydberg-Ritz quantum defects of Connerade 1992, see the comment above these models at the end of this module.
"""

from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetJJ, AngularKetLS
from rydstate.species.eigen_channel_model import EigenChannelModel
from rydstate.species.k_matrix_model import KMatrixModel

REFERENCE_VAILLANT_2024 = (
    "C. L. Vaillant, M. P. A. Jones and R. M. Potvliege, "
    "J. Phys. B: At. Mol. Opt. Phys. 57, 199401 (2024), https://doi.org/10.1088/1361-6455/ad76f0 "
    "(Addendum to J. Phys. B: At. Mol. Opt. Phys. 47, 155001 (2014), https://doi.org/10.1088/0953-4075/47/15/155001)"
)
REFERENCE_CONNERADE_1992 = (
    "J. P. Connerade, W. A. Farooq, H. Ma, M. Nawaz and N. Shen, "
    "J. Phys. B: At. Mol. Opt. Phys. 25, 1405 (1992), https://doi.org/10.1088/0953-4075/25/7/012"
)

# LS-coupled channels with a 4d core, whose j_c is fixed by the angular momentum coupling (e.g. 4dns 3D1 is purely
# 4d_3/2 ns_1/2, 4dns 3D3 purely 4d_5/2 ns_1/2 and 4dnp 3P0 purely 4d_3/2 np_3/2), are labeled to use the averaged
# 4d ionization threshold of the paper instead of the fine structure resolved one
# (see the ionization_threshold_dict of the MQDT class).
LABEL_4D_AVERAGED = "4d threshold averaged"


# --------------------------------------------------------
# S states
# --------------------------------------------------------


class Sr88_S0_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "S J=0, nu > 3.7"
    f_tot, parity = (0, +1)
    nu_range = (3.7, np.inf)  # fitted to 5sns 1S0, n = 7-30, and the 4d2 3P0 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetJJ(l_c=0, l_r=0, j_c=0.5, j_r=0.5, j_tot=0, species="Sr88"),  # 5s_1/2 ns_1/2
        AngularKetJJ(l_c=2, l_r=2, j_c=2.5, j_r=2.5, j_tot=0, species="Sr88"),  # 4d_5/2 nd_5/2
        AngularKetJJ(l_c=2, l_r=2, j_c=1.5, j_r=1.5, j_tot=0, species="Sr88"),  # 4d_3/2 nd_3/2
    ]

    k_matrix = [
        (0, 0, [1.05126086e0, 8.76391110e-1]),
        (0, 1, [3.75986417e-1]),
        (0, 2, [-2.36548468e-2]),
        (1, 1, [-6.40092484e-1, 4.04258438e-1]),
        (1, 2, [-2.06382548e-4]),
        (2, 2, [3.00908733e0, -1.72263065e1]),
    ]


class Sr88_S1_Vaillant2024(KMatrixModel):
    """5sns 3S1 model fitted to the data set (b) of the Addendum (high precision data of Couturier 2019 for n >= 13).

    The Addendum also gives a fit to the older data set (a) of the 2014 paper, which is not included here.
    """

    species = "Sr88"
    name = "S J=1, nu > 3.4"
    f_tot, parity = (1, +1)
    nu_range = (3.4, np.inf)  # fitted to 5sns 3S1, n = 7-23
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, species="Sr88"),  # 5sns 3S1
        AngularKetLS(l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Sr88"),  # 5pnp 3P1
    ]

    k_matrix = [
        (0, 0, [-1.03924403e2, -2.76691239e1]),
        (0, 1, [-1.33451654e2]),
        (1, 1, [-1.68045201e2, 5.51718438e1]),
    ]


# --------------------------------------------------------
# P states
# --------------------------------------------------------


class Sr88_P1_Singlet_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "P J=1 singlet, nu > 2.8"
    f_tot, parity = (1, -1)
    nu_range = (2.8, np.inf)  # fitted to 5snp 1P1, n = 6-29, and the 4d5p 1P1 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Sr88"),  # 5snp 1P1
        AngularKetLS(l_c=2, l_r=1, l_tot=1, s_tot=0, j_tot=1, species="Sr88"),  # 4dnp 1P1
    ]

    k_matrix = [
        (0, 0, [1.11680885e1, -9.09786173e-1]),
        (0, 1, [1.61693288e1]),
        (1, 1, [2.23961659e1, 4.27262604e0]),
    ]


class Sr88_P0_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "P J=0, nu > 2.8"
    f_tot, parity = (0, -1)
    nu_range = (2.8, np.inf)  # fitted to 5snp 3P0, n = 6-15, and the 4d5p 3P0 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Sr88"),  # 5snp 3P0
        AngularKetLS(l_c=2, l_r=1, l_tot=1, s_tot=1, j_tot=0, species="Sr88", label=LABEL_4D_AVERAGED),  # 4dnp 3P0
    ]

    k_matrix = [
        (0, 0, [-4.00956527e-1, 1.03992333e0]),
        (0, 1, [-2.22056880e-1]),
        (1, 1, [-4.02518026e-1, -1.02169594e0]),
    ]


class Sr88_P1_Triplet_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "P J=1 triplet, nu > 2.8"
    f_tot, parity = (1, -1)
    nu_range = (2.8, np.inf)  # fitted to 5snp 3P1, n = 6-15, and the 4d5p 3P1 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Sr88"),  # 5snp 3P1
        AngularKetLS(l_c=2, l_r=1, l_tot=1, s_tot=1, j_tot=1, species="Sr88"),  # 4dnp 3P1
    ]

    k_matrix = [
        (0, 0, [-4.19906665e-1, 1.08261507e0]),
        (0, 1, [-2.29230418e-1]),
        (1, 1, [-3.52617910e-1, -1.30477942e0]),
    ]


class Sr88_P2_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "P J=2, nu > 2.8"
    f_tot, parity = (2, -1)
    nu_range = (2.8, np.inf)  # fitted to 5snp 3P2, n = 6-15, and the 4d5p 3P2 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Sr88"),  # 5snp 3P2
        AngularKetLS(l_c=2, l_r=1, l_tot=1, s_tot=1, j_tot=2, species="Sr88"),  # 4dnp 3P2
    ]

    k_matrix = [
        (0, 0, [-4.53113343e-1, 1.05086557e0]),
        (0, 1, [-2.17961927e-1]),
        (1, 1, [-5.28510182e-1, -4.05119877e-1]),
    ]


# --------------------------------------------------------
# D states
# --------------------------------------------------------


class Sr88_D2_Vaillant2024(KMatrixModel):
    """Combined 6-channel model for the strongly mixed 5snd 1D2 and 5snd 3D2 series."""

    species = "Sr88"
    name = "D J=2, nu > 5.7"
    f_tot, parity = (2, +1)
    nu_range = (5.7, np.inf)  # fitted to 5snd 1D2 and 3D2, n = 8-30, and the 4d2 3P2 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetJJ(l_c=0, l_r=2, j_c=0.5, j_r=2.5, j_tot=2, species="Sr88"),  # 5s_1/2 nd_5/2
        AngularKetJJ(l_c=0, l_r=2, j_c=0.5, j_r=1.5, j_tot=2, species="Sr88"),  # 5s_1/2 nd_3/2
        AngularKetJJ(l_c=2, l_r=0, j_c=2.5, j_r=0.5, j_tot=2, species="Sr88"),  # 4d_5/2 ns_1/2
        AngularKetJJ(l_c=2, l_r=0, j_c=1.5, j_r=0.5, j_tot=2, species="Sr88"),  # 4d_3/2 ns_1/2
        AngularKetLS(l_c=1, l_r=1, l_tot=2, s_tot=0, j_tot=2, species="Sr88"),  # 5pnp 1D2
        AngularKetLS(l_c=2, l_r=2, l_tot=1, s_tot=1, j_tot=2, species="Sr88"),  # 4dnd 3P2
    ]

    # The elements marked with (*) have the opposite sign compared to Table III of the Addendum,
    # due to the different phase convention of the jj-coupled kets, see the module docstring.
    k_matrix = [
        (0, 0, [-3.85388310e-1, -1.77532611e0]),
        (0, 1, [-2.30810347e-1]),  # (*)
        (0, 2, [-2.99689799e-1]),
        (0, 3, [-6.24839129e-1]),  # (*)
        (0, 4, [-2.38162135e-1]),
        (0, 5, [-8.94462406e-2]),
        (1, 1, [-4.88187724e-1, 2.05255442e0]),
        (1, 2, [6.41169781e-1]),  # (*)
        (1, 3, [8.10126226e-6]),
        (1, 4, [4.84958202e-1]),  # (*)
        (1, 5, [-2.42734982e-3]),  # (*)
        (2, 2, [1.13622499e0, 4.73380410e0]),
        (2, 3, [-2.07880487e-1]),  # (*)
        (3, 3, [1.12383070e0, 3.98916238e0]),
        (4, 4, [6.11787736e-1, 5.29286844e0]),
        (5, 5, [2.20539967e0, 6.07956188e0]),
    ]


class Sr88_D1_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "D J=1, nu > 9.5"
    f_tot, parity = (1, +1)
    nu_range = (9.5, np.inf)  # fitted to 5snd 3D1, n = 12-15 and 17-50
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, species="Sr88"),  # 5snd 3D1
        AngularKetLS(l_c=2, l_r=0, l_tot=2, s_tot=1, j_tot=1, species="Sr88", label=LABEL_4D_AVERAGED),  # 4dns 3D1
    ]

    k_matrix = [
        (0, 0, [-7.40335913e-1, 9.68468100e-1]),
        (0, 1, [5.50457207e-1]),
        (1, 1, [1.46140049e0, 2.77735257e-1]),
    ]


class Sr88_D3_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "D J=3, nu > 3.9"
    f_tot, parity = (3, +1)
    nu_range = (3.9, np.inf)  # fitted to 5snd 3D3, n = 6-8, 9-21 and 23-29
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Sr88"),  # 5snd 3D3
        AngularKetLS(l_c=2, l_r=0, l_tot=2, s_tot=1, j_tot=3, species="Sr88", label=LABEL_4D_AVERAGED),  # 4dns 3D3
        AngularKetLS(l_c=2, l_r=2, l_tot=2, s_tot=1, j_tot=3, species="Sr88"),  # 4dnd 3D3
    ]

    k_matrix = [
        (0, 0, [-7.79385678e-1, 1.07199739e0]),
        (0, 1, [4.36019812e-1]),
        (0, 2, [2.22978797e-1]),
        (1, 1, [1.21231421e0, 8.51416131e0]),
        (1, 2, [-1.68322545e-4]),
        (2, 2, [-2.23826516e-1, 5.54442589e0]),
    ]


# --------------------------------------------------------
# F states
# --------------------------------------------------------


class Sr88_F3_Singlet_Vaillant2024(KMatrixModel):
    species = "Sr88"
    name = "F J=3 singlet, nu > 3.5"
    f_tot, parity = (3, -1)
    nu_range = (3.5, np.inf)  # fitted to 5snf 1F3, n = 4-29, and the 4d5p 1F3 perturber
    reference = REFERENCE_VAILLANT_2024

    outer_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, species="Sr88"),  # 5snf 1F3
        AngularKetLS(l_c=2, l_r=1, l_tot=3, s_tot=0, j_tot=3, species="Sr88"),  # 4dnp 1F3
    ]

    k_matrix = [
        (0, 0, [1.71163074e-1, -3.53036781e-1]),
        (0, 1, [4.50595126e-1]),
        (1, 1, [-6.97829350e-1, -1.31850526e0]),
    ]


# The triplet F series are not covered by the Addendum, which revisited only the singlet and triplet S, P and D
# series and the singlet F series. The three-channel models of the original 2014 paper (table IX, with the
# 5snf 3FJ, 4dnp 3FJ and 4dnf 3FJ channels) are not usable instead: they are among the parameters affected by the
# sign error of the fitting program (they give quantum defects of ~0.19 instead of the measured ~0.12, i.e.
# energies that are off by 1-18 1/cm for n = 10-25) and were not refitted.
# We therefore describe the triplet F series by single channel models with the Rydberg-Ritz quantum defects of
# Connerade 1992, which are the values quoted in table 2.3 of the thesis of Vaillant (Durham University, 2014)
# and the ones used by the default Sr88 models (Robicheaux 2019), except for the 3F4 series, where Robicheaux
# uses the delta_2 of the 3F2,3 series.
# These quantum defects were not determined with the ionization threshold of the Addendum, which shifts all
# energies of these three models by a constant <= 0.005 1/cm (150 MHz). This is small compared to the
# uncertainty of the quantum defects themselves (0.001 in delta_0 corresponds to 0.07 1/cm at n = 15).


class Sr88_F2_Connerade1992(EigenChannelModel):
    """5snf 3F2 series, described by its Rydberg-Ritz quantum defects, since Vaillant 2024 has no triplet F models."""

    species = "Sr88"
    name = "F J=2 (Rydberg-Ritz), nu > 9"
    f_tot, parity = (2, -1)
    nu_range = (9.0, np.inf)  # quantum defects fitted to 5snf 3F2, n = 10-24
    reference = REFERENCE_CONNERADE_1992

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, species="Sr88"),  # 5snf 3F2
    ]
    outer_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, species="Sr88"),  # 5snf 3F2
    ]

    eigen_quantum_defects = [
        [0.120, -2.2, 120],
    ]


class Sr88_F3_Triplet_Connerade1992(EigenChannelModel):
    """5snf 3F3 series, described by its Rydberg-Ritz quantum defects, since Vaillant 2024 has no triplet F models.

    The singlet 5snf 1F3 series of the same symmetry is described by :class:`Sr88_F3_Singlet_Vaillant2024`.
    Both series are treated as independent here, which is the same approximation as in the default Sr88 models
    (there, the 1F3 and 3F3 channels are the two uncoupled eigen channels of a single F J=3 model).
    """

    species = "Sr88"
    name = "F J=3 triplet (Rydberg-Ritz), nu > 9"
    f_tot, parity = (3, -1)
    nu_range = (9.0, np.inf)  # quantum defects fitted to 5snf 3F3, n = 10-24
    reference = REFERENCE_CONNERADE_1992

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, species="Sr88"),  # 5snf 3F3
    ]
    outer_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, species="Sr88"),  # 5snf 3F3
    ]

    eigen_quantum_defects = [
        [0.120, -2.2, 120],
    ]


class Sr88_F4_Connerade1992(EigenChannelModel):
    """5snf 3F4 series, described by its Rydberg-Ritz quantum defects, since Vaillant 2024 has no triplet F models."""

    species = "Sr88"
    name = "F J=4 (Rydberg-Ritz), nu > 9"
    f_tot, parity = (4, -1)
    nu_range = (9.0, np.inf)  # quantum defects fitted to 5snf 3F4, n = 10-24
    reference = REFERENCE_CONNERADE_1992

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, species="Sr88"),  # 5snf 3F4
    ]
    outer_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, species="Sr88"),  # 5snf 3F4
    ]

    eigen_quantum_defects = [
        [0.120, -2.4, 120],
    ]
