from __future__ import annotations

import numpy as np

from rydstate.angular.angular_ket import AngularKetFJ, AngularKetLS
from rydstate.angular.core_ket_base import CoreKet
from rydstate.species.mqdt.fmodel import FModel
from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT
from rydstate.units import electron_mass, rydberg_constant


class Strontium87MQDT(SpeciesObjectMQDT):
    name = "Sr87_mqdt"
    Z = 38
    i_c = 4.5
    number_valence_electrons = 2

    potential_type_default = "model_potential_fei_2009"

    _isotope_mass = 86.9088774970  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    _ionization_threshold_dict = {
        CoreKet(i_c, 0.5, 0, 0.5, 4): (45932.287373577, None, "1/cm"),
        CoreKet(i_c, 0.5, 0, 0.5, 5): (45932.120512528, None, "1/cm"),
    }
    core_ground_state = CoreKet(i_c, 0.5, 0, 0.5, 4)
    nuclear_dipole = -1.0936030


# --------------------------------------------------------
# MQDT models valid at large n
# --------------------------------------------------------


class Sr87_S35_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "S F=7/2, nu > 11"
    f_tot = 3.5
    nu_range = (11.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=3.5, species="Sr87"),  # "5sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=4, j_r=0.5, f_tot=3.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [3.370778, 0.418, -0.3],
    ]
    mixing_angles = []


class Sr87_S45_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "S F=9/2, nu > 11"
    f_tot = 4.5
    nu_range = (11.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, f_tot=4.5, species="Sr87"),  # "5sns 1S0"
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=4.5, species="Sr87"),  # "5sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=4, j_r=0.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=5, j_r=0.5, f_tot=4.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [3.26896, -0.138, 0.9],
        [3.370778, 0.418, -0.3],
    ]
    mixing_angles = []


class Sr87_S55_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "S F=11/2, nu > 11"
    f_tot = 5.5
    nu_range = (11.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=1, j_tot=1, f_tot=5.5, species="Sr87"),  # "5sns 3S1"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=0, j_c=0.5, f_c=5, j_r=0.5, f_tot=5.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [3.370778, 0.418, -0.3],
    ]
    mixing_angles = []


# --------------------------------------------------------
# Low-n models
# --------------------------------------------------------


class Sr87_P45_LowN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=9/2 (clock), 1.8 < nu < 2.2"
    f_tot = 4.5
    nu_range = (1.8, 2.2)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=4.5, species="Sr87"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=4.5, species="Sr87"),  # "5snp 3P0"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=4.5, species="Sr87"),  # "5snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=4.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=0.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=1.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=0.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=1.5, f_tot=4.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [0.8720737, 0],
        [0.13689075, 0],
        [0.13143188, 0],
        [0.11955235, 0],
    ]
    mixing_angles = []


# --------------------------------------------------------
# High-n P models
# --------------------------------------------------------


class Sr87_P25_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=5/2, nu > 5"
    f_tot = 2.5
    nu_range = (5.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=2.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=1.5, f_tot=2.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.882, 0.446, -1.9],
    ]
    mixing_angles = []


class Sr87_P35_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=7/2, nu > 5"
    f_tot = 3.5
    nu_range = (5.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=3.5, species="Sr87"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=3.5, species="Sr87"),  # "5snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=3.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=0.5, f_tot=3.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=1.5, f_tot=3.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=1.5, f_tot=3.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.724, -4.67, -157],
        [2.8826, 0.407, -1.3],
        [2.882, 0.446, -1.9],
    ]
    mixing_angles = []


class Sr87_P45_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=9/2, nu > 7"
    f_tot = 4.5
    nu_range = (7.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=4.5, species="Sr87"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=0, f_tot=4.5, species="Sr87"),  # "5snp 3P0"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=4.5, species="Sr87"),  # "5snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=4.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=0.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=1.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=0.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=1.5, f_tot=4.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.724, -4.67, -157],
        [2.8867, 0.44, -1.9],
        [2.8826, 0.407, -1.3],
        [2.882, 0.446, -1.9],
    ]
    mixing_angles = []


class Sr87_P55_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=11/2, nu > 5"
    f_tot = 5.5
    nu_range = (5.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=0, j_tot=1, f_tot=5.5, species="Sr87"),  # "5snp 1P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=1, f_tot=5.5, species="Sr87"),  # "5snp 3P1"
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=5.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=4, j_r=1.5, f_tot=5.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=0.5, f_tot=5.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=1.5, f_tot=5.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.724, -4.67, -157],
        [2.8826, 0.407, -1.3],
        [2.882, 0.446, -1.9],
    ]
    mixing_angles = []


class Sr87_P65_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "P F=13/2, nu > 5"
    f_tot = 6.5
    nu_range = (5.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=1, l_tot=1, s_tot=1, j_tot=2, f_tot=6.5, species="Sr87"),  # "5snp 3P2"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=5, j_r=1.5, f_tot=6.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.882, 0.446, -1.9],
    ]
    mixing_angles = []


# --------------------------------------------------------
# High-n D models
# --------------------------------------------------------


class Sr87_D15_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=3/2, nu > 25"
    f_tot = 1.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=1.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=1.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.655, -41.4, -15363],
    ]
    mixing_angles = []


class Sr87_D25_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=5/2, nu > 25"
    f_tot = 2.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=2.5, species="Sr87"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=2.5, species="Sr87"),  # "5snd 3D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=2.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=1.5, f_tot=2.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=2.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=2.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.3847, -39.41, -1090],
        [2.66149, -16.77, -6656],
        [2.655, -41.4, -15363],
    ]
    mixing_angles = [
        (0, 1, -0.14),
    ]


class Sr87_D35_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=7/2, nu > 25"
    f_tot = 3.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=3.5, species="Sr87"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=3.5, species="Sr87"),  # "5snd 3D1"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=3.5, species="Sr87"),  # "5snd 3D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=3.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=1.5, f_tot=3.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=3.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=1.5, f_tot=3.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=3.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.3847, -39.41, -1090],
        [2.67524, -13.15, -4444],
        [2.66149, -16.77, -6656],
        [2.655, -41.4, -15363],
    ]
    mixing_angles = [
        (0, 2, -0.14),
    ]


class Sr87_D45_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=9/2, nu > 25"
    f_tot = 4.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=4.5, species="Sr87"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=4.5, species="Sr87"),  # "5snd 3D1"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=4.5, species="Sr87"),  # "5snd 3D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=4.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=1.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=1.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=4.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.3847, -39.41, -1090],
        [2.67524, -13.15, -4444],
        [2.66149, -16.77, -6656],
        [2.655, -41.4, -15363],
    ]
    mixing_angles = [
        (0, 2, -0.14),
    ]


class Sr87_D55_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=11/2, nu > 25"
    f_tot = 5.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=5.5, species="Sr87"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=1, f_tot=5.5, species="Sr87"),  # "5snd 3D1"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=5.5, species="Sr87"),  # "5snd 3D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=5.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=1.5, f_tot=5.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=5.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=1.5, f_tot=5.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=5.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.3847, -39.41, -1090],
        [2.67524, -13.15, -4444],
        [2.66149, -16.77, -6656],
        [2.655, -41.4, -15363],
    ]
    mixing_angles = [
        (0, 2, -0.14),
    ]


class Sr87_D65_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=13/2, nu > 25"
    f_tot = 6.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, f_tot=6.5, species="Sr87"),  # "5snd 1D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, f_tot=6.5, species="Sr87"),  # "5snd 3D2"
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=6.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=4, j_r=2.5, f_tot=6.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=1.5, f_tot=6.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=6.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.3847, -39.41, -1090],
        [2.66149, -16.77, -6656],
        [2.655, -41.4, -15363],
    ]
    mixing_angles = [
        (0, 1, -0.14),
    ]


class Sr87_D75_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "D F=15/2, nu > 25"
    f_tot = 7.5
    nu_range = (25.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=3, f_tot=7.5, species="Sr87"),  # "5snd 3D3"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=2, j_c=0.5, f_c=5, j_r=2.5, f_tot=7.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [2.655, -41.4, -15363],
    ]
    mixing_angles = []


# --------------------------------------------------------
# High-n F models
# --------------------------------------------------------


class Sr87_F45_HighN(FModel):
    species_name = "Sr87_mqdt"
    name = "F F=9/2, nu > 9"
    f_tot = 4.5
    nu_range = (9.0, np.inf)
    reference = "10.1088/1361-6455/ab3c26"

    inner_channels = [
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=0, j_tot=3, f_tot=4.5, species="Sr87"),  # "5snf 1F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=2, f_tot=4.5, species="Sr87"),  # "5snf 3F2"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=3, f_tot=4.5, species="Sr87"),  # "5snf 3F3"
        AngularKetLS(l_c=0, l_r=3, l_tot=3, s_tot=1, j_tot=4, f_tot=4.5, species="Sr87"),  # "5snf 3F4"
    ]
    outer_channels = [
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=4, j_r=2.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=4, j_r=3.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=5, j_r=2.5, f_tot=4.5, species="Sr87"),
        AngularKetFJ(l_c=0, l_r=3, j_c=0.5, f_c=5, j_r=3.5, f_tot=4.5, species="Sr87"),
    ]

    eigen_quantum_defects = [
        [0.12, -2.2, 120],
        [0.12, -2.2, 120],
        [0.12, -2.2, 120],
        [0.12, -2.2, 120],
    ]
    mixing_angles = []
