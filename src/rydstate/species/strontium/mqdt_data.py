from __future__ import annotations

from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import Unknown
from rydstate.species.mqdt import MQDT
from rydstate.species.mqdt_model import get_model_classes
from rydstate.species.strontium import (
    sr87_eigen_channel_model_data,
    sr88_eigen_channel_model_data,
    sr88_vaillant2024_k_matrix_model_data,
)


class MQDTStrontium87(MQDT):
    species = "Sr87"
    is_default = True

    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=4): (45932.287373577, "1/cm"),
        CoreKet(i_c=4.5, n_c=5, l_c=0, j_c=0.5, f_c=5): (45932.120512528, "1/cm"),
    }
    # hyperfine centroid of the two F thresholds above, i.e. their (2F+1)-weighted mean
    # (9 * 45932.287373577 + 11 * 45932.120512528) / 20, which is the reference used for the Sr87 quantum defects
    # ("weighted threshold" of F. Robicheaux, J. Phys. B 52, 244001 (2019)).
    reference_ionization_threshold_tuple = (45932.1956, "1/cm")
    model_classes = get_model_classes(sr87_eigen_channel_model_data, species)


class MQDTStrontium88(MQDT):
    species = "Sr88"
    is_default = True

    # Couturier et al., Phys. Rev. A 99, 022503 (2019) (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.022503)
    # I = 1_377_012_721(10) MHz (= 45932.2002 1/cm), same as is used in the Addendum of
    # Vaillant et al. 2024 (https://doi.org/10.1088/1361-6455/ad76f0)
    # Since in sr88_eigen_channel_model_data only this single threshold is used,
    # changing the value here (compared to F. Robicheaux 2019) does not affect the models.
    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=5, l_c=0, j_c=0.5): (1_377_012_721, "MHz"),
    }
    model_classes = get_model_classes(sr88_eigen_channel_model_data, species)


class MQDTStrontium88Vaillant2024(MQDT):
    """MQDT models for Sr88 of Vaillant, Jones and Potvliege in the K-matrix formulation.

    See :mod:`~rydstate.species.strontium.sr88_vaillant2024_k_matrix_model_data` for the models and references.
    Compared to the default models (:class:`MQDTStrontium88`, based on Robicheaux 2019), these models include
    the doubly excited perturber channels (4dnl, 5pnp) explicitly and are therefore valid down to much lower
    principal quantum numbers, but they do not cover the triplet F series.
    """

    species = "Sr88"
    tag = "vaillant2024"
    is_default = False

    # Ionization thresholds as used in the Addendum (Table I), the 5s threshold and the mass corrected Rydberg
    # constant are taken from Couturier 2019 (Phys. Rev. A 99, 022503), the 4d and 5p thresholds are the
    # 5s threshold plus the Sr+ 4d_3/2, 4d_5/2, 5p_1/2 and 5p_3/2 excitation energies (Sansonetti 2012).
    # For LS-coupled channels (4dnl, 5pnp), the unweighted average of the fine structure thresholds is used.
    ionization_threshold_dict: ClassVar = {
        CoreKet(i_c=0, n_c=5, l_c=0, j_c=0.5): (45932.2002, "1/cm"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=1.5): (60488.09, "1/cm"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=2.5): (60768.43, "1/cm"),
        CoreKet(i_c=0, n_c=4, l_c=2, j_c=Unknown, label=Unknown): (60628.26, "1/cm"),
        CoreKet(i_c=0, n_c=5, l_c=1, j_c=Unknown, label=Unknown): (70048.11, "1/cm"),
    }
    reference_ionization_threshold_tuple = (45932.2002, "1/cm")
    model_classes = get_model_classes(sr88_vaillant2024_k_matrix_model_data, species)
