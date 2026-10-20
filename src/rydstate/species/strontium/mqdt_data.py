from __future__ import annotations

from rydstate.qdt.mqdt import ModelSetMQDT
from rydstate.qdt.mqdt.model_mqdt import get_model_classes
from rydstate.species.strontium import (
    sr87_eigen_channel_model_data,
    sr88_eigen_channel_model_data,
    sr88_vaillant2024_k_matrix_model_data,
)


class MQDTStrontium87(ModelSetMQDT):
    """MQDT models for Sr87 of F. Robicheaux in the eigenchannel formulation.

    See :mod:`~rydstate.species.strontium.sr87_eigen_channel_model_data` for the models and references.
    Apart from the Robicheaux 2019 models, it contains a model for the 5s5p states based on NIST data.
    """

    species = "Sr87"

    model_classes = get_model_classes(sr87_eigen_channel_model_data, species)


class MQDTStrontium88(ModelSetMQDT):
    """MQDT models for Sr88 of F. Robicheaux in the eigenchannel formulation.

    See :mod:`~rydstate.species.strontium.sr88_eigen_channel_model_data` for the models and references.
    Apart from the Robicheaux 2019 models, it contains a model for the 5s5p 1P1 and 3P1 states fitted to NIST data.
    The Robicheaux models only contain the 5snl channels and are valid for high effective principal quantum numbers,
    see :class:`MQDTStrontium88Vaillant2024` for models including the doubly excited perturber channels.
    """

    species = "Sr88"

    model_classes = get_model_classes(sr88_eigen_channel_model_data, species)


class MQDTStrontium88Vaillant2024(ModelSetMQDT):
    """MQDT models for Sr88 of Vaillant, Jones and Potvliege in the K-matrix formulation.

    See :mod:`~rydstate.species.strontium.sr88_vaillant2024_k_matrix_model_data` for the models and references.
    Compared to the default models (:class:`MQDTStrontium88`, based on Robicheaux 2019), these models include
    the doubly excited perturber channels (4dnl, 5pnp) explicitly and are therefore valid down to much lower
    principal quantum numbers. The only series not covered by Vaillant 2024 are the 5snf 3FJ series, for which
    single channel models with the Rydberg-Ritz quantum defects of Robertson 2021 (ARC 3.0) are used instead.
    These models are available for the separate species "Sr88_vaillant2024",
    see :class:`~rydstate.species.strontium.ElementPropertiesStrontium88Vaillant2024`.
    """

    species = "Sr88_vaillant2024"

    model_classes = get_model_classes(sr88_vaillant2024_k_matrix_model_data, species)
