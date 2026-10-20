from __future__ import annotations

from rydstate.qdt.mqdt import ModelSetMQDT
from rydstate.qdt.mqdt.model_mqdt import get_model_classes
from rydstate.species.ytterbium import (
    yb171_eigen_channel_model_data,
    yb173_eigen_channel_model_data,
    yb174_eigen_channel_model_data,
)


class MQDTYtterbium171(ModelSetMQDT):
    species = "Yb171"
    is_default = True

    model_classes = get_model_classes(yb171_eigen_channel_model_data, species)


class MQDTYtterbium173(ModelSetMQDT):
    species = "Yb173"
    is_default = True

    model_classes = get_model_classes(yb173_eigen_channel_model_data, species)


class MQDTYtterbium174(ModelSetMQDT):
    species = "Yb174"
    is_default = True

    model_classes = get_model_classes(yb174_eigen_channel_model_data, species)
