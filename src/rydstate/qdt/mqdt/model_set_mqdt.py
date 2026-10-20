from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rydstate.angular.utils import is_not_set
from rydstate.qdt.model_set import ModelSet
from rydstate.qdt.mqdt.model_eigen_channel import TrivialModel
from rydstate.utils import get_all_subclasses

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.core_ket import CoreKet
    from rydstate.qdt.mqdt.model_mqdt import ModelMQDT


class ModelSetMQDT(ModelSet):
    """Base class for the MQDT model sets of the different species."""

    model_classes: ClassVar[list[type[ModelMQDT]]]
    """List of the MQDT :class:`~rydstate.qdt.ModelMQDT` models available for this species.

    :meta hide-value:

    """

    def __init__(self) -> None:
        super().__init__()
        self.models: list[ModelMQDT] = [model_class() for model_class in self.model_classes]

    def __repr__(self) -> str:
        return f"ModelSetMQDT({self.species})"

    def get_mqdt_models(self, outer_channel: AngularKetBase[Any]) -> list[ModelMQDT]:
        """Return a list of MQDT models for the outer_channel."""
        if not is_not_set(outer_channel.m):
            raise ValueError("The m quantum number of the outer_channel must be NotSet.")
        models = [
            model
            for model in self.models
            if any(abs(outer_channel.calc_reduced_overlap(ket)) > 0 for ket in model.outer_channels)
        ]
        if len(models) == 0:
            models = [TrivialModel(self.species, outer_channel)]
        return models

    def get_core_kets(self) -> list[CoreKet]:
        """Return a list of relevant core kets of all MQDT models."""
        core_kets = set()
        for model in self.models:
            core_kets.update(model.get_core_kets())
        return sorted(core_kets, key=lambda ket: (ket.l_c, ket.j_c, ket.f_c, str(ket.label)))


def get_mqdt(species: str) -> ModelSetMQDT:
    """Get an instance of the subclass of ModelSetMQDT for the given species."""
    subclasses = get_all_subclasses(ModelSetMQDT, species)
    if len(subclasses) == 0:
        raise ValueError(f"No subclass of ModelSetMQDT found for {species=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of ModelSetMQDT found for {species=}: {subclasses}.")
