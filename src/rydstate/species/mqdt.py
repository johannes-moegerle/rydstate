from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate.angular.utils import is_not_set
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.species.fmodel import FModelSQDT
from rydstate.species.utils import get_all_subclasses
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.core_ket import CoreKet
    from rydstate.species.fmodel import FModel
    from rydstate.units import PintFloat


class MQDT(ABC, metaclass=CachedABCMeta):
    """Base class for all MQDT classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str | None] = None
    """The tag for these MQDT parameters."""
    is_default: ClassVar[bool] = False
    """Whether this MQDT is the default MQDT for the species."""

    ionization_threshold_dict: ClassVar[dict[CoreKet, tuple[float, str]]]
    """Dictionary containing the ionization thresholds for the different core states.
    The thresholds are given in the form of a tuple (ionization_threshold, unit).
    """
    reference_ionization_threshold_tuple: ClassVar[tuple[float, str] | None] = None
    """The reference ionization threshold, which is used to define the zero of energy for the MQDT models.
    If None, it is defined as the smallest ionization threshold in the ionization_threshold_dict."""

    model_classes: ClassVar[list[type[FModel]]]
    """List of the MQDT :class:`~rydstate.species.fmodel.FModel` models available for this species.

    :meta hide-value:

    """

    def __init__(self) -> None:
        self.models: list[FModel] = [model_class(self) for model_class in self.model_classes]

    def __repr__(self) -> str:
        return f"MQDT({self.species}, {self.tag})"

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: str) -> float: ...

    def get_ionization_threshold(self, core_ket: CoreKet, unit: str | None = "hartree") -> PintFloat | float:
        """Return the ionization threshold of the channel given by the core_ket in the desired unit.

        Args:
            core_ket: The core ket for which to return the ionization threshold.
            unit: Desired unit for the ionization threshold. Default is atomic units "hartree".

        Returns:
            Ionization threshold in the desired unit.

        """
        try:
            matching_core_ket = core_ket.find_matching_core_ket(self.ionization_threshold_dict.keys())
        except ValueError as e:
            raise ValueError(f"Ionization threshold for core ket {core_ket} is not defined.") from e

        ionization_threshold_tuple = self.ionization_threshold_dict[matching_core_ket]
        ionization_threshold: PintFloat = ureg.Quantity(ionization_threshold_tuple[0], ionization_threshold_tuple[1])
        ionization_threshold = ionization_threshold.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_threshold
        if unit == "a.u.":
            return ionization_threshold.magnitude
        return ionization_threshold.to(unit, "spectroscopy").magnitude

    @cached_property
    def reference_ionization_threshold_au(self) -> float:
        """Reference ionization threshold in atomic units (Hartree).

        If no reference_ionization_threshold_tuple is defined, we define the reference ionization threshold as
        the smallest ionization threshold in the ionization_threshold_dict.
        """
        if self.reference_ionization_threshold_tuple is not None:
            return (
                ureg.Quantity(
                    self.reference_ionization_threshold_tuple[0], self.reference_ionization_threshold_tuple[1]
                )
                .to("hartree", "spectroscopy")
                .magnitude
            )
        return min(self.get_ionization_threshold(core_ket, unit="a.u.") for core_ket in self.ionization_threshold_dict)

    def get_mqdt_models(self, outer_channel: AngularKetBase[Any]) -> list[FModel]:
        """Return a list of MQDT models for the outer_channel."""
        if not is_not_set(outer_channel.m):
            raise ValueError("The m quantum number of the outer_channel must be NotSet.")
        models = [
            model
            for model in self.models
            if any(abs(outer_channel.calc_reduced_overlap(ket)) > 0 for ket in model.outer_channels)
        ]
        if len(models) == 0:
            models = [FModelSQDT(self.species, outer_channel, mqdt=self)]
        return models

    def get_core_kets(self) -> list[CoreKet]:
        """Return a list of relevant core kets of all FModels."""
        core_kets = set()
        for model in self.models:
            core_kets.update(model.get_core_kets())
        return sorted(core_kets, key=lambda ket: (ket.l_c, ket.j_c, ket.f_c, str(ket.label)))


def get_mqdt(species: str, tag: str | None = None) -> MQDT:
    """Get an instance of the subclass of MQDT for the given species and tag."""
    subclasses = get_all_subclasses(MQDT, species, tag)

    if tag is None:
        subclasses = [cls for cls in subclasses if getattr(cls, "is_default", False)]

    if len(subclasses) == 0:
        raise ValueError(f"No subclass of MQDT found for {species=} and {tag=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of MQDT found for {species=} and {tag=}: {subclasses}.")
