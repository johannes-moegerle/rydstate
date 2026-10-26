from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, cast

from rydstate.element_properties import get_element_properties

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetBase


class Model(ABC):
    """Base class for a quantum defect theory model of one symmetry block (f_tot and parity) of a species.

    This class holds what all models have in common, independent of the quantum defect theory used,
    see e.g. :class:`~rydstate.qdt.ModelMQDT` and :class:`~rydstate.qdt.ModelSQDT`.
    The attributes are either defined as class attributes of a subclass (as for the MQDT models)
    or set when constructing the model (as for the SQDT models).
    """

    species: str
    """The species for which the model is defined."""
    name: str
    """The name of the model."""

    reference: str | tuple[str, ...] | None = None
    """Reference for the model, e.g., a publication doi where the model is described."""

    f_tot: float
    """Total angular momentum f_tot of the Rydberg state."""

    parity: int
    """Parity (-1 or +1) of the Rydberg state, i.e. (-1)^(l_c + l_r) of its channels."""

    outer_channels: list[AngularKetBase[Any]]
    """List of outer channels of the model (a single channel for SQDT models)."""

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)

        self.ionization_thresholds_au: list[float] = [
            self.element_properties.get_ionization_threshold(channel.get_core_ket(), unit="a.u.")
            for channel in self.outer_channels
        ]
        """Ionization thresholds of the outer channels in atomic units (Hartree),
        see :meth:`~rydstate.element_properties.ElementProperties.get_ionization_threshold`.
        The channel-dependent effective principal quantum numbers nui are defined with respect to these."""
        self.reference_ionization_threshold_au: float = self.element_properties.get_reference_ionization_threshold(
            unit="a.u."
        )
        """Reference ionization threshold in atomic units (Hartree), with respect to which nu is defined,
        see :meth:`~rydstate.element_properties.ElementProperties.get_reference_ionization_threshold`."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.full_name})"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], directory: str | PathLike[str] | None = None) -> Self:
        """Create a model from a dictionary describing it, e.g. as loaded from a json file.

        The "model_class" of the dictionary is the name of the class of the model,
        which must be this class or one of its subclasses.
        The model is created by the ``from_dict`` of that class, see e.g. :meth:`~rydstate.qdt.ModelSQDT.from_dict`.

        Args:
            data: The dictionary describing the model.
            directory: The directory relative to which file paths in the dictionary are resolved,
                e.g. the directory of the json file. Default is None, i.e. the current working directory.

        """
        model_class = data.get("model_class")
        subclasses = [subclass for subclass in _get_subclasses(cls) if subclass.__name__ == model_class]
        if len(subclasses) != 1:
            raise ValueError(
                f"Expected exactly one subclass of {cls.__name__} named {model_class!r}, "
                f"but found {len(subclasses)}: {data}"
            )
        subclass = subclasses[0]
        if subclass.from_dict.__func__ is Model.from_dict.__func__:  # type: ignore [attr-defined]
            raise NotImplementedError(f"{subclass.__name__} cannot be created from a dictionary.")
        return cast("Self", subclass.from_dict(data, directory))

    @property
    def full_name(self) -> str:
        """Return the full name of the model, combining species and model name."""
        return f"{self.species} {self.name}"


def _get_subclasses(cls: type[Model]) -> list[type[Model]]:
    """Return the class itself and all its (direct and indirect) subclasses."""
    subclasses = [cls]
    for subclass in cls.__subclasses__():
        subclasses.extend(_get_subclasses(subclass))
    return subclasses
