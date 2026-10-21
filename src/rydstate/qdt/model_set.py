from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING

from rydstate.element_properties import get_element_properties

if TYPE_CHECKING:
    from rydstate.element_properties import ElementProperties


class ModelSet(ABC):
    """Base class for the sets of quantum defect theory models of a species.

    This holds what all model sets have in common, independent of the quantum defect theory used,
    see e.g. :class:`~rydstate.qdt.ModelSetMQDT` and :class:`~rydstate.qdt.ModelSetSQDT`.
    There is at most one MQDT and one SQDT model set per species, see :func:`~rydstate.qdt.get_mqdt`
    and :func:`~rydstate.qdt.get_sqdt`.
    """

    species: str
    """The short name of the atomic species."""

    @cached_property
    def element_properties(self) -> ElementProperties:
        """The element properties of the species."""
        return get_element_properties(self.species)
