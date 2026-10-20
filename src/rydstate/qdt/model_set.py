from __future__ import annotations

from abc import ABC
from typing import ClassVar

from rydstate.element_properties import get_element_properties
from rydstate.metaclass_cache import CachedABCMeta


class ModelSet(ABC, metaclass=CachedABCMeta):
    """Base class for the sets of quantum defect theory models of a species.

    This holds what all model sets have in common, independent of the quantum defect theory used,
    see e.g. :class:`~rydstate.qdt.ModelSetMQDT` and :class:`~rydstate.qdt.ModelSetSQDT`.
    The concrete model sets of the different species are subclasses of these,
    which are identified by their :attr:`species`, i.e. there is at most one MQDT and one SQDT model set per species.
    """

    species: ClassVar[str]
    """The short name of the atomic species."""

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)
