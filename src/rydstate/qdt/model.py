from __future__ import annotations

from abc import ABC

from rydstate.element_properties import get_element_properties


class Model(ABC):
    """Base class for a quantum defect theory model of one symmetry block (f_tot and parity) of a species.

    The model is valid in a limited range of nu.
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

    nu_range: tuple[float, float]
    """Range of effective principal quantum numbers nu for which the model is valid."""

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.full_name})"

    @property
    def full_name(self) -> str:
        """Return the full name of the model, combining species and model name."""
        return f"{self.species} {self.name}"

    @property
    def nu_min(self) -> float:
        """Minimum nu for which the model is valid."""
        return self.nu_range[0]

    @property
    def nu_max(self) -> float:
        """Maximum nu for which the model is valid."""
        return self.nu_range[1]
