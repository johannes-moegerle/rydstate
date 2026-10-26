from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.utils import is_not_set
from rydstate.qdt.model import Model
from rydstate.qdt.utils import calc_modified_ritz_formula, check_expansion_coefficients

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetLS
    from rydstate.angular.utils import AllKnown
    from rydstate.qdt.utils import ExpansionCoefficients


class ModelSQDTDict(TypedDict):
    """JSON serializable description of a :class:`ModelSQDT`, see :meth:`ModelSQDT.to_dict`."""

    model_class: str
    """Name of the model class, i.e. "ModelSQDT" (also for a :class:`~rydstate.qdt.ModelSQDTFit`),
    which allows to recreate the model with the correct class."""
    species: str
    channel: str
    """Short label of the channel, see :meth:`~rydstate.angular.angular_ket.AngularKetBase.get_short_label`."""
    quantum_defects: list[float]
    reference: NotRequired[str | list[str] | None]
    nu_range: NotRequired[list[float | Literal["inf"]]]
    """[nu_min, nu_max], where an infinite nu_max is given as the string "inf" (JSON has no infinity)."""
    comment: NotRequired[str]
    """Optional comment, e.g. how the quantum defects were derived from the reference.
    It is only meant for the reader of the json file and ignored by :meth:`ModelSQDT.from_dict`."""


class ModelSQDT(Model):
    r"""SQDT model describing the quantum defects of a single channel of a species.

    In contrast to :class:`~rydstate.qdt.ModelMQDT`, the parameters of an SQDT model are not defined
    in a subclass, but simply given when constructing the model, e.g.

    .. code-block:: python

        ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787], reference="Phys. Rev. A 83, 052515 (2011)")

    The quantum defect of the channel is given by the modified Ritz formula

    .. math::
        \delta(n) = \delta_0 + \frac{\delta_2}{(n - \delta_0)^2} + \frac{\delta_4}{(n - \delta_0)^4} + ...

    see :meth:`calc_quantum_defect`.
    """

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        quantum_defects: ExpansionCoefficients,
        reference: str | tuple[str, ...] | None = None,
        nu_range: tuple[float, float | Literal["inf"]] = (0, math.inf),
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, either as angular ket or as its short label
                (e.g. "ns_1/2" or "5s np 3P1",
                see :meth:`~rydstate.angular.angular_ket.AngularKetBase.from_short_label`).
                The channel must be a single LS ket, i.e. have well defined l_r, j_tot and s_tot,
                and its magnetic quantum number m must be NotSet.
            quantum_defects: The expansion coefficients [d0, d2, d4, ...] of the quantum defect,
                see :meth:`calc_quantum_defect`.
            reference: Reference for the model, e.g., a publication doi where the quantum defects are given.
            nu_range: Range (nu_min, nu_max) of effective principal quantum numbers nu for which the model is valid.
                nu_min must be finite, an infinite nu_max can also be given as the string "inf".
                Default is (0, inf), i.e. the model is valid for all (positive) nu.

        """
        if isinstance(channel, str):
            channel = AngularKetBase.from_short_label(channel, species)
        if channel.coupling_scheme != "LS":
            angular_state = channel.to_state("LS")
            if len(angular_state.kets) != 1:
                raise ValueError(f"The channel {channel} of an SQDT model must be a single LS ket.")
            channel = angular_state.kets[0]
        if not is_not_set(channel.m):
            raise ValueError(f"The magnetic quantum number m of the channel {channel} must be NotSet.")
        if channel.contains_unknown:
            raise ValueError(f"The channel {channel} of an SQDT model must not contain unknown quantum numbers.")

        self.species = species
        self.channel: AngularKetLS[AllKnown] = channel  # type: ignore [assignment]
        self.name = self.channel.get_short_label()
        self.reference = reference
        self.f_tot = self.channel.f_tot
        self.parity = self.channel.parity
        nu_min, nu_max = nu_range
        if nu_max == "inf":
            nu_max = math.inf
        elif isinstance(nu_max, str):
            raise ValueError(f'{self.full_name}: nu_max must be a number or the string "inf", but got {nu_max!r}.')
        if not math.isfinite(nu_min):
            raise ValueError(f"{self.full_name}: nu_min must be finite, but got {nu_min}.")
        self.nu_range = (nu_min, nu_max)
        self.outer_channels = [self.channel]
        check_expansion_coefficients(quantum_defects, f"{self.full_name}: quantum_defects")
        self.quantum_defects: tuple[float, ...] = tuple(quantum_defects)
        """The expansion coefficients (d0, d2, d4, ...) of the quantum defect, see :meth:`calc_quantum_defect`.
        They are stored as a tuple, since the models are shared, see :class:`~rydstate.qdt.ModelSQDTFit`."""
        super().__init__()

        self.ionization_threshold_au = self.ionization_thresholds_au[0]
        """Ionization threshold of the single outer channel in atomic units (Hartree), with respect to which
        the effective principal quantum number nui of the channel is defined."""

    def to_dict(self) -> ModelSQDTDict:
        """Return a JSON serializable dictionary describing the model.

        The model can be recreated from the dictionary via :meth:`from_dict`, e.g. to store a model in a json file:

        .. code-block:: python

            with open("model.json", "w") as file:
                json.dump(model.to_dict(), file, indent=2)
            with open("model.json") as file:
                model = ModelSQDT.from_dict(json.load(file))

        The optional keys "reference" and "nu_range" are only included,
        if they differ from their default value (None and (0, inf), respectively).
        An infinite nu_max is given as the string "inf", since JSON has no infinity.

        """
        data: ModelSQDTDict = {
            "model_class": "ModelSQDT",
            "species": self.species,
            "channel": self.channel.get_short_label(),
            "quantum_defects": list(self.quantum_defects),
        }
        if self.reference is not None:
            data["reference"] = list(self.reference) if isinstance(self.reference, tuple) else self.reference
        nu_min, nu_max = self.nu_range
        if (nu_min, nu_max) != (0, math.inf):
            data["nu_range"] = [nu_min, "inf" if math.isinf(nu_max) else nu_max]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Create a model from a dictionary, as returned by :meth:`to_dict`.

        Args:
            data: The dictionary describing the model, see :class:`ModelSQDTDict`.
                Its "model_class" must be "ModelSQDT".
                The keys "reference", "nu_range" and "comment" are optional, where "comment" is ignored.

        """
        unknown_keys = set(data) - set(ModelSQDTDict.__annotations__)
        if unknown_keys:
            raise ValueError(f"Unknown keys {sorted(unknown_keys)} in the description of an SQDT model: {data}")
        if data["model_class"] != "ModelSQDT":
            raise ValueError(f"The description is of a {data['model_class']}, but ModelSQDT was expected: {data}")

        reference = data.get("reference")
        nu_min, nu_max = data.get("nu_range", (0, math.inf))
        return cls(
            data["species"],
            data["channel"],
            list(data["quantum_defects"]),
            reference=tuple(reference) if isinstance(reference, list) else reference,
            nu_range=(nu_min, nu_max),
        )

    def calc_quantum_defect(self, n: int) -> float:
        r"""Calculate the quantum defect of the state with principal quantum number n via the modified Ritz formula.

        .. math::
            \delta(n) = \delta_0 + \frac{\delta_2}{(n - \delta_0)^2} + \frac{\delta_4}{(n - \delta_0)^4} + ...

        References:
            - On a New Law of Series Spectra, Ritz; DOI: 10.1086/141591, https://ui.adsabs.harvard.edu/abs/1908ApJ....28..237R/abstract
            - Rydberg atoms, Gallagher; DOI: 10.1088/0034-4885/51/2/001, (Eq. 16.19)

        Args:
            n: The principal quantum number of the Rydberg state.

        """
        return calc_modified_ritz_formula(n, self.quantum_defects)

    def calc_nui(self, n: int) -> float:
        """Calculate the effective principal quantum number nui = n - delta(n) of the state with the given n.

        Args:
            n: The principal quantum number of the Rydberg state.

        """
        return n - self.calc_quantum_defect(n)
