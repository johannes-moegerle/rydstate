from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from typing_extensions import NotRequired, TypedDict

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.utils import is_not_set
from rydstate.qdt.model import Model
from rydstate.qdt.utils import calc_modified_ritz_formula, check_expansion_coefficients
from rydstate.utils import calc_energy_from_nu

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetLS
    from rydstate.angular.utils import AllKnown
    from rydstate.qdt.utils import ExpansionCoefficients


class ModelSQDTBaseDict(TypedDict):
    """The keys of the JSON serializable description common to all SQDT models, see :meth:`ModelSQDTBase.to_dict`."""

    model_class: str
    """Name of the model class, e.g. "ModelSQDT" (also for a :class:`~rydstate.qdt.ModelSQDTFit`),
    which allows to recreate the model with the correct class, see :meth:`~rydstate.qdt.Model.from_dict`."""
    species: str
    channel: str
    """Short label of the channel, see :meth:`~rydstate.angular.angular_ket.AngularKetBase.get_short_label`."""
    reference: NotRequired[str | list[str] | None]
    n_range: NotRequired[list[int | Literal["inf"]]]
    """[n_min, n_max], where an infinite n_max is given as the string "inf" (JSON has no infinity)."""
    comment: NotRequired[str]
    """Optional comment, e.g. how the data of the model were derived from the reference.
    It is only meant for the reader of the json file and ignored when creating the model from the description."""


class ModelSQDTDict(ModelSQDTBaseDict):
    """JSON serializable description of a :class:`ModelSQDT`, see :meth:`ModelSQDT.to_dict`."""

    quantum_defects: list[float]


class ModelSQDTBase(Model):
    """Base class for the SQDT models, each describing a single channel of a species.

    The concrete models differ in how they calculate the effective principal quantum number nui
    of the state with a given n, see :class:`~rydstate.qdt.ModelSQDT` (via quantum defects)
    and :class:`~rydstate.qdt.ModelSQDTNist` (via measured energy levels).
    Each model describes the states of its channel with principal quantum number n in its :attr:`n_range`.
    """

    _model_class: ClassVar[str]
    """The "model_class" of the description of the model, see :meth:`to_dict`."""

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        reference: str | tuple[str, ...] | None = None,
        n_range: tuple[int, float | Literal["inf"]] = (1, math.inf),
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, either as angular ket or as its short label
                (e.g. "ns_1/2" or "5s np 3P1",
                see :meth:`~rydstate.angular.angular_ket.AngularKetBase.from_short_label`).
                The channel must be a single LS ket, i.e. have well defined l_r, j_tot and s_tot,
                and its magnetic quantum number m must be NotSet.
            reference: Reference for the model, e.g., a publication doi where the data of the model are given.
            n_range: Range (n_min, n_max) of principal quantum numbers n (both inclusive) for which the model is valid.
                n_min must be an integer, n_max an integer or infinite (which can also be given as the string "inf").
                Default is (1, inf), i.e. the model is valid for all n.

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
        n_min, n_max = n_range
        if n_max == "inf":
            n_max = math.inf
        if not isinstance(n_min, int) or not (isinstance(n_max, int) or n_max == math.inf):
            raise ValueError(
                f"{self.full_name}: n_range must be (n_min, n_max) with integers or n_max = inf, but got {n_range!r}."
            )
        if n_min > n_max:
            raise ValueError(f"{self.full_name}: n_min must not be larger than n_max, but got {n_range!r}.")
        self.n_range: tuple[int, float] = (n_min, n_max)
        """Range (n_min, n_max) of principal quantum numbers n (both inclusive) for which the model is valid."""
        self.outer_channels = [self.channel]
        super().__init__()

        self.ionization_threshold_au = self.ionization_thresholds_au[0]
        """Ionization threshold of the single outer channel in atomic units (Hartree), with respect to which
        the effective principal quantum number nui of the channel is defined."""

    def to_dict(self) -> ModelSQDTBaseDict:
        """Return the keys of the JSON serializable description common to all SQDT models.

        The subclasses add their own keys, so that the model can be recreated via their ``from_dict``,
        see :meth:`~rydstate.qdt.Model.from_dict`.
        The key "reference" is always included (None if no reference is given),
        the optional key "n_range" only if it differs from its default value (1, inf).
        An infinite n_max is given as the string "inf", since JSON has no infinity.

        """
        data: ModelSQDTBaseDict = {
            "model_class": self._model_class,
            "species": self.species,
            "channel": self.channel.get_short_label(),
            "reference": list(self.reference) if isinstance(self.reference, tuple) else self.reference,
        }
        n_min, n_max = self.n_range
        if (n_min, n_max) != (1, math.inf):
            data["n_range"] = [n_min, "inf" if math.isinf(n_max) else int(n_max)]
        return data

    @classmethod
    def _parse_dict(cls, data: Mapping[str, Any], dict_type: type[ModelSQDTBaseDict]) -> dict[str, Any]:
        """Check the description of a model and return the keyword arguments common to all SQDT models.

        Args:
            data: The description of the model, as returned by :meth:`to_dict`.
            dict_type: The TypedDict of the description, whose keys are allowed in the description.

        Returns:
            The keyword arguments "species", "channel", "reference" and "n_range".

        """
        unknown_keys = set(data) - set(dict_type.__annotations__)
        if unknown_keys:
            raise ValueError(f"Unknown keys {sorted(unknown_keys)} in the description of a {cls._model_class}: {data}")
        if data["model_class"] != cls._model_class:
            raise ValueError(
                f"The description is of a {data['model_class']}, but {cls._model_class} was expected: {data}"
            )
        n_min, n_max = data.get("n_range", (1, math.inf))
        reference = data.get("reference")
        return {
            "species": data["species"],
            "channel": data["channel"],
            "reference": tuple(reference) if isinstance(reference, list) else reference,
            "n_range": (n_min, n_max),
        }

    @abstractmethod
    def calc_nui(self, n: int) -> float:
        """Calculate the effective principal quantum number nui of the state with the given n.

        Args:
            n: The principal quantum number of the Rydberg state.

        """

    def calc_binding_energy_au(self, *, n: int) -> float:
        """Calculate the binding energy of the state with the given n in atomic units (Hartree).

        The binding energy is negative and given with respect to :attr:`ionization_threshold_au`,
        see :meth:`~rydstate.rydberg_state.RydbergStateSQDT.get_binding_energy`.
        In contrast to :class:`~rydstate.rydberg_state.RydbergStateSQDT`, it is always calculated from this model,
        also if n is outside of its :attr:`n_range` (e.g. to compare or fit models).

        Args:
            n: The principal quantum number of the Rydberg state (keyword-only, see :meth:`calc_energy_au`).

        """
        properties = self.element_properties
        return calc_energy_from_nu(properties.reduced_mass_au, self.calc_nui(n), properties.net_charge)

    def calc_energy_au(self, *, n: int) -> float:
        """Calculate the energy of the state with the given n in atomic units (Hartree), measured from the ground state.

        The energy is :attr:`ionization_threshold_au` plus the binding energy, see :meth:`calc_binding_energy_au`.

        Args:
            n: The principal quantum number of the Rydberg state. It is keyword-only, so that it is not confused with
                the effective principal quantum number nu of :meth:`~rydstate.qdt.ModelMQDT.calc_energy_au`.

        """
        return self.ionization_threshold_au + self.calc_binding_energy_au(n=n)

    @property
    def n_min(self) -> int:
        """Minimum n for which the model is valid."""
        return self.n_range[0]

    @property
    def n_max(self) -> float:
        """Maximum n for which the model is valid (inclusive, may be infinite)."""
        return self.n_range[1]


class ModelSQDT(ModelSQDTBase):
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

    _model_class = "ModelSQDT"

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        quantum_defects: ExpansionCoefficients,
        reference: str | tuple[str, ...] | None = None,
        n_range: tuple[int, float | Literal["inf"]] = (1, math.inf),
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, see :class:`ModelSQDTBase`.
            quantum_defects: The expansion coefficients [d0, d2, d4, ...] of the quantum defect,
                see :meth:`calc_quantum_defect`.
            reference: Reference for the model, e.g., a publication doi where the quantum defects are given.
            n_range: Range (n_min, n_max) of principal quantum numbers n (both inclusive) for which the model is valid,
                see :class:`ModelSQDTBase`. Default is (1, inf), i.e. the model is valid for all n.

        """
        super().__init__(species, channel, reference, n_range)
        check_expansion_coefficients(quantum_defects, f"{self.full_name}: quantum_defects")
        self.quantum_defects: tuple[float, ...] = tuple(quantum_defects)
        """The expansion coefficients (d0, d2, d4, ...) of the quantum defect, see :meth:`calc_quantum_defect`.
        They are stored as a tuple, since the models are shared, see :class:`~rydstate.qdt.ModelSQDTFit`."""

    def to_dict(self) -> ModelSQDTDict:
        """Return a JSON serializable dictionary describing the model.

        The model can be recreated from the dictionary via :meth:`from_dict`, e.g. to store a model in a json file:

        .. code-block:: python

            with open("model.json", "w") as file:
                json.dump(model.to_dict(), file, indent=2)
            with open("model.json") as file:
                model = ModelSQDT.from_dict(json.load(file))

        See :meth:`ModelSQDTBase.to_dict` for the keys common to all SQDT models.

        """
        return {**super().to_dict(), "quantum_defects": list(self.quantum_defects)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], directory: str | PathLike[str] | None = None) -> Self:  # noqa: ARG003
        """Create a model from a dictionary, as returned by :meth:`to_dict`.

        Args:
            data: The dictionary describing the model, see :class:`ModelSQDTDict`.
                Its "model_class" must be "ModelSQDT".
                The keys "reference", "n_range" and "comment" are optional, where "comment" is ignored.
            directory: Not used, since the dictionary contains no file paths, see :meth:`~rydstate.qdt.Model.from_dict`.

        """
        kwargs = cls._parse_dict(data, ModelSQDTDict)
        return cls(quantum_defects=list(data["quantum_defects"]), **kwargs)

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
