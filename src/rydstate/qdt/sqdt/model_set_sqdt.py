from __future__ import annotations

import json
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set
from rydstate.qdt.model_set import ModelSet
from rydstate.qdt.sqdt.model_sqdt import ModelSQDT
from rydstate.qdt.sqdt.model_sqdt_nist import ModelSQDTNist

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.sqdt.model_sqdt import ModelSQDTBase


class ModelSetSQDT(ModelSet):
    """SQDT model set of a species, i.e. the SQDT models of its channels and its low-lying NIST energy levels.

    The model sets of the species are instances of this class, which are registered via
    :func:`register_model_set_sqdt` and looked up via :func:`get_sqdt`, e.g.

    .. code-block:: python

        model_set = ModelSetSQDT("Rb", models=directory / "sqdt_models", nist_data_file=directory / "nist_data.txt")
        register_model_set_sqdt(model_set)

    The models and the NIST energy levels are only loaded when they are needed for the first time,
    so that creating the model sets of all species (e.g. when importing rydstate) is cheap.

    Each channel can be described by several :class:`~rydstate.qdt.ModelSQDTBase` models, whose nu ranges may overlap:
    the :attr:`nist_models` with the measured energy levels (valid in the :attr:`nist_nu_range`)
    and the quantum defect :attr:`models` (usually valid for all nu).
    The models are tried in the order given by :meth:`get_models`, i.e. the NIST models first,
    and the first model that has data for n and whose nu range contains the resulting nu is used.
    Since a NIST model only has data for the n with a measured level, the quantum defects are used
    for all other states, also inside the :attr:`nist_nu_range`.
    """

    def __init__(
        self,
        species: str,
        models: Sequence[ModelSQDT] | str | PathLike[str] | None = None,
        nist_data_file: str | PathLike[str] | None = None,
        nist_nu_range: tuple[float, float] = (0, 15),
    ) -> None:
        """Initialize the SQDT model set.

        Args:
            species: The short name of the atomic species.
            models: The :class:`~rydstate.qdt.ModelSQDT` models of the species,
                either directly as a sequence of models or as the directory with their json files
                (which are loaded on first use via :meth:`load_models`).
                Default is None, i.e. no quantum defects are available for this species,
                in which case only the NIST energy levels can be used.
            nist_data_file: The file with the low-lying NIST energy levels (see
                :func:`~rydstate.qdt.sqdt.nist.parse_nist_energy_levels`), which is parsed on first use.
                Default is None, i.e. no NIST energy levels are available for this species.
                The levels are loaded as :class:`~rydstate.qdt.ModelSQDTNist` models, see :attr:`nist_models`.
            nist_nu_range: Range of effective principal quantum numbers nu for which the NIST energy levels are used.
                Default is (0, 15).

        """
        self.species = species
        self._models_source = models
        self._nist_data_file = nist_data_file
        self.nist_nu_range = nist_nu_range
        self._trivial_models: dict[AngularKetBase[Any], ModelSQDT] = {}

    def __repr__(self) -> str:
        return f"ModelSetSQDT({self.species})"

    @cached_property
    def models(self) -> list[ModelSQDT] | None:
        """List of the :class:`~rydstate.qdt.ModelSQDT` models (loaded on first access).

        Each model describes the quantum defects of one channel (in its range of validity).
        Channels without such a model have zero quantum defects, see :meth:`get_model`.
        None means that no quantum defects are available for this species at all.
        """
        source = self._models_source
        if source is None:
            return None
        models = self.load_models(source) if isinstance(source, (str, PathLike)) else list(source)
        for model in models:
            if model.species != self.species:
                raise ValueError(f"The model {model} is not a model of the species {self.species}.")
        return models

    @classmethod
    def load_models(cls, directory: str | PathLike[str]) -> list[ModelSQDT]:
        """Load all models from the json files in the given directory (sorted by the file names).

        Each json file describes one model, see :meth:`~rydstate.qdt.ModelSQDT.to_dict`.

        Args:
            directory: The directory with the json files of the models.

        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"The directory {directory} with the SQDT models does not exist.")
        models = []
        for file in sorted(directory.glob("*.json")):
            with file.open(encoding="utf-8") as f:
                models.append(ModelSQDT.from_dict(json.load(f)))
        return models

    @cached_property
    def nist_models(self) -> list[ModelSQDTNist]:
        """List of the :class:`~rydstate.qdt.ModelSQDTNist` models (loaded on first access).

        Each model holds the NIST energy levels of one channel and is valid in the :attr:`nist_nu_range`.
        Empty if no NIST data file is given.
        """
        if self._nist_data_file is None:
            return []
        return ModelSQDTNist.from_nist_file(self.species, Path(self._nist_data_file), self.nist_nu_range)

    def get_models(self, angular_ket: AngularKetBase[Any]) -> list[ModelSQDTBase]:
        """Return all models whose channel is the given angular ket (up to the magnetic quantum number m).

        The :attr:`nist_models` come first, so that they take precedence over the quantum defect :attr:`models`
        in their range of validity.

        Args:
            angular_ket: The angular ket of the channel, which must be in LS coupling.

        """
        if angular_ket.coupling_scheme != "LS":
            raise ValueError(f"The angular ket {angular_ket!r} must be in LS coupling.")
        channel = angular_ket if is_not_set(angular_ket.m) else angular_ket.replace_m(NotSet)
        return [model for model in [*self.nist_models, *(self.models or [])] if model.channel == channel]

    def get_model(self, n: int, angular_ket: AngularKetBase[Any]) -> ModelSQDTBase:
        """Return the model describing the state with principal quantum number n and the given angular ket.

        This is the first of the models of the channel (see :meth:`get_models`),
        which has data for n and whose nu range contains the resulting nui of the state.
        I.e. for low lying states this is usually a :class:`~rydstate.qdt.ModelSQDTNist` model
        with the NIST energy levels, otherwise a :class:`~rydstate.qdt.ModelSQDT` model with the quantum defects.
        If there is no quantum defect model for the channel, a model with zero quantum defects is returned.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket of the Rydberg state,
                it is converted to LS coupling if it is a single LS ket in another coupling scheme.

        """
        return self._get_model_and_nui(n, angular_ket)[0]

    def _get_model_and_nui(self, n: int, angular_ket: AngularKetBase[Any]) -> tuple[ModelSQDTBase, float]:
        """Return the model describing the state (see :meth:`get_model`) and the nui of the state."""
        angular_ket = self._to_ls(angular_ket)
        if angular_ket.contains_unknown:
            raise ValueError(f"Cannot calculate nui for unknown quantum numbers of {angular_ket!r}.")

        models = self.get_models(angular_ket)
        for model in models:
            if isinstance(model, ModelSQDTNist) and n not in model.energies:
                continue
            nui = model.calc_nui(n)
            if model.nu_min <= nui <= model.nu_max:
                return model, nui

        if not any(isinstance(model, ModelSQDT) for model in models):
            if self.models is None:
                raise ValueError(
                    f"No quantum defect data available for species {self.species} "
                    f"and no NIST energy level of the channel {angular_ket!r} for n={n}."
                )
            return self._get_trivial_model(angular_ket), float(n)
        raise ValueError(
            f"None of the SQDT models {models} of the channel {angular_ket!r} is valid for n={n}, "
            f"their nu ranges are {[model.nu_range for model in models]}."
        )

    def _get_trivial_model(self, angular_ket: AngularKetBase[Any]) -> ModelSQDT:
        """Return a model with zero quantum defects for the channel of the given LS angular ket."""
        channel = angular_ket if is_not_set(angular_ket.m) else angular_ket.replace_m(NotSet)
        if channel not in self._trivial_models:
            self._trivial_models[channel] = ModelSQDT(self.species, channel, [0.0])
        return self._trivial_models[channel]

    @staticmethod
    def _to_ls(angular_ket: AngularKetBase[Any]) -> AngularKetBase[Any]:
        """Convert the angular ket to LS coupling, if it is a single LS ket in another coupling scheme."""
        if angular_ket.coupling_scheme == "LS":
            return angular_ket
        angular_state = angular_ket.to_state("LS")
        if len(angular_state.kets) != 1:
            raise NotImplementedError("calc_nui is only implemented for AngularKetLS.")
        return angular_state.kets[0]

    def calc_nui(self, n: int, angular_ket: AngularKetBase[Any]) -> float:
        r"""Calculate the effective principal quantum number nui of a Rydberg state with the given n and angular ket.

        The effective principal quantum number nui is defined with reference to the ionization threshold
        of the channel (see :attr:`~rydstate.qdt.Model.ionization_thresholds_au` of the :meth:`get_model`),
        i.e. it describes the binding energy of the Rydberg electron
        :math:`E = I_i - \frac{Z^2 R_M}{\nu_i^2}`
        with the mass corrected Rydberg constant :math:`R_M = R_\infty \mu/m_e`.

        nui is calculated by the model returned by :meth:`get_model`.
        I.e. for low lying states nui is usually calculated from the NIST energy levels
        (see :class:`~rydstate.qdt.ModelSQDTNist`), otherwise via the quantum defect theory,
        where nui is defined as :math:`\nu_i = n^* = n - \delta_{lj}(n)`
        with the quantum defect :math:`\delta_{lj}(n)` (see :class:`~rydstate.qdt.ModelSQDT`).
        If there is no quantum defect model for the channel, the quantum defect is zero.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket specifying l, j_tot, and s_tot of the Rydberg state.

        """
        return self._get_model_and_nui(n, angular_ket)[1]


_SQDT_MODEL_SETS: dict[str, ModelSetSQDT] = {}
"""The registered SQDT model sets by their species, see :func:`register_model_set_sqdt`."""


def register_model_set_sqdt(model_set: ModelSetSQDT) -> ModelSetSQDT:
    """Register the SQDT model set of a species, so that it is returned by :func:`get_sqdt`.

    The species data register their model sets when they are imported, e.g.

    .. code-block:: python

        model_set_sqdt_rubidium = ModelSetSQDT("Rb", models=..., nist_data_file=...)
        register_model_set_sqdt(model_set_sqdt_rubidium)

    Args:
        model_set: The model set to register. There can only be one registered model set per species.

    Returns:
        The registered model set.

    """
    if model_set.species in _SQDT_MODEL_SETS:
        raise ValueError(f"An SQDT model set for the species {model_set.species} is already registered.")
    _SQDT_MODEL_SETS[model_set.species] = model_set
    return model_set


def get_sqdt(species: str) -> ModelSetSQDT:
    """Get the registered SQDT model set of the given species, see :func:`register_model_set_sqdt`.

    The same instance is returned for every call, since the states only store their models
    and look up the model set when needed.
    """
    if species not in _SQDT_MODEL_SETS:
        raise ValueError(f"No ModelSetSQDT registered for {species=}.")
    return _SQDT_MODEL_SETS[species]
