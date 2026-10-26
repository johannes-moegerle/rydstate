from __future__ import annotations

import itertools
import json
import logging
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set, is_unknown
from rydstate.qdt.model_set import ModelSet
from rydstate.qdt.sqdt.model_sqdt import ModelSQDT
from rydstate.qdt.sqdt.nist import parse_nist_energy_levels
from rydstate.utils import calc_nu_from_energy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.sqdt.nist import NistEnergyLevels


logger = logging.getLogger(__name__)


class ModelSetSQDT(ModelSet):
    """SQDT model set of a species, i.e. the SQDT models of its channels and its low-lying NIST energy levels.

    The model sets of the species are instances of this class, which are registered via
    :func:`register_model_set_sqdt` and looked up via :func:`get_sqdt`, e.g.

    .. code-block:: python

        model_set = ModelSetSQDT("Rb", models=directory / "sqdt_models", nist_data_file=directory / "nist_data.txt")
        register_model_set_sqdt(model_set)

    The models and the NIST energy levels are only loaded when they are needed for the first time,
    so that creating the model sets of all species (e.g. when importing rydstate) is cheap.
    """

    def __init__(
        self,
        species: str,
        models: Sequence[ModelSQDT] | str | PathLike[str] | None = None,
        nist_data_file: str | PathLike[str] | None = None,
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

        """
        self.species = species
        self._models_source = models
        self._nist_data_file = nist_data_file
        self._trivial_models: dict[AngularKetBase[Any], ModelSQDT] = {}

    def __repr__(self) -> str:
        return f"ModelSetSQDT({self.species})"

    @cached_property
    def models(self) -> list[ModelSQDT] | None:
        """List of the :class:`~rydstate.qdt.ModelSQDT` models (loaded on first access).

        Each model describes the quantum defects of one channel (in its n range).
        The n ranges of the models of the same channel must not overlap.
        Channels without a model have zero quantum defects, see :meth:`get_model`.
        None means that no quantum defects are available for this species at all.
        """
        source = self._models_source
        if source is None:
            return None
        models = self.load_models(source) if isinstance(source, (str, PathLike)) else list(source)
        for model in models:
            if model.species != self.species:
                raise ValueError(f"The model {model} is not a model of the species {self.species}.")
        self._check_n_ranges_do_not_overlap(models)
        return models

    @staticmethod
    def _check_n_ranges_do_not_overlap(models: Sequence[ModelSQDT]) -> None:
        """Raise an error if the (inclusive) n ranges of two models of the same channel overlap."""
        models_by_channel: dict[AngularKetBase[Any], list[ModelSQDT]] = {}
        for model in models:
            models_by_channel.setdefault(model.channel, []).append(model)
        for channel_models in models_by_channel.values():
            sorted_models = sorted(channel_models, key=lambda model: model.n_min)
            for lower, upper in itertools.pairwise(sorted_models):
                if lower.n_max >= upper.n_min:
                    raise ValueError(
                        f"The n ranges {lower.n_range} of {lower!r} and {upper.n_range} of {upper!r} overlap."
                    )

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
    def _nist_energy_levels(self) -> NistEnergyLevels:
        """The low-lying NIST energy levels, empty if no NIST data file is given."""
        if self._nist_data_file is None:
            return {}
        return parse_nist_energy_levels(Path(self._nist_data_file), self.element_properties)

    def get_models(self, angular_ket: AngularKetBase[Any]) -> list[ModelSQDT]:
        """Return all models whose channel is the given angular ket (up to the magnetic quantum number m).

        The models are sorted by their n range.

        Args:
            angular_ket: The angular ket of the channel, which must be in LS coupling.

        """
        if angular_ket.coupling_scheme != "LS":
            raise ValueError(f"The angular ket {angular_ket!r} must be in LS coupling.")
        channel = angular_ket if is_not_set(angular_ket.m) else angular_ket.replace_m(NotSet)
        models = [model for model in self.models or [] if model.channel == channel]
        return sorted(models, key=lambda model: model.n_min)

    def get_model(self, n: int, angular_ket: AngularKetBase[Any]) -> ModelSQDT:
        """Return the model describing the state with principal quantum number n and the given angular ket.

        This is the model of the channel (see :meth:`get_models`), whose n range contains n.
        If there is no model for the channel, a model with zero quantum defects is returned.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket of the Rydberg state,
                it is converted to LS coupling if it is a single LS ket in another coupling scheme.

        """
        angular_ket = self._to_ls(angular_ket)
        models = self.get_models(angular_ket)
        if len(models) == 0:
            return self._get_trivial_model(angular_ket)
        for model in models:
            if model.n_min <= n <= model.n_max:
                return model
        raise ValueError(
            f"None of the SQDT models {models} of the channel {angular_ket!r} is valid for n={n}, "
            f"their n ranges are {[model.n_range for model in models]}."
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

    def calc_nui(
        self,
        n: int,
        angular_ket: AngularKetBase[Any],
        *,
        use_nist_data: bool = True,
        nist_n_max: int = 15,
    ) -> float:
        r"""Calculate the effective principal quantum number nui of a Rydberg state with the given n and angular ket.

        The effective principal quantum number nui is defined with reference to the ionization threshold
        of the channel (see :attr:`~rydstate.qdt.Model.ionization_thresholds_au` of the :meth:`get_model`),
        i.e. it describes the binding energy of the Rydberg electron
        :math:`E = I_i - \frac{Z^2 R_M}{\nu_i^2}`
        with the mass corrected Rydberg constant :math:`R_M = R_\infty \mu/m_e`.

        I.e. either look up the energy for low lying states in the nist data (if use_nist_data is True),
        and calculate nui from the energy via (see also `calc_nu_from_energy`):

        .. math::
            \nu_i = Z \sqrt{\frac{1}{2} \frac{\mu/m_e}{-(E - I)/E_H}}

        Or calculate nui via the quantum defect theory,
        where nui is defined as :math:`\nu_i = n^* = n - \delta_{lj}(n)`
        with the quantum defect :math:`\delta_{lj}(n)` given by the :class:`~rydstate.qdt.ModelSQDT` model
        whose channel is the angular ket (see :meth:`get_models`) and whose n range contains n
        (see :meth:`~rydstate.qdt.ModelSQDT.calc_quantum_defect`).
        If there is no model for the channel, the quantum defect is zero.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket specifying l, j_tot, and s_tot of the Rydberg state.
            use_nist_data: Whether to use NIST energy data.
                Default is True.
            nist_n_max: Maximum principal quantum number for which to use the NIST energy data.
                Default is 15.

        """
        angular_ket = self._to_ls(angular_ket)

        if use_nist_data:
            nui_nist = self._calc_nist_nui(n, angular_ket, nist_n_max)
            if nui_nist is not None:
                return nui_nist

        if self.models is None:
            raise ValueError(f"No quantum defect data available for species {self.species}.")

        return self.get_model(n, angular_ket).calc_nui(n)

    def _calc_nist_nui(self, n: int, angular_ket: AngularKetBase[Any], nist_n_max: int) -> float | None:
        """Return the nui of the state from the NIST energy levels, or None if they do not contain the state.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket of the Rydberg state in LS coupling.
            nist_n_max: Maximum principal quantum number for which to use the NIST energy data.

        """
        l_r = angular_ket.l_r
        j_tot = angular_ket.get_qn("j_tot", allow_unknown=True)
        s_tot = angular_ket.get_qn("s_tot", allow_unknown=True)
        if is_unknown(j_tot) or is_unknown(s_tot):
            raise ValueError(f"Cannot calculate nui for unknown j_tot or s_tot of {angular_ket!r}.")

        if n > nist_n_max or (n, l_r, j_tot, s_tot) not in self._nist_energy_levels:
            if n <= nist_n_max:
                logger.debug(
                    "NIST energy levels for (n=%d, l_r=%d, j_tot=%s, s_tot=%s) not found, using quantum defect theory.",
                    *(n, l_r, j_tot, s_tot),
                )
            return None

        energy_au = self._nist_energy_levels[(n, l_r, j_tot, s_tot)]
        # all models of a channel share its ionization threshold, so the n range does not matter here
        models = self.get_models(angular_ket) or [self._get_trivial_model(angular_ket)]
        energy_au -= models[0].ionization_threshold_au
        return calc_nu_from_energy(
            self.element_properties.reduced_mass_au, energy_au, self.element_properties.net_charge
        )


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
