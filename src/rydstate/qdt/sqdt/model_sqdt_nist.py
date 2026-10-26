from __future__ import annotations

import math
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rydstate.element_properties import get_element_properties
from rydstate.qdt.sqdt.model_sqdt import ModelSQDTBase, ModelSQDTBaseDict
from rydstate.qdt.sqdt.nist import parse_nist_energy_levels
from rydstate.utils import calc_nu_from_energy

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from typing_extensions import Self

    from rydstate.angular.angular_ket import AngularKetBase


class ModelSQDTNistDict(ModelSQDTBaseDict):
    """JSON serializable description of a :class:`ModelSQDTNist`, see :meth:`ModelSQDTNist.to_dict`."""

    nist_data_file: str
    """Path to the NIST data file with the energy levels, relative paths are relative to the json file."""


@cache
def _load_nist_energies_by_channel(species: str, file: Path) -> dict[tuple[int, float, float], dict[int, float]]:
    """Parse a NIST data file and group its energy levels by channel.

    The result is cached, so that each NIST data file is only parsed once per species.

    Args:
        species: The species of the NIST data file.
        file: The (resolved) path to the NIST data file, see :func:`~rydstate.qdt.sqdt.nist.parse_nist_energy_levels`.

    Returns:
        The energies of the levels in atomic units (Hartree) keyed by n, for each channel (l_r, s_tot, j_tot).

    """
    levels = parse_nist_energy_levels(file, get_element_properties(species))
    energies_by_channel: dict[tuple[int, float, float], dict[int, float]] = {}
    for (n, l_r, j_tot, s_tot), energy in sorted(levels.items()):
        energies_by_channel.setdefault((l_r, s_tot, j_tot), {})[n] = energy
    return energies_by_channel


class ModelSQDTNist(ModelSQDTBase):
    r"""SQDT model describing a single channel of a species by its measured (NIST) energy levels.

    Instead of quantum defects, the model holds the energies of the levels of the channel for some n.
    The effective principal quantum number nui of such a level is calculated from its energy :math:`E`
    and the ionization threshold :math:`I_i` of the channel (see :attr:`ionization_threshold_au`) via

    .. math::
        \nu_i = Z \sqrt{\frac{1}{2} \frac{\mu/m_e}{-(E - I_i)/E_H}}

    see :func:`~rydstate.utils.calc_nu_from_energy`.
    For n without a level (also within the :attr:`n_range` of the model) :meth:`calc_nui` raises an error.

    Usually the models are not constructed directly, but loaded from a NIST data file via :meth:`from_nist_file`
    (or from a json file describing the model, see :meth:`to_dict`).
    """

    _model_class = "ModelSQDTNist"

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        energies: dict[int, float],
        *,
        reference: str | tuple[str, ...] | None = None,
        n_range: tuple[int, float | Literal["inf"]] = (1, math.inf),
        nist_data_file: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, see :class:`~rydstate.qdt.ModelSQDTBase`.
            energies: The energies of the levels of the channel in atomic units (Hartree), keyed by n.
                The energies are given with respect to the same zero as the ionization thresholds,
                i.e. the ground state of the species.
            reference: Reference for the energy levels. Default is None.
            n_range: Range (n_min, n_max) of principal quantum numbers n (both inclusive) for which the model is valid,
                see :class:`~rydstate.qdt.ModelSQDTBase`. Default is (1, inf).
            nist_data_file: The NIST data file the energies are taken from (only needed for :meth:`to_dict`).
                Default is None, i.e. the energies are not taken from a file.

        """
        super().__init__(species, channel, reference, n_range)
        self.energies = energies
        self.nist_data_file = nist_data_file

    @classmethod
    def from_nist_file(
        cls,
        species: str,
        channel: AngularKetBase[Any] | str,
        file: str | PathLike[str],
        *,
        reference: str | tuple[str, ...] | None = None,
        n_range: tuple[int, float | Literal["inf"]] = (1, math.inf),
    ) -> Self:
        """Load the model of the given channel with the energy levels of a NIST data file.

        The file is parsed by :func:`~rydstate.qdt.sqdt.nist.parse_nist_energy_levels`,
        i.e. only the single valence electron levels are used.
        Each file is only parsed once per species, so loading the models of several channels is cheap.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, see :class:`~rydstate.qdt.ModelSQDTBase`.
            file: The path to the NIST data file.
            reference: Reference for the energy levels. Default is None.
            n_range: Range (n_min, n_max) of principal quantum numbers n (both inclusive) for which the model is valid.

        """
        model = cls(species, channel, {}, reference=reference, n_range=n_range, nist_data_file=file)
        key = (model.channel.l_r, model.channel.s_tot, model.channel.j_tot)
        energies_by_channel = _load_nist_energies_by_channel(species, Path(file).resolve())
        if key not in energies_by_channel:
            raise ValueError(f"{model.full_name}: the NIST data file {file} has no energy levels of this channel.")
        model.energies = dict(energies_by_channel[key])
        return model

    def to_dict(self) -> ModelSQDTNistDict:
        """Return a JSON serializable dictionary describing the model.

        The model can be recreated from the dictionary via :meth:`from_dict`.
        Instead of the energies, the dictionary contains the NIST data file they are loaded from,
        so this is only possible for models loaded via :meth:`from_nist_file`.
        See :meth:`~rydstate.qdt.ModelSQDTBase.to_dict` for the keys common to all SQDT models.

        """
        if self.nist_data_file is None:
            raise ValueError(f"{self.full_name}: only models loaded from a NIST data file can be converted to a dict.")
        return {**super().to_dict(), "nist_data_file": Path(self.nist_data_file).as_posix()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], directory: str | PathLike[str] | None = None) -> Self:
        """Create a model from a dictionary, as returned by :meth:`to_dict`.

        Args:
            data: The dictionary describing the model, see :class:`ModelSQDTNistDict`.
                Its "model_class" must be "ModelSQDTNist".
                The keys "reference", "n_range" and "comment" are optional, where "comment" is ignored.
            directory: The directory relative to which a relative "nist_data_file" is resolved,
                e.g. the directory of the json file. Default is None, i.e. the current working directory.

        """
        kwargs = cls._parse_dict(data, ModelSQDTNistDict)
        model = cls.from_nist_file(file=Path(directory or ".") / data["nist_data_file"], **kwargs)
        model.nist_data_file = data["nist_data_file"]  # as given, i.e. relative to the directory
        return model

    def calc_nui(self, n: int) -> float:
        """Calculate the effective principal quantum number nui of the level with the given n from its energy.

        Args:
            n: The principal quantum number of the Rydberg state.

        Returns:
            The effective principal quantum number nui.

        Raises:
            ValueError: If the channel has no level with this n.

        """
        if n not in self.energies:
            raise ValueError(f"{self.full_name}: no energy level for n={n}.")
        return calc_nu_from_energy(
            self.element_properties.reduced_mass_au,
            self.energies[n] - self.ionization_threshold_au,
            self.element_properties.net_charge,
        )
