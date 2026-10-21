from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rydstate.angular import AngularKetLS
from rydstate.element_properties import get_element_properties
from rydstate.qdt.sqdt.model_sqdt import ModelSQDTBase
from rydstate.qdt.sqdt.nist import parse_nist_energy_levels
from rydstate.utils import calc_nu_from_energy

if TYPE_CHECKING:
    from pathlib import Path

    from rydstate.angular.angular_ket import AngularKetBase


class ModelSQDTNist(ModelSQDTBase):
    r"""SQDT model describing a single channel of a species by its measured (NIST) energy levels.

    Instead of quantum defects, the model holds the energies of the levels of the channel for some n.
    The effective principal quantum number nui of such a level is calculated from its energy :math:`E`
    and the ionization threshold :math:`I_i` of the channel (see :attr:`ionization_threshold_au`) via

    .. math::
        \nu_i = Z \sqrt{\frac{1}{2} \frac{\mu/m_e}{-(E - I_i)/E_H}}

    see :func:`~rydstate.utils.calc_nu_from_energy`.
    For n without a level the model has no data and :meth:`calc_nui` raises an error.

    Usually the models are not constructed directly, but loaded from a NIST data file via :meth:`from_nist_file`.
    """

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        energies: dict[int, float],
        *,
        reference: str | tuple[str, ...] | None = "NIST Atomic Spectra Database",
        nu_range: tuple[float, float | Literal["inf"]] = (0, math.inf),
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, see :class:`~rydstate.qdt.ModelSQDTBase`.
            energies: The energies of the levels of the channel in atomic units (Hartree), keyed by n.
                The energies are given with respect to the same zero as the ionization thresholds,
                i.e. the ground state of the species.
            reference: Reference for the energy levels. Default is "NIST Atomic Spectra Database".
            nu_range: Range (nu_min, nu_max) of effective principal quantum numbers nu for which the model is valid,
                see :class:`~rydstate.qdt.ModelSQDTBase`. Default is (0, inf).

        """
        super().__init__(species, channel, reference, nu_range)
        self.energies = energies

    @classmethod
    def from_nist_file(
        cls,
        species: str,
        file: Path,
        nu_range: tuple[float, float | Literal["inf"]] = (0, math.inf),
    ) -> list[ModelSQDTNist]:
        """Load the models of all channels with energy levels in a NIST data file.

        The file is parsed by :func:`~rydstate.qdt.sqdt.nist.parse_nist_energy_levels`,
        i.e. only the single valence electron levels are used.
        Since the NIST levels do not resolve the hyperfine structure, a species with a nuclear spin
        gets one model (with the same energies) for each total angular momentum f_tot of a channel.

        Args:
            species: The species for which the models are defined.
            file: The path to the NIST data file.
            nu_range: Range (nu_min, nu_max) of effective principal quantum numbers nu for which the models are valid.

        Returns:
            One model for each channel, sorted by l_r, s_tot, j_tot and f_tot.

        """
        element_properties = get_element_properties(species)
        levels = parse_nist_energy_levels(file, element_properties)

        energies_by_channel: dict[tuple[int, float, float], dict[int, float]] = {}
        for (n, l_r, j_tot, s_tot), energy in sorted(levels.items()):
            energies_by_channel.setdefault((l_r, s_tot, j_tot), {})[n] = energy

        i_c = element_properties.i_c
        models: list[ModelSQDTNist] = []
        for (l_r, s_tot, j_tot), energies in sorted(energies_by_channel.items()):
            for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                channel = AngularKetLS(l_r=l_r, s_tot=s_tot, j_tot=j_tot, f_tot=float(f_tot), species=species)
                models.append(cls(species, channel, energies, nu_range=nu_range))
        return models

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
