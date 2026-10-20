from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set, is_unknown
from rydstate.qdt.model_set import ModelSet
from rydstate.qdt.sqdt.model_sqdt import ModelSQDT
from rydstate.qdt.sqdt.nist import parse_nist_energy_levels, resolve_species_data_file
from rydstate.utils import calc_nu_from_energy, get_all_subclasses

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.sqdt.nist import NistEnergyLevels


logger = logging.getLogger(__name__)


class ModelSetSQDT(ModelSet):
    """Base class for the SQDT model sets of the different species."""

    nist_data_file: ClassVar[str | None] = None
    """Name of the file (in the species directory) with the low-lying NIST energy levels,
    usually ``"nist_data.txt"``. Left as None for species without a NIST data file."""

    models: ClassVar[list[ModelSQDT] | None] = None
    """List of the :class:`~rydstate.qdt.ModelSQDT` models, each describing the quantum defects of one channel
    (in its range of validity). Channels without a model have zero quantum defects, see :meth:`get_model`.
    The default None means that no quantum defects are available for this species at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self._trivial_models: dict[AngularKetBase[Any], ModelSQDT] = {}

        self._nist_energy_levels: NistEnergyLevels = {}
        if self.nist_data_file is not None:
            # Load the NIST energy levels if a NIST data file is specified
            file = resolve_species_data_file(type(self), self.nist_data_file)
            self._nist_energy_levels = parse_nist_energy_levels(file, self.element_properties)

    def __repr__(self) -> str:
        return f"ModelSetSQDT({self.species})"

    def get_models(self, angular_ket: AngularKetBase[Any]) -> list[ModelSQDT]:
        """Return all models whose channel is the given angular ket (up to the magnetic quantum number m).

        Args:
            angular_ket: The angular ket of the channel, which must be in LS coupling.

        """
        if angular_ket.coupling_scheme != "LS":
            raise ValueError(f"The angular ket {angular_ket!r} must be in LS coupling.")
        channel = angular_ket if is_not_set(angular_ket.m) else angular_ket.replace_m(NotSet)
        return [model for model in self.models or [] if model.channel == channel]

    def get_model(
        self, n: int, angular_ket: AngularKetBase[Any], *, use_nist_data: bool = True, nist_n_max: int = 15
    ) -> ModelSQDT:
        """Return the model describing the state with principal quantum number n and the given angular ket.

        This is the model of the channel (see :meth:`get_models`), whose nu range contains the nui of the state.
        As in :meth:`calc_nui`, the nui of the state is taken from the NIST energy levels if available,
        and otherwise calculated by the model itself.
        If there is no model for the channel, a model with zero quantum defects is returned.

        Args:
            n: The principal quantum number of the Rydberg state.
            angular_ket: The angular ket of the Rydberg state,
                it is converted to LS coupling if it is a single LS ket in another coupling scheme.
            use_nist_data: Whether to use NIST energy data, see :meth:`calc_nui`.
            nist_n_max: Maximum principal quantum number for which to use the NIST energy data, see :meth:`calc_nui`.

        """
        angular_ket = self._to_ls(angular_ket)
        models = self.get_models(angular_ket)
        if len(models) == 0:
            return self._get_trivial_model(angular_ket)

        nui_nist = self._calc_nist_nui(n, angular_ket, nist_n_max) if use_nist_data else None
        for model in models:
            nui = nui_nist if nui_nist is not None else model.calc_nui(n)
            if model.nu_min <= nui <= model.nu_max:
                return model
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
        whose channel is the angular ket (see :meth:`get_models`) and whose nu range contains nui
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

        return self.get_model(n, angular_ket, use_nist_data=False).calc_nui(n)

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
        # all models of a channel share its ionization threshold, so the nu range does not matter here
        models = self.get_models(angular_ket) or [self._get_trivial_model(angular_ket)]
        energy_au -= models[0].ionization_threshold_au
        return calc_nu_from_energy(
            self.element_properties.reduced_mass_au, energy_au, self.element_properties.net_charge
        )


def get_sqdt(species: str) -> ModelSetSQDT:
    """Get an instance of the subclass of ModelSetSQDT for the given species."""
    subclasses = get_all_subclasses(ModelSetSQDT, species)
    if len(subclasses) == 0:
        raise ValueError(f"No subclass of ModelSetSQDT found for {species=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of ModelSetSQDT found for {species=}: {subclasses}.")
