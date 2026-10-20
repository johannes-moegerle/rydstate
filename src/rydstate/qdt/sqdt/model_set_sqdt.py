from __future__ import annotations

import logging
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

from rydstate import units
from rydstate.angular import NotSet
from rydstate.angular.utils import is_not_set, is_unknown
from rydstate.element_properties import get_element_properties
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.qdt.sqdt.nist import parse_nist_energy_levels, resolve_species_data_file
from rydstate.utils import calc_nu_from_energy, get_all_subclasses

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.sqdt.model_sqdt import ModelSQDT
    from rydstate.qdt.sqdt.nist import NistEnergyLevels
    from rydstate.units import PintFloat


logger = logging.getLogger(__name__)


class ModelSetSQDT(ABC, metaclass=CachedABCMeta):
    """Base class for the SQDT model sets of the different species."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str | None] = None
    """The tag for these SQDT parameters."""
    is_default: ClassVar[bool] = False
    """Whether this is the default SQDT model set for the species."""
    nist_data_file: ClassVar[str | None] = None
    """Name of the file (in the species directory) with the low-lying NIST energy levels,
    usually ``"nist_data.txt"``. Left as None for species without a NIST data file."""

    ionization_energy: ClassVar[tuple[float, str]]
    """Ionization energy: (value, unit)."""
    _reference_ionization_energy: ClassVar[tuple[float, str] | None] = None
    """Reference ionization energy: (value, unit), in reference to which nu is defined.
    If None, the ionization_energy is used as the reference and nu = nui."""

    models: ClassVar[list[ModelSQDT] | None] = None
    """List of the :class:`~rydstate.qdt.ModelSQDT` models, each describing the quantum defects of one channel
    (in its range of validity). Channels without a model have zero quantum defects.
    The default None means that no quantum defects are available for this species at all.
    """

    def __init__(self) -> None:
        self.element_properties = get_element_properties(self.species)

        self._nist_energy_levels: NistEnergyLevels = {}
        if self.nist_data_file is not None:
            # Load the NIST energy levels if a NIST data file is specified
            file = resolve_species_data_file(type(self), self.nist_data_file)
            self._nist_energy_levels = parse_nist_energy_levels(file, self.element_properties)

    def __repr__(self) -> str:
        return f"ModelSetSQDT({self.species}, {self.tag})"

    @overload
    def get_ionization_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_energy(self, unit: str) -> float: ...

    def get_ionization_energy(self, unit: str | None = None) -> PintFloat | float:
        """Return the ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is None (returns a Pint quantity).

        Returns:
            Ionization energy in the desired unit.

        """
        ionization_energy_au = units.user_to_au(self.ionization_energy[0], self.ionization_energy[1], "energy")
        return units.au_to_user(ionization_energy_au, "energy", unit)

    @cached_property
    def ionization_energy_au(self) -> float:
        """Ionization energy in atomic units (Hartree)."""
        return self.get_ionization_energy("a.u.")

    @cached_property
    def reference_ionization_energy_au(self) -> float:
        r"""Reference ionization energy in atomic units (Hartree).

        The reference ionization energy defines nu via

        .. math::
            E = I_i - \frac{Z^2 R_M}{\nu_i^2}
              = I_{\text{ref}} - \frac{Z^2 R_M}{\nu^2}

        If no :attr:`_reference_ionization_energy` is defined (as it is the case e.g. for all alkali atoms),
        the :attr:`ionization_energy` is used as the reference and nu = nui.
        """
        if self._reference_ionization_energy is None:
            return self.ionization_energy_au
        value, unit = self._reference_ionization_energy
        return units.user_to_au(value, unit, "energy")

    def get_models(self, angular_ket: AngularKetBase[Any]) -> list[ModelSQDT]:
        """Return all models whose channel is the given angular ket (up to the magnetic quantum number m).

        Args:
            angular_ket: The angular ket of the channel, which must be in LS coupling.

        """
        if angular_ket.coupling_scheme != "LS":
            raise ValueError(f"The angular ket {angular_ket!r} must be in LS coupling.")
        channel = angular_ket if is_not_set(angular_ket.m) else angular_ket.replace_m(NotSet)
        return [model for model in self.models or [] if model.channel == channel]

    def calc_nui(
        self,
        n: int,
        angular_ket: AngularKetBase[Any],
        *,
        use_nist_data: bool = True,
        nist_n_max: int = 15,
    ) -> float:
        r"""Calculate the effective principal quantum number nui of a Rydberg state with the given n and angular ket.

        The effective principal quantum number nui is defined with reference to the
        :attr:`ionization_energy`, i.e. it describes the binding energy of the Rydberg electron
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
        if angular_ket.coupling_scheme != "LS":
            angular_state = angular_ket.to_state("LS")
            if len(angular_state.kets) == 1:
                angular_ket = angular_state.kets[0]
            else:
                raise NotImplementedError("calc_nui is only implemented for AngularKetLS.")

        l_r = angular_ket.l_r
        j_tot = angular_ket.get_qn("j_tot", allow_unknown=True)
        s_tot = angular_ket.get_qn("s_tot", allow_unknown=True)

        if is_unknown(j_tot) or is_unknown(s_tot):
            raise ValueError(f"Cannot calculate nui for unknown j_tot or s_tot of {angular_ket!r}.")

        if n <= nist_n_max and use_nist_data:  # try to use NIST data
            if (n, l_r, j_tot, s_tot) in self._nist_energy_levels:
                energy_au = self._nist_energy_levels[(n, l_r, j_tot, s_tot)]
                energy_au -= self.ionization_energy_au  # use the cached ionization energy for better performance
                return calc_nu_from_energy(
                    self.element_properties.reduced_mass_au, energy_au, self.element_properties.net_charge
                )
            logger.debug(
                "NIST energy levels for (n=%d, l_r=%d, j_tot=%s, s_tot=%s) not found, using quantum defect theory.",
                *(n, l_r, j_tot, s_tot),
            )

        if self.models is None:
            raise ValueError(f"No quantum defect data available for species {self.species}.")

        models = self.get_models(angular_ket)
        if len(models) == 0:
            return float(n)  # no model for this channel, i.e. zero quantum defect

        for model in models:
            nui = model.calc_nui(n)
            if model.nu_min <= nui <= model.nu_max:
                return nui
        raise ValueError(
            f"None of the SQDT models {models} of the channel {angular_ket!r} is valid for n={n}, "
            f"their nu ranges are {[model.nu_range for model in models]}."
        )


def get_sqdt(species: str, tag: str | None = None) -> ModelSetSQDT:
    """Get an instance of the subclass of SQDT for the given species and tag."""
    subclasses = get_all_subclasses(ModelSetSQDT, species, tag)

    if tag is None:
        subclasses = [cls for cls in subclasses if getattr(cls, "is_default", False)]

    if len(subclasses) == 0:
        raise ValueError(f"No subclass of SQDT found for {species=} and {tag=}.")
    if len(subclasses) == 1:
        return subclasses[0]()
    raise ValueError(f"Multiple subclasses of SQDT found for {species=} and {tag=}: {subclasses}.")
