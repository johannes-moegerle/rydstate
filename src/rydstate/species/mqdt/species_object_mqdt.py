from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, overload

from rydstate.angular.core_ket_base import CoreKet, CoreKetDummy
from rydstate.angular.utils import Unknown
from rydstate.species.mqdt.fmodel import FModel, FModelSQDT
from rydstate.species.species_object import SpeciesObject
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetFJ
    from rydstate.units import PintFloat


class SpeciesObjectMQDT(SpeciesObject):
    i_c: ClassVar[float]

    _ionization_threshold_dict: dict[CoreKet, tuple[float, float | None, str]]
    """Dictionary containing the ionization thresholds for the different core states.
    The thresholds are given in the form of a tuple (ionization_threshold, uncertainty, unit).
    """

    core_ground_state: CoreKet
    """The ground state configuration of the atomic core."""

    nuclear_dipole: float
    """Nuclear dipole moment of the species."""

    @cached_property
    def models(self) -> list[FModel]:
        """List of MQDT models available for the species."""
        models = [model for model in FModel.__subclasses__() if getattr(model, "species_name", None) == self.name]
        return [model() for model in models]

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_threshold(self, core_ket: CoreKet, unit: str) -> float: ...

    def get_ionization_threshold(self, core_ket: CoreKet, unit: str | None = "hartree") -> PintFloat | float:
        """Return the ionization energy of the channel given by the core_ket in the desired unit.

        Args:
            core_ket: The core ket for which to return the ionization energy.
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        if core_ket not in self._ionization_threshold_dict:
            if isinstance(core_ket, CoreKetDummy):
                raise ValueError(f"Core ket {core_ket} is a dummy state. Ionization energy is not defined.")
            core_ket = CoreKet(i_c=self.i_c, s_c=core_ket.s_c, l_c=core_ket.l_c, j_c=core_ket.j_c, f_c=Unknown)
            if core_ket not in self._ionization_threshold_dict:
                core_ket = CoreKet(i_c=self.i_c, s_c=core_ket.s_c, l_c=core_ket.l_c, j_c=Unknown, f_c=Unknown)
        if core_ket not in self._ionization_threshold_dict:
            raise ValueError(f"Ionization energy for core ket {core_ket} is not defined.")

        ionization_energy_tuple = self._ionization_threshold_dict[core_ket]
        ionization_energy: PintFloat = ureg.Quantity(ionization_energy_tuple[0], ionization_energy_tuple[2])
        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    @cached_property
    def reference_ionization_energy_au(self) -> float:
        """Ionization energy in atomic units (Hartree)."""
        return self.get_ionization_threshold(self.core_ground_state, unit="hartree")

    def get_mqdt_models(self, outer_channel: AngularKetFJ) -> list[FModel]:
        """Return a list of MQDT models for the outer_channel."""
        models = [model for model in self.models if any(ket == outer_channel for ket in model.outer_channels)]
        if len(models) == 0:
            models = [FModelSQDT(self.name, outer_channel)]
        return models
