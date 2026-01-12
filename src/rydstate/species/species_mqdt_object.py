from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from rydstate.angular.angular_core_ket import AngularCoreKet, AngularCoreKetDummy
from rydstate.angular.utils import Unknown
from rydstate.species.species_object import SpeciesObject
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.units import PintFloat

logger = logging.getLogger(__name__)


class SpeciesMQDTObject(SpeciesObject):
    """Abstract base class for species objects using MQDT and data from MQDT.jl."""

    def __init__(self) -> None:
        """Initialize an species instance."""
        self._nist_energy_levels: dict[tuple[int, int, float, float], float] = {}

        from juliacall import Main as jl  # noqa: N813, PLC0415

        jl.seval("using MQDT")

        i_c = self.i_c if self.i_c is not None else 0
        self.jl_species = jl.Symbol(self.name.replace("_mqdt", ""))
        self.jl_parameters = jl.MQDT.get_parameters(self.jl_species)

        self.ionization_thresholds_dict: dict[AngularCoreKet, float] = {}
        for key, value in self.jl_parameters.thresholds_dict.items():
            core_ket: AngularCoreKet
            if isinstance(key, str):
                core_ket = AngularCoreKetDummy(key)
            else:
                l_c = key.lc
                j_c = key.Jc if str(key.Jc) != "nan" else Unknown
                f_c = key.Fc if str(key.Fc) != "nan" else Unknown
                if i_c == 0 and f_c is Unknown:
                    f_c = j_c
                core_ket = AngularCoreKet(i_c=self.i_c, s_c=1 / 2, l_c=l_c, j_c=j_c, f_c=f_c)
            self.ionization_thresholds_dict[core_ket] = value

        self.jl_models = []
        for l in range(10):
            jtot_min = min(l, abs(l - 1))
            jtot_max = l + 1
            for f_tot in np.arange(abs(jtot_min - i_c), jtot_max + i_c + 1):
                models = jl.MQDT.get_fmodels(self.jl_species, l, float(f_tot))
                self.jl_models.extend(models)

    @overload
    def get_ionization_energy(self, angular_ket: AngularKetBase | None = None, unit: None = None) -> PintFloat: ...

    @overload
    def get_ionization_energy(self, angular_ket: AngularKetBase | None = None, *, unit: str) -> float: ...

    @overload
    def get_ionization_energy(self, angular_ket: AngularKetBase | None, unit: str) -> float: ...

    def get_ionization_energy(
        self, angular_ket: AngularKetBase | None = None, unit: str | None = "hartree"
    ) -> PintFloat | float:
        if angular_ket is None:
            raise ValueError("angular_ket must be provided to get the ionization energy for a MQDT species.")
        core_ket = angular_ket.get_core_ket()
        if core_ket not in self.ionization_thresholds_dict:
            raise ValueError(
                f"Ionic core state {core_ket} not found for {self.name} in {self.ionization_thresholds_dict=}."
            )

        ionization_energy_cm = self.ionization_thresholds_dict[core_ket]  # in cm^-1
        ionization_energy: PintFloat = ureg.Quantity(ionization_energy_cm, "cm^-1")

        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    @overload
    def get_reference_ionization_energy(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_reference_ionization_energy(self, unit: str) -> float: ...

    def get_reference_ionization_energy(self, unit: str | None = "hartree") -> PintFloat | float:
        """Return the reference ionization energy in the desired unit.

        Args:
            unit: Desired unit for the ionization energy. Default is atomic units "hartree".

        Returns:
            Ionization energy in the desired unit.

        """
        ionization_energy_ = self.jl_parameters.threshold  # in cm^-1
        ionization_energy: PintFloat = ureg.Quantity(ionization_energy_, "cm^-1")

        ionization_energy = ionization_energy.to("hartree", "spectroscopy")
        if unit is None:
            return ionization_energy
        if unit == "a.u.":
            return ionization_energy.magnitude
        return ionization_energy.to(unit, "spectroscopy").magnitude

    def get_possible_models(self, angular_ket: AngularKetBase) -> list[Any]:
        if not hasattr(self, "_models_dict"):
            self._create_angular_ket_to_jl_model()
        return self._models_dict.get(angular_ket, [])

    def get_model_indices(self, angular_ket: AngularKetBase) -> list[Any]:
        if not hasattr(self, "_index_dict"):
            self._create_angular_ket_to_jl_model()
        return self._index_dict.get(angular_ket, [])

    def _create_angular_ket_to_jl_model(self) -> None:
        models_dict: dict[AngularKetBase, list[Any]] = {}
        index_dict: dict[AngularKetBase, list[Any]] = {}
        from rydstate.angular.angular_ket_dummy import AngularKetDummy  # noqa: PLC0415
        from rydstate.angular.utils import (  # noqa: PLC0415
            julia_qn_to_dict,
            quantum_numbers_to_angular_ket,
        )

        for model in self.jl_models:
            i_core = 0
            for i, term in enumerate(model.terms):
                angular_ket: AngularKetBase
                if model.core[i]:
                    qn_dict = julia_qn_to_dict(model.outer_channels.i[i_core])
                    angular_ket = quantum_numbers_to_angular_ket(self, **qn_dict)  # type: ignore [arg-type]
                    i_core += 1
                else:
                    name = f"model='{model.name}'; term='{term}'"
                    core_ket_name = term.split("n")[0]
                    core_ket = AngularCoreKetDummy(core_ket_name)
                    angular_ket = AngularKetDummy(name, f_tot=model.f_tot, core_ket=core_ket)

                models_dict.setdefault(angular_ket, [])
                models_dict[angular_ket].append(model)
                index_dict.setdefault(angular_ket, [])
                index_dict[angular_ket].append(i)

        self._models_dict = models_dict
        self._index_dict = index_dict

    def calc_nu(  # type: ignore [override]
        self,
        n: int,
        angular_ket: AngularKetBase,
    ) -> float:
        from juliacall import Main as jl  # noqa: N813, PLC0415

        jl_models = self.get_possible_models(angular_ket)
        indices = self.get_model_indices(angular_ket)
        if len(jl_models) == 0:
            if angular_ket.l_r < 5 and angular_ket.l_c == 0:
                logger.debug("calc_nu No MQDT.jl models found for %s for %s.", angular_ket, self.name)
            return n
        for ind, jl_model in zip(indices, jl_models):  # noqa: B007
            if jl_model.nu_range[0] <= n <= jl_model.nu_range[1]:
                break
        else:
            logger.debug("calc_nu MQDT.jl models found for %s for %s, but not for n=%d.", angular_ket, self.name, n)
            return n

        energy_nu = _calc_energy_nu(self, angular_ket, n)

        t = jl.MQDT.transform(energy_nu, jl_model, self.jl_parameters)
        t = np.array(t)
        nu_i = jl.MQDT.nu(energy_nu, jl_model, self.jl_parameters)
        mu_diag = jl.MQDT.theta(nu_i, jl_model.defects)
        mu_diag = np.diag(mu_diag)
        mu_fj = t @ mu_diag @ np.conjugate(t).T
        mu = mu_fj[ind, ind]

        return float(n - mu)


def _calc_energy_nu(species: SpeciesMQDTObject, angular_ket: AngularKetBase, n: int) -> float:
    core_energy = species.get_ionization_energy(angular_ket, unit="hartree")
    reference_core_energy = species.get_reference_ionization_energy(unit="hartree")

    nu_i = n  # TODO this is an approximation
    eps_i = calc_energy_from_nu(species.reduced_mass_au, nu_i)
    # E_tot = I_i + eps_i = I_ref + eps_ref with eps_i = -1 / (2 * (nu_i^2))
    # => eps_ref = eps_i + I_i - I_ref
    eps_ref = eps_i + core_energy - reference_core_energy
    if eps_ref >= 0:
        # channel state is above reference ionization threshold
        return 120.0  # large nu_ref value for continuum states
        # TODO larger value breaks for Yb171

    return calc_nu_from_energy(species.reduced_mass_au, eps_ref)
