from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import AngularKetFJ
from rydstate.angular.utils import NotSet, is_unknown
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.linalg import calc_nullvector, find_roots
from rydstate.radial import RadialDummy, RadialKet
from rydstate.rydberg_state import RydbergKet, RydbergStateMQDT
from rydstate.species import MQDT, Potential, get_mqdt, get_potential_class

if TYPE_CHECKING:
    from typing_extensions import Self

    from rydstate.species import FModel


logger = logging.getLogger(__name__)


class BasisMQDT(BasisBase[RydbergStateMQDT]):
    states: list[RydbergStateMQDT]

    def __init__(
        self,
        species: str,
        nu: tuple[float, float],
        *,
        l_r: tuple[int, int] | None = None,
        f_tot: tuple[float, float] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        # potential and mqdt parameters
        mqdt: MQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        """Initialize the MQDT basis.

        Args:
            species: Atomic species.
            nu: Tuple of (nu_min, nu_max) for the effective principal quantum number.
            l_r: Optional tuple of (l_r_min, l_r_max) for the Rydberg electron orbital angular momentum.
                This is used to filter models, which include at least one channel with
                l_c=0 and l_r in the specified range.
                Default None, include all models.
            f_tot: Optional tuple of (f_tot_min, f_tot_max) for the total angular momentum.
                Default None, include all f_tot values.
            m: Optional tuple of (m_min, m_max) for the magnetic quantum number range.
                Default NotSet, only include states with m=NotSet.
                If m is given as None, include all allowed m values.
            mqdt: The MQDT data to use for the states.
                Either an instance of an MQDT class
                or a string representing the tag of the MQDT class to use.
            potential_class: The potential class to use for the radial ket.
                Either a a potential class
                or a string representing the tag of the potential class to use.

        """
        super().__init__(species)
        self.mqdt = mqdt if isinstance(mqdt, MQDT) else get_mqdt(species, tag=mqdt)

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        else:
            self.potential_class = get_potential_class(species, tag=potential_class)

        # the maximum l_r is limited by the maximum nu, because l_r < n for bound states
        # and for high l_r the quantum defects are 0, so n = nu
        max_l_r = int(nu[1])
        self._init_models(max_l_r, f_tot, l_r)
        self._init_states(nu, m)

    def shallow_copy(self) -> Self:
        """Return a shallow copy of the basis (with its own independent list of states)."""
        new_basis = super().shallow_copy()
        new_basis.models = self.models.copy()
        return new_basis

    def _init_models(
        self,
        max_l_r: int,
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
    ) -> None:
        self.models: list[FModel] = []
        s_r = 0.5
        i_c = self.element_properties.i_c
        s_c = self.element_properties.s_c
        j_c = s_c
        for l_r in range(max_l_r + 1):
            if not is_allowed_qn(l_r_range, l_r):
                continue
            for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
                for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                    for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                        if not is_allowed_qn(f_tot_range, f_tot):
                            continue
                        channel = AngularKetFJ(
                            l_r=l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(f_tot), species=self.species
                        )
                        for model in self.mqdt.get_mqdt_models(channel):
                            if model not in self.models:
                                self.models.append(model)

    def _init_states(
        self,
        nu_range: tuple[float, float],
        m_range: tuple[float, float] | None | NotSet,
    ) -> None:
        logger.debug("Calculating MQDT states...")
        self.states = []
        for model in self.models:
            logger.debug("  calculating states for model %s with nu_range=%s", model.name, nu_range)
            states = get_mqdt_states_from_fmodel(model, nu_range, m_range, self.potential_class)
            if len(states) == 0:
                logger.debug("  no states found for model %s", model.name)
            else:
                logger.debug(
                    "  model %s: nu_min=%s, nu_max=%s, total states=%d",
                    model.name,
                    states[0].nu,
                    states[-1].nu,
                    len(states),
                )
            self.states.extend(states)

        self.states.sort(key=lambda state: state.nu)


def get_mqdt_states_from_fmodel(
    model: FModel,
    nu_range: tuple[float, float],
    m_range: tuple[float, float] | None | NotSet,
    potential_class: type[Potential],
) -> list[RydbergStateMQDT]:
    """Calculate MQDT states from an FModel by finding zeros of det(M-matrix).

    Args:
        model: The MQDT model to compute states for.
        nu_range: Tuple of (nu_min, nu_max) for the search range.
        m_range: Tuple of (m_min, m_max) for the magnetic quantum number range.
            NotSet will only include states with m=NotSet.
        potential_class: The potential class to use for the radial ket.

    Returns:
        List of :class:`RydbergStateMQDT` objects, one per root of det(M).

    """
    nu_min = max(nu_range[0], model.nu_min)
    nu_max = min(nu_range[1], model.nu_max)
    if np.isinf(nu_max):
        raise ValueError("nu_max must be finite to calculate MQDT states.")

    nu_list = find_roots(lambda nu: np.linalg.det(model.calc_scaled_m_matrix(nu)), nu_min, nu_max)
    if len(nu_list) == 0:
        if nu_max - nu_min > 1.0:
            logger.warning(
                "No MQDT states found for model %s in the range nu_min=%s, nu_max=%s", model.full_name, nu_min, nu_max
            )
        return []

    coefficients_fj: list[float] = []
    angular_kets_fj: list[AngularKetFJ[Any]] = []
    number_kets_fj: list[int] = [0] * len(model.outer_channels)
    for i, angular_ket in enumerate(model.outer_channels):
        for coeff_fj, ket_fj in angular_ket.to_state("FJ"):
            coefficients_fj.append(coeff_fj)
            angular_kets_fj.append(ket_fj)
            number_kets_fj[i] += 1

    states: list[RydbergStateMQDT] = []
    for nu in nu_list:
        nuis = model.calc_channel_nuis(nu)
        coefficients = calc_nullvector(model.calc_scaled_m_matrix(nu))
        coefficients = np.array(
            [coeff * (nui ** (3 / 2)) / np.cos(np.pi * nui) for coeff, nui in zip(coefficients, nuis, strict=True)]
        )
        coefficients /= np.linalg.norm(coefficients)
        arg_max = np.argmax(np.abs(coefficients))
        coefficients *= np.sign(coefficients[arg_max])

        radial_kets: list[RadialKet | RadialDummy] = []
        for nui, angular_ket in zip(nuis, model.outer_channels, strict=True):
            radial: RadialKet | RadialDummy
            if not is_unknown(angular_ket.l_r):
                potential = potential_class(angular_ket.l_r)
                radial = RadialKet(float(nui), potential, sign_convention="positive_at_outer_bound")
            else:
                radial = RadialDummy(1.0, nui)
            radial_kets.append(radial)

        radial_kets_fj = [
            radial for radial, number in zip(radial_kets, number_kets_fj, strict=True) for _ in range(number)
        ]
        coefficients_all = [
            coeff for coeff, number in zip(coefficients, number_kets_fj, strict=True) for _ in range(number)
        ]
        coefficients_all = np.array(coefficients_all) * np.array(coefficients_fj)

        energy_au = model.calc_energy_au(nu)
        for m in get_m_range(model.f_tot, m_range):
            rydberg_kets = [
                RydbergKet(model.species, angular_ket.replace_m(m), radial_ket)
                for angular_ket, radial_ket in zip(angular_kets_fj, radial_kets_fj, strict=True)
            ]
            states.append(
                RydbergStateMQDT(
                    model.species,
                    coefficients_all,
                    rydberg_kets,
                    nu=nu,
                    energy_au=energy_au,
                    model=model,
                    potential_class=potential_class,
                )
            )

    return states
