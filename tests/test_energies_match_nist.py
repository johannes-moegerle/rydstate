"""Check the absolute energies of the SQDT and MQDT models against the experimental NIST levels.

The energy of a Rydberg state is the ionization threshold plus the (negative) binding energy.
Tests that only compare nu or energy differences are blind to the threshold, since a wrong threshold
shifts all levels of a species by the same amount. The tests here therefore compare absolute energies
and look at the median deviation, which is exactly such a constant offset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate.angular import AngularKetLS
from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.element_properties import get_element_properties
from rydstate.qdt import get_mqdt, get_sqdt
from rydstate.qdt.mqdt import ModelSetMQDT
from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT, ModelSQDTNist
from rydstate.qdt.sqdt.model_set_sqdt import _SQDT_MODEL_SETS
from rydstate.radial.potential import get_potential_class
from rydstate.units import ureg
from rydstate.utils import calc_energy_from_nu, calc_nu_from_energy, get_all_subclasses

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rydstate.angular.utils import AllKnown

ALL_SQDT_SPECIES = sorted(_SQDT_MODEL_SETS)
ALL_MQDT_SPECIES = sorted([cls.species for cls in get_all_subclasses(ModelSetMQDT)])

HARTREE_TO_GHZ = ureg.Quantity(1, "hartree").to("GHz", "spectroscopy").magnitude

# The nu and the deviation (calculated - NIST, in GHz) of each level, keyed by (n, l, j_tot, s_tot).
Deviations = dict[tuple[int, int, float, float], tuple[float, float]]


def _nist_models(sqdt: ModelSetSQDT) -> list[ModelSQDTNist]:
    """Return the models of the model set, which describe a channel by its NIST levels."""
    return [model for model in sqdt.models or [] if isinstance(model, ModelSQDTNist)]


def _nist_levels(species: str, threshold_au: float) -> Iterator[tuple[int, AngularKetLS[AllKnown], float, float]]:
    """Yield (n, angular ket, nu, NIST energy in hartree) for all NIST levels.

    The nu is derived from the NIST energy and the threshold only, so that selecting levels by it
    does not depend on the model that is being tested.
    """
    sqdt = get_sqdt(species)
    element_properties = get_element_properties(species)
    levels = {
        (n, model.channel.l_r, model.channel.j_tot, model.channel.s_tot): energy
        for model in _nist_models(sqdt)
        for n, energy in model.energies.items()
    }
    for (n, l, j_tot, s_tot), energy_nist_au in sorted(levels.items()):
        binding_energy_au = energy_nist_au - threshold_au
        nu = calc_nu_from_energy(element_properties.reduced_mass_au, binding_energy_au, element_properties.net_charge)
        angular = AngularKetLS(l_r=l, s_tot=s_tot, j_tot=j_tot, f_tot=j_tot + element_properties.i_c, species=species)
        yield n, angular, nu, energy_nist_au


# only levels with nu >= NU_MIN are compared, for lower nu the quantum defects are not accurate enough
# (and for SQDT the Rydberg-Ritz expansion can even be outside of its range of validity)
NU_MIN = 10


def _assert_no_constant_offset(label: str, deviations: Deviations) -> None:
    """Assert that the deviations scatter around zero instead of around a constant offset.

    A large offset is only blamed on the ionization threshold if it also stands out from the scatter
    of the levels; if it does not, the NIST data (or the quantum defects) are simply too noisy to tell.
    """
    values = [deviation for nu, deviation in deviations.values() if nu >= NU_MIN]
    if len(values) < 10:
        pytest.skip(
            f"{label}: only {len(values)} of the {len(deviations)} NIST levels have nu >= {NU_MIN}, "
            f"which is not enough to resolve the ionization threshold"
        )

    offset = np.median(values)
    scatter = np.median([abs(value - offset) for value in values])

    # A nonzero median deviation shifts all levels of the species alike, i.e. it is an error of the ionization
    # threshold rather than of the quantum defects. Offsets below max_constant_offset_ghz are tolerated.
    # A larger offset is only conclusive if it also stands out from the scatter of the levels around it by
    # min_significance, since that scatter is how well the quantum defects and the tabulated NIST energies describe
    # the levels in the first place.
    max_constant_offset_ghz = 0.5
    min_significance = 2.0

    if max_constant_offset_ghz <= abs(offset) < min_significance * scatter:
        pytest.skip(
            f"{label}: the {len(values)} NIST levels scatter by {scatter:.3f} GHz around their median, "
            f"too much to tell whether the offset of {offset:+.3f} GHz is an error of the ionization threshold"
        )

    assert abs(offset) < max_constant_offset_ghz, (
        f"{label}: the calculated energies are offset from the NIST levels by {offset:+.3f} GHz, "
        f"while the {len(values)} levels scatter by only {scatter:.3f} GHz around it, "
        f"which points to an inaccurate ionization threshold"
    )


@pytest.mark.parametrize("species", ALL_SQDT_SPECIES)
def test_sqdt_energies_match_nist(species: str) -> None:
    """The energies from the quantum defects and the ionization energy must match the NIST levels."""
    sqdt = get_sqdt(species)
    if not any(isinstance(model, ModelSQDT) for model in sqdt.models or []) or not _nist_models(sqdt):
        pytest.skip(f"{species} has no quantum defects and/or no NIST data.")
    element_properties = get_element_properties(species)

    deviations: Deviations = {}
    reference_threshold_au = element_properties.get_reference_ionization_threshold("a.u.")
    for n, angular, nu, energy_nist_au in _nist_levels(species, reference_threshold_au):
        qd_models = [model for model in sqdt.get_models(angular) if isinstance(model, ModelSQDT)]
        if nu < NU_MIN or len(qd_models) == 0:
            continue
        nui = qd_models[0].calc_nui(n)
        binding_energy_au = calc_energy_from_nu(element_properties.reduced_mass_au, nui, element_properties.net_charge)
        energy_au = qd_models[0].ionization_threshold_au + binding_energy_au
        deviations[n, angular.l_r, angular.j_tot, angular.s_tot] = (nu, (energy_au - energy_nist_au) * HARTREE_TO_GHZ)

    _assert_no_constant_offset(f"{species} - SQDT", deviations)


@pytest.mark.parametrize("species", ALL_MQDT_SPECIES)
def test_mqdt_energies_match_nist(species: str) -> None:
    """The energies of the MQDT states must match the NIST levels.

    In contrast to ``test_mqdt_energies_match_nist`` in test_mqdt_references.py, which compares nu,
    this compares the absolute energies and therefore also covers the ionization thresholds
    of the MQDT models (nu is defined with respect to the reference threshold and is blind to it).
    """
    mqdt = get_mqdt(species)
    try:
        sqdt = get_sqdt(species)
    except ValueError:
        pytest.skip(f"{species} has no SQDT class and therefore no NIST data.")
    if not _nist_models(sqdt):
        pytest.skip(f"{species} has no NIST data.")
    potential_class = get_potential_class(species)

    deviations: Deviations = {}
    reference_threshold_au = mqdt.element_properties.get_reference_ionization_threshold("a.u.")
    for n, angular, nu, energy_nist_au in _nist_levels(species, reference_threshold_au):
        states = [
            state
            for model in mqdt.get_mqdt_models(angular)
            for state in get_mqdt_states_from_model(model, (nu - 0.5, nu + 0.5), NotSet, potential_class)
        ]
        if not states:
            continue
        energy_au = min(states, key=lambda state: abs(state.nu - nu)).get_energy("hartree")
        deviations[n, angular.l_r, angular.j_tot, angular.s_tot] = (nu, (energy_au - energy_nist_au) * HARTREE_TO_GHZ)

    _assert_no_constant_offset(f"{species} - MQDT", deviations)
