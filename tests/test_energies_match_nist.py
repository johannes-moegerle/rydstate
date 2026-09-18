"""Check the absolute energies of the SQDT and MQDT models against the experimental NIST levels.

The energy of a Rydberg state is the sum of the ionization threshold and the (negative) binding energy.
The binding energy follows from the quantum defects (SQDT) or from the roots of det(M) (MQDT),
while the ionization threshold is an independent, tabulated constant.
Tests that only compare nu (or energy differences) are therefore blind to the threshold:
a wrong threshold shifts every level of the species by the very same amount.

The tests below compare the absolute energies against the NIST levels shipped with the species
and look at the median deviation, which is exactly such a constant offset.
"""

from __future__ import annotations

from statistics import median
from typing import TYPE_CHECKING, Any

import pytest
from rydstate.angular import AngularKetLS
from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.species import get_all_subclasses, get_element_properties, get_mqdt, get_potential_class, get_sqdt
from rydstate.species.mqdt import MQDT
from rydstate.species.sqdt import SQDT
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy
from rydstate.units import ureg

if TYPE_CHECKING:
    from collections.abc import Callable

ALL_SQDT_SPECIES = sorted([cls.species for cls in get_all_subclasses(SQDT)])
ALL_MQDT_SPECIES = sorted([cls.species for cls in get_all_subclasses(MQDT)])

HARTREE_TO_GHZ = ureg.Quantity(1, "hartree").to("GHz", "spectroscopy").magnitude

# Only NIST levels inside this nu window are used.
# Below NU_MIN the Rydberg-Ritz expansion is not converged (and even diverges for n close to delta_0),
# above NU_MAX the tabulated NIST energies are too imprecise to resolve the threshold
# (e.g. for Li the nu derived from the NIST energies drifts away by more than 0.3 above nu = 30).
NU_MIN = 10.0
NU_MAX = 25.0

# A series is a (l, j_tot, s_tot) Rydberg series; we only use series with enough levels inside the window,
# and only test species for which enough such series and levels remain.
MIN_LEVELS_PER_SERIES = 3
MIN_SERIES = 2
MIN_LEVELS = 10

# A nonzero median deviation shifts all levels of the species alike, i.e. it is an error of the
# ionization threshold rather than of the quantum defects / the MQDT model.
# The offsets that remain are at most ~1.7 GHz (Li) and are dominated by the rounding of the
# tabulated ionization energies (most of them are given as a six digit eV value, i.e. to ~0.24 GHz).
# For comparison, the Sr88 ionization energy this test guards against was 6.5 GHz too low.
MAX_CONSTANT_OFFSET_GHZ = 3.0

# The scatter around that constant offset is dominated by the quality of the quantum defects
# (and, for single series, by NIST levels that are perturbed or misassigned), so it is much looser.
MAX_TYPICAL_DEVIATION_GHZ = 10.0


def _collect_deviations(
    species: str,
    threshold_au: float,
    calc_energy_au: Callable[[int, float, AngularKetLS[Any]], float | None],
) -> dict[tuple[int, float, float], list[float]]:
    """Return the deviations (calculated - NIST, in GHz) of the NIST levels, grouped by Rydberg series.

    The nu used to select the levels is derived from the NIST energy and ``threshold_au`` only,
    so that the selection does not depend on the model that is being tested.

    Args:
        species: The species whose NIST levels are compared.
        threshold_au: The ionization threshold the levels are referenced to, in hartree.
        calc_energy_au: Callback returning the calculated energy (in hartree) of the level with the given
            n, nu and angular ket, or None if the model does not describe this level.

    Returns:
        The deviations in GHz per (l, j_tot, s_tot) series, restricted to series with enough levels.

    """
    sqdt = get_sqdt(species)
    element_properties = get_element_properties(species)

    deviations: dict[tuple[int, float, float], list[float]] = {}
    for (n, l, j_tot, s_tot), energy_nist_au in sorted(sqdt._nist_energy_levels.items()):  # noqa: SLF001
        binding_energy_au = energy_nist_au - threshold_au
        if binding_energy_au >= 0:  # level above the threshold, nu is not defined
            continue
        nu = calc_nu_from_energy(element_properties.reduced_mass_au, binding_energy_au, element_properties.net_charge)
        if not NU_MIN <= nu <= NU_MAX:
            continue

        angular = AngularKetLS(l_r=l, s_tot=s_tot, j_tot=j_tot, f_tot=j_tot + element_properties.i_c, species=species)
        energy_au = calc_energy_au(n, nu, angular)
        if energy_au is None:
            continue
        deviations.setdefault((l, j_tot, s_tot), []).append((energy_au - energy_nist_au) * HARTREE_TO_GHZ)

    return {key: values for key, values in deviations.items() if len(values) >= MIN_LEVELS_PER_SERIES}


def _assert_no_constant_offset(label: str, deviations: dict[tuple[int, float, float], list[float]]) -> None:
    """Assert that the deviations scatter around zero instead of around a constant offset.

    The offset is estimated as the median over the per-series medians, which is robust against
    both individual outliers and a single badly described Rydberg series.
    """
    all_deviations = [deviation for values in deviations.values() for deviation in values]
    if len(deviations) < MIN_SERIES or len(all_deviations) < MIN_LEVELS:
        pytest.skip(
            f"{label}: only {len(all_deviations)} NIST levels in {len(deviations)} series with "
            f"{NU_MIN} <= nu <= {NU_MAX}, which is not enough to resolve the ionization threshold"
        )

    offset = median([median(values) for values in deviations.values()])
    assert abs(offset) < MAX_CONSTANT_OFFSET_GHZ, (
        f"{label}: the calculated energies are offset from the NIST levels by {offset:+.3f} GHz "
        f"(median over {len(all_deviations)} levels in {len(deviations)} series), "
        f"which points to an inaccurate ionization threshold"
    )

    typical_deviation = median([abs(deviation - offset) for deviation in all_deviations])
    assert typical_deviation < MAX_TYPICAL_DEVIATION_GHZ, (
        f"{label}: the calculated energies deviate from the NIST levels by typically "
        f"{typical_deviation:.3f} GHz (median absolute deviation around the offset of {offset:+.3f} GHz)"
    )


@pytest.mark.parametrize("species", ALL_SQDT_SPECIES)
def test_sqdt_energies_match_nist(species: str) -> None:
    """The energies from the quantum defects and the ionization energy must match the NIST levels.

    The quantum defect theory is used explicitly (``use_nist_data=False``), since otherwise
    :meth:`~rydstate.species.sqdt.SQDT.calc_nui` looks the low lying levels up in the very same NIST data
    and reproduces them by construction, for any value of the ionization energy.
    """
    sqdt = get_sqdt(species)
    if sqdt.quantum_defects is None or not sqdt._nist_energy_levels:  # noqa: SLF001
        pytest.skip(f"{species} has no quantum defects and/or no NIST data.")
    element_properties = get_element_properties(species)

    def calc_energy_au(n: int, nu: float, angular: AngularKetLS[Any]) -> float | None:  # noqa: ARG001
        assert sqdt.quantum_defects is not None
        if (angular.l_r, angular.j_tot, angular.s_tot) not in sqdt.quantum_defects:
            return None
        nui = sqdt.calc_nui(n, angular, use_nist_data=False)
        binding_energy_au = calc_energy_from_nu(element_properties.reduced_mass_au, nui, element_properties.net_charge)
        return sqdt.ionization_energy_au + binding_energy_au

    deviations = _collect_deviations(species, sqdt.ionization_energy_au, calc_energy_au)
    _assert_no_constant_offset(f"SQDT({species})", deviations)


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
    if not sqdt._nist_energy_levels:  # noqa: SLF001
        pytest.skip(f"{species} has no NIST data.")
    potential_class = get_potential_class(species)

    def calc_energy_au(n: int, nu: float, angular: AngularKetLS[Any]) -> float | None:  # noqa: ARG001
        states = [
            state
            for model in mqdt.get_mqdt_models(angular)
            for state in get_mqdt_states_from_model(model, (nu - 0.5, nu + 0.5), NotSet, potential_class)
        ]
        if not states:
            return None
        return min(states, key=lambda state: abs(state.nu - nu)).get_energy("hartree")

    deviations = _collect_deviations(species, mqdt.reference_ionization_threshold_au, calc_energy_au)
    _assert_no_constant_offset(f"MQDT({species})", deviations)
