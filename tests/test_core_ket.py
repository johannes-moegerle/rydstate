from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS, determine_n_c
from rydstate.angular.utils import Unknown, is_unknown
from rydstate.species import MQDT, ElementProperties, get_all_subclasses, get_element_properties

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import CouplingScheme

ALL_MQDTS = [cls() for cls in get_all_subclasses(MQDT)]
ALL_ELEMENT_PROPERTIES = [cls() for cls in get_all_subclasses(ElementProperties)]
MONOVALENT_ELEMENT_PROPERTIES = [ep for ep in ALL_ELEMENT_PROPERTIES if ep.number_valence_electrons == 1]
DIVALENT_ELEMENT_PROPERTIES = [ep for ep in ALL_ELEMENT_PROPERTIES if ep.number_valence_electrons == 2]

# The principal quantum number n_c of the core electron for l_c = 0, 1, 2, 3, 4.
# For l_c >= 2 the core shell lies below the ground state shell (e.g. Sr+ 4d and Yb+ 5d),
# i.e. it is taken from ElementProperties.additional_allowed_shells and not from ground_state_shell.
EXPECTED_N_C: dict[str, list[int]] = {
    "Sr87": [5, 5, 4, 4, 5],
    "Sr88": [5, 5, 4, 4, 5],
    "Yb171": [6, 6, 5, 5, 5],
    "Yb173": [6, 6, 5, 5, 5],
    "Yb174": [6, 6, 5, 5, 5],
}


@pytest.mark.parametrize(("species", "expected"), list(EXPECTED_N_C.items()))
def test_determine_n_c_divalent(species: str, expected: list[int]) -> None:
    """determine_n_c returns the lowest allowed core shell of the ion for the given l_c."""
    element_properties = get_element_properties(species)
    n_c_list = [determine_n_c(element_properties, l_c) for l_c in range(len(expected))]
    assert n_c_list == expected


def test_determine_n_c_divalent_all_species_pinned() -> None:
    """All divalent species must have their expected core shells pinned in EXPECTED_N_C."""
    missing = [ep.species for ep in DIVALENT_ELEMENT_PROPERTIES if ep.species not in EXPECTED_N_C]
    assert not missing, f"Add the expected n_c values for {missing} to EXPECTED_N_C."


@pytest.mark.parametrize("element_properties", MONOVALENT_ELEMENT_PROPERTIES, ids=lambda ep: ep.species)
def test_determine_n_c_monovalent(element_properties: ElementProperties) -> None:
    """Species with a single valence electron have no core electron, thus n_c is None."""
    assert all(determine_n_c(element_properties, l_c) is None for l_c in range(5))
    assert determine_n_c(element_properties, Unknown) is None


@pytest.mark.parametrize("element_properties", DIVALENT_ELEMENT_PROPERTIES, ids=lambda ep: ep.species)
def test_determine_n_c_unknown_l_c(element_properties: ElementProperties) -> None:
    """n_c cannot be determined without l_c, so it is Unknown as well."""
    assert is_unknown(determine_n_c(element_properties, Unknown))


@pytest.mark.parametrize("mqdt", ALL_MQDTS, ids=lambda mqdt: f"{mqdt.species}_{mqdt.tag}")
def test_ionization_threshold_core_kets_are_consistent(mqdt: MQDT) -> None:
    """Every core ket of the ionization_threshold_dict must be one that get_core_ket() actually produces.

    Since n_c is part of CoreKet.quantum_number_names, a hardcoded n_c in the ionization_threshold_dict
    that drifts out of sync with determine_n_c would no longer match any channel,
    i.e. the entry silently becomes unreachable and get_ionization_threshold either fails
    or falls back to a broader (wrong) entry.
    """
    element_properties = get_element_properties(mqdt.species)
    # the core kets of all outer channels, i.e. {channel.get_core_ket() for channel in ...}
    channel_core_kets = mqdt.get_core_kets()

    for core_ket in mqdt.ionization_threshold_dict:
        expected_n_c = determine_n_c(element_properties, core_ket.l_c)
        assert core_ket.n_c == expected_n_c, (
            f"{mqdt!r}: ionization threshold entry {core_ket} has n_c={core_ket.n_c}, "
            f"but determine_n_c returns {expected_n_c} for l_c={core_ket.l_c}"
        )
        matching = [
            ket for ket in channel_core_kets if ket.find_matching_core_ket(mqdt.ionization_threshold_dict) is core_ket
        ]
        assert matching, (
            f"{mqdt!r}: ionization threshold entry {core_ket} is not used by any channel core ket "
            f"(candidates: {channel_core_kets})"
        )


SOURCE_KETS: dict[str, AngularKetBase[Any]] = {
    "LS_Sr88": AngularKetLS(l_c=0, l_r=1, s_tot=1, l_tot=1, j_tot=1, f_tot=1, species="Sr88"),
    "JJ_Sr88": AngularKetJJ(l_c=0, l_r=1, j_c=0.5, j_r=1.5, j_tot=1, f_tot=1, species="Sr88"),
    "FJ_Yb171": AngularKetFJ(l_c=0, l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, species="Yb171"),
    "JJ_Yb174_l_c=2": AngularKetJJ(l_c=2, l_r=1, j_c=1.5, j_r=1.5, j_tot=1, f_tot=1, species="Yb174"),
    "LS_unknown": AngularKetLS(l_c=Unknown, f_tot=0.5, parity=1, label="?", allow_unknown=True, species="Yb171"),
    "JJ_unknown": AngularKetJJ(l_c=Unknown, f_tot=0.5, parity=1, label="?", allow_unknown=True, species="Yb171"),
    "FJ_unknown": AngularKetFJ(l_c=Unknown, f_tot=0.5, parity=1, label="?", allow_unknown=True, species="Yb171"),
}


@pytest.mark.parametrize("ket_name", SOURCE_KETS)
def test_transformations_preserve_n_c(ket_name: str, coupling_scheme: CouplingScheme) -> None:
    """n_c is not part of quantum_numbers, so every ket rebuild has to carry it over explicitly.

    Neither replacing m nor changing the coupling scheme changes which core shell the state is built
    on. The rebuilt kets are constructed without a species, i.e. n_c cannot be re-derived: a dropped
    n_c silently becomes None (or None instead of Unknown) and only surfaces much later,
    in get_core_ket / core_state / __eq__.
    """
    ket = SOURCE_KETS[ket_name]
    assert ket.n_c is not None
    assert ket.replace_m(ket.f_tot).n_c == ket.n_c
    assert ket.get_core_ket().n_c == ket.n_c
    converted = ket.to_state(coupling_scheme).kets
    assert converted
    assert all(k.n_c == ket.n_c for k in converted), f"{ket} -> {[k.n_c for k in converted]}"
