import numpy as np
import pytest
from rydstate.rydberg_state import RydbergStateSQDT
from rydstate.species.potential import get_potential_class


def test_potential_is_reused_for_same_class_and_l_r() -> None:
    potential_class = get_potential_class("H")

    potential = potential_class(0)

    assert potential_class(0) is potential
    assert potential_class(l_r=0) is potential


def test_potential_cache_distinguishes_class_and_l_r() -> None:
    hydrogen_potential_class = get_potential_class("H")
    rubidium_potential_class = get_potential_class("Rb")

    potential = hydrogen_potential_class(0)

    assert hydrogen_potential_class(1) is not potential
    assert rubidium_potential_class(0) is not potential


@pytest.mark.parametrize(
    ("species", "n_s", "n_p", "ref_p12", "ref_p32"),
    [
        # reference reduced dipole matrix elements |<ns_1/2||d||np_j>| in e a0
        # Yb+: Chen et al. (2023) Table 2, RCICP values for 6s -> 6p_j
        ("Yb174_ion", 6, 6, 2.63, 3.73),
        # Sr+: no j-dependence check, only that the values are finite and physically reasonable
        ("Sr88_ion", 5, 5, None, None),
    ],
)
def test_ion_dipole_matrix_elements(
    species: str, n_s: int, n_p: int, ref_p12: float | None, ref_p32: float | None
) -> None:
    # This exercises the (default) j-dependent core-polarizability potential through the standard
    # (n, l, j) interface, which uses LS coupling where j_r is derived from j_tot (spinless ion core).
    s = RydbergStateSQDT(species, n=n_s, l_r=0, j_tot=0.5)
    p12 = RydbergStateSQDT(species, n=n_p, l_r=1, j_tot=0.5)
    p32 = RydbergStateSQDT(species, n=n_p, l_r=1, j_tot=1.5)

    d12 = abs(s.calc_reduced_matrix_element(p12, "electric_dipole", unit="e a0"))
    d32 = abs(s.calc_reduced_matrix_element(p32, "electric_dipole", unit="e a0"))

    # both partners have a sizeable dipole and the fine-structure ratio d32/d12 is close to sqrt(2)
    assert np.isfinite(d12)
    assert np.isfinite(d32)
    assert 1.0 < d12 < 6.0
    assert 1.0 < d32 < 8.0
    assert abs(d32 / d12 / np.sqrt(2) - 1) < 0.1

    # where reference values are available, the model potential should reproduce them within ~30 %
    # (the residual is dominated by the missing core-polarization correction to the dipole operator)
    if ref_p12 is not None and ref_p32 is not None:
        assert abs(d12 / ref_p12 - 1) < 0.3, f"{species} <s||d||p1/2>={d12:.3f}, ref={ref_p12}"
        assert abs(d32 / ref_p32 - 1) < 0.3, f"{species} <s||d||p3/2>={d32:.3f}, ref={ref_p32}"
