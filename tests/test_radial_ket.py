from rydstate.radial import RadialKet
from rydstate.radial.potential import get_potential_class


def test_radial_ket_is_reused_for_same_nu_and_potential() -> None:
    potential = get_potential_class("H")(0)

    radial_ket = RadialKet(9.9, potential, n_expected=10)
    reused_radial_ket = RadialKet(9.9, potential, n_expected=10)

    assert reused_radial_ket is radial_ket


def test_radial_ket_cache_distinguishes_nu_and_potential() -> None:
    potential = get_potential_class("H")(0)
    other_potential = get_potential_class("H")(1)

    radial_ket = RadialKet(10, potential)

    assert RadialKet(11, potential) is not radial_ket
    assert RadialKet(10, other_potential) is not radial_ket


def test_radial_ket_is_reused_for_equivalent_potential_requests() -> None:
    potential_class = get_potential_class("H")

    radial_ket = RadialKet(10, potential_class(0))

    assert RadialKet(10, potential_class(l_r=0)) is radial_ket
