from rydstate.radial.potential import get_potential_class


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
