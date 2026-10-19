# ruff: noqa: SLF001

import gc
import weakref

from rydstate.angular import AngularKetLS
from rydstate.radial import RadialKet
from rydstate.radial.potential import get_potential_class


def test_angular_reduced_matrix_element_is_cached() -> None:
    bra = AngularKetLS(l_r=0, f_tot=0.5, species="H")
    ket = AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="H")

    value = bra.calc_reduced_matrix_element(ket, "spherical", 1)

    # the result is stored in the cache keyed by a weakref to the other ket
    assert ket._ref in bra._reduced_matrix_element_cache
    assert bra._reduced_matrix_element_cache[ket._ref][("spherical", 1)] == value

    # overwriting the cached value proves the second call reads from the cache
    bra._reduced_matrix_element_cache[ket._ref][("spherical", 1)] = value + 1
    assert bra.calc_reduced_matrix_element(ket, "spherical", 1) == value + 1


def test_angular_reduced_matrix_element_cache_is_freed_when_other_is_deleted() -> None:
    bra = AngularKetLS(l_r=0, f_tot=0.5, species="H")
    ket = AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="H")

    bra.calc_reduced_matrix_element(ket, "spherical", 1)

    ket_reference = weakref.ref(ket)
    assert len(bra._reduced_matrix_element_cache) == 1

    del ket
    gc.collect()

    # once the other ket is no longer referenced, it (and its cache entry) is freed
    assert ket_reference() is None
    assert len(bra._reduced_matrix_element_cache) == 0


def test_radial_matrix_element_is_cached() -> None:
    potential = get_potential_class("H")(0)
    bra = RadialKet(10, potential)
    ket = RadialKet(12, potential)

    value = bra.calc_matrix_element(ket, k_radial=1, unit="a.u.")

    # the result is stored in the cache keyed by the other ket
    assert ket in bra._matrix_element_cache
    assert bra._matrix_element_cache[ket][(1, "sum")] == value

    # overwriting the cached value proves the second call reads from the cache
    bra._matrix_element_cache[ket][(1, "sum")] = value + 1
    assert bra.calc_matrix_element(ket, k_radial=1, unit="a.u.") == value + 1


def test_radial_matrix_element_reuses_cache_of_reversed_pair() -> None:
    potential = get_potential_class("H")(0)
    bra = RadialKet(10, potential)
    ket = RadialKet(12, potential)

    value = bra.calc_matrix_element(ket, k_radial=1, unit="a.u.")

    # poisoning bra's cache proves the reversed call redirects to it instead of recomputing
    bra._matrix_element_cache[ket][(1, "sum")] = value + 1
    assert ket.calc_matrix_element(bra, k_radial=1, unit="a.u.") == value + 1


def test_radial_matrix_element_cache_is_freed_when_other_is_deleted() -> None:
    potential = get_potential_class("H")(0)
    bra = RadialKet(10, potential)
    ket = RadialKet(12, potential)

    bra.calc_matrix_element(ket, k_radial=1, unit="a.u.")

    ket_reference = weakref.ref(ket)
    assert len(bra._matrix_element_cache) == 1

    del ket
    gc.collect()

    # once the other ket is no longer referenced, it (and its cache entry) is freed
    assert ket_reference() is None
    assert len(bra._matrix_element_cache) == 0
