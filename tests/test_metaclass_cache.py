import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from rydstate.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.qdt import get_mqdt, get_sqdt
from rydstate.species import get_element_properties


class CachedExample(metaclass=CachedABCMeta):
    def __init__(self, value: int, label: str = "default") -> None:
        self.value = value
        self.label = label


def test_cache_normalizes_constructor_arguments() -> None:
    instance = CachedExample(1)

    assert CachedExample(value=1) is instance
    assert CachedExample(1, label="default") is instance


def test_cache_distinguishes_classes_and_arguments() -> None:
    class OtherCachedExample(CachedExample):
        pass

    instance = CachedExample(1)

    assert CachedExample(2) is not instance
    assert OtherCachedExample(1) is not instance


def test_cache_uses_weak_values() -> None:
    instance = CachedExample(3)
    instance_reference = weakref.ref(instance)
    del instance
    gc.collect()

    assert instance_reference() is None

    recreated_instance = CachedExample(3)
    assert recreated_instance is not None


def test_clear_cached_instances_removes_cached_instances_for_that_class_only() -> None:
    class OtherCachedExample(metaclass=CachedABCMeta):
        def __init__(self, value: int) -> None:
            self.value = value

    instance = CachedExample(4)
    other_instance = OtherCachedExample(4)

    CachedExample.clear_cached_instances()

    assert CachedExample(4) is not instance
    assert OtherCachedExample(4) is other_instance


def test_cache_rejects_unhashable_arguments() -> None:
    class UnhashableArgumentExample(metaclass=CachedABCMeta):
        def __init__(self, values: list[int]) -> None:
            self.values = values

    with pytest.raises(TypeError, match="must be hashable"):
        UnhashableArgumentExample([1])


def test_cache_serializes_concurrent_construction() -> None:
    construction_count = 0
    construction_count_lock = threading.Lock()

    class ConcurrentExample(metaclass=CachedABCMeta):
        def __init__(self, value: int) -> None:
            nonlocal construction_count
            with construction_count_lock:
                construction_count += 1
            self.value = value

    with ThreadPoolExecutor(max_workers=8) as executor:
        instances = list(executor.map(ConcurrentExample, [1] * 32))

    assert all(instance is instances[0] for instance in instances)
    assert construction_count == 1


def test_species_data_classes_are_reused() -> None:
    element_properties = get_element_properties("H")
    sqdt = get_sqdt("H")
    mqdt = get_mqdt("Sr87")

    assert get_element_properties("H") is element_properties
    assert get_sqdt("H") is sqdt
    assert get_mqdt("Sr87") is mqdt


@pytest.mark.parametrize(
    "angular_ket_class",
    [AngularKetLS, AngularKetJJ, AngularKetFJ],
)
def test_angular_kets_are_reused(
    angular_ket_class: type[AngularKetLS[Any] | AngularKetJJ[Any] | AngularKetFJ[Any]],
) -> None:
    angular_ket = angular_ket_class(l_r=0, f_tot=0.5, species="H")

    reused_angular_ket = angular_ket_class(
        l_c=0,
        s_r=0.5,
        l_r=0,
        f_tot=0.5,
        species="H",
        allow_unknown=False,
    )

    assert reused_angular_ket is angular_ket


def test_angular_ket_cache_is_constructor_based() -> None:
    species_based_ket = AngularKetLS(l_r=0, f_tot=0.5, species="H")
    explicit_ket = AngularKetLS(i_c=0, s_c=0, l_r=0, f_tot=0.5)

    assert explicit_ket is not species_based_ket
    assert explicit_ket == species_based_ket
