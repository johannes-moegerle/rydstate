import math
from typing import Any

import pytest
from rydstate.angular.utils import NotSet, Unknown, parse_quantum_number, parse_spectroscopic_letter

REAL_NUMBERS = [-math.inf, -3, 0, 0.5, 2, 1e100, math.inf]


@pytest.mark.parametrize("singleton", [Unknown, NotSet])
def test_singleton_sorts_above_every_real_number(singleton: type[Any]) -> None:
    """Unknown and NotSet compare greater than every real number, including inf."""
    for number in REAL_NUMBERS:
        assert singleton > number
        assert singleton >= number
        assert not singleton < number
        assert not singleton <= number
        # reflected operators (number on the left-hand side)
        assert number < singleton
        assert number <= singleton
        assert not number > singleton
        assert not number >= singleton
        assert singleton != number  # type: ignore [comparison-overlap]
        assert number != singleton  # type: ignore [comparison-overlap]


@pytest.mark.parametrize("singleton", [Unknown, NotSet])
def test_singleton_comparison_with_itself(singleton: type[Any]) -> None:
    other = singleton
    assert singleton == other
    assert singleton <= other
    assert singleton >= other
    assert not singleton < other
    assert not singleton > other


def test_singletons_are_not_comparable_to_each_other() -> None:
    assert Unknown != NotSet  # type: ignore [comparison-overlap]
    with pytest.raises(TypeError):
        _ = Unknown < NotSet


@pytest.mark.parametrize("singleton", [Unknown, NotSet])
def test_singleton_arithmetic_returns_singleton(singleton: type[Any]) -> None:
    """Binary and unary arithmetic with real numbers propagates the singleton."""
    assert singleton + 1 is singleton
    assert 1 + singleton is singleton
    assert singleton - 0.5 is singleton
    assert 0.5 - singleton is singleton
    assert singleton * 2 is singleton
    assert 2 * singleton is singleton
    assert +singleton is singleton
    assert -singleton is singleton
    assert singleton + singleton is singleton


def test_singleton_arithmetic_with_non_number_raises() -> None:
    with pytest.raises(TypeError):
        _ = Unknown + "a"
    with pytest.raises(TypeError):
        _ = Unknown + NotSet


def test_sorting_with_unknown() -> None:
    """Unknown values sort after all real numbers, e.g. in filter and sort keys."""
    assert sorted([2.0, Unknown, 1.0]) == [1.0, 2.0, Unknown]  # type: ignore [type-var]
    assert max(3, Unknown) is Unknown
    assert min(3, Unknown) == 3
    # a range filter like in BasisBase.filter_states always excludes Unknown
    assert not 0 <= Unknown <= math.inf


@pytest.mark.parametrize(("string", "value"), [("3/2", 1.5), ("-1/2", -0.5), ("2", 2), ("-3", -3), ("5.5", 5.5)])
def test_parse_quantum_number(string: str, value: float) -> None:
    assert parse_quantum_number(string) == value


@pytest.mark.parametrize("string", ["1/4", "0.25", "a", "3/", ""])
def test_parse_quantum_number_invalid(string: str) -> None:
    with pytest.raises(ValueError, match="parse_quantum_number"):
        parse_quantum_number(string)


@pytest.mark.parametrize(("letter", "l"), [("s", 0), ("P", 1), ("d", 2), ("(l=7)", 7), ("(L=12)", 12)])
def test_parse_spectroscopic_letter(letter: str, l: int) -> None:
    assert parse_spectroscopic_letter(letter) == l


@pytest.mark.parametrize("letter", ["k", "(l=)", "(l=1.5)", "pp"])
def test_parse_spectroscopic_letter_invalid(letter: str) -> None:
    with pytest.raises(ValueError, match="Invalid spectroscopic letter"):
        parse_spectroscopic_letter(letter)
