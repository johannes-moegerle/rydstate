from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from rydstate.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.short_label import format_short_label, parse_short_label
from rydstate.angular.utils import Unknown

if TYPE_CHECKING:
    from rydstate.angular.utils import CouplingScheme

KET_CLASSES: dict[CouplingScheme, type[AngularKetBase[Any]]] = {
    "LS": AngularKetLS,
    "JJ": AngularKetJJ,
    "FJ": AngularKetFJ,
}


@pytest.mark.parametrize(
    ("template", "values", "omit", "label"),
    [
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": 2}, (), "3/2 b=2"),
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": None}, (), "3/2 "),
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": 2}, ("b",), "3/2 "),
        ("{a:qn} b={b:qn}", {"a": 1.5, "b": 2}, ("b",), "3/2 b=2"),  # omit only affects optional parts
        ("{s:mult}{l:L}", {"s": 1, "l": 2}, (), "3D"),
    ],
)
def test_format_short_label(template: str, values: dict[str, float | None], omit: tuple[str], label: str) -> None:
    assert format_short_label(template, values, omit) == label


def test_format_short_label_missing_required_value() -> None:
    with pytest.raises(ValueError, match="required field"):
        format_short_label("{a:qn} {b:qn}", {"a": 1})


@pytest.mark.parametrize("template", ["{a:qn} [b={b:qn}", "{a:qn}] b", "[[{a:qn}]]", "{a:unknown_type}", "{a}"])
def test_invalid_template(template: str) -> None:
    with pytest.raises(ValueError, match="template"):
        parse_short_label(template, "1")


def test_format_short_label_placeholder() -> None:
    """A required field without value shows the placeholder of its type, string values are inserted as is."""
    template = "{n:n}{l:l}_{j:qn}"
    assert format_short_label(template, {"l": 1, "j": 1.5}) == "np_3/2"
    assert format_short_label(template, {"n": None, "l": 1, "j": 1.5}) == "np_3/2"
    assert format_short_label(template, {"n": "[57.3]", "l": 1, "j": 1.5}) == "[57.3]p_3/2"
    assert format_short_label(template, {"n": 60, "l": 1, "j": 1.5}) == "60p_3/2"


def test_parse_short_label_n() -> None:
    """The placeholder "n" is not part of the parsed values, an integer n is."""
    template = "{n:n}{l:l}_{j:qn}"
    assert parse_short_label(template, "np_3/2") == {"l": 1, "j": 1.5}
    assert parse_short_label(template, "60p_3/2") == {"n": 60, "l": 1, "j": 1.5}


def test_parse_short_label() -> None:
    template = "[{n:int}]{l:l}_{j:qn}[ m={m:qn}]"
    assert parse_short_label(template, "5d_5/2 m=-3/2") == {"n": 5, "l": 2, "j": 2.5, "m": -1.5}
    assert parse_short_label(template, "d_5/2") == {"l": 2, "j": 2.5}


@pytest.mark.parametrize(
    ("species", "ket", "label"),
    [
        # single valence electron: the label is the same in all coupling schemes
        ("Rb", AngularKetLS(species="Rb", l_r=0, j_tot=0.5), "ns_1/2"),
        ("Rb", AngularKetLS(species="Rb", l_r=1, j_tot=1.5, m=0.5), "np_3/2 m=1/2"),
        ("Rb", AngularKetJJ(species="Rb", l_r=1, j_r=1.5, m=0.5), "np_3/2 m=1/2"),
        ("Rb", AngularKetFJ(species="Rb", l_r=1, j_r=1.5, m=0.5), "np_3/2 m=1/2"),
        ("Cs", AngularKetJJ(species="Cs", l_r=7, j_r=6.5, m=-5.5), "n(l=7)_6.5 m=-5.5"),
        ("Yb171_ion", AngularKetLS(species="Yb171_ion", l_r=1, j_tot=1.5, f_tot=1, m=0), "np_3/2 F=1 m=0"),
        ("Yb171_ion", AngularKetFJ(species="Yb171_ion", l_r=1, j_r=1.5, f_tot=2), "np_3/2 F=2"),
        # LS coupling
        ("Sr88", AngularKetLS(species="Sr88", l_r=0, s_tot=0, j_tot=0, m=0), "5s ns 1S0 m=0"),
        ("Sr88", AngularKetLS(species="Sr88", l_r=2, s_tot=1, j_tot=3), "5s nd 3D3"),
        ("Yb174", AngularKetLS(species="Yb174", l_c=1, l_r=1, l_tot=1, s_tot=1, j_tot=0), "6p np 3P0"),
        ("Yb174", AngularKetLS(species="Yb174", l_c=2, l_r=7, l_tot=7, s_tot=0, j_tot=7), "5d n(l=7) 1(L=7)7"),
        ("Sr87", AngularKetLS(species="Sr87", l_r=2, s_tot=1, j_tot=1, f_tot=5.5, m=0.5), "5s nd 3D1 F=5.5 m=1/2"),
        ("Yb171", AngularKetLS(species="Yb171", l_r=0, s_tot=0, j_tot=0, f_tot=0.5), "6s ns 1S0 F=1/2"),
        # JJ coupling
        ("Sr88", AngularKetJJ(species="Sr88", l_r=1, j_c=0.5, j_r=1.5, j_tot=1, m=0), "5s_1/2 np_3/2 J=1 m=0"),
        ("Yb174", AngularKetJJ(species="Yb174", l_c=1, l_r=1, j_c=1.5, j_r=0.5, j_tot=2), "6p_3/2 np_1/2 J=2"),
        (
            "Yb171",
            AngularKetJJ(species="Yb171", l_r=1, j_c=0.5, j_r=0.5, j_tot=1, f_tot=1.5),
            "6s_1/2 np_1/2 J=1 F=3/2",
        ),
        # FJ coupling: F is always shown, Fc only if i_c is non-zero
        (
            "Yb171",
            AngularKetFJ(species="Yb171", l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, m=0.5),
            "6s_1/2(Fc=1) np_3/2 F=3/2 m=1/2",
        ),
        ("Yb174", AngularKetFJ(species="Yb174", l_r=1, j_c=0.5, j_r=1.5, f_tot=1), "6s_1/2 np_3/2 F=1"),
    ],
)
def test_short_label_round_trip(species: str, ket: AngularKetBase[Any], label: str) -> None:
    assert ket.get_short_label() == label
    assert type(ket).from_short_label(label, species) == ket


@pytest.mark.parametrize(
    ("species", "label", "coupling_scheme"),
    [
        ("Rb", "np_3/2 m=1/2", "LS"),  # a single valence electron defaults to LS coupling
        ("Sr88", "5s nd 3D3", "LS"),
        ("Sr88", "5s_1/2 np_3/2 J=1 m=0", "JJ"),
        ("Yb171", "6s_1/2(Fc=1) np_3/2 F=3/2 m=1/2", "FJ"),
    ],
)
def test_from_short_label_infers_coupling_scheme(species: str, label: str, coupling_scheme: CouplingScheme) -> None:
    ket = AngularKetBase.from_short_label(label, species)
    assert type(ket) is KET_CLASSES[coupling_scheme]
    assert ket.get_short_label() == label


@pytest.mark.parametrize(
    ("ket_class", "species", "label"),
    [
        (AngularKetLS, "Rb", "60p_3/2 m=1/2"),
        (AngularKetLS, "Sr88", "5s 60d 3D3"),
        (AngularKetJJ, "Sr88", "5s_1/2 60p_3/2 J=1 m=0"),
        (AngularKetFJ, "Yb171", "6s_1/2(Fc=1) 60p_3/2 F=3/2"),
        (AngularKetBase, "Rb", "60p_3/2"),
        (AngularKetBase, "Sr88", "5s 60d 3D3"),
    ],
)
def test_from_short_label_with_n(ket_class: type[AngularKetBase[Any]], species: str, label: str) -> None:
    """The angular ket does not know n, so the label must contain the placeholder "n" instead of a number."""
    with pytest.raises(ValueError, match="use the placeholder 'n' instead") as excinfo:
        ket_class.from_short_label(label, species)
    assert str(excinfo.value).count("placeholder") == 1  # only one error, even if several coupling schemes match


def test_short_label_without_species() -> None:
    """Kets of divalent atoms created without a species have no n_c, which is required for the short label."""
    ket = AngularKetJJ(i_c=0, s_c=0.5, l_r=1, j_c=0.5, j_r=1.5, j_tot=2)
    assert ket.n_c is None
    with pytest.raises(ValueError, match="required field 'n_c'"):
        ket.get_short_label()


@pytest.mark.parametrize(
    ("ket_class", "label", "species"),
    [
        (AngularKetLS, "5s_1/2 np_3/2 J=1", "Sr88"),  # JJ label
        (AngularKetJJ, "5s np 3P1", "Sr88"),  # LS label
        (AngularKetLS, "5s np 3P1", "Rb"),  # divalent label for an alkali atom
        (AngularKetLS, "np_3/2", "Sr88"),  # alkali label for a divalent atom
        (AngularKetLS, "5s np 3P1 m=0 F=1", "Sr88"),  # wrong order
        (AngularKetLS, "5s nk 3P1", "Sr88"),  # no valid spectroscopic letter
        (AngularKetLS, "d nd 1S0", "Yb174"),  # n_c missing
        (AngularKetJJ, "s_1/2 np_3/2 J=1", "Sr88"),  # n_c missing
        (AngularKetBase, "5s np 3P1", "Rb"),
        (AngularKetLS, "5s p 3P1", "Sr88"),  # the n in front of the rydberg electron is missing
    ],
)
def test_from_short_label_invalid_format(ket_class: type[AngularKetBase[Any]], label: str, species: str) -> None:
    with pytest.raises(ValueError, match="does not match"):
        ket_class.from_short_label(label, species)


@pytest.mark.parametrize(
    ("ket_class", "label", "species"),
    [
        (AngularKetLS, "5s np 3P3", "Sr88"),  # j_tot not allowed
        (AngularKetLS, "np_3/2", "Yb171_ion"),  # F missing for a non-zero nuclear spin
        (AngularKetFJ, "6s_1/2(Fc=2) np_3/2 F=3/2", "Yb171"),  # f_c not allowed
        (AngularKetFJ, "6s_1/2 np_3/2 F=3/2", "Yb171"),  # Fc missing for a non-zero nuclear spin
    ],
)
def test_from_short_label_invalid_quantum_numbers(
    ket_class: type[AngularKetBase[Any]], label: str, species: str
) -> None:
    with pytest.raises(ValueError, match=r"spin addition rule|Unknown quantum numbers"):
        ket_class.from_short_label(label, species)


def test_short_label_unknown() -> None:
    """Kets with unknown quantum numbers are labeled by a leading "?", their label (if any), parity, F and m."""
    ket = AngularKetFJ(
        species="Yb171", l_c=Unknown, f_tot=1.5, m=0.5, parity=-1, label="4f13 5d2 6s 3P2", allow_unknown=True
    )
    label = "? 4f13 5d2 6s 3P2 parity=-1 F=3/2 m=1/2"
    assert ket.get_short_label() == label
    assert AngularKetFJ.from_short_label(label, "Yb171") == ket

    ket = AngularKetFJ(species="Yb174", l_c=Unknown, f_tot=0, parity=1, allow_unknown=True)
    assert ket.get_short_label() == "? parity=+1 F=0"
    assert AngularKetFJ.from_short_label("? parity=+1 F=0", "Yb174") == ket

    # the leading "?" distinguishes the label from that of a ket with known quantum numbers
    ket_ls = AngularKetLS(
        species="Sr88", l_c=Unknown, f_tot=1, parity=-1, label="5s np 3P1 label=x", allow_unknown=True
    )
    assert ket_ls.get_short_label() == "? 5s np 3P1 label=x parity=-1 F=1"
    assert AngularKetBase.from_short_label("? 5s np 3P1 label=x parity=-1 F=1", "Sr88") == ket_ls

    # a label "?" is not confused with a missing label
    ket = AngularKetFJ(species="Yb174", l_c=Unknown, f_tot=0, parity=1, label="?", allow_unknown=True)
    assert ket.get_short_label() == "? ? parity=+1 F=0"
    assert AngularKetFJ.from_short_label("? ? parity=+1 F=0", "Yb174") == ket


@pytest.mark.parametrize(
    "label", ["? 4f13 5d2 6s 3P2 F=3/2", "? 4f13 5d2 6s 3P2 parity=-1", "4f13 5d2 6s 3P2 parity=-1 F=3/2"]
)
def test_from_short_label_unknown_invalid(label: str) -> None:
    """For kets with unknown quantum numbers the leading "?", parity and F are required."""
    with pytest.raises(ValueError, match="does not match"):
        AngularKetFJ.from_short_label(label, "Yb171")
