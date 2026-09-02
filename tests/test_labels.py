from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pytest
from rydstate import RydbergStateSQDT, RydbergStateSQDTAlkali, RydbergStateSQDTDivalent
from rydstate.angular import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import Unknown, format_quantum_number, get_spectroscopic_letter
from rydstate.radial import RadialDummy, RadialKet
from rydstate.rydberg_state import RydbergKet, RydbergState
from rydstate.species import get_element_properties
from rydstate.species.potential import get_potential_class

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import CouplingScheme


def _create_rydberg_ket(species: str, angular: AngularKetBase[Any], nu: float, n: int | None = None) -> RydbergKet:
    """Create a RydbergKet without going through a (slow) MQDT model."""
    if angular.contains_unknown:  # e.g. a perturber channel of an MQDT model
        radial = RadialDummy(1.0, nu, element_properties=get_element_properties(species))
        return RydbergKet(species, angular, radial, n=n)
    potential = get_potential_class(species)(angular.l_r)
    return RydbergKet(species, angular, RadialKet(nu, potential), n=n)


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"species": "Rb", "n": 50, "l": 0, "j": 0.5, "m": 0.5}, "Rb:50S_1/2,m=1/2"),
        ({"species": "Rb", "n": 60, "l": 1, "j": 1.5}, "Rb:60P_3/2"),
        ({"species": "Rb", "n": 60, "l": 3, "j": 2.5, "m": 1.5}, "Rb:60F_5/2,m=3/2"),
        # large half-integer quantum numbers are formatted as decimals instead of fractions
        ({"species": "Cs", "n": 60, "l": 5, "j": 5.5, "m": -1.5}, "Cs:60H_5.5,m=-3/2"),
        # l values without a spectroscopic letter fall back to "(l=...)", which is uppercased here as well
        ({"species": "Rb", "n": 60, "l": 7, "j": 7.5, "m": 0.5}, "Rb:60(L=7)_7.5,m=1/2"),
    ],
)
def test_get_label_alkali(kwargs: dict[str, Any], label: str) -> None:
    state = RydbergStateSQDTAlkali(**kwargs)
    assert state.get_label("raw") == label
    assert str(state) == f"|{label}⟩"


def test_get_label_alkali_jj() -> None:
    """For alkali atoms j_tot = j_r, so the label does not depend on the coupling scheme."""
    angular = AngularKetJJ(species="Rb", l_r=1, j_r=1.5, j_tot=1.5, m=0.5)
    assert RydbergStateSQDT("Rb", 60, angular=angular).get_label("raw") == "Rb:60P_3/2,m=1/2"


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"species": "Sr88", "n": 60, "l": 0, "s": 0, "j": 0, "m": 0}, "Sr88:S=0,60S_0,m=0"),
        ({"species": "Sr88", "n": 61, "l": 2, "s": 1, "j": 1, "m": 0}, "Sr88:S=1,61D_1,m=0"),
        ({"species": "Yb174", "n": 55, "l": 1, "s": 1, "j": 2, "m": -1}, "Yb174:S=1,55P_2,m=-1"),
        ({"species": "Yb174", "n": 55, "l": 3, "s": 0, "j": 3}, "Yb174:S=0,55F_3"),
    ],
)
def test_get_label_divalent_ls(kwargs: dict[str, Any], label: str) -> None:
    state = RydbergStateSQDTDivalent(**kwargs)
    assert state.get_label("raw") == label
    assert str(state) == f"|{label}⟩"


@pytest.mark.parametrize(
    ("ket_kwargs", "label"),
    [
        # the inner channels "6pnp 1S0" and "6pnp 3P0" of the Yb174 MQDT model
        ({"l_c": 1, "l_r": 1, "l_tot": 0, "s_tot": 0, "j_tot": 0}, "Yb174:S=0,(6p,[12.3]p)S_0"),
        ({"l_c": 1, "l_r": 1, "l_tot": 1, "s_tot": 1, "j_tot": 0}, "Yb174:S=1,(6p,[12.3]p)P_0"),
    ],
)
def test_get_label_divalent_ls_excited_core(ket_kwargs: dict[str, Any], label: str) -> None:
    """For an excited core (l_c != 0) both shells are shown as (n_c l_c, n l)."""
    angular = AngularKetLS(species="Yb174", **ket_kwargs)
    assert _create_rydberg_ket("Yb174", angular, 12.34).get_label("raw") == label


@pytest.mark.parametrize(
    ("species", "n", "ket_kwargs", "label"),
    [
        ("Sr88", 60, {"l_r": 1, "j_c": 0.5, "j_r": 1.5, "j_tot": 1, "m": 0}, "Sr88:(5s_1/2,60p_3/2),J=1,m=0"),
        ("Yb174", 60, {"l_r": 3, "j_c": 0.5, "j_r": 2.5, "j_tot": 3}, "Yb174:(6s_1/2,60f_5/2),J=3"),
    ],
)
def test_get_label_divalent_jj(species: str, n: int, ket_kwargs: dict[str, Any], label: str) -> None:
    angular = AngularKetJJ(species=species, **ket_kwargs)
    assert _create_rydberg_ket(species, angular, 57.3, n=n).get_label("raw") == label


@pytest.mark.parametrize(
    ("species", "angular", "n", "label"),
    [
        # one valence electron and a nuclear spin (e.g. the ions used as MQDT cores): F is appended
        (
            "Yb171_ion",
            AngularKetLS(species="Yb171_ion", l_r=1, j_tot=1.5, f_tot=1, m=0),
            60,
            "Yb171_ion:60P_3/2,F=1,m=0",
        ),
        # divalent atom with a nuclear spin
        (
            "Sr87",
            AngularKetLS(species="Sr87", l_r=2, s_tot=1, j_tot=1, f_tot=5.5, m=0.5),
            60,
            "Sr87:S=1,60D_1,F=5.5,m=1/2",
        ),
    ],
)
def test_get_label_with_nuclear_spin(species: str, angular: AngularKetBase[Any], n: int, label: str) -> None:
    assert _create_rydberg_ket(species, angular, 57.3, n=n).get_label("raw") == label


@pytest.mark.parametrize(
    ("species", "angular", "nu", "label"),
    [
        ("Rb", AngularKetLS(species="Rb", l_r=1, j_tot=1.5, m=0.5), 47.4, "Rb:[47.4]P_3/2,m=1/2"),
        ("Sr88", AngularKetLS(species="Sr88", l_r=1, s_tot=0, j_tot=1, m=0), 57.32, "Sr88:S=0,[57.3]P_1,m=0"),
        (
            "Sr88",
            AngularKetJJ(species="Sr88", l_r=1, j_c=0.5, j_r=1.5, j_tot=1, m=0),
            57.32,
            "Sr88:(5s_1/2,[57.3]p_3/2),J=1,m=0",
        ),
        (
            "Yb171",
            AngularKetFJ(species="Yb171", l_r=1, j_c=0.5, f_c=1, j_r=1.5, f_tot=1.5, m=0.5),
            59.0586,
            "Yb171:(6s_1/2,[59.1]p_3/2),f_c=1,F=3/2,m=1/2",
        ),
        (
            # for a vanishing nuclear spin, f_c = j_c and j_tot is well defined, so J=j_tot is used instead of f_c
            "Yb174",
            AngularKetFJ(species="Yb174", l_r=1, j_c=0.5, f_c=0.5, j_r=1.5, f_tot=1, m=0),
            59.0,
            "Yb174:(6s_1/2,[59.0]p_3/2),J=1,m=0",
        ),
        (
            "Yb171",
            AngularKetFJ(species="Yb171", f_tot=1.5, m=0.5, parity=-1, label="4f13 5d 6snl a", allow_unknown=True),
            1.8084,
            "Yb171:nu=1.8,4f13 5d 6snl a,F=3/2,m=1/2",
        ),
        (
            # channels with unknown quantum numbers and without a label
            "Yb171",
            AngularKetFJ(species="Yb171", f_tot=1.5, m=0.5, parity=-1, allow_unknown=True),
            1.8084,
            "Yb171:nu=1.8,angular=?,F=3/2,m=1/2",
        ),
    ],
)
def test_get_label_rydberg_ket(species: str, angular: AngularKetBase[Any], nu: float, label: str) -> None:
    ket = _create_rydberg_ket(species, angular, nu)
    assert ket.get_label("raw") == label
    assert str(ket) == f"|{label}⟩"


def test_get_label_unknown_channel_with_n() -> None:
    """A ket with unknown quantum numbers is identified by nu and its label, even if n is given."""
    angular = AngularKetFJ(species="Yb171", f_tot=1.5, m=0.5, parity=-1, label="4f13 5d 6snl a", allow_unknown=True)
    ket = _create_rydberg_ket("Yb171", angular, 1.8084, n=2)
    assert ket.get_label("raw") == "Yb171:nu=1.8,4f13 5d 6snl a,F=3/2,m=1/2"


def test_get_label_rydberg_ket_without_nu() -> None:
    """Radial wavefunctions without a well defined nu (e.g. a superposition of different nu) use [?]."""
    angular = AngularKetLS(species="Sr88", l_r=1, s_tot=0, j_tot=1, m=0)
    potential = get_potential_class("Sr88")(1)
    radial = RadialKet(10.35, potential) + RadialKet(11.35, potential)  # a sum of different nu has no nu
    assert radial.nu is None
    assert RydbergKet("Sr88", angular, radial).get_label("raw") == "Sr88:S=0,[?]P_1,m=0"


def test_get_label_rydberg_ket_with_rescaled_radial() -> None:
    """Scaling a radial wavefunction (or adding wavefunctions with the same nu) keeps nu."""
    angular = AngularKetLS(species="Sr88", l_r=1, s_tot=0, j_tot=1, m=0)
    radial_ket = RadialKet(10.35, get_potential_class("Sr88")(1))
    for radial in [radial_ket * 1.0, radial_ket * 0.5 + radial_ket * 0.5]:
        assert radial.nu == 10.35
        assert RydbergKet("Sr88", angular, radial).get_label("raw") == "Sr88:S=0,[10.3]P_1,m=0"


@pytest.mark.parametrize(
    ("coupling_scheme", "label"),
    [
        ("LS", "Sr88:S=1,60D_1,m=0"),  # the state is already given in LS
        ("JJ", "-Sr88:(5s_1/2,[{nu:.1f}]d_3/2),J=1,m=0"),
        ("FJ", "-Sr88:(5s_1/2,[{nu:.1f}]d_3/2),J=1,m=0"),
    ],
)
def test_get_label_after_to_coupling_scheme(coupling_scheme: CouplingScheme, label: str) -> None:
    """Changing the coupling scheme recombines the radial wavefunctions, but keeps their (common) nu."""
    state = RydbergStateSQDTDivalent("Sr88", 60, l=2, s=1, j=1, m=0)
    converted = state.to_coupling_scheme(coupling_scheme)
    assert converted.get_label("raw") == label.format(nu=state.nu)


def test_get_label_state_and_ket_agree() -> None:
    """The label of an SQDT state and of its single rydberg ket only differ in the n / nu part."""
    state = RydbergStateSQDTDivalent("Sr88", 60, l=2, s=1, j=1, m=0)
    ket = state.rydberg_kets[0]
    assert state.get_label("raw") == "Sr88:S=1,60D_1,m=0"
    # the rydberg ket of an SQDT state knows its n, so it uses the same label as the state
    assert ket.get_label() == state.get_label()
    assert str(ket) == ket.get_label()
    # the same ket without n uses nu instead
    assert RydbergKet("Sr88", ket.angular, ket.radial).get_label("raw") == f"Sr88:S=1,[{state.nu:.1f}]D_1,m=0"


@pytest.mark.parametrize(
    ("fmt", "label"),
    [
        ("raw", "Rb:60P_3/2,m=1/2"),
        ("ket", "|Rb:60P_3/2,m=1/2⟩"),
        ("bra", "⟨Rb:60P_3/2,m=1/2|"),
    ],
)
def test_get_label_fmt(fmt: Literal["raw", "ket", "bra"], label: str) -> None:
    """The fmt argument selects between the raw label and the bra-ket notation (default: ket)."""
    state = RydbergStateSQDTAlkali("Rb", n=60, l=1, j=1.5, m=0.5)
    assert state.get_label(fmt) == label
    assert state.rydberg_kets[0].get_label(fmt) == label


def test_str_multi_channel_state() -> None:
    """A multi-channel state (like an MQDT state) is printed as a sum of its rydberg ket labels."""
    kets = [
        _create_rydberg_ket("Sr88", AngularKetJJ(species="Sr88", l_r=1, j_c=0.5, j_r=j_r, j_tot=1, m=0), 28.5)
        for j_r in [1.5, 0.5]
    ]
    state = RydbergState("Sr88", [0.6, 0.8], kets, nu=28.5, energy_au=-0.001)
    assert str(state) == ("0.6*|Sr88:(5s_1/2,[28.5]p_3/2),J=1,m=0⟩ + 0.8*|Sr88:(5s_1/2,[28.5]p_1/2),J=1,m=0⟩")
    assert state.get_label("raw") == ("0.6*Sr88:(5s_1/2,[28.5]p_3/2),J=1,m=0 + 0.8*Sr88:(5s_1/2,[28.5]p_1/2),J=1,m=0")


def test_str_multi_channel_state_negative_coefficient() -> None:
    """Negative coefficients are printed as a minus sign instead of a '+ -'."""
    kets = [
        _create_rydberg_ket("Sr88", AngularKetJJ(species="Sr88", l_r=1, j_c=0.5, j_r=j_r, j_tot=1, m=0), 28.5)
        for j_r in [1.5, 0.5]
    ]
    state = RydbergState("Sr88", [0.6, -0.8], kets, nu=28.5, energy_au=-0.001)
    assert state.get_label("raw") == ("0.6*Sr88:(5s_1/2,[28.5]p_3/2),J=1,m=0 - 0.8*Sr88:(5s_1/2,[28.5]p_1/2),J=1,m=0")


@pytest.mark.parametrize(("l", "letter"), [(0, "s"), (1, "p"), (2, "d"), (3, "f"), (4, "g"), (7, "(l=7)")])
def test_get_spectroscopic_letter(l: int, letter: str) -> None:
    assert get_spectroscopic_letter(l) == letter


def test_get_spectroscopic_letter_unknown() -> None:
    assert get_spectroscopic_letter(Unknown) == "(l=?)"


@pytest.mark.parametrize("invalid_l", [0.5, -1])
def test_get_spectroscopic_letter_invalid(invalid_l: float) -> None:
    with pytest.raises(ValueError, match="Invalid orbital angular momentum quantum number"):
        get_spectroscopic_letter(invalid_l)  # type: ignore [arg-type]


@pytest.mark.parametrize(
    ("value", "string"),
    [
        (1.5, "3/2"),
        (-0.5, "-1/2"),
        (2.0, "2"),
        (0, "0"),
        (-3, "-3"),
        # half-integer values with |value| >= 5 are formatted as decimals, since fractions get hard to read there
        (4.5, "9/2"),
        (-4.5, "-9/2"),
        (5.5, "5.5"),
        (-5.5, "-5.5"),
        (12.5, "12.5"),
        (-12.5, "-12.5"),
    ],
)
def test_format_quantum_number(value: float, string: str) -> None:
    assert format_quantum_number(value) == string


@pytest.mark.parametrize("invalid_value", [0.25, -1.75])
def test_format_quantum_number_invalid(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="Invalid quantum number"):
        format_quantum_number(invalid_value)
