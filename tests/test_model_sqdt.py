import math

import pytest
from rydstate.angular import AngularKetJJ, AngularKetLS
from rydstate.qdt import ModelSQDT, get_sqdt


def test_channel_from_short_label() -> None:
    """A channel given as short label is converted into the same angular ket."""
    model_label = ModelSQDT("Sr88", "5s np 3P1", [2.88265, 0.39398])
    model_ket = ModelSQDT("Sr88", AngularKetLS(l_r=1, s_tot=1, j_tot=1, f_tot=1, species="Sr88"), [2.88265, 0.39398])
    assert model_label.channel == model_ket.channel
    assert (model_label.channel.l_r, model_label.channel.j_tot, model_label.channel.s_tot) == (1, 1, 1)
    assert (model_label.f_tot, model_label.parity) == (1, -1)
    assert model_label.full_name == "Sr88 5s np 3P1"


def test_channel_is_converted_to_ls() -> None:
    """A channel given in another coupling scheme is converted to LS coupling."""
    model = ModelSQDT("Rb", AngularKetJJ(l_r=1, j_r=1.5, j_tot=1.5, f_tot=1.5, species="Rb"), [2.6416737, 0.295])
    assert model.channel.coupling_scheme == "LS"
    assert model.name == "np_3/2"


def test_channel_must_be_single_ls_ket() -> None:
    with pytest.raises(ValueError, match="single LS ket"):
        ModelSQDT("Sr88", "5s_1/2 np_3/2 J=1", [2.88])


def test_channel_must_not_set_m() -> None:
    with pytest.raises(ValueError, match="must be NotSet"):
        ModelSQDT("Rb", AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, m=0.5, species="Rb"), [3.13])


def test_nu_range() -> None:
    """The default nu range is (0, inf), an infinite nu_max can also be given as the string "inf"."""
    assert ModelSQDT("Rb", "ns_1/2", [3.13]).nu_range == (0, math.inf)
    assert ModelSQDT("Rb", "ns_1/2", [3.13], nu_range=(10, "inf")).nu_range == (10, math.inf)
    with pytest.raises(ValueError, match='nu_max must be a number or the string "inf"'):
        ModelSQDT("Rb", "ns_1/2", [3.13], nu_range=(10, "infinity"))  # type: ignore [arg-type]
    with pytest.raises(ValueError, match="nu_min must be finite"):
        ModelSQDT("Rb", "ns_1/2", [3.13], nu_range=(-math.inf, 10))


def test_quantum_defects_are_checked() -> None:
    with pytest.raises(ValueError, match="must not be an empty list"):
        ModelSQDT("Rb", "ns_1/2", [])


def test_calc_nui() -> None:
    model = ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787])
    n = 50
    assert model.calc_nui(n) == pytest.approx(n - 3.1311807 - 0.1787 / (n - 3.1311807) ** 2)


def test_model_set_uses_model_of_nu_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model set picks the model of the channel, whose nu range contains the state."""
    sqdt = get_sqdt("Rb")  # note: this is a process wide cached instance, so we have to clean up carefully
    low_n = ModelSQDT("Rb", "ns_1/2", [3.1], nu_range=(0, 30))
    high_n = ModelSQDT("Rb", "ns_1/2", [3.2], nu_range=(30, 100))
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="Rb")
    with monkeypatch.context() as patch:
        patch.setattr(sqdt, "models", [low_n, high_n])
        assert sqdt.get_models(angular) == [low_n, high_n]
        # the models are found independent of m
        angular_m = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, m=-0.5, species="Rb")
        assert sqdt.get_models(angular_m) == [low_n, high_n]
        assert sqdt.calc_nui(50, angular_m, use_nist_data=False) == pytest.approx(50 - 3.2)
        # kets in other coupling schemes are converted to LS by calc_nui, but not by get_models
        angular_jj = AngularKetJJ(l_r=0, j_r=0.5, j_tot=0.5, f_tot=0.5, species="Rb")
        assert sqdt.calc_nui(50, angular_jj, use_nist_data=False) == pytest.approx(50 - 3.2)
        with pytest.raises(ValueError, match="must be in LS coupling"):
            sqdt.get_models(angular_jj)
        assert sqdt.calc_nui(20, angular, use_nist_data=False) == pytest.approx(20 - 3.1)
        assert sqdt.calc_nui(50, angular, use_nist_data=False) == pytest.approx(50 - 3.2)
        with pytest.raises(ValueError, match="None of the SQDT models"):
            sqdt.calc_nui(120, angular, use_nist_data=False)

        # channels without a model have a vanishing quantum defect
        angular_p = AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="Rb")
        assert sqdt.get_models(angular_p) == []
        assert sqdt.calc_nui(20, angular_p, use_nist_data=False) == 20
