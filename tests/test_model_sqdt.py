from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rydstate.angular import AngularKetJJ, AngularKetLS
from rydstate.qdt import ModelSetSQDT, ModelSQDT, get_sqdt, register_model_set_sqdt
from rydstate.qdt.sqdt.model_set_sqdt import _SQDT_MODEL_SETS


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

        assert sqdt.get_model(20, angular) is low_n
        assert sqdt.get_model(50, angular_m) is high_n

        # channels without a model have a vanishing quantum defect
        angular_p = AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="Rb")
        assert sqdt.get_models(angular_p) == []
        trivial_model = sqdt.get_model(20, angular_p)
        assert trivial_model.quantum_defects == [0.0]
        assert sqdt.get_model(30, angular_p) is trivial_model
        assert sqdt.calc_nui(20, angular_p, use_nist_data=False) == 20


def test_ionization_thresholds() -> None:
    """The model knows the ionization threshold of its channel and the reference ionization threshold."""
    model = ModelSQDT("Sr88", "5s np 3P1", [2.88265, 0.39398])
    element_properties = model.element_properties
    threshold_au = element_properties.get_ionization_threshold(model.channel.get_core_ket(), "a.u.")
    assert model.outer_channels == [model.channel]
    assert model.ionization_thresholds_au == [threshold_au]
    assert model.ionization_threshold_au == threshold_au
    assert model.reference_ionization_threshold_au == element_properties.get_reference_ionization_threshold("a.u.")


ALL_SQDT_MODEL_SETS = list(_SQDT_MODEL_SETS.values())
ALL_SQDT_MODELS = [model for model_set in ALL_SQDT_MODEL_SETS for model in model_set.models or []]


@pytest.mark.parametrize("model", ALL_SQDT_MODELS, ids=lambda model: model.full_name)
def test_json_round_trip(model: ModelSQDT) -> None:
    """A model can be stored as json and recreated from it."""
    data = json.loads(json.dumps(model.to_dict()))
    loaded = ModelSQDT.from_dict(data)
    assert loaded.species == model.species
    assert loaded.channel == model.channel
    assert loaded.quantum_defects == model.quantum_defects
    assert loaded.reference == model.reference
    assert loaded.nu_range == model.nu_range
    assert loaded.to_dict() == model.to_dict()


def test_to_dict() -> None:
    model = ModelSQDT("Sr88", "5s np 3P1", [2.88265, 0.39398], reference=("Ref A", "Ref B"), nu_range=(10, math.inf))
    assert model.to_dict() == {
        "model_class": "ModelSQDT",
        "species": "Sr88",
        "channel": "5s np 3P1",
        "quantum_defects": [2.88265, 0.39398],
        "reference": ["Ref A", "Ref B"],
        "nu_range": [10, "inf"],
    }
    # optional keys with their default value are not included
    assert ModelSQDT("Rb", "ns_1/2", [3.1311807]).to_dict() == {
        "model_class": "ModelSQDT",
        "species": "Rb",
        "channel": "ns_1/2",
        "quantum_defects": [3.1311807],
    }


def test_from_dict_optional_keys() -> None:
    model = ModelSQDT.from_dict(
        {"model_class": "ModelSQDT", "species": "Rb", "channel": "ns_1/2", "quantum_defects": [3.1311807]}
    )
    assert model.reference is None
    assert model.nu_range == (0, math.inf)


def test_from_dict_ignores_comment() -> None:
    """A comment in the json file is only meant for its reader and not stored in the model."""
    data = {"model_class": "ModelSQDT", "species": "Rb", "channel": "ns_1/2", "quantum_defects": [3.13], "comment": "x"}
    model = ModelSQDT.from_dict(data)
    assert not hasattr(model, "comment")
    assert "comment" not in model.to_dict()


def test_from_dict_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        ModelSQDT.from_dict(
            {"model_class": "ModelSQDT", "species": "Rb", "channel": "ns_1/2", "quantum_defect": [3.1311807]}
        )


def test_from_dict_wrong_model_class() -> None:
    with pytest.raises(ValueError, match="ModelSQDT was expected"):
        ModelSQDT.from_dict(
            {"model_class": "ModelEigenChannel", "species": "Rb", "channel": "ns_1/2", "quantum_defects": [3.13]}
        )


@pytest.mark.parametrize("model_set", ALL_SQDT_MODEL_SETS, ids=lambda model_set: model_set.species)
def test_model_files(model_set: ModelSetSQDT) -> None:
    """The json files of the models are named after their channel and stored as written by to_dict (plus a comment)."""
    directory = model_set._models_source  # noqa: SLF001
    if directory is None:
        assert model_set.models is None
        return
    assert isinstance(directory, Path)
    files = sorted(directory.glob("*.json"))
    assert len(files) == len(model_set.models or [])
    for file, model in zip(files, model_set.models or [], strict=True):
        assert file.name == model.name.replace("/", "_").replace(" ", "_") + ".json"
        data = json.loads(file.read_text(encoding="utf-8"))
        data.pop("comment", None)
        assert data == model.to_dict()


def test_model_sets_are_registered_by_species() -> None:
    for species, model_set in _SQDT_MODEL_SETS.items():
        assert model_set.species == species
        assert get_sqdt(species) is model_set
    with pytest.raises(ValueError, match="No ModelSetSQDT registered"):
        get_sqdt("Unobtainium")


def test_register_model_set_sqdt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only explicitly registered model sets are returned by get_sqdt, and only one per species."""
    monkeypatch.delitem(_SQDT_MODEL_SETS, "Rb")
    model_set = ModelSetSQDT("Rb", models=[])
    with pytest.raises(ValueError, match="No ModelSetSQDT registered"):
        get_sqdt("Rb")
    assert register_model_set_sqdt(model_set) is model_set
    assert get_sqdt("Rb") is model_set
    with pytest.raises(ValueError, match="already registered"):
        register_model_set_sqdt(ModelSetSQDT("Rb"))


def test_models_are_loaded_lazily(tmp_path: Path) -> None:
    """The json files are only read when the models are needed, which keeps creating the model sets cheap."""
    model_set = ModelSetSQDT("Rb", models=tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _ = model_set.models


def test_models_given_directly() -> None:
    """Instead of a directory, the models can also be given directly."""
    model = ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787])
    model_set = ModelSetSQDT("Rb", models=[model])
    assert model_set.models == [model]
    assert model_set.get_model(50, model.channel) is model
    assert ModelSetSQDT("Rb").models is None


def test_models_must_be_of_the_species(tmp_path: Path) -> None:
    model = ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787])
    (tmp_path / "ns_1_2.json").write_text(json.dumps(model.to_dict()))
    assert ModelSetSQDT.load_models(tmp_path)[0].to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="is not a model of the species Cs"):
        _ = ModelSetSQDT("Cs", models=tmp_path).models
    with pytest.raises(ValueError, match="is not a model of the species Cs"):
        _ = ModelSetSQDT("Cs", models=[model]).models


def test_load_models_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ModelSetSQDT.load_models(tmp_path / "missing")
