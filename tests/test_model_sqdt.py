from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest
import rydstate.species.rubidium
from rydstate.angular import AngularKetJJ, AngularKetLS
from rydstate.qdt import Model, ModelSetSQDT, ModelSQDT, ModelSQDTBase, ModelSQDTNist, get_sqdt, register_model_set_sqdt
from rydstate.qdt.sqdt.model_set_sqdt import _SQDT_MODEL_SETS
from rydstate.qdt.sqdt.model_sqdt_nist import _load_nist_energies_by_channel
from rydstate.rydberg_state import RydbergStateSQDTAlkali


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


def test_n_range() -> None:
    """The default n range is (1, inf), an infinite n_max can also be given as the string "inf"."""
    assert ModelSQDT("Rb", "ns_1/2", [3.13]).n_range == (1, math.inf)
    assert ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(10, "inf")).n_range == (10, math.inf)
    assert ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(10, 10)).n_range == (10, 10)
    with pytest.raises(ValueError, match="n_range must be"):
        ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(10, "infinity"))  # type: ignore [arg-type]
    with pytest.raises(ValueError, match="n_range must be"):
        ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(10, 20.5))
    with pytest.raises(ValueError, match="n_range must be"):
        ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(-math.inf, 10))  # type: ignore [arg-type]
    with pytest.raises(ValueError, match="n_min must not be larger than n_max"):
        ModelSQDT("Rb", "ns_1/2", [3.13], n_range=(20, 10))


def test_quantum_defects_are_checked() -> None:
    with pytest.raises(ValueError, match="must not be an empty list"):
        ModelSQDT("Rb", "ns_1/2", [])


def test_calc_nui() -> None:
    model = ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787])
    n = 50
    assert model.calc_nui(n) == pytest.approx(n - 3.1311807 - 0.1787 / (n - 3.1311807) ** 2)


def test_calc_energy() -> None:
    """The energy of a state is the ionization threshold plus the binding energy of nui = n - delta(n)."""
    model = ModelSQDT("Cs", "np_3/2", [3.5590676, 0.37469])
    n = 50
    reduced_mass_au = model.element_properties.reduced_mass_au
    binding_energy_au = -0.5 * reduced_mass_au / model.calc_nui(n) ** 2
    assert model.calc_binding_energy_au(n=n) == pytest.approx(binding_energy_au, rel=1e-14)
    assert model.calc_energy_au(n=n) == pytest.approx(model.ionization_threshold_au + binding_energy_au, rel=1e-14)
    state = RydbergStateSQDTAlkali("Cs", n=n, l=1, j=1.5)
    assert get_sqdt("Cs").get_model(n, state.angular).calc_energy_au(n=n) == pytest.approx(state.get_energy("a.u."))
    # n is keyword-only, so that a nu meant for ModelMQDT.calc_energy_au(nu) is not taken as n
    with pytest.raises(TypeError):
        model.calc_energy_au(48.4)  # type: ignore [misc, arg-type]
    with pytest.raises(TypeError):
        model.calc_energy_au(nu=48.4)  # type: ignore [call-arg]


def test_model_set_uses_model_of_n_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model set picks the model of the channel, whose n range contains n."""
    sqdt = get_sqdt("Rb")  # note: this is a process wide cached instance, so we have to clean up carefully
    low_n = ModelSQDT("Rb", "ns_1/2", [3.1], n_range=(1, 30))
    high_n = ModelSQDT("Rb", "ns_1/2", [3.2], n_range=(31, 100))
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="Rb")
    with monkeypatch.context() as patch:
        patch.setattr(sqdt, "models", [low_n, high_n])
        assert sqdt.get_models(angular) == [low_n, high_n]
        # the models are found independent of m
        angular_m = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, m=-0.5, species="Rb")
        assert sqdt.get_models(angular_m) == [low_n, high_n]
        assert sqdt.calc_nui(50, angular_m) == pytest.approx(50 - 3.2)
        # kets in other coupling schemes are converted to LS by calc_nui, but not by get_models
        angular_jj = AngularKetJJ(l_r=0, j_r=0.5, j_tot=0.5, f_tot=0.5, species="Rb")
        assert sqdt.calc_nui(50, angular_jj) == pytest.approx(50 - 3.2)
        with pytest.raises(ValueError, match="must be in LS coupling"):
            sqdt.get_models(angular_jj)
        assert sqdt.calc_nui(20, angular) == pytest.approx(20 - 3.1)
        assert sqdt.calc_nui(50, angular) == pytest.approx(50 - 3.2)
        with pytest.raises(ValueError, match="None of the SQDT models"):
            sqdt.calc_nui(120, angular)

        assert sqdt.get_model(20, angular) is low_n
        assert sqdt.get_model(30, angular) is low_n  # both bounds are inclusive
        assert sqdt.get_model(31, angular) is high_n
        assert sqdt.get_model(100, angular) is high_n
        assert sqdt.get_model(50, angular_m) is high_n

        # channels without a model have a vanishing quantum defect
        angular_p = AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="Rb")
        assert sqdt.get_models(angular_p) == []
        trivial_model = sqdt.get_model(20, angular_p)
        assert isinstance(trivial_model, ModelSQDT)
        assert trivial_model.quantum_defects == (0.0,)
        assert sqdt.get_model(30, angular_p) is trivial_model
        assert sqdt.calc_nui(20, angular_p) == 20


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
ALL_SQDT_MODELS = [(model_set, model) for model_set in ALL_SQDT_MODEL_SETS for model in model_set.models]
RUBIDIUM_NIST_DATA_FILE = Path(rydstate.species.rubidium.__file__).parent / "nist_data.txt"


@pytest.mark.parametrize(
    ("model_set", "model"),
    ALL_SQDT_MODELS,
    ids=lambda x: f"{x.full_name} ({type(x).__name__})" if isinstance(x, ModelSQDTBase) else "",
)
def test_json_round_trip(model_set: ModelSetSQDT, model: ModelSQDTBase) -> None:
    """A model can be stored as json and recreated from it."""
    data = json.loads(json.dumps(model.to_dict()))
    directory = model_set._models_source  # noqa: SLF001
    assert isinstance(directory, Path)
    loaded = ModelSQDTBase.from_dict(data, directory)
    assert type(loaded) is type(model)
    assert loaded.species == model.species
    assert loaded.channel == model.channel
    assert loaded.reference == model.reference
    assert loaded.n_range == model.n_range
    if isinstance(model, ModelSQDT):
        assert isinstance(loaded, ModelSQDT)
        assert loaded.quantum_defects == model.quantum_defects
    if isinstance(model, ModelSQDTNist):
        assert isinstance(loaded, ModelSQDTNist)
        assert loaded.energies == model.energies
    assert loaded.to_dict() == model.to_dict()


def test_to_dict() -> None:
    model = ModelSQDT("Sr88", "5s np 3P1", [2.88265, 0.39398], reference=("Ref A", "Ref B"), n_range=(10, math.inf))
    assert model.to_dict() == {
        "model_class": "ModelSQDT",
        "species": "Sr88",
        "channel": "5s np 3P1",
        "quantum_defects": [2.88265, 0.39398],
        "reference": ["Ref A", "Ref B"],
        "n_range": [10, "inf"],
    }
    # the reference is always included, n_range only if it differs from its default value
    assert ModelSQDT("Rb", "ns_1/2", [3.1311807]).to_dict() == {
        "model_class": "ModelSQDT",
        "species": "Rb",
        "channel": "ns_1/2",
        "quantum_defects": [3.1311807],
        "reference": None,
    }


def test_from_dict_optional_keys() -> None:
    model = ModelSQDT.from_dict(
        {"model_class": "ModelSQDT", "species": "Rb", "channel": "ns_1/2", "quantum_defects": [3.1311807]}
    )
    assert model.reference is None
    assert model.n_range == (1, math.inf)


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
    """The json files are named after the channel (plus _nist for NIST models) and stored as written by to_dict.

    Only an additional comment is allowed in the json files.
    """
    directory = model_set._models_source  # noqa: SLF001
    assert isinstance(directory, Path)
    files = sorted(directory.glob("*.json"))
    assert len(files) == len(model_set.models)
    for file, model in zip(files, model_set.models, strict=True):
        suffix = "_nist" if isinstance(model, ModelSQDTNist) else ""
        assert file.name == model.name.replace("/", "_").replace(" ", "_") + suffix + ".json"
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
        register_model_set_sqdt(ModelSetSQDT("Rb", models=[]))


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


def test_nist_models_are_loaded() -> None:
    """The NIST levels are given as ModelSQDTNist models for low n, the quantum defects for high n.

    Usually the models switch at nu = 15, but for Rb ns the NIST levels are only used below the
    NIST levels missing from 13s to 38s, i.e. up to 12s.
    """
    sqdt = get_sqdt("Rb")
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="Rb")
    nist_model, qd_model = sqdt.get_models(angular)
    assert isinstance(nist_model, ModelSQDTNist)
    assert isinstance(qd_model, ModelSQDT)
    assert nist_model.name == "ns_1/2"
    assert nist_model.energies[5] == 0  # 5s ground state
    assert nist_model.n_range == (5, 12)
    assert qd_model.n_range == (13, math.inf)
    nist_model, qd_model = sqdt.get_models(AngularKetLS(l_r=1, j_tot=0.5, f_tot=0.5, species="Rb"))
    assert nist_model.n_range == (5, 17)  # 17p has nu = 14.3
    assert qd_model.n_range == (18, math.inf)


def test_nist_model_of_single_channel() -> None:
    """A NIST model can be loaded for a single channel, the NIST data file is only parsed once."""
    model = ModelSQDTNist.from_nist_file("Rb", "ns_1/2", RUBIDIUM_NIST_DATA_FILE, n_range=(5, 20))
    assert model.n_range == (5, 20)
    nist_model = get_sqdt("Rb").get_models(model.channel)[0]
    assert isinstance(nist_model, ModelSQDTNist)
    assert model.energies == nist_model.energies
    assert model.energies is not nist_model.energies
    hits = _load_nist_energies_by_channel.cache_info().hits
    ModelSQDTNist.from_nist_file("Rb", "ns_1/2", RUBIDIUM_NIST_DATA_FILE)
    assert _load_nist_energies_by_channel.cache_info().hits == hits + 1
    channel = AngularKetLS(l_r=20, j_tot=20.5, f_tot=20.5, species="Rb")
    with pytest.raises(ValueError, match="has no energy levels of this channel"):
        ModelSQDTNist.from_nist_file("Rb", channel, RUBIDIUM_NIST_DATA_FILE)


def test_nist_model_calc_energy() -> None:
    """The energy of a NIST model is its measured energy level."""
    model = ModelSQDTNist.from_nist_file("Rb", "np_3/2", RUBIDIUM_NIST_DATA_FILE)
    for n, energy_au in model.energies.items():
        assert model.calc_energy_au(n=n) == pytest.approx(energy_au, rel=1e-12)


def test_nist_model_json_round_trip(tmp_path: Path) -> None:
    """A NIST model is stored with the path of its NIST data file, relative paths are relative to the json file."""
    (tmp_path / "nist_data.txt").write_text(RUBIDIUM_NIST_DATA_FILE.read_text())
    model = ModelSQDTNist.from_nist_file("Rb", "np_1/2", tmp_path / "nist_data.txt", n_range=(5, 20))
    assert model.to_dict()["nist_data_file"] == (tmp_path / "nist_data.txt").as_posix()

    data = {
        "model_class": "ModelSQDTNist",
        "species": "Rb",
        "channel": "np_1/2",
        "reference": "NIST Atomic Spectra Database",
        "nist_data_file": "nist_data.txt",
        "n_range": [5, 20],
    }
    loaded = Model.from_dict(json.loads(json.dumps(data)), tmp_path)
    assert isinstance(loaded, ModelSQDTNist)
    assert loaded.energies == model.energies
    assert loaded.to_dict() == data
    # the reference is None by default and stays None in the round trip
    assert model.reference is None
    assert Model.from_dict(model.to_dict()).reference is None
    with pytest.raises(ValueError, match="only models loaded from a NIST data file"):
        ModelSQDTNist("Rb", "np_1/2", {5: 0.0}).to_dict()


def test_model_from_dict_dispatches_by_model_class() -> None:
    data = {"model_class": "ModelSQDT", "species": "Rb", "channel": "ns_1/2", "quantum_defects": [3.13]}
    assert isinstance(Model.from_dict(data), ModelSQDT)
    assert isinstance(ModelSQDTBase.from_dict(data), ModelSQDT)
    with pytest.raises(ValueError, match="Expected exactly one subclass of ModelSQDTBase named 'ModelEigenChannel'"):
        ModelSQDTBase.from_dict({**data, "model_class": "ModelEigenChannel"})
    with pytest.raises(ValueError, match="named 'ModelUnknown'"):
        Model.from_dict({**data, "model_class": "ModelUnknown"})
    with pytest.raises(NotImplementedError, match="ModelSQDTBase cannot be created from a dictionary"):
        Model.from_dict({**data, "model_class": "ModelSQDTBase"})


def test_load_models_with_nist_model(tmp_path: Path) -> None:
    """The models directory can contain NIST and quantum defect models, which are sorted by their n range."""
    qd_model = ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787], n_range=(46, math.inf))
    nist_model = ModelSQDTNist.from_nist_file("Rb", "ns_1/2", RUBIDIUM_NIST_DATA_FILE.resolve(), n_range=(5, 45))
    (tmp_path / "a_ns_1_2.json").write_text(json.dumps(qd_model.to_dict()))
    (tmp_path / "b_ns_1_2_nist.json").write_text(json.dumps(nist_model.to_dict()))
    model_set = ModelSetSQDT("Rb", models=tmp_path)
    loaded_nist_model, loaded_qd_model = model_set.get_models(qd_model.channel)
    assert isinstance(loaded_nist_model, ModelSQDTNist)
    assert isinstance(loaded_qd_model, ModelSQDT)
    assert model_set.get_model(40, qd_model.channel) is loaded_nist_model
    assert model_set.get_model(50, qd_model.channel) is loaded_qd_model
    # n = 25 is in the n range of the NIST model, but there is no NIST level for it
    assert model_set.get_model(25, qd_model.channel) is loaded_nist_model
    with pytest.raises(ValueError, match="no energy level for n=25"):
        model_set.calc_nui(25, qd_model.channel)
    with pytest.raises(ValueError, match="None of the SQDT models"):
        model_set.get_model(4, qd_model.channel)


def test_n_ranges_of_a_channel_must_not_overlap() -> None:
    """The n ranges are inclusive, so they overlap if they share a single n."""
    low_n = ModelSQDT("Rb", "ns_1/2", [3.1], n_range=(1, 30))
    high_n = ModelSQDT("Rb", "ns_1/2", [3.2], n_range=(30, math.inf))
    with pytest.raises(ValueError, match="overlap"):
        _ = ModelSetSQDT("Rb", models=[high_n, low_n]).models
    # adjacent n ranges and overlapping n ranges of different channels are fine
    other_channel = ModelSQDT("Rb", "np_1/2", [2.6])
    high_n = ModelSQDT("Rb", "ns_1/2", [3.2], n_range=(31, math.inf))
    assert ModelSetSQDT("Rb", models=[high_n, low_n, other_channel]).models == [high_n, low_n, other_channel]


@pytest.mark.parametrize(
    ("model_set", "model"),
    [(model_set, model) for model_set, model in ALL_SQDT_MODELS if isinstance(model, ModelSQDT)],
    ids=lambda x: x.full_name if isinstance(x, ModelSQDTBase) else "",
)
def test_ritz_formula_does_not_diverge_in_n_range(model_set: ModelSetSQDT, model: ModelSQDT) -> None:
    """At the lowest n of the n range, the quantum defect is still close to its leading term delta_0."""
    channel = model.channel
    n_min = next(
        n
        for n in itertools.count(model.n_min)
        if model_set.element_properties.is_allowed_shell(n, channel.l_r, channel.s_tot)
    )
    assert abs(model.calc_quantum_defect(n_min) - model.quantum_defects[0]) < 0.6


def test_model_set_uses_nist_model_in_its_n_range() -> None:
    """Within the n range of the NIST model the NIST levels are used, otherwise the quantum defects."""
    sqdt = get_sqdt("Rb")
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="Rb")
    nist_model, qd_model = sqdt.get_models(angular)
    assert sqdt.get_model(5, angular) is nist_model
    assert sqdt.calc_nui(5, angular) == nist_model.calc_nui(5)
    assert sqdt.calc_nui(5, angular) != pytest.approx(qd_model.calc_nui(5), abs=1e-3)
    with pytest.raises(ValueError, match="no energy level for n=1000"):
        nist_model.calc_nui(1000)
    assert sqdt.get_model(60, angular) is qd_model
    assert sqdt.calc_nui(60, angular) == qd_model.calc_nui(60)
    assert sqdt.get_model(12, angular) is nist_model
    assert sqdt.get_model(13, angular) is qd_model  # NIST levels missing from 13s to 38s
    assert sqdt.get_model(40, angular) is qd_model  # NIST level exists, but n is above the NIST n range


def test_missing_nist_level_in_n_range_raises() -> None:
    """A state in the n range of a NIST model without a NIST level is not available, e.g. Yb174 6s8d 1D2."""
    sqdt = get_sqdt("Yb174")
    angular = AngularKetLS(l_r=2, s_tot=0, j_tot=2, f_tot=2, species="Yb174")
    with pytest.raises(ValueError, match="no energy level for n=8"):
        sqdt.calc_nui(8, angular)


def test_states_between_nist_and_quantum_defect_models_raise() -> None:
    """The Rydberg-Ritz formula of Yb174 6sns 3S1 diverges for n < 20, so these states are not available."""
    sqdt = get_sqdt("Yb174")
    angular = AngularKetLS(l_r=0, s_tot=1, j_tot=1, f_tot=1, species="Yb174")
    assert isinstance(sqdt.get_model(13, angular), ModelSQDTNist)
    for n in range(14, 20):
        with pytest.raises(ValueError, match="None of the SQDT models"):
            sqdt.calc_nui(n, angular)
    assert sqdt.calc_nui(20, angular) == pytest.approx(16.02, abs=0.01)


def test_model_set_without_quantum_defects_uses_nist_models() -> None:
    """Without quantum defects, only the NIST levels in the n range of the NIST models are available.

    Only channels without any model (e.g. higher l) have zero quantum defects.
    """
    sqdt = get_sqdt("Yb174_ion")
    assert all(isinstance(model, ModelSQDTNist) for model in sqdt.models)
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="Yb174_ion")
    assert sqdt.calc_nui(6, angular) == sqdt.get_models(angular)[0].calc_nui(6)
    with pytest.raises(ValueError, match="None of the SQDT models"):
        sqdt.calc_nui(60, angular)
    angular_h = AngularKetLS(l_r=5, j_tot=4.5, f_tot=4.5, species="Yb174_ion")
    assert sqdt.get_models(angular_h) == []
    assert sqdt.calc_nui(60, angular_h) == 60


def test_model_set_without_models() -> None:
    """Without any models (e.g. for hydrogen), all channels have zero quantum defects."""
    sqdt = get_sqdt("H")
    assert sqdt.models == []
    angular = AngularKetLS(l_r=0, j_tot=0.5, f_tot=0.5, species="H")
    assert sqdt.calc_nui(60, angular) == 60
