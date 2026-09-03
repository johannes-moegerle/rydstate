from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
import pytest
from rydstate.angular import AngularKetFJ
from rydstate.angular.utils import is_unknown
from rydstate.species import MQDT, FModel, get_all_subclasses, get_element_properties, get_mqdt

ALL_MODELS = [cls(get_mqdt(cls.species)) for cls in FModel.__subclasses__() if getattr(cls, "name", None) is not None]
ALL_MQDTS = [cls() for cls in get_all_subclasses(MQDT)]


@pytest.fixture(params=ALL_MODELS, ids=lambda cls: cls.full_name)
def model(request: pytest.FixtureRequest) -> FModel:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=ALL_MQDTS, ids=lambda mqdt: f"{mqdt.species}_{mqdt.tag}")
def mqdt(request: pytest.FixtureRequest) -> MQDT:
    return request.param  # type: ignore[no-any-return]


def test_all_models_discovered() -> None:
    """Sanity check: we should find at least 80 FModel subclasses."""
    assert len(ALL_MODELS) >= 80


def test_channel_count_consistency(model: FModel) -> None:
    """Inner channels, outer channels, and eigen quantum defects must have the same length."""
    n_inner = len(model.inner_channels)
    n_outer = len(model.outer_channels)
    n_defects = len(model.eigen_quantum_defects)
    assert n_inner == n_outer, f"{model.full_name}: inner_channels ({n_inner}) != outer_channels ({n_outer})"
    assert n_inner == n_defects, f"{model.full_name}: channels ({n_inner}) != eigen_quantum_defects ({n_defects})"


def test_nu_range_valid(model: FModel) -> None:
    """nu_range must be a 2-tuple with min < max."""
    nu_min, nu_max = model.nu_range
    assert nu_min < nu_max, f"{model.full_name}: nu_range {model.nu_range} has min >= max"
    assert nu_min > 0, f"{model.full_name}: nu_range min must be positive"


def test_f_tot_consistency(model: FModel) -> None:
    """All channels must have f_tot matching the model's f_tot."""
    for i, ch in enumerate(model.inner_channels):
        assert ch.f_tot == model.f_tot, (
            f"{model.full_name}: inner_channels[{i}].f_tot={ch.f_tot} != model.f_tot={model.f_tot}"
        )
    for i, och in enumerate(model.outer_channels):
        assert och.f_tot == model.f_tot, (
            f"{model.full_name}: outer_channels[{i}].f_tot={och.f_tot} != model.f_tot={model.f_tot}"
        )


def test_parity_consistency(model: FModel) -> None:
    """All channels must define the same parity, including channels with unknown orbital quantum numbers."""
    parities = [ch.parity for ch in model.inner_channels]
    parities.extend(och.parity for och in model.outer_channels)
    assert len(set(parities)) <= 1, f"{model.full_name}: channels have inconsistent parity"


def test_all_channels_have_ionization_threshold(model: FModel) -> None:
    """All channels must have ionization thresholds."""
    mqdt = model.mqdt
    try:
        for _i, ch in enumerate(model.outer_channels):
            mqdt.get_ionization_threshold(ch.get_core_ket())
    except ValueError:
        pytest.fail(
            f"{model.full_name}: outer_channels[{_i}] with core ket {ch.get_core_ket()} "
            "has no ionization threshold defined"
        )


def test_mixing_angles_indices_valid(model: FModel) -> None:
    """Mixing angle indices must refer to valid channel positions."""
    if model.mixing_angles is None:
        return
    n_channels = len(model.inner_channels)
    for entry in model.mixing_angles:
        i, j = entry[0], entry[1]
        assert 0 <= i < n_channels, f"{model.full_name}: mixing_angles index {i} out of range [0, {n_channels})"
        assert 0 <= j < n_channels, f"{model.full_name}: mixing_angles index {j} out of range [0, {n_channels})"
        assert i != j, f"{model.full_name}: mixing_angles has self-coupling ({i}, {j})"


def test_no_unknown_class_attributes(model: FModel) -> None:
    """Models must only define attributes known to FModel.

    Since all FModel fields have either a default or are only used if present (e.g. mixing_angles),
    a misspelled field name would otherwise be silently ignored.
    """
    known = set(FModel.__annotations__) | set(dir(FModel))
    unknown = {key for key in type(model).__dict__ if not key.startswith("_")} - known
    assert not unknown, f"{model.full_name}: unknown class attributes {sorted(unknown)} (misspelled field?)"


def test_model_name_contains_quantum_number(model: FModel) -> None:
    """Model name must contain F=X/Y or J=X matching the model's f_tot."""
    assert model.name is not None
    # Match F=X/Y or J=X patterns (integers and fractions)
    match = re.search(r"[FJ]=(\d+/?\d*)", model.name)
    assert match is not None, f"{model.full_name}: name '{model.name}' does not contain F=... or J=..."
    s = match.group(1)
    if "/" in s:
        num, den = s.split("/")
        f_val = int(num) / int(den)
    else:
        f_val = int(s)
    assert f_val == model.f_tot, f"{model.full_name}: name says F/J={f_val} but f_tot={model.f_tot}"


def test_reference_field_set(model: FModel) -> None:
    """Every model must have an explicit reference field (str or None)."""
    assert hasattr(model, "reference"), f"{model.full_name}: missing 'reference' field"


def test_model_name_unique(model: FModel) -> None:
    """Every model must have a unique combination of species and name (full_name)."""
    full_name = model.full_name
    duplicates = [m for m in ALL_MODELS if m.full_name == full_name]
    assert len(duplicates) == 1, f"{model.full_name}: {len(duplicates)} duplicate models found"


def test_species_field_set(model: FModel) -> None:
    """Every model must have a species field."""
    assert model.species is not None, f"{model.full_name}: species is None"


def test_eigen_quantum_defects_format(model: FModel) -> None:
    """Each eigen quantum defect entry must be a list/tuple of numeric values."""
    for i, defect in enumerate(model.eigen_quantum_defects):
        if isinstance(defect, (list, tuple)):
            for j, val in enumerate(defect):
                assert isinstance(val, (int, float)), (
                    f"{model.full_name}: eigen_quantum_defects[{i}][{j}] is {type(val)}, expected numeric"
                )
        else:
            assert isinstance(defect, (int, float)), (
                f"{model.full_name}: eigen_quantum_defects[{i}] is {type(defect)}, expected numeric or list"
            )


def test_at_least_one_real_channel(model: FModel) -> None:
    """Every model must have at least one non-dummy channel."""
    real_channels = [ch for ch in model.inner_channels if not ch.contains_unknown]
    assert len(real_channels) >= 1, f"{model.full_name}: no real (non-dummy) channels"


@pytest.mark.parametrize("channel_type", ["inner", "outer"])
def test_channels_are_orthonormal(model: FModel, channel_type: str) -> None:
    """The channels of a model must form an orthonormal set."""
    channels = model.inner_channels if channel_type == "inner" else model.outer_channels
    overlaps = np.array([[ket1.calc_reduced_overlap(ket2) for ket2 in channels] for ket1 in channels])
    msg = f"{model.full_name}: {channel_type} channels are not orthonormal"
    np.testing.assert_allclose(overlaps, np.eye(len(channels)), atol=1e-10, err_msg=msg)


def test_inner_outer_unitary(model: FModel) -> None:
    """The frame transformation matrix from inner to outer channels must be unitary."""
    unitary = model.calc_frame_transformation_outer_inner()
    msg = f"{model.full_name}: frame transformation (outer - inner) is not unitary"
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-10, err_msg=msg)

    rotation = model.calc_frame_transformation_inner_closecoupling(nu=30.5)
    msg = f"{model.full_name}: frame transformation (inner - closecoupling) is not unitary"
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(rotation.shape[0]), atol=1e-10, err_msg=msg)

    full = model.calc_frame_transformation(nu=30.5)
    msg = f"{model.full_name}: full frame transformation U=QR is not unitary"
    np.testing.assert_allclose(full.conj().T @ full, np.eye(full.shape[0]), atol=1e-10, err_msg=msg)


def test_all_models_found_by_get_mqdt_models(mqdt: MQDT) -> None:
    """Looping over all channels like BasisMQDT._init_models must find every model of the species."""
    element_properties = get_element_properties(mqdt.species)
    i_c = element_properties.i_c
    s_c = element_properties.s_c
    j_c = s_c
    s_r = 0.5

    known_l_r = [
        ch.l_r
        for model in ALL_MODELS
        if model.species == mqdt.species
        for ch in model.outer_channels
        if not is_unknown(ch.l_r)
    ]
    max_l_r = max(known_l_r)

    found_models: list[FModel] = []
    for l_r in range(max_l_r + 1):
        for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
            for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                    channel = AngularKetFJ(
                        l_r=l_r, j_r=float(j_r), f_c=float(f_c), f_tot=float(f_tot), species=mqdt.species
                    )
                    for model in mqdt.get_mqdt_models(channel):
                        if model not in found_models:
                            found_models.append(model)

    # FModel instances are not cached, so compare the models by their (unique) full_name
    found_model_names = [model.full_name for model in found_models]
    missing = [
        model.full_name
        for model in ALL_MODELS
        if model.species == mqdt.species and model.full_name not in found_model_names
    ]
    assert not missing, f"{mqdt!r}: {len(missing)} models not reachable via get_mqdt_models: {missing}"


def test_fj_channels(model: FModel) -> None:
    """fj_channels decomposes every outer channel into FJ kets with the model's f_tot."""
    fj_channels = model.fj_channels
    assert len(fj_channels) >= len(model.outer_channels)
    assert all(isinstance(ket, AngularKetFJ) for ket in fj_channels)
    assert all(ket.f_tot == model.f_tot for ket in fj_channels)


def test_fmodel_get_core_kets(model: FModel) -> None:
    """get_core_kets returns the sorted unique core kets of the outer channels."""
    core_kets = model.get_core_kets()
    assert len(core_kets) == len(set(core_kets))
    assert set(core_kets) == {ch.get_core_ket() for ch in model.outer_channels}
    sort_keys = [(ket.l_c, ket.j_c, ket.f_c, str(ket.label)) for ket in core_kets]
    assert sort_keys == sorted(sort_keys)


def test_mqdt_get_core_kets(mqdt: MQDT) -> None:
    """MQDT.get_core_kets returns the union of core kets over all its models."""
    core_kets = mqdt.get_core_kets()
    assert len(core_kets) == len(set(core_kets))
    assert set(core_kets) == {ck for model in mqdt.models for ck in model.get_core_kets()}


# Models that intentionally only describe an isolated nu window, i.e. no model is available
# directly above them. The gap above these models is therefore not reported as an error.
MODELS_WITH_ISOLATED_NU_RANGE = [
    "Sr87 P F=9/2 (clock), 1.8 < nu < 2.2",
    "Sr88 P J=1 (recombination), 1.8 < nu < 2.2",
]


def _share_channels(model_1: FModel, model_2: FModel) -> bool:
    """Whether two models describe (at least partly) the same physical channels.

    Dummy channels (channels with unknown quantum numbers) are ignored,
    since they only stand in for unidentified perturbers and would match across unrelated models.
    """
    return any(
        abs(ket_1.calc_reduced_overlap(ket_2)) > 0
        for ket_1 in model_1.outer_channels
        if not ket_1.contains_unknown
        for ket_2 in model_2.outer_channels
        if not ket_2.contains_unknown
    )


def _group_models_by_channels(models: list[FModel]) -> list[list[FModel]]:
    """Group the models into sets of models that (transitively) describe the same channels."""
    groups: list[list[FModel]] = []
    for model in models:
        matching = [group for group in groups if any(_share_channels(model, other) for other in group)]
        merged = [model, *(other for group in matching for other in group)]
        groups = [group for group in groups if group not in matching]
        groups.append(merged)
    return groups


def test_nu_ranges_match_at_boundaries(mqdt: MQDT) -> None:
    """Models describing the same channels must tile nu without gaps or overlaps.

    If one model is valid up to nu = 5.7, the model describing the same channels above it must start
    exactly at nu = 5.7 (and not at 6, which would leave a gap of states, nor at 5.5, which would
    make the states in the overlap appear twice).
    """
    errors: list[str] = []
    for group in _group_models_by_channels(mqdt.models):
        ordered = sorted(group, key=lambda model: model.nu_range)
        for model, next_model in pairwise(ordered):
            if model.full_name in MODELS_WITH_ISOLATED_NU_RANGE:
                continue
            if model.nu_max != next_model.nu_min:
                relation = "gap" if model.nu_max < next_model.nu_min else "overlap"
                errors.append(
                    f"{relation} between '{model.full_name}' (nu_max={model.nu_max}) and "
                    f"'{next_model.full_name}' (nu_min={next_model.nu_min})"
                )
    msg = f"{mqdt!r}: nu ranges of models describing the same channels do not match:\n" + "\n".join(errors)
    assert not errors, msg
