from __future__ import annotations

import re

import numpy as np
import pytest
from rydstate.species.mqdt.fmodel import FModel
from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT


def _all_fmodels() -> list[FModel]:
    """Collect all concrete FModel subclasses."""
    return [cls() for cls in FModel.__subclasses__() if hasattr(cls, "name") and cls.name is not None]


ALL_MODELS = _all_fmodels()


@pytest.fixture(params=ALL_MODELS, ids=lambda cls: cls.full_name)
def model(request: pytest.FixtureRequest) -> FModel:
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
    """All non-dummy channels must have the same parity (-1)^(l_c+l_r)."""
    parities: list[int] = [(-1) ** (ch.l_c + ch.l_r) for ch in model.inner_channels if not ch.contains_unknown]
    parities.extend((-1) ** (och.l_c + och.l_r) for och in model.outer_channels if not och.contains_unknown)
    assert len(set(parities)) <= 1, f"{model.full_name}: channels have inconsistent parity"


def test_all_channels_have_ionization_threshold(model: FModel) -> None:
    """All channels must have ionization thresholds."""
    species = SpeciesObjectMQDT.from_name(model.species_name)
    try:
        for _i, ch in enumerate(model.outer_channels):
            species.get_ionization_threshold(ch.get_core_ket())
    except ValueError:
        pytest.fail(
            f"{model.full_name}: outer_channels[{_i}] with core ket {ch.get_core_ket()} "
            "has no ionization threshold defined"
        )


def test_mixing_angles_indices_valid(model: FModel) -> None:
    """Mixing angle indices must refer to valid channel positions."""
    n_channels = len(model.inner_channels)
    for entry in model.mixing_angles:
        i, j = entry[0], entry[1]
        assert 0 <= i < n_channels, f"{model.full_name}: mixing_angles index {i} out of range [0, {n_channels})"
        assert 0 <= j < n_channels, f"{model.full_name}: mixing_angles index {j} out of range [0, {n_channels})"
        assert i != j, f"{model.full_name}: mixing_angles has self-coupling ({i}, {j})"


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
    """Every model must have a unique combination of species_name and name (full_name)."""
    full_name = model.full_name
    duplicates = [m for m in ALL_MODELS if m.full_name == full_name]
    assert len(duplicates) == 1, f"{model.full_name}: {len(duplicates)} duplicate models found"


def test_species_field_set(model: FModel) -> None:
    """Every model must have a species field."""
    assert model.species_name is not None, f"{model.full_name}: species is None"


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


def test_inner_outer_unitary(model: FModel) -> None:
    """The frame transformation matrix from inner to outer channels must be unitary."""
    unitary = model.calc_frame_transformation_outer_inner()
    msg = f"{model.species_name} - {model.full_name}: frame transformation (outer - inner) is not unitary"
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-10, err_msg=msg)

    rotation = model.calc_frame_transformation_inner_closecoupling(nu=30.5)
    msg = f"{model.species_name} - {model.full_name}: frame transformation (inner - closecoupling) is not unitary"
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(rotation.shape[0]), atol=1e-10, err_msg=msg)

    full = model.calc_frame_transformation(nu=30.5)
    msg = f"{model.species_name} - {model.full_name}: full frame transformation U=QR is not unitary"
    np.testing.assert_allclose(full.conj().T @ full, np.eye(full.shape[0]), atol=1e-10, err_msg=msg)
