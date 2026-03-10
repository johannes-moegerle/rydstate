from __future__ import annotations

import re

import numpy as np
import pytest
import rydstate.species.mqdt.sr87
import rydstate.species.mqdt.sr88
import rydstate.species.mqdt.yb171
import rydstate.species.mqdt.yb173
import rydstate.species.mqdt.yb174  # noqa: F401
from rydstate.angular.utils import is_dummy_ket
from rydstate.species.mqdt.fmodel import FModel


def _all_fmodels() -> list[FModel]:
    """Collect all concrete FModel subclasses."""
    return [cls() for cls in FModel.__subclasses__() if hasattr(cls, "name") and cls.name is not None]


ALL_MODELS = _all_fmodels()


@pytest.fixture(params=ALL_MODELS, ids=lambda cls: cls.name)
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
    assert n_inner == n_outer, f"{model.name}: inner_channels ({n_inner}) != outer_channels ({n_outer})"
    assert n_inner == n_defects, f"{model.name}: channels ({n_inner}) != eigen_quantum_defects ({n_defects})"


def test_nu_range_valid(model: FModel) -> None:
    """nu_range must be a 2-tuple with min < max."""
    nu_min, nu_max = model.nu_range
    assert nu_min < nu_max, f"{model.name}: nu_range {model.nu_range} has min >= max"
    assert nu_min > 0, f"{model.name}: nu_range min must be positive"


def test_f_tot_consistency(model: FModel) -> None:
    """All non-dummy channels must have f_tot matching the model's f_tot."""
    for i, ch in enumerate(model.inner_channels):
        if not is_dummy_ket(ch):
            assert ch.f_tot == model.f_tot, (
                f"{model.name}: inner_channels[{i}].f_tot={ch.f_tot} != model.f_tot={model.f_tot}"
            )
    for i, och in enumerate(model.outer_channels):
        if not is_dummy_ket(och):
            assert och.f_tot == model.f_tot, (
                f"{model.name}: outer_channels[{i}].f_tot={och.f_tot} != model.f_tot={model.f_tot}"
            )


def test_parity_consistency(model: FModel) -> None:
    """All non-dummy channels must have the same parity (-1)^(l_c+l_r)."""
    parities: list[int] = [(-1) ** (ch.l_c + ch.l_r) for ch in model.inner_channels if not is_dummy_ket(ch)]
    parities.extend((-1) ** (och.l_c + och.l_r) for och in model.outer_channels if not is_dummy_ket(och))
    assert len(set(parities)) <= 1, f"{model.name}: channels have inconsistent parity"


def test_dummy_channels_match(model: FModel) -> None:
    """Dummy channels must appear at the same positions in inner and outer channel lists."""
    for i, (inner, outer) in enumerate(zip(model.inner_channels, model.outer_channels, strict=True)):
        inner_is_dummy = is_dummy_ket(inner)
        outer_is_dummy = is_dummy_ket(outer)
        assert inner_is_dummy == outer_is_dummy, (
            f"{model.name}: channel {i} dummy mismatch "
            f"(inner={'dummy' if inner_is_dummy else 'real'}, outer={'dummy' if outer_is_dummy else 'real'})"
        )


def test_mixing_angles_indices_valid(model: FModel) -> None:
    """Mixing angle indices must refer to valid channel positions."""
    n_channels = len(model.inner_channels)
    for entry in model.mixing_angles:
        i, j = entry[0], entry[1]
        assert 0 <= i < n_channels, f"{model.name}: mixing_angles index {i} out of range [0, {n_channels})"
        assert 0 <= j < n_channels, f"{model.name}: mixing_angles index {j} out of range [0, {n_channels})"
        assert i != j, f"{model.name}: mixing_angles has self-coupling ({i}, {j})"


def test_model_name_contains_quantum_number(model: FModel) -> None:
    """Model name must contain F=X/Y or J=X matching the model's f_tot."""
    assert model.name is not None
    # Match F=X/Y or J=X patterns (integers and fractions)
    match = re.search(r"[FJ]=(\d+/?\.?\d*)", model.name)
    assert match is not None, f"{model.name}: name '{model.name}' does not contain F=... or J=..."
    s = match.group(1)
    if "/" in s:
        num, den = s.split("/")
        f_val = float(num) / float(den)
    else:
        f_val = float(s)
    assert f_val == model.f_tot, f"{model.name}: name says F/J={f_val} but f_tot={model.f_tot}"


def test_reference_field_set(model: FModel) -> None:
    """Every model must have an explicit reference field (str or None)."""
    assert hasattr(model, "reference"), f"{model.name}: missing 'reference' field"


def test_species_field_set(model: FModel) -> None:
    """Every model must have a species field."""
    assert model.species_name is not None, f"{model.name}: species is None"


def test_eigen_quantum_defects_format(model: FModel) -> None:
    """Each eigen quantum defect entry must be a list/tuple of numeric values."""
    for i, defect in enumerate(model.eigen_quantum_defects):
        if isinstance(defect, (list, tuple)):
            for j, val in enumerate(defect):
                assert isinstance(val, (int, float)), (
                    f"{model.name}: eigen_quantum_defects[{i}][{j}] is {type(val)}, expected numeric"
                )
        else:
            assert isinstance(defect, (int, float)), (
                f"{model.name}: eigen_quantum_defects[{i}] is {type(defect)}, expected numeric or list"
            )


def test_inner_outer_channel_l_r_match(model: FModel) -> None:
    """Non-dummy inner and outer channels at the same position must have the same l_r."""
    for i, (inner, outer) in enumerate(zip(model.inner_channels, model.outer_channels, strict=True)):
        if is_dummy_ket(inner) or is_dummy_ket(outer):
            continue
        assert inner.l_r == outer.l_r, (
            f"{model.name}: channel {i} l_r mismatch (inner.l_r={inner.l_r}, outer.l_r={outer.l_r})"
        )


def test_at_least_one_real_channel(model: FModel) -> None:
    """Every model must have at least one non-dummy channel."""
    real_channels = [ch for ch in model.inner_channels if not is_dummy_ket(ch)]
    assert len(real_channels) >= 1, f"{model.name}: no real (non-dummy) channels"


def test_inner_outer_unitary(model: FModel) -> None:
    """The frame transformation matrix from inner to outer channels must be unitary."""
    unitary = model.calc_frame_transformation_outer_inner()
    msg = f"{model.species_name} - {model.name}: frame transformation (outer - inner) is not unitary"
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-10, err_msg=msg)

    rotation = model.calc_frame_transformation_inner_closecoupling(nu_ref=30.5)
    msg = f"{model.species_name} - {model.name}: frame transformation (inner - closecoupling) is not unitary"
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(rotation.shape[0]), atol=1e-10, err_msg=msg)

    full = model.calc_frame_transformation(nu_ref=30.5)
    msg = f"{model.species_name} - {model.name}: full frame transformation U=QR is not unitary"
    np.testing.assert_allclose(full.conj().T @ full, np.eye(full.shape[0]), atol=1e-10, err_msg=msg)
