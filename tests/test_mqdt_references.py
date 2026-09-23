from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import RydbergStateSQDTDivalent
from rydstate.angular.angular_ket import AngularKetJJ, AngularKetLS
from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.species import EigenChannelModel, get_mqdt, get_potential_class
from rydstate.species.utils import calc_modified_ritz_formula_in_nu

if TYPE_CHECKING:
    from rydstate.units import NDArray


def _get_model(species: str, name: str) -> EigenChannelModel:
    """Return the model of the given species with the given name."""
    model = next(model for model in get_mqdt(species).models if model.name == name)
    assert isinstance(model, EigenChannelModel)
    return model


_YB171_S05 = np.array(
    [
        [1 / 2, 0, 0, 0, 0, 0, np.sqrt(3) / 2],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, np.sqrt(2 / 3), 0, -np.sqrt(1 / 3), 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, np.sqrt(1 / 3), 0, np.sqrt(2 / 3), 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [np.sqrt(3) / 2, 0, 0, 0, 0, 0, -1 / 2],
    ]
)
_YB171_D25 = np.array(
    [
        [np.sqrt(7 / 5) / 2, np.sqrt(7 / 30), 0, 0, 0, -np.sqrt(5 / 3) / 2],
        [-np.sqrt(2 / 5), np.sqrt(3 / 5), 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1 / 2, np.sqrt(1 / 6), 0, 0, 0, np.sqrt(7 / 3) / 2],
    ]
)
_YB174_D2 = np.array(
    [
        [np.sqrt(3 / 5), np.sqrt(2 / 5), 0, 0, 0],
        [-np.sqrt(2 / 5), np.sqrt(3 / 5), 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
)

# The frame transformations, which were previously hardcoded in the model data files
# (as manual_frame_transformation_outer_inner, taken from the papers the models are based on).
# We keep one reference per structurally distinct model: outer channels in JJ coupling (S05),
# outer channels in LS coupling with i_c != 0 (D25) and with i_c = 0 (D2).
REFERENCE_FRAME_TRANSFORMATIONS: list[tuple[str, str, NDArray]] = [
    ("Yb171", "S F=1/2, nu > 26", _YB171_S05),
    ("Yb171", "S F=1/2, 2 < nu < 26", _YB171_S05),
    ("Yb171", "D F=5/2, nu > 30", _YB171_D25),
    ("Yb171", "D F=5/2, 2 < nu < 30", _YB171_D25),
    ("Yb174", "D J=2, nu > 5", _YB174_D2),
]

# Experimentally measured Yb174 levels (from the NIST data shipped with the species),
# which lie inside the nu range of a multi-channel MQDT model: (model name, n, l_r, j_tot, s_tot).
NIST_LEVELS: list[tuple[str, int, int, int, int]] = [
    ("S J=0, nu > 2", 7, 0, 0, 0),  # 6s7s 1S0
    ("S J=0, nu > 2", 8, 0, 0, 0),  # 6s8s 1S0
    ("D J=2, nu > 5", 8, 2, 2, 1),  # 6s8d 3D2
]


def _equal_up_to_channel_signs(a: NDArray, b: NDArray, atol: float = 1e-10) -> bool:
    """Check whether a = diag(row_signs) @ b @ diag(col_signs) for some sign vectors row_signs, col_signs.

    Such sign flips only correspond to a different phase convention of the individual inner and outer channel kets.
    The signs of the columns (inner channels) drop out of K = U Kbar U^T,
    the signs of the rows (outer channels) only flip the sign of the corresponding channel coefficients.
    """
    if not np.allclose(np.abs(a), np.abs(b), atol=atol):
        return False

    ratios = np.where(np.abs(b) > atol, np.sign(a * b), 0)  # a[i, j] = row_signs[i] * b[i, j] * col_signs[j]
    row_signs = np.zeros(a.shape[0])
    col_signs = np.zeros(a.shape[1])
    for start in range(a.shape[0]):
        if row_signs[start] != 0:  # already fixed via a previous connected component
            continue
        row_signs[start] = 1  # the overall sign of each connected component is arbitrary
        stack = [start]
        while stack:  # propagate the sign through the connected component
            i = stack.pop()
            for j in np.flatnonzero(ratios[i]):
                col_signs[j] = ratios[i, j] * row_signs[i]
                for k in np.flatnonzero(ratios[:, j]):
                    sign = ratios[k, j] * col_signs[j]
                    if row_signs[k] == 0:
                        row_signs[k] = sign
                        stack.append(int(k))
                    elif row_signs[k] != sign:
                        return False
    return True


@pytest.mark.parametrize(("species", "name", "reference"), REFERENCE_FRAME_TRANSFORMATIONS)
def test_frame_transformation_matches_reference(species: str, name: str, reference: NDArray) -> None:
    """The calculated frame transformation must match the frame transformation given in the literature.

    The frame transformation is calculated from the overlaps of the inner and outer channel kets,
    so we check here that it still reproduces the published matrices.
    The two may only differ by the sign convention of the individual inner and outer channel kets.
    """
    model = _get_model(species, name)
    calculated = model.calc_frame_transformation_outer_inner()
    assert _equal_up_to_channel_signs(reference, calculated), (
        f"{model.full_name}: calculated frame transformation does not match the reference\n"
        f"reference:\n{np.round(reference, 4)}\ncalculated:\n{np.round(calculated, 4)}"
    )


@pytest.mark.parametrize(("name", "n", "l_r", "j_tot", "s_tot"), NIST_LEVELS)
def test_mqdt_energies_match_nist(name: str, n: int, l_r: int, j_tot: int, s_tot: int) -> None:
    """The multi-channel models must reproduce the experimentally measured Yb174 levels.

    This checks the whole MQDT pipeline (channel definitions, frame transformation, K-matrix, det(M) roots)
    against experiment, without relying on any hardcoded numbers:
    the experimental energies are taken from the NIST data shipped with the species
    (RydbergStateSQDT uses them for the low lying states instead of the Rydberg-Ritz formula).

    The models reproduce these levels to |dnu| < 3e-4, while e.g. mixing up two channels of the
    frame transformation shifts them by |dnu| ~ 1e-1, i.e. the tolerance below is not tight, but still strict.
    """
    nu_experimental = RydbergStateSQDTDivalent("Yb174", n=n, l=l_r, s=s_tot, j=j_tot).nu

    model = _get_model("Yb174", name)
    assert len(model.inner_channels) > 1, f"{model.full_name}: not a multi-channel model"

    nu_range = (nu_experimental - 0.5, nu_experimental + 0.5)
    states = get_mqdt_states_from_model(model, nu_range, NotSet, get_potential_class("Yb174"))
    assert len(states) > 0, f"{model.full_name}: no states found around nu={nu_experimental}"

    closest = min(states, key=lambda state: abs(state.nu - nu_experimental))
    assert abs(closest.nu - nu_experimental) < 1e-3, (
        f"{model.full_name}: the calculated nu={closest.nu} does not match "
        f"the experimental nu={nu_experimental} of the {n=}, {l_r=}, {j_tot=}, {s_tot=} level"
    )


# Low lying states, whose singlet-triplet mixing angle is not determined by the energies the models were fitted to:
# (species, model name, nu range containing the state, l_r). The listed state is the lowest state in the nu range.
SPIN_ORBIT_MIXED_STATES: list[tuple[str, str, tuple[float, float], int]] = [
    ("Sr88", "P J=1 (recombination), 1.8 < nu < 2.2", (1.8, 2.2), 1),  # 5s5p 3P1
    ("Yb174", "P J=1, 1.7 < nu < 2.7", (1.7, 2.7), 1),  # 6s6p 3P1
    ("Yb174", "D J=2, 2 < nu < 5", (2.0, 3.2), 2),  # 6s5d 3D2
]


@pytest.mark.parametrize(("species", "name", "nu_range", "l_r"), SPIN_ORBIT_MIXED_STATES)
def test_singlet_triplet_mixing_towards_jj_coupling(
    species: str, name: str, nu_range: tuple[float, float], l_r: int
) -> None:
    """The singlet-triplet mixing of the low lying states has the sign expected from the spin-orbit interaction.

    The spin-orbit interaction of the valence electron (with normal fine structure, j = l_r - 1/2 below
    j = l_r + 1/2) mixes the lower of the two LS states with the same J towards the jj coupled state with
    j_r = l_r - 1/2. So the lower state must have a larger weight of j_r = l_r - 1/2 than the pure LS state.
    With the opposite sign of the mixing angle, the state is rotated away from it.
    """
    model = _get_model(species, name)
    states = get_mqdt_states_from_model(model, nu_range, NotSet, get_potential_class(species))
    state = min(states, key=lambda state: state.get_energy("1/cm"))
    j_tot = model.f_tot
    s_tot = round(state.calc_exp_qn("s_tot"))
    ls_ket = AngularKetLS(l_c=0, l_r=l_r, l_tot=l_r, s_tot=s_tot, j_tot=j_tot, species=species)
    jj_ket = AngularKetJJ(l_c=0, l_r=l_r, j_c=0.5, j_r=l_r - 0.5, j_tot=j_tot, species=species)
    weight_ls = jj_ket.calc_reduced_overlap(ls_ket) ** 2
    weight = sum(coeff**2 for coeff, ket in state if ket.angular.calc_reduced_overlap(jj_ket) != 0) / state.norm**2
    assert weight > weight_ls, f"{model.full_name}: weight of j_r = l_r - 1/2 {weight} < {weight_ls} of the LS state"


# Yb D J=2 models and nu ranges, in which the relative sign of the 6snd 1D2 and 3D2 components is checked
YB_D2_MODELS: list[tuple[str, str, tuple[float, float]]] = [
    ("Yb174", "D J=2, 2 < nu < 5", (2, 5)),
    ("Yb174", "D J=2, nu > 5", (5, 40)),
    ("Yb171", "D F=3/2, 2 < nu < 30", (2, 30)),
    ("Yb171", "D F=5/2, 2 < nu < 30", (2, 30)),
    ("Yb171", "D F=3/2, nu > 30", (30, 40)),
    ("Yb171", "D F=5/2, nu > 30", (30, 40)),
]


@pytest.mark.parametrize(("species", "name", "nu_range"), YB_D2_MODELS)
def test_yb_d2_singlet_triplet_relative_sign(species: str, name: str, nu_range: tuple[float, float]) -> None:
    """The relative sign of the 6snd 1D2 and 3D2 components is the same in all Yb D J=2 models.

    The sign of the singlet-triplet mixing is fixed by the 171Yb energies of the Rydberg states (hyperfine
    interaction) and, for the low lying states (2 < nu < 5), by the spin-orbit interaction
    (see test_singlet_triplet_mixing_towards_jj_coupling). In all models, the states with dominant triplet character
    have c_S * c_T < 0 and the states with dominant singlet character c_S * c_T > 0
    (c_S, c_T: amplitudes of 6snd 1D2 and 3D2). The mixing angles of the different models (with very different
    parametrizations) are therefore consistent and there is no jump of the relative sign between the models.
    States with a small singlet-triplet mixing (< 1%) or an almost equal mixing are ignored.
    """
    model = _get_model(species, name)
    f_tot = model.f_tot
    ls_kets = [AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=s, j_tot=2, f_tot=f_tot, species=species) for s in (0, 1)]
    n_checked = 0
    for state in get_mqdt_states_from_model(model, nu_range, NotSet, get_potential_class(species)):
        c_s, c_t = (sum(c * ket.angular.calc_reduced_overlap(ls) for c, ket in state) / state.norm for ls in ls_kets)
        if c_s**2 + c_t**2 < 0.5 or min(c_s**2, c_t**2) < 0.01 or abs(c_t**2 - c_s**2) < 0.2:
            continue
        n_checked += 1
        assert (c_s * c_t < 0) == (c_t**2 > c_s**2), f"{model.full_name}: {state.nu=}, {c_s=}, {c_t=}"
    assert n_checked > 0


def _get_nist_fit_models() -> list[EigenChannelModel]:
    """Return the eigenchannel models of all species, which were fitted to NIST data (instead of taken from papers)."""
    models = []
    for species in ("Sr87", "Sr88", "Yb171", "Yb173", "Yb174"):
        for model in get_mqdt(species).models:
            reference = " ".join(model.reference) if isinstance(model.reference, tuple) else str(model.reference)
            if (
                isinstance(model, EigenChannelModel)
                and model.mixing_angles
                and re.search("fit to .*NIST data", reference)
            ):
                models.append(model)
    return models


@pytest.mark.parametrize("model", _get_nist_fit_models(), ids=lambda model: model.full_name)
def test_nist_fit_mixing_angles_smaller_than_pi_half(model: EigenChannelModel) -> None:
    """The mixing angles of the models fitted to NIST data stay within (-pi/2, pi/2) in the whole nu range.

    A rotation by pi/2 just swaps the two eigenchannels, so larger angles are equivalent to smaller angles
    with the eigen quantum defects swapped, but they make the sign of the angle (i.e. the singlet-triplet mixing)
    hard to interpret and compare between models. The angles are evaluated like in the model,
    i.e. with the nui of the first channel, see
    :meth:`~rydstate.species.eigen_channel_model.EigenChannelModel.calc_frame_transformation_inner_closecoupling`.
    """
    for nu in np.linspace(model.nu_min, model.nu_max, 200):
        nui_0 = float(model.calc_channel_nuis(nu)[0])
        for i, j, coefficients in model.mixing_angles or []:
            angle = calc_modified_ritz_formula_in_nu(nui_0, coefficients)
            assert abs(angle) < np.pi / 2, f"{model.full_name}: mixing angle ({i}, {j}) = {angle} at {nu=}"
