from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rydstate import RydbergStateSQDTDivalent
from rydstate.angular.angular_ket import AngularKetJJ, AngularKetLS
from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.species import EigenChannelModel, KMatrixModel, get_mqdt, get_potential_class
from rydstate.species.utils import calc_nu_from_energy
from rydstate.units import ureg

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.rydberg_state import RydbergStateMQDT
    from rydstate.species import MQDTModel
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


def _get_k_matrix_model(name: str) -> KMatrixModel:
    """Return the Sr88 model of Vaillant 2024 (MQDT tag vaillant2024) with the given name."""
    model = next(model for model in get_mqdt("Sr88", "vaillant2024").models if model.name == name)
    assert isinstance(model, KMatrixModel)
    return model


# Theoretical term energies (in 1/cm) of the Sr88 models of Vaillant, Jones and Potvliege 2024, taken from the
# supplementary material of the Addendum (J. Phys. B 57, 199401 (2024), files tables_*.txt, column (c)).
# For each model we check the lowest fitted state, the perturber (if any) and the highest fitted states:
# (model name, [(experimental energy, theoretical energy of the paper), ...])
VAILLANT_REFERENCE_ENERGIES: list[tuple[str, list[tuple[float, float]]]] = [
    ("S J=0, nu > 3.7", [(38444.013, 38444.012512), (44525.838, 44525.837524), (45778.6257, 45778.627021)]),
    ("S J=1, nu > 3.4", [(37424.675, 37424.674450), (44747.64060, 44747.625509), (45647.36527, 45647.359319)]),
    ("P J=1 singlet, nu > 2.8", [(34098.404, 34098.403791), (41172.054, 41172.058686), (45773.14, 45773.255794)]),
    ("P J=0, nu > 2.8", [(33853.490, 33853.489518), (37292.074, 37292.073288), (45183.93, 45183.963970)]),
    ("P J=1 triplet, nu > 2.8", [(33868.317, 33868.316522), (37302.731, 37302.730312), (45184.54, 45184.525966)]),
    ("P J=2, nu > 2.8", [(33973.065, 33973.064509), (37336.591, 37336.590249), (45185.79, 45185.804491)]),
    (
        "D J=2, nu > 5.7",
        [
            (43021.058, 43021.073073),  # 5s8d 1D2
            (43070.268, 43070.274672),  # 5s8d 3D2
            (44729.56, 44730.546738),  # 4d2 3P2 perturber
            (45153.28988, 45153.290438),  # 5s14d 1D2
            (45171.49569, 45171.496996),  # 5s14d 3D2
            (45785.74337, 45785.743140),  # 5s30d 3D2
            (45788.8895, 45788.894714),  # 5s30d 1D2
        ],
    ),
    ("D J=1, nu > 9.5", [(44853.97363, 44853.972136), (45883.21692, 45883.216067)]),
    ("D J=3, nu > 3.9", [(39703.109, 39703.108783), (44865.22, 44865.221482), (45775.60, 45775.601314)]),
    ("F J=3 singlet, nu > 3.5", [(38007.742, 38007.741506), (39539.013, 39539.012578), (45801.03, 45800.940167)]),
]


def _get_states_around_energy(model: MQDTModel, energy: float, delta_nu: float = 0.3) -> list[RydbergStateMQDT]:
    """Return the MQDT states of the model within +- delta_nu around the given term energy (in 1/cm)."""
    energy_au = ureg.Quantity(energy, "1/cm").to("hartree", "spectroscopy").magnitude
    reference_au = model.mqdt.reference_ionization_threshold_au
    nu = calc_nu_from_energy(model.element_properties.reduced_mass_au, energy_au - reference_au)
    nu_range = (max(nu - delta_nu, model.nu_min), nu + delta_nu)
    return get_mqdt_states_from_model(model, nu_range, NotSet, get_potential_class(model.species))


@pytest.mark.parametrize(
    ("name", "energies"), VAILLANT_REFERENCE_ENERGIES, ids=[name for name, _ in VAILLANT_REFERENCE_ENERGIES]
)
def test_vaillant2024_energies_match_publication(name: str, energies: list[tuple[float, float]]) -> None:
    """The Sr88 K-matrix models reproduce the theoretical energies published with them.

    This checks the K-matrix formulation (KMatrixModel, including the energy dependence of the K-matrix via
    epsilon = (I_s - E) / I_s), the channel definitions and the ionization thresholds of the models against the
    supplementary material of the Addendum, which we reproduce to ~1e-5 1/cm (the remaining difference comes from
    the slightly different mass corrected Rydberg constant, 109736.631 1/cm in the paper vs. 109736.6309 1/cm here).
    For comparison: the deviation of the theoretical energies from experiment is > 1e-3 1/cm for most states.
    """
    model = _get_k_matrix_model(name)
    for energy_experiment, energy_theory in energies:
        states = _get_states_around_energy(model, energy_theory)
        assert len(states) > 0, f"{model.full_name}: no states found around E={energy_theory} 1/cm"
        closest = min(states, key=lambda state: abs(state.get_energy("1/cm") - energy_theory))
        assert closest.get_energy("1/cm") == pytest.approx(energy_theory, abs=2e-4), (
            f"{model.full_name}: the calculated energy {closest.get_energy('1/cm')} 1/cm does not match "
            f"the published theoretical energy {energy_theory} 1/cm (experiment: {energy_experiment} 1/cm)"
        )


# Expected spin character <s_tot> (= triplet fraction) of some 5snd J=2 states of the Vaillant 2024 model:
# (experimental energy, expected <s_tot>, tolerance). The values follow from the LS coupled channel fractions
# shown in Figs. 1 and 2 of the supplementary material of the Addendum.
VAILLANT_D2_SPIN_CHARACTER: list[tuple[float, float, float]] = [
    (45788.8895, 0.0, 0.03),  # 5s30d 1D2: almost pure singlet
    (45785.74337, 1.0, 0.03),  # 5s30d 3D2: almost pure triplet
    (45153.28988, 0.18, 0.05),  # 5s14d 1D2: ~0.8 singlet
    (45171.49569, 0.82, 0.05),  # 5s14d 3D2: ~0.8 triplet
    (44829.6648, 0.07, 0.05),  # 5s12d 1D2: ~0.9 singlet
    (44860.06382, 0.93, 0.05),  # 5s12d 3D2: ~0.9 triplet
]


@pytest.mark.parametrize(("energy", "expected", "tolerance"), VAILLANT_D2_SPIN_CHARACTER)
def test_vaillant2024_d2_singlet_triplet_character(energy: float, expected: float, tolerance: float) -> None:
    """The 5snd J=2 states of the K-matrix model have the correct singlet/triplet character.

    The K-matrix of the paper is given in the frame of the jj-coupled channels 5s_1/2 nd_5/2 and 5s_1/2 nd_3/2,
    whose relative phase differs between rydstate and the paper. The corresponding off-diagonal K-matrix elements
    are therefore sign flipped in the model data, which does not change the energies (checked above), but is
    necessary for the correct singlet/triplet character. With the wrong sign, the states of the singlet series
    would come out as mostly triplet (<s_tot> ~ 0.98 instead of 0.003 for 5s30d 1D2).
    """
    model = _get_k_matrix_model("D J=2, nu > 5.7")
    states = _get_states_around_energy(model, energy)
    closest = min(states, key=lambda state: abs(state.get_energy("1/cm") - energy))
    assert abs(closest.get_energy("1/cm") - energy) < 0.01
    assert closest.calc_exp_qn("s_tot") == pytest.approx(expected, abs=tolerance)


def _vaillant2024_ket_phase(ket: AngularKetBase[Any]) -> int:
    """Phase d_i of a channel ket of Vaillant 2024 relative to the rydstate ket (see the model data docstring)."""
    if isinstance(ket, AngularKetJJ):
        return (-1) ** round(ket.j_c + ket.j_r - ket.j_tot)
    assert isinstance(ket, AngularKetLS)
    return (-1) ** round(ket.l_c + ket.l_r - ket.l_tot) * (-1) ** round(1 - ket.s_tot)


# jj->LS recoupling matrices U_{i alphabar} of the mqdtfit driver scripts (folder strontium, driver1and3D2.py and
# driver1S0.py) as (model name, [LS kets], U) with U[i, alphabar] = <jj channel i | LS ket alphabar> in the phase
# convention of the paper (the jj channels are the outer channels of the model, LS channels of the model are skipped).
_SQRT35, _SQRT25 = np.sqrt(3 / 5), np.sqrt(2 / 5)
VAILLANT_RECOUPLING_MATRICES: list[tuple[str, list[AngularKetLS[Any]], NDArray]] = [
    (
        "D J=2, nu > 5.7",
        [
            AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=0, j_tot=2, species="Sr88"),  # 5snd 1D2
            AngularKetLS(l_c=0, l_r=2, l_tot=2, s_tot=1, j_tot=2, species="Sr88"),  # 5snd 3D2
            AngularKetLS(l_c=2, l_r=0, l_tot=2, s_tot=0, j_tot=2, species="Sr88"),  # 4dns 1D2
            AngularKetLS(l_c=2, l_r=0, l_tot=2, s_tot=1, j_tot=2, species="Sr88"),  # 4dns 3D2
        ],
        np.array(
            [
                [_SQRT35, -_SQRT25, 0, 0],  # 5s_1/2 nd_5/2
                [_SQRT25, _SQRT35, 0, 0],  # 5s_1/2 nd_3/2
                [0, 0, _SQRT35, _SQRT25],  # 4d_5/2 ns_1/2
                [0, 0, -_SQRT25, _SQRT35],  # 4d_3/2 ns_1/2
            ]
        ),
    ),
    (
        "S J=0, nu > 3.7",
        [
            AngularKetLS(l_c=0, l_r=0, l_tot=0, s_tot=0, j_tot=0, species="Sr88"),  # 5sns 1S0
            AngularKetLS(l_c=2, l_r=2, l_tot=0, s_tot=0, j_tot=0, species="Sr88"),  # 4dnd 1S0
            AngularKetLS(l_c=2, l_r=2, l_tot=1, s_tot=1, j_tot=0, species="Sr88"),  # 4dnd 3P0
        ],
        np.array(
            [
                [1, 0, 0],  # 5s_1/2 ns_1/2
                [0, _SQRT35, -_SQRT25],  # 4d_5/2 nd_5/2
                [0, _SQRT25, _SQRT35],  # 4d_3/2 nd_3/2
            ]
        ),
    ),
]


@pytest.mark.parametrize(("name", "ls_kets", "u_paper"), VAILLANT_RECOUPLING_MATRICES)
def test_vaillant2024_recoupling_phase_convention(
    name: str, ls_kets: list[AngularKetLS[Any]], u_paper: NDArray
) -> None:
    """The channel kets of the paper are d_i times the rydstate kets (d_i given in the docstring of the model data).

    The paper couples the core electron first, rydstate the Rydberg electron first, which leads to the channel
    dependent phases d_i. Multiplying the rydstate overlaps <jj|LS> by d_jj * d_LS must reproduce the recoupling
    matrices U_{i alphabar} of the mqdtfit drivers exactly.
    """
    model = _get_k_matrix_model(name)
    u_rydstate = np.array(
        [
            [_vaillant2024_ket_phase(jj) * _vaillant2024_ket_phase(ls) * jj.calc_reduced_overlap(ls) for ls in ls_kets]
            for jj in model.outer_channels
            if isinstance(jj, AngularKetJJ)
        ]
    )
    np.testing.assert_allclose(u_rydstate, u_paper, atol=1e-12)


# Off-diagonal K-matrix elements K_ij^(0) of Table III of the Addendum (values with 9 digits from the mqdtfit drivers)
# for the models, in which some signs differ from the paper: (model name, {(i, j): value of the paper})
VAILLANT_PAPER_OFF_DIAGONAL_K: list[tuple[str, dict[tuple[int, int], float]]] = [
    ("S J=1, nu > 3.4", {(0, 1): -1.33451654e2}),
    (
        "D J=2, nu > 5.7",
        {
            (0, 1): 2.30810347e-1,
            (0, 2): -2.99689799e-1,
            (0, 3): 6.24839129e-1,
            (0, 4): -2.38162135e-1,
            (0, 5): -8.94462406e-2,
            (1, 2): -6.41169781e-1,
            (1, 3): 8.10126226e-6,
            (1, 4): -4.84958202e-1,
            (1, 5): 2.42734982e-3,
            (2, 3): 2.07880487e-1,
        },
    ),
]


@pytest.mark.parametrize(("name", "k_paper"), VAILLANT_PAPER_OFF_DIAGONAL_K)
def test_vaillant2024_k_matrix_signs(name: str, k_paper: dict[tuple[int, int], float]) -> None:
    """The K-matrix of the model is D K_paper D with D = diag(d_i) the phases of the channel kets."""
    model = _get_k_matrix_model(name)
    d = [_vaillant2024_ket_phase(ket) for ket in model.outer_channels]
    k_model = {(i, j): coefficients[0] for i, j, coefficients in model.k_matrix}
    for (i, j), value in k_paper.items():
        assert k_model[i, j] == pytest.approx(d[i] * d[j] * value), f"{name}: K-matrix element ({i}, {j})"


def test_vaillant2024_ionization_thresholds() -> None:
    """The channels of the Vaillant 2024 models use the ionization thresholds of Table I of the Addendum.

    In particular, the LS coupled 4dnl channels must use the averaged 4d threshold (60628.26 1/cm) also if their
    core j_c is fixed by the angular momentum coupling (e.g. 4dns 3D1 is purely 4d_3/2 ns_1/2), while the
    jj coupled channels use the fine structure resolved 4d_3/2 (60488.09 1/cm) and 4d_5/2 (60768.43 1/cm) thresholds.
    """
    expected = {
        "S J=0, nu > 3.7": [45932.2002, 60768.43, 60488.09],
        "S J=1, nu > 3.4": [45932.2002, 70048.11],
        "P J=1 singlet, nu > 2.8": [45932.2002, 60628.26],
        "P J=0, nu > 2.8": [45932.2002, 60628.26],
        "P J=1 triplet, nu > 2.8": [45932.2002, 60628.26],
        "P J=2, nu > 2.8": [45932.2002, 60628.26],
        "D J=2, nu > 5.7": [45932.2002, 45932.2002, 60768.43, 60488.09, 70048.11, 60628.26],
        "D J=1, nu > 9.5": [45932.2002, 60628.26],
        "D J=3, nu > 3.9": [45932.2002, 60628.26, 60628.26],
        "F J=3 singlet, nu > 3.5": [45932.2002, 60628.26],
        # the triplet F models are single channel models converging to the first ionization threshold
        "F J=2 (Rydberg-Ritz), nu > 9": [45932.2002],
        "F J=3 triplet (Rydberg-Ritz), nu > 9": [45932.2002],
        "F J=4 (Rydberg-Ritz), nu > 9": [45932.2002],
    }
    mqdt = get_mqdt("Sr88", "vaillant2024")
    assert {model.name for model in mqdt.models} == set(expected)
    for model in mqdt.models:
        thresholds = model.get_ionization_thresholds(unit="1/cm")
        np.testing.assert_allclose(thresholds, expected[model.name], atol=1e-6, err_msg=model.full_name)
