# Generated-by: Anthropic Claude Opus 5.5
# Human-review: Functionality and results cross-checked; code not reviewed line by line.
# If any of the tests in this file fail review the corresponding test in more detail!

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rydstate import RydbergStateSQDTDivalent
from rydstate.angular.angular_ket import AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.angular.utils import NotSet, is_unknown
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.species import EigenChannelModel, KMatrixModel, get_mqdt, get_potential_class, get_sqdt
from rydstate.species.utils import calc_modified_ritz_formula_in_nu, calc_nu_from_energy
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
# Table S10 of R. Kuroda et al., Phys. Rev. A 112, 042817 (2025) (arXiv:2507.11487)
_YB171_P05 = np.array(
    [
        [-np.sqrt(2 / 3), -np.sqrt(1 / 3), 0, 0, 0, 0, 0, 0],
        [1 / (2 * np.sqrt(3)), -np.sqrt(1 / 6), 0, 0, 0, 0, np.sqrt(3) / 2, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [-1 / 2, 1 / np.sqrt(2), 0, 0, 0, 0, 1 / 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
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
    ("Yb171", "P F=1/2, nu > 5.9", _YB171_P05),
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

    Such sign flips correspond to a different phase convention of the individual inner and outer channel kets.
    Note that the signs of the columns (inner channels) only drop out of K = U Kbar U^T if there are no mixing angles,
    otherwise the mixing angles have to be given in the same phase convention as the frame transformation,
    see test_frame_transformation_phase_convention.
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


# The sign of a mixing angle between two channels converging to the same ionization threshold, like the
# singlet-triplet mixing angles of 6snl 1L_J and 3L_J (or of the jj coupled 6sng +G4 and -G4), cannot be determined
# from the energies of the isotopes without hyperfine structure (the energies are invariant under the sign flip),
# although it changes the singlet/triplet character of the states. For the isotopes with hyperfine structure,
# the hyperfine interaction couples these channels to channels with different ionization thresholds, so that the
# energies fix the sign (except for the 6sng series, see test_yb174_g4_g_factors). The models of both isotopes must
# therefore use the same sign, which also ensures that the angles are given in the same phase convention.
ISOTOPE_PAIRS = [("Yb174", "Yb171"), ("Yb174", "Yb173"), ("Sr88", "Sr87")]


def _channel_key(ket: AngularKetBase[Any]) -> tuple[Any, ...] | None:
    """Return the quantum numbers of a channel ket without i_c and f_tot (None for kets with unknown values)."""
    if not isinstance(ket, AngularKetLS | AngularKetJJ):
        return None
    qns = zip(ket.quantum_number_names, ket.quantum_numbers, strict=True)
    return (type(ket).__name__, *(value for qn, value in qns if qn not in ("i_c", "f_tot")))


def _get_mixing_angles_by_channels(model: EigenChannelModel, nu: float) -> dict[tuple[Any, Any], float]:
    """Return the mixing angles of the model at nu, keyed by the channel keys of the mixed channels.

    The angles are reduced to the interval (-pi/2, pi/2], since theta and theta + pi only differ by the sign of the
    eigenchannels. The energy dependence is evaluated with nu instead of the nui of the first channel,
    which is good enough to determine the sign.
    """
    angles = {}
    for i, j, coefficients in model.mixing_angles or []:
        key_i, key_j = _channel_key(model.inner_channels[i]), _channel_key(model.inner_channels[j])
        if key_i is not None and key_j is not None:
            angle = calc_modified_ritz_formula_in_nu(nu, coefficients)
            angle = angle - np.pi * np.ceil(angle / np.pi - 0.5)
            angles[(key_i, key_j)] = angle
            angles[(key_j, key_i)] = -angle  # R_ij(theta) = R_ji(-theta)
    return angles


@pytest.mark.parametrize(("species", "species_hfs"), ISOTOPE_PAIRS)
def test_mixing_angle_signs_consistent_between_isotopes(species: str, species_hfs: str) -> None:
    """Mixing angles between the same channels have the same sign for the isotopes with and without hyperfine structure.

    Only models whose nu ranges overlap by at least 0.5 are compared, at a nu in the middle of the overlap.
    Angles with |theta| > pi/4 are not compared, since a rotation by theta ~ pi/2 mainly swaps the two eigenchannels,
    such that its sign can only be compared together with the eigen quantum defects
    (see test_yb_d2_singlet_triplet_relative_sign for a comparison of the resulting states).
    """
    models = [model for model in get_mqdt(species).models if isinstance(model, EigenChannelModel)]
    models_hfs = [model for model in get_mqdt(species_hfs).models if isinstance(model, EigenChannelModel)]
    n_compared = 0
    for model in models:
        for model_hfs in models_hfs:
            nu_min = max(model.nu_min, model_hfs.nu_min)
            nu_max = min(model.nu_max, model_hfs.nu_max, nu_min + 20)
            if nu_max - nu_min < 0.5:
                continue
            nu = (nu_min + nu_max) / 2
            angles = _get_mixing_angles_by_channels(model, nu)
            for channels, angle_hfs in _get_mixing_angles_by_channels(model_hfs, nu).items():
                if channels not in angles or max(abs(angles[channels]), abs(angle_hfs)) > np.pi / 4:
                    continue
                n_compared += 1
                assert np.sign(angles[channels]) == np.sign(angle_hfs), (
                    f"The mixing angle between the channels {channels} at {nu=} has a different sign in "
                    f"{model.full_name} ({angles[channels]}) and {model_hfs.full_name} ({angle_hfs})"
                )
    assert n_compared > 0, f"No mixing angles of {species} and {species_hfs} were compared"


# Experimental energies (in 1/cm, Table S27) and theoretical g-factors (Table S28) of the 174Yb 6sng +-G4 states of
# R. Kuroda et al., Phys. Rev. A 112, 042817 (2025) (arXiv:2507.11487). The theoretical g-factors agree with the
# measured ones (1.012(9) to 1.018(4) for +G4, 1.021(4) to 1.031(4) for -G4).
KURODA_G4_G_FACTORS: list[tuple[str, float, float]] = [
    ("6s34g +G4", 50347.996116, 1.0174),
    ("6s34g -G4", 50348.000559, 1.0327),
    ("6s37g +G4", 50362.798462, 1.0174),
    ("6s37g -G4", 50362.801915, 1.0327),
    ("6s41g +G4", 50377.706198, 1.0174),
    ("6s41g -G4", 50377.708743, 1.0327),
]


@pytest.mark.parametrize(("label", "energy", "g_factor"), KURODA_G4_G_FACTORS)
def test_yb174_g4_g_factors(label: str, energy: float, g_factor: float) -> None:
    """The 6sng J=4 states of 174Yb have the published g-factors, which fixes the sign of the +G4/-G4 mixing angle.

    The two J=4 channels 6s_1/2 ng_9/2 and 6s_1/2 ng_7/2 converge to the same ionization threshold, and also in 171Yb
    each of them is part of only a single hyperfine channel, so the sign of the mixing angle between them cannot be
    determined from the energies. As in the publication, it is fixed by the g-factors
    g = (1 - <s_tot>) g(1G4) + <s_tot> g(3G4). With the opposite sign of the mixing angle,
    the g-factors would come out as 1.0267 (+G4) and 1.0234 (-G4).
    """
    g_s = 2.00231930436  # electron spin g-factor
    g_singlet, g_triplet = 1.0, 1 + (g_s - 1) * 2 / 40  # g_J of 1G4 and 3G4 (L = 4, J = 4)

    model = _get_model("Yb174", "G J=4, nu > 25")
    states = _get_states_around_energy(model, energy)
    closest = min(states, key=lambda state: abs(state.get_energy("1/cm") - energy))
    assert abs(closest.get_energy("1/cm") - energy) < 5e-4, f"no state found for {label}"
    triplet_fraction = closest.calc_exp_qn("s_tot")
    assert (1 - triplet_fraction) * g_singlet + triplet_fraction * g_triplet == pytest.approx(g_factor, abs=2e-3)


def _kuroda2025_ket_phase(ket: AngularKetBase[Any]) -> int | None:
    """Phase d of a channel ket of the Yb publications relative to the rydstate ket (None for unknown kets).

    The publications of the Thompson group (M. Peper et al., Phys. Rev. X 15, 011009 (2025) and
    R. Kuroda et al., Phys. Rev. A 112, 042817 (2025)) couple all angular momenta in the opposite order
    than rydstate, i.e. the core electron before the Rydberg electron and the spin before the orbital angular momentum,
    ((s_c s_r)S, (l_c l_r)L)J and ((s_c l_c)j_c, (s_r l_r)j_r)J, with the exception of the nuclear spin (J i_c)F and
    (j_c i_c)f_c. Reversing the coupling order of j_1 + j_2 -> j_12 gives a phase (-1)^(j_1 + j_2 - j_12).
    """
    if any(is_unknown(qn) for qn in ket.quantum_numbers):
        return None
    if isinstance(ket, AngularKetLS):
        exponents = [ket.l_c + ket.l_r - ket.l_tot, ket.s_c + ket.s_r - ket.s_tot, ket.l_tot + ket.s_tot - ket.j_tot]
    elif isinstance(ket, AngularKetJJ):
        exponents = [ket.j_c + ket.j_r - ket.j_tot, ket.l_c + ket.s_c - ket.j_c, ket.l_r + ket.s_r - ket.j_r]
    else:
        assert isinstance(ket, AngularKetFJ)
        exponents = [ket.f_c + ket.j_r - ket.f_tot, ket.l_c + ket.s_c - ket.j_c, ket.l_r + ket.s_r - ket.j_r]
    return int((-1) ** round(sum(exponents)))


@pytest.mark.parametrize(("species", "name", "reference"), REFERENCE_FRAME_TRANSFORMATIONS)
def test_frame_transformation_phase_convention(species: str, name: str, reference: NDArray) -> None:
    """The published frame transformations are reproduced exactly with the phases of _kuroda2025_ket_phase.

    Contrary to test_frame_transformation_matches_reference, this also checks the signs of the individual channels,
    which determine whether the mixing angles of the publications can be used unchanged,
    see test_kuroda2025_mixing_angle_phase_convention.
    """
    model = _get_model(species, name)
    d_outer = np.array([_kuroda2025_ket_phase(ket) or 1 for ket in model.outer_channels])
    d_inner = np.array([_kuroda2025_ket_phase(ket) or 1 for ket in model.inner_channels])
    calculated = d_outer[:, None] * model.calc_frame_transformation_outer_inner() * d_inner[None, :]
    np.testing.assert_allclose(calculated, reference, atol=1e-10, err_msg=model.full_name)


def _get_sign_flipped_mixing_angles(model: EigenChannelModel) -> set[tuple[int, int]]:
    """Return the mixing angles, which have the opposite sign in rydstate than in the publications.

    The kets of the publications are related to the rydstate kets by |i>_paper = d_i |i>_rydstate,
    so the frame transformation of the publications is U_paper = D_outer Q D_inner (with Q the frame transformation
    calculated by rydstate) and the rotation matrix of the mixing angles must be transformed as
    R_rydstate = D_inner R_paper D_inner, i.e. theta_ij changes its sign if d_i d_j = -1.
    The phase of the perturber channels (with unknown quantum numbers) is arbitrary, it is chosen such that as few
    angles as possible change their sign.
    """
    phases = [_kuroda2025_ket_phase(ket) for ket in model.inner_channels]
    angles = [(i, j) for i, j, _coefficients in model.mixing_angles or []]
    unknown = sorted({k for angle in angles for k in angle if phases[k] is None})
    best: set[tuple[int, int]] | None = None
    for choice in itertools.product([1, -1], repeat=len(unknown)):
        phases_choice = list(phases)
        for k, phase in zip(unknown, choice, strict=True):
            phases_choice[k] = phase
        flipped = {(i, j) for i, j in angles if phases_choice[i] * phases_choice[j] == -1}  # type: ignore [operator]
        if best is None or len(flipped) < len(best):
            best = flipped
    return best or set()


# Mixing angles of the Yb models, which are given with the opposite sign than in the publications
# (all other mixing angles are taken over unchanged from the publications):
# (species, model name, (i, j)) -> constant part of the mixing angle in the publication
YB_SIGN_FLIPPED_MIXING_ANGLES: dict[tuple[str, str, tuple[int, int]], float] = {
    # theta_27 of Table S10 of Kuroda 2025 (Phys. Rev. A 112, 042817), between 6snp 3P1 and 3P0
    ("Yb171", "P F=1/2, nu > 5.9", (1, 6)): -0.00168607392,
}


def test_kuroda2025_mixing_angle_phase_convention() -> None:
    """The mixing angles of the Yb models are given in the phase convention of the rydstate kets.

    The mixing angles of the publications refer to their kets, see _kuroda2025_ket_phase. For most mixing angles,
    both involved channels have the same phase, so the angle can be taken over unchanged. The remaining ones must
    have the opposite sign than in the publication and are listed in YB_SIGN_FLIPPED_MIXING_ANGLES.
    This test fails, if a model with such an angle is added, to make sure that its sign is adapted.
    """
    flipped = set()
    for species in ("Yb171", "Yb173", "Yb174"):
        for model in get_mqdt(species).models:
            if isinstance(model, EigenChannelModel):
                flipped |= {(species, model.name, angle) for angle in _get_sign_flipped_mixing_angles(model)}
    assert flipped == set(YB_SIGN_FLIPPED_MIXING_ANGLES)

    for (species, name, angle), angle_paper in YB_SIGN_FLIPPED_MIXING_ANGLES.items():
        coefficients = next(c for i, j, c in _get_model(species, name).mixing_angles or [] if (i, j) == angle)
        assert np.sign(coefficients[0]) == -np.sign(angle_paper), f"{species} {name} {angle}: sign not flipped"


def test_yb171_3p1_hyperfine_splitting() -> None:
    """The low lying 171Yb P models reproduce the hyperfine splitting of the 6s6p 3P1 state.

    The splitting E(F=3/2) - E(F=1/2) = 3/2 A(3P1) ~ 5.94 GHz (A(3P1) ~ 3958 MHz,
    e.g. K. Pandey et al., Phys. Rev. A 80, 022518 (2009)) depends on the sign of the 6snp 1P1 - 3P1 mixing angle,
    which is not determined by the 174Yb energies the models were fitted to. With the correct sign, the models give
    ~6.5 GHz, with the opposite sign ~2.8 GHz.
    """
    energies = {}
    for name in ("P F=1/2, 1.6 < nu < 2.6", "P F=3/2, 1.6 < nu < 2.6"):
        model = _get_model("Yb171", name)
        states = get_mqdt_states_from_model(model, (1.6, 2.6), NotSet, get_potential_class("Yb171"))
        energies[model.f_tot] = min((state.get_energy("1/cm") for state in states), key=lambda e: abs(e - 17992))
    splitting = ureg.Quantity(energies[1.5] - energies[0.5], "1/cm").to("MHz", "spectroscopy").magnitude
    assert splitting == pytest.approx(5937, rel=0.15)


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


def test_sr87_5s5p_singlet_admixture() -> None:
    """The 87Sr 5s5p F=9/2 states have the singlet admixture expected from the intercombination and clock lines.

    The singlet fraction of a state decaying to 5s2 1S0 is approximately A / A(1P1) * (omega(1P1) / omega)^3.
    - 5s5p 3P1: A ~ 4.7e4 1/s (7.4 kHz linewidth), A(1P1) ~ 1.9e8 1/s, i.e. singlet fraction ~ 8e-4,
      the same as for the 88Sr 5s5p 3P1 state.
    - 5s5p 3P0 (87Sr clock state): hyperfine quenched lifetime of the order of 150 s, i.e. singlet fraction ~ 1e-10.
      With the opposite sign of the 1P1 - 3P1 mixing angle the singlet fraction is ~ 8e-12, without mixing ~ 3e-11.
    """
    singlet_fractions = {}
    for species, name in [
        ("Sr88", "P J=1 (recombination), 1.8 < nu < 2.2"),
        ("Sr87", "P F=9/2 (clock), 1.8 < nu < 2.2"),
    ]:
        model = _get_model(species, name)
        states = get_mqdt_states_from_model(model, (1.8, 2.2), NotSet, get_potential_class(species))
        for state in states:
            j_tot = round(state.calc_exp_qn("j_tot"))
            if round(state.calc_exp_qn("s_tot")) == 1:
                singlet_fractions[species, j_tot] = 1 - state.calc_exp_qn("s_tot")

    assert singlet_fractions["Sr87", 1] == pytest.approx(singlet_fractions["Sr88", 1], rel=0.01)
    assert singlet_fractions["Sr88", 1] == pytest.approx(8e-4, rel=0.2)
    assert 0.5e-10 < singlet_fractions["Sr87", 0] < 3e-10


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
    if model.name == "P F=3/2, 2.6 < nu < 10":
        pytest.skip(f"{model.full_name}: TODO: the mixing angle is not well defined for this model, fix this!")
    for nu in np.linspace(model.nu_min, model.nu_max, 200):
        nui_0 = float(model.calc_channel_nuis(nu)[0])
        for i, j, coefficients in model.mixing_angles or []:
            angle = calc_modified_ritz_formula_in_nu(nui_0, coefficients)
            assert abs(angle) < np.pi / 2, f"{model.full_name}: mixing angle ({i}, {j}) = {angle} at {nu=}"


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


TRIPLET_F_MODEL_NAMES = {
    2: "F J=2 (Rydberg-Ritz), nu > 9",
    3: "F J=3 triplet (Rydberg-Ritz), nu > 9",
    4: "F J=4 (Rydberg-Ritz), nu > 9",
}


def _get_nist_triplet_f_levels(j_tot: int) -> list[tuple[int, float]]:
    """Return (n, NIST term energy in 1/cm) of the Sr88 5snf 3F_J levels with 10 <= n <= 20."""
    sqdt = get_sqdt("Sr88")
    hartree_to_inverse_cm = ureg.Quantity(1, "hartree").to("1/cm", "spectroscopy").magnitude
    return [
        (n, energy_au * hartree_to_inverse_cm)
        for (n, l_r, j, s_tot), energy_au in sorted(sqdt._nist_energy_levels.items())  # noqa: SLF001
        if l_r == 3 and j == j_tot and s_tot == 1 and 10 <= n <= 20
    ]


@pytest.mark.parametrize("j_tot", [2, 3, 4])
def test_vaillant2024_triplet_f_energies_match_nist(j_tot: int) -> None:
    """The single channel models of the 5snf 3F_J series reproduce the NIST levels with n = 10-20.

    The Rydberg-Ritz quantum defects of Robertson 2021 (ARC 3.0) reproduce these levels to < 0.15 1/cm
    (rms 0.04-0.06 1/cm, comparable to the NIST uncertainty of 0.04 1/cm). The older quantum defects of
    Vaillant 2012 (same parameters for 3F2 and 3F3) and the rounded values of table B.1 of Robertson 2021
    deviate by 0.22-0.28 1/cm at n = 10 for the 3F2 or 3F3 series and fail this test.
    """
    model = next(
        model for model in get_mqdt("Sr88", "vaillant2024").models if model.name == TRIPLET_F_MODEL_NAMES[j_tot]
    )
    levels = _get_nist_triplet_f_levels(j_tot)
    assert len(levels) >= 10, f"only {len(levels)} NIST levels found for 5snf 3F{j_tot}"
    for n, energy_nist in levels:
        states = _get_states_around_energy(model, energy_nist)
        assert len(states) > 0, f"{model.full_name}: no states found around E={energy_nist} 1/cm"
        closest = min(states, key=lambda state: abs(state.get_energy("1/cm") - energy_nist))
        assert closest.get_energy("1/cm") == pytest.approx(energy_nist, abs=0.15), (
            f"{model.full_name}: the calculated energy {closest.get_energy('1/cm')} 1/cm of 5s{n}f 3F{j_tot} "
            f"does not match the NIST energy {energy_nist} 1/cm"
        )


def _vaillant2024_ket_phase(ket: AngularKetBase[Any]) -> int:
    """Phase d_i of a channel ket of Vaillant 2024 relative to the rydstate ket (see the model data docstring)."""
    if isinstance(ket, AngularKetJJ):
        return int((-1) ** round(ket.j_c + ket.j_r - ket.j_tot))
    assert isinstance(ket, AngularKetLS)
    return int((-1) ** round(ket.l_c + ket.l_r - ket.l_tot) * (-1) ** round(1 - ket.s_tot))


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
    k_model = {(i, j): coefficients[0] for i, j, coefficients in model.k_matrix_entries}
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
        np.testing.assert_allclose(thresholds, expected[model.name], rtol=0, atol=1e-6, err_msg=model.full_name)
