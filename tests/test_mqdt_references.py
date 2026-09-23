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
from rydstate.species import EigenChannelModel, get_mqdt, get_potential_class
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
