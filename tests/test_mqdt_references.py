from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import RydbergStateSQDTDivalent
from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import get_mqdt_states_from_fmodel
from rydstate.species import get_mqdt, get_potential_class

if TYPE_CHECKING:
    from rydstate.species import FModel
    from rydstate.units import NDArray


def _get_model(species: str, name: str) -> FModel:
    """Return the model of the given species with the given name."""
    return next(model for model in get_mqdt(species).models if model.name == name)


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
    states = get_mqdt_states_from_fmodel(model, nu_range, NotSet, get_potential_class("Yb174"))
    assert len(states) > 0, f"{model.full_name}: no states found around nu={nu_experimental}"

    closest = min(states, key=lambda state: abs(state.nu - nu_experimental))
    assert abs(closest.nu - nu_experimental) < 1e-3, (
        f"{model.full_name}: the calculated nu={closest.nu} does not match "
        f"the experimental nu={nu_experimental} of the {n=}, {l_r=}, {j_tot=}, {s_tot=} level"
    )
