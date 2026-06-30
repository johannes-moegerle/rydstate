from __future__ import annotations

import inspect
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np

from rydstate.species.element_properties import get_element_properties
from rydstate.species.utils import calc_energy_from_nu, calc_modified_ritz_formula_in_nu, calc_nu_from_energy

if TYPE_CHECKING:
    from types import ModuleType

    from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetJJ, AngularKetLS
    from rydstate.angular.utils import AllKnown
    from rydstate.species.mqdt import MQDT
    from rydstate.species.utils import RydbergRitzParameters
    from rydstate.units import NDArray, PintFloat

logger = logging.getLogger(__name__)


class FModel:
    """Class to store the parameters of a MQDT model for a given species."""

    species: ClassVar[str]
    """The species for which the MQDT model is defined."""
    name: ClassVar[str]
    """The name of the atomic species."""

    reference: ClassVar[str | tuple[str, ...] | None] = None
    """Reference for the MQDT model, e.g., a publication doi where the model is described."""

    f_tot: ClassVar[float]
    """Total angular momentum f_tot of the Rydberg state."""

    nu_range: ClassVar[tuple[float, float]]
    """Range of effective principal quantum numbers nu for which the MQDT model is valid."""

    inner_channels: ClassVar[list[AngularKetBase[Any]]]
    """List of inner channels in the MQDT model."""

    outer_channels: ClassVar[list[AngularKetFJ[Any] | AngularKetJJ[Any] | AngularKetLS[Any]]]
    """List of outer channels in the MQDT model."""

    eigen_quantum_defects: ClassVar[list[RydbergRitzParameters]]
    """List of eigen quantum defects for the close-coupling channels.
    Each entry can be a constant or a list of polynomial coefficients."""

    mixing_angles: ClassVar[list[tuple[int, int, RydbergRitzParameters]]]
    """List of mixing angles between close-coupling channels.
    Each entry is a tuple (i_idx, j_idx, params) where i_idx and j_idx are the indices of the involved channels
    and params are the parameters for the energy dependence of the angle (constant or polynomial coefficients)."""

    def __init__(self, mqdt: MQDT) -> None:
        self.mqdt = mqdt
        self.element_properties = get_element_properties(self.species)

    @property
    def full_name(self) -> str:
        """Return the full name of the model, combining species and model name."""
        return f"{self.species} {self.name}"

    @property
    def nu_min(self) -> float:
        """Minimum nu for which the model is valid."""
        return self.nu_range[0]

    @property
    def nu_max(self) -> float:
        """Maximum nu for which the model is valid."""
        return self.nu_range[1]

    @overload
    def get_ionization_thresholds(self, unit: None = None) -> list[PintFloat]: ...

    @overload
    def get_ionization_thresholds(self, unit: str) -> list[float]: ...

    def get_ionization_thresholds(self, unit: str | None = "hartree") -> list[PintFloat] | list[float]:
        """Return the ionization thresholds for all channels.

        Args:
            unit: Desired unit for the ionization thresholds. Default is atomic units "hartree".

        Returns:
            List of ionization thresholds in the desired unit.

        """
        return [self.mqdt.get_ionization_threshold(ket.get_core_ket(), unit=unit) for ket in self.outer_channels]  # type: ignore [return-value]

    @cached_property  # don't remove this caching without benchmarking it!!!
    def ionization_thresholds_au(self) -> list[float]:
        """Return the ionization thresholds for all channels in atomic units."""
        return self.get_ionization_thresholds(unit="hartree")

    def calc_energy_au(self, nu: float) -> float:
        """Calculate the energy of the Rydberg state.

        The energy is calculated for an effective principal quantum number nu,
        which is defined with reference to the lowest ionization threshold of the MQDT model.
        """
        return (
            calc_energy_from_nu(self.element_properties.reduced_mass_au, nu, self.element_properties.net_charge)
            + self.mqdt.reference_ionization_energy_au
        )

    def calc_channel_nuis(self, nu: float) -> NDArray:
        r"""Return the channel-dependent effective principal quantum numbers nui.

        The channel dependent effective principal quantum numbers nui are defined via

        .. math::
            E = I_i - \frac{Ry}{2 \nu_i^2} = I_{\text{ref}} - \frac{Ry}{nu^2}

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            List of channel nui values.

        """
        energy_au = self.calc_energy_au(nu)
        reduced_mass_au = self.element_properties.reduced_mass_au
        net_charge = self.element_properties.net_charge
        nuis = [
            calc_nu_from_energy(reduced_mass_au, energy_au - threshold, net_charge)
            for threshold in self.ionization_thresholds_au
        ]
        return np.array(nuis)

    def calc_eigen_quantum_defects(self, nu: float) -> NDArray:
        r"""Return the eigen quantum defects evaluated at the channel-dependent effective principal quantum numbers nui.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Array of eigen quantum defects.

        """
        nuis = self.calc_channel_nuis(nu)
        eigen_quantum_defects = [
            calc_modified_ritz_formula_in_nu(nui, params)
            for nui, params in zip(nuis, self.eigen_quantum_defects, strict=True)
        ]
        return np.array(eigen_quantum_defects)

    def calc_k_matrix_closecoupling(self, nu: float) -> NDArray:
        r"""Return diagonal K-matrix in the close-coupling frame.

        Diagonal entries are tan(\pi * \mu_\alpha) where \mu_\alpha are the eigen quantum defects
        evaluated at the channel-dependent effective principal quantum numbers nui.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Diagonal K-matrix in the close-coupling frame.

        """
        return np.diag(np.tan(np.pi * np.array(self.calc_eigen_quantum_defects(nu))))

    def calc_frame_transformation_outer_inner(self) -> NDArray:
        """Return the frame transformation matrix Q mapping inner to outer channels.

        Computed from the overlaps (Wigner coefficients) between inner_channels and outer_channels.

        Returns:
            Unitary transformation matrix Q (n_outer, n_inner).

        """
        n = len(self.inner_channels)
        u = np.zeros((n, n))

        for i, outer in enumerate(self.outer_channels):
            for j, inner in enumerate(self.inner_channels):
                u[i, j] = outer.calc_reduced_overlap(inner)

        return u

    @cached_property  # don't remove this caching without benchmarking it!!!
    def frame_transformation_outer_inner(self) -> NDArray:
        """Cached version of calc_frame_transformation_outer_inner."""
        return self.calc_frame_transformation_outer_inner()

    def calc_frame_transformation_inner_closecoupling(self, nu: float) -> NDArray:
        """Return the frame transformation matrix R mapping close-coupling to inner channels.

        Computed as rotation matrix from the mixing angles.
        Applies successive 2x2 rotations between the channels specified by mixing_angles.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Unitary transformation matrix R (n_inner, n_closecoupling).

        """
        n = len(self.inner_channels)
        if len(self.mixing_angles) == 0:
            return np.eye(n)
        # Find reference channel nu for energy-dependent angles
        # convention: first involved channel of first energy-dependent mixing entry
        ref_nu: float | None = None
        for i_idx, _j_idx, params in self.mixing_angles:
            if isinstance(params, list) and len(params) > 1:
                nuis = self.calc_channel_nuis(nu)
                ref_nu = float(nuis[i_idx])
                break
        if ref_nu is None:
            ref_nu = 0.0  # unused; angles are constant
        rot = np.eye(n)
        for i_idx, j_idx, params in self.mixing_angles:
            angle = calc_modified_ritz_formula_in_nu(ref_nu, params)
            r = np.eye(n)
            r[i_idx, i_idx] = np.cos(angle)
            r[i_idx, j_idx] = -np.sin(angle)
            r[j_idx, i_idx] = np.sin(angle)
            r[j_idx, j_idx] = np.cos(angle)
            rot = rot @ r
        return rot

    def calc_frame_transformation(self, nu: float) -> NDArray:
        """Return the full frame transformation U from close-coupling to outer channel frame.

        Combines the unitary frame transformation Q with the rotation matrix R.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Frame transformation matrix U = Q R (n_outer, n_closecoupling).

        """
        return self.frame_transformation_outer_inner @ self.calc_frame_transformation_inner_closecoupling(nu)

    def calc_k_matrix(self, nu: float) -> NDArray:
        r"""Return the K-matrix in the collision (outer) channel frame.

        The K-matrix is defined as

        .. math::
            K = tan(\pi \mu) = U tan(\pi \mu_{\alpha}) U^T

        where U is the frame transformation matrix and \mu_{\alpha} are the eigen quantum defects.
        The transpose :math:`U^T = U^{-1}` holds because U is real and orthogonal.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            K-matrix in the collision (outer) channel frame, K = tan(\pi \mu).

        """
        transform = self.calc_frame_transformation(nu)
        kbar = self.calc_k_matrix_closecoupling(nu)
        return transform @ kbar @ transform.T

    def calc_m_matrix(self, nu: float) -> NDArray:
        r"""Return the M-matrix in the collision (outer) channel frame.

        The M-matrix is defined as

        .. math::
            M = tan(β) + K = tan(\pi \nu) + tan(\pi \mu)

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            M-matrix in the collision (outer) channel frame, M = tan(β) + K.

        """
        kmat = self.calc_k_matrix(nu)
        nuis = self.calc_channel_nuis(nu)
        return np.array(np.diag(np.tan(np.pi * nuis)) + kmat)

    def calc_scaled_m_matrix(self, nu: float) -> NDArray:
        r"""Return the scaled M-matrix in the collision (outer) channel frame.

        The scaled M-matrix is defined as

        .. math::
            M_{\text{scaled}} = \cos(\pi \nu) M = \sin(\pi \nu) + \cos(\pi \nu) K

        We use this to improve numerical stability when finding roots of det(M) = 0.
        This is especially important for states with nu close to half integer.
        """
        kmat = self.calc_k_matrix(nu)
        nuis = self.calc_channel_nuis(nu)
        return np.array(np.diag(np.sin(np.pi * nuis)) + np.diag(np.cos(np.pi * nuis)) @ kmat)

    def calc_det_m_matrix(self, nu: float) -> float:
        """Calculate the determinant of the M-matrix at a given nu value.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Determinant of the M-matrix at the given nu value.

        """
        return float(np.linalg.det(self.calc_m_matrix(nu)))

    def calc_det_scaled_m_matrix(self, nu: float) -> float:
        """Calculate the determinant of the scaled M-matrix at a given nu value.

        Args:
            nu: Effective principal quantum number with reference to the lowest ionization threshold.

        Returns:
            Determinant of the scaled M-matrix at the given nu value.

        """
        return float(np.linalg.det(self.calc_scaled_m_matrix(nu)))

    def calc_nu(self, n: int, angular_ket: AngularKetFJ[Any], prev_nus: list[float] | None = None) -> float:
        ind = self.outer_channels.index(angular_ket)

        prev_nus = [n] if prev_nus is None else prev_nus
        nu_ref = _calc_nu_ref(self.mqdt, angular_ket, prev_nus[-1])

        trafo = self.calc_frame_transformation(nu_ref)
        mu_closecoupling = np.diag(self.calc_eigen_quantum_defects(nu_ref))
        mu_fj = trafo @ mu_closecoupling @ trafo.T

        mu1 = mu_fj[ind, ind]

        k = trafo @ np.tan(np.pi * mu_closecoupling) @ trafo.T
        tan_mu = k[ind, ind]
        mu = np.arctan(tan_mu) / np.pi
        # TODO this is a bit hacky
        mu += round(mu1)

        nu = float(n - mu)
        prev_nus.append(nu)
        if abs(prev_nus[-1] - prev_nus[-2]) < 1e-3:
            return nu

        if len(prev_nus) > 10:
            logger.warning(
                "calc_nu did not converge for %s at %d after %s iterations, returning last value %f.",
                *(angular_ket, n, prev_nus, nu),
            )
            return nu

        return self.calc_nu(n, angular_ket, prev_nus=prev_nus)


def _calc_nu_ref(mqdt: MQDT, angular_ket: AngularKetFJ[Any], prev_nu: float) -> float:
    element_properties = get_element_properties(mqdt.species)
    core_energy = mqdt.get_ionization_threshold(angular_ket.get_core_ket(), unit="hartree")
    reference_core_energy = mqdt.reference_ionization_energy_au

    eps_i = calc_energy_from_nu(element_properties.reduced_mass_au, prev_nu)
    # E_tot = I_i + eps_i = I_ref + eps_ref with eps_i = -1 / (2 * (nu_i^2))
    # => eps_ref = eps_i + I_i - I_ref
    eps_ref = eps_i + core_energy - reference_core_energy
    if eps_ref >= 0:
        # channel state is above reference ionization threshold
        return 120.0  # large nu_ref value for continuum states
        # TODO larger value breaks for Yb171

    return calc_nu_from_energy(element_properties.reduced_mass_au, eps_ref)


def get_fmodels(module: ModuleType, species: str) -> list[type[FModel]]:
    """Return all FModel subclasses defined in ``module`` that match the given species.

    Args:
        module: The module to inspect for FModel subclasses.
        species: The species the returned FModel subclasses must match.

    Returns:
        List of all FModel subclasses defined in the module for the given species.

    """
    fmodels = [
        obj
        for obj in vars(module).values()
        if (
            inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, FModel)
            and obj is not FModel
            and getattr(obj, "species", None) == species
        )
    ]
    if len(fmodels) == 0:
        raise ValueError(f"No FModel subclasses for species {species!r} found in {module.__name__}.")
    return fmodels


class FModelSQDT(FModel):
    def __init__(self, species: str, channel: AngularKetFJ[AllKnown], mqdt: MQDT) -> None:
        self.species = species  # type: ignore [misc]
        self.name = f"SQDT l_r={channel.l_r}, j_r={channel.j_r}, f_tot={channel.f_tot}, nu > {channel.l_r + 1}"  # type: ignore [misc]
        self.f_tot = channel.f_tot  # type: ignore [misc]
        self.nu_range = (channel.l_r + 1, math.inf)  # type: ignore [misc]
        self.inner_channels = [channel]  # type: ignore [misc]
        self.outer_channels = [channel]  # type: ignore [misc]
        self.eigen_quantum_defects = [0]  # type: ignore [misc]
        self.mixing_angles = []  # type: ignore [misc]

        super().__init__(mqdt)
