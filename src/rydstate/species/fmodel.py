from __future__ import annotations

import inspect
import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np

from rydstate.angular.utils import is_not_set
from rydstate.species.element_properties import get_element_properties
from rydstate.species.utils import calc_energy_from_nu, calc_modified_ritz_formula_in_nu, calc_nu_from_energy

if TYPE_CHECKING:
    from types import ModuleType

    from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.angular.utils import AllKnown
    from rydstate.species.mqdt import MQDT
    from rydstate.species.utils import RydbergRitzParameters
    from rydstate.units import NDArray, PintFloat


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

    outer_channels: ClassVar[list[AngularKetBase[Any]]]
    """List of outer channels in the MQDT model."""

    eigen_quantum_defects: ClassVar[list[RydbergRitzParameters]]
    """List of eigen quantum defects for the close-coupling channels.
    Each entry can be a constant or a list of polynomial coefficients."""

    mixing_angles: ClassVar[list[tuple[int, int, RydbergRitzParameters]] | None] = None
    """List of mixing angles between close-coupling channels.
    Each entry is a tuple (i_idx, j_idx, params) where i_idx and j_idx are the indices of the involved channels
    and params are the parameters for the energy dependence of the angle (constant or polynomial coefficients).
    The default None means no mixing between the close-coupling channels."""

    def __init__(self, mqdt: MQDT) -> None:
        self.mqdt = mqdt
        self.element_properties = get_element_properties(self.species)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.full_name})"

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

    @cached_property
    def fj_channels(self) -> list[AngularKetFJ[Any]]:
        """Return a list of FJ channels in the model."""
        return [ket_fj for angular_ket in self.outer_channels for ket_fj in angular_ket.to_state("FJ").kets]

    def get_core_kets(self) -> list[CoreKet]:
        """Return a list of relevant core kets of the model."""
        core_kets = {channel.get_core_ket() for channel in self.outer_channels}
        return sorted(core_kets, key=lambda ket: (ket.l_c, ket.j_c, ket.f_c, str(ket.label)))

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
        which is defined with reference to the reference ionization threshold of the MQDT model,
        see :attr:`~rydstate.species.mqdt.MQDT.reference_ionization_threshold_au`.
        """
        return (
            calc_energy_from_nu(self.element_properties.reduced_mass_au, nu, self.element_properties.net_charge)
            + self.mqdt.reference_ionization_threshold_au
        )

    def calc_channel_nuis(self, nu: float) -> NDArray:
        r"""Return the channel-dependent effective principal quantum numbers nui.

        The channel dependent effective principal quantum numbers nui are defined via

        .. math::
            E = I_i - \frac{Ry}{2 \nu_i^2} = I_{\text{ref}} - \frac{Ry}{nu^2}

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            List of channel nui values.

        """
        reduced_mass_au = self.element_properties.reduced_mass_au
        net_charge = self.element_properties.net_charge
        reference_threshold_au = self.mqdt.reference_ionization_threshold_au
        # we calculate binding_energy_au here directly from nu (and dont use calc_energy_au) to avoid numerical issues
        binding_energy_au = calc_energy_from_nu(reduced_mass_au, nu, net_charge)
        nuis = [
            calc_nu_from_energy(reduced_mass_au, binding_energy_au - (threshold - reference_threshold_au), net_charge)
            for threshold in self.ionization_thresholds_au
        ]
        return np.array(nuis)

    def calc_eigen_quantum_defects(self, nu: float) -> NDArray:
        r"""Return the eigen quantum defects evaluated at the channel-dependent effective principal quantum numbers nui.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

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
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            Diagonal K-matrix in the close-coupling frame.

        """
        return np.diag(np.tan(np.pi * self.calc_eigen_quantum_defects(nu)))

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
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            Unitary transformation matrix R (n_inner, n_closecoupling).

        """
        n = len(self.inner_channels)
        rot = np.eye(n)
        if self.mixing_angles is None:
            return rot
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
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            Frame transformation matrix U = Q R (n_outer, n_closecoupling).

        """
        if self.mixing_angles is None:
            return self.frame_transformation_outer_inner
        return self.frame_transformation_outer_inner @ self.calc_frame_transformation_inner_closecoupling(nu)

    def calc_k_matrix(self, nu: float) -> NDArray:
        r"""Return the K-matrix in the collision (outer) channel frame.

        The K-matrix is defined as

        .. math::
            K = tan(\pi \mu) = U tan(\pi \mu_{\alpha}) U^T

        where U is the frame transformation matrix and \mu_{\alpha} are the eigen quantum defects.
        The transpose :math:`U^T = U^{-1}` holds because U is real and orthogonal.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

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
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            M-matrix in the collision (outer) channel frame, M = tan(β) + K.

        """
        kmat = self.calc_k_matrix(nu)
        nuis = self.calc_channel_nuis(nu)
        return np.diag(np.tan(np.pi * nuis)) + kmat

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
    def __init__(self, species: str, channel: AngularKetBase[AllKnown], mqdt: MQDT) -> None:
        if not is_not_set(channel.m):
            raise ValueError("The m quantum number of the channel must be NotSet.")
        self.species = species  # type: ignore [misc]
        self.name = f"SQDT {channel}, nu >= {channel.l_r + 1}"  # type: ignore [misc]
        self.f_tot = channel.f_tot  # type: ignore [misc]
        self.nu_range = (channel.l_r + 1, math.inf)  # type: ignore [misc]
        self.inner_channels = [channel]  # type: ignore [misc]
        self.outer_channels = [channel]  # type: ignore [misc]
        self.eigen_quantum_defects = [0]  # type: ignore [misc]

        super().__init__(mqdt)

    def calc_scaled_m_matrix(self, nu: float) -> NDArray:
        # Fast path for SQDT models: the single channel has a vanishing quantum defect, so K = 0 and the
        # scaled M-matrix reduces to the 1x1 matrix sin(pi * nui) (see FModel.calc_scaled_m_matrix).
        nui = self.calc_channel_nuis(nu)[0]
        return np.array([[math.sin(math.pi * nui)]])
