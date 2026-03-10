from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, overload

import numpy as np

from rydstate.angular.utils import is_unknown
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy
from rydstate.utils.linalg import find_roots

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ
    from rydstate.angular.angular_ket_dummy import AngularKetDummy
    from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT
    from rydstate.species.utils import RydbergRitzParameters
    from rydstate.units import NDArray, PintFloat


class FModel:
    """Class to store the parameters of a MQDT model for a given species."""

    reference: ClassVar[str | None] = None
    """Reference for the MQDT model, e.g., a publication doi where the model is described."""

    species_name: ClassVar[str]
    """The species for which the MQDT model is defined."""

    name: ClassVar[str | None]
    """The name of the atomic species."""

    f_tot: ClassVar[float]
    """Total angular momentum f_tot of the Rydberg state."""

    nu_range: ClassVar[tuple[float, float]]
    """Range of effective principal quantum numbers nu for which the MQDT model is valid."""

    inner_channels: ClassVar[list[AngularKetBase]]
    """List of inner channels in the MQDT model."""
    outer_channels: ClassVar[list[AngularKetFJ | AngularKetDummy]]
    """List of outer channels in the MQDT model."""

    eigen_quantum_defects: ClassVar[list[RydbergRitzParameters]]
    mixing_angles: ClassVar[list[tuple[int, int, RydbergRitzParameters]]]

    manual_frame_transformation_outer_inner: ClassVar[NDArray | None] = None
    """Optional manually specified frame transformation matrix Q mapping inner to outer channels.
    This is mainly needed for models with unknown quantum numbers,
    where the frame transformation cannot (yet) be computed from Wigner coefficients.
    """

    @property
    def nu_min(self) -> float:
        """Minimum nu for which the model is valid."""
        return self.nu_range[0]

    @property
    def nu_max(self) -> float:
        """Maximum nu for which the model is valid."""
        return self.nu_range[1]

    @cached_property
    def species(self) -> SpeciesObjectMQDT:
        """Return the SpeciesObjectMQDT associated with this model."""
        from rydstate.species.mqdt.species_object_mqdt import SpeciesObjectMQDT  # noqa: PLC0415

        return SpeciesObjectMQDT.from_name(self.species_name)

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
        return [self.species.get_ionization_threshold(ket.get_core_ket(), unit=unit) for ket in self.outer_channels]  # type: ignore [return-value]

    @cached_property  # don't remove this caching without benchmarking it!!!
    def ionization_thresholds_au(self) -> list[float]:
        """Return the ionization thresholds for all channels in atomic units."""
        return self.get_ionization_thresholds(unit="hartree")

    def calc_channel_nuis(self, nu_ref: float) -> NDArray:
        r"""Return the channel-dependent effective principal quantum numbers nui.

        The channel dependent effective principal quantum numbers nui are defined via

        .. math::
            E = I_i - \frac{Ry}{2 \nu_i^2} = I_{\text{ref}} - \frac{Ry}{nu_ref^2}

        Args:
            nu_ref: Reference effective principal quantum number.

        Returns:
            List of channel nui values.

        """
        thresholds = self.ionization_thresholds_au
        ioniz_ref = self.species.reference_ionization_energy_au
        reduced_mass_au = self.species.reduced_mass_au
        energy_from_nu_ref = calc_energy_from_nu(reduced_mass_au, nu_ref)
        nuis = [
            calc_nu_from_energy(reduced_mass_au, ioniz_ref + energy_from_nu_ref - threshold) for threshold in thresholds
        ]
        return np.array(nuis)

    def calc_k_matrix_closecoupling(self, nu_ref: float) -> NDArray:
        r"""Return diagonal K-matrix in the close-coupling frame.

        Diagonal entries are tan(\pi * \mu_\alpha) where \mu_\alpha are the eigen quantum defects
        evaluated at the channel-dependent effective principal quantum numbers nui.

        Args:
            nu_ref: Reference effective principal quantum number.

        Returns:
            Diagonal K-matrix in the close-coupling frame.

        """
        nuis = self.calc_channel_nuis(nu_ref)
        defects = [
            calc_energy_dependent_quantity(nu, params)
            for nu, params in zip(nuis, self.eigen_quantum_defects, strict=True)
        ]
        return np.diag(np.tan(np.pi * np.array(defects)))

    def calc_frame_transformation_outer_inner(self) -> NDArray:
        """Return the frame transformation matrix Q mapping inner to outer channels.

        Computed from the overlaps (Wigner coefficients) between inner_channels and outer_channels.

        Returns:
            Unitary transformation matrix Q (n_outer, n_inner).

        """
        if self.manual_frame_transformation_outer_inner is not None:
            return self.manual_frame_transformation_outer_inner

        n = len(self.inner_channels)
        u = np.zeros((n, n))
        for i, outer in enumerate(self.outer_channels):
            for j, inner in enumerate(self.inner_channels):
                if any(is_unknown(qn) for qn in outer.quantum_numbers) or any(
                    is_unknown(qn) for qn in inner.quantum_numbers
                ):
                    raise ValueError(
                        "Cannot compute frame transformation for channels with unknown quantum numbers."
                        f"Please provide a manual frame transformation matrix for {self.name}."
                    )
                u[i, j] = outer.calc_reduced_overlap(inner)
        return u

    @cached_property  # don't remove this caching without benchmarking it!!!
    def frame_transformation_outer_inner(self) -> NDArray:
        """Cached version of calc_frame_transformation_outer_inner."""
        return self.calc_frame_transformation_outer_inner()

    def calc_frame_transformation_inner_closecoupling(self, nu_ref: float) -> NDArray:
        """Return the frame transformation matrix R mapping close-coupling to inner channels.

        Computed as rotation matrix from the mixing angles.
        Applies successive 2x2 rotations between the channels specified by mixing_angles.

        Args:
            nu_ref: Reference effective principal quantum number.

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
                nuis = self.calc_channel_nuis(nu_ref)
                ref_nu = float(nuis[i_idx])
                break
        if ref_nu is None:
            ref_nu = 0.0  # unused; angles are constant
        rot = np.eye(n)
        for i_idx, j_idx, params in self.mixing_angles:
            angle = calc_energy_dependent_quantity(ref_nu, params)
            r = np.eye(n)
            r[i_idx, i_idx] = np.cos(angle)
            r[i_idx, j_idx] = -np.sin(angle)
            r[j_idx, i_idx] = np.sin(angle)
            r[j_idx, j_idx] = np.cos(angle)
            rot = rot @ r
        return rot

    def calc_frame_transformation(self, nu_ref: float) -> NDArray:
        """Return the full frame transformation U from close-coupling to outer channel frame.

        Combines the unitary frame transformation Q with the rotation matrix R.

        Args:
            nu_ref: Reference effective principal quantum number.

        Returns:
            Frame transformation matrix U = Q R (n_outer, n_closecoupling).

        """
        return self.frame_transformation_outer_inner @ self.calc_frame_transformation_inner_closecoupling(nu_ref)

    def calc_k_matrix(self, nu_ref: float) -> NDArray:
        r"""Return the K-matrix in the collision (outer) channel frame.

        The K-matrix is defined as

        .. math::
            K = tan(\pi \mu) = U tan(\pi \mu_{\alpha}) U^T

        where U is the frame transformation matrix and \mu_{\alpha} are the eigen quantum defects.

        Args:
            nu_ref: Reference effective principal quantum number.

        Returns:
            K-matrix in the collision (outer) channel frame, K = tan(\pi \mu).

        """
        transform = self.calc_frame_transformation(nu_ref)
        kbar = self.calc_k_matrix_closecoupling(nu_ref)
        return transform @ kbar @ transform.T

    def calc_m_matrix(self, nu_ref: float) -> NDArray:
        r"""Return the M-matrix in the collision (outer) channel frame.

        The M-matrix is defined as

        .. math::
            M = tan(β) + K = tan(\pi \nu) + tan(\pi \mu)

        Args:
            nu_ref: Reference effective principal quantum number.

        Returns:
            M-matrix in the collision (outer) channel frame, M = tan(β) + K.

        """
        kmat = self.calc_k_matrix(nu_ref)
        nuis = self.calc_channel_nuis(nu_ref)
        return np.array(np.diag(np.tan(np.pi * nuis)) + kmat)

    def _calc_scaled_m_matrix(self, nu_ref: float) -> NDArray:
        # we scale the M-matrix by cos(pi * nui) to improve numerical stability when finding roots of det(M) = 0
        # this is especially important for states with nu close to half integer
        kmat = self.calc_k_matrix(nu_ref)
        nuis = self.calc_channel_nuis(nu_ref)
        return np.array(np.diag(np.sin(np.pi * nuis)) + np.cos(np.pi * nuis) * kmat)

    def calc_detm_roots(self, nu_min: float, nu_max: float) -> list[float]:
        """Find zeros of det(mmat(nu_ref)) in [nu_min, nu_max] using scipy root finding.

        Args:
            nu_min: Lower bound of the search range.
            nu_max: Upper bound of the search range.

        Returns:
            List of nu_ref values where det(mmat) ≈ 0.

        """

        def det_mmat(nu: float) -> float:
            return float(np.linalg.det(self._calc_scaled_m_matrix(nu)))

        return find_roots(det_mmat, nu_min, nu_max)


class FModelSQDT(FModel):
    def __init__(self, species_name: str, channel: AngularKetFJ) -> None:
        self.species_name = species_name  # type: ignore [misc]
        self.name = f"SQDT l_r={channel.l_r}, j_r={channel.j_r}, f_tot={channel.f_tot}, nu > {channel.l_r + 1}"  # type: ignore [misc]
        self.f_tot = channel.f_tot  # type: ignore [misc]
        self.nu_range = (channel.l_r + 1, math.inf)  # type: ignore [misc]
        self.inner_channels = [channel]  # type: ignore [misc]
        self.outer_channels = [channel]  # type: ignore [misc]
        self.eigen_quantum_defects = [0]  # type: ignore [misc]
        self.mixing_angles = []  # type: ignore [misc]


def calc_energy_dependent_quantity(nui: float, qantity: RydbergRitzParameters) -> float:
    """Evaluate a quantity at channel nu using polynomial convention: q₀ + q₁/ν² + q₂/ν⁴ + ...

    Args:
        nui: Channel-dependent effective principal quantum number.
        qantity: Quantity parameters. A single float is a constant value; a list gives
            polynomial coefficients [q₀, q₁, q₂, ...].

    Returns:
        Quantity value at the given nu.

    """
    if isinstance(qantity, (int, float)):
        return float(qantity)
    result = 0.0
    for i, coeff in enumerate(qantity):
        result += float(coeff) * (1.0 / nui**2) ** i
    return result
