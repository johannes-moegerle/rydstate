from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from rydstate.angular.utils import is_not_set
from rydstate.species.mqdt_model import MQDTModel
from rydstate.species.utils import calc_modified_ritz_formula_in_nu

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.angular.utils import AllKnown
    from rydstate.species.mqdt import MQDT
    from rydstate.species.utils import RydbergRitzParameters
    from rydstate.units import NDArray


class EigenChannelModel(MQDTModel):
    """MQDT model formulated in terms of eigenchannels and a frame transformation.

    The K-matrix is diagonal in the close-coupling (eigenchannel) frame, with the eigen quantum defects
    on its diagonal. It is transformed to the outer channel frame via the frame transformation
    :math:`U = Q R`, where Q is given by the overlaps between inner and outer channels and R is the
    rotation given by the mixing angles between the close-coupling channels,
    see :meth:`calc_frame_transformation`.
    """

    inner_channels: ClassVar[list[AngularKetBase[Any]]]
    """List of inner channels in the MQDT model."""

    eigen_quantum_defects: ClassVar[list[RydbergRitzParameters]]
    """List of eigen quantum defects for the close-coupling channels.
    Each entry can be a constant or a list of polynomial coefficients."""

    mixing_angles: ClassVar[list[tuple[int, int, RydbergRitzParameters]] | None] = None
    """List of mixing angles between close-coupling channels.
    Each entry is a tuple (i_idx, j_idx, params) where i_idx and j_idx are the indices of the involved channels
    and params are the parameters for the energy dependence of the angle (constant or polynomial coefficients).
    The default None means no mixing between the close-coupling channels."""

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
        r"""Return the K-matrix in the outer channel frame.

        The K-matrix is defined as

        .. math::
            K = tan(\pi \mu) = U tan(\pi \mu_{\alpha}) U^T

        where U is the frame transformation matrix and \mu_{\alpha} are the eigen quantum defects.
        The transpose :math:`U^T = U^{-1}` holds because U is real and orthogonal.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            K-matrix in the outer channel frame, K = tan(\pi \mu).

        """
        transform = self.calc_frame_transformation(nu)
        kbar = self.calc_k_matrix_closecoupling(nu)
        return transform @ kbar @ transform.T

    def calc_approximate_quantum_defects(self, nu: float) -> NDArray:
        r"""Return approximate quantum defects of the outer channels, including their integer part.

        The eigen quantum defects of the close-coupling channels are tabulated including their integer
        part, so we recover the integer part of the outer channel quantum defects by transforming them
        to the outer channel frame with the frame transformation U, exactly like the K-matrix
        (see :meth:`calc_k_matrix`) but without the tangent, which is what would throw the integer part away:

        .. math::
            \mu \approx U \mu_{\alpha} U^T

        Note that transforming the eigen quantum defects instead of their tangents is only an
        approximation, which is why the result must not be used for the fractional part of the
        quantum defects, but is good enough to determine their integer part.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            Array of approximate quantum defects of the outer channels.

        """
        transform = self.calc_frame_transformation(nu)
        eigen_quantum_defects = np.diag(self.calc_eigen_quantum_defects(nu))
        return np.diag(transform @ eigen_quantum_defects @ transform.T)


class TrivialModel(EigenChannelModel):
    """Trivial single channel model with a vanishing quantum defect (i.e. a hydrogen-like channel).

    This is used as fallback for channels for which no MQDT model is available.
    """

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
        # Fast path for single channel models: the single channel has a vanishing quantum defect, so K = 0 and
        # the scaled M-matrix reduces to the 1x1 matrix sin(pi * nui) (see MQDTModel.calc_scaled_m_matrix).
        nui = self.calc_channel_nuis(nu)[0]
        return np.array([[math.sin(math.pi * nui)]])
