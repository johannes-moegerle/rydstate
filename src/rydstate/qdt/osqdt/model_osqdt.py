from __future__ import annotations

import logging
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.linalg import find_roots
from rydstate.qdt.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.mqdt.model_mqdt import ModelMQDT


logger = logging.getLogger(__name__)


class ModelOSQDT(Model):
    r"""Outer single channel quantum defects model for a specific channel derived from a single MQDT model.

    On initialization, all roots of the OSQDT condition (see :meth:`calc_condition`) inside the range of
    validity of the MQDT model are determined and labeled by their principal quantum number n
    (see :meth:`calc_ns`).
    The resulting states are stored in the attribute ``solutions``, which maps the principal quantum
    number n of each state to its channel nui.

    The single outer channel of this model is the outer channel of the MQDT model, which overlaps with the channel,
    so that both models use the same ionization threshold for it
    (e.g. the averaged 6p threshold for the 6pnp perturber channels of the Yb models).
    """

    def __init__(self, model: ModelMQDT, channel: AngularKetBase[Any], nu_range: tuple[float, float]) -> None:
        """Initialize the OSQDT model for one outer channel of the given MQDT model.

        Args:
            model: The MQDT model, which must contain the channel as exactly one of its outer channels.
            channel: The outer channel, whose quantum defects are described by this object.
            nu_range: Tuple of (nu_min, nu_max) for the effective principal quantum number nu,
                restricting the nu range, in which the roots of the OSQDT condition are searched,
                to the range of interest (intersected with the range of validity of the model,
                which must overlap with it).

        """
        self.index = self.get_channel_index(model, channel)
        self.model = model
        self.channel = channel

        self.species = model.species
        self.name = f"{model.name}, OSQDT {channel}"
        self.reference = model.reference
        self.f_tot = channel.f_tot
        self.parity = channel.parity
        self.nu_range: tuple[float, float] = (max(model.nu_min, nu_range[0]), min(model.nu_max, nu_range[1]))
        """Range of effective principal quantum numbers nu for which the model is valid."""
        if self.nu_range[0] > self.nu_range[1]:
            raise ValueError(
                f"The requested nu range {nu_range} does not overlap with the range of validity "
                f"{model.nu_range} of {model.full_name}."
            )
        self.outer_channels = [model.outer_channels[self.index]]
        super().__init__()

        self.ionization_threshold_au = self.ionization_thresholds_au[0]
        """Ionization threshold of the single outer channel in atomic units (Hartree), with respect to which
        the effective principal quantum number nui of the channel is defined."""

        # always use the model nu_min, so determining the principal quantum number n is more robust
        # but only roots with nu in nu_range are kept, see self.solutions below
        self.nui_min = self.model.calc_channel_nuis(self.model.nu_min)[self.index]
        self.nui_max = self.model.calc_channel_nuis(min(self.model.nu_max, nu_range[1]))[self.index]

        # find all roots of the OSQDT condition in the valid range
        self._nuis = find_roots(self.calc_condition, self.nui_min, self.nui_max)

        # determine the principal quantum number n for all roots
        self._ns = self.calc_ns(self._nuis)

        self.solutions = {
            n: nui
            for n, nui in zip(self._ns, self._nuis, strict=True)
            if nu_range[0] <= self.calc_nu(nui) <= nu_range[1]
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model.full_name}, {self.channel})"

    @property
    def nu_min(self) -> float:
        """Minimum nu for which the model is valid."""
        return self.nu_range[0]

    @property
    def nu_max(self) -> float:
        """Maximum nu for which the model is valid."""
        return self.nu_range[1]

    @staticmethod
    def get_channel_index(model: ModelMQDT, channel: AngularKetBase[Any]) -> int:
        """Return the index of the single outer channel of the MQDT model, which overlaps with the channel."""
        indices = [i for i, ket in enumerate(model.outer_channels) if channel.calc_reduced_overlap(ket) != 0]
        if len(indices) != 1:
            raise ValueError(
                f"The channel {channel} overlaps with {len(indices)} outer channels of {model.full_name}, "
                "but it must overlap with exactly one."
            )
        return indices[0]

    def calc_nu(self, nui: float) -> float:
        """Convert the channel nui into the nu of the state.

        Args:
            nui: Effective principal quantum number with reference to the ionization threshold
                of the channel.

        Returns:
            Effective principal quantum number with reference to the reference ionization threshold
            of the MQDT model.

        """
        return self.model.calc_nu_from_channel_nui(nui, self.index)

    def calc_k_ii(self, nui: float) -> float:
        """Return the diagonal element K_ii of the K-matrix belonging to this channel.

        Args:
            nui: The channel nui at which to evaluate the K-matrix.

        Returns:
            The diagonal element of the K-matrix of the MQDT model in the outer channel
            frame (see :meth:`~rydstate.qdt.ModelMQDT.calc_k_matrix`).
            NaN for a state above the reference ionization threshold, where the K-matrix is not defined.

        """
        nu = self.calc_nu(nui)
        if not np.isfinite(nu):
            return np.nan
        return float(self.model.calc_k_matrix(nu)[self.index, self.index])

    def calc_condition(self, nui: float) -> float:
        r"""Calculate the condition, whose roots define the OSQDT states of this channel.

        This is the determinant condition of MQDT, where the K-matrix is assumed to be diagonal
        in the outer channel frame (i.e. the off-diagonal elements are set to zero).
        The determinant condition then decouples into one equation per outer channel

        .. math::
            \sin(\pi \nu_i) + \cos(\pi \nu_i) K_{ii}(\nu) = 0

        I.e. this is the diagonal element of the scaled M-matrix belonging to this channel
        (see :meth:`~rydstate.qdt.ModelMQDT.calc_scaled_m_matrix`), where the scaling with
        :math:`\cos(\pi \nu_i)` improves the numerical stability of the root finding.

        Args:
            nui: The channel nui at which to evaluate the condition.

        Returns:
            Value of the OSQDT condition of this channel at the given nui.

        """
        kii = self.calc_k_ii(nui)
        return float(np.sin(np.pi * nui) + np.cos(np.pi * nui) * kii)

    def calc_approximate_quantum_defect(self, nui: float) -> float:
        r"""Calculate the approximate full quantum defect of this channel, including its integer part.

        The OSQDT condition only determines the quantum defect modulo one
        (:math:`K_{ii} = \tan(\pi \mu_i)`), which is not enough to label the states with their principal
        quantum number n. The integer part is taken from the approximate quantum defects of the model
        (see :meth:`~rydstate.qdt.ModelMQDT.calc_approximate_quantum_defects`), which
        must not be used for the fractional part of the quantum defect (this one is given exactly by
        the OSQDT condition), but is good enough to determine the integer part.

        Args:
            nui: The channel nui at which to evaluate the quantum defect.

        Returns:
            Approximate quantum defect of this channel, including its integer part.

        """
        nu = self.calc_nu(nui)
        return float(self.model.calc_approximate_quantum_defects(nu)[self.index])

    def calc_ns(self, nuis: Sequence[float]) -> list[int]:
        r"""Determine the principal quantum numbers of all given roots of the OSQDT condition at once.

        The principal quantum number of a state is defined via the full quantum defect
        including its integer part.

        .. math::
            n = \nu_i + \mu_i

        At a root the fractional part of the quantum defect is exact, so

        .. math::
            n = \mathrm{round}(\nu_i + \arctan(K_{ii}) / \pi) + m

        where m is the integer part of the quantum defect.
        The rounded part is exact for every state on its own, so m is the only quantity which has to be
        estimated, and we take it from the approximated quantum defect (see :meth:`calc_approximate_quantum_defect`)
        at the state where it is the least ambiguous, i.e. where nui + mu lies closest to an integer.

        The quantum defect gains an integer wherever it passes a half integer, i.e. wherever its
        K-matrix element has a pole (a perturber of the series). This is taken into account by
        unwrapping the fractional quantum defect from state to state, since the rounded part alone
        would give both states the same n.

        Args:
            nuis: The channel nui of all OSQDT states of this channel, sorted in ascending order.

        Returns:
            The principal quantum number of each given nui.

        """
        if len(nuis) == 0:
            return []

        # wherever the fractional quantum defect jumps by more than 0.5 from one state to the next, the
        # quantum defect passed a half integer (where the K-matrix element has a pole), i.e. the channel
        # gained an integer, which unwrapping adds to all the following states
        fractional_quantum_defects = np.unwrap([np.arctan(self.calc_k_ii(nui)) / np.pi for nui in nuis], period=1)

        exact = [round(nui + mu) for nui, mu in zip(nuis, fractional_quantum_defects, strict=True)]
        approximated = [nui + self.calc_approximate_quantum_defect(nui) for nui in nuis]
        ambiguities = [abs(value - round(value)) for value in approximated]
        # prefer the largest nui if several states are equally unambiguous, since there quantum defects vary the least
        anchor = min(range(len(nuis)), key=lambda index: (ambiguities[index], -nuis[index]))
        ns = [n + round(approximated[anchor]) - exact[anchor] for n in exact]

        if ambiguities[anchor] > 0.3:
            logger.warning(
                "%s: the approximated quantum defect of channel %s determines the n of its best "
                "resolved state (nui=%.3f) only up to +-%.2f, so all its n might be shifted by one.",
                *(self.model.full_name, self.channel, nuis[anchor], ambiguities[anchor]),
            )
        duplicates = sorted(n for n, count in Counter(ns).items() if count > 1)
        if duplicates:
            logger.warning(
                "%s: channel %s has more than one state with n=%s, i.e. the OSQDT condition has roots "
                "which are not states of the channel.",
                *(self.model.full_name, self.channel, duplicates),
            )
        gaps = [n for n, next_n in pairwise(ns) if next_n - n > 1]
        if gaps:
            logger.warning(
                "%s: channel %s has no state above n=%s, i.e. the OSQDT condition has states which were not found.",
                *(self.model.full_name, self.channel, gaps),
            )
        return ns
