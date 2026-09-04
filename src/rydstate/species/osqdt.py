from __future__ import annotations

import logging
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.utils import is_unknown
from rydstate.linalg import find_roots
from rydstate.species.sqdt import SQDT
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.species.fmodel import FModel
    from rydstate.species.mqdt import MQDT


logger = logging.getLogger(__name__)


class NoOSQDTStateError(ValueError):
    """Raised if the OSQDT models do not contain a state with the requested principal quantum number."""


class OSQDT(SQDT):
    r"""Outer single channel quantum defects for a specific channel derived from the MQDT models.

    In contrast to a normal :class:`~rydstate.species.SQDT`, the quantum defects are not taken from
    tabulated Rydberg-Ritz parameters, but from the diagonal elements of the K-matrix of the
    :class:`~rydstate.species.MQDT` models in the outer channel frame
    (and setting the couplings of the K matrix in the outer channel basis to zero).

    Since one MQDT model is only valid in a limited nu range, the channel is described by one
    :class:`OSQDTModel` per model, which each solve the OSQDT condition in their own range of validity.
    :meth:`calc_nui` then simply picks the model which contains the requested state.
    """

    def __init__(self, models: tuple[FModel, ...], channel: AngularKetBase[Any], nu_range: tuple[float, float]) -> None:
        """Initialize the OSQDT object from the given MQDT models and the outer channel of interest.

        Args:
            models: All MQDT models which contain the channel as one of their outer channels.
                The models must all belong to the same :class:`~rydstate.species.MQDT` instance,
                usually they are simply the models valid in the different nu ranges of the channel.
            channel: The outer channel, whose quantum defects are described by this object.
            nu_range: Tuple of (nu_min, nu_max) for the effective principal quantum number nu.
                The OSQDT states of the channel are only calculated inside this range.

        """
        if len(models) == 0:
            raise ValueError(f"No MQDT model given for the channel {channel}.")
        if not all(model.species == models[0].species for model in models):
            raise ValueError("All models must have the same species.")
        if not all(model.mqdt is models[0].mqdt for model in models):
            raise ValueError("All models must belong to the same MQDT instance.")
        self.species = models[0].species  # type: ignore [misc]
        self.mqdt: MQDT = models[0].mqdt
        super().__init__()

        self.nu_range = nu_range
        self.channel = channel
        self.osqdt_models = [OSQDTModel(model, channel, nu_range=nu_range) for model in models]

        # all models describe the same channel, so they must agree on its ionization threshold
        thresholds = {osqdt_model.ionization_energy_au for osqdt_model in self.osqdt_models}
        if len(thresholds) > 1:
            raise ValueError(f"The models do not agree on the ionization threshold of the channel {channel}.")
        self.ionization_energy = (thresholds.pop(), "hartree")  # type: ignore [misc]
        # nu is defined with respect to the reference ionization threshold of the MQDT models
        self._reference_ionization_energy = (self.mqdt.reference_ionization_threshold_au, "hartree")  # type: ignore [misc]

        self.n_min = min(
            min(osqdt_model.solutions.keys(), default=int(nu_range[1]) + 100) for osqdt_model in self.osqdt_models
        )
        self.n_max = max(max(osqdt_model.solutions.keys(), default=0) for osqdt_model in self.osqdt_models)

        self._sanity_check()

    def _sanity_check(self) -> None:
        tol = 1e-3

        osqdt_models = self.osqdt_models[0]
        wanted_nui_min = osqdt_models.model.calc_channel_nuis(self.nu_range[0])[osqdt_models.index]
        wanted_nui_max = osqdt_models.model.calc_channel_nuis(self.nu_range[1])[osqdt_models.index]
        l_r = self.channel.l_r if not is_unknown(self.channel.l_r) else 0

        if wanted_nui_max < max(wanted_nui_min + 1, l_r + 2) + tol:
            return  # the nui range is too small to expect any OSQDT states, so no warning is needed

        if all(len(osqdt_model.solutions) == 0 for osqdt_model in self.osqdt_models):
            if (self.nu_range[1] - self.nu_range[0]) > 1 and wanted_nui_max > 2:
                logger.warning(
                    "The channel %s has no OSQDT states in the nu range %s (nui_range=(%s, %s)).",
                    *(self.channel, self.nu_range, wanted_nui_min, wanted_nui_max),
                )
            return

        model_min_max = [(osqdt_model, osqdt_model.nui_min, osqdt_model.nui_max) for osqdt_model in self.osqdt_models]
        model_min_max = sorted(model_min_max, key=lambda item: item[1])  # sort by nui_min
        possible_nui_min = model_min_max[0][1]

        if possible_nui_min > max(wanted_nui_min, 2, l_r + 1) + tol:
            logger.warning(
                "The channel %s has no model for nui<%.3f, but the requested nu range starts at nui=%.3f.",
                *(self.channel, possible_nui_min, wanted_nui_min),
            )

        old_nui_max = model_min_max[0][2]
        for _osqdt_model, nui_min, nui_max in model_min_max[1:]:
            if wanted_nui_max <= old_nui_max + tol:
                break

            if nui_min > max(wanted_nui_min, old_nui_max) + tol:
                logger.warning(
                    "The channel %s has no model for nui in (%.3f, %.3f).", *(self.channel, old_nui_max, nui_min)
                )
            old_nui_max = max(nui_max, old_nui_max)

        possible_nui_max = model_min_max[-1][2]
        if wanted_nui_max > possible_nui_max + tol:
            logger.warning(
                "The channel %s has no model for nui>%.3f, but the requested nu range ends at nui=%.3f.",
                *(self.channel, possible_nui_max, wanted_nui_max),
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.species}, {self.channel})"

    def calc_nui(self, n: int, angular_ket: AngularKetBase[Any]) -> float:  # type: ignore [override]
        r"""Calculate the effective principal quantum number nui of the state with principal quantum number n.

        The state is given by the root of the OSQDT condition (see :meth:`OSQDTModel.calc_condition`)
        which belongs to the principal quantum number n (see :meth:`OSQDTModel.calc_ns`).

        Args:
            n: The principal quantum number of the Rydberg state.
                This is the true principal quantum number, i.e. it includes the integer part of the
                quantum defect (see :meth:`OSQDTModel.calc_ns`).
            angular_ket: The angular ket of the state, which must be the channel of this OSQDT object
                (its m quantum number is ignored).

        Returns:
            The effective principal quantum number nui of the channel, defined with reference to the
            ionization threshold of the channel.

        """
        if angular_ket.replace_m(NotSet) != self.channel:
            raise ValueError(f"The angular ket {angular_ket} is not the OSQDT channel {self.channel}.")

        osqdt_models = [osqdt_model for osqdt_model in self.osqdt_models if n in osqdt_model.solutions]

        if len(osqdt_models) == 0:
            raise NoOSQDTStateError(
                f"None of the models {[m.model.full_name for m in self.osqdt_models]} "
                f"has a state with n={n} for the channel {self.channel}."
            )
        if len(osqdt_models) > 1:
            logger.warning(
                "Found %d OSQDT models with n=%d for the channel %s, using the one of %s.",
                *(len(osqdt_models), n, self.channel, osqdt_models[0].model.full_name),
            )
        return osqdt_models[0].solutions[n]


class OSQDTModel:
    r"""Outer single channel quantum defects model for a specific channel derived from a single MQDT model.

    On initialization, all roots of the OSQDT condition (see :meth:`calc_condition`) inside the range of
    validity of the MQDT model are determined and labeled by their principal quantum number n
    (see :meth:`calc_ns`).
    The resulting states are stored in the attribute ``solutions``, which maps the principal quantum
    number n of each state to its channel nui.
    """

    def __init__(self, model: FModel, channel: AngularKetBase[Any], nu_range: tuple[float, float]) -> None:
        """Initialize the OSQDT model for one outer channel of the given MQDT model.

        Args:
            model: The MQDT model, which must contain the channel as exactly one of its outer channels.
            channel: The outer channel, whose quantum defects are described by this object.
            nu_range: Tuple of (nu_min, nu_max) for the effective principal quantum number nu,
                restricting the nu range, in which the roots of the OSQDT condition are searched,
                to the range of interest (intersected with the range of validity of the model).

        """
        indices = [i for i, ket in enumerate(model.outer_channels) if channel.calc_reduced_overlap(ket) != 0]
        if len(indices) != 1:
            raise ValueError(
                f"The channel {channel} overlaps with {len(indices)} outer channels of {model.full_name}, "
                "but it must overlap with exactly one."
            )
        self.index = indices[0]
        self.model = model
        self.channel = channel

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
    def ionization_energy_au(self) -> float:
        """The ionization threshold of the channel in atomic units (Hartree)."""
        return self.model.ionization_thresholds_au[self.index]

    def calc_nu(self, nui: float) -> float:
        """Convert the channel nui into the nu of the state, i.e. change the reference ionization threshold.

        Args:
            nui: Effective principal quantum number with reference to the ionization threshold
                of the channel.

        Returns:
            Effective principal quantum number with reference to the reference ionization threshold
            of the MQDT model, which is infinity for a state above that threshold.

        """
        element_properties = self.model.element_properties
        mqdt = self.model.mqdt

        reduced_mass_au = element_properties.reduced_mass_au
        net_charge = element_properties.net_charge
        reference_threshold_au = mqdt.reference_ionization_threshold_au
        binding_energy_au = calc_energy_from_nu(reduced_mass_au, nui, net_charge) + (
            self.ionization_energy_au - reference_threshold_au
        )
        return calc_nu_from_energy(reduced_mass_au, binding_energy_au, net_charge)

    def calc_k_ii(self, nui: float) -> float:
        """Return the diagonal element K_ii of the K-matrix belonging to this channel.

        Args:
            nui: The channel nui at which to evaluate the K-matrix.

        Returns:
            The diagonal element of the K-matrix of the MQDT model in the outer channel
            frame (see :meth:`~rydstate.species.fmodel.FModel.calc_k_matrix`).
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
        (see :meth:`~rydstate.species.fmodel.FModel.calc_scaled_m_matrix`), where the scaling with
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
        quantum number n. The integer part is recovered from the eigen quantum defects of the
        close-coupling channels, which are tabulated including their integer part, by transforming
        them to the outer channel frame with the frame transformation U, exactly like the K-matrix
        (see :meth:`~rydstate.species.fmodel.FModel.calc_k_matrix`) but without the tangent, which
        is what would throw the integer part away:

        .. math::
            \mu \approx U \mu_{\alpha} U^T

        Note that transforming the eigen quantum defects instead of their tangents is only an
        approximation, which is why the result must not be used for the fractional part of the
        quantum defect (this one is given exactly by the OSQDT condition), but is good enough to
        determine the integer part.

        Args:
            nui: The channel nui at which to evaluate the quantum defect.

        Returns:
            Approximate quantum defect of this channel, including its integer part.

        """
        nu = self.calc_nu(nui)
        transform = self.model.calc_frame_transformation(nu)
        eigen_quantum_defects = np.diag(self.model.calc_eigen_quantum_defects(nu))
        quantum_defects = transform @ eigen_quantum_defects @ transform.T
        return float(quantum_defects[self.index, self.index])

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
