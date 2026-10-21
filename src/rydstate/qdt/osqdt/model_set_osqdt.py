from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rydstate.angular import NotSet
from rydstate.angular.utils import is_unknown
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.qdt.osqdt.model_osqdt import ModelOSQDT
from rydstate.qdt.sqdt.model_set_sqdt import ModelSetSQDT

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetBase
    from rydstate.qdt.mqdt.model_mqdt import ModelMQDT


logger = logging.getLogger(__name__)


class NoOSQDTStateError(ValueError):
    """Raised if the OSQDT models do not contain a state with the requested principal quantum number."""


class ModelSetOSQDT(ModelSetSQDT, metaclass=CachedABCMeta):
    r"""Outer single channel quantum defects for a specific channel derived from the MQDT models.

    In contrast to a normal :class:`~rydstate.qdt.ModelSetSQDT`, the quantum defects are not taken from
    tabulated Rydberg-Ritz parameters, but from the diagonal elements of the K-matrix of the
    :class:`~rydstate.qdt.ModelSetMQDT` models in the outer channel frame
    (and setting the couplings of the K matrix in the outer channel basis to zero).

    Since one MQDT model is only valid in a limited nu range, the channel is described by one
    :class:`ModelOSQDT` per model, which each solve the OSQDT condition in their own range of validity.
    :meth:`calc_nui` then simply picks the model which contains the requested state.
    """

    def __init__(
        self, models: tuple[ModelMQDT, ...], channel: AngularKetBase[Any], nu_range: tuple[float, float]
    ) -> None:
        """Initialize the OSQDT object from the given MQDT models and the outer channel of interest.

        Args:
            models: All MQDT models which contain the channel as one of their outer channels.
                The models must all be of the same species,
                usually they are simply the models valid in the different nu ranges of the channel.
            channel: The outer channel, whose quantum defects are described by this object.
            nu_range: Tuple of (nu_min, nu_max) for the effective principal quantum number nu.
                The OSQDT states of the channel are only calculated inside this range.

        """
        if len(models) == 0:
            raise ValueError(f"No MQDT model given for the channel {channel}.")
        if not all(model.species == models[0].species for model in models):
            raise ValueError("All models must have the same species.")
        super().__init__(models[0].species)

        self.nu_range = nu_range
        self.channel = channel
        self.mqdt_models = models
        # only models valid somewhere in the requested nu range can have states in it
        self.osqdt_models = [
            ModelOSQDT(model, channel, nu_range=nu_range)
            for model in models
            if model.nu_min <= nu_range[1] and nu_range[0] <= model.nu_max
        ]

        ns = [n for osqdt_model in self.osqdt_models for n in osqdt_model.solutions]
        self.n_min = min(ns, default=int(nu_range[1]) + 100)
        self.n_max = max(ns, default=0)

        self._sanity_check()

    def _sanity_check(self) -> None:
        tol = 1e-3

        model = self.mqdt_models[0]
        index = ModelOSQDT.get_channel_index(model, self.channel)
        wanted_nui_min = model.calc_channel_nuis(self.nu_range[0])[index]
        wanted_nui_max = model.calc_channel_nuis(self.nu_range[1])[index]
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

    def get_model(self, n: int, angular_ket: AngularKetBase[Any]) -> ModelOSQDT:  # type: ignore [override]
        """Return the OSQDT model, which has a state with the principal quantum number n.

        Args:
            n: The principal quantum number of the Rydberg state, see :meth:`calc_nui`.
            angular_ket: The angular ket of the state, which must be the channel of this OSQDT object
                (its m quantum number is ignored).

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
        return osqdt_models[0]

    def calc_nui(self, n: int, angular_ket: AngularKetBase[Any]) -> float:
        r"""Calculate the effective principal quantum number nui of the state with principal quantum number n.

        The state is given by the root of the OSQDT condition (see :meth:`ModelOSQDT.calc_condition`)
        which belongs to the principal quantum number n (see :meth:`ModelOSQDT.calc_ns`).

        Args:
            n: The principal quantum number of the Rydberg state.
                This is the true principal quantum number, i.e. it includes the integer part of the
                quantum defect (see :meth:`ModelOSQDT.calc_ns`).
            angular_ket: The angular ket of the state, which must be the channel of this OSQDT object
                (its m quantum number is ignored).

        Returns:
            The effective principal quantum number nui of the channel, defined with reference to the
            ionization threshold of the channel.

        """
        return self.get_model(n, angular_ket).solutions[n]
