from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np

from rydstate.species.element_properties import get_element_properties
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy

if TYPE_CHECKING:
    from types import ModuleType

    from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ
    from rydstate.angular.core_ket import CoreKet
    from rydstate.species.mqdt import MQDT
    from rydstate.units import NDArray, PintFloat


class MQDTModel(ABC):
    r"""Base class for an MQDT model describing one symmetry block (f_tot and parity) of a species.

    The model defines a set of outer channels and, via :meth:`calc_k_matrix`,
    the K-matrix coupling them. How the K-matrix is parametrized is up to the subclasses, e.g.
    :class:`~rydstate.species.eigen_channel_model.EigenChannelModel` parametrizes it by eigen quantum
    defects together with a frame transformation to the outer channels.
    The MQDT states of the model are then given by the roots of det(M) = det(tan(\pi \nu) + K) = 0,
    see :meth:`calc_m_matrix`.
    """

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

    outer_channels: ClassVar[list[AngularKetBase[Any]]]
    """List of outer channels in the MQDT model."""

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
            E = I_i - \frac{Z^2 R_M}{\nu_i^2}
              = I_{\text{ref}} - \frac{Z^2 R_M}{\nu^2}

        where :math:`R_M = R_\infty \mu/m_e` is the mass corrected Rydberg constant and
        :math:`Z` is the net charge of the ionic core seen by the Rydberg electron.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            Array of channel nui values.

        """
        reduced_mass_au = self.element_properties.reduced_mass_au
        net_charge = self.element_properties.net_charge
        reference_threshold_au = self.mqdt.reference_ionization_threshold_au
        # we calculate binding_energy_au here directly from nu (and dont use calc_energy_au) to avoid numerical issues
        binding_energy_au = calc_energy_from_nu(reduced_mass_au, nu, net_charge)
        energies = [
            binding_energy_au - (threshold - reference_threshold_au) for threshold in self.ionization_thresholds_au
        ]
        nuis = [calc_nu_from_energy(reduced_mass_au, energy, net_charge) for energy in energies]
        return np.array(nuis)

    @abstractmethod
    def calc_k_matrix(self, nu: float) -> NDArray:
        r"""Return the K-matrix in the outer channel frame.

        The K-matrix is defined as :math:`K = \tan(\pi \mu)`, where :math:`\mu` are the quantum defects
        of the outer channels. How it is parametrized depends on the concrete model.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            K-matrix in the outer channel frame, K = tan(\pi \mu).

        """

    def calc_approximate_quantum_defects(self, nu: float) -> NDArray:
        raise NotImplementedError(
            f"{type(self).__name__} does not provide quantum defects including their integer part."
        )

    def calc_m_matrix(self, nu: float) -> NDArray:
        r"""Return the M-matrix in the outer channel frame.

        The M-matrix is defined as

        .. math::
            M = tan(β) + K = tan(\pi \nu) + tan(\pi \mu)

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            M-matrix in the outer channel frame, M = tan(β) + K.

        """
        kmat = self.calc_k_matrix(nu)
        nuis = self.calc_channel_nuis(nu)
        return np.diag(np.tan(np.pi * nuis)) + kmat

    def calc_scaled_m_matrix(self, nu: float) -> NDArray:
        r"""Return the scaled M-matrix in the outer channel frame.

        The scaled M-matrix is defined as

        .. math::
            M_{\text{scaled}} = \cos(\pi \nu) M = \sin(\pi \nu) + \cos(\pi \nu) K

        We use this to improve numerical stability when finding roots of det(M) = 0.
        This is especially important for states with nu close to half integer.
        """
        kmat = self.calc_k_matrix(nu)
        nuis = self.calc_channel_nuis(nu)
        return np.array(np.diag(np.sin(np.pi * nuis)) + np.diag(np.cos(np.pi * nuis)) @ kmat)


def get_model_classes(module: ModuleType, species: str) -> list[type[MQDTModel]]:
    """Return all MQDTModel subclasses defined in ``module`` that match the given species.

    Args:
        module: The module to inspect for MQDTModel subclasses.
        species: The species the returned MQDTModel subclasses must match.

    Returns:
        List of all MQDTModel subclasses defined in the module for the given species.

    """
    model_classes = [
        obj
        for obj in vars(module).values()
        if (
            inspect.isclass(obj)
            and obj.__module__ == module.__name__
            and issubclass(obj, MQDTModel)
            and getattr(obj, "species", None) == species
        )
    ]
    if len(model_classes) == 0:
        raise ValueError(f"No MQDTModel subclasses for species {species!r} found in {module.__name__}.")
    return model_classes


class ScaledOffDiagonalModel(MQDTModel):
    r"""MQDT model with the off-diagonal elements of the K-matrix scaled by a constant factor.

    The K-matrix in the outer channel frame describes the coupling between the outer channels,
    which enters the M-matrix only via its off-diagonal elements
    (:math:`M = \text{diag}(\tan(\pi \nu_i)) + K`, i.e. the off-diagonal part of M is the one of K).
    Scaling them down therefore continuously turns the MQDT model into a set of uncoupled channels,
    without changing the quantum defects of the individual outer channels.

    This is used by :class:`~rydstate.basis.BasisTunableMQDT`.
    """

    def __init__(self, model: MQDTModel, scale_off_diagonal: float) -> None:
        """Initialize the scaled model from an existing model.

        Args:
            model: The model to scale the off-diagonal elements of the K-matrix of.
            scale_off_diagonal: The factor by which to scale the off-diagonal elements of the K-matrix.
                A value of 1 reproduces the given model, a value of 0 fully decouples its outer channels.

        """
        self.model = model
        self.scale_off_diagonal = scale_off_diagonal

        self.species = model.species  # type: ignore [misc]
        self.name = f"{model.name} (off-diagonal K scaled by {scale_off_diagonal})"  # type: ignore [misc]
        self.reference = model.reference  # type: ignore [misc]
        self.f_tot = model.f_tot  # type: ignore [misc]
        self.nu_range = model.nu_range  # type: ignore [misc]
        self.outer_channels = model.outer_channels  # type: ignore [misc]

        super().__init__(model.mqdt)

    def calc_approximate_quantum_defects(self, nu: float) -> NDArray:
        """Return the approximate quantum defects of the wrapped model.

        The scaling only affects the off-diagonal elements of the K-matrix,
        i.e. the quantum defects of the individual outer channels are unchanged.
        """
        return self.model.calc_approximate_quantum_defects(nu)

    def calc_k_matrix(self, nu: float) -> NDArray:
        """Return the K-matrix of the model with its off-diagonal elements scaled by scale_off_diagonal."""
        kmat = self.model.calc_k_matrix(nu)
        kmat_diag = np.diag(np.diag(kmat))
        return self.scale_off_diagonal * (kmat - kmat_diag) + kmat_diag
