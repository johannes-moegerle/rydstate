from __future__ import annotations

from typing import TYPE_CHECKING

from rydstate.angular.utils import NotSet
from rydstate.basis.basis_mqdt import BasisMQDT
from rydstate.species import FModelScaledOffDiagonal

if TYPE_CHECKING:
    from rydstate.species import MQDT, Potential


class BasisTunableMQDT(BasisMQDT):
    r"""MQDT basis with a tunable coupling between the outer channels.

    This basis behaves exactly like :class:`~rydstate.basis.BasisMQDT`, except that all its models are
    wrapped in a :class:`~rydstate.species.FModelScaledOffDiagonal`, which scales the off-diagonal
    elements of the K-matrix in the outer channel frame by ``coupling_factor``.
    This tunes how much of the coupling between the outer channels is taken into account,
    while leaving the quantum defects of the individual outer channels untouched.

    For ``coupling_factor=1`` this basis reproduces :class:`~rydstate.basis.BasisMQDT` exactly,
    for ``coupling_factor=0`` the channels decouple and the states are the single channel states
    also given by :class:`~rydstate.basis.BasisOSQDT`.
    """

    def __init__(
        self,
        species: str,
        nu: tuple[float, float],
        *,
        l_r: tuple[int, int] | None = None,
        f_tot: tuple[float, float] | None = None,
        m: tuple[float, float] | NotSet | None = NotSet,
        # potential and mqdt parameters
        potential_class: type[Potential] | str | None = None,
        mqdt: MQDT | str | None = None,
        coupling_factor: float = 0.0,
    ) -> None:
        """Initialize the TunableMQDT basis.

        Args:
            species: Atomic species.
            nu: Tuple of (nu_min, nu_max) for the effective principal quantum number.
            l_r: Optional tuple of (l_r_min, l_r_max) for the Rydberg electron orbital angular momentum.
                This is used to filter models, which include at least one channel with
                l_c=0 and l_r in the specified range.
                Default None, include all models.
            f_tot: Optional tuple of (f_tot_min, f_tot_max) for the total angular momentum.
                Default None, include all f_tot values.
            m: Optional tuple of (m_min, m_max) for the magnetic quantum number range.
                Default NotSet, only include states with m=NotSet.
                If m is given as None, include all allowed m values.
            potential_class: The potential class to use for the radial ket.
                Either a a potential class
                or a string representing the tag of the potential class to use.
            mqdt: The MQDT data to use for the states.
                Either an instance of an MQDT class
                or a string representing the tag of the MQDT class to use.
            coupling_factor: The factor by which to scale the off-diagonal elements of the K-matrix,
                i.e. the coupling between the outer channels.
                Default 0, i.e. fully decoupled outer channels.

        """
        self.coupling_factor = coupling_factor
        super().__init__(species, nu, l_r=l_r, f_tot=f_tot, m=m, potential_class=potential_class, mqdt=mqdt)

    def _init_models(
        self,
        max_l_r: int,
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
    ) -> None:
        super()._init_models(max_l_r, f_tot_range, l_r_range)
        # models with a single outer channel have no off-diagonal elements at all, so we leave them untouched
        # (this also keeps the fast path of FModelSQDT)
        self.models = [
            FModelScaledOffDiagonal(model, self.coupling_factor) if len(model.outer_channels) > 1 else model
            for model in self.models
        ]
