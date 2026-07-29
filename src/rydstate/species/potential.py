from __future__ import annotations

import logging
import math
from abc import ABC
from typing import TYPE_CHECKING, ClassVar, TypeVar

import numpy as np

from rydstate.angular.utils import is_unknown
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.species.element_properties import get_element_properties
from rydstate.species.utils import get_all_subclasses

if TYPE_CHECKING:
    from rydstate.units import NDArray

XType = TypeVar("XType", "NDArray", float)


logger = logging.getLogger(__name__)


class Potential(ABC, metaclass=CachedABCMeta):
    """Base class for all potential classes."""

    species: ClassVar[str]
    """The short name of the atomic species."""
    tag: ClassVar[str]
    """The tag for these potential parameters."""
    is_default: ClassVar[bool] = False
    """Whether this potential is the default potential for the species."""

    def __init__(self, l_r: int) -> None:
        r"""Initialize the potential.

        Args:
            l_r: Orbital angular momentum of the Rydberg electron.

        """
        self.element_properties = get_element_properties(self.species)

        if is_unknown(l_r) or not ((isinstance(l_r, int) or l_r.is_integer()) and l_r >= 0):
            raise ValueError(f"l_r must be an integer, and larger or equal 0, but {l_r=}")
        self.l_r = int(l_r)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(l_r={self.l_r})"

    def calc_potential_coulomb(self, x: XType) -> XType:
        r"""Calculate the Coulomb potential V_Col(x) in atomic units.

        The Coulomb potential is given as

        .. math::
            V_{Col}(x) = -Z / x

        where x = r / a_0 and Z is the net charge of the ionic core seen by the Rydberg electron.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_Col: The Coulomb potential V_Col(x) in atomic units.

        """
        return -self.element_properties.net_charge / x

    def calc_effective_potential_centrifugal(self, x: XType) -> XType:
        r"""Calculate the effective centrifugal potential V_l(x) in atomic units.

        The effective centrifugal potential is given as

        .. math::
            V_{l_r}(x) = \frac{l_r(l_r+1)}{2x^2}

        where x = r / a_0 and l_r is the orbital angular momentum quantum number of the Rydberg electron.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_{l_r}: The effective centrifugal potential V_{l_r}(x) in atomic units.

        """
        x2 = x * x
        return (1 / self.element_properties.reduced_mass_au) * self.l_r * (self.l_r + 1) / (2 * x2)

    def calc_effective_potential_sqrt(self, x: XType) -> XType:
        r"""Calculate the effective potential V_sqrt(x) from the sqrt transformation in atomic units.

        The sqrt transformation potential arises from the transformation from the wavefunction u(x) to w(z),
        where x = r / a_0 and w(z) = z^{-1/2} u(x=z^2) = (r/a_0)^{-1/4} sqrt(a_0) r R(r).
        Due to the transformation, an additional term is added to the radial Schrödinger equation,
        which can be written as effective potential V_{sqrt}(x) and is given by

        .. math::
            V_{sqrt}(x) = \frac{3}{32x^2}

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V_sqrt: The sqrt transformation potential V_sqrt(x) in atomic units.

        """
        x2 = x * x
        return (1 / self.element_properties.reduced_mass_au) * (3 / 32) / x2

    def calc_model_potential(self, x: XType) -> XType:
        raise NotImplementedError(
            f"Subclasses of Potential ({self.__class__.__name__}) must implement the calc_model_potential method."
        )

    def calc_total_effective_potential(self, x: XType) -> XType:
        r"""Calculate the total effective potential V_eff(x) in atomic units.

        The total effective potential includes all physical and effective potentials:

        .. math::
            V_{eff}(x) = V(x) + V_{l_r}(x) + V_{sqrt}(x)

        where V(x) is the physical potential (either Coulomb or a model potential),
        V_{l_r}(x) is the effective centrifugal potential,
        and V_{sqrt}(x) is the effective potential from the sqrt transformation.

        Note that we on purpose do not include the spin-orbit potential for several reasons:

        i) The fine structure corrections are important for the energies of the states.
           This includes a) spin-orbit coupling, b) Darwin term, and c) relativistic corrections to the kinetic energy.
           Since we (obviously) can not include the latter two in the potential,
           it is only consistent to not include the spin-orbit term either.

        ii) The model potentials are generated without the spin-orbit term,
            since their accuracy is not sufficient to resolve the fine structure corrections at small distances.
            (This can also be seen by running Numerov for low lying states with an energy changed by e.g. 1%,
            which will lead to almost no change in the wavefunction.)

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_eff: The total potential V_eff(x) in atomic units.

        """
        # Note: we do not include the spin-orbit potential, see docstring for details.
        v = self.calc_model_potential(x)
        v += self.calc_effective_potential_centrifugal(x)
        v += self.calc_effective_potential_sqrt(x)
        return v

    def calc_hydrogen_turning_point_z(self, n: int) -> float:
        r"""Calculate the classical turning point z_i of the state if it would be a hydrogen atom.

        The hydrogen turning point is defined as the point,
        where for the idealized hydrogen(-like) atom the potential equals the energy,
        i.e. V_Col(r_i) + V_l(r_i) = E.
        This is exactly the case at

        .. math::
            r_i = \frac{n^2 - n \sqrt{n^2 - l_r(l_r + 1)}}{Z}

        and z_i = sqrt{r_i / a_0}, where Z is the net charge of the ionic core seen by the Rydberg electron.

        Args:
            n: Principal quantum number of the state.

        Returns:
            z_i: The inner hydrogen turning point z_i in the scaled dimensionless coordinate z_i = sqrt{r_i / a_0}.

        """
        net_charge = self.element_properties.net_charge
        return math.sqrt((n * n - n * math.sqrt(n * n - self.l_r * (self.l_r + 1))) / net_charge)

    def calc_turning_point_z(self, energy_au: float, dz: float = 1e-3) -> float:
        r"""Calculate the classical inner turning point z_i for the given state.

        The classical turning point is defined as the point,
        where the total effective potential of the Rydberg model equals the energy,
        i.e. V_eff(r_i) = E.

        Note: Because we use the total effective potential, even for l=0 the turning point is not at r=0.
        The advantage of this is, that this definition of the turning point should correspond to
        where w(z) should have its last change of sign in the second derivative.

        Args:
            energy_au: The energy, for which to calculate the classical turning point in atomic units.
            dz: The precision of the turning point calculation.

        Returns:
            z_i: The inner turning point z_i in the scaled dimensionless coordinate z_i = sqrt{r_i / a_0}.

        """
        # for a given hydrogen turning point z_hyd, the classical turning point usually lies within z_hyd \pm 5
        # for a given l, the hydrogen turning point is bound by
        # z_lower = z_hyd(n=inf, l_r)  = \sqrt{l_r * (l_r+1) / (2 Z)} <= z_hyd(n, l_r) <= z_hyd(n=l_r+1, l_r) = z_upper
        z_lower = math.sqrt(self.l_r * (self.l_r + 1) / (2 * self.element_properties.net_charge))
        z_upper = self.calc_hydrogen_turning_point_z(n=self.l_r + 1)

        z_min_orig, z_max_orig = max(z_lower - 5, dz), z_upper + 5
        z_min, z_max = z_min_orig, z_max_orig

        while z_max - z_min > dz:
            z_list = np.linspace(z_min, z_max, 1_000, endpoint=True)
            v_list = self.calc_total_effective_potential(z_list**2) - energy_au

            inds = np.argwhere(np.diff(np.sign(v_list)) < 0).flatten()
            if len(inds) == 0:
                raise ValueError("Effective potential is always above or below the energy, this should not happen!")
            ind = inds[-1]  # take the last index, where a sign change from positive to negative occurs
            # because for some potentials, the potential for small distances gets negative again,
            # but the classical forbidden region was already reached for a larger distance

            z_min = z_list[ind]
            z_max = z_list[ind + 1]

        if z_min == z_min_orig or z_max == z_max_orig:
            logger.warning(
                "The turning point calculation did converge to the original z_min or z_max. "
                "This should not happen and is probably a bug!"
            )

        return z_min + (z_max - z_min) * v_list[ind] / (v_list[ind] - v_list[ind + 1])  # type: ignore [no-any-return]


class PotentialCoulomb(Potential):
    """Simple Coulomb potential, without any additional terms."""

    tag = "coulomb"

    def calc_model_potential(self, x: XType) -> XType:
        r"""Calculate the model potential V(x) in atomic units.

        Default implementation returns the Coulomb potential, but this can be overridden by subclasses to implement
        different model potentials.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate the potential.

        Returns:
            V: The model potential V(x) in atomic units.

        """
        return self.calc_potential_coulomb(x)


class PotentialMarinescu1994(Potential):
    """Model potential for alkali atoms from Marinescu et al. (1994).

    See also: Phys. Rev. A 49, 982 (1994)
    """

    tag = "marinescu_1994"

    alpha_core_marinescu_1994: ClassVar[float]
    """Static dipole polarizability (a.u.) of the core, used for the parametric model potential."""
    r_c_dict_marinescu_1994: ClassVar[dict[int, float]]
    """Cutoff radius {l: r_c} to truncate the unphysical short-range contribution of the polarization potential."""
    model_potential_parameter_marinescu_1994: ClassVar[dict[int, tuple[float, float, float, float]]]
    """Parameters {l: (a_1, a_2, a_3, a_4)} for the parametric model potential."""
    reference: ClassVar[str] = (
        "M. Marinescu et al., Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982"
    )

    def calc_model_potential(self, x: XType) -> XType:
        r"""Calculate the model potential by Marinescu et al. (1994) in atomic units.

        The model potential, see :attr:`~PotentialMarinescu1994.reference`, is given by

        .. math::
            V_{mp,marinescu}(x) = - \frac{Z_{l}}{x} - \frac{\alpha_{core}}{2x^4} (1 - e^{-x^6/x_c**6})

        where Z_{l} is the effective nuclear charge,
        :math:`\alpha_{core}` is the static dipole polarizability of the ionic core,
        and x_c is the effective core size.

        .. math::
            Z_{l} = Z_{net} + (Z - Z_{net}) \exp(-a_1 x) - x (a_3 + a_4 x) \exp(-a_2 x)

        with the nuclear charge Z and the net charge :math:`Z_{net}` of the ionic core seen by the Rydberg electron.
        The effective charge interpolates from Z (at x = 0) to :math:`Z_{net}` (at x -> infinity).

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_{mp,marinescu}: The four parameter potential V_{mp,marinescu}(x) in atomic units.

        """
        parameter_dict = self.model_potential_parameter_marinescu_1994
        if len(parameter_dict) == 0:
            raise ValueError(f"No parametric model potential parameters defined for the species {self.species}.")
        # default to parameters for the maximum l
        a1, a2, a3, a4 = parameter_dict.get(self.l_r, parameter_dict[max(parameter_dict.keys())])
        z_net = self.element_properties.net_charge
        exp_a1 = np.exp(-a1 * x)
        exp_a2 = np.exp(-a2 * x)
        z_nl: XType = z_net + (self.element_properties.Z - z_net) * exp_a1 - x * (a3 + a4 * x) * exp_a2
        v_c = -z_nl / x

        alpha_core = self.alpha_core_marinescu_1994
        if alpha_core == 0:
            v_p = 0
        else:
            r_c_dict = self.r_c_dict_marinescu_1994
            if len(r_c_dict) == 0:
                raise ValueError(f"No parametric model potential parameters defined for the species {self.species}.")
            # default to x_c for the maximum l
            x_c = r_c_dict.get(self.l_r, r_c_dict[max(r_c_dict.keys())])
            x2: XType = x * x
            x4: XType = x2 * x2
            x6: XType = x4 * x2
            exp_x6 = np.exp(-(x6 / x_c**6))
            v_p = -alpha_core / (2 * x4) * (1 - exp_x6)

        return v_c + v_p


class PotentialFei2009(Potential):
    """Model potential for alkaline earth atoms from Fei et al. (2009).

    See also: Phys. Rev. A 79, 052507 (2009)
    """

    tag = "fei_2009"

    model_potential_parameter_fei_2009: ClassVar[tuple[float, float, float, float]]
    """Parameters (delta, alpha, beta, gamma) for the four-parameter potential used in the model potential."""
    reference: ClassVar[str] = (
        "Y. Fei et al., Chinese Phys. B 18 4234 (2009), https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025"
    )

    def calc_model_potential(self, x: XType) -> XType:
        r"""Calculate the model potential by Fei et al. (2009) in atomic units.

        The four parameter potential, see :attr:`~PotentialFei2009.reference`, is given by

        .. math::
            V_{mp,fei}(x) = - \frac{1}{x}
                - \frac{Z-1}{x} \cdot [1 - \alpha + \alpha e^{\beta x^\delta + \gamma x^{2\delta}}]^{-1}

        where Z is the nuclear charge.

        Args:
            x: The dimensionless radial coordinate x = r / a_0, for which to calculate potential.

        Returns:
            V_{mp,fei}: The four parameter potential V_{mp,fei}(x) in atomic units.

        """
        if self.element_properties.net_charge != 1:
            raise ValueError("PotentialFei2009 is only valid for neutral atoms with a core net charge of 1")
        delta, alpha, beta, gamma = self.model_potential_parameter_fei_2009
        with np.errstate(over="ignore"):
            denom: XType = 1 - alpha + alpha * np.exp(beta * x**delta + gamma * x ** (2.0 * delta))
            return -1 / x - (self.element_properties.Z - 1) / (x * denom)


def get_potential_class(species: str, tag: str | None = None) -> type[Potential]:
    """Get the subclass of Potential for the given species and tag."""
    subclasses = get_all_subclasses(Potential, species, tag)

    if tag is None:
        subclasses = [cls for cls in subclasses if getattr(cls, "is_default", False)]

    if len(subclasses) == 0:
        raise ValueError(f"No subclass of Potential found for {species=} and {tag=}.")
    if len(subclasses) == 1:
        return subclasses[0]
    raise ValueError(f"Multiple subclasses of Potential found for {species=} and {tag=}: {subclasses}.")
