from __future__ import annotations

import logging
import math
import weakref
from numbers import Number
from typing import TYPE_CHECKING, Literal

import numpy as np
from mpmath import whitw
from scipy.special import gamma

from rydstate.angular.utils import is_unknown
from rydstate.metaclass_cache import CachedABCMeta
from rydstate.radial.numerov import _run_numerov_integration_python, run_numerov_integration
from rydstate.radial.radial_base import Radial
from rydstate.species.utils import calc_energy_from_nu

if TYPE_CHECKING:
    from rydstate.radial.radial_matrix_element import INTEGRATION_METHODS, RadialMatrixElementOperator
    from rydstate.species.element_properties import ElementProperties
    from rydstate.species.potential import Potential
    from rydstate.units import NDArray

logger = logging.getLogger(__name__)


WavefunctionSignConvention = Literal["positive_at_outer_bound", "n_l_1"] | None


class RadialKet(Radial, metaclass=CachedABCMeta):
    r"""Class representing a radial Rydberg state."""

    element_properties: ElementProperties
    nu: float

    def __init__(
        self,
        nu: float,
        potential: Potential,
        *,
        n_expected: int | None = None,
        dz: float = 1e-2,
        x_min: float | None = None,
        x_max: float | None = None,
        run_backward: bool = True,
        w0: float = 1e-10,
        use_njit: bool = True,
        integration_method: Literal["numerov", "whittaker"] = "numerov",
        sign_convention: WavefunctionSignConvention = "positive_at_outer_bound",
    ) -> None:
        r"""Initialize the radial ket.

        Args:
            nu: Effective principal quantum number of the rydberg electron,
                which is used to calculate the energy of the state.
            potential: The potential object.
            n_expected: Optional principal quantum number of the rydberg electron, used for additional
                sanity checks of the radial wavefunction (e.g. checking that the number of nodes matches
                n - l - 1). It is also required for the "n_l_1" sign convention.
            dz: The step size of the integration in the scaled dimensionless
                coordinate :math:`z = \sqrt{r/a_0}`.
            x_min: The minimum value of the radial coordinate in dimensionless
                units :math:`x = r/a_0`. None means a sensible value is calculated automatically.
            x_max: The maximum value of the radial coordinate in dimensionless
                units :math:`x = r/a_0`. None means a sensible value is calculated automatically.
            run_backward: Whether to integrate the radial Schrödinger equation
                "backward" or "forward".
            w0: The initial magnitude of the radial wavefunction at the boundary.
            use_njit: Whether to use the fast njit version of the Numerov integration.
            integration_method: The method used to integrate the wavefunction,
                either "numerov" or "whittaker".
            sign_convention: The sign convention applied to the wavefunction after the integration.
                One of "positive_at_outer_bound", "n_l_1" or None (see :meth:`_apply_sign_convention`).
                The "n_l_1" convention requires ``n_expected`` to be set.

        """
        self._matrix_element_cache: weakref.WeakKeyDictionary[
            RadialKet, dict[tuple[RadialMatrixElementOperator, str], float]
        ] = weakref.WeakKeyDictionary()

        self.potential = potential
        self.element_properties = potential.element_properties
        self.l_r = potential.l_r

        if not nu > 0:
            raise ValueError(f"nu must be larger than 0, but is {nu=}")
        self.nu = nu

        self._dz = dz
        self._x_min = x_min
        self._x_max = x_max
        self._run_backward = run_backward
        self._w0 = w0
        self._use_njit = use_njit
        self._integration_method = integration_method
        self._sign_convention: WavefunctionSignConvention = sign_convention

        self.n_expected: int | None = n_expected
        if self.n_expected is not None:
            self._sanity_check_n_expected(self.n_expected)

    def __repr__(self) -> str:
        nu, potential = self.nu, self.potential
        return f"{self.__class__.__name__}({nu=}, {potential=})"

    @property
    def z_list(self) -> NDArray:
        if not hasattr(self, "_z_list"):
            self._create_grid_points()
        return self._z_list

    @property
    def w_list(self) -> NDArray:
        if not hasattr(self, "_w_list"):
            self.integrate_wavefunction()
        return self._w_list

    def _sanity_check_n_expected(self, n_expected: int) -> None:
        """Validate n_expected, which is used for additional sanity checks of the radial wavefunction.

        E.g. if n_expected is provided, we can check whether the number of nodes in the wavefunction matches
        n_expected - l - 1.

        Args:
            n_expected: Principal quantum number of the rydberg electron.

        """
        if not ((isinstance(n_expected, int) or n_expected.is_integer()) and n_expected >= 1):
            raise ValueError(f"n_expected must be an integer, and larger or equal 1, but {n_expected=}")

        if n_expected > 10 and n_expected < (self.nu - 1e-5):
            # if n_expected <= 10, we use NIST energy data for low n, which sometimes results in nu > n_expected
            # -1e-5: avoid issues due to numerical precision and due to NIST data
            raise ValueError(f"n_expected must be larger or equal to nu, but {n_expected=}, nu={self.nu} for {self}")
        if not is_unknown(self.l_r) and n_expected <= self.l_r:
            raise ValueError(f"n_expected must be larger than l_r, but {n_expected=}, l_r={self.l_r} for {self}")

    def _create_grid_points(self) -> None:
        r"""Create the grid points for the integration of the radial Schrödinger equation.

        The grid is controlled by the ``x_min``, ``x_max`` and ``dz`` arguments
        passed to :meth:`__init__` (see there for details).
        """
        x_min = self._x_min
        x_max = self._x_max
        dz = self._dz
        if hasattr(self, "_z_list"):
            raise RuntimeError("The grid points were already created, you should not create them again.")

        net_charge = self.element_properties.net_charge
        if x_min is None:
            # we set z_min explicitly too small,
            # since the integration will automatically stop after the turning point,
            # and as soon as the wavefunction is close to zero
            if self.l_r <= 10:
                z_min = 0.0
            else:
                # we use reduced_mass_au=1, which overestimates the energy
                # and thus underestimates the lower turning point
                energy_au = calc_energy_from_nu(1, self.nu, net_charge)
                z_min = self.potential.calc_turning_point_z(energy_au)
                z_min = math.sqrt(0.5) * z_min - 3  # see also compare_z_min_cutoff.ipynb
        else:
            z_min = math.sqrt(x_min)
        # Since the potential diverges at z=0 we set the minimum z_min to dz
        z_min = max(z_min, dz)

        if x_max is None:
            # This is an empirical formula for the maximum value of the radial coordinate
            # it takes into account that for large n but small l the wavefunction is very extended.
            # The outer classical turning point scales as ~2 nu^2 / Z, hence the division by the net charge.
            x_max = 2 * self.nu * (self.nu + 20 + (self.nu - self.l_r) / 4) / net_charge + 5
        z_max = math.sqrt(x_max)

        # put all grid points on a standard grid, i.e. [dz, 2*dz, 3*dz, ...]
        # this is necessary to allow integration of two different wavefunctions
        # Note: using np.arange((z_min // dz) * dz, z_max + dz / 2, dz)
        # would lead to 'quite big' inprecisions (1e-10) between grid points of different grids,
        # because of floating point errors
        self._z_list: NDArray = np.arange(0, z_max + dz / 2, dz)[round(z_min / dz) :]

    def integrate_wavefunction(self) -> None:
        """Integrate the wavefunction using the method given by the ``integration_method`` parameter."""
        method = self._integration_method
        if method == "numerov":
            self._integrate_numerov()
        elif method == "whittaker":
            self._integrate_whittaker()
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _integrate_numerov(self) -> None:
        r"""Run the Numerov integration of the radial Schrödinger equation.

        The resulting radial wavefunctions are then stored as attributes, where
        - w_list is the dimensionless and scaled wavefunction w(z)
        - u_list is the dimensionless wavefunction \tilde{u}(x)
        - r_list is the radial wavefunction R(r) in atomic units

        The radial wavefunction are related as follows:

        .. math::
            \tilde{u}(x) = \sqrt{a_0} r R(r)

        .. math::
            w(z) = z^{-1/2} \tilde{u}(x=z^2) = (r/a_0)^{-1/4} \sqrt{a_0} r R(r)


        where z = \sqrt{r/a_0} is the dimensionless scaled coordinate.

        The resulting radial wavefunction is normalized such that

        .. math::
            \int_{0}^{\infty} r^2 |R(r)|^2 dr
            = \int_{0}^{\infty} |\tilde{u}(x)|^2 dx
            = \int_{0}^{\infty} 2 z^2 |w(z)|^2 dz
            = 1

        The integration is controlled by the ``run_backward``, ``w0`` and ``use_njit`` arguments
        passed to :meth:`__init__`:

        - ``run_backward``: Whether to integrate the radial Schrödinger equation "backward" or "forward".
        - ``w0``: The initial magnitude of the wavefunction at the inner/outer boundary for forward/backward integration
        - ``use_njit``: Whether to use the fast njit version of the Numerov integration.
        """
        run_backward = self._run_backward
        w0 = self._w0
        use_njit = self._use_njit

        if hasattr(self, "_w_list"):
            raise RuntimeError("The wavefunction was already integrated, you should not create it again.")

        # Note: Inside this method we use y and x like it is used in the numerov function
        # and not like in the rest of this class, i.e. y = w(z) and x = z
        element_properties = self.element_properties
        energy_au = calc_energy_from_nu(element_properties.reduced_mass_au, self.nu, element_properties.net_charge)
        v_eff = self.potential.calc_total_effective_potential(self.x_list)
        glist = 8 * element_properties.reduced_mass_au * self.z_list * self.z_list * (energy_au - v_eff)

        if run_backward:
            # During the Numerov integration we start the wavefunction at the outer boundary with +w0.
            # Note: n - l - 1 is the number of nodes of the radial wavefunction
            # Thus, the sign of the wavefunction at the inner boundary is (-1)^{(n - l - 1) % 2}
            # You can choose a different sign convention via the sign_convention parameter
            y0, y1 = 0, w0
            x_start, x_stop, dx = self.z_list[-1], self.z_list[0], -self.dz
            g_list_directed = glist[::-1]
            # We set x_min to the classical turning point
            # after x_min is reached in the integration, the integration stops, as soon as it crosses the x-axis again
            # or it reaches a local minimum (thus going away from the x-axis)
            # the reason for this is that the second derivative of the wavefunction d^2/dz^2 w(z) (= concavity)
            # can only vanish at either
            # i) where w(z) = 0 or ii) where the potential is equal to the energy (-> classical turning point)
            # If we further assume, that the wavefunction converges to zero at the inner boundary,
            # we know that after the inner classical turning point
            # the wavefunction should never increase the distance from the x-axis again.
            x_min = self.potential.calc_turning_point_z(energy_au)

        else:  # forward
            y0, y1 = 0, w0
            x_start, x_stop, dx = self.z_list[0], self.z_list[-1], self.dz
            g_list_directed = glist
            x_min = math.sqrt(self.nu * (self.nu + 15) / element_properties.net_charge)

        if use_njit:
            w_list_list = run_numerov_integration(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)
        else:
            logger.warning("Using python implementation of Numerov integration, this is much slower!")
            w_list_list = _run_numerov_integration_python(x_start, x_stop, dx, y0, y1, g_list_directed, x_min)

        # add zeros for the part of the grid that was not integrated
        w_list_list += [0] * (len(self.z_list) - len(w_list_list))
        if run_backward:
            w_list_list = w_list_list[::-1]
        self._w_list = np.array(w_list_list)

        # normalize the wavefunction, see docstring
        self._w_list /= self.norm

        self._apply_sign_convention()
        self._sanity_check_wavefunction(x_stop, run_backward)

    def _integrate_whittaker(self) -> None:
        if hasattr(self, "_w_list"):
            raise RuntimeError("The wavefunction was already integrated, you should not create it again.")

        logger.warning("Using Whittaker to get the wavefunction is not recommended! Use this only for comparison.")

        whitw_vectorized = np.vectorize(whitw, otypes=[float])
        m_star = self.element_properties.reduced_mass_au
        net_charge = self.element_properties.net_charge
        whitw_list = whitw_vectorized(self.nu, self.l_r + 0.5, m_star * 2 * net_charge * self.x_list / self.nu)

        # to get the correct whittaker functions, u_list should be multiplied with nu^(3/2)
        # however, to normalize them, we divide again by nu^(3/2) and thus we can skip this step.
        # Note: for net_charge != 1 (ions) the analytic normalization below is not exact; the Whittaker
        # path is only intended for comparison (the production path is Numerov, which normalizes
        # numerically via self.norm).
        u_list: NDArray = whitw_list / np.sqrt(self.nu**2 * gamma(self.nu + self.l_r + 1) * gamma(self.nu - self.l_r))
        w_list: NDArray = u_list / np.sqrt(self.z_list)

        self._w_list = w_list
        self._apply_sign_convention()

    def _sanity_check_wavefunction(self, z_force_stop: float, run_backward: bool) -> bool:  # noqa: C901, PLR0915, PLR0912
        """Do some sanity checks on the wavefunction.

        Check if the wavefuntion fulfills the following conditions:
        - The wavefunction is positive (or zero) at the inner boundary.
        - The wavefunction is close to zero at the inner boundary.
        - The wavefunction is close to zero at the outer boundary.
        - The wavefunction has exactly (n - l - 1) nodes.
        - The integration stopped before z_force_stop (for l>0)
        """
        warning_msgs: list[str] = []
        start_id = np.argwhere(self.w_list != 0).flatten()[0]

        # Check and Correct if divergence of the wavefunction
        w_list_abs = np.abs(self.w_list)
        idmax = np.argmax(w_list_abs)
        w_abs_max = w_list_abs[idmax]
        outer_max = next(
            (
                w_list_abs[i]
                for i in range(len(w_list_abs) - 2, 0, -1)
                if w_list_abs[i] > w_list_abs[i - 1] and w_list_abs[i] > w_list_abs[i + 1]
            ),
            0,
        )
        if outer_max == 0:
            warning_msgs.append("Could not find a local maximum of the wavefunction at the outer boundary.")
        elif idmax <= start_id + 5 and w_abs_max / outer_max > 5:
            warning_msgs.append(
                f"Wavefunction diverges at the inner boundary, w_abs_max / outer_max={w_abs_max / outer_max:.2e}",
            )
            warning_msgs.append("Trying to correct the wavefunction.")
            first_ind = next(ind for ind in np.argwhere(w_list_abs < outer_max).flatten() if ind > start_id)
            self._w_list[:first_ind] = 0
            self._w_list /= self.norm

        # From here on, we want to keep the logic from the old sanity check,
        # where the wavefunction is restricted to the region where numerov actually ran
        # and not set to 0 in the region where numerov did not run
        w_list = self.w_list[start_id:]
        z_list = self.z_list[start_id:]
        steps = len(w_list)

        # Check the maximum of the wavefunction
        idmax = np.argmax(np.abs(w_list))
        if idmax < 0.05 * steps:
            warning_msgs.append(
                f"The maximum of the wavefunction is close to the inner boundary (idmax={idmax}) "
                "probably due to inner divergence of the wavefunction. "
            )

        # Check the weight of the wavefunction at the inner boundary
        inner_ind = 10
        inner_weight = (
            2 * np.sum(w_list[:inner_ind] * w_list[:inner_ind] * z_list[:inner_ind] * z_list[:inner_ind]) * self.dz
        )
        inner_weight_scaled_to_whole_grid = inner_weight * steps / inner_ind

        tol = 1e-4
        # for low nu the wavefunction converges not as good and still has more weight at the inner boundary
        if self.nu <= 10:
            tol = 8e-3
        elif self.nu <= 16:
            tol = 2e-3

        # for divalent atoms, the wavefunction (currently) is not expected to be close to zero at the inner boundary
        is_alkali_like = self.element_properties.number_valence_electrons == 1
        if is_alkali_like and inner_weight_scaled_to_whole_grid > tol:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the inner boundary"
                f" (inner_weight_scaled_to_whole_grid={inner_weight_scaled_to_whole_grid:.2e})"
            )

        # Check the wavefunction at the outer boundary
        outer_ind = int(0.95 * steps)
        outer_wf = np.abs(w_list[outer_ind:])
        if np.max(outer_wf) > 1e-7:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the outer boundary, max={np.max(outer_wf):.2e}"
            )

        outer_weight = 2 * np.sum(outer_wf * outer_wf * z_list[outer_ind:] * z_list[outer_ind:]) * self.dz
        outer_weight_scaled_to_whole_grid = outer_weight * steps / len(outer_wf)
        if outer_weight_scaled_to_whole_grid > 1e-10:
            warning_msgs.append(
                f"The wavefunction is not close to zero at the outer boundary,"
                f" (outer_weight_scaled_to_whole_grid={outer_weight_scaled_to_whole_grid:.2e})"
            )

        # Check the number of nodes
        nodes = self.nodes
        if self.n_expected is not None and nodes != self.n_expected - self.l_r - 1:
            warning_msgs.append(
                f"The wavefunction has {nodes} nodes, but should have {self.n_expected - self.l_r - 1} nodes."
            )

        # Check that numerov stopped and did not run until z_force_stop
        if run_backward:
            z_stop = z_list[np.argwhere(w_list != 0).flatten()[0]]
            z_tol = 0.035 if is_alkali_like else 0.05
            if self.l_r == 0 and z_stop > z_tol:  # z_stop should run almost to zero for l=0
                warning_msgs.append(f"The integration for l=0 did stop at {z_stop} (should be close to zero).")
            if self.l_r > 0 and z_force_stop > z_stop - self.dz / 2 and inner_weight_scaled_to_whole_grid > 1e-6:
                warning_msgs.append(
                    f"The integration did not stop before z_force_stop, z={z_stop}, z_force_stop={z_force_stop}"
                )
        else:
            z_stop = z_list[np.argwhere(w_list != 0).flatten()[-1]]
            if self.l_r > 0 and z_force_stop < z_stop + self.dz / 2:
                warning_msgs.append(f"The integration did not stop before z_force_stop, z={z_stop}")

        if warning_msgs:
            msg = f"The wavefunction for the radial_ket {self} has some issues:"
            msg += "\n      ".join(["", *warning_msgs])
            logger.warning(msg)
            return False

        return True

    def _apply_sign_convention(self) -> None:
        """Set the sign of the wavefunction according to the ``sign_convention`` parameter.

        The sign convention is taken from the ``sign_convention`` argument passed to :meth:`__init__`:
            - None: Leave the wavefunction as it is.
            - "n_l_1": The wavefunction is defined to have the sign of (-1)^{(n_expected - l - 1)} at the outer boundary
            - "positive_at_outer_bound": The wavefunction is defined to be positive at the outer boundary.
        """
        sign_convention: WavefunctionSignConvention = self._sign_convention
        if sign_convention is None:
            return

        current_outer_sign = self.get_outer_sign()
        if sign_convention == "positive_at_outer_bound":
            if current_outer_sign < 0:
                self._w_list = -self._w_list
        elif sign_convention == "n_l_1":
            if self.n_expected is None:
                raise ValueError("n_expected must be given to apply the 'n_l_1' sign convention.")
            if current_outer_sign * (-1) ** (self.n_expected - self.l_r - 1) < 0:
                self._w_list = -self._w_list
        else:
            raise ValueError(f"Unknown sign convention: {sign_convention}")

    def _calc_matrix_element_au(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        *,
        integration_method: INTEGRATION_METHODS = "sum",
    ) -> float:
        if not isinstance(other, RadialKet):
            return super()._calc_matrix_element_au(other, k_radial=k_radial, integration_method=integration_method)

        if other not in self._matrix_element_cache and self in other._matrix_element_cache:  # noqa: SLF001
            return other._calc_matrix_element_au(self, k_radial=k_radial, integration_method=integration_method)  # noqa: SLF001

        cache = self._matrix_element_cache.get(other)
        if cache is None:
            cache = self._matrix_element_cache[other] = {}
        cache_key = (k_radial, integration_method)
        if cache_key not in cache:
            cache[cache_key] = super()._calc_matrix_element_au(
                other, k_radial=k_radial, integration_method=integration_method
            )

        return cache[cache_key]


class RadialDummy(Radial, metaclass=CachedABCMeta):
    _is_dummy = True
    nu: float

    def __init__(self, coeff: float, nu: float, *, element_properties: ElementProperties | None = None) -> None:
        if not nu > 0:
            raise ValueError(f"nu must be larger than 0, but is {nu=}")
        self.nu = nu

        self._coeff = coeff
        self.element_properties = element_properties

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._coeff}, nu={self.nu})"

    @property
    def norm(self) -> float:
        return abs(self._coeff)

    def __mul__(self, scalar: float) -> RadialDummy:
        if not isinstance(scalar, Number):
            return NotImplemented
        return RadialDummy(scalar * self._coeff, self.nu, element_properties=self.element_properties)

    def get_outer_sign(self) -> int:
        return int(np.sign(self._coeff))

    def _calc_matrix_element_au(
        self,
        other: Radial,
        k_radial: RadialMatrixElementOperator,
        *,
        integration_method: INTEGRATION_METHODS = "sum",  # noqa: ARG002
    ) -> float:
        if not other._is_dummy or k_radial != 0:  # noqa: SLF001
            return 0.0
        # the wavefunction of a dummy is unknown, so we approximate it by a Coulomb (Whittaker) function,
        # for which the overlap can be approximated as
        # 2 * sqrt(nu1 * nu2) / (nu1 + nu2) * sinc(nu1 - nu2), with the normalized sinc(x) = sin(pi x) / (pi x)
        # it is one for equal nu and vanishes if the nu differ by a non-zero integer
        overlap = 2 * np.sqrt(self.nu * other.nu) / (self.nu + other.nu) * np.sinc(self.nu - other.nu)  # type: ignore [operator]
        return self._coeff.conjugate() * other._coeff * overlap  # type: ignore [attr-defined,no-any-return]  # noqa: SLF001
