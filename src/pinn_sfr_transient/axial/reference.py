"""Stiff reference solver for the axial channel (milestones M2 and M4).

Method of lines: :mod:`pinn_sfr_transient.axial.physics` discretises Chapter 3's
energy equations in ``z``, and an implicit Radau integrator advances the
resulting ODE system in ``t``. This is the **held-out truth** for every
downstream milestone — the PINN never sees it during training, only at test time
(the protocol the 0D model already follows, ``docs/neural_network.md`` §7).

The system is genuinely stiff. With the default parameters the coolant transit
time across one axial cell is ~2.4 ms while the pump coasts down over 5 s and the
horizon is 60 s: a spread of ~2.5e4. An explicit integrator would be useless.
Removing the prompt neutron mode (deviation D-KIN-1) buys a factor of ~2300, not
a non-stiff problem — hence Radau, exactly as the milestone plan anticipated.

**Steady state is exact, not iterated.** Dropping the time derivatives — which is
how the manual obtains its own steady state (section 3.4) — leaves a closed
marching solution, so :func:`steady_state` is an *analytic* answer for the
discretised system rather than a numerical one. That makes it a genuine
verification oracle: the solver has to reproduce it to round-off from a steady
start, and nothing about the discretisation is being assumed correct in the
process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.physics import (
    N_FIELDS,
    clad_coolant_flux,
    flow_rate,
    make_rhs,
    nodal_power,
    node_geometry,
    unpack,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.config import FloatArray

_NEWTON_ITERS: int = 60
_NEWTON_TOL: float = 1e-12


@dataclass(slots=True)
class AxialTrajectory:
    """Solved single-phase transient on a uniform time grid.

    Every temperature field has shape ``(n_axial, n_time)``: axial node first,
    time second.
    """

    t: FloatArray
    zeta: FloatArray
    T_f: FloatArray
    T_cl: FloatArray
    T_s: FloatArray
    T_c: FloatArray
    alpha: FloatArray  # void fraction, dimensionless
    flow: FloatArray  # mass flow rate w(t) [kg/s]
    H: float = 1.0  # active height [m], for the voided-length integral

    @property
    def T_out(self) -> FloatArray:
        """Coolant temperature leaving the top node [K]."""
        return self.T_c[-1]

    @property
    def peak_clad(self) -> float:
        """Hottest cladding temperature anywhere, any time [K]."""
        return float(self.T_cl.max())

    @property
    def voided_length(self) -> FloatArray:
        """Voided length ``L_void(t) = integral of alpha dz`` [m].

        The metric M4 is judged on rather than a relative ``L2`` of ``alpha``:
        the void field is zero over most of the domain for most of the transient,
        so a relative norm there is dominated by its denominator and reads as
        noise. An absolute length in metres is what a reactor engineer checks.
        """
        return self.alpha.sum(axis=0) * (self.zeta[1] - self.zeta[0]) * self.H

    def onset(self, threshold: float = 0.01) -> tuple[float, float]:
        """First time and normalised height at which the void exceeds ``threshold``.

        Returns ``(nan, nan)`` if boiling never starts.
        """
        hit = self.alpha > threshold
        if not hit.any():
            return float("nan"), float("nan")
        i = int(np.argmax(hit.any(axis=0)))
        return float(self.t[i]), float(self.zeta[int(np.argmax(hit[:, i]))])


def steady_state(p: AxialParams) -> tuple[FloatArray, ...]:
    """Exact steady state of the discretised system, as ``(T_f, T_cl, T_s, T_c, alpha)``.

    Obtained by dropping the time derivatives, which is how the manual derives
    its own steady state (section 3.4). The system then unwinds in closed form:

    * structure — no source, so ``q_sc = 0`` and ``T_s = T_c``;
    * cladding and fuel — ``q_ec = q_fe = Q_fuel``, the nodal fission power;
    * coolant — ``T_c[j] = T_c[j-1] + Q_total[j] / (w c)``, marching up from the
      inlet. Summing telescopes to ``T_out - T_in = P_0 / (w_0 c_c)``, so the
      whole-core temperature rise is exact regardless of mesh.

    Only the fuel needs iteration, and only because Eq. 3.3-4 carries the
    radiation term: ``h_b dT + eps sigma dT^4 = Q_fuel`` is solved per node by
    Newton from the conduction-only guess, which converges in a handful of steps.
    """
    geo = node_geometry(p)
    f_nodes = p.power_shape(p.zeta_nodes())
    q_fuel, q_cool = nodal_power(1.0, f_nodes, p)
    w = p.w_0  # g(0) = 1

    # Coolant: march the telescoping sum up the channel.
    rise = (q_fuel + q_cool) / (w * p.c_c)
    T_c = p.T_in + np.cumsum(rise)
    T_s = T_c.copy()

    # Cladding: linear film, so direct.
    T_cl = T_c + q_fuel / (p.h_clad_coolant * geo.A_ec)

    # Fuel: gap conductance plus radiation, Newton on each node.
    T_f = _fuel_from_flux(q_fuel, T_cl, geo.A_fe, p)
    # The nominal state is subcooled everywhere, so the channel starts void-free.
    return T_f, T_cl, T_s, T_c, np.zeros_like(T_c)


def _fuel_from_flux(q: FloatArray, T_cl: FloatArray, area: float, p: AxialParams) -> FloatArray:
    """Invert Eq. 3.3-4 for the fuel temperature; radiation makes it nonlinear."""
    sigma = 5.670374419e-8
    T_f = T_cl + q / (p.h_gap * area)
    for _ in range(_NEWTON_ITERS):
        f = area * (p.h_gap * (T_f - T_cl) + p.emissivity * sigma * (T_f**4 - T_cl**4)) - q
        step = f / (area * (p.h_gap + 4.0 * p.emissivity * sigma * T_f**3))
        T_f = T_f - step
        if np.max(np.abs(step)) < _NEWTON_TOL:
            break
    else:  # pragma: no cover - defensive; Newton converges in a handful of steps
        msg = "fuel temperature did not converge"
        raise RuntimeError(msg)
    return T_f


def steady_profile(p: AxialParams, zeta: FloatArray) -> tuple[FloatArray, ...]:
    """Continuous steady state at arbitrary heights, ``(T_f, T_cl, T_s, T_c, alpha)``.

    The mesh-free twin of :func:`steady_state`, and the PINN's hard initial
    condition. Where the discrete version telescopes a nodal sum, this uses the
    closed-form power integral,
    ``T_c(zeta) = T_in + (P_0 / (w_0 c_c)) F(zeta)``, so the ansatz can satisfy
    both the initial condition and the inlet boundary condition *identically*
    rather than approximately.

    The two steady states differ by the midpoint-rule error of the axial shape,
    ``O(dz^2)``, and converge to each other as the mesh refines — asserted in the
    tests rather than assumed.
    """
    dT = p.P_0 / (p.w_0 * p.c_c)
    T_c: FloatArray = np.asarray(p.T_in + dT * p.power_shape_integral(zeta), dtype=np.float64)
    T_s = T_c
    # Per unit length, i.e. the fluxes of `node_geometry` with dz = 1.
    q_fuel: FloatArray = np.asarray(
        (1.0 - p.gamma_c) * p.P_0 * p.power_shape(zeta) / p.H, dtype=np.float64
    )
    T_cl = T_c + q_fuel / (p.h_clad_coolant * 2.0 * np.pi * p.r_co)
    T_f = _fuel_from_flux(q_fuel, T_cl, 2.0 * np.pi * p.r_fo, p)
    return T_f, T_cl, T_s, T_c, np.zeros_like(T_c)


def jacobian_sparsity(p: AxialParams) -> Any:  # noqa: ANN401 - scipy sparse matrix
    """Sparsity pattern of the method-of-lines Jacobian.

    Every coupling in :func:`~pinn_sfr_transient.axial.physics.derivatives` is
    local: a node talks to the material above and below it in the resistance
    chain, and the coolant additionally to its **upwind neighbour** ``j-1``. So
    of the ``(4n)^2`` entries only ``O(n)`` are non-zero.

    Handing this to Radau matters a great deal in practice. Without it the
    integrator finite-differences a dense Jacobian and factorises it densely,
    which is ``O(n^3)`` per step: the mesh-convergence study at ``n = 640``
    (2560 states) does not finish in reasonable time. With it, seconds.
    """
    n = p.n_axial
    pattern = lil_matrix((N_FIELDS * n, N_FIELDS * n), dtype=np.int8)
    f, cl, st, c = 0, n, 2 * n, 3 * n
    j = np.arange(n)
    pattern[f + j, f + j] = 1  # fuel <- fuel, cladding
    pattern[f + j, cl + j] = 1
    pattern[cl + j, f + j] = 1  # cladding <- fuel, cladding, coolant
    pattern[cl + j, cl + j] = 1
    pattern[cl + j, c + j] = 1
    pattern[st + j, st + j] = 1  # structure <- structure, coolant
    pattern[st + j, c + j] = 1
    pattern[c + j, cl + j] = 1  # coolant <- cladding, structure, coolant
    pattern[c + j, st + j] = 1
    pattern[c + j, c + j] = 1
    pattern[c + j[1:], c + j[:-1]] = 1  # ... and its upwind neighbour
    a = 4 * n
    pattern[a + j, cl + j] = 1  # void <- the wall heat that makes it ...
    pattern[a + j, st + j] = 1
    pattern[a + j, c + j] = 1  # ... the superheat that gates it ...
    pattern[a + j, a + j] = 1
    pattern[a + j[1:], a + j[:-1]] = 1  # ... and its own upwind neighbour
    return pattern.tocsr()


def steady_state_vector(p: AxialParams) -> FloatArray:
    """Steady state packed into the flat state vector used by the solver."""
    return np.concatenate(steady_state(p))


def solve_reference(  # noqa: PLR0913 - solver knobs are clearer flat than bundled
    p: AxialParams,
    *,
    n_out: int = 601,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-8,
    amplitude: Callable[[float], float] | None = None,
) -> AxialTrajectory:
    """Integrate the channel from its exact steady state over ``[0, t_end]``.

    ``amplitude`` prescribes the normalised power ``P(t)/P_0`` and defaults to a
    constant 1.0. M2 is the prescribed-power milestone: the transient is driven
    entirely by the pump coast-down.
    """
    y0 = steady_state_vector(p)
    t_eval = np.linspace(0.0, p.t_end, n_out)
    sol = solve_ivp(
        make_rhs(p, amplitude),
        (0.0, p.t_end),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        jac_sparsity=jacobian_sparsity(p),
    )
    if not sol.success:  # pragma: no cover - solver failure is not expected
        msg = f"reference integration failed: {sol.message}"
        raise RuntimeError(msg)

    T_f, T_cl, T_s, T_c, alpha = unpack(sol.y, p.n_axial)
    return AxialTrajectory(
        t=sol.t,
        zeta=p.zeta_nodes(),
        T_f=T_f,
        T_cl=T_cl,
        T_s=T_s,
        T_c=T_c,
        alpha=alpha,
        flow=flow_rate(sol.t, p),
        H=p.H,
    )


def energy_balance(
    traj: AxialTrajectory,
    p: AxialParams,
    amplitude: Callable[[float], float] | None = None,
) -> float:
    """Relative closure error of the channel energy balance over the transient.

    Summing the four nodal energy equations over ``z`` cancels every internal
    flux — ``q_fe`` between fuel and cladding, ``q_ec`` and ``q_sc`` into the
    coolant — and the upwind advection telescopes, leaving

    ``dE/dt = P_discrete(t) - w(t) c (T_out(t) - T_in)``

    with ``E`` the total stored energy. Comparing the change in ``E`` against the
    time integral of the right-hand side tests the *discretisation*, not just the
    integrator: a wrong surface area, heat capacity or stencil breaks it even
    when the solution looks plausible.

    ``P_discrete`` is the sum of the nodal sources, **not** the nominal ``P_0``.
    Those differ by the midpoint-rule error of the axial power shape (~1.6e-4 at
    ``n_axial = 40``), and using ``P_0`` here would floor the closure at that
    value — measuring the quadrature of the source rather than the conservation
    property being tested. That convergence is checked separately, by
    ``test_discrete_power_converges_to_the_nominal_rating``.

    Returns the absolute closure error divided by the total energy delivered.
    """
    geo = node_geometry(p)
    f_nodes = p.power_shape(p.zeta_nodes())
    q_fuel, q_cool = nodal_power(1.0, f_nodes, p)
    p_discrete = float((q_fuel + q_cool).sum())
    amp = np.ones_like(traj.t) if amplitude is None else np.array([amplitude(t) for t in traj.t])

    # Sensible storage in the four materials, plus the LATENT heat locked up in
    # the vapour. Omitting the latent term would make the balance fail the moment
    # boiling starts -- which is exactly how the M4 conservation defect surfaced.
    T_sat = sodium.saturation_temperature(p.p_system)
    latent_per_unit_void = sodium.vapor_density(T_sat) * sodium.latent_heat(T_sat) * p.A_c * geo.dz
    stored = (
        geo.C_f * traj.T_f.sum(axis=0)
        + geo.C_cl * traj.T_cl.sum(axis=0)
        + geo.C_s * traj.T_s.sum(axis=0)
        + geo.C_c * traj.T_c.sum(axis=0)
        + latent_per_unit_void * traj.alpha.sum(axis=0)
    )
    # Convected out of the top: sensible heat in the liquid, AND latent heat in
    # the vapour that leaves with it. Omitting the second term leaves a residual
    # of exactly the vapour outflow -- about 45 W against 50 kW here, i.e. 9e-4,
    # which is what the closure floored at before this term was added.
    u_out = traj.flow / (p.rho_c * p.A_c)
    latent_out = (
        u_out * traj.alpha[-1] * sodium.vapor_density(T_sat) * sodium.latent_heat(T_sat) * p.A_c
    )
    net = amp * p_discrete - traj.flow * p.c_c * (traj.T_out - p.T_in) - latent_out
    delivered = _trapezoid(amp * p_discrete, traj.t)
    return float(abs((stored[-1] - stored[0]) - _trapezoid(net, traj.t)) / delivered)


def clad_coolant_heat(traj: AxialTrajectory, p: AxialParams) -> FloatArray:
    """Total cladding-to-coolant heat flow at each time [W] (diagnostic)."""
    geo = node_geometry(p)
    return clad_coolant_flux(traj.T_cl, traj.T_c, geo, p).sum(axis=0)


def _trapezoid(y: FloatArray, x: FloatArray) -> float:
    """Trapezoidal integral.

    Hand-rolled because ``np.trapezoid`` needs numpy >= 2.0 and ``np.trapz`` is
    deprecated there, while this package's floor is 1.26 so it installs on Colab
    without upgrading numpy in the running kernel.
    """
    return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x)))
