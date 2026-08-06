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
from scipy.sparse import csr_matrix

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.physics import (
    N_FIELDS,
    N_GROUPS,
    clad_coolant_flux,
    flow_rate,
    kinetics_weights,
    make_rhs,
    nodal_power,
    node_geometry,
    prompt_jump_power,
    reactivity_components,
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
    c: FloatArray  # delayed-neutron precursors, shape (6, n_time)
    power: FloatArray  # normalised power P(t)/P_0
    rho: FloatArray  # net reactivity, absolute units — never dollars
    rho_doppler: FloatArray  # Doppler component of `rho` (Eq. 4.5-3)
    rho_void: FloatArray  # coolant-density/void component of `rho` (Eq. 4.5-25)
    flow: FloatArray  # mass flow rate w(t) [kg/s]
    H: float = 1.0  # active height [m], for the voided-length integral
    stopped_early: bool = False  # True if the run hit the validity limit

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

    @property
    def peak_rho_over_beta(self) -> float:
        """``max_t rho/beta`` — the prompt-jump pole tripwire (deviation D-KIN-1).

        The closure has a pole at ``rho = beta``. Every run must report this and
        anything approaching 1 is outside the approximation, not a result.
        """
        return float(self.rho.max() / self._beta)

    def void_worth_is_exercised(self, tol: float = 1e-6) -> bool:
        """Report whether the void feedback ever inserted **positive** reactivity.

        The sodium void worth changes sign near the top of the core, and boiling
        starts at the top because that is where the coolant is hottest. If the
        voided region never reaches below that sign change, every void
        contribution is negative and the transient is self-limiting for a reason
        that has nothing to do with the physics under study — the positive void
        feedback this project exists to examine is simply never sampled.

        Measured with the shipped defaults this is **False**: all voiding sits
        above ``zeta_sign``, the void component spans -0.036 to 0 in units of
        ``beta``, and :attr:`peak_rho_over_beta` is exactly 0. Report it
        alongside the tripwire so a null result is never mistaken for a
        stabilising one.

        ``tol`` is a fraction of ``beta``, because the void integral picks up
        positive values of order 1e-20 from the first few steps — roundoff in a
        positive-worth cell whose ``alpha`` is numerically zero. A bare ``> 0``
        answers "yes" to noise.
        """
        return bool(np.any(self.rho_void > tol * self._beta))

    _beta: float = 3.5e-3

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
    # Subcooled everywhere, so void-free; and c_i = 1 makes P(0) = 1 exactly,
    # since sum(beta_i) / (beta - 0) = 1. No criticality offset is needed.
    return T_f, T_cl, T_s, T_c, np.zeros_like(T_c), np.ones(N_GROUPS)


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
    return T_f, T_cl, T_s, T_c, np.zeros_like(T_c), np.ones(N_GROUPS)


def jacobian_sparsity(p: AxialParams, *, feedback: bool = False) -> Any:  # noqa: ANN401
    """Sparsity pattern of the method-of-lines Jacobian.

    Without feedback every coupling is local — a node talks to its material
    neighbours and, for the advected fields, to its upwind neighbour — so of the
    ``(5n+6)^2`` entries only ``O(n)`` are non-zero. Supplying that to Radau took
    the ``n = 640`` convergence study from intractable to seconds.

    **With feedback the picture changes qualitatively.** The reactivity is an
    *integral* over the whole channel (Eq. 4.5-3, Eq. 4.5-25) and the resulting
    power amplitude drives every node, so every row depends on every ``T_f`` and
    every ``alpha``. Those blocks are genuinely dense and the pattern says so;
    the saving is correspondingly smaller, which is a property of the physics
    rather than of the implementation.

    Built as a dense boolean array and converted at the end. An earlier version
    used ``lil_matrix`` slice assignment for the kinetics blocks, which silently
    did nothing: Radau then had a Jacobian missing 312 entries and could not
    advance past ``t = 0.5 s``. A wrong pattern is worse than no pattern, and it
    fails in a way that looks like a stiff-solver problem rather than a bug.
    """
    n = p.n_axial
    size = N_FIELDS * n + N_GROUPS
    m = np.zeros((size, size), dtype=np.int8)
    f, cl, st, c, a = 0, n, 2 * n, 3 * n, 4 * n
    j = np.arange(n)

    m[f + j, f + j] = 1  # fuel <- fuel, cladding
    m[f + j, cl + j] = 1
    m[cl + j, f + j] = 1  # cladding <- fuel, cladding, coolant, void (film)
    m[cl + j, cl + j] = 1
    m[cl + j, c + j] = 1
    m[cl + j, a + j] = 1
    m[st + j, st + j] = 1  # structure <- structure, coolant, void (film)
    m[st + j, c + j] = 1
    m[st + j, a + j] = 1
    m[c + j, cl + j] = 1  # coolant <- cladding, structure, coolant, void
    m[c + j, st + j] = 1
    m[c + j, c + j] = 1
    m[c + j, a + j] = 1
    m[c + j[1:], c + j[:-1]] = 1  # ... and its upwind neighbour
    m[a + j, cl + j] = 1  # void <- the wall heat that makes it ...
    m[a + j, st + j] = 1
    m[a + j, c + j] = 1  # ... the superheat that gates it ...
    m[a + j, a + j] = 1
    m[a + j[1:], a + j[:-1]] = 1  # ... and its own upwind neighbour
    kin = N_FIELDS * n
    g = np.arange(N_GROUPS)
    m[kin + g, kin + g] = 1  # precursors always decay into themselves

    if feedback:
        m[:, kin:] = 1  # every field is driven by the power amplitude
        m[kin:, :] = 1  # the precursors see every field through the integrals
        m[:, f : f + n] = 1  # ... and the Doppler integral couples all T_f ...
        m[:, a : a + n] = 1  # ... the void integral all alpha
    return csr_matrix(m)


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
    T_limit: float = sodium.T_MAX,
    feedback: bool = False,
) -> AxialTrajectory:
    """Integrate the channel from its exact steady state over ``[0, t_end]``.

    ``amplitude`` prescribes the normalised power ``P(t)/P_0`` and defaults to a
    constant 1.0 — M2 and M3 are the prescribed-power milestones.

    **The run terminates when any temperature reaches ``T_limit``**, which
    defaults to :data:`~pinn_sfr_transient.axial.sodium.T_MAX`, the upper bound
    of the section 12.13 property fits. That is not a numerical convenience.
    Past dryout this model has no melting, no cladding motion and no fuel
    relocation (Chapters 8-16, deviation D-SCOPE-1), and the sodium correlations
    themselves stop at 2270 K; integrating on would extrapolate three models at
    once and present the result as a prediction. Stopping makes the model state
    in its own output exactly where it ceases to apply, and ``stopped_early``
    records that it did.
    """
    y0 = steady_state_vector(p)
    T_f0 = steady_state(p)[0] if feedback else None
    t_eval = np.linspace(0.0, p.t_end, n_out)
    n_temp = (N_FIELDS - 1) * p.n_axial  # the void is not a temperature

    def _too_hot(_t: float, y: FloatArray) -> float:
        return float(T_limit - np.max(y[:n_temp]))

    _too_hot.terminal = True  # ty: ignore[unresolved-attribute]
    _too_hot.direction = -1.0  # ty: ignore[unresolved-attribute]

    sol = solve_ivp(
        make_rhs(p, amplitude, T_f0),
        (0.0, p.t_end),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        jac_sparsity=jacobian_sparsity(p, feedback=feedback),
        events=_too_hot,
    )
    if not sol.success:  # pragma: no cover - solver failure is not expected
        msg = f"reference integration failed: {sol.message}"
        raise RuntimeError(msg)

    T_f, T_cl, T_s, T_c, alpha, c = unpack(sol.y, p.n_axial)
    if feedback:
        w_D, w_void = kinetics_weights(p)
        parts = [
            reactivity_components(T_f[:, i], alpha[:, i], T_f0, w_D, w_void, p)
            for i in range(sol.t.size)
        ]
        rho_doppler = np.array([d for d, _ in parts])
        rho_void = np.array([v for _, v in parts])
        rho = p.rho_ext + rho_doppler + rho_void
        power = np.array([prompt_jump_power(c[:, i], rho[i], p) for i in range(sol.t.size)])
    else:
        amp = amplitude if amplitude is not None else (lambda _t: 1.0)
        power = np.array([amp(float(t)) for t in sol.t])
        rho = np.zeros_like(power)
        rho_doppler = np.zeros_like(power)
        rho_void = np.zeros_like(power)
    return AxialTrajectory(
        t=sol.t,
        zeta=p.zeta_nodes(),
        T_f=T_f,
        T_cl=T_cl,
        T_s=T_s,
        T_c=T_c,
        alpha=alpha,
        c=c,
        power=power,
        rho=rho,
        rho_doppler=rho_doppler,
        rho_void=rho_void,
        flow=flow_rate(sol.t, p),
        H=p.H,
        _beta=p.beta_eff,
        stopped_early=bool(sol.status == 1),
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

    **The amplitude comes from the trajectory, not from a default of one.**
    ``traj.power`` is the normalised power actually delivered — the prescribed
    ``amplitude`` sampled on the output grid under Plan B, and the prompt-jump
    output under Plan A. Defaulting to one silently mis-stated the closed-loop
    balance as **0.382** where the true figure is 1.5e-5, because feedback drives
    the power down to 0.498 of nominal. Pass ``amplitude`` only to override.

    Returns the absolute closure error divided by the total energy delivered.
    """
    geo = node_geometry(p)
    f_nodes = p.power_shape(p.zeta_nodes())
    q_fuel, q_cool = nodal_power(1.0, f_nodes, p)
    p_discrete = float((q_fuel + q_cool).sum())
    amp = traj.power if amplitude is None else np.array([amplitude(t) for t in traj.t])

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
