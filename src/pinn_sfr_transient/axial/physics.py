"""Single-phase axial thermal-hydraulics (SAS4A/SASSYS-1 Chapter 3).

The pre-boiling energy balance for one channel, discretised on the axial mesh of
:class:`~pinn_sfr_transient.axial.config.AxialParams`. Four temperature fields —
fuel, cladding, structure, coolant — one lumped node per material per axial
position, following the manual's equations:

* **Eq. 3.3-4** fuel-to-cladding flux, gap conductance **plus radiation**;
* **Eq. 3.3-5** coolant energy in conservative flux form, with **all three**
  source terms ``Q_c + Q_ec + Q_sc``;
* **Eq. 3.3-6** direct neutron and gamma heating deposited in the coolant, the
  fraction ``gamma_c`` of total power — it bypasses the fuel and cladding
  thermal lag entirely, so power reaches the coolant with no delay;
* **Eq. 3.9-1** pre-boiling momentum, whose stated assumption is that the flow
  rate ``w = w(t)`` is **independent of z** (incompressible liquid, one slug
  filling the channel). That is what makes the advection term below exact rather
  than approximate — but only until voiding starts (deviation D-TH-2).

:func:`derivatives` is **backend-agnostic**: numpy, torch and JAX all evaluate
the same expression tree, so the M3 PINN residual and this reference solver
cannot drift apart. :func:`make_rhs` is the thin scipy adapter that packs the
four fields into one state vector.

Boiling is **not** here — this is Chapter 3's regime. Void onset and the mixture
field arrive at M4; the kinetics closure at M6. Until then power is prescribed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial._backend import xp as _xp

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.config import FloatArray

type Field = Any
"""A numeric field: numpy array, torch tensor or JAX array, or a plain float.

Named rather than bare ``Any`` so the backend-agnostic signatures below say what
they mean, and so they do not each need an ``ANN401`` suppression.
"""

STEFAN_BOLTZMANN: float = 5.670374419e-8
"""Stefan-Boltzmann constant [W/m^2-K^4], radiation term of Eq. 3.3-4."""

N_FIELDS: int = 5
"""Fields per axial node: fuel, cladding, structure and coolant temperatures, plus void."""


@dataclass(frozen=True, slots=True)
class NodeGeometry:
    """Per-axial-node volumes, surfaces and heat capacities [SI].

    Pure geometry, so it is computed once and shared by every backend.
    """

    dz: float
    A_fe: float  # fuel outer surface, fuel -> cladding path
    A_ec: float  # cladding outer surface, cladding -> coolant path
    A_sc: float  # structure wetted surface, structure -> coolant path
    C_f: float  # fuel heat capacity [J/K]
    C_cl: float  # cladding heat capacity [J/K]
    C_s: float  # structure heat capacity [J/K]
    C_c: float  # coolant heat capacity [J/K]


def node_geometry(p: AxialParams) -> NodeGeometry:
    """Derive the per-node geometry and heat capacities from ``p``."""
    dz = p.H / p.n_axial
    a_fe = 2.0 * np.pi * p.r_fo * dz
    a_ec = 2.0 * np.pi * p.r_co * dz
    a_sc = p.gamma_2 * a_ec
    v_f = np.pi * p.r_fo**2 * dz
    v_cl = np.pi * (p.r_co**2 - p.r_ci**2) * dz
    v_s = p.gamma_2 * 2.0 * np.pi * p.r_co * p.t_struct * dz
    v_c = p.A_c * dz
    return NodeGeometry(
        dz=dz,
        A_fe=a_fe,
        A_ec=a_ec,
        A_sc=a_sc,
        C_f=p.rho_f * p.c_f * v_f,
        C_cl=p.rho_cl * p.c_cl * v_cl,
        C_s=p.rho_s * p.c_s * v_s,
        C_c=p.rho_c * p.c_c * v_c,
    )


def line_geometry(p: AxialParams) -> NodeGeometry:
    """Per-unit-length geometry: :func:`node_geometry` with ``dz = 1``.

    Every area and heat capacity in :class:`NodeGeometry` scales linearly with
    ``dz``, so setting it to one turns the nodal quantities into per-metre ones.
    That lets :func:`continuous_derivatives` and :func:`derivatives` share the
    exact same flux functions — the PDE form and its discretisation cannot drift
    apart, because there is only one of each.
    """
    from dataclasses import replace  # noqa: PLC0415 - local to keep the import graph flat

    per_node = node_geometry(p)
    scale = 1.0 / per_node.dz
    return replace(
        per_node,
        dz=1.0,
        A_fe=per_node.A_fe * scale,
        A_ec=per_node.A_ec * scale,
        A_sc=per_node.A_sc * scale,
        C_f=per_node.C_f * scale,
        C_cl=per_node.C_cl * scale,
        C_s=per_node.C_s * scale,
        C_c=per_node.C_c * scale,
    )


# --- flow ------------------------------------------------------------------
def flow_fraction(t: Field, p: AxialParams) -> Field:
    """Normalised pump coast-down ``g(t)``, with ``g(0) = 1`` (deviation D-FLOW-1).

    The manual's table-look-up pump prescribes *head*, ``H(t) = H_r f(t)`` with
    ``f(0) = 1`` (Eq. 5.3-61), and lets the loop momentum equation produce the
    flow. We prescribe the flow directly and decay it exponentially to a
    natural-circulation floor. Because ``f_nc > 0`` the flow never reverses, so
    the single upstream boundary condition of Eq. 3.9-1 stays valid by
    construction — a guarantee that does **not** survive M4's boiling-induced
    pressure (risk R6).
    """
    xp = _xp(t)
    return p.f_nc + (1.0 - p.f_nc) * xp.exp(-t / p.tau_pump)


def flow_rate(t: Field, p: AxialParams) -> Field:
    """Coolant mass flow rate ``w(t)`` [kg/s], ``z``-independent per Eq. 3.9-1."""
    return p.w_0 * flow_fraction(t, p)


# --- heat transfer paths ---------------------------------------------------
def gap_flux(T_f: Field, T_cl: Field, geo: NodeGeometry, p: AxialParams) -> Field:
    """Fuel-to-cladding heat flow per node [W] — Eq. 3.3-4.

    ``q_fe = h_b (T_f - T_cl) + eps sigma (T_f^4 - T_cl^4)``. The radiation term
    is not decorative: above 1000 K it carries a meaningful share, and it is
    *stabilising*, growing faster than linearly as the fuel heats.
    """
    conduction = p.h_gap * (T_f - T_cl)
    radiation = p.emissivity * STEFAN_BOLTZMANN * (T_f**4 - T_cl**4)
    return geo.A_fe * (conduction + radiation)


def film_coefficient(h_wet: float, alpha: Field, p: AxialParams) -> Field:
    """Effective wall-to-coolant film coefficient as the node voids (milestone M5).

    Section 12.5.1 writes the wall-to-vapour resistance as ``1/h_ec + R_ehf``,
    where ``h_ec`` carries the *combined* liquid-film and vapour resistance. Here
    that combination is a void-weighted blend between the wetted value and a
    vapour value two to three orders smaller:

    ``h_eff = (1 - alpha) h_wet + alpha h_vapour``

    This is the safety-relevant coupling and the reason M4 alone was not enough.
    Losing the liquid does not merely stop the coolant heating up — it removes
    the heat path, so the cladding temperature runs away. Before M5 the model
    kept full wetted heat transfer in a fully voided node, which made a boiling
    run indistinguishable from one that could not boil.
    """
    return (1.0 - alpha) * h_wet + alpha * p.h_vapour


def coolant_capacity(alpha: Field, p: AxialParams) -> Field:
    """Volumetric heat capacity of the two-phase mixture [J/m^3-K].

    ``(1 - alpha) rho_l c_l + alpha rho_v c_g``: a voided node has almost no
    thermal inertia, so it tracks the wall almost instantly.

    **Not used in the residuals, deliberately.** Substituting it into the
    *temperature*-form energy equation breaks conservation: as ``alpha`` changes
    the capacity changes, and the correct accounting then needs an enthalpy
    formulation rather than ``c dT/dt``. Measured cost of using it anyway: the
    energy closure degrades from 3.6e-6 to ~2e-2. So M5 degrades the **film
    coefficient** — which is the mechanism section 12.5.1 actually describes —
    and leaves the mixture capacity to a future enthalpy-form revision. Kept here
    because it is the right expression and the tests pin its value.
    """
    T_sat = sodium.saturation_temperature(p.p_system)
    vapour = sodium.vapor_density(T_sat) * sodium.vapor_heat_capacity(T_sat)
    return (1.0 - alpha) * p.rho_c * p.c_c + alpha * vapour


def clad_coolant_flux(
    T_cl: Field, T_c: Field, geo: NodeGeometry, p: AxialParams, alpha: Field = 0.0
) -> Field:
    """Cladding-to-coolant heat flow per node [W] — the ``Q_ec`` term of Eq. 3.3-5."""
    return film_coefficient(p.h_clad_coolant, alpha, p) * geo.A_ec * (T_cl - T_c)


def struct_coolant_flux(
    T_s: Field, T_c: Field, geo: NodeGeometry, p: AxialParams, alpha: Field = 0.0
) -> Field:
    """Structure-to-coolant heat flow per node [W] — the ``Q_sc`` term of Eq. 3.3-5.

    Vanishes when ``gamma_2 = 0``, i.e. when the structure node is disabled.
    """
    return film_coefficient(p.h_struct_coolant, alpha, p) * geo.A_sc * (T_s - T_c)


def nodal_power(amplitude: Field, f_nodes: Field, p: AxialParams) -> tuple[Any, Any]:
    """Split the nodal power into its fuel and direct-coolant parts [W].

    Returns ``(Q_fuel, Q_coolant)``. The manual deposits a fraction ``gamma_c``
    of total power straight into the coolant by neutron and gamma heating
    (Eq. 3.3-6); the remainder is generated in the fuel. ``f_nodes`` is the
    normalised axial shape sampled at the node centres, so
    ``sum(f_nodes) / n_axial == 1``.
    """
    total = amplitude * p.P_0 * f_nodes / p.n_axial
    return (1.0 - p.gamma_c) * total, p.gamma_c * total


# --- boiling (milestone M4) -------------------------------------------------
def boiling_fraction(T_c: Field, p: AxialParams) -> Field:
    """Fraction of the wall heat going into vaporisation rather than sensible heat.

    The manual's onset criterion (section 12.4) is a hard threshold: a bubble
    forms when the coolant exceeds saturation by at least ``DTS``, about 10 K for
    the first bubble. A hard threshold has no usable autodiff gradient, so it is
    replaced by a logistic of width ``dT_smooth`` centred on the same criterion —
    the same smoothing the 0D model applies, but now around a *physical*
    saturation temperature from Eq. 12.13-4 rather than a demonstration value.

    ``dT_smooth`` is a real accuracy/trainability knob, not a fudge: too wide and
    boiling starts early and gradually, too narrow and the front is too sharp for
    a smooth network. It is swept, not guessed.
    """
    xp = _xp(T_c)
    T_sat = sodium.saturation_temperature(p.p_system)
    x = (T_c - T_sat - p.dT_superheat) / p.dT_smooth
    return 0.5 * (1.0 + xp.tanh(0.5 * x))


def latent_fraction(T_c: Field, alpha: Field, p: AxialParams) -> Field:
    """Fraction of the wall heat that becomes vapour, ``b (1 - alpha)``.

    ``b`` is the superheat switch of :func:`boiling_fraction`; the ``(1 - alpha)``
    factor shuts the source off as a node empties, so the void cannot leave
    ``[0, 1)`` even before the network's sigmoid clamps it.

    **The same fraction must be removed from the sensible balance and no more.**
    An earlier version diverted the full ``b q_wall`` from the coolant but only
    vaporised ``b (1 - alpha) q_wall``, so energy quietly vanished once a node
    approached dryout — caught by the energy-balance test, not by inspection.
    Returning the difference to the sensible term conserves energy exactly.
    Physically, at ``alpha -> 1`` the wall heat goes on heating what is now
    vapour; that the node keeps the liquid heat capacity is a documented
    simplification which M5's film and dryout model replaces.
    """
    return boiling_fraction(T_c, p) * (1.0 - alpha)


def vapour_source(T_c: Field, alpha: Field, q_wall: Field, p: AxialParams) -> Field:
    """Void generation rate ``d alpha/dt`` from wall heat [1/s].

    Once the superheat criterion is met the wall heat stops raising the coolant
    temperature and starts making vapour: ``Gamma = b_eff q_wall / lambda_vap``
    kilograms per second per metre, filling the flow area at the saturated vapour
    density of Eq. 12.13-6.
    """
    T_sat = sodium.saturation_temperature(p.p_system)
    rho_v = sodium.vapor_density(T_sat)
    lam = sodium.latent_heat(T_sat)
    return latent_fraction(T_c, alpha, p) * q_wall / (lam * rho_v * p.A_c)


# --- the coupled right-hand side -------------------------------------------
def derivatives(  # noqa: PLR0913 - five coupled fields plus the driving terms
    t: Field,
    T_f: Field,
    T_cl: Field,
    T_s: Field,
    T_c: Field,
    alpha: Field,
    p: AxialParams,
    geo: NodeGeometry,
    f_nodes: Field,
    amplitude: Field = 1.0,
) -> tuple[Any, ...]:
    """Time derivatives of the four temperatures [K/s] and the void fraction [1/s].

    Backend-agnostic: arithmetic plus ``exp``, ``tanh`` and ``concatenate``, all
    provided under the same name by numpy, torch and JAX, so the reference solver
    and the PINN residual share one expression tree.

    Both advected fields use **first-order upwind** differencing of the
    conservative flux form of Eq. 3.3-5. Upwind rather than a higher order was
    chosen at M2 precisely because it is monotone and will not oscillate across
    the boiling front introduced here.
    """
    xp = _xp(T_c)
    q_fe = gap_flux(T_f, T_cl, geo, p)
    q_ec = clad_coolant_flux(T_cl, T_c, geo, p, alpha)
    q_sc = struct_coolant_flux(T_s, T_c, geo, p, alpha)
    q_fuel, q_cool = nodal_power(amplitude, f_nodes, p)

    # Wall heat reaching the coolant. Below the superheat criterion it raises the
    # temperature; above it, it makes vapour instead (manual section 12.4).
    q_wall = q_cool + q_ec + q_sc
    b = latent_fraction(T_c, alpha, p)

    inlet = T_c[:1] * 0.0 + p.T_in
    T_up = xp.concatenate([inlet, T_c[:-1]])
    advection = flow_rate(t, p) * p.c_c * (T_up - T_c)

    # Void advects at the liquid velocity; the inlet is subcooled, so none enters.
    a_up = xp.concatenate([alpha[:1] * 0.0, alpha[:-1]])
    u = flow_rate(t, p) / (p.rho_c * p.A_c)
    d_alpha = vapour_source(T_c, alpha, q_wall / geo.dz, p) + u * (a_up - alpha) / geo.dz

    dT_s = (
        -film_coefficient(p.h_struct_coolant, alpha, p)
        * (T_s - T_c)
        / (p.rho_s * p.c_s * p.t_struct)
    )
    return (
        (q_fuel - q_fe) / geo.C_f,
        (q_fe - q_ec) / geo.C_cl,
        dT_s,
        (advection + (1.0 - b) * q_wall) / geo.C_c,
        d_alpha,
    )


def continuous_derivatives(  # noqa: PLR0913 - five coupled fields plus the driving terms
    t: Field,
    T_f: Field,
    T_cl: Field,
    T_s: Field,
    T_c: Field,
    alpha: Field,
    dTc_dz: Field,
    dalpha_dz: Field,
    p: AxialParams,
    geo: NodeGeometry,
    f_zeta: Field,
    amplitude: Field = 1.0,
) -> tuple[Any, ...]:
    """PDE right-hand sides — the mesh-free form of :func:`derivatives`.

    Identical physics, with the discrete upwind differences replaced by the true
    axial gradients supplied by the caller (autodiff, for the PINN). ``geo`` must
    be :func:`line_geometry` and ``f_zeta`` the axial shape at the same points.

    Sharing the flux and boiling closures with the reference solver means the
    network and its ground truth solve the same equations by construction rather
    than by review.
    """
    q_fe = gap_flux(T_f, T_cl, geo, p)
    q_ec = clad_coolant_flux(T_cl, T_c, geo, p, alpha)
    q_sc = struct_coolant_flux(T_s, T_c, geo, p, alpha)
    # Power per METRE. `nodal_power` returns per-node watts (it divides by
    # n_axial) and a node spans dz = H / n_axial, so the per-metre form is simply
    # the total divided by H -- no factor of n_axial anywhere.
    total = amplitude * p.P_0 * f_zeta / p.H
    q_cool, q_fuel = p.gamma_c * total, (1.0 - p.gamma_c) * total

    q_wall = q_cool + q_ec + q_sc
    b = latent_fraction(T_c, alpha, p)
    advection = flow_rate(t, p) * p.c_c * dTc_dz
    u = flow_rate(t, p) / (p.rho_c * p.A_c)
    dT_s = (
        -film_coefficient(p.h_struct_coolant, alpha, p)
        * (T_s - T_c)
        / (p.rho_s * p.c_s * p.t_struct)
    )
    return (
        (q_fuel - q_fe) / geo.C_f,
        (q_fe - q_ec) / geo.C_cl,
        dT_s,
        ((1.0 - b) * q_wall - advection) / geo.C_c,
        vapour_source(T_c, alpha, q_wall, p) - u * dalpha_dz,
    )


def unpack(y: Field, n: int) -> tuple[Any, ...]:
    """Split a flat state vector into ``(T_f, T_cl, T_s, T_c, alpha)``."""
    return tuple(y[k * n : (k + 1) * n] for k in range(N_FIELDS))


def make_rhs(
    p: AxialParams,
    amplitude: Callable[[float], float] | None = None,
) -> Callable[[float, FloatArray], FloatArray]:
    """Build ``f(t, y)`` for :func:`scipy.integrate.solve_ivp`.

    Parameters
    ----------
    p
        Channel configuration.
    amplitude
        Prescribed normalised power ``P(t)/P_0``. Defaults to a constant 1.0 —
        M2 is the *prescribed-power* milestone ("Plan B" of the milestone plan);
        the kinetics closure that makes this an output rather than an input
        arrives at M6.
    """
    geo = node_geometry(p)
    f_nodes = p.power_shape(p.zeta_nodes())
    amp = amplitude if amplitude is not None else (lambda _t: 1.0)
    n = p.n_axial

    def rhs(t: float, y: FloatArray) -> FloatArray:
        T_f, T_cl, T_s, T_c, alpha = unpack(y, n)
        d = derivatives(t, T_f, T_cl, T_s, T_c, alpha, p, geo, f_nodes, amp(t))
        return np.concatenate(d)

    return rhs
