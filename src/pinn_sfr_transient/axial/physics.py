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

from pinn_sfr_transient.axial._backend import xp as _xp

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.config import FloatArray

STEFAN_BOLTZMANN: float = 5.670374419e-8
"""Stefan-Boltzmann constant [W/m^2-K^4], radiation term of Eq. 3.3-4."""

N_FIELDS: int = 4
"""Temperature fields per axial node: fuel, cladding, structure, coolant."""


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


# --- flow ------------------------------------------------------------------
def flow_fraction(t: Any, p: AxialParams) -> Any:  # noqa: ANN401 - backend-agnostic
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


def flow_rate(t: Any, p: AxialParams) -> Any:  # noqa: ANN401 - backend-agnostic
    """Coolant mass flow rate ``w(t)`` [kg/s], ``z``-independent per Eq. 3.9-1."""
    return p.w_0 * flow_fraction(t, p)


# --- heat transfer paths ---------------------------------------------------
def gap_flux(T_f: Any, T_cl: Any, geo: NodeGeometry, p: AxialParams) -> Any:  # noqa: ANN401
    """Fuel-to-cladding heat flow per node [W] — Eq. 3.3-4.

    ``q_fe = h_b (T_f - T_cl) + eps sigma (T_f^4 - T_cl^4)``. The radiation term
    is not decorative: above 1000 K it carries a meaningful share, and it is
    *stabilising*, growing faster than linearly as the fuel heats.
    """
    conduction = p.h_gap * (T_f - T_cl)
    radiation = p.emissivity * STEFAN_BOLTZMANN * (T_f**4 - T_cl**4)
    return geo.A_fe * (conduction + radiation)


def clad_coolant_flux(T_cl: Any, T_c: Any, geo: NodeGeometry, p: AxialParams) -> Any:  # noqa: ANN401
    """Cladding-to-coolant heat flow per node [W] — the ``Q_ec`` term of Eq. 3.3-5."""
    return p.h_clad_coolant * geo.A_ec * (T_cl - T_c)


def struct_coolant_flux(T_s: Any, T_c: Any, geo: NodeGeometry, p: AxialParams) -> Any:  # noqa: ANN401
    """Structure-to-coolant heat flow per node [W] — the ``Q_sc`` term of Eq. 3.3-5.

    Vanishes when ``gamma_2 = 0``, i.e. when the structure node is disabled.
    """
    return p.h_struct_coolant * geo.A_sc * (T_s - T_c)


def nodal_power(amplitude: Any, f_nodes: Any, p: AxialParams) -> tuple[Any, Any]:  # noqa: ANN401
    """Split the nodal power into its fuel and direct-coolant parts [W].

    Returns ``(Q_fuel, Q_coolant)``. The manual deposits a fraction ``gamma_c``
    of total power straight into the coolant by neutron and gamma heating
    (Eq. 3.3-6); the remainder is generated in the fuel. ``f_nodes`` is the
    normalised axial shape sampled at the node centres, so
    ``sum(f_nodes) / n_axial == 1``.
    """
    total = amplitude * p.P_0 * f_nodes / p.n_axial
    return (1.0 - p.gamma_c) * total, p.gamma_c * total


# --- the coupled right-hand side -------------------------------------------
def derivatives(  # noqa: PLR0913 - four coupled fields plus the driving terms
    t: Any,  # noqa: ANN401
    T_f: Any,  # noqa: ANN401
    T_cl: Any,  # noqa: ANN401
    T_s: Any,  # noqa: ANN401
    T_c: Any,  # noqa: ANN401
    p: AxialParams,
    geo: NodeGeometry,
    f_nodes: Any,  # noqa: ANN401
    amplitude: Any = 1.0,  # noqa: ANN401
) -> tuple[Any, Any, Any, Any]:
    """Time derivatives of the four temperature fields [K/s].

    Backend-agnostic: every operation is arithmetic plus ``exp`` and
    ``concatenate``, all of which numpy, torch and JAX provide under the same
    name, so the reference solver and the M3 PINN residual share one expression
    tree.

    The coolant advection term is the conservative flux form of Eq. 3.3-5,
    ``d(w c T)/dz``, discretised **first-order upwind**. Upwind rather than a
    higher order deliberately: it is monotone, so it will not oscillate across
    the void front that M4 introduces. The price is first-order mesh
    convergence, which ``tests/axial/test_axial_reference.py`` measures.
    """
    xp = _xp(T_c)
    q_fe = gap_flux(T_f, T_cl, geo, p)
    q_ec = clad_coolant_flux(T_cl, T_c, geo, p)
    q_sc = struct_coolant_flux(T_s, T_c, geo, p)
    q_fuel, q_cool = nodal_power(amplitude, f_nodes, p)

    # Upwind neighbour: the inlet feeds node 0, node j-1 feeds node j. Building
    # the inlet element as `T_c[:1] * 0 + T_in` keeps dtype, device and backend
    # without a literal-array constructor that differs between frameworks.
    inlet = T_c[:1] * 0.0 + p.T_in
    T_up = xp.concatenate([inlet, T_c[:-1]])
    advection = flow_rate(t, p) * p.c_c * (T_up - T_c)

    # The structure derivative is written with the gamma_2 factor CANCELLED:
    # q_sc carries gamma_2 through A_sc, and C_s carries it through the structure
    # volume, so the ratio is gamma_2-free. Dividing them as written would be
    # 0/0 at gamma_2 = 0 — which deviation D-GEOM-2 documents as the supported
    # way to disable the structure node. How fast a given piece of duct responds
    # cannot depend on how much of it the coolant sees; only the coolant's q_sc
    # does, and that keeps its gamma_2.
    dT_s = -p.h_struct_coolant * (T_s - T_c) / (p.rho_s * p.c_s * p.t_struct)

    return (
        (q_fuel - q_fe) / geo.C_f,
        (q_fe - q_ec) / geo.C_cl,
        dT_s,
        (advection + q_cool + q_ec + q_sc) / geo.C_c,
    )


def unpack(y: Any, n: int) -> tuple[Any, Any, Any, Any]:  # noqa: ANN401 - backend-agnostic
    """Split a flat state vector into ``(T_f, T_cl, T_s, T_c)``."""
    return y[:n], y[n : 2 * n], y[2 * n : 3 * n], y[3 * n :]


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
        T_f, T_cl, T_s, T_c = unpack(y, n)
        d = derivatives(t, T_f, T_cl, T_s, T_c, p, geo, f_nodes, amp(t))
        return np.concatenate(d)

    return rhs
