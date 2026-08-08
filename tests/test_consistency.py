"""Consistency tests (numpy/scipy only -- no torch required).

Validates that:

1. the nominal state is an exact fixed point of the RHS;
2. the reference trajectory satisfies the physical ODE residuals;
3. the normalized residual formulation used in :mod:`pinn_sfr_transient.pinn_torch`
   is algebraically equivalent to the physical ODEs; and
4. the delayed-precursor steady-state scaling is consistent.
"""

import numpy as np

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.physics import flow_fraction, make_rhs, reactivity
from pinn_sfr_transient.reference import solve_reference


def test_steady_state_is_fixed_point() -> None:
    p = SFRParams()
    f = make_rhs(p)(0.0, p.steady_state())
    assert np.allclose(f, 0.0, atol=1e-9), f"steady state not fixed: {f}"


def test_reference_satisfies_physical_ode() -> None:
    p = SFRParams()
    traj = solve_reference(p, n_out=4000)
    dydt = np.gradient(traj.y, traj.t, axis=1)
    f = np.empty_like(traj.y)
    rhs = make_rhs(p)
    for j in range(traj.t.size):
        f[:, j] = rhs(float(traj.t[j]), traj.y[:, j])

    s = slice(5, -5)
    rel_p = np.abs(dydt[0, s] - f[0, s]).max() / (np.abs(f[0]).max() + 1e-30)
    rel_tf = np.abs(dydt[7, s] - f[7, s]).max() / (np.abs(f[7]).max() + 1e-30)
    rel_tc = np.abs(dydt[8, s] - f[8, s]).max() / (np.abs(f[8]).max() + 1e-30)
    assert rel_p < 5e-2
    assert rel_tf < 5e-2
    assert rel_tc < 5e-2


def test_normalized_residual_equivalence() -> None:
    """The normalized residuals (as in pinn_torch) must vanish on the reference."""
    p = SFRParams()
    traj = solve_reference(p, n_out=4000)
    t, y = traj.t, traj.y
    dydt = np.gradient(y, t, axis=1)

    P, Tf, Tc = y[0], y[7], y[8]
    C = y[1:7]
    C_i0 = (p.beta_i / (p.Lambda * p.lambda_i))[:, None]
    c = C / C_i0
    dc = dydt[1:7] / C_i0

    rho = reactivity(Tf, Tc, p)
    g = flow_fraction(t, p)

    R_p = p.Lambda * dydt[0] - ((rho - p.beta_eff) * P + (p.beta_i[:, None] * c).sum(0))
    R_c = dc - p.lambda_i[:, None] * (P[None, :] - c)
    R_tf = dydt[7] - ((p.P0 / p.Cf) * P - (p.UA / p.Cf) * (Tf - Tc))
    R_tc = dydt[8] - ((p.UA / p.Cc) * (Tf - Tc) - (p.W0 * g / p.Cc) * (Tc - p.Tin))

    s = slice(5, -5)
    assert np.abs(R_p[s]).max() < 1e-3
    assert np.abs(R_c[:, s]).max() < 1e-2
    assert np.abs(R_tf[s]).max() < 1e-2
    assert np.abs(R_tc[s]).max() < 1e-2


def test_precursor_scaling() -> None:
    p = SFRParams()
    Ci0 = p.steady_precursors(1.0)
    assert np.allclose(p.beta_i / p.Lambda, p.lambda_i * Ci0, rtol=1e-12)
