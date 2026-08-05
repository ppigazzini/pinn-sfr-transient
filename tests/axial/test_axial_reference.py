"""M2 acceptance tests for the single-phase axial reference solver.

The reference is the held-out truth for every later milestone, so it has to be
verified against something other than itself. Four independent handles are used,
none of which relies on the solver being right:

1. **An exact steady state.** Dropping the time derivatives leaves a closed
   marching solution, so ``rhs(0, y_steady)`` must vanish to round-off and a
   run started there with constant flow must not move.
2. **Mesh convergence at the order the scheme actually has** — and the two
   regimes differ, which is itself a check (see the convergence tests).
3. **A conservation law the discretisation cannot fake**: total stored energy
   against power in minus power convected out.
4. **Backend parity**: numpy, torch and JAX evaluating the same residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_sfr_transient.axial import AxialParams, sodium
from pinn_sfr_transient.axial.physics import (
    derivatives,
    flow_fraction,
    make_rhs,
    nodal_power,
    node_geometry,
)
from pinn_sfr_transient.axial.reference import (
    energy_balance,
    jacobian_sparsity,
    solve_reference,
    steady_state,
    steady_state_vector,
)


@pytest.fixture(scope="module")
def traj():
    """Default transient, shared across tests (a Radau solve is not free)."""
    return solve_reference(AxialParams(), n_out=241)


# --- 1. the exact steady state ---------------------------------------------
def test_steady_state_annihilates_the_right_hand_side():
    """The strongest single check available: an oracle the solver cannot influence."""
    p = AxialParams()
    residual = make_rhs(p)(0.0, steady_state_vector(p))
    assert np.max(np.abs(residual)) < 1e-8


def test_steady_state_temperature_rise_is_the_telescoped_source():
    """Upwind advection makes the steady coolant profile a telescoping sum -- exactly."""
    p = AxialParams()
    _, _, _, T_c, _ = steady_state(p)
    q_fuel, q_cool = nodal_power(1.0, p.power_shape(p.zeta_nodes()), p)
    expected = (q_fuel + q_cool).sum() / (p.w_0 * p.c_c)
    assert T_c[-1] - p.T_in == pytest.approx(expected, rel=1e-12)


def test_discrete_power_converges_to_the_nominal_rating():
    """The midpoint sum of the axial shape approaches unity as the mesh refines."""
    err = []
    for n in (20, 40, 80):
        p = AxialParams(n_axial=n)
        q_f, q_c = nodal_power(1.0, p.power_shape(p.zeta_nodes()), p)
        err.append(abs((q_f + q_c).sum() - p.P_0))
    assert err[1] < err[0] / 3.0  # midpoint rule is second order
    assert err[2] < err[1] / 3.0


def test_constant_flow_holds_the_steady_state_for_all_time():
    """With `f_nc = 1` there is no coast-down, so nothing may drift."""
    p = AxialParams(f_nc=1.0, t_end=30.0)
    tr = solve_reference(p, n_out=61)
    T_f0, T_cl0, T_s0, T_c0, _ = steady_state(p)
    assert np.max(np.abs(tr.T_c - T_c0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_f - T_f0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_cl - T_cl0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_s - T_s0[:, None])) < 1e-6


def test_structure_sits_at_the_coolant_temperature_in_steady_state():
    """No source in the duct wall, so `q_sc = 0` and it equilibrates."""
    _, _, T_s, T_c, _ = steady_state(AxialParams())
    np.testing.assert_allclose(T_s, T_c, rtol=0.0, atol=0.0)


def test_temperatures_are_ordered_fuel_above_cladding_above_coolant():
    T_f, T_cl, _, T_c, _ = steady_state(AxialParams())
    assert np.all(T_f > T_cl)
    assert np.all(T_cl > T_c)


def test_coolant_heats_monotonically_up_the_channel():
    _, _, _, T_c, _ = steady_state(AxialParams())
    assert np.all(np.diff(T_c) > 0.0)


# --- 2. mesh convergence ---------------------------------------------------
def _outlet(n, n_out=241):
    return solve_reference(AxialParams(n_axial=n), n_out=n_out).T_out


def _richardson(metric):
    """Observed order from three grids, with no fine reference to trust.

    ``||u_h - u_h/2|| / ||u_h/2 - u_h/4|| -> 2^p``. Using successive differences
    rather than an error against a "converged" run removes the assumption that
    the finest grid is itself converged — and costs three solves instead of four.
    """
    a, b, c = (_outlet(n) for n in (20, 40, 80))
    return metric(a - b) / metric(b - c)


def test_transient_converges_at_first_order():
    """Upwind advection is first order, and during the coast-down that dominates.

    Upwind is chosen over a higher-order scheme deliberately: it is monotone, so
    it does not oscillate across the boiling front M4 introduces. First order is
    the price, and this measures that we pay exactly that and no more.
    """
    ratio = _richardson(lambda d: float(np.sqrt(np.mean(d**2))))
    assert 1.6 < ratio < 2.8


def test_quasi_steady_state_converges_at_second_order():
    """At the end of the coast-down the advection error cancels, exposing the source.

    In steady state the upwind stencil telescopes to the *exact* integral of the
    nodal sources and so contributes no truncation error at all; what remains is
    the second-order midpoint quadrature of the axial power shape. Seeing first
    order in the transient and second here is much stronger evidence that both
    pieces behave as designed than one number twice would be.
    """
    ratio = _richardson(lambda d: abs(float(d[-1])))
    assert 3.0 < ratio < 5.5


# --- 3. conservation -------------------------------------------------------
def test_energy_balance_closes(traj):
    """Catches a wrong area, heat capacity or stencil even when the curves look fine."""
    assert energy_balance(traj, AxialParams()) < 1e-4


def test_energy_balance_converges_to_zero_not_to_a_floor():
    """The discretisation conserves energy *exactly*; the residual is only quadrature.

    A floor here would mean a real conservation defect. This caught one during
    M2: the check originally compared against the nominal `P_0` rather than the
    summed nodal sources, which floored it at the midpoint-rule error of the
    axial power shape (~1.6e-4) and hid the true behaviour.
    """
    p = AxialParams()
    err = [energy_balance(solve_reference(p, n_out=n), p) for n in (61, 121, 241)]
    assert err[0] / err[1] > 3.0  # trapezoid quadrature is second order
    assert err[1] / err[2] > 3.0
    assert err[2] < 1e-4


# --- risk R6: the single upstream boundary condition stays valid ------------
def test_flow_never_reverses(traj):
    """Eq. 3.9-1 admits one upstream BC; a sign change would invalidate the model."""
    assert np.all(traj.flow > 0.0)
    assert traj.flow[0] == pytest.approx(AxialParams().w_0)


def test_flow_decays_to_the_natural_circulation_floor():
    p = AxialParams(t_end=200.0)
    assert flow_fraction(np.array([p.t_end]), p)[0] == pytest.approx(p.f_nc, abs=1e-6)


# --- physical behaviour of the transient -----------------------------------
def test_losing_flow_heats_the_coolant(traj):
    assert traj.T_out[-1] > traj.T_out[0] + 100.0


def test_transient_stays_inside_the_sodium_property_range(traj):
    """M4 will call the section 12.13 correlations on these fields."""
    assert bool(sodium.in_range(traj.T_c).all())
    assert bool(sodium.in_range(traj.T_f).all())


def test_boiling_starts_where_and_when_the_criterion_says(traj):
    """M4's headline acceptance: onset time and location.

    The manual's criterion (section 12.4) is ``T_c > T_sat + DTS``. The outlet is
    the hottest point in the channel, so boiling must start at the top, and only
    once the coast-down has driven the coolant there past saturation plus the
    superheat.
    """
    p = AxialParams()
    t0, z0 = traj.onset()
    assert 5.0 < t0 < 30.0
    assert z0 > 0.9  # the top of the channel is the hottest point
    # The criterion is smoothed, so onset happens *within a few smoothing widths*
    # of `T_sat + DTS` rather than exactly at it: the logistic still passes ~0.7%
    # of the wall heat below threshold, and because voiding is explosive that is
    # enough to start it early. The sweep below quantifies the shift.
    T_onset = sodium.saturation_temperature(p.p_system) + p.dT_superheat
    i = int(np.argmin(np.abs(traj.t - t0)))
    assert traj.T_out[i] > T_onset - 6.0 * p.dT_smooth


def test_onset_is_insensitive_to_the_smoothing_width():
    """M4's kill criterion: onset must not drift by more than 2 s as `dT_smooth` varies.

    `dT_smooth` is the one knob with no counterpart in the manual — it exists
    purely so the section 12.4 threshold has a usable autodiff gradient. If the
    answer depended strongly on it, the smoothing would be setting the physics.
    Across a 16x range the onset time moves by ~1.2 s and the location stays
    within one axial cell, so it is not.
    """
    times, places = [], []
    for d in (0.5, 2.0, 8.0):
        t0, z0 = solve_reference(AxialParams(dT_smooth=d), n_out=241).onset()
        times.append(t0)
        places.append(z0)
    assert max(times) - min(times) < 2.0
    # The measured location also depends on the output cadence, since the front
    # advances between samples; at n_out = 241 it is resolved and identical
    # across the sweep. One cell is the physical tolerance.
    assert max(places) - min(places) <= 1.0 / AxialParams().n_axial + 1e-9


def test_no_boiling_when_saturation_is_out_of_reach():
    """Control: raise the system pressure and the void field must stay identically zero."""
    high = AxialParams(p_system=1.6e7)  # T_sat ~ 2280 K, above anything reached
    tr = solve_reference(high, n_out=121)
    assert not bool((tr.alpha > 1e-9).any())
    assert np.isnan(tr.onset()[0])


def test_void_stays_within_bounds(traj):
    """`(1 - alpha)` shuts the source off as a node empties; nothing may escape [0, 1]."""
    assert traj.alpha.min() > -1e-9
    assert traj.alpha.max() < 1.0 + 1e-9


def test_voided_length_grows_and_is_bounded(traj):
    """`L_void` is the metric M4 is judged on -- absolute metres, not a relative L2."""
    L = traj.voided_length
    assert L[0] == pytest.approx(0.0, abs=1e-12)
    assert L[-1] > 0.1
    assert L[-1] <= AxialParams().H + 1e-12
    assert np.all(np.diff(L) > -1e-9)  # the front only advances here


def test_voiding_is_explosive_once_it_starts(traj):
    """Filling a node with vapour takes ~1 J; the wall delivers ~1 kW. Seconds, not minutes.

    This is the physical reason Chapter 12 is a slug-*ejection* model: the vapour
    mass needed to void a channel is negligible, so the front runs away as soon as
    the superheat criterion is met.
    """
    t0, _ = traj.onset()
    i = int(np.argmin(np.abs(traj.t - (t0 + 5.0))))
    assert traj.alpha[:, i].max() > 0.99


def test_post_dryout_response_is_out_of_scope_and_says_so():
    """Documents the M5 gap instead of letting it look like a result.

    Once a node reaches `alpha = 1` the latent sink switches off (correctly --
    there is no liquid left to boil) and the wall heat returns to sensible
    heating. But this model still gives that node *liquid* heat capacity and
    liquid advection, so its post-dryout temperature is not physical. Comparing
    against a run that cannot boil at all shows the difference is ~1 K: after
    dryout the two models agree, which is precisely the symptom. M5's film and
    dryout heat path is what closes this.
    """
    boiling = solve_reference(AxialParams(), n_out=121)
    never = solve_reference(AxialParams(p_system=1.6e7), n_out=121)
    assert abs(float(boiling.T_out[-1] - never.T_out[-1])) < 5.0


def test_direct_coolant_heating_bypasses_the_fuel_thermal_lag():
    """`gamma_c` (Eq. 3.3-6) reaches the coolant instantly; the fuel path cannot."""
    p_on = AxialParams(gamma_c=0.10)
    p_off = AxialParams(gamma_c=0.0)
    geo_on, geo_off = node_geometry(p_on), node_geometry(p_off)
    f_on = p_on.power_shape(p_on.zeta_nodes())
    args_on = steady_state(p_off)  # same start state for both
    d_on = derivatives(0.0, *args_on, p_on, geo_on, f_on, 1.0)
    d_off = derivatives(0.0, *args_on, p_off, geo_off, f_on, 1.0)
    assert np.all(d_on[3] > d_off[3])  # coolant sees it immediately


def test_radiation_term_is_not_negligible():
    """Eq. 3.3-4 carries `eps sigma dT^4`; dropping it shifts the fuel temperature."""
    with_rad = steady_state(AxialParams(emissivity=0.7))[0].max()
    without = steady_state(AxialParams(emissivity=0.0))[0].max()
    assert without > with_rad + 1.0


def test_structure_node_can_be_disabled_without_dividing_by_zero():
    """`gamma_2 = 0` is the documented way to drop the structure (D-GEOM-2)."""
    p = AxialParams(gamma_2=0.0)
    residual = make_rhs(p)(0.0, steady_state_vector(p))
    assert np.all(np.isfinite(residual))
    assert np.max(np.abs(residual)) < 1e-8


def test_structure_slows_the_coolant_heat_up():
    """The duct wall is a heat sink; removing it must make the transient faster."""
    with_s = solve_reference(AxialParams(gamma_2=0.5), n_out=61).T_out
    without = solve_reference(AxialParams(gamma_2=0.0), n_out=61).T_out
    assert without[10] > with_s[10]


# --- solver mechanics ------------------------------------------------------
def test_jacobian_sparsity_covers_every_real_coupling():
    """A missing entry would silently degrade Radau's Newton solve."""
    p = AxialParams(n_axial=6)
    pattern = np.asarray(jacobian_sparsity(p).todense())
    rhs = make_rhs(p)
    y0 = steady_state_vector(p)
    f0 = rhs(0.0, y0)
    for k in range(y0.size):
        y = y0.copy()
        y[k] += 1e-4 * max(1.0, abs(y0[k]))
        touched = np.abs(rhs(0.0, y) - f0) > 0.0
        assert np.all(pattern[touched, k] == 1), f"missing sparsity entry in column {k}"


def test_trajectory_shapes_and_helpers(traj):
    p = AxialParams()
    assert traj.T_c.shape == (p.n_axial, traj.t.size)
    assert traj.zeta.shape == (p.n_axial,)
    np.testing.assert_allclose(traj.T_out, traj.T_c[-1])
    assert traj.peak_clad == pytest.approx(float(traj.T_cl.max()))


# --- 4. backend parity of the residual algebra -----------------------------
def _perturbed_state(p):
    """A state well away from equilibrium.

    Comparing backends *at* the steady state is comparing zeros: the derivatives
    there are ~1e-13 K/s of cancellation noise, so a relative tolerance measures
    rounding rather than agreement. Perturbing puts them at 1e2-1e3 K/s, where
    the comparison means something.
    """
    T_f, T_cl, T_s, T_c, alpha = steady_state(p)
    return T_f + 50.0, T_cl - 20.0, T_s + 30.0, T_c + 10.0, alpha


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_derivatives_match_across_backends(backend):
    """Revised decision D1: the residual algebra carries all three backends.

    The M3 PINN will evaluate this same function on tensors, so a backend
    disagreement here would surface as an unexplained training failure later.
    """
    mod = pytest.importorskip(backend)
    p = AxialParams()
    geo = node_geometry(p)
    f_nodes = p.power_shape(p.zeta_nodes())
    state = _perturbed_state(p)
    ref = derivatives(3.0, *state, p, geo, f_nodes, 1.0)
    assert max(np.max(np.abs(r)) for r in ref) > 1.0  # the comparison is non-trivial

    if backend == "torch":
        conv = lambda a: mod.tensor(a, dtype=mod.float64)  # noqa: E731
        t_arg = mod.tensor(3.0, dtype=mod.float64)
    else:
        mod.config.update("jax_enable_x64", True)
        conv = lambda a: mod.numpy.asarray(a, dtype=mod.numpy.float64)  # noqa: E731
        t_arg = mod.numpy.asarray(3.0, dtype=mod.numpy.float64)

    got = derivatives(t_arg, *(conv(a) for a in state), p, geo, conv(f_nodes), 1.0)
    names = ("T_f", "T_cl", "T_s", "T_c", "alpha")
    for g, r, name in zip(got, ref, names, strict=True):
        np.testing.assert_allclose(np.asarray(g), r, rtol=1e-13, atol=0.0, err_msg=name)
