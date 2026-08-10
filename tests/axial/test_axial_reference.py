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

import importlib

import numpy as np
import pytest

from pinn_sfr_transient.axial import AxialParams, sodium
from pinn_sfr_transient.axial.physics import (
    N_GROUPS,
    coolant_capacity,
    decay_heat_derivatives,
    derivatives,
    film_coefficient,
    flow_fraction,
    kinetics_weights,
    latent_fraction,
    make_rhs,
    n_decay,
    nodal_power,
    node_geometry,
    prompt_jump_power,
    reactivity,
    reactivity_components,
    residual_normalisation,
    residual_scales,
    total_power,
    vaporisation_time,
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
    T_c = steady_state(p)[3]
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
    T_f0, T_cl0, T_s0, T_c0 = steady_state(p)[:4]
    assert np.max(np.abs(tr.T_c - T_c0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_f - T_f0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_cl - T_cl0[:, None])) < 1e-6
    assert np.max(np.abs(tr.T_s - T_s0[:, None])) < 1e-6


def test_structure_sits_at_the_coolant_temperature_in_steady_state():
    """No source in the duct wall, so `q_sc = 0` and it equilibrates."""
    _, _, T_s, T_c = steady_state(AxialParams())[:4]
    np.testing.assert_allclose(T_s, T_c, rtol=0.0, atol=0.0)


def test_temperatures_are_ordered_fuel_above_cladding_above_coolant():
    T_f, T_cl, _, T_c = steady_state(AxialParams())[:4]
    assert np.all(T_f > T_cl)
    assert np.all(T_cl > T_c)


def test_coolant_heats_monotonically_up_the_channel():
    T_c = steady_state(AxialParams())[3]
    assert np.all(np.diff(T_c) > 0.0)


# --- 2. mesh convergence ---------------------------------------------------
# The convergence studies run the NON-BOILING case (`p_system` raised so
# saturation is out of reach). Two reasons, both necessary: the orders being
# measured are properties of the single-phase discretisation, isolated from the
# boiling nonlinearity; and a boiling run terminates at the validity limit at a
# mesh-dependent time, so the trajectories would not even be comparable.
_BENIGN = {"p_system": 1.6e7}


def _outlet(n, n_out=241):
    return solve_reference(AxialParams(n_axial=n, **_BENIGN), n_out=n_out).T_out


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


def test_energy_balance_uses_the_trajectory_power_not_a_default_of_one():
    """The closed loop delivers ~0.5 of nominal, so a hard-coded amplitude is wrong.

    `energy_balance` used to default `amp = 1`, which mis-stated the Plan A
    closure as 0.382 — a number that reads as a catastrophic conservation defect
    when the discretisation is in fact conserving to 1.5e-5. No test covered the
    feedback trajectory, so it went unnoticed.
    """
    p = AxialParams()
    traj = solve_reference(p, n_out=241, feedback=True)
    assert traj.power.min() < 0.6  # the amplitude really does move
    assert energy_balance(traj, p) < 1e-4
    # Forcing the old behaviour reproduces the bad number, so this is the cause.
    assert energy_balance(traj, p, amplitude=lambda _t: 1.0) > 0.1


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


def test_dryout_collapses_the_heat_path_and_spikes_the_cladding():
    """M5's whole point: losing the liquid removes the heat path, not just the coolant.

    Section 12.5.1 puts the liquid film and the vapour in series in the
    wall-to-coolant resistance. Blending toward the vapour value as a node voids
    is what turns boiling from a temperature *plateau* into a cladding
    *excursion* — the safety-relevant behaviour, and the reason M4 alone was not
    enough. Before M5, a boiling run and one that could not boil agreed to ~1 K.
    """
    boiling = solve_reference(AxialParams(), n_out=121)
    never = solve_reference(AxialParams(p_system=1.6e7), n_out=121)
    assert boiling.peak_clad > never.peak_clad + 300.0


def test_film_coefficient_blends_between_wetted_and_vapour():
    p = AxialParams()
    assert film_coefficient(p.h_clad_coolant, 0.0, p) == pytest.approx(p.h_clad_coolant)
    assert film_coefficient(p.h_clad_coolant, 1.0, p) == pytest.approx(p.h_vapour)
    assert p.h_vapour < p.h_clad_coolant / 100.0  # orders apart, which is the point


def test_run_stops_at_the_validity_limit_rather_than_extrapolating():
    """The model states in its own output where it stops applying.

    Past dryout there is no melting, no cladding motion and no fuel relocation
    here (Chapters 8-16), and the section 12.13 correlations themselves stop at
    2270 K. Integrating on would extrapolate three models at once.
    """
    tr = solve_reference(AxialParams(), n_out=121)
    assert tr.stopped_early
    assert tr.t[-1] < AxialParams().t_end
    assert bool(sodium.in_range(tr.T_c).all())
    assert tr.T_f.max() <= sodium.T_MAX + 1.0


def test_a_benign_run_does_not_stop_early():
    """The termination must trigger on the physics, not on every run."""
    tr = solve_reference(AxialParams(p_system=1.6e7), n_out=61)
    assert not tr.stopped_early
    assert tr.t[-1] == pytest.approx(AxialParams().t_end)


def test_mixture_capacity_is_correct_but_deliberately_unused():
    """It is the right expression; using it in the temperature form breaks conservation.

    Documented in `coolant_capacity`: substituting it into `c dT/dt` degrades the
    energy closure from 3.6e-6 to ~2e-2, because a changing capacity needs an
    enthalpy formulation. M5 therefore degrades the film coefficient — the
    mechanism section 12.5.1 actually describes — and leaves this to a future
    enthalpy-form revision.
    """
    p = AxialParams()
    assert coolant_capacity(0.0, p) == pytest.approx(p.rho_c * p.c_c)
    assert coolant_capacity(1.0, p) < coolant_capacity(0.0, p) / 100.0


def test_direct_coolant_heating_bypasses_the_fuel_thermal_lag():
    """`gamma_c` (Eq. 3.3-6) reaches the coolant instantly; the fuel path cannot."""
    p_on = AxialParams(gamma_c=0.10)
    p_off = AxialParams(gamma_c=0.0)
    geo_on, geo_off = node_geometry(p_on), node_geometry(p_off)
    f_on = p_on.power_shape(p_on.zeta_nodes())
    args_on = steady_state(p_off)[:5]  # same start state for both
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
    T_f, T_cl, T_s, T_c, alpha = steady_state(p)[:5]
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


# --- M6: the prompt-jump kinetics closure ----------------------------------
@pytest.fixture(scope="module")
def plan_a():
    """Closed-loop run: power is an output, not an input."""
    return solve_reference(AxialParams(), n_out=241, feedback=True)


def test_nominal_state_is_exactly_critical():
    """No criticality offset is needed — a free consequence of the log Doppler.

    At nominal `T_f = T_f0` so `ln(T_f/T_f0) = 0`, and `alpha = 0`, so both
    reactivity integrals vanish identically. With `c_i = 1` the prompt-jump
    closure then gives `P = sum(beta_i)/beta = 1` exactly. The 0D model has to
    absorb a residual into `rho_ext` to achieve the same thing.
    """
    p = AxialParams()
    T_f0, _, _, _, alpha, c = steady_state(p)
    w_D, w_void = kinetics_weights(p)
    rho = reactivity(T_f0, alpha, T_f0, w_D, w_void, p)
    assert abs(float(rho)) < 1e-15
    assert float(prompt_jump_power(c, rho, p)) == pytest.approx(1.0, rel=1e-15)


def test_closed_loop_starts_at_nominal_power(plan_a):
    assert plan_a.power[0] == pytest.approx(1.0, rel=1e-12)


def test_feedback_is_self_limiting(plan_a):
    """Both feedbacks are negative here, so power can only fall."""
    assert plan_a.power.max() == pytest.approx(1.0, rel=1e-9)
    assert plan_a.power[-1] < 0.8
    assert plan_a.rho.max() <= 1e-12


def test_pole_tripwire_is_reported_and_far_from_one(plan_a):
    """M6's kill criterion: `rho/beta -> 1` means the prompt jump is invalid.

    The closure has a pole at prompt criticality. Every run reports
    `max_t rho/beta`; the milestone bar is < 0.5.
    """
    assert plan_a.peak_rho_over_beta < 0.5
    assert plan_a.rho.min() / AxialParams().beta_eff > -50.0  # sane, not runaway


def test_power_decay_respects_the_eighty_second_floor(plan_a):
    """The stopwatch test: no delayed-neutron decay can be faster than `1/lambda_1`.

    Under the prompt-jump closure the tail time constant is bounded below by
    `1/lambda_1 = 80.6 s` at *any* reactivity — as the reactivity goes to minus
    infinity the dominant inhour root only approaches `-lambda_1`. A faster tail
    would mean the closure had been violated or bypassed, not that the physics
    was severe.
    """
    p = AxialParams()
    late = plan_a.t > 20.0
    tau = -1.0 / np.polyfit(plan_a.t[late], np.log(plan_a.power[late]), 1)[0]
    assert tau >= 1.0 / p.lambda_i.min()


def test_feedback_delays_boiling_and_lowers_the_cladding_peak(plan_a):
    """Closing the loop is stabilising: less power, later onset, cooler cladding."""
    plan_b = solve_reference(AxialParams(), n_out=241)
    assert plan_a.onset()[0] > plan_b.onset()[0] + 2.0
    assert plan_a.peak_clad < plan_b.peak_clad - 100.0
    assert not plan_a.stopped_early  # and it no longer runs out of validity
    assert plan_b.stopped_early


def test_doppler_dominates_the_reactivity_balance(plan_a):
    """Reconstruct rho from the fields and check it matches what the solver used."""
    p = AxialParams()
    T_f0 = steady_state(p)[0]
    w_D, w_void = kinetics_weights(p)
    i = len(plan_a.t) // 2
    rebuilt = reactivity(plan_a.T_f[:, i], plan_a.alpha[:, i], T_f0, w_D, w_void, p)
    assert float(rebuilt) == pytest.approx(plan_a.rho[i], rel=1e-12)


def test_void_block_is_normalised_by_transport_not_by_its_source():
    """The void's *dynamical* rate is advection; its source is 160x faster but local.

    Normalising the residual by the vaporisation time made a spurious void in the
    subcooled bulk cost 3.9e-5 against 1.0 at the front, and the network duly
    voided the whole channel from t = 0. The block must be scaled by the rate
    that governs it everywhere — the same transit time as the coolant.
    """
    p = AxialParams()
    tau = residual_scales(p)
    assert tau[4] == pytest.approx(tau[3], rel=1e-12)  # advected with the liquid


def test_vaporisation_time_is_reported_but_never_used_to_normalise():
    """The 160x source stiffness is real and diagnostic; it is not the scaling."""
    p = AxialParams()
    T_sat = sodium.saturation_temperature(p.p_system)
    expected = sodium.latent_heat(T_sat) * sodium.vapor_density(T_sat) * p.A_c * p.H / p.P_0
    assert vaporisation_time(p) == pytest.approx(expected, rel=1e-12)
    assert vaporisation_time(p) < 1e-3
    # 160x faster than the transport, and NOT what the normalisation uses.
    assert vaporisation_time(p) < residual_scales(p)[4] / 100.0
    assert residual_normalisation(p)[4] != pytest.approx(vaporisation_time(p) / p.t_end)


def test_reactivity_components_sum_to_the_net(plan_a):
    """The reported split is the same quantity the solver integrated, not a re-derivation."""
    p = AxialParams()
    np.testing.assert_allclose(
        p.rho_ext + plan_a.rho_doppler + plan_a.rho_void, plan_a.rho, rtol=1e-14
    )


def test_void_feedback_is_never_positive_at_the_shipped_defaults(plan_a):
    """`max rho/beta = 0` is an artefact of *where* the channel boils, not stability.

    Boiling starts at the top of the channel because that is where the coolant is
    hottest, and the void field can only advect upward. With `zeta_sign = 0.80`
    the voided region therefore never reaches the positive-worth part of the
    core, so the coolant/void term is negative for the whole transient and the
    positive sodium-void feedback this project exists to study is never sampled.
    Pinned here so it is a stated property rather than an unnoticed one; raising
    `zeta_sign` above the onset location is what would exercise it.
    """
    p = AxialParams()
    assert not plan_a.void_worth_is_exercised()
    assert plan_a.rho_void.max() / p.beta_eff < 1e-6  # roundoff only, never physical
    assert plan_a.rho_void.min() / p.beta_eff < -1e-3  # the term is active, just negative
    voided = plan_a.alpha.max(axis=1) > 1e-2
    assert plan_a.zeta[voided].min() > p.zeta_sign


def test_prompt_jump_pole_is_clamped_not_crossed():
    """A guard, so a bad parameter set cannot silently produce nonsense."""
    p = AxialParams()
    c = np.ones(6)
    huge = prompt_jump_power(c, np.array(0.999 * p.beta_eff), p)
    assert np.isfinite(huge)
    assert huge <= 1.0 / 0.05 + 1e-9  # the floor caps the amplification


def test_sparsity_covers_the_feedback_couplings():
    """The gap that let a broken pattern through: the old test never ran Plan A.

    With feedback the reactivity is an integral over the channel, so every row
    depends on every `T_f` and every `alpha`. A `lil_matrix` slice assignment
    silently dropped 312 entries and Radau then could not advance past 0.5 s —
    which looks like a stiff-solver problem, not a bug.
    """
    p = AxialParams(n_axial=6)
    y0 = steady_state_vector(p)
    rhs = make_rhs(p, None, steady_state(p)[0])
    f0 = rhs(1.0, y0)
    pattern = np.asarray(jacobian_sparsity(p, feedback=True).todense())
    for k in range(y0.size):
        y = y0.copy()
        y[k] += 1e-4 * max(1.0, abs(y0[k]))
        touched = np.abs(rhs(1.0, y) - f0) > 0.0
        assert np.all(pattern[touched, k] == 1), f"missing sparsity entry in column {k}"


# --- condensation (manual section 12.5, experimental) -----------------------
def test_condensation_is_off_by_default_so_the_vapour_source_cannot_be_negative():
    """At `condensation = 0` the void is monotone along a characteristic.

    That monotonicity is what makes `alpha` slaved to `T_c`, which is the premise
    of the algebraic closure D-TH-3.
    """
    p = AxialParams()
    assert p.condensation == 0.0
    T_c = np.linspace(600.0, 1400.0, 41)
    alpha = np.linspace(0.0, 0.99, 41)
    assert bool((latent_fraction(T_c, alpha, p) >= 0.0).all())


def test_condensation_makes_the_phase_change_signed():
    """Vapour in a region that is no longer superheated must be able to shrink."""
    p = AxialParams(condensation=1.0)
    subcooled, voided = np.array([700.0]), np.array([0.8])
    assert latent_fraction(subcooled, voided, p).item() < 0.0
    # ... and it still vanishes as the vapour runs out, so alpha cannot go negative.
    assert latent_fraction(subcooled, np.array([0.0]), p).item() == 0.0


def test_condensation_preserves_the_energy_balance():
    """The signed fraction is removed from the sensible term and no more."""
    p = AxialParams(condensation=1.0)
    assert energy_balance(solve_reference(p, n_out=241), p) < 1e-4


def test_condensation_is_inert_in_this_scenario():
    """Correct physics, no effect here — and the reason is measurable.

    Condensation needs the film heat flow to reverse (the manual: `Q_e`, `Q_s`
    negative). Vapour only ever exists where dryout has driven the cladding
    *hotter* than the coolant, so the cladding branch never reverses and the net
    wall heat stays positive. Voided length moves by 0.3%.
    """
    base = solve_reference(AxialParams(n_axial=80), n_out=161)
    cond = solve_reference(AxialParams(n_axial=80, condensation=1.0), n_out=161)
    assert abs(cond.voided_length.max() - base.voided_length.max()) < 0.02
    assert cond.onset()[0] == pytest.approx(base.onset()[0], abs=0.5)


# --- Annex A, N3: the void worth can be exercised ----------------------------
def test_default_parameters_never_sample_the_void_worth_positive():
    """D49: `max rho/beta = 0` is about where the channel boils, not stability."""
    traj = solve_reference(AxialParams(n_axial=80), n_out=161, feedback=True)
    assert not traj.void_worth_is_exercised()
    assert traj.rho_void.min() < 0.0


def test_the_alternative_set_does_sample_it_positive():
    """Objective 2 needs a set in which the positive branch exists at all."""
    p = AxialParams.with_positive_void_worth(n_axial=80)
    assert p.zeta_sign > 0.96  # above the onset location
    traj = solve_reference(p, n_out=161, feedback=True)
    assert traj.void_worth_is_exercised()
    assert traj.rho_void.max() / p.beta_eff > 0.01


# --- Annex A, N4: decay heat -------------------------------------------------
def test_decay_heat_is_off_by_default():
    assert AxialParams().decay_fraction == 0.0
    assert n_decay(AxialParams()) == 0


def test_decay_heat_adds_states_and_keeps_the_steady_state_exact():
    """`psi_t = psi_f + psi_h` must still give exactly 1 at nominal."""
    p = AxialParams(n_axial=40, decay_fraction=0.065)
    assert n_decay(p) == 3
    y0 = steady_state_vector(p)
    assert y0.size == 5 * p.n_axial + N_GROUPS + 3
    assert np.max(np.abs(make_rhs(p)(0.0, y0))) < 1e-8
    assert p.steady_decay_heat(1.0).sum() == pytest.approx(0.065, rel=1e-12)


def test_decay_heat_removes_the_zero_power_attractor():
    """With `psi_f = 0` the total power is still positive — the point of §4.4.

    Without it, `P = c = 0` is an exact solution of the whole coupled system,
    which is the collapse mode REPORT-01 §5.2 exists to diagnose.
    """
    p = AxialParams(decay_fraction=0.065)
    h = p.steady_decay_heat(1.0)
    assert total_power(0.0, h, p) == pytest.approx(0.065, rel=1e-12)
    assert total_power(1.0, h, p) == pytest.approx(1.0, rel=1e-12)


def test_decay_heat_groups_relax_toward_their_share_of_fission_power():
    p = AxialParams(decay_fraction=0.065)
    h_eq = p.steady_decay_heat(1.0)
    np.testing.assert_allclose(decay_heat_derivatives(h_eq, 1.0, p), 0.0, atol=1e-18)
    assert bool((decay_heat_derivatives(np.zeros(3), 1.0, p) > 0).all())


# --- Annex A, N7: axial expansion feedback -----------------------------------
def test_axial_expansion_is_off_by_default_and_must_be_stabilising():
    """D-FB-3: omitting it over-predicts the excursion, so its sign is load-bearing."""
    assert AxialParams().alpha_expansion == 0.0
    with pytest.raises(ValueError, match="alpha_expansion must be <= 0"):
        AxialParams(alpha_expansion=1e-6)


def test_axial_expansion_leaves_the_nominal_state_exactly_critical():
    """Linear in `T_f - T_f0`, so it vanishes at nominal and needs no offset."""
    p = AxialParams(n_axial=40, alpha_expansion=-1e-6)
    T_f0 = steady_state(p)[0]
    w_D, w_void = kinetics_weights(p)
    doppler, void = reactivity_components(T_f0, np.zeros_like(T_f0), T_f0, w_D, w_void, p)
    assert abs(float(doppler)) < 1e-15
    assert abs(float(void)) < 1e-15


def test_axial_expansion_adds_negative_reactivity_as_the_fuel_heats():
    off = AxialParams(n_axial=40)
    on = AxialParams(n_axial=40, alpha_expansion=-1e-6)
    T_f0 = steady_state(off)[0]
    w_D, w_void = kinetics_weights(off)
    zero = np.zeros_like(T_f0)
    d_off = float(reactivity_components(T_f0 * 1.3, zero, T_f0, w_D, w_void, off)[0])
    d_on = float(reactivity_components(T_f0 * 1.3, zero, T_f0, w_D, w_void, on)[0])
    assert d_on < d_off


def _measured_validity_frac() -> float:
    """Return the fraction of `t_end` over which the model is actually valid.

    The reference terminates when any temperature reaches the top of the section 12.13
    sodium property fits (D-SCOPE-1), so where it stops *is* the validity horizon.
    """
    p = AxialParams(n_axial=40)
    traj = solve_reference(p, n_out=241)
    assert traj.t[-1] < p.t_end, "the run must stop early, or there is no validity limit to track"
    return float(traj.t[-1] / p.t_end)


def test_the_reference_stops_before_its_property_fits_run_out():
    """There must be a validity limit to track, on every install.

    Split out from the backend check below so it runs in the core-only lane too: it is
    a statement about the physics and needs no network.
    """
    assert 0.0 < _measured_validity_frac() < 1.0


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_the_training_horizon_matches_the_models_validity_range(backend):
    """`t_train_frac` must track where the model actually stops being valid.

    Training past that asks the network to satisfy residuals where the model does not
    apply, and because the ansatz is one smooth function of `t_hat` that state
    propagates back to `t = 0`.

    This was a defect, not a hypothetical: the default was 1.0, which trains over 72% of
    an invalid horizon and forms no boiling front at all, while every published table was
    measured at 0.275 with the value recorded nowhere. Tie the default to the measurement
    so a change in the physics cannot silently invalidate it.

    **Both backends, and via `importorskip`.** The earlier version imported the torch
    config unconditionally and checked that backend alone. So it errored rather than
    skipped wherever torch is absent — which broke the two CI lanes that exist precisely
    to prove each backend stands up without the other — and it never checked that the JAX
    default agrees, which is the divergence the two-backend rule exists to catch.
    """
    pytest.importorskip(backend)
    mod = f"pinn_sfr_transient.axial.{backend}pinn.config"
    cfg = importlib.import_module(mod).AxialTrainConfig()

    validity_frac = _measured_validity_frac()
    assert cfg.t_train_frac == pytest.approx(validity_frac, abs=0.02), (
        f"{backend} default t_train_frac {cfg.t_train_frac} does not match the measured "
        f"validity horizon {validity_frac:.4f}"
    )
