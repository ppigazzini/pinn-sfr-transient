"""M3 acceptance tests for the axial PINN.

Deliberately *tiny* networks and iteration counts: the suite has to stay fast, so
these check the things that must hold for **any** weights — the hard constraints,
the residual identity, backend agreement — rather than accuracy, which is a
property of a converged run and is recorded in ``docs/axial_physics.md``.

The load-bearing test is :func:`test_residual_is_the_shared_physics_to_machine_precision`.
It is this model's answer to ``tests/test_consistency.py``: it proves the
network's normalised residual, un-normalised, *is* the same equation the M2
reference solves, so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_sfr_transient.axial import AxialParams
from pinn_sfr_transient.axial.physics import (
    continuous_derivatives,
    line_geometry,
    quasi_steady_void,
    residual_normalisation,
    residual_scales,
)
from pinn_sfr_transient.axial.pinn_torch import (
    FIELDS,
    AxialPinn,
    AxialTrainConfig,
    Trainer,
    _bounded_weights,
    _precursors,
    relative_l2,
)
from pinn_sfr_transient.axial.reference import solve_reference, steady_profile

TINY = AxialTrainConfig(width=8, depth=2, n_colloc=64, adam_iters=3, lbfgs_iters=2, log_every=100)


@pytest.fixture
def model():
    return AxialPinn(AxialParams(), TINY)


# --- hard constraints: must hold for ANY weights ---------------------------
def test_initial_condition_is_exact(model):
    """`theta = theta_0 + t_hat N` — the IC cannot be violated, untrained or not.

    Asserted against the model's *own* ``theta0``, which is the actual claim: the
    ``t_hat`` gate is identically zero at ``t = 0``, so the network contributes
    nothing there for any weights. Agreement between the torch and numpy steady
    profiles is a separate question, tested below — the two run independent Newton
    solves for the fuel temperature and differ in the last bits.
    """
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 33)
    got = model.predict(zeta, np.array([0.0]))
    zt = torch.tensor(zeta.reshape(-1, 1), dtype=torch.float64)
    theta0 = model.theta0(zt).detach().numpy()
    for k in range(4):
        expected = p.T_in + theta0[:, k] * model.dT
        np.testing.assert_allclose(got[k][:, 0], expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got[4][:, 0], 0.0, rtol=0.0, atol=0.0)  # void-free


def test_initial_condition_matches_the_reference_steady_state(model):
    """And that hard IC is the same state the reference solver starts from."""
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 33)
    got = model.predict(zeta, np.array([0.0]))
    names = ("T_f", "T_cl", "T_s", "T_c", "alpha")
    for f, ref, name in zip(got, steady_profile(p, zeta)[:5], names, strict=True):
        np.testing.assert_allclose(f[:, 0], ref, atol=1e-9, err_msg=name)


def test_inlet_boundary_condition_is_exact(model):
    """Eq. 3.9-1 admits one upstream condition; the `zeta` factor pins it identically."""
    p = AxialParams()
    T_c = model.predict(np.array([0.0]), np.linspace(0.0, p.t_end, 21))[3]
    np.testing.assert_allclose(T_c, p.T_in, rtol=0.0, atol=0.0)


def test_hard_constraints_survive_training(model):
    """Training moves the weights; it must not be able to move the constraints."""
    p = AxialParams()
    Trainer(model, TINY).train(verbose=False)
    zeta = np.linspace(0.0, 1.0, 17)
    np.testing.assert_allclose(
        model.predict(zeta, np.array([0.0]))[3], steady_profile(p, zeta)[3][:, None], atol=1e-12
    )
    np.testing.assert_allclose(
        model.predict(np.array([0.0]), np.linspace(0.0, p.t_end, 9))[3], p.T_in, atol=1e-12
    )


def test_torch_steady_profile_matches_the_numpy_one(model):
    """Two implementations of the hard IC exist because `jvp` cannot trace numpy."""
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 101)
    theta = model.theta0(torch.tensor(zeta.reshape(-1, 1), dtype=torch.float64)).numpy()
    ref = steady_profile(p, zeta)[:5]
    for k, name in enumerate(("T_f", "T_cl", "T_s", "T_c")):
        got = p.T_in + theta[:, k] * model.dT
        np.testing.assert_allclose(got, ref[k], atol=1e-9, err_msg=name)
    np.testing.assert_allclose(theta[:, 4], 0.0)  # void-free at nominal


# --- the consistency test --------------------------------------------------
def test_residual_is_the_shared_physics_to_machine_precision(model):
    """The normalised residual, un-normalised, equals the reference's own equations.

    This is the axial counterpart of `tests/test_consistency.py`. It rebuilds the
    residual by hand from `continuous_derivatives` — the function the M2 solver
    discretises — and asserts the network's internal one matches. If these ever
    diverge, the PINN and its ground truth are solving different problems, and no
    accuracy number would mean anything.

    Variable scaling multiplies block `k` by `res_norm[k]` before squaring, so
    the identity to check is that dividing it back out recovers the physics
    exactly. That is the point of the scaling: it changes the *loss*, never the
    equation. This test caught the scaling landing on the residual before the
    scaling was checked against the physics, which is what it exists for.
    """
    p = AxialParams()
    gen = torch.Generator().manual_seed(0)
    zeta = torch.rand(128, 1, dtype=torch.float64, generator=gen)
    that = torch.rand(128, 1, dtype=torch.float64, generator=gen)

    blocks = model.residual_blocks(zeta, that)

    theta, d_dt, d_dz = model.state_and_grads(zeta, that)
    rhs = continuous_derivatives(
        that * model.t_end,
        *model.to_physical(theta),
        d_dz[:, 3:4] * model.dT / p.H,
        d_dz[:, 4:5] / p.H,
        p,
        line_geometry(p),
        _shape(p, zeta),
        1.0,
    )
    scales = [model.t_end / model.dT] * 4 + [model.t_end]
    assert len(blocks) == model.n_blocks
    for k in range(model.n_blocks):
        physical = (d_dt[:, k : k + 1] - scales[k] * rhs[k]).pow(2).squeeze(1)
        # Un-scale the network's block and it must be the physical residual.
        unscaled = blocks[k] / model.res_norm[k] ** 2
        np.testing.assert_allclose(
            unscaled.detach().numpy(), physical.detach().numpy(), rtol=1e-14, atol=0.0
        )


def test_variable_scaling_changes_the_loss_but_never_the_equation():
    """Scaling is a reweighting: the residual's *zero set* must be untouched.

    Two models with identical weights, one scaled and one not, must have blocks
    that differ by exactly `res_norm ** 2` — so a state that annihilates one
    annihilates the other. Anything else would mean the scaling had altered the
    physics rather than the optimisation.
    """
    p = AxialParams()
    on = AxialPinn(p, AxialTrainConfig(width=8, depth=2, residual_scaling=True))
    off = AxialPinn(p, AxialTrainConfig(width=8, depth=2, residual_scaling=False))
    off.load_state_dict(on.state_dict())  # same weights, different scaling
    gen = torch.Generator().manual_seed(3)
    zeta = torch.rand(64, 1, dtype=torch.float64, generator=gen)
    that = torch.rand(64, 1, dtype=torch.float64, generator=gen)
    b_on, b_off = on.residual_blocks(zeta, that), off.residual_blocks(zeta, that)
    for k in range(on.n_blocks):
        np.testing.assert_allclose(
            b_on[k].detach().numpy(),
            (b_off[k] * on.res_norm[k] ** 2).detach().numpy(),
            rtol=1e-14,
        )
    np.testing.assert_allclose(off.res_norm[: off.n_blocks], np.ones(off.n_blocks))


def test_variable_scaling_brings_every_block_to_the_same_order():
    """23x of spread in the blocks' natural rates becomes O(1) each.

    `residual_normalisation` is `tau_k / t_end`, and a block's normalised
    derivative is of order `t_end / tau_k`, so the product is order one.

    The spread is 23x and not the 813x an earlier version of this test asserted:
    that number came from normalising the void by its *source* rate, which is
    160x faster than its transport and active on under 4% of the domain. Doing
    so left a spurious void costing 3.9e-5 against 1.0 at the front, and the
    network voided the whole channel — see `vaporisation_time`.
    """
    p = AxialParams()
    tau = residual_scales(p)
    nrm = residual_normalisation(p)
    rates = [p.t_end / t for t in tau]
    assert 20.0 < max(rates) / min(rates) < 30.0
    scaled = [r * n for r, n in zip(rates, nrm, strict=True)]
    np.testing.assert_allclose(scaled, np.ones(5), rtol=1e-12)


def test_a_shorter_training_horizon_rescales_normalised_time_consistently():
    """`t_train_frac` moves `t_hat = 1`, so every scale that references it must move.

    Training over the model's validity window rather than the full `t_end` is a
    scope decision (72% of the 60 s horizon lies past the sodium property range
    under prescribed power). It only works if the residual normalisation follows
    the trained horizon rather than the parameter.
    """
    p = AxialParams()
    frac = 0.275
    m = AxialPinn(p, AxialTrainConfig(width=8, depth=2, t_train_frac=frac))
    assert m.t_end == pytest.approx(p.t_end * frac)
    # res_norm carries one extra entry for the interface block; compare the fields.
    np.testing.assert_allclose(
        m.res_norm[: len(FIELDS)], residual_normalisation(p, m.t_end), rtol=1e-14
    )
    # t = 0 is still exactly the steady state, and the inlet is still pinned.
    zeta = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(
        m.predict(zeta, np.array([0.0]))[3], steady_profile(p, zeta)[3][:, None], atol=1e-9
    )


def _shape(p, zeta):
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (np.pi * k)) * np.sin(0.5 * np.pi * k)
    return torch.cos(np.pi * k * (zeta - 0.5)) / norm


def test_continuous_steady_profile_annihilates_the_pde_residual():
    """The strongest physics check available, and it needs no network at all.

    The continuous steady profile is an *exact* solution of the steady PDE:
    ``w c dT_c/dz = P_0 f(zeta) / H`` holds identically because
    ``T_c = T_in + (P_0 / (w_0 c_c)) F(zeta)`` and ``F' = f``. So feeding it to
    `continuous_derivatives` with the analytic gradient must return zero to
    round-off. A sign error, a missing source or a wrong scale factor anywhere in
    the PDE form fails here immediately.
    """
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 257)
    T_f, T_cl, T_s, T_c, alpha = steady_profile(p, zeta)[:5]
    dT = p.P_0 / (p.w_0 * p.c_c)
    dTc_dz = dT * p.power_shape(zeta) / p.H  # analytic: F' = f
    zeros = np.zeros_like(zeta)
    rhs = continuous_derivatives(
        0.0,
        T_f,
        T_cl,
        T_s,
        T_c,
        alpha,
        dTc_dz,
        zeros,
        p,
        line_geometry(p),
        p.power_shape(zeta),
        1.0,
    )
    for k, name in enumerate(("T_f", "T_cl", "T_s", "T_c", "alpha")):
        assert np.max(np.abs(rhs[k])) < 1e-9, f"{name}: {np.max(np.abs(rhs[k])):.3e} K/s"


def test_reference_solution_approaches_the_pde_as_the_mesh_refines():
    """The discrete solver and the continuous PDE are the same equation in the limit.

    Measured at a fixed interior point during the transient. First-order upwind
    means the gap closes like `dz`, so this asserts convergence rather than a
    tolerance — the absolute residual at any single mesh is dominated by
    discretisation error, not by a modelling mistake.
    """
    errs = []
    for n in (40, 80, 160):
        # Non-boiling: the statement is about the discretisation, and a boiling
        # run stops at the validity limit at a mesh-dependent time.
        p = AxialParams(n_axial=n, p_system=1.6e7)
        traj = solve_reference(p, n_out=121)
        j, i = n // 2, 60
        dz, dt = p.H / n, traj.t[1] - traj.t[0]
        dTc_dz = (traj.T_c[j + 1, i] - traj.T_c[j - 1, i]) / (2.0 * dz)
        dalpha_dz = (traj.alpha[j + 1, i] - traj.alpha[j - 1, i]) / (2.0 * dz)
        rhs = continuous_derivatives(
            traj.t[i],
            traj.T_f[j, i],
            traj.T_cl[j, i],
            traj.T_s[j, i],
            traj.T_c[j, i],
            traj.alpha[j, i],
            dTc_dz,
            dalpha_dz,
            p,
            line_geometry(p),
            float(p.power_shape(traj.zeta[j])),
            1.0,
        )
        measured = (traj.T_c[j, i + 1] - traj.T_c[j, i - 1]) / (2.0 * dt)
        errs.append(abs(float(measured - rhs[3])))
    assert errs[2] < errs[0]


# --- training mechanics ----------------------------------------------------
def test_training_runs_and_reduces_the_loss():
    cfg = AxialTrainConfig(
        width=8, depth=2, n_colloc=64, adam_iters=60, lbfgs_iters=0, log_every=1000
    )
    model = AxialPinn(AxialParams(), cfg)
    trainer = Trainer(model, cfg)
    zeta, that = trainer.collocation()
    before = trainer.causal_loss(zeta, that).item()
    trainer.train(verbose=False)
    assert trainer.causal_loss(zeta, that).item() < before


def test_block_weights_and_rar_update(model):
    trainer = Trainer(model, TINY)
    trainer.update_block_weights(*trainer.collocation())
    assert torch.all(trainer.block_w > 0)
    assert trainer.rar.numel() == 0
    trainer.cfg.rar_pool, trainer.cfg.rar_add = 128, 8
    trainer.rar_refine()
    assert trainer.rar.shape == (8, 2)


def test_seed_makes_training_reproducible():
    a = AxialPinn(AxialParams(), TINY)
    b = AxialPinn(AxialParams(), TINY)
    zeta, t = np.linspace(0.0, 1.0, 9), np.array([30.0])
    np.testing.assert_allclose(a.predict(zeta, t)[0], b.predict(zeta, t)[0])


def test_relative_l2_reports_every_field(model):
    err = relative_l2(model, solve_reference(AxialParams(), n_out=21))
    assert set(err) == {"T_f", "T_cl", "T_s", "T_c", "L_void_max_err_m"}
    assert all(np.isfinite(v) for v in err.values())


# --- M6: the PINN with the kinetics closed ---------------------------------
PLAN_A = AxialTrainConfig(
    width=8, depth=2, feedback=True, n_time=16, adam_iters=3, lbfgs_iters=2, log_every=100
)


@pytest.fixture
def plan_a_model():
    return AxialPinn(AxialParams(), PLAN_A)


def test_precursors_start_at_one_and_stay_positive(plan_a_model):
    """`c = exp(t_hat N)` pins `c(0) = 1` exactly and makes `c > 0` unconditional.

    Positivity is not decoration: with `c > 0` and the pole guard on `beta - rho`,
    `P = sum(beta_i c_i)/(beta - rho)` cannot reach zero. That is the structural
    answer to the power-collapse question REPORT-01 section 5.2 is about — the
    trivial solution is removed by construction rather than avoided by training.
    """
    c0 = _precursors(plan_a_model, torch.zeros(1, 1, dtype=torch.float64))
    np.testing.assert_allclose(c0.detach().numpy(), 1.0, rtol=0.0, atol=0.0)
    c = _precursors(plan_a_model, torch.rand(64, 1, dtype=torch.float64))
    assert bool((c > 0).all())


def test_closed_loop_power_starts_at_nominal(plan_a_model):
    """`P(0) = sum(beta_i)/beta = 1` exactly, for any weights and with no offset."""
    power, rho = plan_a_model.predict_power(np.array([0.0]))
    assert power[0] == pytest.approx(1.0, rel=1e-12)
    assert abs(rho[0]) < 1e-15


def test_closed_loop_has_six_blocks_and_they_are_per_time(plan_a_model):
    that = torch.rand(16, 1, dtype=torch.float64)
    blocks = plan_a_model.closed_loop_blocks(that)
    assert len(blocks) == plan_a_model.n_blocks + 1  # fields + precursors
    assert all(b.shape == (16,) for b in blocks)
    assert all(bool(torch.isfinite(b).all()) for b in blocks)


def test_closed_loop_gradients_reach_both_networks(plan_a_model):
    """The field net and the precursor net must both be trained by the loss."""
    blocks = plan_a_model.closed_loop_blocks(torch.rand(16, 1, dtype=torch.float64))
    grads = torch.autograd.grad(
        sum(b.mean() for b in blocks), list(plan_a_model.parameters()), allow_unused=True
    )
    assert all(g is not None for g in grads)


def test_plan_a_training_runs_and_reduces_the_loss():
    cfg = AxialTrainConfig(
        width=8, depth=2, feedback=True, n_time=16, adam_iters=40, lbfgs_iters=0, log_every=1000
    )
    model = AxialPinn(AxialParams(), cfg)
    trainer = Trainer(model, cfg)
    zeta, that = trainer.collocation()
    before = trainer.causal_loss(zeta, that).item()
    trainer.train(verbose=False)
    assert trainer.causal_loss(zeta, that).item() < before


def test_plan_a_collocates_in_time_only():
    """The axial direction is the fixed quadrature the reactivity integral needs."""
    cfg = AxialTrainConfig(width=8, depth=2, feedback=True, n_time=16)
    trainer = Trainer(AxialPinn(AxialParams(), cfg), cfg)
    zeta, that = trainer.collocation()
    assert zeta.shape == that.shape
    assert torch.equal(zeta, that)


def test_block_weight_spread_is_bounded(model):
    """The unbounded scheme ran to 6.2e6 against 0.451 and every field was worse for it.

    `lambda_k = mean(g)/g_k` gives a block that is being fitted well an
    ever-larger weight, with nothing to stop the feedback. Bounding the ratio is
    the whole fix; measured in `docs/axial_nn.md` §7.2.
    """
    cfg = AxialTrainConfig(width=8, depth=2, n_colloc=64, weight_max_ratio=10.0)
    trainer = Trainer(model, cfg)
    for _ in range(40):  # far past where the unbounded version had diverged
        trainer.update_block_weights(*trainer.collocation())
    w = trainer.block_w
    assert float(w.max() / w.min()) <= 10.0**2 + 1e-9
    assert torch.all(w > 0.0)


def test_block_weighting_can_be_switched_off_by_the_ratio_knob(model):
    """`weight_max_ratio = 1` is the measured-equivalent "no weighting" setting."""
    cfg = AxialTrainConfig(width=8, depth=2, n_colloc=64, weight_max_ratio=1.0)
    trainer = Trainer(model, cfg)
    for _ in range(5):
        trainer.update_block_weights(*trainer.collocation())
    np.testing.assert_allclose(trainer.block_w.numpy(), np.ones(model.n_blocks))


def test_bounded_weights_preserves_ratios_below_the_cap():
    """Only ratios matter (Adam is scale-invariant), so renormalising must not move them."""
    raw = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.5], dtype=torch.float64)
    out = _bounded_weights(raw, cap=100.0)  # nothing clamps: spread is 16 < 100
    np.testing.assert_allclose((out / out[0]).numpy(), (raw / raw[0]).numpy(), rtol=1e-14)
    assert float(torch.log(out).mean().abs()) < 1e-14  # unit geometric mean


def test_pseudo_time_anchor_uses_real_zeta_under_feedback():
    """Plan A collocates in time only, so the anchor has to rebuild the tensor grid.

    `collocation()` returns times in *both* slots under feedback. Passing that
    pair straight to `normalised_state` evaluated the ansatz with times standing
    in for `zeta`, so the proximal pull of arXiv:2604.23528 aimed at a state on
    the wrong manifold.
    """
    cfg = AxialTrainConfig(
        width=8, depth=2, n_time=8, feedback=True, pts_every=1, log_every=100, adam_iters=2
    )
    trainer = Trainer(AxialPinn(AxialParams(), cfg), cfg)
    zeta, that = trainer._anchor_points(1.0)
    n_z = trainer.model.zeta_q.shape[0]
    assert zeta.shape == that.shape
    assert zeta.shape[0] % n_z == 0
    np.testing.assert_allclose(zeta[:n_z].detach().numpy(), trainer.model.zeta_q.detach().numpy())
    trainer.pseudo_time_step(1.0)
    assert torch.isfinite(trainer._pts_penalty())


def test_rar_is_disabled_under_feedback():
    """RAR adds arbitrary points; a quadrature rule cannot absorb them."""
    cfg = AxialTrainConfig(width=8, depth=2, feedback=True, n_time=16)
    trainer = Trainer(AxialPinn(AxialParams(), cfg), cfg)
    trainer.rar_refine()
    assert trainer.rar.numel() == 0


# --- D-TH-3: the void eliminated algebraically ------------------------------
def test_void_closure_removes_the_void_residual_block():
    """QSSA drops the block whose normalised rate is 8.5e4; the other four remain."""
    p = AxialParams()
    on = AxialPinn(p, AxialTrainConfig(width=8, depth=2, void_closure=True))
    off = AxialPinn(p, AxialTrainConfig(width=8, depth=2, void_closure=False))
    assert on.n_blocks == 4
    assert off.n_blocks == 5
    gen = torch.Generator().manual_seed(0)
    z, t = (torch.rand(32, 1, dtype=torch.float64, generator=gen) for _ in range(2))
    assert len(on.residual_blocks(z, t)) == 4
    assert len(off.residual_blocks(z, t)) == 5


def test_void_closure_slaves_alpha_to_the_coolant_temperature():
    """`alpha` is a function of the network's own `T_c`, not a free output."""
    p = AxialParams()
    m = AxialPinn(p, AxialTrainConfig(width=8, depth=2, void_closure=True))
    gen = torch.Generator().manual_seed(1)
    z, t = (torch.rand(48, 1, dtype=torch.float64, generator=gen) for _ in range(2))
    state = m.normalised_state(z, t)
    T_c = p.T_in + state[:, 3:4] * m.dT
    np.testing.assert_allclose(
        state[:, 4:5].detach().numpy(),
        quasi_steady_void(T_c, p).detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_void_closure_gives_the_hard_constraints_for_free():
    """`b` underflows to exactly zero below saturation, so no gate is needed.

    The initial condition and the inlet condition on `alpha` fall out of the
    closure rather than being imposed, which is why the void head no longer
    starts half open (the asymmetry a bias offset failed to fix).
    """
    p = AxialParams()
    m = AxialPinn(p, AxialTrainConfig(width=8, depth=2, void_closure=True))
    zeta = torch.linspace(0.0, 1.0, 33, dtype=torch.float64).reshape(-1, 1)
    a0 = m.normalised_state(zeta, torch.zeros_like(zeta))[:, 4].detach()
    assert float(a0.abs().max()) == 0.0
    that = torch.linspace(0.0, 1.0, 33, dtype=torch.float64).reshape(-1, 1)
    a_in = m.normalised_state(torch.zeros_like(that), that)[:, 4].detach()
    assert float(a_in.abs().max()) == 0.0


def test_void_closure_is_differentiable_where_the_switch_is_off():
    """`sqrt(b)` was the first choice and returns NaN: `b` is exactly 0 on 85% of the domain."""
    p = AxialParams()
    T = torch.tensor([700.0, 900.0, 1169.0, 1200.0], dtype=torch.float64, requires_grad=True)
    (grad,) = torch.autograd.grad(quasi_steady_void(T, p).sum(), T)
    assert bool(torch.isfinite(grad).all())


# --- M8 option 2: the front-position network (measured worse; kept as a knob) --
def test_front_network_adds_an_interface_block_and_a_level_set_input():
    """`z_f(t)` gives a fifth block and a third network input, `phi = zeta - z_f`."""
    p = AxialParams()
    off = AxialPinn(p, AxialTrainConfig(width=16, depth=2))
    on = AxialPinn(p, AxialTrainConfig(width=16, depth=2, front_net=True))
    assert not off.use_front
    assert on.use_front
    assert (off.n_blocks, on.n_blocks) == (4, 5)
    assert off.net.net[0].in_features == 2
    assert on.net.net[0].in_features == 3


def test_front_network_keeps_every_hard_constraint():
    """The extra input channel must not loosen the IC, the inlet, or the void bound."""
    p = AxialParams()
    m = AxialPinn(p, AxialTrainConfig(width=16, depth=2, front_net=True))
    zeta = np.linspace(0.0, 1.0, 17)
    np.testing.assert_allclose(
        m.predict(zeta, np.array([0.0]))[3], steady_profile(p, zeta)[3][:, None], atol=1e-9
    )
    np.testing.assert_allclose(m.predict(np.array([0.0]), np.linspace(0, 20, 5))[3], p.T_in, atol=0)
    a = m.predict(zeta, np.linspace(0, 20, 5))[4]
    assert a.min() >= 0.0
    assert a.max() <= 1.0


def test_front_position_can_leave_the_channel_to_mean_no_front():
    """Before onset there is no interface to pin, so `z_f` must be free to exceed 1."""
    p = AxialParams()
    m = AxialPinn(p, AxialTrainConfig(width=16, depth=2, front_net=True))
    z_f = m.front_position(torch.linspace(0, 1, 21, dtype=torch.float64).reshape(-1, 1)).detach()
    assert float(z_f.min()) > 0.0
    assert float(z_f.max()) < 1.25 + 1e-12


def test_front_residual_is_masked_off_while_the_outlet_is_subcooled():
    """The interface condition has no solution before the channel top boils."""
    p = AxialParams()
    m = AxialPinn(p, AxialTrainConfig(width=16, depth=2, front_net=True))
    # Untrained: the outlet sits at the steady profile, far below saturation.
    r = m.front_residual(torch.zeros(8, 1, dtype=torch.float64)).detach()
    assert float(r.abs().max()) == 0.0
