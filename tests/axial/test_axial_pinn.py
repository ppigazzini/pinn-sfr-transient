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
    residual_normalisation,
    residual_scales,
)
from pinn_sfr_transient.axial.pinn_torch import (
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
    for k in range(5):
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
    for k in range(5):
        np.testing.assert_allclose(
            b_on[k].detach().numpy(),
            (b_off[k] * on.res_norm[k] ** 2).detach().numpy(),
            rtol=1e-14,
        )
    np.testing.assert_allclose(off.res_norm, np.ones(5))


def test_variable_scaling_brings_every_block_to_the_same_order():
    """The whole point: 813x of spread in the natural rates becomes O(1) per block.

    `residual_normalisation` is `tau_k / t_end`, and a block's normalised
    derivative is of order `t_end / tau_k`, so the product is order one. Without
    it the void block carries a term of order 8.5e4 against 104 for the fuel.
    """
    p = AxialParams()
    tau = residual_scales(p)
    nrm = residual_normalisation(p)
    rates = [p.t_end / t for t in tau]
    assert max(rates) / min(rates) > 500.0  # the raw spread is severe
    scaled = [r * n for r, n in zip(rates, nrm, strict=True)]
    np.testing.assert_allclose(scaled, np.ones(5), rtol=1e-12)


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
    assert len(blocks) == 6  # four temperatures, void, precursors
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
    np.testing.assert_allclose(trainer.block_w.numpy(), np.ones(5))


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
