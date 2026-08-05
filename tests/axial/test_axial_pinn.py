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
from pinn_sfr_transient.axial.physics import continuous_derivatives, line_geometry
from pinn_sfr_transient.axial.pinn_torch import (
    AxialPinn,
    AxialTrainConfig,
    Trainer,
    relative_l2,
)
from pinn_sfr_transient.axial.reference import solve_reference, steady_profile

TINY = AxialTrainConfig(width=8, depth=2, n_colloc=64, adam_iters=3, lbfgs_iters=2, log_every=100)


@pytest.fixture
def model():
    return AxialPinn(AxialParams(), TINY)


# --- hard constraints: must hold for ANY weights ---------------------------
def test_initial_condition_is_exact(model):
    """`theta = theta_0 + t_hat * N` — the IC cannot be violated, untrained or not."""
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 33)
    got = model.predict(zeta, np.array([0.0]))
    names = ("T_f", "T_cl", "T_s", "T_c", "alpha")
    for f, ref, name in zip(got, steady_profile(p, zeta), names, strict=True):
        np.testing.assert_allclose(f[:, 0], ref, rtol=0.0, atol=0.0, err_msg=name)


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
    ref = steady_profile(p, zeta)
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
        expected = (d_dt[:, k : k + 1] - scales[k] * rhs[k]).pow(2).squeeze(1)
        np.testing.assert_allclose(
            blocks[k].detach().numpy(), expected.detach().numpy(), rtol=0.0, atol=0.0
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
    T_f, T_cl, T_s, T_c, alpha = steady_profile(p, zeta)
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
        p = AxialParams(n_axial=n)
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
