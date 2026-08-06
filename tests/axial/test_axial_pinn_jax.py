"""M7 tests for the JAX/Equinox twin of the axial PINN.

Two things are checked: that the JAX backend satisfies the same hard constraints
the torch one does, and that the two **agree** where they independently implement
the same quantity. The second is the point of having two backends at all —
``docs/neural_network.md`` §9 records that the 0D model's PyTorch init bug was
identified because the JAX twin fit well at the same budget.

Everything is tiny and short: these assert properties that hold for any weights,
not accuracy, which belongs to a converged run.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("optax")

import jax.numpy as jnp

from pinn_sfr_transient.axial import AxialParams
from pinn_sfr_transient.axial import pinn_jax as pj
from pinn_sfr_transient.axial.reference import steady_profile

TINY = pj.AxialTrainConfig(width=16, depth=2, n_colloc=128, adam_iters=3, lbfgs_iters=2)


@pytest.fixture
def model():
    return pj.AxialPinn(TINY, jax.random.PRNGKey(0))


# --- hard constraints -------------------------------------------------------
def test_initial_condition_matches_the_reference_steady_state(model):
    """The ansatz starts exactly on the state the reference solver starts from."""
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 33)
    got = pj.predict(model, p, zeta, np.array([0.0]))
    names = ("T_f", "T_cl", "T_s", "T_c", "alpha")
    for f, ref, name in zip(got, steady_profile(p, zeta)[:5], names, strict=True):
        np.testing.assert_allclose(f[:, 0], ref, atol=1e-9, err_msg=name)


def test_inlet_boundary_condition_is_exact(model):
    """Eq. 3.9-1 admits one upstream condition; the `zeta` factor pins it identically."""
    p = AxialParams()
    fields = pj.predict(model, p, np.array([0.0]), np.linspace(0.0, p.t_end, 17))
    np.testing.assert_allclose(fields[3], p.T_in, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(fields[4], 0.0, rtol=0.0, atol=0.0)  # no void at the inlet


def test_void_stays_within_bounds(model):
    """A sigmoid behind a gate: `alpha` cannot leave [0, 1) for any weights."""
    p = AxialParams()
    alpha = pj.predict(model, p, np.linspace(0, 1, 21), np.linspace(0, p.t_end, 21))[4]
    assert alpha.min() >= 0.0
    assert alpha.max() < 1.0


def test_precursors_start_at_one_and_stay_positive():
    """`c = exp(t_hat N)` — `c(0) = 1` exact, `c > 0` unconditional."""
    cfg = pj.AxialTrainConfig(width=16, depth=2, feedback=True, n_time=16)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    c0 = pj.precursors(model, jnp.zeros(1))
    np.testing.assert_allclose(np.asarray(c0), 1.0, rtol=0.0, atol=0.0)
    rng = np.random.default_rng(0)
    c = jax.vmap(lambda x: pj.precursors(model, x))(jnp.asarray(rng.random((32, 1))))
    assert bool((np.asarray(c) > 0).all())


def test_closed_loop_power_starts_at_nominal():
    """`P(0) = sum(beta_i)/beta = 1` exactly, with no criticality offset."""
    cfg = pj.AxialTrainConfig(width=16, depth=2, feedback=True, n_time=16)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    power, rho = pj.predict_power(model, AxialParams(), np.array([0.0]))
    assert power[0] == pytest.approx(1.0, rel=1e-12)
    assert abs(rho[0]) < 1e-15


# --- cross-backend agreement ------------------------------------------------
def test_theta0_matches_the_torch_backend():
    """Two independent implementations of the hard IC must not drift apart.

    Both backends reimplement the analytic steady profile — JAX cannot trace the
    numpy one, and torch cannot either. Two transcriptions of the same closed
    form is exactly the situation where a silent divergence hides, so it is
    asserted rather than assumed.
    """
    torch = pytest.importorskip("torch")
    # Local, so the JAX CI job (which has no torch) skips rather than errors.
    from pinn_sfr_transient.axial.pinn_torch import AxialPinn as TorchPinn
    from pinn_sfr_transient.axial.pinn_torch import AxialTrainConfig as TorchCfg

    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 65)
    tmodel = TorchPinn(p, TorchCfg(width=8, depth=2))
    zt = torch.tensor(zeta.reshape(-1, 1), dtype=torch.float64)
    t_theta = tmodel.theta0(zt).detach().numpy()
    j_theta = np.asarray(jax.vmap(lambda z: pj.theta0(p, z))(jnp.asarray(zeta.reshape(-1, 1))))
    np.testing.assert_allclose(j_theta, t_theta, rtol=1e-12, atol=0.0)


def test_shape_and_integral_match_the_numpy_definitions():
    """The JAX power shape is the same closed form `AxialParams` exposes."""
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 101)
    z = jnp.asarray(zeta)
    np.testing.assert_allclose(np.asarray(pj._power_shape(p, z)), p.power_shape(zeta), rtol=1e-13)
    np.testing.assert_allclose(
        np.asarray(pj._power_integral(p, z)), p.power_shape_integral(zeta), rtol=1e-13
    )


# --- training mechanics -----------------------------------------------------
@pytest.mark.parametrize("feedback", [False, True])
def test_training_runs_end_to_end(feedback):
    """Adam then the L-BFGS polish, both plans, including the jit path."""
    cfg = pj.AxialTrainConfig(
        width=8,
        depth=2,
        n_colloc=64,
        n_time=8,
        adam_iters=4,
        lbfgs_iters=2,
        feedback=feedback,
        log_every=1000,
    )
    model, p, out_cfg = pj.train(AxialParams(), cfg, verbose=False)
    assert out_cfg.feedback == feedback
    fields = pj.predict(model, p, np.linspace(0, 1, 5), np.array([0.0, 30.0]))
    assert all(np.all(np.isfinite(f)) for f in fields)


def test_residual_blocks_are_finite(model):
    p = AxialParams()
    rng = np.random.default_rng(0)
    zeta = jnp.asarray(rng.random((64, 1)))
    that = jnp.asarray(rng.random((64, 1)))
    blocks = pj.residual_blocks(model, p, zeta, that)
    assert len(blocks) == 5
    assert all(bool(np.isfinite(np.asarray(b)).all()) for b in blocks)


def test_closed_loop_blocks_are_per_time():
    cfg = pj.AxialTrainConfig(width=8, depth=2, feedback=True, n_time=8)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    p = AxialParams()
    that, zeta_q, weights = pj._collocation(p, cfg, jax.random.PRNGKey(1))
    blocks = pj.closed_loop_blocks(model, p, that, zeta_q, weights)
    assert len(blocks) == 6
    assert all(b.shape == (that.shape[0],) for b in blocks)


def test_rar_is_disabled_under_feedback():
    """The axial direction is a quadrature rule; RAR would break it."""
    cfg = pj.AxialTrainConfig(width=8, depth=2, feedback=True, n_time=8)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    p = AxialParams()
    pts = pj._rar_points(model, p, cfg, jax.random.PRNGKey(2), jnp.ones(6))
    assert len(pts) == 3  # the Plan A tuple, unchanged
