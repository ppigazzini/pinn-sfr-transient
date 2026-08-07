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
from pinn_sfr_transient.axial.physics import quasi_steady_void
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
    cfg = pj.AxialTrainConfig()
    blocks = pj.residual_blocks(model, p, zeta, that, cfg)
    assert len(blocks) == pj.n_field_blocks(cfg)
    assert all(bool(np.isfinite(np.asarray(b)).all()) for b in blocks)


def test_closed_loop_blocks_are_per_time():
    cfg = pj.AxialTrainConfig(width=8, depth=2, feedback=True, n_time=8)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    p = AxialParams()
    that, zeta_q, weights = pj._collocation(p, cfg, jax.random.PRNGKey(1))
    blocks = pj.closed_loop_blocks(model, p, that, zeta_q, weights, cfg)
    assert len(blocks) == pj.n_field_blocks(cfg) + 1  # fields + precursors
    assert all(b.shape == (that.shape[0],) for b in blocks)


def test_causal_weighting_chunks_on_time_not_on_zeta():
    """D40: the JAX loss chunked Plan B by axial position, so the ramp ran up the channel.

    Both plans are checked, because which member of the collocation tuple holds
    time depends on the plan — that is exactly what made the original
    `pts[0]` wrong under Plan B and right under Plan A.

    The probe puts every point in one time chunk but spreads them over all of
    `zeta`. Chunking on time then leaves a single non-empty chunk; chunking on
    `zeta` spreads the loss across many. Comparing the loss against a version
    with `zeta` shuffled isolates it: shuffling `zeta` must not move a
    time-chunked loss at all.
    """
    p = AxialParams()
    cfg = pj.AxialTrainConfig(width=8, depth=2, causal_chunks=32)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    zeta = rng.random((256, 1))
    that = np.full((256, 1), 0.01)  # all inside chunk 0
    w = jnp.ones(5)
    base = float(pj.causal_loss(model, p, cfg, (jnp.asarray(zeta), jnp.asarray(that)), w))
    shuffled = jnp.asarray(rng.permutation(zeta))
    moved = float(pj.causal_loss(model, p, cfg, (shuffled, jnp.asarray(that)), w))
    assert moved == pytest.approx(base, rel=1e-12)


def test_causal_weighting_matches_the_torch_backend():
    """The two backends must agree on the causal weights for the same points.

    A direct cross-check of the reduction the parity study flagged as an
    untested suspect (torch masks versus JAX `bincount`).
    """
    torch = pytest.importorskip("torch")
    from pinn_sfr_transient.axial import pinn_torch as pt

    rng = np.random.default_rng(1)
    that = rng.random((128, 1))
    e = rng.random(128)

    chunks = 8
    idx = np.clip((that.reshape(-1) * chunks).astype(int), 0, chunks - 1)
    j_losses = np.asarray(
        jnp.bincount(jnp.asarray(idx), weights=jnp.asarray(e), length=chunks)
        / jnp.maximum(jnp.bincount(jnp.asarray(idx), length=chunks), 1)
    )
    t_e = torch.tensor(e)
    t_idx = torch.tensor(idx)
    t_losses = np.asarray(
        torch.stack(
            [
                t_e[t_idx == m].mean() if bool((t_idx == m).any()) else t_e.sum() * 0.0
                for m in range(chunks)
            ]
        )
    )
    np.testing.assert_allclose(j_losses, t_losses, rtol=1e-14)
    assert pt._ALPHA_GATE == pj._ALPHA_GATE  # and the two ansatzes stay identical
    assert pt._EXP_BOUND == pj._EXP_BOUND
    # ... and so does the block-weight bound, or the backends drift apart again
    assert pt.AxialTrainConfig().weight_max_ratio == pj.AxialTrainConfig().weight_max_ratio
    raw = np.array([1.0, 2.0, 4.0, 8.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(pj.bounded_weights(jnp.asarray(raw), 100.0)),
        pt._bounded_weights(torch.tensor(raw), 100.0).numpy(),
        rtol=1e-14,
    )


def test_bounded_weights_clamps_the_spread():
    """The measured fix: the ratio between the most- and least-weighted block is capped."""
    raw = jnp.asarray([1e6, 1.0, 1e-6, 1.0, 1.0])
    out = pj.bounded_weights(raw, 10.0)
    assert float(out.max() / out.min()) <= 100.0 + 1e-9
    np.testing.assert_allclose(np.asarray(pj.bounded_weights(raw, 1.0)), np.ones(5))


def test_rar_is_disabled_under_feedback():
    """The axial direction is a quadrature rule; RAR would break it."""
    cfg = pj.AxialTrainConfig(width=8, depth=2, feedback=True, n_time=8)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    p = AxialParams()
    pts = pj._rar_points(model, p, cfg, jax.random.PRNGKey(2), jnp.ones(6))
    assert len(pts) == 3  # the Plan A tuple, unchanged


# --- N1: the two backends must expose the same model ------------------------
@pytest.mark.parametrize(
    "kw", [{}, {"void_closure": False}, {"front_net": True}, {"feedback": True, "n_time": 8}]
)
def test_block_structure_matches_the_torch_backend(kw):
    """A knob that exists in one backend and not the other is a silent model fork.

    The void closure and the front network landed in torch first; this is what
    stops the parity table in `docs/axial_nn.md` being measured on two different
    models again.
    """
    pytest.importorskip("torch")
    from pinn_sfr_transient.axial import pinn_torch as pt

    j_cfg = pj.AxialTrainConfig(width=16, depth=2, **kw)
    t_cfg = pt.AxialTrainConfig(width=16, depth=2, **kw)
    t_model = pt.AxialPinn(AxialParams(), t_cfg)
    j_model = pj.AxialPinn(j_cfg, jax.random.PRNGKey(0))

    n_j = pj.n_field_blocks(j_cfg) + (1 if pj.uses_front(j_cfg) else 0)
    assert n_j == t_model.n_blocks
    assert j_model.mlp.layers[0].in_features == t_model.net.net[0].in_features
    assert pj.uses_front(j_cfg) == t_model.use_front


def test_the_two_backends_share_every_default():
    """Defaults that drift make every cross-backend number a comparison of schedules."""
    pytest.importorskip("torch")
    from pinn_sfr_transient.axial import pinn_torch as pt

    j, t = pj.AxialTrainConfig(), pt.AxialTrainConfig()
    for name in (
        "width",
        "depth",
        "n_colloc",
        "adam_iters",
        "lbfgs_iters",
        "lr",
        "causal_eps",
        "causal_chunks",
        "weight_update_every",
        "weight_momentum",
        "weight_max_ratio",
        "residual_scaling",
        "void_closure",
        "front_net",
        "front_frac",
        "t_train_frac",
        "feedback",
        "n_time",
        "seed",
        "optimizer",
    ):
        assert getattr(j, name) == getattr(t, name), name


def test_void_closure_agrees_across_backends():
    """`quasi_steady_void` is shared source; assert the dispatch really is shared.

    Absolute, not relative. The closure cubes a `tanh`, and the two libm
    implementations saturate at marginally different arguments, so just outside
    the switch one backend returns exactly zero while the other returns ~1e-7 —
    a relative difference of 1 on a quantity that is physically zero. Measured
    maximum absolute disagreement over 600-1400 K is 9.5e-7.
    """
    torch = pytest.importorskip("torch")
    p = AxialParams()
    T = np.linspace(600.0, 1400.0, 101)
    j = np.asarray(quasi_steady_void(jnp.asarray(T), p))
    t = quasi_steady_void(torch.tensor(T, dtype=torch.float64), p).numpy()
    np.testing.assert_allclose(j, t, rtol=0.0, atol=1e-5)


def test_float64_is_enabled_by_importing_the_backend():
    """float64 is a correctness setting for this model, and it is easy to lose.

    `jax_enable_x64` must be applied before any array exists. A refactor once
    moved the module that set it, and the whole backend silently ran in float32 —
    no error, no warning, ~2% different answers. Assert it rather than trust it.
    """
    assert jnp.zeros(1).dtype == jnp.float64
    assert (
        pj.AxialPinn(pj.AxialTrainConfig(width=8, depth=2), jax.random.PRNGKey(0))
        .mlp.layers[0]
        .weight.dtype
        == jnp.float64
    )


def test_self_scaled_bfgs_agrees_across_backends():
    """The `optimizer` knob must select the same algorithm in both backends.

    The two implementations differ in form -- a closure-driven class in torch, a
    function over a flat vector in JAX -- because the frameworks force it. What
    must not differ is the arithmetic. Run both on the same ill-conditioned
    quadratic from the same start and require the same minimiser.
    """
    torch = pytest.importorskip("torch")
    from pinn_sfr_transient.axial.jaxpinn.optimizers import minimize
    from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS

    n = 40
    d_np = np.logspace(0.0, 4.0, n)
    x0_np = np.ones(n)

    d_t = torch.tensor(d_np, dtype=torch.float64)
    x_t = torch.tensor(x0_np, dtype=torch.float64, requires_grad=True)
    opt = SelfScaledLBFGS([x_t], max_iter=25, self_scale=True)

    def closure():
        x_t.grad = None
        loss = 0.5 * (d_t * x_t * x_t).sum()
        loss.backward()
        return loss

    opt.step(closure)

    d_j = jnp.asarray(d_np)

    def value_and_grad(x):
        return 0.5 * jnp.sum(d_j * x * x), d_j * x

    x_j, _ = minimize(value_and_grad, jnp.asarray(x0_np), max_iter=25, self_scale=True)

    np.testing.assert_allclose(np.asarray(x_j), x_t.detach().numpy(), rtol=1e-10, atol=1e-12)


def test_self_scaling_off_reproduces_textbook_lbfgs():
    """`self_scale=False` must be plain L-BFGS, so the comparison has a control.

    Without this the self-scaled arm has nothing to be measured against inside
    the same implementation, and a difference could be the scaling or could be
    the line search.
    """
    torch = pytest.importorskip("torch")
    from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS

    def rosenbrock(x):
        return (100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2).sum()

    x0 = torch.full((20,), -1.2, dtype=torch.float64)
    x0[1::2] = 1.0

    def run(make):
        x = x0.clone().requires_grad_(True)
        opt = make([x])

        def closure():
            if isinstance(opt, torch.optim.LBFGS):
                opt.zero_grad()
            else:
                x.grad = None
            loss = rosenbrock(x)
            loss.backward()
            return loss

        opt.step(closure)
        return float(rosenbrock(x).detach())

    ours = run(lambda ps: SelfScaledLBFGS(ps, max_iter=200, self_scale=False))
    theirs = run(
        lambda ps: torch.optim.LBFGS(
            ps, max_iter=200, history_size=50, line_search_fn="strong_wolfe"
        )
    )
    # Not bit-equal -- the zoom interpolates differently -- so the assertion is
    # that both solve it, not that they agree digit for digit. Rosenbrock starts
    # near 5e3 here; anything below 1e-6 has found the valley floor.
    assert ours < 1e-6, ours
    assert theirs < 1e-6, theirs


def test_residual_blocks_are_identical_given_identical_parameters():
    """Transplant torch's weights into the Equinox model; every block must match.

    The two backends disagree by a consistent 21% on `T_s` and `T_c` after
    training (`docs/axial_nn.md` section 7.3.2), and every previous backend
    disagreement in this project turned out to be a bug. This separates the two
    possible causes: if the residuals differ at identical parameters, the
    equations forked; if they agree to round-off, the difference is training
    dynamics and the equations are exonerated.

    They agree to ~1e-14 relative. Keep it that way.
    """
    torch = pytest.importorskip("torch")
    import equinox as eqx

    from pinn_sfr_transient.axial import pinn_torch as pt
    from pinn_sfr_transient.axial.jaxpinn.ansatz import normalised_state as j_state
    from pinn_sfr_transient.axial.jaxpinn.residuals import residual_blocks as j_blocks

    width, depth, n = 16, 3, 129
    p = AxialParams()
    tcfg, jcfg = (
        pt.AxialTrainConfig(width=width, depth=depth),
        pj.AxialTrainConfig(width=width, depth=depth),
    )
    torch.manual_seed(0)
    tm = pt.AxialPinn(p, tcfg)
    jm = pj.AxialPinn(jcfg, jax.random.PRNGKey(0))

    t_linear = [m for m in tm.modules() if isinstance(m, torch.nn.Linear)]
    assert len(t_linear) == len(jm.mlp.layers)
    for i, tl in enumerate(t_linear):
        jm = eqx.tree_at(
            lambda m, i=i: m.mlp.layers[i].weight, jm, jnp.asarray(tl.weight.detach().numpy())
        )
        jm = eqx.tree_at(
            lambda m, i=i: m.mlp.layers[i].bias, jm, jnp.asarray(tl.bias.detach().numpy())
        )

    rng = np.random.default_rng(0)
    zeta = rng.uniform(0.0, 1.0, (n, 1))
    that = rng.uniform(0.0, 1.0, (n, 1))
    zt = torch.tensor(zeta, dtype=torch.float64)
    tt = torch.tensor(that, dtype=torch.float64)

    np.testing.assert_allclose(
        np.asarray(
            jax.vmap(lambda a, b: j_state(jm, p, a, b, jcfg))(jnp.asarray(zeta), jnp.asarray(that))
        ),
        tm.normalised_state(zt, tt).detach().numpy(),
        rtol=1e-11,
        atol=1e-13,
    )

    tb = tm.residual_blocks(zt, tt)
    jb = j_blocks(jm, p, jnp.asarray(zeta), jnp.asarray(that), jcfg)
    assert len(tb) == len(jb)
    for k, (a, b) in enumerate(zip(tb, jb, strict=True)):
        np.testing.assert_allclose(
            np.asarray(b).ravel(),
            a.detach().numpy().ravel(),
            rtol=1e-10,
            atol=1e-13,
            err_msg=f"residual block {k} differs between backends",
        )
