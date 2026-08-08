"""Smoke + correctness tests for the JAX/Equinox PINN.

Skipped entirely when the optional ``jax`` extra is not installed
(``uv sync --extra jax-cpu``). Configs are deliberately tiny so the whole module
runs quickly on CPU.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.pinn_jax import (
    SFRPinn,
    TrainConfig,
    _rar_points,
    causal_loss,
    predict,
    relative_l2,
    train,
)


def _tiny_cfg(**kw: object) -> TrainConfig:
    base = {
        "width": 8,
        "depth": 2,
        "n_colloc": 64,
        "adam_iters": 3,
        "lbfgs_iters": 0,
        "causal_chunks": 4,
        "log_every": 1,
    }
    return TrainConfig(**{**base, **kw})


def _model(p: SFRParams, cfg: TrainConfig) -> SFRPinn:
    return SFRPinn(p, cfg, jax.random.key(0))


def test_hard_initial_condition() -> None:
    p = SFRParams()
    model = _model(p, _tiny_cfg())
    # s(0) must equal the normalized nominal state s0 = ones(9), by construction.
    assert jnp.allclose(model.state(jnp.array(0.0)), jnp.ones(9))


def test_forward_mode_derivative_shapes() -> None:
    p = SFRParams()
    model = _model(p, _tiny_cfg())
    t = jnp.linspace(0.0, p.t_end, 8)
    s, ds = model.state_and_deriv(t)
    assert s.shape == (8, 9)
    assert ds.shape == (8, 9)
    assert jnp.all(jnp.isfinite(ds))


def test_tiny_training_runs_and_predicts() -> None:
    p = SFRParams()
    model = train(p, _tiny_cfg(), verbose=False)
    pred = predict(model, p, n=16)
    assert pred["P"].shape == (16,)
    assert all(np.all(np.isfinite(pred[k])) for k in ("t", "P", "Tf", "Tc"))
    # relative_l2 against an identical reference must be ~0.
    errs = relative_l2(pred, {k: pred[k] for k in ("t", "P", "Tf", "Tc")})
    assert max(errs.values()) < 1e-9


def test_rar_points_are_fixed_size_and_in_domain() -> None:
    p = SFRParams()
    cfg = _tiny_cfg(rar_pool=256, n_rar=16)
    pts = _rar_points(_model(p, cfg), p, cfg, jnp.ones(4), jax.random.key(1))
    assert pts.shape == (cfg.n_rar,)  # fixed size -> jit-stable
    assert float(pts.min()) >= 0.0
    assert float(pts.max()) <= p.t_end


def test_causal_loss_and_lbfgs_branch() -> None:
    p = SFRParams()
    cfg = _tiny_cfg(lbfgs_iters=2)
    model = _model(p, cfg)
    t = jnp.linspace(0.0, p.t_end, cfg.n_colloc)
    loss = causal_loss(model, t, jnp.ones(4), p, cfg)
    assert jnp.isfinite(loss)

    trained = train(p, cfg, verbose=False)  # exercises the L-BFGS polish branch
    assert np.all(np.isfinite(predict(trained, p, n=4)["P"]))
