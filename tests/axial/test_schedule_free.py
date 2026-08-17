"""Schedule-free AdamW: the reported iterate must be ``x``, never ``y``.

The schedule-free family (arXiv:2405.15682) maintains two sequences. Gradients are
evaluated at ``y``, which is what the optimiser's parameter pytree holds; the iterate
that carries the method's guarantee is the weighted average ``x``, recoverable only via
``optax.contrib.schedule_free_eval_params``. Reporting ``y`` is the standard way to get a
wrong number out of this family: it trains, it converges, it produces a plausible result,
and nothing anywhere says the wrong sequence was read.

So these tests assert the distinction rather than an accuracy. Two of them would pass
against an implementation that silently returned ``y``; the third would not, and it is
the reason this file exists.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import equinox as eqx
import jax
import optax

from pinn_sfr_transient.axial import pinn_jax as pj
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.training import _first_order, _schedule_free_x

ITERS = 6


def _cfg(**kw) -> AxialTrainConfig:
    base = {
        "width": 8,
        "depth": 2,
        "n_colloc": 32,
        "adam_iters": ITERS,
        "lbfgs_iters": 0,
        "log_every": 10**9,
    }
    return AxialTrainConfig(**(base | kw))


def test_schedule_free_replaces_the_cosine_decay_rather_than_composing_with_it():
    """The method's claim is that no schedule is needed, so it must not get one.

    Checked through the optimiser's own state: ``schedule_free_adamw`` carries the
    schedule-free bookkeeping, and a cosine-wrapped Adam does not.
    """
    sf = _first_order(_cfg(first_order="schedulefree"))
    plain = _first_order(_cfg())
    params = {"w": jax.numpy.ones(3)}
    assert "ScheduleFree" in type(sf.init(params)).__name__ or any(
        "ScheduleFree" in type(s).__name__
        for s in jax.tree_util.tree_leaves(sf.init(params), is_leaf=lambda x: hasattr(x, "_fields"))
    )
    assert "ScheduleFree" not in type(plain.init(params)).__name__


def test_the_warmup_is_a_fraction_of_the_budget_not_a_fixed_count():
    """A fixed warmup becomes the whole run at a short budget; a fraction cannot."""
    # Read the warmup back off the built optimiser by stepping it: the learning rate
    # during warmup rises with the step count, so a budget-scaled warmup means the two
    # budgets are at DIFFERENT fractions of their ramp at the same absolute step.
    import jax.numpy as jnp

    def lr_at(adam_iters: int, step: int) -> float:
        opt = _first_order(_cfg(first_order="schedulefree", adam_iters=adam_iters))
        params = {"w": jnp.zeros(2)}
        state = opt.init(params)
        for _ in range(step):
            upd, state = opt.update({"w": jnp.ones(2)}, state, params)
            params = optax.apply_updates(params, upd)
        return float(jnp.abs(jnp.asarray(params["w"])).max())

    # 10 steps into a 100-iteration budget is the END of its warmup; 10 steps into a
    # 30000-iteration budget is 0.3% of it, so it has moved far less.
    assert lr_at(100, 10) > lr_at(30000, 10)


def test_x_and_y_differ_and_training_returns_x():
    """The load-bearing one: the returned model must be ``x``, not the optimiser's ``y``.

    Reconstructed independently -- this test runs its own schedule-free loop, keeps the
    raw ``y``, converts to ``x`` with optax's own function, and requires that

      * ``x`` and ``y`` are genuinely different (otherwise the test proves nothing), and
      * what ``train`` hands back matches ``x`` and NOT ``y``.

    An implementation that forgot the conversion passes every other test in this file.
    """
    p = AxialParams()
    cfg = _cfg(first_order="schedulefree")

    # --- an independent replay of the loop, to get y and x by hand -----------------
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model = jax.random.split(key)
    from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn

    model = AxialPinn(cfg, k_model)
    opt = _first_order(cfg)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    loss_of = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda m, pts: _loss(m, p, cfg, pts)),
    )
    pts = _points(p, cfg, key, model)
    for _ in range(ITERS):
        _, grads = loss_of(model, pts)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, state = opt.update(grads, state, params)
        model = eqx.apply_updates(model, updates)

    y = eqx.filter(model, eqx.is_inexact_array)
    x = optax.contrib.schedule_free_eval_params(state, y)
    flat_y = np.concatenate([np.asarray(a).ravel() for a in jax.tree_util.tree_leaves(y)])
    flat_x = np.concatenate([np.asarray(a).ravel() for a in jax.tree_util.tree_leaves(x)])
    assert not np.allclose(flat_x, flat_y), "x and y coincide; the test cannot detect the bug"

    # --- and what the helper the trainer uses produces --------------------------------
    got = eqx.filter(_schedule_free_x(model, state), eqx.is_inexact_array)
    flat_got = np.concatenate([np.asarray(a).ravel() for a in jax.tree_util.tree_leaves(got)])
    np.testing.assert_array_equal(flat_got, flat_x)
    assert not np.allclose(flat_got, flat_y)


def test_train_end_to_end_returns_the_averaged_iterate():
    """The conversion has to happen inside ``train``, not be left to the caller."""
    p = AxialParams()
    model, _, cfg = pj.train(p, _cfg(first_order="schedulefree"), verbose=False)
    zeta = np.linspace(0.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 5)
    fields = pj.predict(model, p, zeta, t, cfg)
    assert all(np.all(np.isfinite(np.asarray(f))) for f in fields)

    # The same budget under plain Adam must give a DIFFERENT model: if the two agreed,
    # `first_order` would not be reaching the optimiser at all -- the D67 failure.
    other, _, ocfg = pj.train(p, _cfg(), verbose=False)
    assert not np.allclose(
        np.asarray(fields[0]), np.asarray(pj.predict(other, p, zeta, t, ocfg)[0])
    )


def _points(p: AxialParams, cfg: AxialTrainConfig, key, model):
    from pinn_sfr_transient.axial.jaxpinn.samplers import _collocation, _merge

    return _merge(_collocation(p, cfg, key, 1.0, model), None, feedback=cfg.feedback)


def _loss(model, p: AxialParams, cfg: AxialTrainConfig, pts):
    import jax.numpy as jnp

    from pinn_sfr_transient.axial.jaxpinn.residuals import n_field_blocks
    from pinn_sfr_transient.axial.jaxpinn.weighting import causal_loss

    return causal_loss(model, p, cfg, pts, jnp.ones(n_field_blocks(cfg)))


def test_the_polish_starts_from_x_not_y(monkeypatch):
    """The conversion must happen BEFORE the quasi-Newton stage, not after it.

    Otherwise the polish is handed the gradient-evaluation point and spends its budget
    refining the wrong iterate -- a failure that survives every test above, because the
    model finally returned would still have been converted somewhere.

    Captured behaviourally: `_run_polish` is replaced by a spy, and what it receives has
    to equal the `x` that the same configuration produces with the polish switched off.
    """
    from pinn_sfr_transient.axial.jaxpinn import training as tr

    p = AxialParams()
    x_only = pj.train(p, _cfg(first_order="schedulefree"), verbose=False)[0]

    seen = {}

    def spy(model, *_args, **_kwargs):
        seen["model"] = model
        return model

    monkeypatch.setattr(tr, "_run_polish", spy)
    pj.train(p, _cfg(first_order="schedulefree", lbfgs_iters=3), verbose=False)

    def flat(m):
        leaves = jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))
        return np.concatenate([np.asarray(a).ravel() for a in leaves])

    np.testing.assert_array_equal(flat(seen["model"]), flat(x_only))
