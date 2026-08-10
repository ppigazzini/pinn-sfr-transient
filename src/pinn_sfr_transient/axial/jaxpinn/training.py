"""The training loop: Adam then an L-BFGS polish.

Orchestration only. Every piece it composes — architecture, ansatz, residuals,
weighting, samplers — lives in its own module and can be replaced without editing
this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

import equinox as eqx
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.jaxpinn.ansatz import normalised_state
from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.optimizers import minimize as ssbfgs_minimize
from pinn_sfr_transient.axial.jaxpinn.residuals import n_field_blocks, uses_front
from pinn_sfr_transient.axial.jaxpinn.samplers import _collocation, _merge, _rar_points
from pinn_sfr_transient.axial.jaxpinn.weighting import (
    _block_grad_norms,
    bounded_weights,
    causal_loss,
)


# --- training ---------------------------------------------------------------
def train(
    p: AxialParams | None = None, cfg: AxialTrainConfig | None = None, *, verbose: bool = True
) -> tuple[AxialPinn, AxialParams, AxialTrainConfig]:
    """Adam (causal + adaptive block weights + RAR) then an L-BFGS polish."""
    p = p or AxialParams()
    cfg = cfg or AxialTrainConfig()
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model = jax.random.split(key)
    model = AxialPinn(cfg, k_model)

    n_blocks = (
        n_field_blocks(cfg)
        + (1 if uses_front(cfg) else 0)
        + (1 if cfg.feedback else 0)
        + (1 if cfg.onset_head else 0)
    )
    w = jnp.ones(n_blocks)
    sched = optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.adam_iters), alpha=0.1)
    optimizer = (
        optax.contrib.ademamix(sched) if cfg.first_order == "ademamix" else optax.adam(sched)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        model: AxialPinn, opt_state: optax.OptState, pts: tuple, w: jax.Array, anchor: tuple | None
    ) -> tuple:
        loss, grads = eqx.filter_value_and_grad(lambda m: causal_loss(m, p, cfg, pts, w, anchor))(
            model
        )
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(model, updates), opt_state, loss

    rar: tuple | None = None
    anchor: tuple | None = None
    dtau = cfg.pts_dtau
    # Drawn before the loop, not inside it: with `adam_iters = 0` the loop never
    # runs and the quasi-Newton polish below would have no points to train on --
    # a NameError, and the reason "is Adam needed at all?" had never been tested.
    # The torch twin draws its own set inside `_lbfgs`, so it was never exposed.
    key, ck0 = jax.random.split(key)
    pts = _merge(_collocation(p, cfg, ck0, 1.0, model), rar, feedback=cfg.feedback)
    for it in range(cfg.adam_iters):
        # Time-window curriculum: the horizon opens from 1/n_windows to 1 over
        # training, matching the torch twin. With n_windows = 1 this is a no-op.
        stage = min(int(it / max(cfg.adam_iters, 1) * cfg.n_windows) + 1, cfg.n_windows)
        t_max = stage / cfg.n_windows
        if it and it % cfg.rar_every == 0:
            key, rk = jax.random.split(key)
            rar = _rar_points(model, p, cfg, rk, w)
        # FRESH points every step, plus the fixed-size RAR set. The count is
        # constant so the jitted step never recompiles. Resampling is not a
        # detail: training on a frozen set is the collocation-overfitting mode of
        # arXiv:2605.30910, and holding the set fixed between RAR refreshes is
        # what made this backend stall at ~0.24 relative L2 while the torch twin
        # — which resamples every step — reached ~0.06 on the identical budget.
        key, ck = jax.random.split(key)
        pts = _merge(_collocation(p, cfg, ck, t_max, model), rar, feedback=cfg.feedback)
        if cfg.pts_every and it % cfg.pts_every == 0:
            # Re-anchor and relax. The anchor points are RESAMPLED every step:
            # the paper is explicit that pseudo-time stepping and resampling work
            # together, since a fixed anchor set is one more thing to overfit.
            key, ak = jax.random.split(key)
            a_zeta, a_that = _collocation(p, cfg, ak, t_max, model)[:2]
            a_state = jax.lax.stop_gradient(
                jax.vmap(lambda a, b, m=model: normalised_state(m, p, a, b, cfg))(a_zeta, a_that)
            )
            anchor = (a_zeta, a_that, a_state, dtau)
            dtau *= cfg.pts_growth
        if cfg.weight_max_ratio > 1.0 and it and it % cfg.weight_update_every == 0:
            gn = _block_grad_norms(model, p, cfg, pts)
            target = bounded_weights(gn.mean() / (gn + 1e-12), cfg.weight_max_ratio)
            w = cfg.weight_momentum * w + (1.0 - cfg.weight_momentum) * target
        model, opt_state, loss = step(model, opt_state, pts, w, anchor)
        if verbose and it % cfg.log_every == 0:
            print(f"[adam {it:6d}] loss={float(loss):.3e}")

    if cfg.lbfgs_iters > 0:
        model = _lbfgs_polish(model, p, cfg, pts, w, verbose=verbose)
    return model, p, cfg


def _polish_spec(cfg: AxialTrainConfig, model: AxialPinn):  # noqa: ANN202
    """Which parameters the quasi-Newton stage is allowed to move.

    Everything, unless ``freeze_encoder`` is set, in which case the **first Linear** --
    the projection from the Fourier features into the trunk -- is held fixed and the
    polish optimises the trunk alone.

    The argument is a counting one. That layer is an encoder: it decides which of the
    512 embedded frequencies the network uses, which is a *representation* choice, and
    Adam with its fresh-sample-per-step stream is the tool suited to it. It is also 66%
    of the model -- 32 832 of 49 797 parameters. Holding it fixed turns the polish from a
    2.1x **under**determined problem into a 1.4x **over**determined one (24 000 residual
    entries against 16 965 trainable weights), which is the side of the line the
    literature prescribes, and it drops two-thirds of the parameters from every
    curvature pair as well.

    Off by default, so no published number moves when it lands.
    """
    spec = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    if not cfg.freeze_encoder:
        return spec
    return eqx.tree_at(
        lambda m: (m.mlp.layers[0].weight, m.mlp.layers[0].bias),
        spec,
        replace=(False, False),
    )


def _lbfgs_polish(  # noqa: PLR0913 - polish needs the model, params, points and weights
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    pts: tuple,
    w: jax.Array,
    *,
    verbose: bool,
) -> AxialPinn:
    """Quasi-Newton polish on a fixed collocation set via ``optax.lbfgs``."""
    params, static = eqx.partition(model, _polish_spec(cfg, model))

    def loss_fn(params: AxialPinn) -> jax.Array:
        return causal_loss(eqx.combine(params, static), p, cfg, pts, w)  # no proximal term

    before = float(loss_fn(params))
    if cfg.optimizer in ("ssbfgs", "lbfgs-shared", "ssbroyden"):
        flat0, unravel = ravel_pytree(params)
        vg = eqx.filter_jit(jax.value_and_grad(lambda z: loss_fn(unravel(z))))
        flat, after = ssbfgs_minimize(
            vg,
            flat0,
            max_iter=cfg.lbfgs_iters,
            history_size=cfg.lbfgs_history,
            self_scale=cfg.optimizer in ("ssbfgs", "ssbroyden"),
            broyden_phi=0.5 if cfg.optimizer == "ssbroyden" else 0.0,
        )
        params = unravel(flat)
        if not np.isfinite(after) or after > before:
            if verbose:
                print(f"[{cfg.optimizer}] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
            return model
        if verbose:
            print(f"[{cfg.optimizer} done] loss={after:.3e}")
        return eqx.combine(params, static)

    # Never call this bare: the default memory_size is 10 against torch's 50,
    # and that single argument was the entire cross-backend accuracy gap.
    opt = optax.lbfgs(memory_size=cfg.lbfgs_history)
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def body(_: int, carry: tuple) -> tuple:
        params, state = carry
        loss, grads = value_and_grad(params, state=state)
        updates, state = opt.update(grads, state, params, value=loss, grad=grads, value_fn=loss_fn)
        return optax.apply_updates(params, updates), state

    params, _ = jax.lax.fori_loop(0, cfg.lbfgs_iters, body, (params, state))
    after = float(loss_fn(params))
    # Same divergence guard as the torch twin: a bad line-search step can only
    # cost time, never accuracy.
    if not np.isfinite(after) or after > before:
        if verbose:
            print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        return model
    if verbose:
        print(f"[lbfgs done] loss={after:.3e}")
    return eqx.combine(params, static)
