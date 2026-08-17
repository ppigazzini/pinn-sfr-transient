"""The training loop: Adam then an L-BFGS polish.

Orchestration only. Every piece it composes — architecture, ansatz, residuals,
weighting, samplers — lives in its own module and can be replaced without editing
this file.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinn_sfr_transient.axial.config import AxialParams

from dataclasses import replace

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
    p: AxialParams | None = None,
    cfg: AxialTrainConfig | None = None,
    *,
    verbose: bool = True,
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
) -> tuple[AxialPinn, AxialParams, AxialTrainConfig]:
    """Adam (causal + adaptive block weights + RAR) then an L-BFGS polish.

    ``on_checkpoint`` receives ``(cumulative quasi-Newton iterations, model)`` at each
    entry of ``cfg.polish_checkpoints``, so one run can be scored at several budgets
    instead of being re-run once per budget.
    """
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
    optimizer = _first_order(cfg)
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
    # Adam gets its own collocation count: `adam_colloc` if set, else `n_colloc`.
    a_cfg = replace(cfg, n_colloc=cfg.adam_colloc) if cfg.adam_colloc else cfg
    key, ck0 = jax.random.split(key)
    pts = _merge(_collocation(p, a_cfg, ck0, 1.0, model), rar, feedback=cfg.feedback)
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
        pts = _merge(_collocation(p, a_cfg, ck, t_max, model), rar, feedback=cfg.feedback)
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

    # Schedule-free keeps two sequences: the gradients were evaluated at `y`, which is
    # what `model` holds, and the iterate to REPORT is the running average `x`. Convert
    # before anything downstream sees the model -- the polish, the caller, `predict`.
    if cfg.first_order == "schedulefree" and cfg.adam_iters > 0:
        model = _schedule_free_x(model, opt_state)

    if cfg.lbfgs_iters > 0:
        key, pk = jax.random.split(key)
        model = _run_polish(model, p, cfg, rar, w, pk, on_checkpoint, verbose=verbose)
    return model, p, cfg


def _first_order(cfg: AxialTrainConfig) -> optax.GradientTransformation:
    """Build the first-order optimiser with its learning-rate schedule.

    ``schedulefree`` gets a CONSTANT learning rate and its own warmup, not the cosine
    decay the other arms use: the method's claim (arXiv:2405.15682) is that a schedule
    is unnecessary, so composing it with one would measure a hybrid nobody proposed.
    """
    if cfg.first_order == "schedulefree":
        return optax.contrib.schedule_free_adamw(
            cfg.lr,
            warmup_steps=max(1, int(cfg.sf_warmup_frac * cfg.adam_iters)),
            # weight_decay stays at optax's 0.0. AdamW with no decay is Adam plus the
            # schedule-free averaging, which is the one difference this arm is testing;
            # turning decay on as well would make any result unattributable.
            weight_decay=0.0,
        )
    sched = optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.adam_iters), alpha=0.1)
    if cfg.first_order == "ademamix":
        return optax.contrib.ademamix(sched)
    return optax.adam(sched)


def _schedule_free_x(model: AxialPinn, opt_state: optax.OptState) -> AxialPinn:
    """Replace the model's ``y`` iterate with the ``x`` one that should be reported.

    This is the whole hazard of the schedule-free family. The optimiser's parameters are
    the point gradients are taken at; the answer is the weighted average maintained
    alongside it, and the two are NOT close early in a run. Returning ``y`` produces a
    plausible, worse number with nothing to indicate anything went wrong.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(optax.contrib.schedule_free_eval_params(opt_state, params), static)


def _polish_spec(cfg: AxialTrainConfig, model: AxialPinn):  # noqa: ANN202
    """Which parameters the quasi-Newton stage is allowed to move.

    Everything, unless ``freeze_encoder`` is set, in which case the **first Linear** --
    the projection from the Fourier features into the trunk -- is held fixed and the
    polish optimises the trunk alone.

    That layer is an encoder: it decides which of the embedded frequencies the network
    uses, which is a *representation* choice.

    The determinacy argument this docstring used to make is **withdrawn**. Freezing takes
    the trainable count from 17 029 to 16 965 -- 0.4% -- because the projection was never
    fitting capacity (section 7.5.37a), so it cannot turn an underdetermined problem into
    an overdetermined one. What it does change is the **curvature dimension**: the space
    L-BFGS builds its pairs in drops from 49 797 to 16 965 at f256 and from 25 221 to
    16 965 at f64. That is a conditioning argument, and it predicts the benefit should
    scale with the embedding width.

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


def _run_polish(  # noqa: PLR0913, PLR0917 - the polish needs all of this state
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    rar: tuple | None,
    w: jax.Array,
    key: jax.Array,
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
    *,
    verbose: bool,
) -> AxialPinn:
    """Run the quasi-Newton stage, in one block or in several with a fresh set each.

    ``polish_refresh = 0`` keeps the single fixed set every published number here used.
    Above zero the set is redrawn every that many iterations and the optimiser restarted,
    so its curvature history is consistent WITHIN a block and never spans two objectives
    -- which is the point of a fixed set -- while the stage as a whole stops being able
    to overfit one draw. arXiv:2605.24278 runs its BFGS baseline in blocks of 1000.
    """
    q_cfg = replace(cfg, n_colloc=cfg.polish_colloc) if cfg.polish_colloc else cfg

    def draw(k: jax.Array) -> tuple:
        return _merge(_collocation(p, q_cfg, k, 1.0, model), rar, feedback=cfg.feedback)

    cps = tuple(cfg.polish_checkpoints)

    if cfg.freeze_after > 0:
        if cfg.polish_refresh > 0:
            msg = "freeze_after and polish_refresh both set; they schedule the same stage"
            raise ValueError(msg)
        # One set for both halves: the switch already discards the curvature history --
        # the parameter vector changes length -- so redrawing here would confound a
        # restart with a change of objective, and 7.5.37 measured the redraw at 1.5x worse.
        pts = draw(key)
        n1 = min(cfg.freeze_after, cfg.lbfgs_iters)
        free = replace(cfg, lbfgs_iters=n1, freeze_encoder=False)
        model = _lbfgs_polish(
            model, p, free, pts, w, tuple(c for c in cps if c < n1), on_checkpoint, verbose=verbose
        )
        if verbose:
            print(f"[lbfgs] encoder frozen after {n1} of {cfg.lbfgs_iters}", flush=True)
        rest = replace(cfg, lbfgs_iters=cfg.lbfgs_iters - n1, freeze_encoder=True)
        return _lbfgs_polish(
            model,
            p,
            rest,
            pts,
            w,
            tuple(c - n1 for c in cps if c > n1),
            None if on_checkpoint is None else lambda n, m: on_checkpoint(n + n1, m),
            verbose=verbose,
        )

    if cfg.polish_refresh <= 0:
        return _lbfgs_polish(model, p, cfg, draw(key), w, cps, on_checkpoint, verbose=verbose)

    done, blk = 0, cfg.polish_refresh
    while done < cfg.lbfgs_iters:
        n = min(blk, cfg.lbfgs_iters - done)
        key, bk = jax.random.split(key)
        model = _lbfgs_polish(model, p, replace(cfg, lbfgs_iters=n), draw(bk), w, verbose=False)
        done += n
    if verbose:
        print(f"[lbfgs] {cfg.lbfgs_iters} iterations in blocks of {blk}, set redrawn each")
    return model


def _ssbfgs_polish(  # noqa: PLR0913, PLR0917 - the same state the caller holds
    model: AxialPinn,
    static: AxialPinn,
    params: AxialPinn,
    loss_fn,  # noqa: ANN001
    cfg: AxialTrainConfig,
    before: float,
    *,
    verbose: bool,
) -> AxialPinn:
    """Minimise with the self-scaled quasi-Newton family, on a flattened vector."""
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
    if not np.isfinite(after) or after > before:
        if verbose:
            print(f"[{cfg.optimizer}] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        return model
    if verbose:
        print(f"[{cfg.optimizer} done] loss={after:.3e}")
    return eqx.combine(unravel(flat), static)


def _segmented(body, params, state, total: int, checkpoints: tuple[int, ...]):  # noqa: ANN001, ANN202
    """Run ``total`` iterations of ``body``, snapshotting the parameters at each stop.

    The optimiser is **not** restarted at a stop: ``state`` is carried across the segment
    boundary, so the trajectory is the one a single uninterrupted `fori_loop` would take
    and a checkpoint costs one copy. Restarting instead would turn every checkpoint into
    a blocked restart, which section 7.5.37 measured at 1.5x worse -- the intermediate
    rows would then be measuring the checkpointing, not the budget.
    """
    wanted = {b for b in checkpoints if 0 < b <= total}
    snaps, done = [], 0
    for b in [*sorted(wanted - {total}), total]:
        params, state = jax.lax.fori_loop(0, b - done, body, (params, state))
        done = b
        # Only what was asked for. A stage boundary is not a checkpoint: with
        # `freeze_after` the polish runs in two stages, and snapshotting each stage's
        # end silently added a row at the freeze point that no caller requested.
        if done in wanted:
            snaps.append((done, params))
    return params, snaps


def _lbfgs_polish(  # noqa: PLR0913, PLR0917 - polish needs model, params, points, weights
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    pts: tuple,
    w: jax.Array,
    checkpoints: tuple[int, ...] = (),
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
    *,
    verbose: bool,
) -> AxialPinn:
    """Quasi-Newton polish on a fixed collocation set via ``optax.lbfgs``.

    ``checkpoints`` are iteration counts *within this call* at which the model is handed
    to ``on_checkpoint``, so one run can be scored at several budgets. They fire only
    after the divergence guard passes -- a reverted polish has no intermediate states
    worth reporting, and emitting them would put rows in a study file that no returned
    model corresponds to.
    """
    params, static = eqx.partition(model, _polish_spec(cfg, model))

    def loss_fn(params: AxialPinn) -> jax.Array:
        return causal_loss(eqx.combine(params, static), p, cfg, pts, w)  # no proximal term

    # `float(...)` is a HOST SYNC, and every JAX wall-clock this project quotes
    # depends on one. JAX dispatches asynchronously, so a timer around `train()` that
    # returned before the device finished would measure Python dispatch, not the
    # solve -- the classic mistake `block_until_ready` exists to prevent. This call,
    # and its twin after the polish, are what actually await the computation; the
    # divergence guard is load-bearing for the timings as well as for the accuracy.
    # Measured: `train()` as timed today and `train()` followed by an explicit
    # `jax.block_until_ready` agree to run-to-run noise. Remove the guard and the
    # timings silently become dispatch measurements.
    before = float(loss_fn(params))
    if cfg.optimizer in ("ssbfgs", "lbfgs-shared", "ssbroyden"):
        if checkpoints:
            msg = f"polish_checkpoints is not implemented for optimizer={cfg.optimizer!r}"
            raise NotImplementedError(msg)
        return _ssbfgs_polish(model, static, params, loss_fn, cfg, before, verbose=verbose)

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

    params, snaps = _segmented(body, params, state, cfg.lbfgs_iters, checkpoints)
    after = float(loss_fn(params))
    # Same divergence guard as the torch twin: a bad line-search step can only
    # cost time, never accuracy.
    if not np.isfinite(after) or after > before:
        if verbose:
            print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        return model
    if verbose:
        print(f"[lbfgs done] loss={after:.3e}")
    if on_checkpoint is not None:
        for n, snap in snaps:
            on_checkpoint(n, eqx.combine(snap, static))
    return eqx.combine(params, static)
