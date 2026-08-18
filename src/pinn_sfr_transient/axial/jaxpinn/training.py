"""The training loop: Adam then an L-BFGS polish.

Orchestration only. Every piece it composes — architecture, ansatz, residuals,
weighting, samplers — lives in its own module and can be replaced without editing
this file.
"""

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    import chex

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
def _next_boundary(cfg: AxialTrainConfig, it: int, *, verbose: bool, checkpointing: bool) -> int:
    """How many iterations may run fused before Python must intervene again.

    Everything the loop does per-iteration is compiled; everything it does on a
    *cadence* -- RAR, weights, pseudo-time, logging, checkpoints -- is Python. This
    returns the distance to the nearest such event so no event moves: with all
    cadences off it returns the whole remaining budget and the run is one compiled
    loop, which is what the companion's does.
    """
    left = cfg.adam_iters - it
    cadences = []
    if cfg.rar_every:
        cadences.append(cfg.rar_every - it % cfg.rar_every)
    if cfg.pts_every:
        cadences.append(cfg.pts_every - it % cfg.pts_every)
    if cfg.weight_max_ratio > 1.0 and cfg.weight_update_every:
        cadences.append(cfg.weight_update_every - it % cfg.weight_update_every)
    if verbose and cfg.log_every:
        cadences.append(cfg.log_every - it % cfg.log_every)
    if checkpointing and cfg.adam_checkpoint_every:
        cadences.append(cfg.adam_checkpoint_every - it % cfg.adam_checkpoint_every)
    return max(1, min([left, *cadences]))


def train(  # noqa: C901, PLR0915 - one loop with five cadences; splitting it would
    # hide which events fire on which iteration, and that ordering is the contract
    # `_next_boundary` exists to preserve.
    p: AxialParams | None = None,
    cfg: AxialTrainConfig | None = None,
    *,
    verbose: bool = True,
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
) -> tuple[AxialPinn, AxialParams, AxialTrainConfig]:
    """Adam (causal + adaptive block weights + RAR) then an L-BFGS polish.

    ``on_checkpoint`` receives ``(cumulative iterations, model)``: at each entry of
    ``cfg.polish_checkpoints`` during the quasi-Newton stage, and every
    ``cfg.adam_checkpoint_every`` iterations during the first-order one. Either way a
    single run is scored at several budgets instead of being re-run once per budget --
    which for a 10-rung first-order ladder is one run instead of ten.
    """
    p = p or AxialParams()
    cfg = cfg or AxialTrainConfig()
    # THE REFERENCE IMPLEMENTATION'S DERIVATION, EXACTLY. `k_model` is `split(key)[0]`,
    # the fixed collocation set comes off `k_points = split(key)[1]`, and the first-order
    # stream is `fold_in(key, 1)` rather than a wider split -- splitting differently moves
    # both of the others.
    #
    # This read `key, k_model = jax.random.split(key)`, taking the WRONG HALF for the
    # model and deriving every draw from the other one. Same seed, different weights,
    # different points: `seed = 0` here was not `seed = 0` there, so no number from this
    # backend could be compared with a reference one, and the seed spread on this problem
    # reaches 12.5x. The reference measures the key derivation alone at 0.0314 s -> 0.0103 s
    # of boiling-onset error.
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_points = jax.random.split(key)
    k_step = jax.random.fold_in(key, 1)
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

    key = k_step  # the first-order stream; the polish takes `k_points` untouched
    rar: tuple | None = None
    anchor: tuple | None = None
    dtau = cfg.pts_dtau
    # Adam gets its own collocation count: `adam_colloc` if set, else `n_colloc`.
    a_cfg = replace(cfg, n_colloc=cfg.adam_colloc) if cfg.adam_colloc else cfg

    # --- the fused inner loop -------------------------------------------------
    #
    # The draw and the update run INSIDE one compiled region, and the loop over
    # iterations is `lax.fori_loop`, not a Python `for`. Previously each iteration
    # made three dispatch round-trips -- `_collocation`, `_merge`, then a jitted
    # `step` -- and the cores idled through the Python between them: measured at
    # **15.6% of wall time outside the fused region**, which is most of the CPU-
    # utilisation gap against the companion's loop (it drew its batch inside
    # `lax.fori_loop` from the start).
    #
    # Python still runs, but only at CADENCE BOUNDARIES -- a RAR refresh, a weight
    # update, a pseudo-time re-anchor, a checkpoint, a log line. `_next_boundary`
    # returns how many iterations may run before the next such event, so the schedule
    # is exact: every event fires on the iteration the cadence says it does, including
    # for periods that do not divide the budget. `tests/axial/test_fused_loop.py`
    # reconstructs the whole event set from `_next_boundary` and pins it.
    #
    # That -- not agreement with the previous loop -- is the contract. The previous loop
    # drew the early-time cluster on every iteration, which was measured worse in every
    # time window and is retired (see `samplers.py`); reproducing its numbers is not a
    # goal and matching them would be a defect.
    static = eqx.partition(model, eqx.is_inexact_array)[1]

    @eqx.filter_jit
    def run_block(  # noqa: PLR0913, PLR0917 - the block's whole state, flat is clearer
        params: AxialPinn,
        opt_state: optax.OptState,
        key: jax.Array,
        w: jax.Array,
        rar: tuple | None,
        t_max: float,
        n: jax.Array,
    ) -> tuple:
        def body(_i: jax.Array, carry: tuple) -> tuple:
            params, opt_state, key, _loss = carry
            model = eqx.combine(params, static)
            key, ck = jax.random.split(key)
            pts = _merge(_collocation(p, a_cfg, ck, t_max, model), rar, feedback=cfg.feedback)
            loss, grads = eqx.filter_value_and_grad(
                lambda m: causal_loss(m, p, cfg, pts, w, anchor)
            )(model)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_inexact_array), opt_state, params
            )
            return optax.apply_updates(params, updates), opt_state, key, loss

        return jax.lax.fori_loop(0, n, body, (params, opt_state, key, jnp.zeros(())))

    params = eqx.filter(model, eqx.is_inexact_array)
    it = 0
    loss = jnp.zeros(())
    while it < cfg.adam_iters:
        stage = min(int(it / max(cfg.adam_iters, 1) * cfg.n_windows) + 1, cfg.n_windows)
        t_max = stage / cfg.n_windows
        model = eqx.combine(params, static)
        if cfg.rar_every and it and it % cfg.rar_every == 0:
            key, rk = jax.random.split(key)
            rar = _rar_points(model, p, cfg, rk, w)
        if cfg.pts_every and it % cfg.pts_every == 0:
            key, ak = jax.random.split(key)
            a_zeta, a_that = _collocation(p, cfg, ak, t_max, model)[:2]
            a_state = jax.lax.stop_gradient(
                jax.vmap(lambda a, b, m=model: normalised_state(m, p, a, b, cfg))(a_zeta, a_that)
            )
            anchor = (a_zeta, a_that, a_state, dtau)
            dtau *= cfg.pts_growth
        if cfg.weight_max_ratio > 1.0 and it and it % cfg.weight_update_every == 0:
            gn = _block_grad_norms(
                model,
                p,
                cfg,
                _merge(
                    _collocation(p, a_cfg, jax.random.fold_in(key, it), t_max, model),
                    rar,
                    feedback=cfg.feedback,
                ),
            )
            target = bounded_weights(gn.mean() / (gn + 1e-12), cfg.weight_max_ratio)
            w = cfg.weight_momentum * w + (1.0 - cfg.weight_momentum) * target

        n = _next_boundary(cfg, it, verbose=verbose, checkpointing=on_checkpoint is not None)
        params, opt_state, key, loss = run_block(
            params, opt_state, key, w, rar, t_max, jnp.asarray(n)
        )
        it += n

        if verbose and it % cfg.log_every == 0:
            print(f"[adam {it - 1:6d}] loss={float(loss):.3e}", flush=True)
        every = cfg.adam_checkpoint_every
        if on_checkpoint is not None and every and it % every == 0:
            model = eqx.combine(params, static)
            snap = (
                _schedule_free_x(model, opt_state) if cfg.first_order == "schedulefree" else model
            )
            on_checkpoint(it, snap)

    model = eqx.combine(params, static)

    # Schedule-free keeps two sequences: the gradients were evaluated at `y`, which is
    # what `model` holds, and the iterate to REPORT is the running average `x`. Convert
    # before anything downstream sees the model -- the polish, the caller, `predict`.
    if cfg.first_order == "schedulefree" and cfg.adam_iters > 0:
        model = _schedule_free_x(model, opt_state)

    if cfg.lbfgs_iters > 0:
        # `k_points`, not a key derived from the first-order stream: the reference draws
        # its fixed set from `split(key)[1]` whether or not a first-order stage ran, so an
        # `adam_iters = 0` arm here draws exactly the reference's set.
        model = _run_polish(model, p, cfg, rar, w, k_points, on_checkpoint, verbose=verbose)
    return model, p, cfg


def _first_order(cfg: AxialTrainConfig) -> optax.GradientTransformation:
    """Build the first-order optimiser with its learning-rate schedule.

    ``schedulefree`` gets a CONSTANT learning rate and its own warmup, not the cosine
    decay the other arms use: the method's claim (arXiv:2405.15682) is that a schedule
    is unnecessary, so composing it with one would measure a hybrid nobody proposed.
    """
    if cfg.first_order == "schedulefree":
        if cfg.lr_warmup:
            # Refused, not ignored. Schedule-free replaces the learning-rate schedule
            # with an averaging sequence and warms the step size internally over the
            # same `sf_warmup_frac`; wrapping it in an external warmup would either be
            # silently dropped (as it was) or measure a hybrid nobody proposed.
            msg = (
                "lr_warmup and first_order='schedulefree' both schedule the step size; "
                "schedule-free warms up internally over sf_warmup_frac. Pick one."
            )
            raise ValueError(msg)
        return optax.contrib.schedule_free_adamw(
            cfg.lr,
            warmup_steps=max(1, int(cfg.sf_warmup_frac * cfg.adam_iters)),
            # weight_decay stays at optax's 0.0. AdamW with no decay is Adam plus the
            # schedule-free averaging, which is the one difference this arm is testing;
            # turning decay on as well would make any result unattributable.
            weight_decay=0.0,
        )
    sched = _lr_schedule(cfg)
    if cfg.first_order == "ademamix":
        return optax.contrib.ademamix(sched, b3=_b3_warmup(cfg), alpha=_alpha_warmup(cfg))
    return optax.adam(sched)


#: AdEMAMix's final slow-EMA weight and the two decay endpoints its warmup runs between.
#: optax's own defaults, restated because the warmup schedules need the endpoints.
_ADEMAMIX_ALPHA: float = 5.0
_ADEMAMIX_B1: float = 0.9
_ADEMAMIX_B3: float = 0.9999


def _warmup_steps(cfg: AxialTrainConfig) -> int:
    """Warmup length in steps, as a fraction of the first-order budget.

    A fraction rather than a count so it scales with the budget instead of silently
    becoming the whole run at a short one.
    """
    return max(1, int(cfg.sf_warmup_frac * cfg.adam_iters))


def _lr_schedule(cfg: AxialTrainConfig) -> optax.Schedule:
    """Cosine decay, optionally with a linear warmup in front of it.

    ``optax.warmup_cosine_decay_schedule`` rather than a hand-rolled ramp -- optax ships
    warmup composed with cosine decay, and a local reimplementation of a library
    schedule is a second thing to get wrong.

    Off by default. Warmup changes the trajectory of **every** first-order arm, and the
    plain Adam arm is where every published first-order number in this repository comes
    from, so switching it on globally would move numbers that nothing else in the change
    touches.
    """
    if not cfg.lr_warmup:
        return optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.adam_iters), alpha=0.1)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.lr,
        warmup_steps=_warmup_steps(cfg),
        decay_steps=max(1, cfg.adam_iters),
        end_value=cfg.lr * 0.1,
    )


def _alpha_warmup(cfg: AxialTrainConfig) -> optax.Schedule:
    """Ramp AdEMAMix's slow-EMA weight linearly from zero, then hold.

    **Without this the method diverges on this problem, and slowly enough to waste a
    run.** At its defaults AdEMAMix mixes a slow average whose half-life is about 7000
    steps into the update at five times the weight of the fast one; applying both from
    step zero amplifies gradients taken before either average is populated. Measured on
    the companion repository at 10 000 points and f256: loss 5.9e+06 by 200 000 steps at
    lr 1e-4 (1.3e+07 at 1e-3), saturation margins above +8000 K, a voided length of
    0.65 m in a 0.4 m channel, and a NaN onset at one first checkpoint.

    A short smoke test does not catch it. 400 steps cleared the unwarmed version, because
    400 steps is far too few for a 7000-step average to accumulate. **A first-order arm
    needs a smoke test longer than the slowest moving average it configures.**

    ``optax.linear_schedule`` holds ``end_value`` past ``transition_steps``, which is
    exactly the ramp-then-hold wanted; it agrees with the hand-rolled version to 2e-07.

    **Driven from step 1, not from optax's count 0.** optax calls a schedule with the
    number of updates already applied, so the first update would see ``alpha = 0`` and
    ``b3 = b1`` and the ramp would finish one step late; `pytorch_optimizer.AdEMAMix`,
    which the torch backend uses, counts its own steps from 1. That one-step offset is
    the whole difference between the two implementations, and it is not small where it
    matters: on a cond-1e6 quadratic with a 40-step warmup the two disagreed by **32.3x**
    and agree to **1.002x** once aligned. At the real budget -- a warmup of 10% of the
    first-order iterations -- it is one part in tens of thousands and moves nothing, but a
    short cross-backend check is exactly where the offset dominates, and a short
    cross-backend check is how this project finds defects.
    """
    ramp = optax.linear_schedule(
        init_value=0.0, end_value=_ADEMAMIX_ALPHA, transition_steps=_warmup_steps(cfg)
    )
    return lambda count: ramp(jnp.asarray(count) + 1)


def _b3_warmup(cfg: AxialTrainConfig) -> Callable[[chex.Numeric], jax.Array]:
    """Ramp AdEMAMix's slow-EMA decay from ``b1`` to ``b3`` over the same warmup.

    The one schedule here that optax does **not** ship, and the one that has to be
    hand-written:
    ``exp(ln a ln b / ((1 - s) ln b + s ln a))`` at warmup fraction ``s``. It is even in
    the **half-life**, not in the decay, and the difference is not cosmetic. At the
    midpoint this gives a half-life of 3469 steps -- half of the final 6931, as intended.
    A linear ramp between the same endpoints gives a half-life of **13 steps**, i.e. no
    slow memory at all until the very end of the warmup, which is not a warmup of the
    thing the method depends on.
    """
    warm = _warmup_steps(cfg)
    la, lb = math.log(_ADEMAMIX_B1), math.log(_ADEMAMIX_B3)

    def schedule(count: chex.Numeric) -> jax.Array:
        # `count + 1` for the same reason as `_alpha_warmup`: optax counts updates
        # already applied, the torch twin counts steps taken, and the two must index the
        # same warmup or they are not running the same method.
        s = jnp.minimum((jnp.asarray(count) + 1) / warm, 1.0)
        return jnp.minimum(jnp.exp(la * lb / ((1.0 - s) * lb + s * la)), _ADEMAMIX_B3)

    return schedule


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
