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
from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.optimizers import minimize as ssbfgs_minimize
from pinn_sfr_transient.axial.jaxpinn.residuals import n_field_blocks
from pinn_sfr_transient.axial.jaxpinn.samplers import _collocation, _merge, _rar_points
from pinn_sfr_transient.axial.jaxpinn.weighting import (
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
    if verbose and cfg.log_every:
        cadences.append(cfg.log_every - it % cfg.log_every)
    if checkpointing and cfg.adam_checkpoint_every:
        cadences.append(cfg.adam_checkpoint_every - it % cfg.adam_checkpoint_every)
    return max(1, min([left, *cadences]))


def train(
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
    # THE REFERENCE IMPLEMENTATION'S DERIVATION, EXACTLY. `k_model` is `split(key)[0]`
    # and the fixed collocation set comes off `k_points = split(key)[1]`; the first-order
    # stream is `fold_in(key, 1)` rather than a wider split, because splitting differently
    # moves both of the others.
    #
    # This used to read `key, k_model = jax.random.split(key)`, taking the WRONG HALF for
    # the model and then deriving every draw from the other one. Same seed, different
    # weights, different points: `seed = 0` here was not `seed = 0` there, so no number
    # from this backend could be compared with the reference's, and the seed spread on
    # this problem reaches 12.5x. The reference measures the key derivation alone at
    # 0.0314 s -> 0.0103 s of boiling-onset error.
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_points = jax.random.split(key)
    k_step = jax.random.fold_in(key, 1)
    model = AxialPinn(cfg, k_model)

    w = jnp.ones(n_field_blocks(cfg) + (1 if cfg.feedback else 0))
    optimizer = _first_order(cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model: AxialPinn, opt_state: optax.OptState, pts: tuple, w: jax.Array) -> tuple:
        loss, grads = eqx.filter_value_and_grad(lambda m: causal_loss(m, p, cfg, pts, w))(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(model, updates), opt_state, loss

    key = k_step  # the first-order stream; the polish takes `k_points` untouched
    rar: tuple | None = None
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
        n: jax.Array,
    ) -> tuple:
        def body(_i: jax.Array, carry: tuple) -> tuple:
            params, opt_state, key, _loss = carry
            model = eqx.combine(params, static)
            key, ck = jax.random.split(key)
            pts = _merge(_collocation(p, a_cfg, ck), rar, feedback=cfg.feedback)
            loss, grads = eqx.filter_value_and_grad(lambda m: causal_loss(m, p, cfg, pts, w))(model)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_inexact_array), opt_state, params
            )
            return optax.apply_updates(params, updates), opt_state, key, loss

        return jax.lax.fori_loop(0, n, body, (params, opt_state, key, jnp.zeros(())))

    params = eqx.filter(model, eqx.is_inexact_array)
    it = 0
    loss = jnp.zeros(())
    while it < cfg.adam_iters:
        model = eqx.combine(params, static)
        if cfg.rar_every and it and it % cfg.rar_every == 0:
            key, rk = jax.random.split(key)
            rar = _rar_points(model, p, cfg, rk, w)
        n = _next_boundary(cfg, it, verbose=verbose, checkpointing=on_checkpoint is not None)
        params, opt_state, key, loss = run_block(params, opt_state, key, w, rar, jnp.asarray(n))
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

    **Driven by optax's own count, which starts at 0**, exactly as the companion
    implementation does. `pytorch_optimizer.AdEMAMix` counts its own steps from 1, so the
    two libraries index their warmups one step apart. That is each library's convention,
    not a defect in either: what is set externally is the same total warmup length, and
    over one it is one part in the warmup. Do not shift either to match the other --
    a `count + 1` here makes this arm stop reproducing the reference it is meant to.

    The offset only looks large at a warmup so short that one step is a large fraction of
    it: on a cond-1e6 quadratic with a 40-step warmup the two implementations differ by
    32x. At the 100 000-step warmup a 1M-iteration arm uses it is 1e-5. A cross-library
    check must therefore compare at a realistic warmup, or it measures the convention
    rather than the method.
    """
    return optax.linear_schedule(
        init_value=0.0, end_value=_ADEMAMIX_ALPHA, transition_steps=_warmup_steps(cfg)
    )


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
        # optax's own count, from 0, as the companion implementation has it. See
        # `_alpha_warmup` on why this is not shifted to match the torch twin.
        s = jnp.minimum(jnp.asarray(count) / warm, 1.0)
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


def _polish_spec(cfg: AxialTrainConfig, model: AxialPinn):  # noqa: ANN202, ARG001
    """Which parameters the quasi-Newton stage may move: all of them.

    ``freeze_encoder`` and ``freeze_after`` used to hold the Fourier-to-trunk projection
    fixed for part or all of the stage. Both are retired: freezing throughout was measured
    worse at either Adam budget (7.5.32) and freezing part-way was **2.5x worse** (7.5.41).
    """
    return jax.tree_util.tree_map(eqx.is_inexact_array, model)


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
    """Run the quasi-Newton stage: one solve, on one fixed collocation set.

    One path, as the reference implementation has. ``polish_refresh`` used to redraw the
    set in blocks and restart the optimiser; 7.5.37 measured that the blocked restart
    **hurts**, and it is retired along with the two freeze schedules.
    """
    q_cfg = replace(cfg, n_colloc=cfg.polish_colloc) if cfg.polish_colloc else cfg
    pts = _merge(_collocation(p, q_cfg, key), rar, feedback=cfg.feedback)
    return _lbfgs_polish(
        model, p, cfg, pts, w, tuple(cfg.polish_checkpoints), on_checkpoint, verbose=verbose
    )


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


def _segmented(  # noqa: ANN202, PLR0913, PLR0917 - one loop's state, flat is clearer
    body,  # noqa: ANN001
    params,  # noqa: ANN001
    state,  # noqa: ANN001
    total: int,
    checkpoints: tuple[int, ...],
    emit=None,  # noqa: ANN001
):
    """Run ``total`` iterations of ``body``, emitting the parameters at each stop.

    The optimiser is **not** restarted at a stop: ``state`` is carried across the segment
    boundary, so the trajectory is the one a single uninterrupted `fori_loop` would take
    and a checkpoint costs one copy. Restarting instead would turn every checkpoint into
    a blocked restart, which section 7.5.37 measured at 1.5x worse -- the intermediate
    rows would then be measuring the checkpointing, not the budget.

    **``emit`` is called the moment the rung is reached**, not after the stage finishes.
    A checkpoint held until the end is worth nothing to a run that is stopped, and this
    stage is hours long. Only the rungs asked for are emitted: a segment boundary that
    nobody requested is not a checkpoint.
    """
    wanted = {b for b in checkpoints if 0 < b <= total}
    snaps, done = [], 0
    for b in [*sorted(wanted - {total}), total]:
        params, state = jax.lax.fori_loop(0, b - done, body, (params, state))
        done = b
        if done in wanted:
            snaps.append((done, params))
            if emit is not None:
                emit(done, params)
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

    emit = None if on_checkpoint is None else lambda n, q: on_checkpoint(n, eqx.combine(q, static))
    params, _ = _segmented(body, params, state, cfg.lbfgs_iters, checkpoints, emit)
    after = float(loss_fn(params))
    # Same divergence guard as the torch twin: a bad line-search step can only cost time,
    # never accuracy. It governs what the CALLER is handed and nothing else -- the rungs
    # were emitted as they were reached, and a rung is a fact about the run. Discarding
    # them here, as this used to, threw away the evidence that would show where the
    # divergence began.
    if not np.isfinite(after) or after > before:
        if verbose:
            print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        return model
    if verbose:
        print(f"[lbfgs done] loss={after:.3e}")
    return eqx.combine(params, static)
