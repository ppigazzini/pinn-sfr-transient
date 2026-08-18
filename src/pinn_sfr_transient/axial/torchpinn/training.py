"""Sampling and the training loop.

``Trainer`` owns where the residual is evaluated and how the run is driven. Both
are separate modules in the JAX twin; here they share a class because the sampler
needs the model to place points on the predicted front, and the loop needs
mutable optimiser state.
"""

import contextlib
from typing import TYPE_CHECKING

try:
    import torch  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pinn_sfr_transient.axial.config import AxialParams

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.torchpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.torchpinn.model import AxialPinn
from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS

# Candidates drawn per kept point when sampling the level set.
_LEVEL_SET_POOL = 8

#: AdEMAMix's final slow-EMA weight. `optax.contrib.ademamix`'s default and
#: `pytorch_optimizer.AdEMAMix`'s, restated because the warmup is configured against it.
_ADEMAMIX_ALPHA = 5.0


class Trainer:
    """Adam (causal weighting + adaptive block weights + RAR) then an L-BFGS polish."""

    def __init__(self, model: AxialPinn, cfg: AxialTrainConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.dev = cfg.device
        self.rar = torch.empty(0, 2, dtype=torch.float64, device=self.dev)
        # Explicit generator: collocation draws must not depend on global RNG
        # state, or a run is only reproducible if nothing else touched it.
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed)
        self._compiled: Callable[..., torch.Tensor] | None = None
        p = model.p
        self.T_boil = float(sodium.saturation_temperature(p.p_system) + p.dT_superheat)

    def _rand(self, *shape: int) -> torch.Tensor:
        """Uniform draw from this trainer's own generator, never the global one."""
        return torch.rand(*shape, dtype=torch.float64, device=self.dev, generator=self.gen)

    def _blocks(self, zeta: torch.Tensor, that: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Residual blocks for whichever plan is active."""
        if self.cfg.feedback:
            return self.model.closed_loop_blocks(that)
        return self.model.residual_blocks(zeta, that)

    def collocation(self, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform points over ``(zeta, t_hat)``, plus the RAR set.

        ``n`` overrides ``cfg.n_colloc`` for one draw, which is how the two stages get
        their own counts: ``adam_colloc`` for the first-order loop and ``polish_colloc``
        for the quasi-Newton one. Both were fields the torch backend accepted and then
        ignored -- ``n_colloc`` drove every draw, so an arm configured for a small Adam
        batch silently ran full batch, which is a different algorithm.

        **No early-time cluster.** It was drawn unconditionally here and is retired; see
        the JAX twin's sampler for the measurement that retired it.
        """
        if self.cfg.feedback:
            # Plan A collocates in TIME only: the axial direction is the fixed
            # quadrature the reactivity integral needs (section 3.5a).
            that = self._rand(self.cfg.n_time, 1)
            return that, that
        n = n or self.cfg.n_colloc
        pts = self._rand(n, 2)
        parts = [pts]
        if self.rar.numel():
            parts.append(self.rar)
        allp = torch.cat(parts, dim=0)
        return allp[:, 0:1], allp[:, 1:2]

    def _pointwise(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(zeta, that)
        return sum(blocks)

    def causal_loss(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        """Squared residual, summed over the equations and averaged over time windows.

        The average is per time window and then across windows, so sampling density and
        loss weighting stay independent. The blocks enter with equal weight: the variable
        scaling has already put them on a common magnitude.

        Scatter-reduce rather than a Python loop over masks -- the loop cost a graph break
        per chunk under `torch.compile`, and this is the same operation the JAX twin does
        with `bincount`.
        """
        e = sum(self._blocks(zeta, that))
        chunks = self.cfg.causal_chunks
        idx = torch.clamp((that.reshape(-1) * chunks).long(), max=chunks - 1)
        sums = torch.zeros(chunks, dtype=e.dtype, device=e.device).index_add_(0, idx, e)
        counts = torch.zeros(chunks, dtype=e.dtype, device=e.device).index_add_(
            0, idx, torch.ones_like(e)
        )
        return (sums / counts.clamp(min=1.0)).mean()

    @torch.no_grad()
    def rar_refine(self) -> None:
        """Append the worst-residual candidates to the reservoir [Wu et al. 2023]."""
        if self.cfg.feedback:
            return  # Plan A's collocation is a tensor grid; RAR would break the quadrature
        pool = self._rand(self.cfg.rar_pool, 2)
        e = self._pointwise(pool[:, 0:1], pool[:, 1:2])
        top = torch.topk(e, min(self.cfg.rar_add, e.numel())).indices
        self.rar = torch.cat([self.rar, pool[top]], dim=0)[-self.cfg.rar_cap :]

    def _reset_rar(self) -> None:
        """Drop the reservoir when the window grows; its points are stale."""
        self.rar = torch.empty(0, 2, dtype=torch.float64, device=self.dev)

    def train(
        self,
        *,
        verbose: bool = True,
        on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
    ) -> AxialPinn:
        """Run the full schedule and return the trained model.

        ``on_checkpoint`` receives ``(cumulative quasi-Newton iterations, model)`` at each
        entry of ``cfg.polish_checkpoints``, so one run can be scored at several budgets
        instead of being re-run once per budget. The JAX twin takes the same argument.
        """
        cfg = self.cfg
        # `foreach=True` explicitly. PyTorch's auto-selection does NOT enable it on
        # CPU -- `_default_to_fused_or_foreach` returns `(False, False)` there, so
        # leaving it unset silently takes the single-tensor for-loop path on the only
        # device this project runs on. Measured on this model: 1.50x on the optimiser
        # step alone and **1.061x end-to-end**, the gap being that the step is mostly
        # forward-plus-backward through the residuals, not the update.
        #
        # It is free rather than a trade: horizontal fusion here leaves the trained
        # fields **bitwise identical** (checked at 200 Adam iterations), so unlike a
        # thread-count change it does not move a published number.
        opt = self._first_order()
        # Schedule-free REPLACES the learning-rate schedule with an averaging sequence
        # and warms the step size internally, so it runs at a constant rate. Composing
        # it with the cosine would measure a hybrid nobody proposed; the JAX twin makes
        # the same exception for the same reason.
        sched = None if cfg.first_order == "schedulefree" else self._lr_schedule(opt)
        # The loss, compiled or not -- shared with the quasi-Newton stage, which needs it
        # more. `fullgraph=True` because a partial compile here is
        # worth ~nothing and hides the reason: the residual stack broke into eight
        # graphs until `_backend.xp` stopped sniffing `__module__` on Python scalars and
        # `state_and_grads` stopped marking its inputs `requires_grad`. A silent
        # fallback to eager would have left both defects in place looking like a 3%
        # regression. `dynamic=False` is NOT tuning: automatic dynamic shapes -- which
        # switch on by themselves at the second distinct collocation count, and RAR
        # produces one every `rar_every` -- make `torch._make_dual` fail on a symbolic
        # size, so forward-mode AD and dynamic shapes cannot be combined in 2.13. Static
        # shapes recompile per size instead, which is the cost the config comment quotes.
        loss_fn = self._loss_fn()
        n_adam = cfg.adam_colloc
        for it in range(cfg.adam_iters):
            if cfg.rar_every and it and it % cfg.rar_every == 0:
                self.rar_refine()
            opt.zero_grad()
            loss = loss_fn(*self.collocation(n_adam))
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if verbose and it % cfg.log_every == 0:
                print(f"[adam {it:6d}] loss={loss.item():.3e}")
            # First-order checkpoints, on the same cadence and the same 1-indexed count
            # as the JAX twin. Without this a pure first-order arm (`lbfgs_iters = 0`)
            # emitted nothing at all -- the only callback site was the polish -- so a
            # ten-rung budget ladder cost ten runs of the longest rung.
            every = cfg.adam_checkpoint_every
            if on_checkpoint is not None and every and (it + 1) % every == 0:
                with self._reportable(opt):
                    on_checkpoint(it + 1, self.model)
        # Schedule-free keeps two sequences and the gradients were taken at `y`; the
        # iterate to REPORT is the running average `x`. Swap before anything downstream
        # sees the model -- the polish, the caller, `predict`.
        self._to_reportable(opt)
        if cfg.lbfgs_iters > 0:
            self._lbfgs(verbose=verbose, on_checkpoint=on_checkpoint)
        return self.model

    def _loss_fn(self) -> Callable[..., torch.Tensor]:
        """Return the loss, compiled once per Trainer when ``cfg.compile`` is set.

        **Both stages use it.** The quasi-Newton polish evaluates this loss far more
        often than the first-order loop does -- once per line-search probe, several times
        per iteration -- and it used to call `causal_loss` directly, so `compile` did
        nothing for it and nothing at all for an `adam_iters = 0` arm. That is most of
        this project's funded configurations.

        The optimiser itself cannot be compiled and is not: `torch.optim.LBFGS` is
        wrapped in `torch._dynamo.disable` upstream. It does not need to be -- the step
        is a two-loop recursion over 17k parameters and the cost is the residual forward
        and backward, which is exactly what this compiles.

        The polish is the *easier* case of the two. It runs on a FIXED collocation set,
        so there is one input shape and one compilation; the first-order loop redraws and
        grows a RAR reservoir, which is where the per-shape recompiles come from.
        """
        if self._compiled is None:
            self._compiled = (
                torch.compile(self.causal_loss, fullgraph=True, dynamic=False)
                if self.cfg.compile
                else self.causal_loss
            )
        return self._compiled

    def _first_order(self) -> torch.optim.Optimizer:
        """Build the first-order optimiser named by ``cfg.first_order``.

        ``adam`` is `torch.optim.Adam`. The other two are **not in `torch.optim`** --
        AdEMAMix is pytorch/pytorch#135609 and still a proposal, and schedule-free has
        never been proposed -- so they come from `pytorch_optimizer`, which is the same
        choice the JAX twin makes in reaching for `optax.contrib` rather than writing
        them. Hand-rolling either is what AGENTS.md forbids; taking a maintained
        implementation is not.

        The hyper-parameters are matched to the JAX twin's deliberately and the match was
        checked rather than assumed, because an unset argument across two implementations
        of one algorithm is this project's most expensive recurring defect (7.5.17):

        * `AdEMAMix(betas=(0.9, 0.999, 0.9999), alpha=5.0, eps=1e-8, weight_decay=0.0)`
          are `optax.contrib.ademamix`'s defaults exactly.
        * `t_alpha_beta3` is the warmup length, and both libraries ramp identically:
          `alpha` linearly as ``min(step alpha / t, alpha)``, and `b3` through
          ``exp(ln b1 ln b3 / ((1 - s) ln b3 + s ln b1))``. The JAX side builds those two
          schedules by hand from `optax.linear_schedule`; here they are the library's own
          `schedule_alpha` and `schedule_beta3`, and the formulae agree term for term.
        * `ScheduleFreeAdamW(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
          weight_lr_power=2.0)` are `optax.contrib.schedule_free_adamw`'s defaults exactly.

        `weight_decay` stays at zero on both arms. AdamW with no decay is Adam plus the
        schedule-free averaging, which is the one difference that arm is testing.
        """
        cfg = self.cfg
        params = self.model.parameters()
        if cfg.first_order == "adam":
            # `foreach=True` for the same reason as the comment in `train`; the two
            # library optimisers do their own fusion and take no such argument.
            return torch.optim.Adam(params, lr=cfg.lr, foreach=True)
        try:
            import pytorch_optimizer as po  # noqa: PLC0415 - optional, only these arms
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            msg = (
                f"first_order={cfg.first_order!r} needs `pytorch-optimizer`, which ships "
                f"with the torch extras: `uv sync --extra torch-cpu`. Neither AdEMAMix nor "
                f"schedule-free is in torch.optim."
            )
            raise SystemExit(msg) from exc
        if cfg.first_order == "ademamix":
            return po.AdEMAMix(
                params, lr=cfg.lr, alpha=_ADEMAMIX_ALPHA, t_alpha_beta3=self._warmup_steps()
            )
        if cfg.first_order == "schedulefree":
            if cfg.lr_warmup:
                # Refused, not ignored. Schedule-free warms the step size internally over
                # the same `sf_warmup_frac`, so an external warmup would either be dropped
                # or measure a hybrid nobody proposed. The JAX twin refuses this too.
                msg = (
                    "lr_warmup and first_order='schedulefree' both schedule the step size; "
                    "schedule-free warms up internally over sf_warmup_frac. Pick one."
                )
                raise ValueError(msg)
            return po.ScheduleFreeAdamW(params, lr=cfg.lr, warmup_steps=self._warmup_steps())
        msg = f"unknown first_order={cfg.first_order!r}; expected adam, ademamix or schedulefree"
        raise ValueError(msg)

    def _warmup_steps(self) -> int:
        """Warmup length in steps, as a fraction of the first-order budget.

        A fraction rather than a count so it scales with the budget instead of silently
        becoming the whole run at a short one. Same expression as the JAX twin's.
        """
        return max(1, int(self.cfg.sf_warmup_frac * self.cfg.adam_iters))

    def _to_reportable(self, opt: torch.optim.Optimizer) -> None:
        """Put the model into the iterate that should be REPORTED, in place.

        Schedule-free carries two sequences: gradients are evaluated at ``y``, which is
        what the parameters hold while training, and the iterate to report is the running
        average ``x``. `pytorch_optimizer` swaps between them with `train()`/`eval()`.
        Everything downstream -- the polish, the caller, `predict`, a checkpoint -- must
        see ``x``, and the JAX twin converts at exactly the same points. Reporting ``y``
        is a silent accuracy loss that no test of the loss would catch.

        A no-op for every other optimiser.
        """
        if hasattr(opt, "eval"):
            opt.eval()

    @contextlib.contextmanager
    def _reportable(self, opt: torch.optim.Optimizer) -> Iterator[None]:
        """Hold the model at the reportable iterate, then hand it back to training."""
        self._to_reportable(opt)
        try:
            yield
        finally:
            if hasattr(opt, "train"):
                opt.train()

    def _lr_schedule(self, opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """Cosine decay, optionally with a linear warmup in front of it.

        Composed from ``LinearLR`` and ``CosineAnnealingLR`` through ``SequentialLR``,
        which is torch's shipped equivalent of ``optax.warmup_cosine_decay_schedule`` --
        the JAX twin's `_lr_schedule`. Both run the cosine over ``adam_iters - warmup``
        down to ``0.1 lr``, so the two curves agree away from the first step.

        They cannot agree *during* it: optax starts the ramp at exactly zero, and torch's
        ``start_factor`` must be positive. ``1 / warmup`` makes the first step the first
        rung of the same ladder rather than a no-op, and both reach the peak on the same
        step; the ramps differ by at most ``lr / warmup`` -- at the first step, decaying
        linearly to nothing. Measured over 100 iterations at ``lr = 1e-3``: 1.0e-04
        during the warmup and **9.4e-11 after it**.

        ``lr_warmup`` was a field this backend accepted and ignored -- the schedule was
        unconditionally plain cosine.
        """
        cfg = self.cfg
        total = max(1, cfg.adam_iters)
        eta_min = cfg.lr * 0.1
        if not cfg.lr_warmup:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=eta_min)
        warm = max(1, int(cfg.sf_warmup_frac * cfg.adam_iters))
        return torch.optim.lr_scheduler.SequentialLR(
            opt,
            [
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1.0 / warm, end_factor=1.0, total_iters=warm
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total - warm), eta_min=eta_min
                ),
            ],
            milestones=[warm],
        )

    def _emit_checkpoint(
        self, cb: Callable[[int, AxialPinn], None] | None, n: int, saved: list
    ) -> None:
        """Hand the caller the model as it stood at iteration ``n``, then restore it.

        The JAX twin can pass a snapshot directly because its models are immutable
        pytrees. Here the parameters are mutated in place, so the model is rewound,
        handed over, and put back -- the callback sees the intermediate state and the
        caller's model is left exactly as the polish finished it.
        """
        if cb is None:
            return
        current = [q.detach().clone() for q in self.model.parameters()]
        with torch.no_grad():
            for q, old in zip(self.model.parameters(), saved, strict=True):
                q.copy_(old)
        cb(n, self.model)
        with torch.no_grad():
            for q, new in zip(self.model.parameters(), current, strict=True):
                q.copy_(new)

    def _make_opt(self, iters: int) -> torch.optim.Optimizer:
        """Build the quasi-Newton optimiser for a block of ``iters`` iterations."""
        if self.cfg.optimizer in ("ssbfgs", "lbfgs-shared", "ssbroyden"):
            return SelfScaledLBFGS(
                self.model.parameters(),
                max_iter=iters,
                history_size=self.cfg.lbfgs_history,
                self_scale=self.cfg.optimizer in ("ssbfgs", "ssbroyden"),
                # SSBroyden is the self-scaled Broyden class at the midpoint of the
                # family. `phi = 0` would be SSBFGS exactly, so 0.5 is what makes it
                # a distinct arm rather than a rename.
                broyden_phi=0.5 if self.cfg.optimizer == "ssbroyden" else 0.0,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            )
        return torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=iters,
            history_size=self.cfg.lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

    def _run_stages(self, closure, *, want_snaps: bool) -> list:  # noqa: ANN001
        """Step the polish, snapshotting at each requested budget. One solve, one set.

        ``torch.optim.LBFGS`` keeps its curvature history in ``state``, so repeated
        ``.step()`` calls continue the same solve and a checkpoint costs a copy rather
        than a restart -- the rows are one trajectory, not a ladder of short runs.
        """
        cps = sorted(set(self.cfg.polish_checkpoints))
        snaps: list = []
        iters = self.cfg.lbfgs_iters
        opt, run = self._make_opt(iters), 0
        for seg in [*[b for b in cps if 0 < b < iters], iters]:
            # BOTH limits, per segment. `torch.optim.LBFGS` derives `max_eval` from
            # `max_iter` at construction and enforces it PER `.step()` call, so setting
            # only `max_iter` leaves the segment capped by the constructor's evaluation
            # budget -- which silently truncated every segment to one function evaluation
            # and made a checkpointed run a different run.
            opt.param_groups[0]["max_iter"] = seg - run
            opt.param_groups[0]["max_eval"] = (seg - run) * 5 // 4 + 1
            opt.step(closure)
            run = seg
            if want_snaps and run in cps:
                snaps.append((run, [q.detach().clone() for q in self.model.parameters()]))
        return snaps

    def _lbfgs(
        self, *, verbose: bool, on_checkpoint: Callable[[int, AxialPinn], None] | None = None
    ) -> None:
        """Quasi-Newton polish on a fixed collocation set, with a divergence guard.

        One block, or two when ``freeze_after`` splits it, or many when
        ``polish_refresh`` redraws. The guard spans the whole stage rather than each
        block: the question a caller cares about is whether the polish as a whole
        improved on what Adam handed it.
        """
        zeta, that = self.collocation(n=self.cfg.polish_colloc)
        before = self._loss_fn()(zeta, that).item()
        snapshot = [q.detach().clone() for q in self.model.parameters()]

        def closure() -> torch.Tensor:
            for q in self.model.parameters():
                q.grad = None
            loss = self._loss_fn()(zeta, that)
            loss.backward()
            return loss

        snaps = self._run_stages(closure, want_snaps=on_checkpoint is not None)
        after = self._loss_fn()(zeta, that).item()
        if not np.isfinite(after) or after > before:
            with torch.no_grad():
                for q, saved in zip(self.model.parameters(), snapshot, strict=True):
                    q.copy_(saved)
            if verbose:
                print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
            return
        if verbose:
            print(f"[lbfgs done] loss={after:.3e}")
        for n, saved in snaps:  # only a polish that improved has states worth reporting
            self._emit_checkpoint(on_checkpoint, n, saved)


def train(
    p: AxialParams | None = None,
    cfg: AxialTrainConfig | None = None,
    *,
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
) -> AxialPinn:
    """Build and train the axial PINN."""
    p = p or AxialParams()
    cfg = cfg or AxialTrainConfig()
    return Trainer(AxialPinn(p, cfg), cfg).train(on_checkpoint=on_checkpoint)
