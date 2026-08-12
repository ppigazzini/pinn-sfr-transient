"""Sampling and the training loop.

``Trainer`` owns where the residual is evaluated and how the run is driven. Both
are separate modules in the JAX twin; here they share a class because the sampler
needs the model to place points on the predicted front, and the loop needs
mutable optimiser state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pinn_sfr_transient.axial.config import AxialParams

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.torchpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.torchpinn.model import AxialPinn
from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS
from pinn_sfr_transient.axial.torchpinn.weighting import (
    _bounded_weights,
    _causal_weights,
)

# Candidates drawn per kept point when sampling the level set.
_LEVEL_SET_POOL = 8


class Trainer:
    """Adam (causal weighting + adaptive block weights + RAR) then an L-BFGS polish."""

    def __init__(self, model: AxialPinn, cfg: AxialTrainConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.dev = cfg.device
        n_blocks = model.n_blocks + (1 if cfg.feedback else 0)
        self.block_w = torch.ones(n_blocks, dtype=torch.float64, device=self.dev)
        self.rar = torch.empty(0, 2, dtype=torch.float64, device=self.dev)
        # Pseudo-time stepping state: the anchor is the previous pseudo-step's
        # solution, held fixed while the network takes an implicit step toward it.
        self.pts_anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.pts_dtau = cfg.pts_dtau
        # Explicit generator: collocation draws must not depend on global RNG
        # state, or a run is only reproducible if nothing else touched it.
        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed)
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

    def collocation(self, t_max: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform points over ``(zeta, t_hat)``, clustered early, plus the RAR reservoir."""
        if self.cfg.feedback:
            # Plan A collocates in TIME only: the axial direction is the fixed
            # quadrature the reactivity integral needs (section 3.5a).
            that = self._rand(self.cfg.n_time, 1) * t_max
            early = self._rand(self.cfg.n_time // 2, 1) * 0.4 * t_max
            allt = torch.cat([that, early], dim=0)
            return allt, allt
        n = self.cfg.n_colloc
        pts = self._rand(n, 2)
        pts[:, 1] *= t_max
        early = self._rand(n // 2, 2)
        early[:, 1] *= 0.4 * t_max  # fastest dynamics live early in the window
        parts = [pts, early, *([self.rar] if self.rar.numel() else [])]
        if self.model.use_front and self.cfg.front_frac > 0.0:
            parts.append(self._front_points(int(n * self.cfg.front_frac), t_max))
        elif self.cfg.front_level_set and self.cfg.front_frac > 0.0:
            parts.append(self._level_set_points(int(n * self.cfg.front_frac), t_max))
        allp = torch.cat(parts, dim=0)
        return allp[:, 0:1], allp[:, 1:2]

    @torch.no_grad()
    def _front_points(self, n: int, t_max: float) -> torch.Tensor:
        """Collocation clustered on the predicted front.

        RAR cannot supply these: it samples by residual magnitude, and once the
        void is closed algebraically the field residual is small *everywhere*,
        including across the front. The front's own position is the only signal
        left that says where the interesting 2% of the domain is.
        """
        that = self._rand(n, 1) * t_max
        z_f = self.model.front_position(that)
        spread = 0.05 * torch.randn(n, 1, dtype=torch.float64, device=self.dev, generator=self.gen)
        zeta = (z_f + spread).clamp(0.0, 1.0)
        return torch.cat([zeta, that], dim=1)

    @torch.no_grad()
    def _level_set_points(self, n: int, t_max: float) -> torch.Tensor:
        """Collocation on the saturation level set, found from the model's own ``T_c``.

        Rejection-sample: draw a candidate cloud, evaluate ``T_c``, keep the ``n``
        points closest to ``T_sat + dT_superheat``. That is importance sampling by
        the front indicator rather than by residual magnitude, which is what RAR
        does and what cannot work here — after the algebraic closure the residual
        is small everywhere, including across the front.

        The point is to fix a **measure**, not to add a term. The loss averages
        over the domain and the front is a few percent of it, so the objective
        under-weights the front no matter how long training runs; more iterations
        then converge more precisely to a minimiser whose peak is wrong.
        """
        cand = self._rand(n * _LEVEL_SET_POOL, 2)
        cand[:, 1] *= t_max
        state = self.model.normalised_state(cand[:, 0:1], cand[:, 1:2])
        T_c = self.model.to_physical(state)[3]
        idx = torch.topk((T_c - self.T_boil).abs().squeeze(1), n, largest=False).indices
        return cand[idx]

    def _anchor_points(self, t_max: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a genuine ``(zeta, t_hat)`` pair for the pseudo-time anchor.

        Under Plan A :meth:`collocation` returns *times* in both slots, because
        the residual there needs only the time axis — the axial direction is the
        fixed quadrature. Feeding that pair straight to ``normalised_state``
        evaluated the ansatz with times standing in for ``zeta``, so the proximal
        term pulled toward a state on the wrong manifold. Rebuild the tensor grid
        the Plan A state is actually defined on.
        """
        zeta, that = self.collocation(t_max)
        if not self.cfg.feedback:
            return zeta, that
        n_z = self.model.zeta_q.shape[0]
        return self.model.zeta_q.repeat(that.shape[0], 1), that.repeat_interleave(n_z, dim=0)

    def _pointwise(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(zeta, that)
        return sum(self.block_w[k] * blocks[k] for k in range(len(blocks)))

    def pseudo_time_step(self, t_max: float = 1.0) -> None:
        """Re-anchor the pseudo-time step and relax it [arXiv:2604.23528].

        Plain residual minimisation is free to sit in any low-residual basin,
        including ones no physical solution occupies — the paper's central point,
        and something this model has already been bitten by once (``T_f`` driven
        negative while the loss fell). Pseudo-time stepping instead solves the
        implicit-Euler problem

        ``(u - u_prev) / dtau + R(u) = 0``

        which adds a proximal pull toward the previous iterate. A jump to a
        distant spurious basin costs ``||u - u_prev||^2 / dtau``, so the optimiser
        has to *walk* to a solution rather than teleport to one. ``dtau`` grows
        each step, so the anchor relaxes and the limit recovers ordinary residual
        minimisation.

        The anchor points are **resampled** at every step: the paper is explicit
        that pseudo-time stepping and resampling work together, since a fixed
        anchor set is one more thing to overfit.
        """
        zeta, that = self._anchor_points(t_max)
        with torch.no_grad():
            state = self.model.normalised_state(zeta, that).detach()
        self.pts_anchor = (zeta.detach(), that.detach(), state)
        self.pts_dtau *= self.cfg.pts_growth

    def _pts_penalty(self) -> torch.Tensor:
        """Proximal term ``||u - u_prev||^2 / dtau``; zero when the anchor is unset."""
        if self.pts_anchor is None:
            return torch.zeros((), dtype=torch.float64, device=self.dev)
        zeta, that, prev = self.pts_anchor
        now = self.model.normalised_state(zeta, that)
        return (now - prev).pow(2).mean() / self.pts_dtau

    def causal_loss(
        self, zeta: torch.Tensor, that: torch.Tensor, t_max: float = 1.0
    ) -> torch.Tensor:
        """Time-chunked loss with causal weights [Wang, Sankaran & Perdikaris 2024].

        The chunks span the *current window*, not the whole horizon, so causal
        weighting keeps its resolution as the window grows.
        """
        e = self._pointwise(zeta, that)
        chunks = self.cfg.causal_chunks
        idx = torch.clamp((that.reshape(-1) / max(t_max, 1e-12) * chunks).long(), max=chunks - 1)
        # Scatter-reduce, not a Python loop over boolean masks. The loop cost a
        # graph break per chunk under `torch.compile` -- `bool(mask.any())` is a
        # host synchronisation and `e[mask]` a data-dependent shape -- so the whole
        # step fell back to eager. This is also exactly what the JAX twin does with
        # `bincount`, so the two reductions are now the same operation.
        sums = torch.zeros(chunks, dtype=e.dtype, device=e.device).index_add_(0, idx, e)
        counts = torch.zeros(chunks, dtype=e.dtype, device=e.device).index_add_(
            0, idx, torch.ones_like(e)
        )
        losses = sums / counts.clamp(min=1.0)
        with torch.no_grad():
            w = _causal_weights(losses, self.cfg.causal_eps)
        return (w * losses).mean() + self._pts_penalty()

    def update_block_weights(self, zeta: torch.Tensor, that: torch.Tensor) -> None:
        """Balance the blocks by gradient norm [Wang, Teng & Perdikaris 2021], **bounded**.

        The scheme sets ``lambda_k = mean(g)/g_k``, so a block whose gradient
        falls as it is fitted earns an ever-larger weight — a positive feedback
        with nothing to stop it. Measured over three seeds, the weights reached
        3.1e5 to 6.2e6 on ``T_f`` while the void block pinned at 0.451, a spread
        of up to 5e6, and the run-to-run spread in the ``T_f`` error was 10.4x.
        Bounding the ratio removes both: every field improves and the seed spread
        collapses to 1.1-1.2x. The full table is in ``docs/axial_nn.md``
        section 7.2.

        Only *ratios* between blocks can matter — Adam is scale-invariant to a
        global factor, which is exactly the argument REPORT-01 D39 uses to show
        fixed per-equation scaling is a no-op — so the target is renormalised to
        unit geometric mean before clamping. That leaves the relative balance
        untouched and bounds only the spread.
        """
        blocks = self._blocks(zeta, that)
        params = [q for q in self.model.parameters() if q.requires_grad]
        norms = []
        for b in blocks:
            grads = torch.autograd.grad(b.mean(), params, retain_graph=True, allow_unused=True)
            sq = sum(
                (g.pow(2).sum() for g in grads if g is not None),
                start=torch.zeros((), device=self.dev),
            )
            norms.append(torch.sqrt(sq + 1e-30))
        gn = torch.stack(norms)
        with torch.no_grad():
            target = _bounded_weights(gn.mean() / (gn + 1e-12), self.cfg.weight_max_ratio)
            m = self.cfg.weight_momentum
            self.block_w = m * self.block_w + (1.0 - m) * target

    @torch.no_grad()
    def rar_refine(self, t_max: float = 1.0) -> None:
        """Append the worst-residual candidates to the reservoir [Wu et al. 2023]."""
        if self.cfg.feedback:
            return  # Plan A's collocation is a tensor grid; RAR would break the quadrature
        pool = self._rand(self.cfg.rar_pool, 2)
        pool[:, 1] *= t_max
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
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.adam_iters), eta_min=cfg.lr * 0.1
        )
        for it in range(cfg.adam_iters):
            # Time-window curriculum: the horizon opens from `1/n_windows` to 1
            # over training, so the network solves a short transient first and
            # extends it. Causal weighting *re-weights* a globally posed problem;
            # windowing makes the problem itself local in time, which is the
            # stronger form of the same idea. With `n_windows = 1` this is a
            # no-op and the schedule is the original one.
            stage = min(int(it / cfg.adam_iters * cfg.n_windows) + 1, cfg.n_windows)
            t_max = stage / cfg.n_windows
            # Skip the gradient-norm pass entirely when weighting is off: it costs
            # one backward per block and the answer is known to be ones.
            if cfg.weight_max_ratio > 1.0 and it and it % cfg.weight_update_every == 0:
                self.update_block_weights(*self.collocation(t_max))
            if it and it % cfg.rar_every == 0:
                self.rar_refine(t_max)
            if cfg.pts_every and it % cfg.pts_every == 0:
                self.pseudo_time_step(t_max)
            opt.zero_grad()
            loss = self.causal_loss(*self.collocation(t_max), t_max)
            loss.backward()
            opt.step()
            sched.step()
            if verbose and it % cfg.log_every == 0:
                w = [f"{v:.1e}" for v in self.block_w.tolist()]
                print(f"[adam {it:6d}] t<={t_max:.2f} loss={loss.item():.3e} w=[{','.join(w)}]")
        if cfg.lbfgs_iters > 0:
            self._lbfgs(verbose=verbose, on_checkpoint=on_checkpoint)
        return self.model

    def _polish_stages(self) -> list[tuple[int, bool]]:
        """Return ``(iterations, freeze the encoder)`` for each block of the polish.

        One block unless ``freeze_after`` splits it, in which case the projection is
        trainable for the first ``freeze_after`` iterations and held for the rest -- on
        the same collocation set, since the switch already discards the curvature history
        and redrawing would confound a restart with a change of objective.
        """
        cfg = self.cfg
        if cfg.freeze_after <= 0:
            return [(cfg.lbfgs_iters, cfg.freeze_encoder)]
        if cfg.polish_refresh > 0:
            msg = "freeze_after and polish_refresh both set; they schedule the same stage"
            raise ValueError(msg)
        n1 = min(cfg.freeze_after, cfg.lbfgs_iters)
        return [(n1, False), (cfg.lbfgs_iters - n1, True)]

    def _freeze_encoder(self, *, force: bool = False) -> list:
        """Hold the Fourier-to-trunk projection fixed; return the parameters frozen.

        The JAX twin does this by partitioning that layer out of the optimised pytree.
        Here it is ``requires_grad_(False)`` -- a parameter with no gradient contributes
        zeros to every curvature pair, so the space L-BFGS searches is the same one --
        and ``_lbfgs`` restores it afterwards so the model is left exactly as found
        whatever the divergence guard decides.
        """
        if not ((force or self.cfg.freeze_encoder) and self.model.embed is not None):
            return []
        first = next(m for m in self.model.net.modules() if isinstance(m, torch.nn.Linear))
        frozen = [q for q in first.parameters() if q.requires_grad]
        for q in frozen:
            q.requires_grad_(False)
        return frozen

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

    def _run_stages(self, closure, *, want_snaps: bool, verbose: bool) -> list:  # noqa: ANN001
        """Step the polish through its stages and segments, returning the snapshots.

        One optimiser per *stage*, stepped once per *segment*. ``torch.optim.LBFGS``
        keeps its history in ``state``, so repeated ``.step()`` calls continue the same
        solve and a segment boundary costs a copy rather than a restart. A **new**
        optimiser at the freeze switch is deliberate: the history there refers to
        coordinates the next stage holds fixed.
        """
        cps = sorted(set(self.cfg.polish_checkpoints))
        snaps: list = []
        done = 0
        for iters, freeze in self._polish_stages():
            if iters <= 0:
                continue
            frozen = self._freeze_encoder(force=freeze)
            stops = [b - done for b in cps if done < b < done + iters]
            opt, run = self._make_opt(iters), 0
            for seg in [*stops, iters]:  # the last entry ends the stage, requested or not
                # BOTH limits, per segment. `torch.optim.LBFGS` derives `max_eval` from
                # `max_iter` at construction and enforces it PER `.step()` call, so
                # setting only `max_iter` leaves the segment capped by the constructor's
                # evaluation budget -- which silently truncated every segment to one
                # function evaluation and made a checkpointed run a different run.
                opt.param_groups[0]["max_iter"] = seg - run
                opt.param_groups[0]["max_eval"] = (seg - run) * 5 // 4 + 1
                opt.step(closure)
                run = seg
                # Only what was asked for -- a stage boundary is not a checkpoint.
                if want_snaps and done + run in cps:
                    state = [q.detach().clone() for q in self.model.parameters()]
                    snaps.append((done + run, state))
            done += iters
            for q in frozen:  # leave the model as it was found, whatever the guard decides
                q.requires_grad_(True)
            if verbose and self.cfg.freeze_after > 0:
                print(
                    f"[lbfgs] {iters} iterations, encoder {'frozen' if freeze else 'free'}",
                    flush=True,
                )
        return snaps

    def _lbfgs(
        self, *, verbose: bool, on_checkpoint: Callable[[int, AxialPinn], None] | None = None
    ) -> None:
        """Quasi-Newton polish on a fixed collocation set, with a divergence guard.

        One block, or two when ``freeze_after`` splits it. The guard spans the whole
        stage rather than each block: the question a caller cares about is whether the
        polish as a whole improved on what Adam handed it.
        """
        zeta, that = self.collocation()
        before = self.causal_loss(zeta, that).item()
        snapshot = [q.detach().clone() for q in self.model.parameters()]

        def closure() -> torch.Tensor:
            for q in self.model.parameters():
                q.grad = None
            loss = self.causal_loss(zeta, that)
            loss.backward()
            return loss

        snaps = self._run_stages(closure, want_snaps=on_checkpoint is not None, verbose=verbose)
        after = self.causal_loss(zeta, that).item()
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
