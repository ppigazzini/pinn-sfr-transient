"""Physics-Informed Neural Network for the SFR ULOF transient (PyTorch 2.12).

Built from scratch with the current SOTA training recipe for stiff,
multi-equation ODE PINNs (see ``docs/references.md``):

* **Non-dimensionalised states + equations** so the stiff PKE residuals are
  O(1)-balanced [Viet Cuong et al. 2024; VS-PINN 2024].
* **Causal temporal weighting** so earlier times are fitted before later ones,
  respecting the arrow of time across the void-onset front
  [Wang, Sankaran & Perdikaris 2024].
* **Gradient-norm adaptive loss weights** that auto-balance the four residual
  blocks [Wang, Teng & Perdikaris 2021].
* **Residual-based adaptive refinement (RAR)** of collocation points
  [Wu et al. 2023].
* **Forward-mode autodiff** for the time derivative via ``torch.func.jvp`` +
  ``vmap`` — a single fused pass for the whole state vector (the modern,
  efficient idiom), with a reverse-mode fallback.
* Optional ``torch.compile``; float64 throughout; hard initial conditions.

``pinn_jax.py`` is the functional JAX/Equinox twin (same residuals and recipe);
``docs/neural_network.md`` §9 compares the two.

Run (after ``uv sync --extra torch-cpu``; use ``--extra torch-gpu`` for a CUDA build)::

    uv run python -m pinn_sfr_transient.pinn_torch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from pinn_sfr_transient.config import SFRParams

try:
    import torch  # ty: ignore
    from torch import nn  # ty: ignore
    from torch.func import jvp, vmap  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.12 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from pinn_sfr_transient.plotting import PinnPrediction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrainConfig:
    """Hyper-parameters for the PINN and its training schedule."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
    adam_iters: int = 15000
    lbfgs_iters: int = 600
    lr: float = 1e-3

    # Causal weighting [Wang, Sankaran & Perdikaris 2024]
    causal_eps: float = 1.0
    causal_chunks: int = 32

    # Gradient-norm adaptive block weights [Wang, Teng & Perdikaris 2021]
    weight_update_every: int = 250
    weight_momentum: float = 0.9  # EMA factor for the weight update

    # Residual-based adaptive refinement / RAR [Wu et al. 2023]
    rar_every: int = 2000
    rar_pool: int = 20000
    rar_add: int = 200
    rar_cap: int = 4000

    # Modern-PyTorch knobs
    jacobian: Literal["forward", "reverse"] = "forward"
    compile: bool = False
    device: str = "cpu"
    seed: int = 0

    # diagnostics
    log_every: int = 1000


# ---------------------------------------------------------------------------
# Network + physics ansatz
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Plain tanh MLP, 1 -> n_out.

    Initialised like Equinox's ``eqx.nn.MLP`` -- weights *and* biases ~ U(-k, k)
    with k = 1/sqrt(fan_in) -- so both backends start from the same distribution.
    This matters for the fit, not just for parity: with the previous
    ``xavier_normal`` + zero-bias init the gradient-norm weighting could not lift
    the stiff power-block weight high enough, and the power trajectory was fit
    poorly (P relative-L2 ~0.28 vs ~5e-3 here). See ``docs/neural_network.md`` §9.
    """

    def __init__(self, n_out: int = 9, width: int = 64, depth: int = 5) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                k = m.in_features**-0.5
                nn.init.uniform_(m.weight, -k, k)
                nn.init.uniform_(m.bias, -k, k)

    def forward(self, tn: torch.Tensor) -> torch.Tensor:
        return self.net(tn)


class SFRPinn(nn.Module):
    """Normalized-state PINN with hard initial conditions for the ULOF system."""

    def __init__(self, p: SFRParams, cfg: TrainConfig) -> None:
        super().__init__()
        self.p = p
        self.cfg = cfg
        dev = cfg.device
        # Seed BEFORE the weights are created: nn.init draws from the global RNG, so
        # the init must be seeded here, not later in Trainer.train(). Otherwise every
        # fresh kernel starts from entropy -> a different basin each run (the JAX twin
        # gets the same determinism from its explicit jax.random key).
        torch.manual_seed(cfg.seed)
        self.net = MLP(9, cfg.width, cfg.depth).to(dev).double()

        self.lam = torch.tensor(p.lambda_i, dtype=torch.float64, device=dev)
        self.beta_i = torch.tensor(p.beta_i, dtype=torch.float64, device=dev)
        self.beta = float(p.beta_eff)
        self.Lam = float(p.Lambda)
        self.t_end = float(p.t_end)
        self.DTf = p.Tf0 - p.Tin
        self.DTc = p.Tc0 - p.Tin
        # nominal normalized state s0 = [p=1, c_i=1, th_f=1, th_c=1]
        self.s0 = torch.ones(9, dtype=torch.float64, device=dev)

    # -- state, with hard IC s(0) = s0 --------------------------------------
    def _state(self, t: torch.Tensor) -> torch.Tensor:
        """Map time ``t`` (..., 1) to the normalized state (..., 9)."""
        tn = t / self.t_end
        return self.s0 + tn * self.net(tn)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self._state(t)

    def to_physical(
        self, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        P = s[..., 0:1]
        C = s[..., 1:7] * (self.beta_i / (self.Lam * self.lam))
        Tf = self.p.Tin + s[..., 7:8] * self.DTf
        Tc = self.p.Tin + s[..., 8:9] * self.DTc
        return P, C, Tf, Tc

    # -- closures used by feedback ------------------------------------------
    def _void(self, Tc: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.tanh(0.5 * (Tc - self.p.T_onset) / self.p.dT_void))

    def _flow(self, t: torch.Tensor) -> torch.Tensor:
        return self.p.f_nc + (1.0 - self.p.f_nc) * torch.exp(-t / self.p.tau_pump)

    def _reactivity(self, Tf: torch.Tensor, Tc: torch.Tensor) -> torch.Tensor:
        return (
            self.p.rho_ext
            + self.p.alpha_f * (Tf - self.p.Tf0)
            + self.p.alpha_c * (Tc - self.p.Tc0)
            + self.p.alpha_void * self._void(Tc)
        )

    # -- state + time-derivative -------------------------------------------
    def _deriv_forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-mode (jvp+vmap): the whole 9-vector derivative in one pass."""

        def f(tau: torch.Tensor) -> torch.Tensor:  # tau: () -> (9,)
            return self._state(tau.reshape(1, 1)).reshape(-1)

        tt = t.reshape(-1)
        ones = torch.ones_like(tt)
        return vmap(lambda a, b: jvp(f, (a,), (b,)))(tt, ones)

    def _deriv_reverse(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse-mode fallback: per-output autograd (always available)."""
        t = t.clone().requires_grad_(True)
        s = self._state(t)
        grads = [
            torch.autograd.grad(
                s[:, k : k + 1],
                t,
                grad_outputs=torch.ones_like(s[:, k : k + 1]),
                create_graph=True,
                retain_graph=True,
            )[0]
            for k in range(9)
        ]
        return s, torch.cat(grads, dim=1)

    def state_and_deriv(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (s, ds/dt), each (N, 9).

        Default is forward-mode (the efficient PyTorch-2.x idiom for a
        scalar-input/vector-output map). It falls back to reverse mode if the
        ``torch.func`` composition either raises *or* — on the first call —
        disagrees with reverse mode: some torch builds (e.g. older ones shipped by
        hosted notebooks) silently miscompute ``jvp``+``vmap``, which would corrupt
        the residual and ruin training. Reverse mode is correct everywhere.
        """
        if self.cfg.jacobian == "forward":
            try:
                s, d = self._deriv_forward(t)
            except Exception:  # noqa: BLE001 - robust fallback for any backend/build quirk
                self._fall_back_to_reverse("forward-mode autodiff unavailable")
                return self._deriv_reverse(t)
            if not getattr(self, "_fwd_checked", False):
                self._fwd_checked = True
                _, d_rev = self._deriv_reverse(t)
                if not torch.allclose(d, d_rev, atol=1e-6, rtol=1e-4):
                    self._fall_back_to_reverse("forward-mode autodiff disagrees with reverse mode")
                    return self._deriv_reverse(t)
            return s, d
        return self._deriv_reverse(t)

    def _fall_back_to_reverse(self, why: str) -> None:
        """Switch to reverse-mode autodiff for the rest of training (once)."""
        if not getattr(self, "_warned_forward", False):
            print(f"[pinn] {why}; using reverse mode.")
            self._warned_forward = True
        self.cfg.jacobian = "reverse"

    # -- per-point squared residual blocks ----------------------------------
    def residual_blocks(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (e_p, e_c, e_tf, e_tc), each (N,), squared residuals."""
        s, ds = self.state_and_deriv(t)
        p_, c, th_f, th_c = s[:, 0:1], s[:, 1:7], s[:, 7:8], s[:, 8:9]
        dp, dc, dth_f, dth_c = ds[:, 0:1], ds[:, 1:7], ds[:, 7:8], ds[:, 8:9]

        Tf = self.p.Tin + th_f * self.DTf
        Tc = self.p.Tin + th_c * self.DTc
        P = p_
        rho = self._reactivity(Tf, Tc)
        g = self._flow(t)

        R_p = self.Lam * dp - ((rho - self.beta) * p_ + (self.beta_i * c).sum(1, keepdim=True))
        R_c = dc - self.lam * (p_ - c)
        R_tf = self.DTf * dth_f - (
            (self.p.P0 / self.p.Cf) * P - (self.p.UA / self.p.Cf) * (Tf - Tc)
        )
        R_tc = self.DTc * dth_c - (
            (self.p.UA / self.p.Cc) * (Tf - Tc) - (self.p.W0 * g / self.p.Cc) * (Tc - self.p.Tin)
        )
        return (
            R_p.pow(2).squeeze(1),
            R_c.pow(2).mean(1),
            R_tf.pow(2).squeeze(1),
            R_tc.pow(2).squeeze(1),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """SOTA training loop: causal weighting + adaptive block weights + RAR."""

    def __init__(self, model: SFRPinn, p: SFRParams, cfg: TrainConfig) -> None:
        self.model = model
        self.p = p
        self.cfg = cfg
        self.dev = cfg.device
        # adaptive per-block weights [lambda_p, lambda_c, lambda_tf, lambda_tc]
        self.block_w = torch.ones(4, dtype=torch.float64, device=self.dev)
        # RAR collocation reservoir (high-residual points), grown over training
        self.rar_points = torch.empty(0, 1, dtype=torch.float64, device=self.dev)

    # -- collocation: uniform + early-time cluster + RAR reservoir -----------
    def collocation(self) -> torch.Tensor:
        n = self.cfg.n_colloc
        t1 = torch.rand(n // 2, 1, dtype=torch.float64, device=self.dev) * self.p.t_end
        t2 = torch.rand(n - n // 2, 1, dtype=torch.float64, device=self.dev) * (0.4 * self.p.t_end)
        pts = [t1, t2]
        if self.rar_points.numel() > 0:
            pts.append(self.rar_points)
        return torch.cat(pts, dim=0)

    def _pointwise_total(self, t: torch.Tensor) -> torch.Tensor:
        e_p, e_c, e_tf, e_tc = self.model.residual_blocks(t)
        w = self.block_w
        return w[0] * e_p + w[1] * e_c + w[2] * e_tf + w[3] * e_tc

    # -- causal temporal weighting [Wang, Sankaran & Perdikaris 2024] -------
    def causal_loss(self, t: torch.Tensor) -> torch.Tensor:
        e = self._pointwise_total(t)  # (N,)
        tt = t.reshape(-1)
        edges = torch.linspace(0.0, self.p.t_end, self.cfg.causal_chunks + 1, device=self.dev)
        idx = torch.bucketize(tt, edges[1:-1].contiguous())  # chunk index in [0, M-1]

        chunk_losses = []
        for m in range(self.cfg.causal_chunks):
            mask = idx == m
            chunk_losses.append(e[mask].mean() if bool(mask.any()) else e.sum() * 0.0)
        L = torch.stack(chunk_losses)  # (M,)

        with torch.no_grad():
            cum_before = torch.cumsum(L, 0) - L  # sum of strictly-earlier chunks
            cw = torch.exp(-self.cfg.causal_eps * cum_before)
        return (cw * L).mean()

    # -- gradient-norm adaptive block weights [Wang, Teng & Perdikaris 2021]-
    def update_block_weights(self, t: torch.Tensor) -> None:
        e_p, e_c, e_tf, e_tc = self.model.residual_blocks(t)
        losses = [e_p.mean(), e_c.mean(), e_tf.mean(), e_tc.mean()]
        params = [q for q in self.model.parameters() if q.requires_grad]
        norms = []
        for lk in losses:
            grads = torch.autograd.grad(lk, params, retain_graph=True, allow_unused=True)
            sq = sum(
                (gi.pow(2).sum() for gi in grads if gi is not None),
                start=torch.zeros((), device=self.dev),
            )
            norms.append(torch.sqrt(sq + 1e-30))
        gn = torch.stack(norms)  # (4,)
        with torch.no_grad():
            lam_hat = gn.mean() / (gn + 1e-12)  # balance: small grad -> larger weight
            mom = self.cfg.weight_momentum
            self.block_w = mom * self.block_w + (1.0 - mom) * lam_hat

    # -- RAR: add worst-residual points to the reservoir [Wu et al. 2023] ---
    @torch.no_grad()
    def rar_refine(self) -> None:
        pool = torch.rand(self.cfg.rar_pool, 1, dtype=torch.float64, device=self.dev) * self.p.t_end
        e = self._pointwise_total(pool)
        top = torch.topk(e, min(self.cfg.rar_add, e.numel())).indices
        self.rar_points = torch.cat([self.rar_points, pool[top]], dim=0)
        if self.rar_points.shape[0] > self.cfg.rar_cap:
            self.rar_points = self.rar_points[-self.cfg.rar_cap :]

    # -- full schedule: Adam (+ adaptivity) then L-BFGS polish --------------
    def train(self, *, verbose: bool = True) -> SFRPinn:
        """Run Adam (with adaptivity) then the L-BFGS polish; return the model."""
        torch.manual_seed(self.cfg.seed)
        model = self.model
        if self.cfg.compile:
            model = torch.compile(model)  # type: ignore[assignment]

        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        # Cosine decay over the Adam phase to lr/10, so the optimiser settles into a
        # lower-loss basin before the L-BFGS polish. (The old StepLR(step_size=4000)
        # only decayed at the very end -- i.e. never, for a typical run.)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.cfg.adam_iters), eta_min=self.cfg.lr * 0.1
        )

        for it in range(self.cfg.adam_iters):
            if it % self.cfg.weight_update_every == 0 and it > 0:
                self.update_block_weights(self.collocation())
            if it % self.cfg.rar_every == 0 and it > 0:
                self.rar_refine()

            opt.zero_grad()
            loss = self.causal_loss(self.collocation())
            loss.backward()
            opt.step()
            sched.step()
            if verbose and it % self.cfg.log_every == 0:
                w = self.block_w.tolist()
                print(
                    f"[adam {it:6d}] loss={loss.item():.3e}  "
                    f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]  "
                    f"rar={self.rar_points.shape[0]}"
                )

        if self.cfg.lbfgs_iters > 0:
            self._lbfgs_polish(verbose=verbose)
        return self.model

    def _lbfgs_polish(self, *, verbose: bool) -> None:
        """Quasi-Newton polish on a fixed collocation set, with a divergence guard."""
        t_fixed = self.collocation()
        loss_before = self.causal_loss(t_fixed).item()
        # Snapshot the Adam result so a diverging L-BFGS step can be rolled back.
        snapshot = [q.detach().clone() for q in self.model.parameters()]
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.cfg.lbfgs_iters,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = self.causal_loss(t_fixed)
            loss.backward()
            return loss

        opt.step(closure)
        loss_after = self.causal_loss(t_fixed).item()
        # Safety net: torch's L-BFGS can diverge on a bad line-search step. If the
        # polish gave a non-finite or worse loss, restore the Adam result.
        if not np.isfinite(loss_after) or loss_after > loss_before:
            with torch.no_grad():
                for q, saved in zip(self.model.parameters(), snapshot, strict=True):
                    q.copy_(saved)
            if verbose:
                print(f"[lbfgs] reverted: {loss_before:.3e} -> {loss_after:.3e} (kept Adam)")
        elif verbose:
            print(f"[lbfgs done] loss={loss_after:.3e}")


# ---------------------------------------------------------------------------
# Inference / validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: SFRPinn, n: int = 2000) -> PinnPrediction:
    """Evaluate the trained model on a uniform grid, returning physical fields."""
    p = model.p
    t = torch.linspace(0.0, p.t_end, n, dtype=torch.float64, device=model.cfg.device).unsqueeze(1)
    P, _C, Tf, Tc = model.to_physical(model.forward(t))
    return {
        "t": t.squeeze(1).cpu().numpy(),
        "P": P.squeeze(1).cpu().numpy(),
        "Tf": Tf.squeeze(1).cpu().numpy(),
        "Tc": Tc.squeeze(1).cpu().numpy(),
    }


def relative_l2(pinn: PinnPrediction, ref: dict[str, np.ndarray]) -> dict[str, float]:
    """Relative L2 error of the prediction against the reference, per field."""

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

    return {
        "P": rel(pinn["P"], np.interp(pinn["t"], ref["t"], ref["P"])),
        "Tf": rel(pinn["Tf"], np.interp(pinn["t"], ref["t"], ref["Tf"])),
        "Tc": rel(pinn["Tc"], np.interp(pinn["t"], ref["t"], ref["Tc"])),
    }


def train(p: SFRParams | None = None, cfg: TrainConfig | None = None) -> SFRPinn:
    """Build the model and trainer, then run the full training schedule."""
    p = p or SFRParams()
    cfg = cfg or TrainConfig()
    model = SFRPinn(p, cfg)
    return Trainer(model, p, cfg).train()


def main() -> None:
    """Train the PINN and report relative L2 error against the reference run."""
    p = SFRParams()
    cfg = TrainConfig()
    model = train(p, cfg)
    pinn = predict(model)

    refpath = Path("results/ulof_reference.npz")
    if refpath.exists():
        ref = dict(np.load(refpath))
        print("\nRelative L2 error vs reference:")
        for k, v in relative_l2(pinn, ref).items():
            print(f"  {k:3s}: {v:.3e}")
    else:
        print("Run `pinn-sfr reference` first to generate the reference trajectory.")


if __name__ == "__main__":
    main()
