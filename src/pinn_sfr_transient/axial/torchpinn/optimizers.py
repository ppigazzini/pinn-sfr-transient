"""Limited-memory self-scaled BFGS with a strong-Wolfe line search.

**Why this exists.** `docs/axial_nn.md` section 7.3.4 measured that the
quasi-Newton stage is not polishing anything here: with it disabled,
``max alpha = 0.0000`` and ``L_void = 0.0000`` in both backends on every seed.
Three thousand Adam iterations never form the boiling front; three hundred L-BFGS
iterations do. That makes the quasi-Newton stage the component that decides
whether the physics appears at all, and it is the one component that had never
been varied — `torch.optim.LBFGS` at its defaults, and nothing to compare it to.

**The method.** Standard L-BFGS applies the Oren-Luenberger scaling once, to the
initial inverse-Hessian approximation:

    H_0 = gamma I,   gamma = (s'y) / (y'y)

Self-scaled BFGS applies a scaling at *every* update [Oren & Luenberger 1974;
Al-Baali 1998], which is what Kiyani, Shukla, Urban, Darbon & Karniadakis
(arXiv:2501.16371) find beats L-BFGS across PINN benchmarks:

    H_{k+1} = (I - rho s y') (tau_k H_k) (I - rho y s') + rho s s'
    rho     = 1 / (y's)
    tau_k   = min(1, 1 / b_k),   b_k = (y' H_k y) / (y's)

The secant condition ``H_{k+1} y = s`` holds for *any* symmetric matrix in the
middle, so ``tau_k`` is free to choose. It is chosen so that the scaled operator
reproduces the observed curvature along ``y``: requiring
``y'(tau H_k)y = y's`` gives ``tau = 1/b_k`` exactly. ``b_k > 1`` means ``H_k``
overestimates the inverse curvature along that direction and the step it proposes
is too long. Capping at 1 makes the scaling a damper only: it can never inflate
the approximation.

**The direction of that scaling is the whole method, and getting it backwards is
silent.** Multiplying where one should divide still satisfies the secant
condition, still descends, and still converges -- just worse. It was caught here
only because the implementation is checked against `torch.optim.LBFGS` on
problems with known minima before it is allowed near the PINN.

**In limited memory** the scaling enters the second loop of the two-loop
recursion as a division by ``tau_i`` immediately before pair ``i``'s correction,
because at that point the accumulator holds ``H_i`` applied to the forward-loop
vector. ``b_i`` is evaluated when the pair is pushed, using the operator built
from the pairs already stored -- one extra two-loop recursion per iteration,
which is negligible against one loss-and-gradient evaluation.

Setting ``self_scale=False`` sets every ``tau_i`` to 1 and recovers textbook
L-BFGS, which is how the implementation is checked against `torch.optim.LBFGS`
rather than trusted.

**The line search** is the standard strong-Wolfe bracketing-and-zoom of Nocedal &
Wright (Algorithms 3.5 and 3.6) at ``c1 = 1e-4``, ``c2 = 0.9`` -- the constants
arXiv:2501.16371 states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Strong-Wolfe constants, as stated in arXiv:2501.16371.
C1 = 1e-4
C2 = 0.9
_MAX_LS_ITERS = 25
_MIN_CURVATURE = 1e-12


class SelfScaledLBFGS:
    """Limited-memory (self-scaled) BFGS driven by a closure, as `torch.optim.LBFGS` is.

    Parameters
    ----------
    params
        The tensors to optimise. Flattened into one vector internally.
    max_iter
        Quasi-Newton iterations. Each costs one gradient plus the line search's
        function evaluations.
    history_size
        Number of ``(s, y)`` pairs retained.
    self_scale
        Apply ``tau_k`` at every update (self-scaled BFGS). ``False`` gives
        textbook L-BFGS and exists so the two can be compared inside one
        implementation.
    tolerance_grad, tolerance_change
        Stopping tolerances on the infinity norm of the gradient and on the step.
    """

    def __init__(  # noqa: PLR0913 - these are the optimiser's knobs, flat is clearer
        self,
        params: Iterable[torch.Tensor],
        *,
        max_iter: int = 500,
        history_size: int = 50,
        self_scale: bool = True,
        cap_tau: bool = True,
        h0_scaling: bool = True,
        tolerance_grad: float = 1e-12,
        tolerance_change: float = 1e-14,
    ) -> None:
        self.params = [q for q in params if q.requires_grad]
        self.max_iter = max_iter
        self.history_size = history_size
        self.self_scale = self_scale
        # `cap_tau` is Al-Baali's min(1, .) safeguard, which makes the scaling a
        # damper only. `h0_scaling` is the Shanno-Phua gamma on H_0 that plain
        # L-BFGS uses. Both are exposed because self-scaling and H_0 scaling
        # address the same defect, and applying both compounds the damping: over
        # a 50-pair history the cumulative factor is the product of every tau.
        self.cap_tau = cap_tau
        self.h0_scaling = h0_scaling
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.n_iter = 0
        self.n_evals = 0

    # -- flat parameter/gradient views -------------------------------------
    def _flat_params(self) -> torch.Tensor:
        return torch.cat([q.detach().reshape(-1) for q in self.params])

    def _set_flat_params(self, x: torch.Tensor) -> None:
        i = 0
        with torch.no_grad():
            for q in self.params:
                n = q.numel()
                q.copy_(x[i : i + n].view_as(q))
                i += n

    def _flat_grad(self) -> torch.Tensor:
        return torch.cat(
            [(torch.zeros_like(q) if q.grad is None else q.grad).reshape(-1) for q in self.params]
        )

    # -- the two-loop recursion --------------------------------------------
    def _apply_H(
        self,
        v: torch.Tensor,
        pairs: list[tuple[torch.Tensor, torch.Tensor, float, float]],
        gamma: float,
    ) -> torch.Tensor:
        """Return ``H v`` for the operator built from ``pairs`` with ``H_0 = gamma I``."""
        q = v.clone()
        alphas: list[float] = []
        for s, y, rho, _tau in reversed(pairs):
            a = rho * float(torch.dot(s, q))
            q = q - a * y
            alphas.append(a)
        r = gamma * q
        for (s, y, rho, tau), a in zip(pairs, reversed(alphas), strict=True):
            if tau != 1.0:
                r = r * tau  # H_i <- tau_i H_i, i.e. SHRINK when tau < 1
            b = rho * float(torch.dot(y, r))
            r = r + s * (a - b)
        return r

    # -- strong-Wolfe line search ------------------------------------------
    def _line_search(  # noqa: C901, PLR0912, PLR0911 - this branch structure IS
        # Nocedal & Wright Algorithms 3.5 and 3.6; splitting it hides the
        # correspondence that makes it auditable against the reference.
        self,
        closure: Callable[[], torch.Tensor],
        x0: torch.Tensor,
        d: torch.Tensor,
        f0: float,
        g0: torch.Tensor,
    ) -> tuple[float, float, torch.Tensor]:
        """Bracket then zoom for a step satisfying the strong Wolfe conditions.

        Returns ``(step, f, grad)``; ``step == 0.0`` means the search failed and
        the caller must stop rather than take an unvalidated step.
        """
        dphi0 = float(torch.dot(g0, d))
        if dphi0 >= 0:  # not a descent direction; the caller resets the history
            return 0.0, f0, g0

        def evaluate(step: float) -> tuple[float, torch.Tensor, float]:
            self._set_flat_params(x0 + step * d)
            f = float(closure().detach())
            self.n_evals += 1
            g = self._flat_grad()
            return f, g, float(torch.dot(g, d))

        # A point on the line: (step, phi, grad, phi').
        def point(step: float) -> tuple[float, float, torch.Tensor, float]:
            f, g, dphi = evaluate(step)
            return step, f, g, dphi

        def sufficient_decrease(step: float, f: float) -> bool:
            return f <= f0 + C1 * step * dphi0

        def curvature(dphi: float) -> bool:
            return abs(dphi) <= -C2 * dphi0

        # Phase 1 -- bracket (Nocedal & Wright, Algorithm 3.5).
        prev = (0.0, f0, g0, dphi0)
        cur = point(1.0)
        lo = hi = None
        for i in range(_MAX_LS_ITERS):
            step, f, _g, dphi = cur
            if not sufficient_decrease(step, f) or (i > 0 and f >= prev[1]):
                lo, hi = prev, cur
                break
            if curvature(dphi):
                return step, f, cur[2]
            if dphi >= 0:
                lo, hi = cur, prev
                break
            prev = cur
            cur = point(min(2.0 * step, 1e10))
        if lo is None or hi is None:
            # Never bracketed: the last point still satisfied Armijo, so keep it.
            step, f, g, _ = cur
            if f < f0:
                self._set_flat_params(x0 + step * d)
                return step, f, g
            return 0.0, f0, g0

        # Phase 2 -- zoom (Algorithm 3.6). Bisection: robust, and the cost here is
        # dominated by the closure, not by the interpolation's iteration count.
        for _ in range(_MAX_LS_ITERS):
            if abs(hi[0] - lo[0]) < 1e-16:
                break
            mid = point(0.5 * (lo[0] + hi[0]))
            step, f, g, dphi = mid
            if not sufficient_decrease(step, f) or f >= lo[1]:
                hi = mid
            else:
                if curvature(dphi):
                    return step, f, g
                if dphi * (hi[0] - lo[0]) >= 0:
                    hi = lo
                lo = mid
        # No Wolfe point found; fall back to the best validated decrease.
        if lo[1] < f0:
            self._set_flat_params(x0 + lo[0] * d)
            return lo[0], lo[1], lo[2]
        return 0.0, f0, g0

    # -- the driver ---------------------------------------------------------
    def step(self, closure: Callable[[], torch.Tensor]) -> float:  # noqa: C901
        """Run up to ``max_iter`` iterations. ``closure`` must zero, evaluate and backward."""
        pairs: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []
        gamma = 1.0
        f = float(closure().detach())
        self.n_evals += 1
        g = self._flat_grad()

        for _ in range(self.max_iter):
            if float(g.abs().max()) <= self.tolerance_grad:
                break
            d = -self._apply_H(g, pairs, gamma) if pairs else -g / max(float(g.norm()), 1.0)
            x0 = self._flat_params()
            step, f_new, g_new = self._line_search(closure, x0, d, f, g)
            if step == 0.0:
                if not pairs:  # no direction and no history left to blame
                    break
                pairs.clear()  # reset to steepest descent and try once more
                gamma = 1.0
                self._set_flat_params(x0)
                continue
            s = step * d
            y = g_new - g
            sy = float(torch.dot(s, y))
            if sy > _MIN_CURVATURE * float(s.norm()) * float(y.norm()):
                # tau uses the operator BEFORE this pair is stored.
                tau = 1.0
                if self.self_scale and pairs:
                    Hy = self._apply_H(y, pairs, gamma)
                    b = float(torch.dot(y, Hy)) / sy
                    if b > 0:
                        tau = min(1.0, 1.0 / b) if self.cap_tau else 1.0 / b
                pairs.append((s, y, 1.0 / sy, tau))
                if len(pairs) > self.history_size:
                    pairs.pop(0)
                if self.h0_scaling:
                    gamma = sy / float(torch.dot(y, y))
            small = float(s.abs().max()) < self.tolerance_change
            if abs(f_new - f) < self.tolerance_change and small:
                f, g = f_new, g_new
                break
            f, g = f_new, g_new
            self.n_iter += 1
        return f
