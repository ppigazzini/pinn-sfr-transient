"""Limited-memory self-scaled BFGS with a strong-Wolfe line search — the JAX twin.

The method, its motivation and the derivation of ``tau_k`` are documented once,
in :mod:`pinn_sfr_transient.axial.torchpinn.optimizers`. This module implements
the **same algorithm** so the ``optimizer`` knob means the same thing in both
backends, as `AGENTS.md` requires.

Two deliberate asymmetries, both framework-imposed:

* The torch twin is a class driven by a ``closure``, mirroring
  `torch.optim.LBFGS`. Here it is a function over a flat parameter vector,
  because an Equinox model is an immutable PyTree and there is nothing to mutate
  in place.
* The iteration is a Python loop rather than `lax.fori_loop`, and it is **not**
  jitted as a whole. A strong-Wolfe line search has data-dependent control flow
  and a data-dependent trip count, which `jit` cannot express without padding
  every search to its worst case. The expensive part -- the loss and its
  gradient -- is jitted by the caller, and the vector algebra around it is
  ~17k-element dot products, so the Python overhead is not measurable against
  one residual evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

# Strong-Wolfe constants, as stated in arXiv:2501.16371. Identical to the torch twin.
C1 = 1e-4
C2 = 0.9
_MAX_LS_ITERS = 25
_MIN_CURVATURE = 1e-12

Pair = tuple[jax.Array, jax.Array, float, float]


def _apply_H(v: jax.Array, pairs: list[Pair], gamma: float) -> jax.Array:
    """Two-loop recursion: ``H v`` from ``pairs`` with ``H_0 = gamma I``."""
    q = v
    alphas: list[float] = []
    for s, y, rho, _tau in reversed(pairs):
        a = rho * float(jnp.vdot(s, q))
        q = q - a * y
        alphas.append(a)
    r = gamma * q
    for (s, y, rho, tau), a in zip(pairs, reversed(alphas), strict=True):
        if tau != 1.0:
            r = r * tau  # H_i <- tau_i H_i, i.e. SHRINK when tau < 1
        b = rho * float(jnp.vdot(y, r))
        r = r + s * (a - b)
    return r


def _line_search(  # noqa: PLR0913, PLR0917, C901, PLR0912 - a line search needs the
    # whole line, and this branch structure IS Nocedal & Wright Algorithms 3.5
    # and 3.6; splitting it hides the correspondence to the reference.
    value_and_grad: Callable[[jax.Array], tuple[float, jax.Array]],
    x0: jax.Array,
    d: jax.Array,
    f0: float,
    g0: jax.Array,
    counter: list[int],
) -> tuple[float, float, jax.Array]:
    """Bracket then zoom (Nocedal & Wright 3.5/3.6). ``step == 0.0`` means failure."""
    dphi0 = float(jnp.vdot(g0, d))
    if dphi0 >= 0:
        return 0.0, f0, g0

    def point(step: float) -> tuple[float, float, jax.Array, float]:
        f, g = value_and_grad(x0 + step * d)
        counter[0] += 1
        return step, float(f), g, float(jnp.vdot(g, d))

    def armijo(step: float, f: float) -> bool:
        return f <= f0 + C1 * step * dphi0

    def curvature(dphi: float) -> bool:
        return abs(dphi) <= -C2 * dphi0

    prev = (0.0, f0, g0, dphi0)
    cur = point(1.0)
    lo = hi = None
    for i in range(_MAX_LS_ITERS):
        step, f, _g, dphi = cur
        if not armijo(step, f) or (i > 0 and f >= prev[1]):
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
        step, f, g, _ = cur
        return (step, f, g) if f < f0 else (0.0, f0, g0)

    for _ in range(_MAX_LS_ITERS):
        if abs(hi[0] - lo[0]) < 1e-16:
            break
        mid = point(0.5 * (lo[0] + hi[0]))
        step, f, g, dphi = mid
        if not armijo(step, f) or f >= lo[1]:
            hi = mid
        else:
            if curvature(dphi):
                return step, f, g
            if dphi * (hi[0] - lo[0]) >= 0:
                hi = lo
            lo = mid
    if lo[1] < f0:
        return lo[0], lo[1], lo[2]
    return 0.0, f0, g0


def minimize(  # noqa: PLR0913, C901 - these are the optimiser's knobs, flat is clearer
    value_and_grad: Callable[[jax.Array], tuple[float, jax.Array]],
    x0: jax.Array,
    *,
    max_iter: int = 500,
    history_size: int = 50,
    self_scale: bool = True,
    cap_tau: bool = True,
    h0_scaling: bool = True,
    tolerance_grad: float = 1e-12,
    tolerance_change: float = 1e-14,
) -> tuple[jax.Array, float]:
    """Minimise from ``x0``; return the final point and its value."""
    pairs: list[Pair] = []
    gamma = 1.0
    x = x0
    f_val, g = value_and_grad(x)
    f = float(f_val)
    counter = [1]

    for _ in range(max_iter):
        if float(jnp.abs(g).max()) <= tolerance_grad:
            break
        d = -_apply_H(g, pairs, gamma) if pairs else -g / max(float(jnp.linalg.norm(g)), 1.0)
        step, f_new, g_new = _line_search(value_and_grad, x, d, f, g, counter)
        if step == 0.0:
            if not pairs:
                break
            pairs.clear()
            gamma = 1.0
            continue
        s = step * d
        x = x + s
        y = g_new - g
        sy = float(jnp.vdot(s, y))
        if sy > _MIN_CURVATURE * float(jnp.linalg.norm(s)) * float(jnp.linalg.norm(y)):
            tau = 1.0
            if self_scale and pairs:
                b = float(jnp.vdot(y, _apply_H(y, pairs, gamma))) / sy
                if b > 0:
                    tau = min(1.0, 1.0 / b) if cap_tau else 1.0 / b
            pairs.append((s, y, 1.0 / sy, tau))
            if len(pairs) > history_size:
                pairs.pop(0)
            if h0_scaling:
                gamma = sy / float(jnp.vdot(y, y))
        converged = abs(f_new - f) < tolerance_change and float(jnp.abs(s).max()) < tolerance_change
        f, g = f_new, g_new
        if converged:
            break
    return x, f
