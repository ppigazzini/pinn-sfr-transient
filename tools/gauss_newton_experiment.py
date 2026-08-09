"""Gauss-Newton against the shipped default, at equal wall-clock — roadmap D.4.

Run **after** `gauss_newton_probe.py`, which checked the machinery against things
with known answers. This measures whether it is worth anything.

**The configuration is the shipped default** — `t_train_frac = 0.275`, f256, width 64,
depth 5 — and the comparison is against the default budget, 30 Adam + 30000
quasi-Newton, which §7.5.20 measured reaching `T_s = 0.0017`. Running an optimiser
comparison at some other budget is how the first bake-off wasted a day: it used the
study's hardcoded 3000/300, which §7.5.11 had already shown is the regime where the
quasi-Newton stage does not matter.

**Equal wall-clock, not equal iterations.** §7.5.17a is why: the curvature-memory
ladder was monotone at equal iterations and reversed against a clock. A Gauss-Newton
step costs a Jacobian's worth of work, so it must earn that rather than merely reduce
the loss faster per step.

**Matrix-free.** At this size the Jacobian is about 30000 x 83589 and the Gramian
83589 squared; neither is formable. The Gramian acts through ``J' (J v)`` — one
``jvp`` and one ``vjp``, i.e. two gradient-equivalents per matrix-vector product —
and the step is taken by conjugate gradients with the randomized Nyström
preconditioner of arXiv:2505.11638, whose sketch is built the same matrix-free way.

Scored by the same `relative_l2` every other study uses, against the same ruler, so
the number is comparable to every table in `docs/axial_nn.md`.

    OMP_NUM_THREADS=8 uv run python tools/gauss_newton_experiment.py --seconds 9000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pinn_sfr_transient.axial import pinn_jax as pj
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.jaxpinn.residuals import residual_vector
from pinn_sfr_transient.axial.reference import solve_reference
from pinn_sfr_transient.axial.scoring import relative_l2


def make_problem(cfg, seed: int, n_colloc: int):  # noqa: ANN001, ANN201
    """Build the shipped model, a fixed collocation set, and the flat residual."""
    p = AxialParams()
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1000)
    k1, k2 = jax.random.split(key)
    zeta = jax.random.uniform(k1, (n_colloc, 1))
    that = jax.random.uniform(k2, (n_colloc, 1)) * 1.0
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat0, unravel = ravel_pytree(params)

    @eqx.filter_jit
    def residuals(flat: jax.Array) -> jax.Array:
        m = eqx.combine(unravel(flat), static)
        return jnp.concatenate(residual_vector(m, p, zeta, that, cfg))

    @eqx.filter_jit
    def loss(flat: jax.Array) -> jax.Array:
        return 0.5 * jnp.mean(residuals(flat) ** 2)

    return p, static, unravel, flat0, residuals, loss


def gauss_newton(  # noqa: PLR0913, PLR0917
    flat0, residuals, loss, budget: float, rank: int, cg_iters: int
):
    """Matrix-free Levenberg-Marquardt with a randomized Nyström preconditioner."""

    @eqx.filter_jit
    def gram_mv(x: jax.Array, v: jax.Array) -> jax.Array:
        _, jv = jax.jvp(residuals, (x,), (v,))
        return jax.vjp(residuals, x)[1](jv)[0]

    @eqx.filter_jit
    def grad_of(x: jax.Array) -> jax.Array:
        r, vjp = jax.vjp(residuals, x)
        return vjp(r)[0]

    # lam starts at 1e2 because that is where it was MEASURED to work, not at a
    # textbook 1e-3. One step at lam = 1e2 takes the loss 6.15e-2 -> 8.20e-3, a 7.5x
    # reduction; at 1e-2 the same step increases it. Starting small and growing on
    # rejection wastes a full sketch and CG solve per rejection, and the first smoke
    # run spent its whole budget climbing from 1e-2 without ever reaching the useful
    # range -- the method looked broken and was only mis-damped.
    x, lam, el, steps = flat0, 1e2, 0.0, 0
    trace = [(0.0, float(loss(x)))]
    while el < budget:
        t0 = time.perf_counter()
        g = grad_of(x)
        # The Nystrom sketch depends on the Gramian, not on lam, so it is built once
        # per accepted point and reused across damping retries -- which is what makes
        # a rejection cheap rather than a wasted step.
        key = jax.random.PRNGKey(steps)
        omega = jax.random.normal(key, (x.size, rank))
        y = jnp.stack([gram_mv(x, omega[:, i]) for i in range(rank)], axis=1) + 1e-8 * omega
        c = jnp.linalg.cholesky(omega.T @ y + 1e-10 * jnp.eye(rank))
        b = jax.scipy.linalg.solve_triangular(c, y.T, lower=True).T
        u, sv, _ = jnp.linalg.svd(b, full_matrices=False)
        ev = jnp.maximum(sv**2 - 1e-8, 0.0)
        f_old = float(loss(x))
        accepted = False
        for _ in range(4):  # damping retries, reusing the sketch
            inv = 1.0 / (ev + lam) - 1.0 / lam

            def precond(
                v: jax.Array, u: jax.Array = u, inv: jax.Array = inv, lam: float = lam
            ) -> jax.Array:
                return u @ (inv * (u.T @ v)) + v / lam

            step, _ = jax.scipy.sparse.linalg.cg(
                lambda v, x=x, lam=lam: gram_mv(x, v) + lam * v,
                -g,
                M=precond,
                tol=1e-6,
                maxiter=cg_iters,
            )
            f_new = float(loss(x + step))
            if np.isfinite(f_new) and f_new < f_old:
                x, lam, accepted = x + step, max(lam * 0.5, 1e-6), True
                break
            lam = min(lam * 6.0, 1e10)
        el += time.perf_counter() - t0
        steps += 1
        trace.append((el, float(loss(x))))
        print(
            f"  gn step {steps:4d}  t={el:7.0f}s  loss={trace[-1][1]:.4e}  "
            f"lam={lam:.1e}  {'ok' if accepted else 'reject'}",
            flush=True,
        )
    return x, trace, steps


def main() -> int:
    """Train the shipped model with Gauss-Newton and score it against the ruler."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seconds",
        type=float,
        default=9000.0,
        help="wall-clock budget; the default budget's measured cost is ~9060 s",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=20, help="Nystrom sketch rank")
    ap.add_argument("--cg-iters", type=int, default=40)
    ap.add_argument("--colloc", type=int, default=6000)
    ap.add_argument("--out", type=Path, default=Path("gn_experiment.json"))
    args = ap.parse_args()

    cfg = pj.AxialTrainConfig(seed=args.seed, log_every=10**9)
    p, static, unravel, flat0, residuals, loss = make_problem(cfg, args.seed, args.colloc)
    print(
        f"shipped default: f{cfg.fourier_features}, width {cfg.width}, depth {cfg.depth}, "
        f"t_train_frac {cfg.t_train_frac}\n"
        f"{int(flat0.size)} parameters, {int(residuals(flat0).size)} residual entries, "
        f"budget {args.seconds:.0f} s\nstart loss {float(loss(flat0)):.4e}\n",
        flush=True,
    )
    x, trace, steps = gauss_newton(flat0, residuals, loss, args.seconds, args.rank, args.cg_iters)

    traj = solve_reference(AxialParams(n_axial=160), n_out=241)
    model = eqx.combine(unravel(x), static)
    s = relative_l2(pj.predict(model, p, traj.zeta, traj.t, cfg), traj, p)
    print(
        f"\nGauss-Newton, {steps} steps in {trace[-1][0]:.0f} s:\n"
        f"  T_s      {s['T_s']:.4f}   (default budget reaches 0.0017; the bar is 0.01)\n"
        f"  L_void   {s['L_void_max']:.4f}  "
        f"({100 * s['L_void_max'] / s['L_void_max_ref']:.0f}% of reference)\n"
        f"  margin   {s['margin_K']:+.1f} K  (reference {s['margin_K_ref']:+.1f} K)"
    )
    args.out.write_text(json.dumps({"trace": trace, "score": s, "steps": steps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
