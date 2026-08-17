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

**Two solvers, because the shape of the problem chooses one.** The Gauss-Newton step
can be posed in parameter space, where the system is ``n x n``, or in residual space,
where it is ``m x m``, and the two give the *same* step. Here ``m`` (24 000 residuals)
is smaller than ``n`` (50 309 parameters), so the **dual** is the smaller system and
the primal is the wrong side to iterate on (arXiv:2505.21404).

* ``--solver primal`` — matrix-free conjugate gradients on ``J'J + lambda I``, with the
  randomized Nyström preconditioner of arXiv:2505.11638. This is what the first run
  used. Its rank-40 sketch failed to capture a spectrum with condition number 1e8, and
  that failure is *predicted*: the effective dimension of the regularized kernel
  plateaus above half the batch size, so a sketch of 0.17% was never going to work.
* ``--solver dual`` — subsample ``m_sub`` residuals, form the ``m_sub x m_sub`` dual
  Gramian ``J J'`` densely, and solve it exactly by Cholesky. At a few thousand
  residuals that is sub-second in float64 and it removes the sketch rank, the CG
  tolerance and the preconditioner from the hyper-parameter surface **in one move**.

The dual step is recovered as ``d = -J' (J J' + lambda I)^-1 r``, which is the same
vector the primal normal equations give — a fact worth asserting rather than trusting,
and `gauss_newton_probe.py` does assert it.

Scored by the same `relative_l2` every other study uses, against the same ruler, so
the number is comparable to every table in `docs/axial_nn.md`.

    OMP_NUM_THREADS=8 uv run python tools/gauss_newton_experiment.py --seconds 9000
"""

import argparse
import json
import time
from collections.abc import Callable
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


# Rows of the Jacobian built per reverse-mode sweep. This number is a MEMORY bound, not
# a tuning knob, and it does not change the answer at all: reverse mode holds one copy of
# the forward tape per cotangent in flight, and `jax.jacrev` batches every row at once.
#
# Measured at the shipped default (6000 collocation points, 50 309 parameters, float64),
# XLA's own buffer assignment for one sweep is linear in the chunk —
#
#     chunk     8      16      32      64
#     GB     0.380   0.659   1.217   2.494        => 38.7 MB per row, 0.08 GB fixed
#
# — so the 3000-row `jacrev` this replaced asked for **113 GB**, and took a 64 GB host
# down. At 64 rows it is 2.5 GB, and the assembled `J_sub` is the same matrix either way.
JAC_CHUNK = 64

# Refuse to start above this. Well under any plausible host, because the cost of a cap
# that is too low is one error message and the cost of one that is too high was measured.
DEFAULT_MAX_GB = 8.0


def compiled_bytes(jitted: Callable, *args: object) -> float:
    """Return XLA's own peak-bytes figure for ``jitted(*args)``, or ``nan`` if unavailable.

    Asking the compiler beats estimating. `memory_analysis` reports the buffers XLA has
    assigned for a compiled executable, and for a reverse-mode sweep those are dominated
    by the forward tape — the term that scales with how many rows are differentiated at
    once, and the one whose size was assumed rather than checked. Lowering happens before
    any buffer is allocated, so this is safe to call on a shape that would not fit.

    ``jitted`` must be wrapped with **`jax.jit`, not `eqx.filter_jit`**: the Equinox
    wrapper's `Compiled` object does not forward `memory_analysis`, and the first version
    of this guard called it through Equinox, caught the `AttributeError`, and disabled
    itself with a warning. A guard that fails open is not a guard, so the fallback below
    exists only for a backend that genuinely reports nothing.
    """
    try:
        ana = jitted.lower(*args).compile().memory_analysis()
    except Exception:  # noqa: BLE001 - any analysis failure must fall back, not abort
        return float("nan")
    if ana is None:
        return float("nan")
    return float(ana.temp_size_in_bytes + ana.output_size_in_bytes + ana.argument_size_in_bytes)


def check_budget(peak_gb: float, max_gb: float, what: str) -> None:
    """Abort before allocating when the plan does not fit, rather than after.

    This project has taken the host down once by handing `jax.jacrev` a 3000-row output:
    reverse mode holds one forward tape per cotangent in flight, so the allocation was
    ~80 GB and nothing in the code said so. The rule that follows is that any routine
    whose memory scales with a *tunable* must state its peak and refuse to exceed it.
    """
    if not np.isfinite(peak_gb):
        msg = (
            f"could not measure the memory of {what}, and this tool has already taken "
            f"one host down. Pass --max-gb 0 to run unguarded deliberately."
        )
        raise SystemExit(msg)
    print(f"  {what}: {peak_gb:.2f} GB peak (cap {max_gb:.1f} GB)", flush=True)
    if peak_gb > max_gb:
        msg = (
            f"{what} needs {peak_gb:.1f} GB, above the {max_gb:.1f} GB cap. "
            f"Lower --m-sub or --jac-chunk, or raise --max-gb if the host really has it."
        )
        raise SystemExit(msg)


def make_dual_jacobian(
    residuals: Callable[[jax.Array], jax.Array], jac_chunk: int
) -> tuple[Callable, Callable]:
    """Return ``(jac_rows, dual_gram)``, the chunked Jacobian build for the dual solver.

    A module-level factory rather than closures inside the solver, so the one piece of
    this tool that has taken a host down is reachable from a test without a training run.
    """

    @jax.jit  # jax.jit, not filter_jit: `compiled_bytes` needs `memory_analysis`
    def jac_rows(x: jax.Array, idx: jax.Array, ks: jax.Array) -> jax.Array:
        """Return one chunk of rows of ``J_sub``, differentiating only those rows.

        The vjp closure is built **once, outside the vmap**, so the forward tape is a
        captured constant and `vmap` batches the backward sweep alone. Writing it the
        other way round — a vmap over a function that re-opens the tape — costs one tape
        copy per row again and gives back exactly the blow-up this exists to avoid.
        """
        r0, vjp_rows = jax.vjp(lambda z: residuals(z)[idx], x)
        basis = jax.nn.one_hot(ks, idx.size, dtype=r0.dtype)
        return jax.vmap(lambda c: vjp_rows(c)[0])(basis)

    def dual_gram(x: jax.Array, idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``J_sub`` and ``J_sub J_sub'`` for a subsample of rows, built in row blocks.

        A subsample is taken because the dual system is dense: at a few thousand rows a
        Cholesky is sub-second, and beyond that it is not. The rows are redrawn every
        step, so no subset is fitted preferentially.

        The blocks are a fixed ``jac_chunk`` wide including the last one — the tail is
        padded with a repeat of its final index and trimmed afterwards — so every call
        hits one compiled shape instead of two.
        """
        m = int(idx.size)
        blocks = []
        for i in range(0, m, jac_chunk):
            ks = jnp.arange(i, i + jac_chunk)
            take = min(jac_chunk, m - i)
            rows = jac_rows(x, idx, jnp.minimum(ks, m - 1))
            blocks.append(rows[:take])
        j_sub = jnp.concatenate(blocks, axis=0)
        return j_sub, j_sub @ j_sub.T

    return jac_rows, dual_gram


def preflight_dual(
    jac_rows: Callable, flat0: jax.Array, m_rows: int, jac_chunk: int, max_gb: float
) -> None:
    """Cost the dual step in bytes and abort if it does not fit, before allocating.

    Two terms. The reverse sweep is transient and XLA reports it; ``J_sub`` and the dual
    Gramian are held for the whole step and are exact arithmetic. They are summed rather
    than maxed because the sweep for block ``k+1`` runs while the blocks before it are
    still assembled.
    """
    sweep = compiled_bytes(jac_rows, flat0, jnp.arange(m_rows), jnp.arange(jac_chunk))
    held = (m_rows * int(flat0.size) + m_rows * m_rows) * 8.0
    check_budget((sweep + held) / 2**30, max_gb, "dual Gauss-Newton step")


def _nystrom_precond(gram_mv, x, rank: int, key):  # noqa: ANN001, ANN202
    """Randomized Nyström preconditioner, built matrix-free from ``rank`` products."""
    omega = jax.random.normal(key, (x.size, rank))
    y = jnp.stack([gram_mv(x, omega[:, i]) for i in range(rank)], axis=1) + 1e-8 * omega
    c = jnp.linalg.cholesky(omega.T @ y + 1e-10 * jnp.eye(rank))
    b = jax.scipy.linalg.solve_triangular(c, y.T, lower=True).T
    u, sv, _ = jnp.linalg.svd(b, full_matrices=False)
    ev = jnp.maximum(sv**2 - 1e-8, 0.0)
    return u, ev


def gauss_newton(  # noqa: C901, PLR0913, PLR0917 - two solvers plus damping retries
    # is genuinely two branches and a loop; splitting them would hide that the two
    # produce the SAME step, which is the point the sub-command exists to make.
    flat0: jax.Array,
    residuals: Callable[[jax.Array], jax.Array],
    loss: Callable[[jax.Array], jax.Array],
    budget: float,
    rank: int,
    cg_iters: int,
    solver: str = "dual",
    m_sub: int = 3000,
    jac_chunk: int = JAC_CHUNK,
    max_gb: float = DEFAULT_MAX_GB,
) -> tuple:
    """Levenberg-Marquardt Gauss-Newton, solved on whichever side is smaller.

    ``lam`` starts at 1e2 because that is where it was MEASURED to work: one step there
    takes the loss 6.15e-2 -> 8.20e-3, while the same step at 1e-2 increases it. The
    first version started at 1e-3 and grew on rejection, and spent its whole budget
    climbing without reaching the useful range -- the method looked broken and was only
    mis-damped.
    """

    @eqx.filter_jit
    def jvp_fn(x: jax.Array, v: jax.Array) -> jax.Array:
        return jax.jvp(residuals, (x,), (v,))[1]

    @eqx.filter_jit
    def vjp_fn(x: jax.Array, u: jax.Array) -> jax.Array:
        return jax.vjp(residuals, x)[1](u)[0]

    @eqx.filter_jit
    def gram_mv(x: jax.Array, v: jax.Array) -> jax.Array:
        return vjp_fn(x, jvp_fn(x, v))

    jac_rows, dual_gram = make_dual_jacobian(residuals, jac_chunk)

    x, lam, el, steps = flat0, 1e2, 0.0, 0
    trace = [(0.0, float(loss(x)))]
    n_res = int(residuals(flat0).size)

    if solver == "dual":
        preflight_dual(jac_rows, flat0, min(m_sub, n_res), jac_chunk, max_gb)

    while el < budget:
        t0 = time.perf_counter()
        f_old = float(loss(x))
        accepted = False
        if solver == "dual":
            key = jax.random.PRNGKey(steps)
            idx = jax.random.choice(key, n_res, (min(m_sub, n_res),), replace=False)
            j_sub, gg = dual_gram(x, idx)
            r_sub = residuals(x)[idx]
            eye = jnp.eye(gg.shape[0])
            for _ in range(4):
                # d = -J'(JJ' + lam I)^-1 r : the same step as the primal normal
                # equations, obtained from the smaller system.
                step = -j_sub.T @ jnp.linalg.solve(gg + lam * eye, r_sub)
                f_new = float(loss(x + step))
                if np.isfinite(f_new) and f_new < f_old:
                    x, lam, accepted = x + step, max(lam * 0.5, 1e-6), True
                    break
                lam = min(lam * 6.0, 1e10)
        else:
            g = vjp_fn(x, residuals(x))
            u, ev = _nystrom_precond(gram_mv, x, rank, jax.random.PRNGKey(steps))
            for _ in range(4):
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
            f"  {solver} step {steps:4d}  t={el:7.0f}s  loss={trace[-1][1]:.4e}  "
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
    ap.add_argument(
        "--solver",
        choices=("dual", "primal"),
        default="dual",
        help="dual solves the m x m system exactly; primal is CG on n x n",
    )
    ap.add_argument("--m-sub", type=int, default=3000, help="residual rows per dual solve")
    ap.add_argument(
        "--rank",
        type=int,
        default=200,
        help="Nystrom sketch rank (primal only). Literature reports 100-500; "
        "the first run used 40, which is 0.17%% of the batch",
    )
    ap.add_argument(
        "--jac-chunk",
        type=int,
        default=JAC_CHUNK,
        help="Jacobian rows per reverse sweep. Sets peak memory, not the answer: "
        "reverse mode holds one forward tape per row in flight",
    )
    ap.add_argument(
        "--max-gb",
        type=float,
        default=DEFAULT_MAX_GB,
        help="refuse to start if XLA's own peak-bytes figure exceeds this",
    )
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
    x, trace, steps = gauss_newton(
        flat0,
        residuals,
        loss,
        args.seconds,
        args.rank,
        args.cg_iters,
        solver=args.solver,
        m_sub=args.m_sub,
        jac_chunk=args.jac_chunk,
        max_gb=args.max_gb,
    )

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
