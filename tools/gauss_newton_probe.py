"""Does the Gauss-Newton machinery work? — a correctness probe, not a benchmark.

Roadmap item D.4 proposes energy natural gradient / Gauss-Newton with randomized
preconditioning. Before any of that is worth measuring, the pieces have to be right,
and this project's record says that is not a formality: an inverted self-scaling and
a Broyden update contracting the wrong vector both trained fine and were both wrong.

So this checks the machinery against things with known answers, and stops. It does
**not** compare against L-BFGS, does not train to convergence, and produces no number
that belongs in a results table.

Six checks:

1. ``residual_vector`` squared is exactly ``residual_blocks`` — the signed residual
   was factored out of the squared one, so the factoring must be exact.
2. The Jacobian from ``jax.jacrev`` matches central differences. Everything below is
   built on ``J``; if it is wrong, nothing else means anything.
3. The Gramian ``J'J`` is symmetric positive semi-definite, which it is by
   construction and therefore a check on the assembly rather than the mathematics.
4. The Levenberg-Marquardt step actually solves its own normal equations.
5. **Gauss-Newton converges in one step on a linear-residual problem.** This is the
   defining property of the method: where the residual is linear in the parameters,
   the Gauss-Newton model is exact and one step is the answer. Nothing else here
   distinguishes a correct implementation from a plausible one.
6. The randomized Nyström-preconditioned CG solve actually solves its system. This
   is the arXiv:2505.11638 construction, checked on the **linear-system residual**
   rather than by comparing solutions: the Gramian's condition number is ~1e8, so a
   solution comparison has a floor of ``cond * eps`` and would be measuring the
   arithmetic rather than the solver.

    OMP_NUM_THREADS=8 uv run python tools/gauss_newton_probe.py
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from pinn_sfr_transient.axial import pinn_jax as pj
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.jaxpinn.residuals import residual_blocks, residual_vector

WIDTH, DEPTH, FEATS, N_PTS = 12, 2, 8, 96


def _setup():  # noqa: ANN202
    p = AxialParams()
    cfg = pj.AxialTrainConfig(width=WIDTH, depth=DEPTH, fourier_features=FEATS)
    model = pj.AxialPinn(cfg, jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    zeta = jnp.asarray(rng.random((N_PTS, 1)))
    that = jnp.asarray(rng.random((N_PTS, 1)))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    flat0, unravel = ravel_pytree(params)

    def residuals(flat: jax.Array) -> jax.Array:
        m = eqx.combine(unravel(flat), static)
        return jnp.concatenate(residual_vector(m, p, zeta, that, cfg))

    return p, cfg, model, zeta, that, flat0, residuals


def nystrom_solve(gram, g, lam, rank, key):  # noqa: ANN001, ANN201
    """Solve ``(gram + lam I) d = -g`` by CG with a randomized Nyström preconditioner."""
    n = gram.shape[0]
    omega = jax.random.normal(key, (n, rank))
    y = gram @ omega + 1e-10 * omega
    c = jnp.linalg.cholesky(omega.T @ y + 1e-12 * jnp.eye(rank))
    b = jax.scipy.linalg.solve_triangular(c, y.T, lower=True).T
    u, sv, _ = jnp.linalg.svd(b, full_matrices=False)
    ev = jnp.maximum(sv**2 - 1e-10, 0.0)
    inv = 1.0 / (ev + lam) - 1.0 / lam

    def precond(v: jax.Array) -> jax.Array:
        return u @ (inv * (u.T @ v)) + v / lam

    step, _ = jax.scipy.sparse.linalg.cg(
        lambda v: gram @ v + lam * v, -g, M=precond, tol=1e-12, maxiter=500
    )
    return step


def main() -> int:  # noqa: PLR0915 - a checklist reads better flat
    """Run the six checks and report pass or fail for each."""
    p, cfg, model, zeta, that, flat0, residuals = _setup()
    ok = True

    # 1. the factoring is exact
    sq = residual_blocks(model, p, zeta, that, cfg)
    sg = residual_vector(model, p, zeta, that, cfg)
    same = all(bool(jnp.array_equal(a, b**2)) for a, b in zip(sq[: len(sg)], sg, strict=True))
    print(
        f"1. residual_vector**2 is bit-identical to residual_blocks : {'PASS' if same else 'FAIL'}"
    )
    ok &= same

    # 2. the Jacobian, against central differences
    jac = jax.jacrev(residuals)(flat0)
    rng = np.random.default_rng(1)
    cols = rng.choice(int(flat0.size), size=6, replace=False)
    h = 1e-6
    worst = 0.0
    for c in cols:
        e = jnp.zeros_like(flat0).at[int(c)].set(h)
        fd = (residuals(flat0 + e) - residuals(flat0 - e)) / (2 * h)
        worst = max(
            worst, float(jnp.max(jnp.abs(fd - jac[:, int(c)])) / (jnp.max(jnp.abs(fd)) + 1e-30))
        )
    good = worst < 1e-5
    print(
        f"2. jacrev matches central differences (6 columns)          : "
        f"{'PASS' if good else 'FAIL'}  worst rel {worst:.2e}"
    )
    ok &= good

    # 3. the Gramian is symmetric PSD
    gram = jac.T @ jac
    asym = float(jnp.max(jnp.abs(gram - gram.T)) / (jnp.max(jnp.abs(gram)) + 1e-30))
    eigmin = float(jnp.linalg.eigvalsh(gram).min())
    good = asym < 1e-12 and eigmin > -1e-8 * float(jnp.max(jnp.abs(gram)))
    print(
        f"3. J'J symmetric and positive semi-definite                : "
        f"{'PASS' if good else 'FAIL'}  asym {asym:.1e}, min eig {eigmin:.2e}"
    )
    ok &= good

    # 4. the LM step solves its own normal equations
    r = residuals(flat0)
    g = jac.T @ r
    lam = 1e-4
    n = gram.shape[0]
    step = jnp.linalg.solve(gram + lam * jnp.eye(n), -g)
    resid = float(jnp.linalg.norm((gram + lam * jnp.eye(n)) @ step + g) / jnp.linalg.norm(g))
    good = resid < 1e-8
    print(
        f"4. the LM step solves (J'J + lam I) d = -J'r               : "
        f"{'PASS' if good else 'FAIL'}  relative residual {resid:.2e}"
    )
    ok &= good

    # 5. THE check: exact in one step where the residual is linear
    rng = np.random.default_rng(2)
    a_mat = jnp.asarray(rng.normal(size=(60, 20)))
    b_vec = jnp.asarray(rng.normal(size=60))

    def lin_res(x: jax.Array) -> jax.Array:
        return a_mat @ x - b_vec

    x0 = jnp.zeros(20)
    j_lin = jax.jacrev(lin_res)(x0)
    d = jnp.linalg.solve(j_lin.T @ j_lin, -(j_lin.T @ lin_res(x0)))
    x1 = x0 + d
    want = jnp.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    err = float(jnp.linalg.norm(x1 - want) / jnp.linalg.norm(want))
    good = err < 1e-10
    print(
        f"5. one Gauss-Newton step is exact on a linear residual     : "
        f"{'PASS' if good else 'FAIL'}  relative error {err:.2e}"
    )
    ok &= good

    # 6. Nystrom-preconditioned CG solves the system it was given
    #
    # Asserted on the LINEAR-SYSTEM RESIDUAL, not on the solution. Written first as
    # a solution comparison at 1e-6, it failed at 1.8e-3 -- and that was the check
    # being wrong, not the solver: the Gramian's condition number is ~1e8, so
    # `cond * eps` puts a floor of ~2e-8 on any solution comparison and a converged
    # CG residual of 3.5e-7 lands exactly where 1.8e-3 of solution error is expected.
    # The same mistake as the quadrature tolerance in `test_hostile_audit.py`: an
    # absolute bound that fails for the right reason and would then be silenced for
    # the wrong one.
    a_op = gram + lam * jnp.eye(n)
    ev_all = jnp.linalg.eigvalsh(a_op)
    cond = float(ev_all.max() / ev_all.min())
    approx = nystrom_solve(gram, g, lam, rank=min(40, n // 2), key=jax.random.PRNGKey(3))
    res_ap = float(jnp.linalg.norm(a_op @ approx + g) / jnp.linalg.norm(g))
    sol = float(jnp.linalg.norm(approx - step) / jnp.linalg.norm(step))
    good = res_ap < 1e-5
    print(
        f"6. Nystrom-preconditioned CG solves the linear system      : "
        f"{'PASS' if good else 'FAIL'}  residual {res_ap:.2e}"
    )
    ok &= good

    print(
        f"\n   conditioning of (J'J + lam I): {cond:.2e}\n"
        f"   solution difference vs the direct solve: {sol:.2e}, which is the\n"
        f"   residual amplified by that conditioning rather than an error.\n"
        f"   A rank-{min(40, n // 2)} sketch does not capture this spectrum -- the very\n"
        f"   ill-conditioning that motivates roadmap item D.4, measured here rather\n"
        f"   than assumed. Rank and damping are what a real implementation must sweep."
    )

    print(f"\n{n} parameters, {int(r.size)} residual entries.")
    print("ALL CHECKS PASSED" if ok else "SOMETHING IS WRONG — do not measure anything yet")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
