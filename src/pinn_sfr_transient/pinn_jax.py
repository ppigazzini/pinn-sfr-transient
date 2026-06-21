"""Physics-Informed Neural Network for the SFR ULOF transient (JAX + Equinox).

The JAX implementation of the same PINN as ``pinn_torch.py``: identical
normalized residuals, hard initial conditions, and Adam -> L-BFGS schedule with
causal weighting and gradient-norm adaptive loss weights. Stack: **Equinox**
(model as a PyTree), **Optax** (``optax.adam``/``optax.lbfgs``; jaxopt is
deprecated), ``jax.jvp`` + ``jax.vmap`` for forward-mode derivatives, float64 via
``jax_enable_x64``. ``docs/neural_network.md`` §9 compares the two backends.
Runs on CPU or GPU; **not TPU** (the required float64 is unsupported there).

The ``SFRPinn`` module holds *only* the network; ``SFRParams`` is passed to the
free residual/loss functions, keeping array constants out of the PyTree metadata
(Optax's tree ops require it to be hashable). RAR augments the collocation set
with a *fixed* number of high-residual points so shapes stay static under
``jit`` (the torch backend grows an unbounded reservoir instead).

Run (after ``uv sync --extra jax-cpu``; use ``--extra jax-gpu`` for CUDA)::

    uv run python -m pinn_sfr_transient.pinn_jax
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinn_sfr_transient.config import SFRParams

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "JAX backend requires `uv sync --extra jax-cpu` (or `--extra jax-gpu`)"
    raise SystemExit(msg) from exc

# float64 is essential for this stiff problem; set before any array is created.
jax.config.update("jax_enable_x64", True)


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
    weight_momentum: float = 0.9

    # Residual-adaptive refinement — RAR [Wu et al. 2023]. Every `rar_every` steps a
    # fixed-size set of the highest-(weighted-)residual points, drawn from a dense
    # pool, AUGMENTS the base collocation set (it never replaces it, so coverage is
    # preserved). The torch backend grows an unbounded reservoir; keeping the
    # augmentation count constant here keeps shapes static, so `jit` never recompiles.
    rar_every: int = 2000
    rar_pool: int = 20000
    n_rar: int = 1000

    seed: int = 0
    log_every: int = 1000


# ---------------------------------------------------------------------------
# Network + hard-IC ansatz  (an eqx.Module *is* a PyTree -> jit/grad/vmap work)
# ---------------------------------------------------------------------------
class SFRPinn(eqx.Module):
    """Normalized-state network with hard initial conditions s(0) = ones(9)."""

    mlp: eqx.nn.MLP
    t_end: float = eqx.field(static=True)

    def __init__(self, p: SFRParams, cfg: TrainConfig, key: jax.Array) -> None:
        self.t_end = p.t_end
        # 1 -> width -> ... -> 9, tanh activations (smooth, well-behaved AD).
        self.mlp = eqx.nn.MLP(
            in_size=1,
            out_size=9,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnp.tanh,
            key=key,
        )

    def state(self, t: jax.Array) -> jax.Array:
        """Map scalar time ``t`` to the normalized state (9,); s(0) = ones(9)."""
        tn = t / self.t_end
        return jnp.ones(9) + tn * self.mlp(jnp.reshape(tn, (1,)))

    def __call__(self, t: jax.Array) -> jax.Array:
        return self.state(t)

    def state_and_deriv(self, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return (s, ds/dt), each (N, 9), via a single forward-mode pass."""

        def one(ti: jax.Array) -> tuple[jax.Array, jax.Array]:
            return jax.jvp(self.state, (ti,), (jnp.ones_like(ti),))

        return jax.vmap(one)(t)


# ---------------------------------------------------------------------------
# Physics + residuals  (free functions of (model, t, p); mirror of physics.py)
# ---------------------------------------------------------------------------
def _void(tc: jax.Array, p: SFRParams) -> jax.Array:
    return 0.5 * (1.0 + jnp.tanh(0.5 * (tc - p.T_onset) / p.dT_void))


def _reactivity(tf: jax.Array, tc: jax.Array, p: SFRParams) -> jax.Array:
    return (
        p.rho_ext
        + p.alpha_f * (tf - p.Tf0)
        + p.alpha_c * (tc - p.Tc0)
        + p.alpha_void * _void(tc, p)
    )


def residual_blocks(
    model: SFRPinn, t: jax.Array, p: SFRParams
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (e_p, e_c, e_tf, e_tc), each (N,), squared normalized residuals."""
    s, ds = model.state_and_deriv(t)
    p_, c, th_f, th_c = s[:, 0], s[:, 1:7], s[:, 7], s[:, 8]
    dp, dc, dth_f, dth_c = ds[:, 0], ds[:, 1:7], ds[:, 7], ds[:, 8]

    d_tf = p.Tf0 - p.Tin
    d_tc = p.Tc0 - p.Tin
    tf = p.Tin + th_f * d_tf
    tc = p.Tin + th_c * d_tc
    rho = _reactivity(tf, tc, p)
    g = p.f_nc + (1.0 - p.f_nc) * jnp.exp(-t / p.tau_pump)

    lam = jnp.asarray(p.lambda_i)
    beta_i = jnp.asarray(p.beta_i)

    r_p = p.Lambda * dp - ((rho - p.beta_eff) * p_ + (c * beta_i).sum(axis=1))
    r_c = dc - lam * (p_[:, None] - c)
    r_tf = d_tf * dth_f - ((p.P0 / p.Cf) * p_ - (p.UA / p.Cf) * (tf - tc))
    r_tc = d_tc * dth_c - ((p.UA / p.Cc) * (tf - tc) - (p.W0 * g / p.Cc) * (tc - p.Tin))
    return r_p**2, (r_c**2).mean(axis=1), r_tf**2, r_tc**2


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _pointwise_total(model: SFRPinn, t: jax.Array, w: jax.Array, p: SFRParams) -> jax.Array:
    e_p, e_c, e_tf, e_tc = residual_blocks(model, t, p)
    return w[0] * e_p + w[1] * e_c + w[2] * e_tf + w[3] * e_tc


def causal_loss(
    model: SFRPinn, t: jax.Array, w: jax.Array, p: SFRParams, cfg: TrainConfig
) -> jax.Array:
    """Causally-weighted physics loss [Wang, Sankaran & Perdikaris 2024]."""
    e = _pointwise_total(model, t, w, p)
    edges = jnp.linspace(0.0, p.t_end, cfg.causal_chunks + 1)
    idx = jnp.clip(jnp.searchsorted(edges, t, side="right") - 1, 0, cfg.causal_chunks - 1)

    sums = jnp.zeros(cfg.causal_chunks).at[idx].add(e)
    counts = jnp.zeros(cfg.causal_chunks).at[idx].add(1.0)
    chunk_loss = sums / jnp.maximum(counts, 1.0)  # (M,)

    cum_before = jnp.cumsum(chunk_loss) - chunk_loss  # earlier-chunk losses
    cw = jax.lax.stop_gradient(jnp.exp(-cfg.causal_eps * cum_before))
    return (cw * chunk_loss).mean()


def _block_grad_norms(model: SFRPinn, t: jax.Array, p: SFRParams) -> jax.Array:
    """L2 gradient norm of each residual block w.r.t. the network params."""
    norms = []
    for k in range(4):
        grads = eqx.filter_grad(lambda m, k=k: residual_blocks(m, t, p)[k].mean())(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        sq = sum(jnp.vdot(leaf, leaf) for leaf in leaves)
        norms.append(jnp.sqrt(sq + 1e-30))
    return jnp.stack(norms)


# ---------------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------------
def _collocation(p: SFRParams, cfg: TrainConfig, key: jax.Array) -> jax.Array:
    """Uniform points on [0, t_end] plus an early-time cluster (fastest dynamics)."""
    k1, k2 = jax.random.split(key)
    n = cfg.n_colloc
    t1 = jax.random.uniform(k1, (n // 2,)) * p.t_end
    t2 = jax.random.uniform(k2, (n - n // 2,)) * (0.4 * p.t_end)
    return jnp.concatenate([t1, t2])


def _rar_points(
    model: SFRPinn, p: SFRParams, cfg: TrainConfig, w: jax.Array, key: jax.Array
) -> jax.Array:
    """Residual-adaptive refinement, RAR variant [Wu et al. 2023].

    Draw a dense pool and return its ``n_rar`` highest-(weighted-)residual points.
    These *augment* the base collocation set; the fixed count keeps the jitted step
    from recompiling (the torch backend instead grows an unbounded reservoir).
    """
    pool = jax.random.uniform(key, (cfg.rar_pool,)) * p.t_end
    e = jax.lax.stop_gradient(_pointwise_total(model, pool, w, p))
    idx = jax.lax.top_k(e, cfg.n_rar)[1]
    return pool[idx]


# ---------------------------------------------------------------------------
# Training: Adam (+ adaptive weights) then L-BFGS polish
# ---------------------------------------------------------------------------
def train(
    p: SFRParams | None = None, cfg: TrainConfig | None = None, *, verbose: bool = True
) -> SFRPinn:
    """Build the model and run the full training schedule; return the model."""
    p = p or SFRParams()
    cfg = cfg or TrainConfig()
    key, mkey = jax.random.split(jax.random.key(cfg.seed))
    model = SFRPinn(p, cfg, mkey)

    block_w = jnp.ones(4)
    # Cosine decay over the Adam phase to lr/10 (alpha), so the optimiser settles into
    # a lower-loss basin before the L-BFGS polish (mirrors the torch backend).
    lr_sched = optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.adam_iters), alpha=0.1)
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def adam_step(
        model: SFRPinn, opt_state: optax.OptState, t: jax.Array, w: jax.Array
    ) -> tuple[SFRPinn, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(lambda m: causal_loss(m, t, w, p, cfg))(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    rar = _rar_points(model, p, cfg, block_w, key)  # high-residual augmentation
    for it in range(cfg.adam_iters):
        if it > 0 and it % cfg.rar_every == 0:
            key, rk = jax.random.split(key)
            rar = _rar_points(model, p, cfg, block_w, rk)
        # fresh base each step (uniform + early cluster) ++ the RAR points; the
        # total size is constant, so the jitted step never recompiles.
        key, ck = jax.random.split(key)
        t = jnp.concatenate([_collocation(p, cfg, ck), rar])
        if it % cfg.weight_update_every == 0 and it > 0:
            gn = _block_grad_norms(model, t, p)
            lam_hat = gn.mean() / (gn + 1e-12)
            block_w = cfg.weight_momentum * block_w + (1.0 - cfg.weight_momentum) * lam_hat
        model, opt_state, loss = adam_step(model, opt_state, t, block_w)
        if verbose and it % cfg.log_every == 0:
            w0, w1, w2, w3 = block_w.tolist()
            print(f"[adam {it:6d}] loss={float(loss):.3e}  w=[{w0:.2f},{w1:.2f},{w2:.2f},{w3:.2f}]")

    if cfg.lbfgs_iters > 0:
        key, ck = jax.random.split(key)
        t_fixed = _collocation(p, cfg, ck)
        model = _lbfgs_polish(model, t_fixed, block_w, p, cfg)
        if verbose:
            print(f"[lbfgs done] loss={float(causal_loss(model, t_fixed, block_w, p, cfg)):.3e}")
    return model


def _lbfgs_polish(
    model: SFRPinn, t: jax.Array, w: jax.Array, p: SFRParams, cfg: TrainConfig
) -> SFRPinn:
    """Quasi-Newton polish on a fixed collocation set via ``optax.lbfgs``."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params: SFRPinn) -> jax.Array:
        return causal_loss(eqx.combine(params, static), t, w, p, cfg)

    opt = optax.lbfgs()
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def step(_: int, carry: tuple) -> tuple:
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=loss_fn)
        return optax.apply_updates(params, updates), state

    params, _ = jax.lax.fori_loop(0, cfg.lbfgs_iters, step, (params, state))
    return eqx.combine(params, static)


# ---------------------------------------------------------------------------
# Inference / validation
# ---------------------------------------------------------------------------
def predict(model: SFRPinn, p: SFRParams, n: int = 2000) -> dict[str, np.ndarray]:
    """Evaluate the trained model on a uniform grid, returning physical fields."""
    t = jnp.linspace(0.0, p.t_end, n)
    s = jax.vmap(model.state)(t)
    tf = p.Tin + s[:, 7] * (p.Tf0 - p.Tin)
    tc = p.Tin + s[:, 8] * (p.Tc0 - p.Tin)
    return {
        "t": np.asarray(t),
        "P": np.asarray(s[:, 0]),
        "Tf": np.asarray(tf),
        "Tc": np.asarray(tc),
    }


def relative_l2(pinn: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, float]:
    """Relative L2 error of the prediction against the reference, per field."""

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

    return {
        "P": rel(pinn["P"], np.interp(pinn["t"], ref["t"], ref["P"])),
        "Tf": rel(pinn["Tf"], np.interp(pinn["t"], ref["t"], ref["Tf"])),
        "Tc": rel(pinn["Tc"], np.interp(pinn["t"], ref["t"], ref["Tc"])),
    }


def main() -> None:
    """Train the PINN and report relative L2 error against the reference run."""
    from pathlib import Path  # noqa: PLC0415  (CLI-only)

    p = SFRParams()
    model = train(p, TrainConfig())
    pinn = predict(model, p)

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
