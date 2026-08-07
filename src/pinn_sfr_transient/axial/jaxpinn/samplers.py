"""Collocation samplers.

Where the residual is evaluated, independent of what the residual is. All draws
are keyed, and every set has a *constant* size so the jitted step never
recompiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.jaxpinn.ansatz import front_position, normalised_state
from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.residuals import residual_blocks, uses_front
from pinn_sfr_transient.axial.physics import kinetics_weights

# Candidates drawn per kept point when sampling the level set. Matches the torch twin.
_LEVEL_SET_POOL = 8


# --- collocation ------------------------------------------------------------
def _collocation(
    p: AxialParams,
    cfg: AxialTrainConfig,
    key: jax.Array,
    t_max: float = 1.0,
    model: AxialPinn | None = None,
) -> tuple:
    """Uniform points with an early-time cluster; time-only when feedback is on.

    ``t_max`` is the time-window curriculum of ``n_windows``. When the front
    network is on, a share ``front_frac`` of the points is drawn near the
    predicted front: RAR cannot supply those, because it samples by residual
    magnitude and the field residual is small everywhere once the void is closed
    algebraically.
    """
    if cfg.feedback:
        k1, k2 = jax.random.split(key)
        that = jnp.concatenate(
            [
                jax.random.uniform(k1, (cfg.n_time, 1)) * t_max,
                jax.random.uniform(k2, (cfg.n_time // 2, 1)) * 0.4 * t_max,
            ]
        )
        zeta_q = jnp.asarray(p.zeta_nodes().reshape(-1, 1))
        weights = tuple(jnp.asarray(w) for w in kinetics_weights(p))
        return that, zeta_q, weights
    k1, k2, k3, k4 = jax.random.split(key, 4)
    pts = jax.random.uniform(k1, (cfg.n_colloc, 2)).at[:, 1].multiply(t_max)
    early = jax.random.uniform(k2, (cfg.n_colloc // 2, 2)).at[:, 1].multiply(0.4 * t_max)
    parts = [pts, early]
    if model is not None and uses_front(cfg) and cfg.front_frac > 0.0:
        n = int(cfg.n_colloc * cfg.front_frac)
        t_f = jax.random.uniform(k3, (n, 1)) * t_max
        z_f = jax.vmap(lambda h: front_position(model, h))(t_f)
        spread = 0.05 * jax.random.normal(k4, (n, 1))
        parts.append(jnp.concatenate([jnp.clip(z_f + spread, 0.0, 1.0), t_f], axis=1))
    elif model is not None and cfg.front_level_set and cfg.front_frac > 0.0:
        parts.append(_level_set_points(model, p, cfg, k3, t_max))
    allp = jnp.concatenate(parts)
    return allp[:, 0:1], allp[:, 1:2]


def _level_set_points(
    model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, key: jax.Array, t_max: float
) -> jax.Array:
    """Collocation on the saturation level set, found from the model's own ``T_c``.

    The torch twin carries the full rationale. In short: the loss is a mean over
    the domain and the front is a few percent of it, so the objective under-weights
    the front however long training runs. RAR cannot supply these points because
    the residual is small everywhere once the void is closed algebraically, and no
    front-position network is needed because under D-TH-3 the front IS the level
    set ``T_c = T_sat + dT_superheat``.

    Rejection sampling keeps the shape static, so ``jit`` never recompiles.
    """
    n = int(cfg.n_colloc * cfg.front_frac)
    cand = jax.random.uniform(key, (n * _LEVEL_SET_POOL, 2)).at[:, 1].multiply(t_max)
    dT = p.P_0 / (p.w_0 * p.c_c)
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(cand[:, 0:1], cand[:, 1:2])
    T_c = p.T_in + theta[:, 3] * dT
    T_boil = sodium.saturation_temperature(p.p_system) + p.dT_superheat
    idx = jnp.argsort(jnp.abs(T_c - T_boil))[:n]
    return cand[idx]


def _merge(base: tuple, rar: tuple | None, *, feedback: bool) -> tuple:
    """Append the RAR points to a freshly drawn base set, keeping the size constant."""
    if rar is None or feedback:
        return base
    return (jnp.concatenate([base[0], rar[0]]), jnp.concatenate([base[1], rar[1]]))


def _rar_points(
    model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, key: jax.Array, w: jax.Array
) -> tuple:
    """Select the worst-residual points from a fresh pool [Wu et al. 2023].

    A *fixed* count, so shapes stay static and ``jit`` never recompiles; the
    caller appends them to a freshly drawn base set each step. Disabled under
    feedback, where the axial direction is a quadrature rule that cannot absorb
    arbitrary points.
    """
    if cfg.feedback:
        return _collocation(p, cfg, key)
    pool = jax.random.uniform(key, (cfg.rar_pool, 2))
    blocks = residual_blocks(model, p, pool[:, 0:1], pool[:, 1:2], cfg)
    e = sum(w[k] * blocks[k] for k in range(len(blocks)))
    top = jnp.argsort(e)[-cfg.rar_keep :]
    return pool[top, 0:1], pool[top, 1:2]
