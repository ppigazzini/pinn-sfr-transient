"""Collocation samplers.

Where the residual is evaluated, independent of what the residual is. All draws
are keyed, and every set has a *constant* size so the jitted step never
recompiles.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.residuals import residual_blocks
from pinn_sfr_transient.axial.physics import kinetics_weights


# --- collocation ------------------------------------------------------------
def _collocation(
    p: AxialParams,
    cfg: AxialTrainConfig,
    key: jax.Array,
) -> tuple:
    """Uniform points over ``(zeta, t_hat)``; time-only when feedback is on.

    The uniform draw is bit-identical to the reference implementation's: same key
    derivation, same shape, same order.

    **Uniform, with no early-time cluster.** An extra ``n_colloc // 2`` points crammed
    into the first 40% of the window used to be drawn unconditionally here. It is
    retired: the companion repository measured it worse in every time window and worst
    where its own density was highest, and it was not defensible on its own terms
    either -- it made ``n_colloc = 500`` draw 750 points, so the batch knob did not mean
    what it said, and it put 60% of the sample in ``t_hat < 0.4`` where nothing happens
    while boiling onset at ``t_hat = 0.665`` sat in a window getting 13% instead of 20%.

    Two arguments and no options: the reference implementation's signature, with ``p``
    added only because the feedback plan needs the axial quadrature nodes. The front and
    level-set draws that used to take a ``model`` here, and the ``t_max`` window
    curriculum, are retired -- each was measured and none earned its branch.
    """
    # THE UNIFORM DRAW CONSUMES `split(key)[0]`, which is the reference implementation's
    # derivation exactly. It used to consume `split(key, 4)[0]`, a different stream, and
    # the comment defending that split argued it must not move -- while it was already
    # not the stream every published reference number was drawn from. The optional
    # features below take FOLDED keys instead of wider splits, so switching one on cannot
    # perturb the base draw; that is what a four-way split could not promise.
    k_uniform = jax.random.split(key)[0]
    if cfg.feedback:
        that = jax.random.uniform(k_uniform, (cfg.n_time, 1))
        zeta_q = jnp.asarray(p.zeta_nodes().reshape(-1, 1))
        weights = tuple(jnp.asarray(w) for w in kinetics_weights(p))
        return that, zeta_q, weights
    pts = jax.random.uniform(k_uniform, (cfg.n_colloc, 2))
    return pts[:, 0:1], pts[:, 1:2]


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
