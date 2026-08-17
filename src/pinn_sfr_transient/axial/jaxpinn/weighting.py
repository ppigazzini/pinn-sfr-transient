"""Loss weighting: causal ramp, block balance, proximal anchor.

How the residual blocks are combined into one scalar. Kept apart from the blocks
themselves so a weighting scheme can be swapped, or removed, without touching the
physics — which is what makes an ablation a config change.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

import equinox as eqx

from pinn_sfr_transient.axial.jaxpinn.ansatz import normalised_state
from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.residuals import _blocks, n_field_blocks, uses_front


def causal_weights(losses: jax.Array, eps: float) -> jax.Array:
    """``exp(-eps * prefix / total)`` — causal weights, normalised so ``eps`` is unitless.

    The un-normalised form makes ``eps`` carry the reciprocal units of the loss,
    so variable scaling silently switched causality off (the ramp collapsed to a
    2% tilt). See the torch twin for the measurement.
    """
    prefix = jnp.cumsum(losses) - losses
    return jnp.exp(-eps * prefix / (losses.sum() + 1e-30))


def pts_penalty(
    model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, anchor: tuple | None
) -> jax.Array:
    """Proximal term ``||u - u_prev||^2 / dtau``; zero when the anchor is unset.

    Plain residual minimisation may sit in any low-residual basin, including ones
    no physical solution occupies. The implicit-Euler form adds a pull toward the
    previous iterate, so the optimiser has to walk to a solution rather than
    teleport to one [arXiv:2604.23528]. Measured harmful here (§7.2.5).
    """
    if anchor is None:
        return jnp.zeros(())
    zeta, that, prev, dtau = anchor
    now = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(zeta, that)
    return ((now - prev) ** 2).mean() / dtau


def causal_loss(  # noqa: PLR0913, PLR0917 - the proximal anchor is optional state
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    pts: tuple,
    w: jax.Array,
    anchor: tuple | None = None,
) -> jax.Array:
    """Time-chunked loss with causal weights [Wang, Sankaran & Perdikaris 2024].

    The chunking variable is **time**, and which member of ``pts`` holds it
    depends on the plan: Plan A collocates in time only, so ``pts = (that, ...)``,
    while Plan B collocates over ``(zeta, t)`` and ``_collocation`` returns them
    in that order. Reading ``pts[0]`` unconditionally chunked the Plan B loss by
    **axial position**, which made the causal ramp run up the channel instead of
    forward in time — the cause of the 3-10x backend disagreement recorded as D40
    in ``docs/axial_nn.md`` section 7.2, where "the causal-chunk reduction" was
    listed as an untested suspect. The torch twin was always correct.
    """
    blocks = _blocks(model, p, cfg, pts)
    e = sum(w[k] * blocks[k] for k in range(len(blocks)))
    that = pts[0] if cfg.feedback else pts[1]
    idx = jnp.clip((that.reshape(-1) * cfg.causal_chunks).astype(int), 0, cfg.causal_chunks - 1)
    counts = jnp.bincount(idx, length=cfg.causal_chunks)
    sums = jnp.bincount(idx, weights=e, length=cfg.causal_chunks)
    losses = sums / jnp.maximum(counts, 1)
    cw = jax.lax.stop_gradient(causal_weights(losses, cfg.causal_eps))
    return (cw * losses).mean() + pts_penalty(model, p, cfg, anchor)


def _block_grad_norms(
    model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, pts: tuple
) -> jax.Array:
    """Gradient norm of each residual block [Wang, Teng & Perdikaris 2021]."""
    n = n_field_blocks(cfg) + (1 if uses_front(cfg) else 0) + (1 if cfg.feedback else 0)
    norms = []
    for k in range(n):
        grads = eqx.filter_grad(lambda m, k=k: _blocks(m, p, cfg, pts)[k].mean())(model)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
        norms.append(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves) + 1e-30))
    return jnp.stack(norms)


def bounded_weights(target: jax.Array, cap: float) -> jax.Array:
    """Renormalise block weights to unit geometric mean, then clamp to ``[1/cap, cap]``.

    The unbounded scheme lets a well-fitted block's weight run away — measured to
    3.1e5-6.2e6 against 0.451 on the void block, with every field worse for it
    (``docs/axial_nn.md`` section 7.2). Only ratios can matter, since Adam is
    scale-invariant to a global factor, so renormalising first bounds the spread
    without touching the balance. ``cap = 1`` disables the weighting entirely.
    """
    if cap <= 1.0:
        return jnp.ones_like(target)
    return jnp.clip(target / jnp.exp(jnp.log(target).mean()), 1.0 / cap, cap)
