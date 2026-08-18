"""The objective: squared residuals, averaged per time window and then across windows.

How the residual blocks are combined into one scalar. Kept apart from the blocks
themselves so a weighting scheme can be swapped, or removed, without touching the
physics — which is what makes an ablation a config change.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams


from pinn_sfr_transient.axial.jaxpinn.archs import AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.residuals import _blocks


def causal_loss(
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    pts: tuple,
    w: jax.Array | None = None,
) -> jax.Array:
    """Squared residual, summed over the equations and averaged over time windows.

    The average is taken **per time window and then across windows**, not over the points
    directly. A plain mean would let wherever the points happen to lie become a statement
    about where the residual matters, and boiling onset falls at 10.98 s of a 16.5 s
    window, so any density tilted toward early time would weight against the event the
    model exists to predict. Averaging within each of ``causal_chunks`` windows first
    keeps sampling density and loss weighting independent.

    No initial- or boundary-condition term appears: the ansatz satisfies them exactly. The
    four blocks enter with **equal weight**, because the variable scaling has already put
    them on a common magnitude.

    This is the reference implementation's objective, and it is now the only one. The
    causal ramp, the gradient-norm block balance and the pseudo-time proximal term have
    all been retired -- each was measured harmful, each defaulted to off, and each cost a
    branch on the hot path while being inert. ``w`` is accepted and ignored so a caller
    passing block weights still runs; it will go with the last of them.

    The chunking variable is **time**, and which member of ``pts`` holds it depends on the
    plan: the feedback plan collocates in time only, so ``pts = (that, ...)``, while the
    scattered plan returns ``(zeta, that)``. Reading ``pts[0]`` unconditionally chunked
    the scattered loss by *axial position*, which ran the average up the channel instead
    of forward in time.
    """
    del w  # accepted for call compatibility; the blocks are equally weighted
    blocks = _blocks(model, p, cfg, pts)
    e = sum(blocks)
    that = pts[0] if cfg.feedback else pts[1]
    idx = jnp.clip((that.reshape(-1) * cfg.causal_chunks).astype(int), 0, cfg.causal_chunks - 1)
    counts = jnp.bincount(idx, length=cfg.causal_chunks)
    sums = jnp.bincount(idx, weights=e, length=cfg.causal_chunks)
    return jnp.mean(sums / jnp.maximum(counts, 1))
