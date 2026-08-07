"""Residual blocks: the physics the network is trained on.

Every block comes from ``axial.physics``, the same functions the reference solver
discretises, so the network and its ground truth cannot drift apart. Nothing here
knows how the blocks will be weighted or where the points came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.jaxpinn.ansatz import (
    _power_shape,
    front_position,
    horizon,
    normalised_state,
    precursors,
    state_and_grads,
    theta0,
)
from pinn_sfr_transient.axial.jaxpinn.archs import FIELDS, N_TEMPS, AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.physics import (
    boiling_fraction,
    continuous_derivatives,
    line_geometry,
    precursor_derivatives,
    prompt_jump_power,
    reactivity,
    residual_normalisation,
)


def front_residual(
    model: AxialPinn, p: AxialParams, that: jax.Array, cfg: AxialTrainConfig
) -> jax.Array:
    """Squared interface residual, masked by the outlet superheat switch.

    ``[T_c(z_f, t) - T_sat - dT_sup] / dT``. Before onset the condition has no
    solution to pin, so the mask -- the network's own outlet temperature through
    ``boiling_fraction`` -- switches it off. Nothing here consults the reference.
    """
    dT = p.P_0 / (p.w_0 * p.c_c)
    z_f = jax.vmap(lambda h: front_position(model, h))(that)
    state_at = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))
    T_c_front = p.T_in + state_at(z_f, that)[:, 3:4] * dT
    T_c_top = p.T_in + state_at(jnp.ones_like(that), that)[:, 3:4] * dT
    mask = boiling_fraction(T_c_top, p)
    T_boil = sodium.saturation_temperature(p.p_system) + p.dT_superheat
    return ((mask * (T_c_front - T_boil) / dT) ** 2).squeeze(1)


def uses_front(cfg: AxialTrainConfig) -> bool:
    """Report whether the front-position network is active."""
    return bool(cfg.front_net and cfg.void_closure)


def n_field_blocks(cfg: AxialTrainConfig) -> int:
    """Field residual blocks: four when the void is closed algebraically, else five."""
    return N_TEMPS if cfg.void_closure else len(FIELDS)


def _norm(p: AxialParams, cfg: AxialTrainConfig) -> tuple[float, ...]:
    """Per-block variable scaling, or ones when it is switched off."""
    if not cfg.residual_scaling:
        return (1.0,) * len(FIELDS)
    return residual_normalisation(p, horizon(p, cfg))


def residual_blocks(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array, cfg: AxialTrainConfig
) -> tuple[jax.Array, ...]:
    """Plan B blocks: prescribed power, scattered ``(zeta, t)`` collocation."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    t_end = horizon(p, cfg)
    theta, d_dt, d_dz = state_and_grads(model, p, zeta, that, cfg)
    temps = tuple(p.T_in + theta[:, k : k + 1] * dT for k in range(N_TEMPS))
    fields = (*temps, theta[:, N_TEMPS : N_TEMPS + 1])
    rhs = continuous_derivatives(
        that * t_end,
        *fields,
        d_dz[:, 3:4] * dT / p.H,
        d_dz[:, 4:5] / p.H,
        p,
        line_geometry(p),
        _power_shape(p, zeta),
        1.0,
    )
    scales = [t_end / dT] * N_TEMPS + [t_end]
    nrm = _norm(p, cfg)
    blocks = [
        (((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * nrm[k]) ** 2).squeeze(1)
        for k in range(n_field_blocks(cfg))
    ]
    if uses_front(cfg):
        blocks.append(front_residual(model, p, that, cfg))
    return tuple(blocks)


def closed_loop_blocks(  # noqa: PLR0913, PLR0917 - the tensor grid needs all of them
    model: AxialPinn,
    p: AxialParams,
    that: jax.Array,
    zeta_q: jax.Array,
    weights: tuple,
    cfg: AxialTrainConfig,
) -> tuple[jax.Array, ...]:
    """Plan A blocks: power is an output of the prompt-jump closure.

    The reactivity is an axial *integral*, so one amplitude couples every node at
    a given time to every other and the collocation has to be a tensor grid —
    ``n_time`` times the fixed axial quadrature. Field blocks are reduced over
    that quadrature so all six are one value per time, which is also the shape
    causal weighting wants.
    """
    dT = p.P_0 / (p.w_0 * p.c_c)
    t_end = horizon(p, cfg)
    w_D, w_void = weights
    n_t, n_z = that.shape[0], zeta_q.shape[0]
    zeta = jnp.tile(zeta_q, (n_t, 1))
    t_rep = jnp.repeat(that, n_z, axis=0)

    theta, d_dt, d_dz = state_and_grads(model, p, zeta, t_rep, cfg)
    temps = tuple(p.T_in + theta[:, k : k + 1] * dT for k in range(N_TEMPS))
    fields = (*temps, theta[:, N_TEMPS : N_TEMPS + 1])

    T_f0 = p.T_in + jax.vmap(lambda z: theta0(p, z))(zeta_q)[:, 0] * dT
    rho = reactivity(
        fields[0].reshape(n_t, n_z), fields[4].reshape(n_t, n_z), T_f0[None, :], w_D, w_void, p
    )

    batched = jax.vmap(lambda x: precursors(model, x))
    c, dc = jax.jvp(batched, (that,), (jnp.ones_like(that),))
    power = prompt_jump_power(c, rho.reshape(-1, 1), p)

    rhs = continuous_derivatives(
        t_rep * t_end,
        *fields,
        d_dz[:, 3:4] * dT / p.H,
        d_dz[:, 4:5] / p.H,
        p,
        line_geometry(p),
        _power_shape(p, zeta),
        jnp.repeat(power, n_z, axis=0),
    )
    scales = [t_end / dT] * N_TEMPS + [t_end]
    nrm = _norm(p, cfg)
    blocks = [
        (((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * nrm[k]) ** 2).reshape(n_t, n_z).mean(1)
        for k in range(n_field_blocks(cfg))
    ]
    if uses_front(cfg):
        blocks.append(front_residual(model, p, that, cfg))
    # Precursors carry their own rate per group, `t_end * lambda_i`.
    c_norm = 1.0 / (t_end * jnp.asarray(p.lambda_i)) if cfg.residual_scaling else 1.0
    blocks.append((((dc - t_end * precursor_derivatives(c, power, p)) * c_norm) ** 2).mean(1))
    return tuple(blocks)


# --- loss -------------------------------------------------------------------
def _blocks(model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, pts: tuple) -> tuple:
    if cfg.feedback:
        that, zeta_q, weights = pts
        return closed_loop_blocks(model, p, that, zeta_q, weights, cfg)
    return residual_blocks(model, p, pts[0], pts[1], cfg)
