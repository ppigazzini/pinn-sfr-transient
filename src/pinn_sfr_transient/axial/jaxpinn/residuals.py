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


def onset_point(model: AxialPinn) -> tuple[jax.Array, jax.Array]:
    """Return the head's ``(zeta*, t_hat*)``, both in ``(0, 1)`` by construction."""
    if model.onset_raw is None:
        msg = "onset_point() requires cfg.onset_head"
        raise RuntimeError(msg)
    pt = jax.nn.sigmoid(model.onset_raw)
    return pt[0:1], pt[1:2]


def onset_residual(model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig) -> jax.Array:
    """Square the two tangency conditions that define onset — the torch twin's rationale.

    ``R1 = (T_c - T_boil)/dT`` says the field reaches saturation; ``R2 =
    (dT_c/dzeta)/dT`` says it reaches it *tangentially*, which is what makes the
    point a first touch rather than any later crossing.

    Neither condition consults the reference: the threshold is a sodium property
    and both are statements about the network's own field. Neither picks the
    *earliest* tangency either — the initialisation puts the point where onset is,
    but a later tangency satisfies both residuals too, and whether that matters is
    what the isolated study measures.
    """
    dT = p.P_0 / (p.w_0 * p.c_c)
    z, t = onset_point(model)
    _, _, d_dz = state_and_grads(model, p, z.reshape(1, 1), t.reshape(1, 1), cfg)
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(
        z.reshape(1, 1), t.reshape(1, 1)
    )
    T_boil = sodium.saturation_temperature(p.p_system) + p.dT_superheat
    r_value = (p.T_in + theta[:, 3:4] * dT - T_boil) / dT
    r_slope = d_dz[:, 3:4]
    return (jnp.concatenate([r_value, r_slope], axis=1) ** 2).mean(1)


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


def residual_vector(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array, cfg: AxialTrainConfig
) -> tuple[jax.Array, ...]:
    """Return the **signed** field residuals, one array per block, before squaring.

    :func:`residual_blocks` squares these and is what training minimises. The signed
    form exists because a Gauss-Newton or natural-gradient method needs the Jacobian
    of ``r``, not of ``r**2`` — and taking a square root back would lose the sign,
    which is the one thing those methods need.

    Factored out rather than transcribed: two copies of a residual is the defect
    class this project has hit most often, so `residual_blocks` calls this and
    squares the result. The front and onset blocks are **not** included, because
    both are already-squared quantities with their own masks and neither is part of
    the least-squares system a Gauss-Newton step forms.
    """
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
    return tuple(
        ((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * nrm[k]).squeeze(1)
        for k in range(n_field_blocks(cfg))
    )


def residual_blocks(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array, cfg: AxialTrainConfig
) -> tuple[jax.Array, ...]:
    """Plan B blocks: prescribed power, scattered ``(zeta, t)`` collocation."""
    blocks = [r**2 for r in residual_vector(model, p, zeta, that, cfg)]
    if uses_front(cfg):
        blocks.append(front_residual(model, p, that, cfg))
    if cfg.onset_head:
        blocks.append(onset_residual(model, p, cfg))
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
    if cfg.onset_head:
        blocks.append(onset_residual(model, p, cfg))
    return tuple(blocks)


# --- loss -------------------------------------------------------------------
def _blocks(model: AxialPinn, p: AxialParams, cfg: AxialTrainConfig, pts: tuple) -> tuple:
    if cfg.feedback:
        that, zeta_q, weights = pts
        return closed_loop_blocks(model, p, that, zeta_q, weights, cfg)
    return residual_blocks(model, p, pts[0], pts[1], cfg)
