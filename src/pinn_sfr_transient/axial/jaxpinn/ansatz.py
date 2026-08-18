"""The ansatz: analytic steady profile plus every hard constraint.

This layer decides what the network *cannot* get wrong — the initial condition,
the inlet, positivity, the void bound — independently of which architecture
produces the raw outputs and of which residual consumes them.
"""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

from pinn_sfr_transient.axial.jaxpinn.archs import (
    _ALPHA_GATE,
    _NEWTON_ITERS,
    N_TEMPS,
    AxialPinn,
    _bounded_exp,
)
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.physics import quasi_steady_void


# --- the analytic steady profile, in JAX ------------------------------------
def _power_shape(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Axial power shape, closed form so it is autodiff-safe."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (jnp.pi * k)) * jnp.sin(0.5 * jnp.pi * k)
    return jnp.cos(jnp.pi * k * (zeta - 0.5)) / norm


def _power_integral(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Cumulative axial power fraction ``F(zeta)``; ``F(0) = 0``, ``F(1) = 1``."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    half = 0.5 * jnp.pi * k
    return (jnp.sin(jnp.pi * k * (zeta - 0.5)) + jnp.sin(half)) / (2.0 * jnp.sin(half))


def _fuel_temperature(q: jax.Array, T_cl: jax.Array, area: float, p: AxialParams) -> jax.Array:
    """Invert Eq. 3.3-4 for the fuel temperature; the radiation term makes it nonlinear.

    A fixed unrolled Newton, because the iteration count must not depend on the
    data for the graph to be traceable. Five steps: measured, four already reach
    machine precision.
    """
    sigma = 5.670374419e-8
    T_f = T_cl + q / (p.h_gap * area)
    for _ in range(_NEWTON_ITERS):
        f = area * (p.h_gap * (T_f - T_cl) + p.emissivity * sigma * (T_f**4 - T_cl**4)) - q
        T_f = T_f - f / (area * (p.h_gap + 4.0 * p.emissivity * sigma * T_f**3))
    return T_f


def theta0(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Analytic steady profile in normalised variables — the hard initial condition."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    T_c = p.T_in + dT * _power_integral(p, zeta)
    q_fuel = (1.0 - p.gamma_c) * p.P_0 * _power_shape(p, zeta) / p.H
    T_cl = T_c + q_fuel / (p.h_clad_coolant * 2.0 * jnp.pi * p.r_co)
    T_f = _fuel_temperature(q_fuel, T_cl, 2.0 * jnp.pi * p.r_fo, p)
    cols = [(T - p.T_in) / dT for T in (T_f, T_cl, T_c, T_c)]
    return jnp.concatenate([*cols, jnp.zeros_like(T_c)], axis=-1)


# --- ansatz -----------------------------------------------------------------
def _raw(model: AxialPinn, x: jax.Array) -> jax.Array:
    """Network output for a prepared input, with the embedding if one is set."""
    return model.mlp(model.embed(x) if model.embed is not None else x)


def normalised_state(
    model: AxialPinn,
    p: AxialParams,
    zeta: jax.Array,
    that: jax.Array,
    cfg: AxialTrainConfig | None = None,
) -> jax.Array:
    """``theta(zeta, t_hat)`` with every hard constraint satisfied identically.

    Same constraints as the torch twin, and for the same reasons. The ansatz is
    **multiplicative**, ``theta = theta_0 exp(t_hat N)``: ``exp(0) = 1`` makes the
    initial condition exact, ``theta_0 >= 0`` with a positive exponential keeps
    every temperature at or above the inlet, and ``theta_c0(0) = 0`` pins
    ``T_c(0, t) = T_in`` for free — Eq. 3.9-1 admits exactly one upstream
    condition and this is it, with no separate gate.

    The additive form it replaced let the optimiser drive ``T_f`` negative while
    the loss fell, which made the logarithmic Doppler of Eq. 4.5-3 return NaN.
    """
    cfg = cfg or AxialTrainConfig()
    x = jnp.concatenate([zeta, that])
    raw = _raw(model, x)
    base = theta0(p, zeta)
    temps = base[:N_TEMPS] * _bounded_exp(that * raw[:N_TEMPS])
    if cfg.void_closure:
        # `b` underflows to exactly zero below saturation, so the void-free
        # initial and inlet conditions fall out of the closure -- no gate needed.
        alpha = quasi_steady_void(p.T_in + temps[3:4] * (p.P_0 / (p.w_0 * p.c_c)), p)
    else:
        gate = jnp.tanh(_ALPHA_GATE * that) * jnp.tanh(_ALPHA_GATE * zeta)
        alpha = gate * jax.nn.sigmoid(raw[-1:])
    return jnp.concatenate([temps, alpha])


def precursors(model: AxialPinn, that: jax.Array) -> jax.Array:
    """``c_i(t_hat) = exp(t_hat N(t_hat))`` — ``c(0) = 1`` exact, ``c > 0`` always.

    Positivity is structural: with ``c > 0`` and the pole guard on ``beta - rho``,
    ``P = sum(beta_i c_i)/(beta - rho)`` cannot reach zero, so the power-collapse
    mode is removed by construction rather than avoided by training.
    """
    return _bounded_exp(that * model.kin(that))


def state_and_grads(
    model: AxialPinn,
    p: AxialParams,
    zeta: jax.Array,
    that: jax.Array,
    cfg: AxialTrainConfig | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(theta, d theta/d t_hat, d theta/d zeta)`` at a batch of points.

    Two forward-mode passes for a map ``R^2 -> R^5``: one ``jvp`` per input
    direction yields all five components at once, which is cheaper than five
    reverse passes.
    """

    def one(z: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        f = lambda a, b: normalised_state(model, p, a, b, cfg)  # noqa: E731
        theta, d_dt = jax.jvp(lambda b: f(z, b), (h,), (jnp.ones_like(h),))
        _, d_dz = jax.jvp(lambda a: f(a, h), (z,), (jnp.ones_like(z),))
        return theta, d_dt, d_dz

    return jax.vmap(one)(zeta, that)


# --- residuals --------------------------------------------------------------
def horizon(p: AxialParams, cfg: AxialTrainConfig) -> float:
    """End of the trained window [s]; ``t_hat = 1`` maps here, not to ``p.t_end``."""
    return float(p.t_end) * cfg.t_train_frac
