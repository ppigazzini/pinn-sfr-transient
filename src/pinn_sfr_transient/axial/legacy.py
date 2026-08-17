"""Frozen reader for the `pinn-ulof` checkpoint corpus. Do not extend this module.

334 checkpoints, 89 MB, roughly 200 CPU-hours of training, produced by the companion
repository before this one had any way to save a model. They are the only surviving
evidence behind several results — two AdEMAMix divergences, the schedule-free stall, and
the collocation-count rungs where the boiling front never forms — and re-training one
costs hours against seconds to re-score.

They do not load with :mod:`pinn_sfr_transient.axial.checkpoint`, and the reason is
structural rather than cosmetic. ``equinox`` deserialises into a *skeleton*, so the
pytree has to match leaf for leaf:

=========================  ==================================  ==========================
                           ``pinn-ulof``                       this repository
=========================  ==================================  ==========================
``AxialPinn`` fields       ``(embed, mlp)``                    ``(mlp, kin, front,
                                                               embed, onset_raw)``
key splits                 2                                   4
``FourierEmbedding`` args  ``(n_in, n_features, scale, key)``  ``+ scale vector, bands``
optional trunks            none                                Beignet, Laplace, mod. MLP
budget field names         ``points``, ``iters``               ``n_colloc``,
                                                               ``lbfgs_iters``
=========================  ==================================  ==========================

The field *order* alone would misfill the tree, and the renamed budget fields would be
dropped by the unknown-key filter — silently losing the two axes a ladder groups by.

So this is a verbatim copy of that repository's network and configuration, and **frozen**
means frozen: no features, no refactors, no sharing with the live backends. Its single
job is to open 334 files that will never be written again. If
:mod:`pinn_sfr_transient.axial.jaxpinn` grows a layer, nothing here moves.

The *physics* is deliberately **not** copied. ``AxialParams`` is identical between the two
repositories field for field, and ``quasi_steady_void`` is the same function, so the void
closure is imported from :mod:`pinn_sfr_transient.axial.physics`. That is the right
coupling: if the closure changes, the reference changes with it, and the corpus *should*
be re-scored rather than quietly keep answering under the old truth.

    from pinn_sfr_transient.axial import legacy
    model, cfg, saved = legacy.load(Path("models/p10000_i50000_f64_s0_....eqx"))
    fields = legacy.predict(model, AxialParams(), cfg, traj.zeta, traj.t)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax

# float64 BEFORE any array is created, exactly as `jaxpinn/__init__.py` does it, and for
# a sharper reason here: the corpus was TRAINED in double precision, so a skeleton built
# at the default float32 would down-cast every weight on the way in. That is a silent
# loss -- the file still opens, the fields still look plausible, and every score is
# subtly wrong. This module can be imported without `jaxpinn`, so it cannot rely on that
# package having set the flag.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402 - must follow the x64 flag above
import numpy as np  # noqa: E402 - grouped with the import it follows

from pinn_sfr_transient.axial.physics import quasi_steady_void  # noqa: E402 - as above

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams, FloatArray
    from pinn_sfr_transient.axial.reference import AxialTrajectory

#: Temperatures in the state vector; the fifth field is the void fraction.
N_TEMPS: int = 4
#: Bound on the ansatz exponent, so a diverging iterate cannot overflow to inf.
_EXP_BOUND: float = 4.0


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The companion repository's hyper-parameters, exactly as its files record them.

    Field names are its names — ``points`` and ``iters``, not ``n_colloc`` and
    ``lbfgs_iters``. Renaming them here would make the headers unreadable, which is the
    whole problem this module exists to solve.
    """

    points: int = 10000
    iters: int = 50000
    fourier_features: int = 64
    fourier_scale: float = 2.0
    width: int = 64
    depth: int = 5
    lbfgs_history: int = 50
    seed: int = 0
    t_train_frac: float = 0.275


def _bounded_exp(x: jax.Array) -> jax.Array:
    """``exp`` with a smooth ceiling and floor."""
    return jnp.exp(_EXP_BOUND * jnp.tanh(x / _EXP_BOUND))


class FourierEmbedding(eqx.Module):
    """Random Fourier features, ``x -> [sin(2 pi B x), cos(2 pi B x)]``, frozen."""

    B: jax.Array

    def __init__(self, n_in: int, n_features: int, scale: float, key: jax.Array) -> None:
        self.B = jax.random.normal(key, (n_in, n_features)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map two inputs to ``2 * n_features`` sinusoidal features."""
        proj = 2.0 * jnp.pi * (x @ jax.lax.stop_gradient(self.B))
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])


class AxialPinn(eqx.Module):
    """Frozen embedding followed by a ``tanh`` MLP with five outputs.

    **Two fields, in this order, and one key split into two.** That is the pytree the
    334 files were serialised against; changing either breaks all of them at once.
    """

    embed: FourierEmbedding
    mlp: eqx.nn.MLP

    def __init__(self, cfg: TrainConfig, key: jax.Array) -> None:
        k_embed, k_mlp = jax.random.split(key)
        self.embed = FourierEmbedding(2, cfg.fourier_features, cfg.fourier_scale, k_embed)
        self.mlp = eqx.nn.MLP(
            in_size=2 * cfg.fourier_features,
            out_size=N_TEMPS + 1,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnp.tanh,
            key=k_mlp,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Raw network output at one point; the ansatz wraps this."""
        return self.mlp(self.embed(x))


# --- the analytic steady state, which is also the exact initial condition ----------
# Closed forms rather than calls into `AxialParams`, which is numpy and would break under
# a JAX tracer. Same expressions, differentiable.
def _power_shape(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Axial power shape, normalised so its mean over the channel is one."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (jnp.pi * k)) * jnp.sin(0.5 * jnp.pi * k)
    return jnp.cos(jnp.pi * k * (zeta - 0.5)) / norm


def _power_integral(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Cumulative axial power fraction: zero at the inlet, one at the outlet."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    half = 0.5 * jnp.pi * k
    return (jnp.sin(jnp.pi * k * (zeta - 0.5)) + jnp.sin(half)) / (2.0 * jnp.sin(half))


def _fuel_temperature(q: jax.Array, T_cl: jax.Array, area: float, p: AxialParams) -> jax.Array:
    """Invert the gap flux for ``T_f``: conduction plus radiation, by Newton iteration."""
    sigma = 5.670374419e-8
    T = T_cl + q / (p.h_gap * area)
    for _ in range(5):
        f = p.h_gap * area * (T - T_cl) + sigma * p.emissivity * area * (T**4 - T_cl**4) - q
        df = p.h_gap * area + 4.0 * sigma * p.emissivity * area * T**3
        T = T - f / df
    return T


def theta0(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Analytic steady profile, in normalised variables."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    T_c = p.T_in + dT * _power_integral(p, zeta)
    q_fuel = (1.0 - p.gamma_c) * p.P_0 * _power_shape(p, zeta) / p.H
    T_cl = T_c + q_fuel / (p.h_clad_coolant * 2.0 * jnp.pi * p.r_co)
    T_f = _fuel_temperature(q_fuel, T_cl, 2.0 * jnp.pi * p.r_fo, p)
    cols = [(T - p.T_in) / dT for T in (T_f, T_cl, T_c, T_c)]
    return jnp.concatenate([*cols, jnp.zeros_like(T_c)], axis=-1)


def normalised_state(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array
) -> jax.Array:
    """``theta(zeta, t_hat)`` with the hard constraints satisfied identically.

    Multiplicative ansatz ``theta = theta_0(zeta) exp(t_hat N(zeta, t_hat))``: ``exp(0) =
    1`` makes the initial condition exactly the steady profile, positivity keeps the
    logarithmic Doppler defined, and ``theta_0`` vanishing at ``zeta = 0`` for the coolant
    pins the single upstream boundary condition.
    """
    raw = model(jnp.concatenate([zeta, that]))
    base = theta0(p, zeta)
    temps = base[:N_TEMPS] * _bounded_exp(that * raw[:N_TEMPS])
    dT = p.P_0 / (p.w_0 * p.c_c)
    alpha = quasi_steady_void(p.T_in + temps[3:4] * dT, p)
    return jnp.concatenate([temps, alpha])


def horizon(p: AxialParams, cfg: TrainConfig) -> float:
    """End of the trained window [s]. ``t_hat = 1`` maps here, not to ``p.t_end``."""
    return float(p.t_end) * cfg.t_train_frac


def predict(
    model: AxialPinn, p: AxialParams, cfg: TrainConfig, zeta: FloatArray, t: FloatArray
) -> tuple[FloatArray, ...]:
    """Evaluate the five fields on a ``(zeta, t)`` grid, in physical units."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    zz, tt = np.meshgrid(np.asarray(zeta), np.asarray(t) / horizon(p, cfg), indexing="ij")
    flat_z = jnp.asarray(zz.reshape(-1, 1))
    flat_t = jnp.asarray(tt.reshape(-1, 1))
    theta = np.asarray(
        jax.vmap(lambda a, b: normalised_state(model, p, a, b))(flat_z, flat_t)
    ).reshape(*zz.shape, N_TEMPS + 1)
    temps = [p.T_in + theta[..., k] * dT for k in range(N_TEMPS)]
    return (*temps, theta[..., N_TEMPS])


def load(path: Path) -> tuple[AxialPinn, TrainConfig, str]:
    """Read a legacy checkpoint: its model, its configuration and when it was saved.

    Keys the configuration no longer defines are dropped rather than raised on. Files
    written before the clustered-collocation removal still carry ``drift_points``,
    ``points_file``, ``residual_scaling``, ``uniform_collocation`` and ``sf_warmup_frac``;
    without this filter all 334 stop opening at once.
    """
    with Path(path).open("rb") as f:
        head = json.loads(f.readline().decode())
        known = {fld.name for fld in dc_fields(TrainConfig)}
        cfg = TrainConfig(**{k: v for k, v in head["config"].items() if k in known})
        skeleton = AxialPinn(cfg, jax.random.PRNGKey(cfg.seed))
        return eqx.tree_deserialise_leaves(f, skeleton), cfg, head["saved_utc"]


def header(path: Path) -> dict:
    """Read a legacy checkpoint's header without loading the weights."""
    with Path(path).open("rb") as f:
        return json.loads(f.readline().decode())


def score(path: Path, traj: AxialTrajectory, p: AxialParams | None = None) -> dict[str, float]:
    """Score a legacy checkpoint by **this** repository's scorer, not the companion's.

    That is the point of importing the corpus rather than the numbers: one definition of
    every metric, applied to both the old models and any new one. Verified against the
    companion's own committed ladder — see :func:`pinn_sfr_transient.axial.ladder.errors`.
    """
    from pinn_sfr_transient.axial import config as _config  # noqa: PLC0415 - avoids a cycle
    from pinn_sfr_transient.axial.scoring import relative_l2  # noqa: PLC0415 - avoids a cycle

    p = p or _config.AxialParams()
    model, cfg, _ = load(path)
    return relative_l2(predict(model, p, cfg, traj.zeta, traj.t), traj, p)
