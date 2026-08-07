"""Network architectures for the JAX axial PINN.

Function approximators only: no physics, no residuals, no sampling. The
separation follows jaxpi2, where architectures are swappable precisely because
they know nothing about the equations they are used on.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.physics import N_GROUPS

FIELDS: tuple[str, ...] = ("T_f", "T_cl", "T_s", "T_c", "alpha")
N_TEMPS: int = 4
_ALPHA_GATE: float = 10.0
_FRONT_MAX: float = 1.25
_NEWTON_ITERS: int = 5
_EXP_BOUND: float = 4.0
"""Bound on the exponent of the multiplicative ansatz; see the torch twin."""

"""The void head starts at ``sigmoid(0) ~ 0.5``, as in the torch twin.

Biasing it toward zero was tried and measured away; the torch module records the
numbers and ``docs/axial_nn.md`` section 7.1 the table.
"""


def _bounded_exp(x: jax.Array) -> jax.Array:
    """``exp`` with a smooth ceiling and floor, so the ansatz cannot overflow."""
    return jnp.exp(_EXP_BOUND * jnp.tanh(x / _EXP_BOUND))


class FourierEmbedding(eqx.Module):
    """Random Fourier features, ``x -> [sin(2 pi B x), cos(2 pi B x)]``.

    ``B`` is frozen: it is drawn once and held under ``stop_gradient``, matching
    the torch twin's ``register_buffer``.
    """

    B: jax.Array

    def __init__(self, n_in: int, n_features: int, scale: float, key: jax.Array) -> None:
        self.B = jax.random.normal(key, (n_in, n_features)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        proj = 2.0 * jnp.pi * (x @ jax.lax.stop_gradient(self.B))
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])


class ModifiedMLP(eqx.Module):
    """Two-encoder MLP of [Wang, Teng & Perdikaris 2021], the architecture jaxpi uses.

    Encoders ``U`` and ``V`` are computed once from the input and mixed into every
    hidden layer, ``h <- (1 - z) U + z V``, so the input reaches the last layer
    undiminished. Equinox's default ``Linear`` init is ``U(+/-1/sqrt(fan_in))`` on
    weights and biases, which is what the torch twin sets explicitly.
    """

    u: eqx.nn.Linear
    v: eqx.nn.Linear
    first: eqx.nn.Linear
    hidden: list
    out: eqx.nn.Linear

    def __init__(self, n_in: int, n_out: int, width: int, depth: int, key: jax.Array) -> None:
        keys = jax.random.split(key, depth + 3)
        self.u = eqx.nn.Linear(n_in, width, key=keys[0])
        self.v = eqx.nn.Linear(n_in, width, key=keys[1])
        self.first = eqx.nn.Linear(n_in, width, key=keys[2])
        self.hidden = [eqx.nn.Linear(width, width, key=k) for k in keys[3 : depth + 2]]
        self.out = eqx.nn.Linear(width, n_out, key=keys[depth + 2])

    def __call__(self, x: jax.Array) -> jax.Array:
        u, v = jnp.tanh(self.u(x)), jnp.tanh(self.v(x))
        h = jnp.tanh(self.first(x))
        for layer in self.hidden:
            z = jnp.tanh(layer(h))
            h = (1.0 - z) * u + z * v
        return self.out(h)


class AxialPinn(eqx.Module):
    """Field network, plus a precursor network when the kinetics loop is closed.

    Holds *only* the networks. ``AxialParams`` is passed to the free functions
    below so its numpy arrays stay out of the PyTree metadata, which Optax's tree
    operations require to be hashable — the same split the 0D JAX backend uses.
    """

    mlp: eqx.nn.MLP | ModifiedMLP
    kin: eqx.nn.MLP | None
    front: eqx.nn.MLP | None
    embed: FourierEmbedding | None

    def __init__(self, cfg: AxialTrainConfig, key: jax.Array) -> None:
        k_field, k_kin, k_front, k_emb = jax.random.split(key, 4)
        use_front = bool(cfg.front_net and cfg.void_closure)
        n_in = 3 if use_front else 2
        self.embed = (
            FourierEmbedding(n_in, cfg.fourier_features, cfg.fourier_scale, k_emb)
            if cfg.fourier_features
            else None
        )
        if cfg.fourier_features:
            n_in = 2 * cfg.fourier_features
        self.mlp = (
            ModifiedMLP(n_in, len(FIELDS), cfg.width, cfg.depth, k_field)
            if cfg.modified_mlp
            else eqx.nn.MLP(
                in_size=n_in,
                out_size=len(FIELDS),
                width_size=cfg.width,
                depth=cfg.depth,
                activation=jnp.tanh,
                key=k_field,
            )
        )
        # Precursors are functions of time alone, so a separate smaller network.
        self.kin = (
            eqx.nn.MLP(
                in_size=1,
                out_size=N_GROUPS,
                width_size=cfg.width // 2,
                depth=2,
                activation=jnp.tanh,
                key=k_kin,
            )
            if cfg.feedback
            else None
        )
        # Front-position network: one input, one output, so it is cheap next to
        # the field network. Off by default, like the torch twin.
        self.front = (
            eqx.nn.MLP(
                in_size=1,
                out_size=1,
                width_size=max(8, cfg.width // 4),
                depth=2,
                activation=jnp.tanh,
                key=k_front,
            )
            if use_front
            else None
        )
