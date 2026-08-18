"""Network architectures for the JAX axial PINN.

Function approximators only: no physics, no residuals, no sampling. The
separation follows jaxpi2, where architectures are swappable precisely because
they know nothing about the equations they are used on.
"""

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

    ``B`` is frozen: drawn once and held under ``stop_gradient``, matching the torch
    twin's ``register_buffer``. ``scale_per_input`` gives each coordinate its own
    bandwidth — the rationale is in the torch twin.
    """

    B: jax.Array

    # PLR0913/PLR0917: five knobs plus JAX's explicit PRNG key. The torch twin
    # carries exactly the same five and passes; bundling them into a spec object
    # here would fork the two signatures, which is the thing AGENTS.md forbids.
    def __init__(  # noqa: PLR0913
        self,
        n_in: int,
        n_features: int,
        scale: float,
        key: jax.Array,
        scale_per_input: tuple[float, ...] | None = None,
        *,
        bands: tuple[float, ...] = (),
    ) -> None:
        s = jnp.full((n_in, 1), float(scale))
        if scale_per_input is not None:
            if len(scale_per_input) != n_in:
                msg = f"scale_per_input has {len(scale_per_input)} entries, need {n_in}"
                raise ValueError(msg)
            s = jnp.asarray(scale_per_input, dtype=jnp.float64).reshape(n_in, 1)
        mult = tuple(bands) or (1.0,)
        per = n_features // len(mult)
        raw = jax.random.normal(key, (n_in, n_features))
        cols, start = [], 0
        for k, b in enumerate(mult):
            n = per if k < len(mult) - 1 else n_features - per * (len(mult) - 1)
            cols.append(raw[:, start : start + n] * s * float(b))
            start += n
        self.B = jnp.concatenate(cols, axis=1)

    def __call__(self, x: jax.Array) -> jax.Array:
        proj = 2.0 * jnp.pi * (x @ jax.lax.stop_gradient(self.B))
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])


def fourier_scale_vector(cfg, n_in: int) -> tuple[float, ...] | None:  # noqa: ANN001
    """Per-input Fourier bandwidths, or ``None`` for an isotropic basis.

    Input order is ``(zeta, t)``, plus the level-set coordinate when it is on. Only
    ``zeta`` is scaled: the front is sharp in space and smooth in time, so raising
    the time bandwidth buys nothing and costs conditioning.
    """
    if cfg.fourier_scale_zeta is None:
        return None
    base = float(cfg.fourier_scale)
    return (base * float(cfg.fourier_scale_zeta),) + (base,) * (n_in - 1)


class AxialPinn(eqx.Module):
    """Field network, plus a precursor network when the kinetics loop is closed.

    Holds *only* the networks. ``AxialParams`` is passed to the free functions
    below so its numpy arrays stay out of the PyTree metadata, which Optax's tree
    operations require to be hashable — the same split the 0D JAX backend uses.
    """

    mlp: eqx.nn.MLP
    kin: eqx.nn.MLP | None
    embed: FourierEmbedding | None

    def __init__(self, cfg: AxialTrainConfig, key: jax.Array) -> None:
        # THE REFERENCE IMPLEMENTATION'S SPLIT: embedding takes `split(key)[0]` and the
        # field network `split(key)[1]`, so the same seed builds the same weights. This
        # was `split(key, 4)`, which handed both a different key -- identical shapes and
        # an identical 25 221 parameter count, entirely different values, and no way to
        # compare a run here with a reference one.
        #
        # The optional precursor head takes a FOLDED key rather than widening the split,
        # so turning it on cannot move the field network or the embedding.
        k_emb, k_field = jax.random.split(key)
        k_kin = jax.random.fold_in(key, 2)
        n_in = 2
        self.embed = (
            FourierEmbedding(
                n_in,
                cfg.fourier_features,
                cfg.fourier_scale,
                k_emb,
                fourier_scale_vector(cfg, n_in),
                bands=cfg.fourier_bands,
            )
            if cfg.fourier_features
            else None
        )
        if cfg.fourier_features:
            n_in = 2 * cfg.fourier_features
        self.mlp = eqx.nn.MLP(
            in_size=n_in,
            out_size=len(FIELDS),
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnp.tanh,
            key=k_field,
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
