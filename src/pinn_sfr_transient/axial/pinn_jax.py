"""JAX/Equinox twin of the axial PINN (milestone M7).

The functional counterpart of :mod:`pinn_sfr_transient.axial.pinn_torch`: same
ansatz, same hard constraints, same Adam-to-L-BFGS schedule with causal weighting
and gradient-norm block weights, and — the point of the exercise — **the same
residual functions**. ``continuous_derivatives``, ``reactivity``,
``prompt_jump_power`` and ``precursor_derivatives`` all dispatch on their
argument type, so this backend re-uses them rather than re-deriving them. Only
the training mechanics differ.

Carrying two backends is a **correctness mechanism**, not bookkeeping.
``docs/neural_network.md`` §9 records that the 0D model's PyTorch
initialisation bug — power fit at ``L2 ~ 0.3`` — was identified because the JAX
twin fit well at the same budget. Two implementations that disagree tell you
something is wrong; one tells you nothing.

Framework differences, all mechanical:

* ``eqx.Module`` is an immutable PyTree, so a training step returns a *new*
  model; ``nn.Module`` mutates in place.
* ``eqx.filter_value_and_grad`` *returns* a gradient PyTree for Optax to consume;
  ``loss.backward()`` writes ``.grad``.
* ``eqx.filter_jit`` compiles the whole step, which needs **static shapes** — so
  RAR augments the collocation set with a *fixed* number of high-residual points
  rather than growing an unbounded reservoir.
* float64 comes from one global ``jax_enable_x64`` rather than per-tensor
  ``.double()``. It is essential here, not optional: at float32 the section 12.13
  correlations lose about eight significant digits.

Run (after ``uv sync --extra jax-cpu``; ``--extra jax-gpu`` for CUDA)::

    uv run python -m pinn_sfr_transient.axial.pinn_jax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.physics import (
    N_GROUPS,
    boiling_fraction,
    continuous_derivatives,
    kinetics_weights,
    line_geometry,
    precursor_derivatives,
    prompt_jump_power,
    quasi_steady_void,
    reactivity,
    residual_normalisation,
)
from pinn_sfr_transient.axial.reference import solve_reference

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "JAX backend requires `uv sync --extra jax-cpu` (or `--extra jax-gpu`)"
    raise SystemExit(msg) from exc

# float64 before any array is created: this problem does not survive float32.
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from pinn_sfr_transient.config import FloatArray

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


@dataclass(slots=True)
class AxialTrainConfig:
    """Hyper-parameters for the JAX axial PINN. Mirrors the torch config."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
    # Budget knobs match the torch twin exactly. They did not — 3000/300/1000
    # against 8000/500/2000 — so the default-configuration comparison the parity
    # study rests on was never a like-for-like one. Anything measured across
    # backends must be at an identical budget or it is measuring the schedule.
    adam_iters: int = 8000
    lbfgs_iters: int = 500
    lr: float = 1e-3

    # Dimensionless: the prefix sum is normalised by the total. See the torch twin.
    causal_eps: float = 0.0
    causal_chunks: int = 32
    # Fraction of `p.t_end` trained over — a scope decision; see the torch twin.
    t_train_frac: float = 1.0
    weight_update_every: int = 250
    weight_momentum: float = 0.9
    # Bound on the block-weight spread; off by default, see the torch twin.
    weight_max_ratio: float = 1.0
    # Variable scaling per residual block; see the torch twin.
    residual_scaling: bool = True
    # Eliminate the void algebraically (D-TH-3); see the torch twin.
    void_closure: bool = True
    # Front-position network (M8 option 2). Measured worse on every metric; see
    # the torch twin for the table.
    front_net: bool = False
    front_frac: float = 0.25  # share of collocation drawn near the predicted front

    # --- remedies for the moving front; every one measured in docs/axial_nn.md --
    # Time-window curriculum: train [0, w t_end] for growing w. Neutral in the
    # section 7.2.5 re-ablation.
    n_windows: int = 1
    # Random Fourier features against spectral bias [Tancik et al. 2020]. Measured
    # -11.1% at three seeds (section 7.2.6) but NOT adopted: it does not compose
    # with `modified_mlp`. 0 disables.
    fourier_features: int = 0
    fourier_scale: float = 2.0
    # Two-encoder "modified MLP" [Wang, Teng & Perdikaris 2021]. Measured -16.1%,
    # and likewise not adopted -- see section 7.2.6.
    modified_mlp: bool = False
    # Pseudo-time stepping [Wang, Koohy, Lu & Perdikaris, arXiv:2604.23528].
    # Measured harmful: under it the boiling front does not form at all. 0 disables.
    pts_every: int = 0
    pts_dtau: float = 1.0
    pts_growth: float = 1.5

    # RAR keeps a FIXED count so `jit` never recompiles (the torch twin grows an
    # unbounded reservoir instead — same idea, framework-appropriate form).
    rar_every: int = 2000
    rar_pool: int = 20000
    rar_keep: int = 400

    feedback: bool = False
    n_time: int = 128
    seed: int = 0
    log_every: int = 1000


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
def front_position(model: AxialPinn, that: jax.Array) -> jax.Array:
    """Predicted front height in ``(0, _FRONT_MAX)``; above 1 means "no front yet"."""
    return _FRONT_MAX * jax.nn.sigmoid(model.front(that))


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
    use_front = bool(cfg.front_net and cfg.void_closure and model.front is not None)
    x = (
        jnp.concatenate([zeta, that, zeta - front_position(model, that)])
        if use_front
        else (jnp.concatenate([zeta, that]))
    )
    raw = model.mlp(model.embed(x) if model.embed is not None else x)
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
    allp = jnp.concatenate(parts)
    return allp[:, 0:1], allp[:, 1:2]


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


# --- training ---------------------------------------------------------------
def train(
    p: AxialParams | None = None, cfg: AxialTrainConfig | None = None, *, verbose: bool = True
) -> tuple[AxialPinn, AxialParams, AxialTrainConfig]:
    """Adam (causal + adaptive block weights + RAR) then an L-BFGS polish."""
    p = p or AxialParams()
    cfg = cfg or AxialTrainConfig()
    key = jax.random.PRNGKey(cfg.seed)
    key, k_model = jax.random.split(key)
    model = AxialPinn(cfg, k_model)

    n_blocks = n_field_blocks(cfg) + (1 if uses_front(cfg) else 0) + (1 if cfg.feedback else 0)
    w = jnp.ones(n_blocks)
    sched = optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.adam_iters), alpha=0.1)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(
        model: AxialPinn, opt_state: optax.OptState, pts: tuple, w: jax.Array, anchor: tuple | None
    ) -> tuple:
        loss, grads = eqx.filter_value_and_grad(lambda m: causal_loss(m, p, cfg, pts, w, anchor))(
            model
        )
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return eqx.apply_updates(model, updates), opt_state, loss

    rar: tuple | None = None
    anchor: tuple | None = None
    dtau = cfg.pts_dtau
    for it in range(cfg.adam_iters):
        # Time-window curriculum: the horizon opens from 1/n_windows to 1 over
        # training, matching the torch twin. With n_windows = 1 this is a no-op.
        stage = min(int(it / max(cfg.adam_iters, 1) * cfg.n_windows) + 1, cfg.n_windows)
        t_max = stage / cfg.n_windows
        if it and it % cfg.rar_every == 0:
            key, rk = jax.random.split(key)
            rar = _rar_points(model, p, cfg, rk, w)
        # FRESH points every step, plus the fixed-size RAR set. The count is
        # constant so the jitted step never recompiles. Resampling is not a
        # detail: training on a frozen set is the collocation-overfitting mode of
        # arXiv:2605.30910, and holding the set fixed between RAR refreshes is
        # what made this backend stall at ~0.24 relative L2 while the torch twin
        # — which resamples every step — reached ~0.06 on the identical budget.
        key, ck = jax.random.split(key)
        pts = _merge(_collocation(p, cfg, ck, t_max, model), rar, feedback=cfg.feedback)
        if cfg.pts_every and it % cfg.pts_every == 0:
            # Re-anchor and relax. The anchor points are RESAMPLED every step:
            # the paper is explicit that pseudo-time stepping and resampling work
            # together, since a fixed anchor set is one more thing to overfit.
            key, ak = jax.random.split(key)
            a_zeta, a_that = _collocation(p, cfg, ak, t_max, model)[:2]
            a_state = jax.lax.stop_gradient(
                jax.vmap(lambda a, b, m=model: normalised_state(m, p, a, b, cfg))(a_zeta, a_that)
            )
            anchor = (a_zeta, a_that, a_state, dtau)
            dtau *= cfg.pts_growth
        if cfg.weight_max_ratio > 1.0 and it and it % cfg.weight_update_every == 0:
            gn = _block_grad_norms(model, p, cfg, pts)
            target = bounded_weights(gn.mean() / (gn + 1e-12), cfg.weight_max_ratio)
            w = cfg.weight_momentum * w + (1.0 - cfg.weight_momentum) * target
        model, opt_state, loss = step(model, opt_state, pts, w, anchor)
        if verbose and it % cfg.log_every == 0:
            print(f"[adam {it:6d}] loss={float(loss):.3e}")

    if cfg.lbfgs_iters > 0:
        model = _lbfgs_polish(model, p, cfg, pts, w, verbose=verbose)
    return model, p, cfg


def _lbfgs_polish(  # noqa: PLR0913 - polish needs the model, params, points and weights
    model: AxialPinn,
    p: AxialParams,
    cfg: AxialTrainConfig,
    pts: tuple,
    w: jax.Array,
    *,
    verbose: bool,
) -> AxialPinn:
    """Quasi-Newton polish on a fixed collocation set via ``optax.lbfgs``."""
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params: AxialPinn) -> jax.Array:
        return causal_loss(eqx.combine(params, static), p, cfg, pts, w)  # no proximal term

    before = float(loss_fn(params))
    opt = optax.lbfgs()
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(loss_fn)

    def body(_: int, carry: tuple) -> tuple:
        params, state = carry
        loss, grads = value_and_grad(params, state=state)
        updates, state = opt.update(grads, state, params, value=loss, grad=grads, value_fn=loss_fn)
        return optax.apply_updates(params, updates), state

    params, _ = jax.lax.fori_loop(0, cfg.lbfgs_iters, body, (params, state))
    after = float(loss_fn(params))
    # Same divergence guard as the torch twin: a bad line-search step can only
    # cost time, never accuracy.
    if not np.isfinite(after) or after > before:
        if verbose:
            print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        return model
    if verbose:
        print(f"[lbfgs done] loss={after:.3e}")
    return eqx.combine(params, static)


# --- inference --------------------------------------------------------------
def predict(
    model: AxialPinn,
    p: AxialParams,
    zeta: FloatArray,
    t: FloatArray,
    cfg: AxialTrainConfig | None = None,
) -> tuple[FloatArray, ...]:
    """Evaluate on a ``(zeta, t)`` grid, returning physical fields ``(n_z, n_t)``."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    cfg = cfg or AxialTrainConfig()
    t_end = horizon(p, cfg)
    zz, tt = np.meshgrid(zeta, t, indexing="ij")
    z = jnp.asarray(zz.reshape(-1, 1))
    h = jnp.asarray((tt / t_end).reshape(-1, 1))
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(z, h)
    out = [np.asarray(p.T_in + theta[:, k] * dT).reshape(zz.shape) for k in range(N_TEMPS)]
    out.append(np.asarray(theta[:, N_TEMPS]).reshape(zz.shape))
    return tuple(out)


def predict_power(
    model: AxialPinn, p: AxialParams, t: FloatArray, cfg: AxialTrainConfig | None = None
) -> tuple[FloatArray, FloatArray]:
    """Normalised power and net reactivity on a time grid (Plan A only)."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    cfg = cfg or AxialTrainConfig()
    t_end = horizon(p, cfg)
    zeta_q = jnp.asarray(p.zeta_nodes().reshape(-1, 1))
    w_D, w_void = (jnp.asarray(x) for x in kinetics_weights(p))
    that = jnp.asarray((np.asarray(t) / t_end).reshape(-1, 1))
    n_t, n_z = that.shape[0], zeta_q.shape[0]
    zeta = jnp.tile(zeta_q, (n_t, 1))
    t_rep = jnp.repeat(that, n_z, axis=0)
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(zeta, t_rep)
    T_f = p.T_in + theta[:, 0].reshape(n_t, n_z) * dT
    alpha = theta[:, N_TEMPS].reshape(n_t, n_z)
    T_f0 = p.T_in + jax.vmap(lambda z: theta0(p, z))(zeta_q)[:, 0] * dT
    rho = reactivity(T_f, alpha, T_f0[None, :], w_D, w_void, p)
    c = jax.vmap(lambda x: precursors(model, x))(that)
    power = prompt_jump_power(c, rho.reshape(-1, 1), p)
    return np.asarray(power).ravel(), np.asarray(rho).ravel()


def relative_l2(
    model: AxialPinn, p: AxialParams, traj: object, cfg: AxialTrainConfig | None = None
) -> dict[str, float]:
    """Relative ``L2`` per temperature field, plus the absolute voided-length error."""
    fields = predict(model, p, traj.zeta, traj.t, cfg)  # type: ignore[attr-defined]
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)  # type: ignore[attr-defined]
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS[:N_TEMPS], fields[:N_TEMPS], ref, strict=True)
    }
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H  # type: ignore[attr-defined]
    out["L_void_max_err_m"] = float(
        np.max(np.abs(fields[N_TEMPS].sum(axis=0) * dz - traj.voided_length))  # type: ignore[attr-defined]
    )
    return out


def main() -> None:
    """Train and report the relative L2 error against the reference."""
    model, p, cfg = train()
    traj = solve_reference(p, n_out=201, feedback=cfg.feedback)
    print("\nRelative L2 vs the reference:")
    for k, v in relative_l2(model, p, traj, cfg).items():
        print(f"  {k:18s}: {v:.3e}")


if __name__ == "__main__":
    main()
