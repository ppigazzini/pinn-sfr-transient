"""PINN for the single-phase axial channel — milestone M3 ("Plan B").

A network of ``(zeta, t)`` trained on the Chapter 3 residuals alone; no
reference data enters the loss. Power is *prescribed*, so this is the milestone
plan's Plan B: the thermal-hydraulics is validated before the kinetics feedback
is closed at M6.

**One set of equations.** The residual calls
:func:`~pinn_sfr_transient.axial.physics.continuous_derivatives`, the same
function — and therefore the same flux expressions — that the M2 reference
solver discretises. The network and its ground truth cannot drift apart by
construction rather than by review.

**Hard constraints instead of penalties**, following the 0D model's approach and
this project's stated preference:

* the **initial condition** is exact — the ansatz is
  ``theta = theta_0(zeta) + t_hat * N(...)``, so at ``t = 0`` every field equals
  the analytic steady profile for *any* network weights;
* the **inlet boundary condition** is exact — the coolant output carries an
  extra factor ``zeta``, so ``T_c(0, t) = T_in`` identically. Eq. 3.9-1 admits
  exactly one upstream condition, and this is it.

Neither appears in the loss, so the objective is pure physics and there is one
fewer competing term for the weighting to balance.

Run (after ``uv sync --extra torch-cpu``)::

    uv run python -m pinn_sfr_transient.axial.pinn_torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.physics import (
    N_GROUPS,
    continuous_derivatives,
    kinetics_weights,
    line_geometry,
    precursor_derivatives,
    prompt_jump_power,
    reactivity,
)
from pinn_sfr_transient.axial.reference import solve_reference

try:
    import torch  # ty: ignore
    from torch import nn  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.12 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from pinn_sfr_transient.config import FloatArray

FIELDS: tuple[str, ...] = ("T_f", "T_cl", "T_s", "T_c", "alpha")
N_TEMPS: int = 4


@dataclass(slots=True)
class AxialTrainConfig:
    """Hyper-parameters for the axial PINN and its schedule."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
    adam_iters: int = 8000
    lbfgs_iters: int = 500
    lr: float = 1e-3

    # Causal temporal weighting [Wang, Sankaran & Perdikaris 2024]
    causal_eps: float = 1.0
    causal_chunks: int = 32

    # Gradient-norm adaptive block weights [Wang, Teng & Perdikaris 2021]
    weight_update_every: int = 250
    weight_momentum: float = 0.9
    # Largest factor by which any block weight may depart from the geometric mean,
    # so the spread between the most- and least-weighted block is bounded by its
    # square. **Measured, not chosen**: unbounded (the original scheme) the
    # weights ran to 3.1e5-6.2e6 on `T_f` against 0.451 on the void, and every
    # field was worse for it — see `docs/axial_nn.md` section 7.2. 1.0 disables
    # the weighting entirely, which measures the same as 10.0 to within seed
    # noise; 10.0 keeps the mechanism live where it is useful.
    weight_max_ratio: float = 10.0

    # Residual-based adaptive refinement [Wu et al. 2023]
    rar_every: int = 2000
    rar_pool: int = 20000
    rar_add: int = 200
    rar_cap: int = 4000

    # Milestone M6: close the kinetics loop. With `feedback = False` the power is
    # prescribed (Plan B, M3); with it on, power becomes an output of the
    # prompt-jump closure and the reactivity integrals bring every axial node into
    # every residual — which is what forces the tensor collocation below.
    feedback: bool = False
    n_time: int = 128  # collocation times when feedback is on

    # --- remedies for the moving front (all measured in docs/axial_nn.md) ----
    # Time-window curriculum: train [0, w t_end] for growing w. Causal weighting
    # re-weights an already-global problem; windowing makes the problem itself
    # local in time, which is the stronger form of the same idea and the standard
    # treatment for long horizons with a front (jaxpi2's time-windowed training).
    n_windows: int = 1
    # Random Fourier feature embedding of the inputs, against spectral bias
    # [Tancik et al. 2020; Wang, Wang & Perdikaris 2021]. 0 disables.
    fourier_features: int = 0
    fourier_scale: float = 2.0
    # Two-encoder "modified MLP" [Wang, Teng & Perdikaris 2021], the architecture
    # jaxpi uses by default; multiplicative interactions carry the inputs to every
    # layer instead of letting them wash out with depth.
    modified_mlp: bool = False

    # Pseudo-time stepping against spurious solutions
    # [Wang, Koohy, Lu & Perdikaris, arXiv:2604.23528]. 0 disables.
    pts_every: int = 0
    pts_dtau: float = 1.0
    pts_growth: float = 1.5

    device: str = "cpu"
    seed: int = 0
    log_every: int = 1000


class FourierEmbedding(nn.Module):
    """Random Fourier features, ``x -> [sin(2 pi B x), cos(2 pi B x)]``.

    A plain MLP is spectrally biased toward smooth functions, which is the wrong
    prior for a boiling front that moves through the domain. Lifting the inputs
    into a random Fourier basis restores the high frequencies
    [Tancik et al. 2020; Wang, Wang & Perdikaris 2021].
    """

    def __init__(self, n_in: int, n_features: int, scale: float) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(n_in, n_features, dtype=torch.float64) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ModifiedMLP(nn.Module):
    """Two-encoder MLP of [Wang, Teng & Perdikaris 2021], as used by jaxpi.

    Two encoders ``U`` and ``V`` are computed once from the input and mixed into
    every hidden layer, ``h <- (1 - z) * U + z * V``. The inputs therefore reach
    the last layer undiminished, which a plain feed-forward stack cannot manage
    at depth.
    """

    def __init__(self, n_in: int, n_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.u = nn.Linear(n_in, width)
        self.v = nn.Linear(n_in, width)
        self.first = nn.Linear(n_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, n_out)
        for m in [self.u, self.v, self.first, self.out, *self.hidden]:
            k = m.in_features**-0.5
            nn.init.uniform_(m.weight, -k, k)
            nn.init.uniform_(m.bias, -k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.tanh(self.u(x)), torch.tanh(self.v(x))
        h = torch.tanh(self.first(x))
        for layer in self.hidden:
            z = torch.tanh(layer(h))
            h = (1.0 - z) * u + z * v
        return self.out(h)


class MLP(nn.Module):
    """Plain tanh MLP, ``n_in -> n_out``.

    Weights *and* biases from ``U(-k, k)`` with ``k = 1/sqrt(fan_in)``, matching
    Equinox. That is not cosmetic: ``docs/neural_network.md`` §9 records that the
    0D model's earlier ``xavier_normal`` + zero-bias init left the gradient-norm
    scheme unable to lift the stiff block's weight, and the fit was an order of
    magnitude worse. Same recipe, same init.
    """

    def __init__(self, n_out: int = 4, width: int = 64, depth: int = 5, n_in: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                k = m.in_features**-0.5
                nn.init.uniform_(m.weight, -k, k)
                nn.init.uniform_(m.bias, -k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AxialPinn(nn.Module):
    """Normalised-state PINN with hard initial and inlet conditions."""

    def __init__(self, p: AxialParams, cfg: AxialTrainConfig) -> None:
        super().__init__()
        self.p = p
        self.cfg = cfg
        torch.manual_seed(cfg.seed)  # before init: nn.init draws from the global RNG
        n_in = 2
        self.embed: nn.Module | None = None
        if cfg.fourier_features:
            self.embed = (
                FourierEmbedding(2, cfg.fourier_features, cfg.fourier_scale).to(cfg.device).double()
            )
            n_in = 2 * cfg.fourier_features
        core = (
            ModifiedMLP(n_in, len(FIELDS), cfg.width, cfg.depth)
            if cfg.modified_mlp
            else MLP(len(FIELDS), cfg.width, cfg.depth, n_in=n_in)
        )
        self.net = core.to(cfg.device).double()
        # Precursors are functions of time alone, so they get their own smaller
        # one-input network rather than a head on the (zeta, t) field network.
        kin = MLP(N_GROUPS, cfg.width // 2, 2, n_in=1) if cfg.feedback else None
        self.kin = kin.to(cfg.device).double() if kin is not None else None
        self.geo = line_geometry(p)
        # Fixed axial quadrature for the reactivity integrals (deviation note
        # section 3.5a of the plan): RAR may add arbitrary points to the field
        # residual, but an integral needs a rule, so the two collocation sets are
        # kept separate. Same nodes and weights as the reference's own sum.
        self.w_D, self.w_void = (torch.tensor(w, dtype=torch.float64) for w in kinetics_weights(p))
        self.zeta_q = torch.tensor(p.zeta_nodes().reshape(-1, 1), dtype=torch.float64)
        self.dT = p.P_0 / (p.w_0 * p.c_c)  # nominal core rise, the temperature scale
        self.t_end = float(p.t_end)

    # -- normalisation ------------------------------------------------------
    def to_physical(self, theta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Map the normalised state back to kelvin, plus the (already unitless) void."""
        temps = tuple(self.p.T_in + theta[:, k : k + 1] * self.dT for k in range(N_TEMPS))
        return (*temps, theta[:, N_TEMPS : N_TEMPS + 1])

    def theta0(self, zeta: torch.Tensor) -> torch.Tensor:
        """Analytic steady profile in normalised variables — the hard IC.

        Written in torch rather than lifted from
        :func:`~pinn_sfr_transient.axial.reference.steady_profile`, because the
        ansatz is differentiated with respect to ``zeta``: a detour through numpy
        would break the tangent (and does — ``jvp`` cannot trace ``.numpy()``).
        The two are asserted equal in the tests, so there is still one definition
        being checked rather than two being trusted.
        """
        p = self.p
        T_c = p.T_in + self.dT * _power_integral(p, zeta)
        q_fuel = (1.0 - p.gamma_c) * p.P_0 * _power_shape(p, zeta) / p.H
        T_cl = T_c + q_fuel / (p.h_clad_coolant * 2.0 * np.pi * p.r_co)
        T_f = _fuel_temperature(q_fuel, T_cl, 2.0 * np.pi * p.r_fo, p)
        cols = [(T - p.T_in) / self.dT for T in (T_f, T_cl, T_c, T_c)]
        cols.append(torch.zeros_like(T_c))  # the nominal channel is void-free
        return torch.cat(cols, dim=1)

    # -- the ansatz ---------------------------------------------------------
    def normalised_state(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        """``theta(zeta, t_hat)`` with every hard constraint satisfied identically.

        Three constraints, none of them in the loss:

        * initial condition — ``exp(0) = 1``, so ``t = 0`` is exact;
        * **positivity** — ``theta_0 >= 0`` and the exponential is positive, so
          every temperature stays at or above the inlet, ``T >= T_in``;
        * coolant inlet — ``theta_c0(0) = 0``, so the multiplicative form pins
          ``T_c(0, t) = T_in`` for free, with no separate gate;
        * void — the fifth column is a **sigmoid**, so ``alpha`` cannot leave
          ``[0, 1)`` for any weights, and its own gate enforces both that the
          channel starts void-free and that no void crosses the subcooled inlet.
          The void equation is advective with ``u > 0`` and so, like the coolant
          temperature, admits exactly one upstream condition.

        **The multiplicative form replaced an additive one, and that was a
        formulation fix rather than a refactor.** With ``theta = theta_0 + t_hat
        N`` nothing bounded the temperatures below: measured, the optimiser drove
        ``T_f`` from 722 K to -1 K over 115 iterations *while the loss fell*, and
        the logarithmic Doppler of Eq. 4.5-3 then returned NaN. The residual was
        perfectly content in that nonphysical region — the spurious-solution mode
        of arXiv:2604.23528, and exactly what REPORT-01 section 5.2 item 8 says to
        parameterise away. Constraining the ansatz to the physical manifold
        removes the region rather than penalising it.

        The structure is pinned to ``T_in`` at ``zeta = 0`` as a side effect, which
        is correct: its only coupling is to the coolant, held at ``T_in`` there by
        the inlet condition.
        """
        x = torch.cat([zeta, that], dim=1)
        raw = self.net(self.embed(x) if self.embed is not None else x)
        temps = self.theta0(zeta)[:, :N_TEMPS] * _bounded_exp(that * raw[:, :N_TEMPS])
        gate = torch.tanh(_ALPHA_GATE * that) * torch.tanh(_ALPHA_GATE * zeta)
        alpha = gate * torch.sigmoid(raw[:, N_TEMPS : N_TEMPS + 1])
        return torch.cat([temps, alpha], dim=1)

    def state_and_grads(
        self, zeta: torch.Tensor, that: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(theta, d theta/d t_hat, d theta/d zeta)``.

        Two forward-mode passes rather than five reverse ones: the map is
        ``R^2 -> R^4``, so a single ``jvp`` per input direction yields all four
        components at once.
        """
        z = zeta.detach().requires_grad_(requires_grad=True)
        h = that.detach().requires_grad_(requires_grad=True)
        theta, d_dt = torch.func.jvp(
            lambda a: self.normalised_state(z, a), (h,), (torch.ones_like(h),)
        )
        _, d_dz = torch.func.jvp(lambda a: self.normalised_state(a, h), (z,), (torch.ones_like(z),))
        return theta, d_dt, d_dz

    # -- residuals ----------------------------------------------------------
    def closed_loop_blocks(self, that: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Residual blocks with the kinetics closed (milestone M6).

        The reactivity is an axial *integral* (Eq. 4.5-3, Eq. 4.5-25), so a single
        power amplitude couples every node at a given time to every other. The
        collocation therefore has to be a **tensor grid** — ``n_time`` times times
        the fixed axial quadrature — rather than scattered points: you cannot
        evaluate an integral on a random cloud. This is the two-collocation-set
        design of the plan's section 3.5a, in its simplest correct form.

        Returns the four field blocks, the void block, and the precursor block.
        """
        n_t, n_z = that.shape[0], self.zeta_q.shape[0]
        zeta = self.zeta_q.repeat(n_t, 1)
        t_rep = that.repeat_interleave(n_z, dim=0)
        theta, d_dt, d_dz = self.state_and_grads(zeta, t_rep)
        fields = self.to_physical(theta)

        T_f0 = self.p.T_in + self.theta0(self.zeta_q)[:, 0:1] * self.dT
        T_f_g = fields[0].reshape(n_t, n_z)
        alpha_g = fields[4].reshape(n_t, n_z)
        rho = reactivity(T_f_g, alpha_g, T_f0.reshape(1, n_z), self.w_D, self.w_void, self.p)

        # Precursors and their time derivative in one forward-mode pass, exactly
        # as the fields get theirs.
        c, dc = torch.func.jvp(lambda h: _precursors(self, h), (that,), (torch.ones_like(that),))
        power = prompt_jump_power(c, rho.reshape(-1, 1), self.p)  # (n_t, 1)

        amp = power.repeat_interleave(n_z, dim=0)
        rhs = continuous_derivatives(
            t_rep * self.t_end,
            *fields,
            d_dz[:, 3:4] * self.dT / self.p.H,
            d_dz[:, 4:5] / self.p.H,
            self.p,
            self.geo,
            _power_shape(self.p, zeta),
            amp,
        )
        # Reduce each field block over the axial quadrature so every block is one
        # value per time. That is what makes the six comparable, and it is also
        # the natural shape for causal weighting, which chunks in time.
        scales = [self.t_end / self.dT] * N_TEMPS + [self.t_end]
        blocks = [
            (d_dt[:, k : k + 1] - scales[k] * rhs[k]).pow(2).reshape(n_t, n_z).mean(1)
            for k in range(len(FIELDS))
        ]

        # Precursors: dc_i/dt_hat = t_end * lambda_i (P - c_i), Eq. 4.3-1 in
        # normalised time. Averaged over the six groups, as the 0D model does.
        d_phys = precursor_derivatives(c, power, self.p)
        blocks.append((dc - self.t_end * d_phys).pow(2).mean(1))
        return tuple(blocks)

    def residual_blocks(self, zeta: torch.Tensor, that: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Squared residual of each field, shape ``(N,)`` each.

        In normalised variables ``d theta / d t_hat = (t_end / dT) dT/dt``, so a
        residual is ``d theta/d t_hat - (t_end/dT) * f_physical``. Every physical
        term comes from :func:`continuous_derivatives`, shared with the reference.
        """
        theta, d_dt, d_dz = self.state_and_grads(zeta, that)
        fields = self.to_physical(theta)
        rhs = continuous_derivatives(
            that * self.t_end,
            *fields,
            d_dz[:, 3:4] * self.dT / self.p.H,
            d_dz[:, 4:5] / self.p.H,
            self.p,
            self.geo,
            _power_shape(self.p, zeta),
            1.0,
        )
        # Temperatures are scaled by t_end/dT; the void is already dimensionless,
        # so it only picks up the time scale.
        scales = [self.t_end / self.dT] * N_TEMPS + [self.t_end]
        return tuple(
            (d_dt[:, k : k + 1] - scales[k] * rhs[k]).pow(2).squeeze(1) for k in range(len(FIELDS))
        )

    @torch.no_grad()
    def predict_power(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Normalised power and net reactivity on a time grid (Plan A only)."""
        that = torch.tensor(
            (np.asarray(t) / self.t_end).reshape(-1, 1), dtype=torch.float64, device=self.cfg.device
        )
        n_z = self.zeta_q.shape[0]
        zeta = self.zeta_q.repeat(that.shape[0], 1)
        fields = self.to_physical(self.normalised_state(zeta, that.repeat_interleave(n_z, dim=0)))
        T_f0 = self.p.T_in + self.theta0(self.zeta_q)[:, 0:1] * self.dT
        rho = reactivity(
            fields[0].reshape(-1, n_z),
            fields[4].reshape(-1, n_z),
            T_f0.reshape(1, n_z),
            self.w_D,
            self.w_void,
            self.p,
        )
        c = _precursors(self, that)
        power = prompt_jump_power(c, rho.reshape(-1, 1), self.p)
        return power.cpu().numpy().ravel(), rho.cpu().numpy().ravel()

    @torch.no_grad()
    def predict(self, zeta: FloatArray, t: FloatArray) -> tuple[FloatArray, ...]:
        """Evaluate on a ``(zeta, t)`` mesh grid, returning physical fields ``(n_z, n_t)``."""
        zz, tt = np.meshgrid(zeta, t, indexing="ij")
        z = torch.tensor(zz.reshape(-1, 1), dtype=torch.float64, device=self.cfg.device)
        h = torch.tensor(
            (tt / self.t_end).reshape(-1, 1), dtype=torch.float64, device=self.cfg.device
        )
        fields = self.to_physical(self.normalised_state(z, h))
        return tuple(f.cpu().numpy().reshape(zz.shape) for f in fields)


_ALPHA_GATE: float = 10.0
"""Sharpness of the void gate; large enough to saturate away from the boundaries."""

"""**On the void head's initialisation — a real asymmetry, and a cure that failed.**

The output layer is initialised ``U(+/-1/sqrt(width))``, so ``sigmoid(raw)`` sits
at ~0.5 at every interior point on iteration zero, while the reference void field
is identically zero over ~96% of the channel and over all of ``t < 10.8 s``. The
void is not an inert output either — it degrades ``film_coefficient``, shifts
``alpha_D`` between its flooded and voided values, and enters Eq. 4.5-25. The
temperatures, by contrast, start *exactly* on the steady profile because their
ansatz is multiplicative with ``exp(0) = 1``. The asymmetry is genuine.

The obvious cure — ``sigmoid(raw - 4)``, so the head starts at 0.018 — was
implemented and then **measured, at three seeds against an ``n_axial = 160``
reference, and it does not earn its place**: mean relative ``L2`` went 0.226 to
0.227 on ``T_cl``, 0.068 to 0.103 on ``T_s`` and 0.114 to 0.124 on ``T_c``. It
did cut the ``T_f`` seed spread from 12.5x to 1.8x, which is interesting and not
sufficient. Reverted, and recorded in ``docs/axial_nn.md`` section 7.1 with the
table, because the measurement is worth more than the change would have been:
seed variance dominates this model, and the void's real blocker is the block
weighting, not the initialisation.
"""

_EXP_BOUND: float = 4.0
"""Bound on the exponent of the multiplicative ansatz.

``exp(S tanh(x/S))`` equals ``exp(x)`` for small ``x`` but saturates at
``exp(+/-S)``, so the ansatz cannot overflow however hard the optimiser pushes.
``S = 4`` allows a factor ``e^4 ~ 55`` on the steady excess temperature; the
reference needs at most 9.6, so the bound never binds and ``tanh`` stays in its
linear region where the gradient is healthy.
"""


def _bounded_exp(x: torch.Tensor) -> torch.Tensor:
    """``exp`` with a smooth ceiling and floor — see :data:`_EXP_BOUND`."""
    return torch.exp(_EXP_BOUND * torch.tanh(x / _EXP_BOUND))


def _bounded_weights(target: torch.Tensor, cap: float) -> torch.Tensor:
    """Renormalise block weights to unit geometric mean, then clamp to ``[1/cap, cap]``.

    ``cap = 1`` collapses every weight to one, i.e. no weighting at all.
    """
    if cap <= 1.0:
        return torch.ones_like(target)
    return (target / torch.exp(torch.log(target).mean())).clamp(1.0 / cap, cap)


def _precursors(model: AxialPinn, that: torch.Tensor) -> torch.Tensor:
    """``c_i(t_hat)`` with ``c(0) = 1`` exact and ``c > 0`` guaranteed.

    ``c = exp(t_hat N(t_hat))``, bounded by :func:`_bounded_exp`, does both: the
    ``t_hat`` factor pins the initial condition for any weights, and the
    exponential makes the precursors positive by construction. Positivity is
    what makes
    ``P = sum(beta_i c_i)/(beta - rho)`` unable to reach zero — the collapse mode
    that REPORT-01 section 5.2 spends its length on.
    """
    return _bounded_exp(that * model.kin(that))


def _power_shape(p: AxialParams, zeta: torch.Tensor) -> torch.Tensor:
    """Axial power shape on a tensor (closed form, so autodiff-safe)."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (np.pi * k)) * np.sin(0.5 * np.pi * k)
    return torch.cos(np.pi * k * (zeta - 0.5)) / norm


def _power_integral(p: AxialParams, zeta: torch.Tensor) -> torch.Tensor:
    """Cumulative axial power fraction ``F(zeta)`` on a tensor; ``F(0)=0``, ``F(1)=1``."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    half = 0.5 * np.pi * k
    return (torch.sin(np.pi * k * (zeta - 0.5)) + np.sin(half)) / (2.0 * np.sin(half))


def _fuel_temperature(
    q: torch.Tensor, T_cl: torch.Tensor, area: float, p: AxialParams, iters: int = 5
) -> torch.Tensor:
    """Invert Eq. 3.3-4 for the fuel temperature; radiation makes it nonlinear.

    A fixed unrolled Newton rather than a convergence loop: the iteration count
    must not depend on the data for the graph to be traceable, and 40 steps is
    past convergence for this smooth scalar problem: measured, 4 iterations already
    reach machine precision, so 5 carries a margin of one.
    """
    sigma = 5.670374419e-8
    T_f = T_cl + q / (p.h_gap * area)
    for _ in range(iters):
        f = area * (p.h_gap * (T_f - T_cl) + p.emissivity * sigma * (T_f**4 - T_cl**4)) - q
        T_f = T_f - f / (area * (p.h_gap + 4.0 * p.emissivity * sigma * T_f**3))
    return T_f


class Trainer:
    """Adam (causal weighting + adaptive block weights + RAR) then an L-BFGS polish."""

    def __init__(self, model: AxialPinn, cfg: AxialTrainConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.dev = cfg.device
        n_blocks = len(FIELDS) + (1 if cfg.feedback else 0)
        self.block_w = torch.ones(n_blocks, dtype=torch.float64, device=self.dev)
        self.rar = torch.empty(0, 2, dtype=torch.float64, device=self.dev)
        # Pseudo-time stepping state: the anchor is the previous pseudo-step's
        # solution, held fixed while the network takes an implicit step toward it.
        self.pts_anchor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.pts_dtau = cfg.pts_dtau

    def _blocks(self, zeta: torch.Tensor, that: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Residual blocks for whichever plan is active."""
        if self.cfg.feedback:
            return self.model.closed_loop_blocks(that)
        return self.model.residual_blocks(zeta, that)

    def collocation(self, t_max: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform points over ``(zeta, t_hat)``, clustered early, plus the RAR reservoir."""
        if self.cfg.feedback:
            # Plan A collocates in TIME only: the axial direction is the fixed
            # quadrature the reactivity integral needs (section 3.5a).
            that = torch.rand(self.cfg.n_time, 1, dtype=torch.float64, device=self.dev) * t_max
            early = (
                torch.rand(self.cfg.n_time // 2, 1, dtype=torch.float64, device=self.dev)
                * 0.4
                * t_max
            )
            allt = torch.cat([that, early], dim=0)
            return allt, allt
        n = self.cfg.n_colloc
        pts = torch.rand(n, 2, dtype=torch.float64, device=self.dev)
        pts[:, 1] *= t_max
        early = torch.rand(n // 2, 2, dtype=torch.float64, device=self.dev)
        early[:, 1] *= 0.4 * t_max  # fastest dynamics live early in the window
        allp = (
            torch.cat([pts, early, self.rar], dim=0)
            if self.rar.numel()
            else torch.cat([pts, early], dim=0)
        )
        return allp[:, 0:1], allp[:, 1:2]

    def _anchor_points(self, t_max: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a genuine ``(zeta, t_hat)`` pair for the pseudo-time anchor.

        Under Plan A :meth:`collocation` returns *times* in both slots, because
        the residual there needs only the time axis — the axial direction is the
        fixed quadrature. Feeding that pair straight to ``normalised_state``
        evaluated the ansatz with times standing in for ``zeta``, so the proximal
        term pulled toward a state on the wrong manifold. Rebuild the tensor grid
        the Plan A state is actually defined on.
        """
        zeta, that = self.collocation(t_max)
        if not self.cfg.feedback:
            return zeta, that
        n_z = self.model.zeta_q.shape[0]
        return self.model.zeta_q.repeat(that.shape[0], 1), that.repeat_interleave(n_z, dim=0)

    def _pointwise(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(zeta, that)
        return sum(self.block_w[k] * blocks[k] for k in range(len(blocks)))

    def pseudo_time_step(self, t_max: float = 1.0) -> None:
        """Re-anchor the pseudo-time step and relax it [arXiv:2604.23528].

        Plain residual minimisation is free to sit in any low-residual basin,
        including ones no physical solution occupies — the paper's central point,
        and something this model has already been bitten by once (``T_f`` driven
        negative while the loss fell). Pseudo-time stepping instead solves the
        implicit-Euler problem

        ``(u - u_prev) / dtau + R(u) = 0``

        which adds a proximal pull toward the previous iterate. A jump to a
        distant spurious basin costs ``||u - u_prev||^2 / dtau``, so the optimiser
        has to *walk* to a solution rather than teleport to one. ``dtau`` grows
        each step, so the anchor relaxes and the limit recovers ordinary residual
        minimisation.

        The anchor points are **resampled** at every step: the paper is explicit
        that pseudo-time stepping and resampling work together, since a fixed
        anchor set is one more thing to overfit.
        """
        zeta, that = self._anchor_points(t_max)
        with torch.no_grad():
            state = self.model.normalised_state(zeta, that).detach()
        self.pts_anchor = (zeta.detach(), that.detach(), state)
        self.pts_dtau *= self.cfg.pts_growth

    def _pts_penalty(self) -> torch.Tensor:
        """Proximal term ``||u - u_prev||^2 / dtau``; zero when the anchor is unset."""
        if self.pts_anchor is None:
            return torch.zeros((), dtype=torch.float64, device=self.dev)
        zeta, that, prev = self.pts_anchor
        now = self.model.normalised_state(zeta, that)
        return (now - prev).pow(2).mean() / self.pts_dtau

    def causal_loss(
        self, zeta: torch.Tensor, that: torch.Tensor, t_max: float = 1.0
    ) -> torch.Tensor:
        """Time-chunked loss with causal weights [Wang, Sankaran & Perdikaris 2024].

        The chunks span the *current window*, not the whole horizon, so causal
        weighting keeps its resolution as the window grows.
        """
        e = self._pointwise(zeta, that)
        chunks = self.cfg.causal_chunks
        idx = torch.clamp((that.reshape(-1) / max(t_max, 1e-12) * chunks).long(), max=chunks - 1)
        losses = torch.stack(
            [e[idx == m].mean() if bool((idx == m).any()) else e.sum() * 0.0 for m in range(chunks)]
        )
        with torch.no_grad():
            w = torch.exp(-self.cfg.causal_eps * (torch.cumsum(losses, 0) - losses))
        return (w * losses).mean() + self._pts_penalty()

    def update_block_weights(self, zeta: torch.Tensor, that: torch.Tensor) -> None:
        """Balance the blocks by gradient norm [Wang, Teng & Perdikaris 2021], **bounded**.

        The scheme sets ``lambda_k = mean(g)/g_k``, so a block whose gradient
        falls as it is fitted earns an ever-larger weight — a positive feedback
        with nothing to stop it. Measured over three seeds, the weights reached
        3.1e5 to 6.2e6 on ``T_f`` while the void block pinned at 0.451, a spread
        of up to 5e6, and the run-to-run spread in the ``T_f`` error was 10.4x.
        Bounding the ratio removes both: every field improves and the seed spread
        collapses to 1.1-1.2x. The full table is in ``docs/axial_nn.md``
        section 7.2.

        Only *ratios* between blocks can matter — Adam is scale-invariant to a
        global factor, which is exactly the argument REPORT-01 D39 uses to show
        fixed per-equation scaling is a no-op — so the target is renormalised to
        unit geometric mean before clamping. That leaves the relative balance
        untouched and bounds only the spread.
        """
        blocks = self._blocks(zeta, that)
        params = [q for q in self.model.parameters() if q.requires_grad]
        norms = []
        for b in blocks:
            grads = torch.autograd.grad(b.mean(), params, retain_graph=True, allow_unused=True)
            sq = sum(
                (g.pow(2).sum() for g in grads if g is not None),
                start=torch.zeros((), device=self.dev),
            )
            norms.append(torch.sqrt(sq + 1e-30))
        gn = torch.stack(norms)
        with torch.no_grad():
            target = _bounded_weights(gn.mean() / (gn + 1e-12), self.cfg.weight_max_ratio)
            m = self.cfg.weight_momentum
            self.block_w = m * self.block_w + (1.0 - m) * target

    @torch.no_grad()
    def rar_refine(self, t_max: float = 1.0) -> None:
        """Append the worst-residual candidates to the reservoir [Wu et al. 2023]."""
        if self.cfg.feedback:
            return  # Plan A's collocation is a tensor grid; RAR would break the quadrature
        pool = torch.rand(self.cfg.rar_pool, 2, dtype=torch.float64, device=self.dev)
        pool[:, 1] *= t_max
        e = self._pointwise(pool[:, 0:1], pool[:, 1:2])
        top = torch.topk(e, min(self.cfg.rar_add, e.numel())).indices
        self.rar = torch.cat([self.rar, pool[top]], dim=0)[-self.cfg.rar_cap :]

    def _reset_rar(self) -> None:
        """Drop the reservoir when the window grows; its points are stale."""
        self.rar = torch.empty(0, 2, dtype=torch.float64, device=self.dev)

    def train(self, *, verbose: bool = True) -> AxialPinn:
        """Run the full schedule and return the trained model."""
        cfg = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.adam_iters), eta_min=cfg.lr * 0.1
        )
        for it in range(cfg.adam_iters):
            # Time-window curriculum: the horizon opens from `1/n_windows` to 1
            # over training, so the network solves a short transient first and
            # extends it. Causal weighting *re-weights* a globally posed problem;
            # windowing makes the problem itself local in time, which is the
            # stronger form of the same idea. With `n_windows = 1` this is a
            # no-op and the schedule is the original one.
            stage = min(int(it / cfg.adam_iters * cfg.n_windows) + 1, cfg.n_windows)
            t_max = stage / cfg.n_windows
            if it and it % cfg.weight_update_every == 0:
                self.update_block_weights(*self.collocation(t_max))
            if it and it % cfg.rar_every == 0:
                self.rar_refine(t_max)
            if cfg.pts_every and it % cfg.pts_every == 0:
                self.pseudo_time_step(t_max)
            opt.zero_grad()
            loss = self.causal_loss(*self.collocation(t_max), t_max)
            loss.backward()
            opt.step()
            sched.step()
            if verbose and it % cfg.log_every == 0:
                w = [f"{v:.1e}" for v in self.block_w.tolist()]
                print(f"[adam {it:6d}] t<={t_max:.2f} loss={loss.item():.3e} w=[{','.join(w)}]")
        if cfg.lbfgs_iters > 0:
            self._lbfgs(verbose=verbose)
        return self.model

    def _lbfgs(self, *, verbose: bool) -> None:
        """Quasi-Newton polish on a fixed collocation set, with a divergence guard."""
        zeta, that = self.collocation()
        before = self.causal_loss(zeta, that).item()
        snapshot = [q.detach().clone() for q in self.model.parameters()]
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=self.cfg.lbfgs_iters,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = self.causal_loss(zeta, that)
            loss.backward()
            return loss

        opt.step(closure)
        after = self.causal_loss(zeta, that).item()
        if not np.isfinite(after) or after > before:
            with torch.no_grad():
                for q, saved in zip(self.model.parameters(), snapshot, strict=True):
                    q.copy_(saved)
            if verbose:
                print(f"[lbfgs] reverted: {before:.3e} -> {after:.3e} (kept Adam)")
        elif verbose:
            print(f"[lbfgs done] loss={after:.3e}")


def relative_l2(model: AxialPinn, traj: object) -> dict[str, float]:
    """Relative ``L2`` error of every field against the held-out reference."""
    fields = model.predict(traj.zeta, traj.t)  # type: ignore[attr-defined]
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)  # type: ignore[attr-defined]
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS[:N_TEMPS], fields[:N_TEMPS], ref, strict=True)
    }
    # The void is near zero over most of the domain, so a relative L2 there is
    # dominated by its denominator. Report the absolute voided-length error in
    # metres instead -- the quantity M4 is actually judged on.
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H  # type: ignore[attr-defined]
    out["L_void_max_err_m"] = float(
        np.max(np.abs(fields[N_TEMPS].sum(axis=0) * dz - traj.voided_length))  # type: ignore[attr-defined]
    )
    return out


def train(p: AxialParams | None = None, cfg: AxialTrainConfig | None = None) -> AxialPinn:
    """Build and train the axial PINN."""
    p = p or AxialParams()
    cfg = cfg or AxialTrainConfig()
    return Trainer(AxialPinn(p, cfg), cfg).train()


def main() -> None:
    """Train, then report the relative L2 error against the M2 reference."""
    p = AxialParams()
    model = train(p)
    traj = solve_reference(p, n_out=201)
    print("\nRelative L2 vs the M2 reference:")
    for k, v in relative_l2(model, traj).items():
        print(f"  {k:4s}: {v:.3e}")


if __name__ == "__main__":
    main()
