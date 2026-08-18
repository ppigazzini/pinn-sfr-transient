"""The model: ansatz and residuals.

Two responsibilities the JAX twin keeps in separate modules — what the network
cannot get wrong, and the equations it is scored on. They live together here
because ``nn.Module`` owns its parameters and its forward pass, so splitting them
would mean splitting the class. That is torch's idiom rather than a design
choice, and it is the one place this package does not mirror ``axial.jaxpinn``.

Every residual calls ``axial.physics``, the same functions the reference solver
discretises, so the network and its ground truth cannot drift apart.
"""

from typing import TYPE_CHECKING

try:
    import torch  # ty: ignore
    from torch import nn  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.config import FloatArray

import numpy as np

from pinn_sfr_transient.axial import sodium
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
    reactivity_components,
    residual_normalisation,
)
from pinn_sfr_transient.axial.torchpinn.ansatz import (
    _fuel_temperature,
    _power_integral,
    _power_shape,
    _precursors,
)
from pinn_sfr_transient.axial.torchpinn.archs import (
    _ALPHA_GATE,
    _FRONT_MAX,
    FIELDS,
    MLP,
    N_TEMPS,
    BeignetPyramid,
    FourierEmbedding,
    LaplaceMix,
    ModifiedMLP,
    _bounded_exp,
    fourier_scale_vector,
    laplace_width,
)
from pinn_sfr_transient.axial.torchpinn.config import AxialTrainConfig


class AxialPinn(nn.Module):
    """Normalised-state PINN with hard initial and inlet conditions."""

    def __init__(self, p: AxialParams, cfg: AxialTrainConfig) -> None:
        super().__init__()
        self.p = p
        self.cfg = cfg
        torch.manual_seed(cfg.seed)  # before init: nn.init draws from the global RNG
        n_in = 3 if (cfg.front_net and cfg.void_closure) or cfg.level_set_input else 2
        self.embed: nn.Module | None = None
        if cfg.beignet_levels:
            # Replaces the random Fourier embedding rather than wrapping it: the paper
            # substitutes one for the other, and composing them would make any gain
            # unattributable (the compound-arm mistake of section 7.5.4).
            self.embed = (
                BeignetPyramid(
                    cfg.beignet_levels,
                    cfg.beignet_features,
                    cfg.beignet_base,
                    cfg.beignet_noise,
                    cfg.beignet_pad,
                )
                .to(cfg.device)
                .double()
            )
            n_in = self.embed.n_out
        elif cfg.fourier_features:
            self.embed = (
                FourierEmbedding(
                    n_in,
                    cfg.fourier_features,
                    cfg.fourier_scale,
                    fourier_scale_vector(cfg, n_in),
                    bands=cfg.fourier_bands,
                )
                .to(cfg.device)
                .double()
            )
            n_in = 2 * cfg.fourier_features
        if cfg.laplace_rates:
            # Wraps whatever embedding exists (or none, for `alone`), so the two
            # bases compose rather than one replacing the other.
            self.embed = (
                LaplaceMix(
                    self.embed,
                    cfg.laplace_rates,
                    cfg.laplace_mode,
                    float(p.t_end) * cfg.t_train_frac,
                )
                .to(cfg.device)
                .double()
            )
            n_in = laplace_width(
                cfg, 3 if (cfg.front_net and cfg.void_closure) or cfg.level_set_input else 2
            )
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
        # Front-position network: t_hat -> z_f. One input, one output, so it is
        # cheap next to the field network.
        self.use_front = bool(cfg.front_net and cfg.void_closure)
        self.T_boil = float(sodium.saturation_temperature(p.p_system) + p.dT_superheat)
        self.front = (
            MLP(1, max(8, cfg.width // 4), 2, n_in=1).to(cfg.device).double()
            if self.use_front
            else None
        )
        # Onset head: `(zeta*, t*)` as two raw scalars pushed through a sigmoid, so
        # both stay inside the domain by construction rather than by a penalty.
        # Initialised at logit 2.0 -> ~0.88, i.e. high in the channel and late in
        # the window, which is where onset is in every regime the reference maps.
        # A parameter, not a network: onset is TWO NUMBERS for a fixed set of
        # parameters. A network here would only be needed to make onset a function
        # of `void_worth_net`/`tau_pump` for the M9 sweep, which is a later step.
        self.onset_raw = (
            torch.nn.Parameter(torch.full((2,), 2.0, dtype=torch.float64, device=cfg.device))
            if cfg.onset_head
            else None
        )
        self.geo = line_geometry(p)
        # Fixed axial quadrature for the reactivity integrals (deviation note
        # section 3.5a of the plan): RAR may add arbitrary points to the field
        # residual, but an integral needs a rule, so the two collocation sets are
        # kept separate. Same nodes and weights as the reference's own sum.
        self.w_D, self.w_void = (torch.tensor(w, dtype=torch.float64) for w in kinetics_weights(p))
        self.zeta_q = torch.tensor(p.zeta_nodes().reshape(-1, 1), dtype=torch.float64)
        self.dT = p.P_0 / (p.w_0 * p.c_c)  # nominal core rise, the temperature scale
        # `t_hat = 1` is the end of the TRAINED horizon, which need not be
        # `p.t_end` — see `AxialTrainConfig.t_train_frac`.
        self.t_end = float(p.t_end) * cfg.t_train_frac
        # Variable scaling per residual block; ones when disabled.
        # The void block is absent when it is closed algebraically; the interface
        # block replaces it when the front network is on.
        self.n_fields = N_TEMPS if cfg.void_closure else len(FIELDS)
        self.n_blocks = self.n_fields + (1 if self.use_front else 0) + (1 if cfg.onset_head else 0)
        self.res_norm: tuple[float, ...] = (
            residual_normalisation(p, self.t_end) if cfg.residual_scaling else (1.0,) * len(FIELDS)
        )
        # The interface residual is already dimensionless and O(1).
        self.res_norm = (*self.res_norm, 1.0)
        # Precursors carry their own rate per group, `t_end * lambda_i`.
        lam = torch.tensor(p.lambda_i, dtype=torch.float64)
        self.c_norm = 1.0 / (self.t_end * lam) if cfg.residual_scaling else torch.ones_like(lam)

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

    def _raw(self, x: torch.Tensor) -> torch.Tensor:
        """Network output for a prepared input, with the embedding if one is set."""
        return self.net(self.embed(x) if self.embed is not None else x)

    def _level_set_coord(self, zeta: torch.Tensor, that: torch.Tensor) -> torch.Tensor:
        """``phi = (T_c - T_sat - dT_sup) / dT`` from a bootstrap pass (idea 3).

        `T_c` is this network's own output, so the coordinate depends on the thing
        it feeds. Resolved by evaluating once with ``phi = 0`` and using the
        resulting ``T_c``.

        **No `detach` here, deliberately.** The residual needs the *total*
        derivative of the state with respect to ``(zeta, t)``, and with ``phi`` in
        the input that includes the term flowing through ``phi``. Detaching would be
        cheaper, would still train, and would silently drop that term -- a wrong
        residual that produces plausible numbers, which is the defect class this
        project keeps finding. The cost is a second forward pass and a deeper graph.
        """
        zero = torch.zeros_like(zeta)
        raw0 = self._raw(torch.cat([zeta, that, zero], dim=1))
        temps0 = self.theta0(zeta)[:, :N_TEMPS] * _bounded_exp(that * raw0[:, :N_TEMPS])
        T_c0 = self.p.T_in + temps0[:, 3:4] * self.dT
        return (T_c0 - self.T_boil) / self.dT

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
        of arXiv:2604.23528, and exactly the thing to parameterise away. Constraining
        the ansatz to the physical manifold
        removes the region rather than penalising it.

        The structure is pinned to ``T_in`` at ``zeta = 0`` as a side effect, which
        is correct: its only coupling is to the coolant, held at ``T_in`` there by
        the inlet condition.
        """
        if self.use_front:
            # Signed distance to the front. A field that is kinked in (zeta, t) is
            # smooth in phi, which is what lets `T_c` carry the kink at all.
            x = torch.cat([zeta, that, zeta - self.front_position(that)], dim=1)
        elif self.cfg.level_set_input:
            x = torch.cat([zeta, that, self._level_set_coord(zeta, that)], dim=1)
        else:
            x = torch.cat([zeta, that], dim=1)
        raw = self._raw(x)
        temps = self.theta0(zeta)[:, :N_TEMPS] * _bounded_exp(that * raw[:, :N_TEMPS])
        if self.cfg.void_closure:
            # `b` underflows to exactly zero below saturation, so alpha = 0 at
            # t = 0 and at the inlet fall out of the closure -- no gate needed.
            T_c = self.p.T_in + temps[:, 3:4] * self.dT
            alpha = quasi_steady_void(T_c, self.p)
        else:
            gate = torch.tanh(_ALPHA_GATE * that) * torch.tanh(_ALPHA_GATE * zeta)
            alpha = gate * torch.sigmoid(raw[:, N_TEMPS : N_TEMPS + 1])
        return torch.cat([temps, alpha], dim=1)

    def front_position(self, that: torch.Tensor) -> torch.Tensor:
        """Predicted front height ``z_f(t_hat)``, in normalised height.

        Mapped into ``(0, _FRONT_MAX)`` so the network can place the front *above*
        the channel, which is how it says "no front yet" — the state for the first
        third of this transient. Values above one are never evaluated as a field
        position; they only switch the interface residual off through its mask.
        """
        return _FRONT_MAX * torch.sigmoid(self.front(that))

    def front_residual(self, that: torch.Tensor) -> torch.Tensor:
        """Squared interface residual ``[T_c(z_f, t) - T_sat - dT_sup] / dT``.

        Masked by ``b(T_c(1, t))``, the superheat switch at the channel outlet: a
        front exists inside the channel only once the top is boiling, and before
        that the condition has no solution to pin. The mask is the network's own
        output, so nothing here consults the reference.
        """
        p = self.p
        z_f = self.front_position(that)
        T_c_front = self.p.T_in + self.normalised_state(z_f, that)[:, 3:4] * self.dT
        top = torch.ones_like(that)
        T_c_top = self.p.T_in + self.normalised_state(top, that)[:, 3:4] * self.dT
        mask = boiling_fraction(T_c_top, p)
        T_boil = sodium.saturation_temperature(p.p_system) + p.dT_superheat
        return (mask * (T_c_front - T_boil) / self.dT).pow(2).squeeze(1)

    def state_and_grads(
        self, zeta: torch.Tensor, that: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(theta, d theta/d t_hat, d theta/d zeta)``.

        Two forward-mode passes rather than five reverse ones: the map is
        ``R^2 -> R^4``, so a single ``jvp`` per input direction yields all four
        components at once.

        The coordinates are **detached but not marked** ``requires_grad``. Forward mode
        carries the derivative in the tangent, so nothing here needs a reverse-mode leaf
        at the inputs; marking them built a backward graph down to ``zeta`` and
        ``t_hat`` whose gradients were computed by every ``loss.backward()`` and then
        discarded. It also made the whole residual stack uncompilable -- AOTAutograd
        refuses a graph that returns a tensor derived from an in-graph
        ``requires_grad_()``, because functionalisation drops the flag. Removing it
        leaves the loss bitwise identical and the parameter gradients agreeing to
        2.2e-16, one ulp against a maximum gradient of 8.2.

        It is also a plain eager win, the discarded backward work being real: together
        with the scalar fast path in ``_backend.xp``, the eager loop went from 7.2 to
        10.7 iterations per second at f256 with 500 points on 8 pinned cores.
        """
        z, h = zeta.detach(), that.detach()
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
            ((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * self.res_norm[k])
            .pow(2)
            .reshape(n_t, n_z)
            .mean(1)
            for k in range(self.n_fields)
        ]
        if self.use_front:
            blocks.append(self.front_residual(that))

        # Precursors: dc_i/dt_hat = t_end * lambda_i (P - c_i), Eq. 4.3-1 in
        # normalised time. Averaged over the six groups, as the 0D model does.
        d_phys = precursor_derivatives(c, power, self.p)
        blocks.append(((dc - self.t_end * d_phys) * self.c_norm).pow(2).mean(1))
        if self.onset_raw is not None:
            blocks.append(self.onset_residual())
        return tuple(blocks)

    def onset_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the head's ``(zeta*, t_hat*)``, both in ``(0, 1)`` by construction."""
        if self.onset_raw is None:
            msg = "onset_point() requires cfg.onset_head"
            raise RuntimeError(msg)
        pt = torch.sigmoid(self.onset_raw)
        return pt[0:1].reshape(1, 1), pt[1:2].reshape(1, 1)

    def onset_residual(self) -> torch.Tensor:
        """Square the two tangency conditions that define onset.

        ``R1 = (T_c - T_boil)/dT`` says the field reaches saturation; ``R2 =
        (dT_c/dzeta)/dT`` says it reaches it *tangentially*, which is what makes
        the point a first touch rather than any later crossing. Both are divided by
        the temperature scale so the block is O(1) like the others.

        Two things this does NOT do, deliberately. It does not consult the
        reference — the threshold is a sodium property, and the conditions are
        statements about the network's own field. And it does not pick the
        *earliest* solution: the sigmoid keeps the point in the domain and the
        initialisation puts it where onset is, but a later tangency would also
        satisfy both residuals. Whether that matters is exactly what the isolated
        study measures; a barrier could be added if it does.
        """
        z, t = self.onset_point()
        _, _, d_dz = self.state_and_grads(z, t)
        T_c = self.p.T_in + self.normalised_state(z, t)[:, 3:4] * self.dT
        r_value = (T_c - self.T_boil) / self.dT
        r_slope = d_dz[:, 3:4]
        return torch.cat([r_value, r_slope], dim=1).pow(2).mean(1)

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
        # so it only picks up the time scale. `res_norm` then divides each block
        # by its own characteristic rate so all five are O(1) (variable scaling).
        scales = [self.t_end / self.dT] * N_TEMPS + [self.t_end]
        blocks = [
            ((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * self.res_norm[k]).pow(2).squeeze(1)
            for k in range(self.n_fields)
        ]
        if self.use_front:
            blocks.append(self.front_residual(that))
        if self.onset_raw is not None:
            blocks.append(self.onset_residual())
        return tuple(blocks)

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
    def predict_reactivity_components(self, t: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Doppler and void reactivity separately, as the reference reports them.

        `predict_power` returns the net, and the net hides which mechanism is
        wrong. Plan A under-predicts `min rho/beta` by 26-28% on every seed
        (`docs/axial_nn.md` section 7.4.1); attributing that needs the split the
        reference already exposes as `rho_doppler` and `rho_void`.
        """
        that = torch.tensor(
            (np.asarray(t) / self.t_end).reshape(-1, 1), dtype=torch.float64, device=self.cfg.device
        )
        n_z = self.zeta_q.shape[0]
        zeta = self.zeta_q.repeat(that.shape[0], 1)
        fields = self.to_physical(self.normalised_state(zeta, that.repeat_interleave(n_z, dim=0)))
        T_f0 = self.p.T_in + self.theta0(self.zeta_q)[:, 0:1] * self.dT
        dop, void = reactivity_components(
            fields[0].reshape(-1, n_z),
            fields[4].reshape(-1, n_z),
            T_f0.reshape(1, n_z),
            self.w_D,
            self.w_void,
            self.p,
        )
        return dop.cpu().numpy().ravel(), void.cpu().numpy().ravel()

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
