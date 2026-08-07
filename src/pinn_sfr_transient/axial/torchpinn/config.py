"""Training hyper-parameters for the torch axial PINN.

Separated from the networks, the model and the trainer so a knob can be read
without importing torch, and so the dependency graph stays a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AxialTrainConfig:
    """Hyper-parameters for the axial PINN and its schedule."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
    adam_iters: int = 8000
    lbfgs_iters: int = 500
    lr: float = 1e-3

    # Causal temporal weighting [Wang, Sankaran & Perdikaris 2024]. `causal_eps`
    # is DIMENSIONLESS — the prefix sum is normalised by the total (see
    # `_causal_weights`), so it sets the ramp's log dynamic range and survives a
    # change in the loss scale. It did not before, and `residual_scaling` moved
    # that scale by ~1e10 and switched causality off without saying so.
    # Default 0.0 (off): measured, causality costs accuracy on every field here
    # (`docs/axial_nn.md` section 7.2.4). Non-zero re-enables it meaningfully.
    causal_eps: float = 0.0
    causal_chunks: int = 32

    # Fraction of `AxialParams.t_end` the network is trained over.
    #
    # **A scope decision, not a fitting knob.** With prescribed power the channel
    # reaches the top of the section 12.13 sodium property range at ~16.5 s and
    # the reference solver stops there, by design (deviation D-SCOPE-1: no
    # melting, no cladding motion, no relocation). Training to `t_end = 60 s`
    # therefore asks the network to satisfy residuals over a region where the
    # model itself does not apply — 72% of the horizon — and because the ansatz
    # is one smooth function of `t_hat`, the fully-voided state it settles on
    # there propagates back to `t = 0`. Measured: void at t = 0.25 s at the
    # channel *inlet*, against a reference that is identically zero until 10.75 s
    # and then boils at the top.
    #
    # This does not use the reference in the loss (REPORT-01 section 4.1). The
    # horizon is a property of the model's validity range, not of its solution.
    # Plan A needs no truncation: with feedback the transient is self-limiting
    # and completes 60 s inside the property range.
    t_train_frac: float = 1.0

    # Gradient-norm adaptive block weights [Wang, Teng & Perdikaris 2021]
    weight_update_every: int = 250
    weight_momentum: float = 0.9
    # Largest factor by which any block weight may depart from the geometric mean,
    # so the spread between the most- and least-weighted block is bounded by its
    # square. **Measured, not chosen**, twice: unbounded, the weights ran to
    # 3.1e5-6.2e6 on `T_f` against 0.451 on the void and every field was worse
    # for it; and once `residual_scaling` removes the *static* imbalance the
    # adaptive part is worse than nothing on all four fields. So the default is
    # off. Values > 1 re-enable it, bounded. See `docs/axial_nn.md` section 7.2.
    weight_max_ratio: float = 1.0
    # Variable scaling: divide each residual block by its own characteristic rate
    # so all blocks are O(1) [Ko & Park, JCP 529 113860 (2025)]. The natural rates
    # span 813x here (`physics.residual_scales`), almost all of it the void, which
    # is 8x beyond what `weight_max_ratio` can undo — so the fixed part has to
    # come out analytically. REPORT-01 D39 measured this as a no-op, correctly,
    # back when the adaptive weights were unbounded and cancelled it.
    residual_scaling: bool = True

    # Eliminate the void algebraically instead of solving it (deviation D-TH-3).
    # `alpha` fills a node in 0.71 ms against a 0.113 s transport time, so it is a
    # fast variable slaved to `T_c` -- the same elimination D-KIN-1 makes for the
    # prompt neutron mode. Removes a residual block whose normalised rate is 8.5e4
    # and lets the front appear analytically where `T_c` crosses saturation.
    void_closure: bool = True

    # Milestone M8, option 2: a second network for the front position `z_f(t)`,
    # following Chang, Lin & Lai (arXiv:2512.14010). Requires `void_closure`,
    # under which the front IS the level set `T_c = T_sat + DTS`, so the front
    # network adds no new information -- its value is mechanical:
    #   * a dedicated, localised residual for a front the field residual averages
    #     over a domain in which it occupies 2%;
    #   * `phi = zeta - z_f(t)` as a third network input, which is what lets `T_c`
    #     carry a kink at the front -- the mechanism that attacks `T_c`, and `T_c`
    #     is now what bounds the front's accuracy;
    #   * collocation concentrated on the front, which RAR cannot supply because it
    #     samples by residual magnitude and the front residual is small everywhere.
    # It also makes onset time and location direct network outputs, which is the
    # form the M4 acceptance criteria ask for.
    # **Default off: measured worse on every metric.** Three seeds against
    # option 1 alone: T_f +1.6%, T_cl +1.3%, T_s +2.8%, T_c +3.0%, onset time
    # +2.1%, voided length -27%, and `max alpha` fell from 1.0000 to 0.92 because
    # one seed's front only half formed. The prediction that motivated it holds
    # against it: under `void_closure` the interface condition is *implied* by
    # `T_c`, so the block adds a competing loss term and no information, and the
    # three mechanical benefits do not pay for it. Kept as a knob with the
    # measurement, like the other remedies in `docs/axial_nn.md`.
    front_net: bool = False
    front_frac: float = 0.25  # share of collocation drawn near the predicted front

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
