"""Training hyper-parameters for the JAX axial PINN.

Separated from the networks, the residuals and the samplers so a knob can be
read without importing a model, and so the dependency graph stays a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    # **Measured, not chosen.** The default must (a) form the boiling front on every
    # seed of both backends, (b) beat what it replaces on accuracy, and (c) not cost
    # more. `300/3000 + f256` is the only configuration measured that does all three:
    #
    #   configuration          T_s torch   sec    worst margin   front
    #   8000/500  f0 (old)     0.0434      2337   -2.3 K         on NO seed
    #   300/3000  f128         0.0314      1636   +17.5 K        every seed
    #   300/3000  f256 (this)  0.0282      2209   +24.4 K        every seed
    #   300/3000  f512         0.0216      3392   +34.6 K        every seed
    #
    # f512 is more accurate and is the documented best (`axial_nn.md` 0.6), but costs
    # 45% MORE than the default it would replace; a default should not impose that
    # silently. f128 is cheaper still, at a +9.5 K worst-seed margin on JAX -- thinner
    # than is comfortable for a threshold crossing (7.5.4).
    #
    # The old default was dominated on every axis: slower than 300/3000+f512, 40% less
    # accurate, L_void at a tenth of this, and it produced NO boiling front on any
    # seed of either backend (7.2.9) -- the repository failing to reproduce its own
    # headline result.
    adam_iters: int = 300
    lbfgs_iters: int = 3000
    lr: float = 1e-3

    # Dimensionless: the prefix sum is normalised by the total. See the torch twin.
    causal_eps: float = 0.0
    causal_chunks: int = 32
    # Fraction of `p.t_end` trained over — a scope decision; see the torch twin.
    # 16.5 s of the 60 s horizon: the reference stops exactly there, on every mesh
    # from n = 40 to n = 640 (`axial_nn.md` section 6.5), because that is where the
    # channel leaves the section 12.13 property range.
    #
    # **This default was 1.0, and 1.0 does not reproduce any published number.**
    # Every table in `axial_nn.md` sections 7.2.5 onward was measured at 0.275 and
    # described as "the current defaults"; the value was recorded nowhere. At 1.0
    # the shipped configuration trains over 72% of a horizon where the model does
    # not apply and **forms no boiling front at all** — `max alpha = 0.0000` — so
    # the repository's headline result was not what its own entry point produced.
    #
    # 0.25 scores better on all four fields (T_f 0.1250 against 0.1379) and is
    # **rejected**: 15 s is not where this model stops being valid, and choosing
    # the validity window by its score against the reference is fitting the
    # problem statement to the ruler. The horizon is a property of the model's
    # validity range, not of its solution.
    #
    # It is also a cliff, not a slope. 0.300 gives `max alpha = 0.0000` — no front
    # at all — so the published configuration sits 0.025 from where its headline
    # result vanishes. See `axial_nn.md` section 7.2.7.
    t_train_frac: float = 0.275
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
    # Share of collocation drawn on the **saturation level set** -- the points
    # where the network's own `T_c` is near `T_sat + dT_superheat`.
    #
    # **This is a fix for a measure bug, not another loss term.** The loss is a
    # mean over the domain and the front occupies a few percent of the channel, so
    # the front region contributes a few percent of the objective no matter how
    # long training runs. More optimisation therefore converges more accurately to
    # a minimiser whose peak is wrong: 8000+500 iterations beat 3000+300 by 47% on
    # `T_s` and lose the front entirely (`docs/axial_nn.md` section 7.5.5).
    #
    # RAR cannot supply these points. Once the void is closed algebraically the
    # field residual is small *everywhere*, including across the front, so
    # residual-magnitude sampling has no signal to follow -- the torch sampler has
    # carried a comment saying exactly that since M8.
    #
    # It needs no front-position network: under D-TH-3 the front IS the level set
    # `T_c = T_sat + dT_superheat`, and the network knows its own `T_c`.
    # `front_level_set` selects that; `front_net` selects the M8 front network,
    # which measured worse on every metric.
    front_level_set: bool = False
    # Feed the level-set coordinate `phi = (T_c - T_sat - dT_sup) / dT` as a third
    # network input (idea 3).
    #
    # Under D-TH-3 the front IS the level set `phi = 0`, and the solution is
    # **smooth in phi** where it is kinked in `zeta`. Giving the network that
    # coordinate removes the sharpness rather than resolving it, which is a
    # different move from the Fourier basis and, on paper, a stronger one.
    #
    # `T_c` is the network's own output, so the input depends on the output. This is
    # resolved with a **bootstrap pass**: evaluate once with `phi = 0`, take the
    # resulting `T_c`, and evaluate again with the real `phi`. **Gradients flow
    # through both passes**, so the total derivative the residual needs includes the
    # term through `phi`. Detaching `phi` would be cheaper and would silently make
    # `d/dz` wrong -- it would train, produce plausible numbers, and corrupt the
    # residual, which is the exact defect class this project keeps finding.
    #
    # Costs two forward passes and a deeper graph. Distinct from `front_net`, which
    # fed `zeta - z_f(t)` from a SEPARATE learned network and measured worse.
    level_set_input: bool = False

    # --- remedies for the moving front; every one measured in docs/axial_nn.md --
    # Time-window curriculum: train [0, w t_end] for growing w. Neutral in the
    # section 7.2.5 re-ablation.
    n_windows: int = 1
    # Random Fourier features against spectral bias [Tancik et al. 2020]. Measured
    # -11.1% at three seeds (section 7.2.6) but NOT adopted: it does not compose
    # with `modified_mlp`. 0 disables.
    # 256: the capacity rung of the shipped default. The ladder's measured
    # endpoint is 512 (`axial_nn.md` 7.5.8); 256 is the rung that keeps the
    # default no more expensive than the one it replaced.
    fourier_features: int = 256
    fourier_scale: float = 2.0
    # Bandwidth of the Fourier basis in `zeta`, as a multiple of `fourier_scale`.
    # `None` keeps the basis isotropic, which is what a single `scale` assumes.
    #
    # **The solution is not isotropic.** The front is a near-discontinuity in `zeta`
    # and smooth in `t`, so one bandwidth either under-resolves the front or spends
    # conditioning on time structure that is not there. Values > 1 sharpen the basis
    # in space only.
    fourier_scale_zeta: float | None = None
    # Multi-scale Fourier: several frozen `B` blocks at different bandwidths,
    # concatenated (idea 2). `()` is a single band and reproduces the default.
    #
    # A single bandwidth has to be chosen, and the solution has structure at more
    # than one: a smooth bulk and a near-discontinuous front. Bands of, say,
    # (1, 4, 16) cover both without picking one. The feature count is split evenly
    # across bands, so `fourier_features` still sets the total width and this trades
    # resolution *within* a band for coverage *across* bands.
    fourier_bands: tuple[float, ...] = ()

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
    # Quasi-Newton stage. `docs/axial_nn.md` section 7.3.4 measured that this
    # stage is not a polish here -- it is the step that forms the boiling front,
    # since Adam alone leaves `max alpha = 0` in both backends. So the choice of
    # implementation is a physics question, not a tuning one.
    #
    # "lbfgs"        -- the framework's own: `torch.optim.LBFGS` in torch,
    #                   `optax.lbfgs` in JAX. The default, unchanged, so no
    #                   published number moves. NOTE that this is the one knob
    #                   whose *implementation* differs between the backends.
    # "lbfgs-shared" -- this repository's own L-BFGS, bit-comparable across
    #                   backends (a test pins the two to 1e-10 on a quadratic).
    #                   Exists to answer section 7.3.4: it removes the last
    #                   implementation difference between the backends, so if the
    #                   21% `T_s`/`T_c` gap survives it, the framework L-BFGS is
    #                   not the cause.
    # "ssbfgs"       -- limited-memory self-scaled BFGS [Oren & Luenberger 1974;
    #                   Al-Baali 1998], the family Kiyani et al. (arXiv:2501.16371)
    #                   report beating L-BFGS across PINN benchmarks. Measured here
    #                   on stiff quadratics and Rosenbrock BEFORE the PINN: it loses
    #                   to plain L-BFGS on both. Kept because that comparison is the
    #                   point, and because a PINN loss is not a quadratic.
    optimizer: str = "lbfgs"

    seed: int = 0
    log_every: int = 1000
