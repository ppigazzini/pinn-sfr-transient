"""Training hyper-parameters for the torch axial PINN.

Separated from the networks, the model and the trainer so a knob can be read
without importing torch, and so the dependency graph stays a DAG.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class AxialTrainConfig:
    """Hyper-parameters for the axial PINN and its schedule."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
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
    # 30 Adam / 30000 quasi-Newton (7.5.20). Chosen because at that budget the model
    # MEETS ITS 1% BAR -- T_s 0.0017 at three seeds, 5.9x inside it -- after being
    # stuck at 2x outside it for the project's whole life. The whole of that came
    # from the quasi-Newton count: 3000 -> 30000 improves T_s 15x monotonically on
    # every seed, L_void reaches 99.3% of the reference, and the saturation margin
    # +67.6 K against the reference's +69.2 K.
    #
    # It costs 10x the wall-clock, 9060 s against 905 s, and that breaks the rule the
    # previous default was chosen by -- form the front, beat what it replaces, cost no
    # more. The rule is overridden deliberately: a default that MEETS the acceptance
    # criterion in 2.5 hours is worth more than one that misses it in fifteen minutes,
    # and 7.5.4b's cheaper configurations remain one field away for anyone iterating.
    #
    # Adam stays at 30 rather than 0 because 30 is measurably better (0.0044 against
    # 0.0084 at qn10000) and costs nothing, and rather than 300 because 7.5.11 showed
    # that axis flat -- the shipped 300 was never doing measurable work.
    #
    # Measured on JAX at three seeds. Torch is unmeasured at this budget and running;
    # both backends have the same monotone quasi-Newton axis in 7.5.11, which is why
    # this ships before that confirms rather than after.
    adam_iters: int = 30
    lbfgs_iters: int = 30000
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

    # The first-order stage's algorithm. "adam" is every published number here.
    # "ademamix" adds a slow gradient EMA (arXiv:2409.03137); REPORT-01 records it ~2x
    # better than Adam beyond 8000 iterations and ~260x WORSE at 3000, because the slow
    # EMA has not warmed up -- so it is gated on budget, never swapped in blind.
    # JAX only: torch has no AdEMAMix and one is deliberately not written.
    first_order: str = "adam"

    # Warmup as a FRACTION of the first-order budget, used by the JAX backend's
    # "schedulefree" arm. Present here so the two configs stay field-for-field equal --
    # the parity check in `tools/backend_smoke.py` compares fields and defaults, and a
    # knob that exists on one side only is exactly the silent fork AGENTS.md forbids.
    # Torch has no schedule-free optimiser and this value is unused there.
    sf_warmup_frac: float = 0.1

    # Linear warmup in front of the cosine decay, over `sf_warmup_frac` of the
    # first-order budget. `optax.warmup_cosine_decay_schedule` on the JAX side; the
    # torch twin owns the field for config parity and does not implement it, like
    # `first_order="ademamix"`.
    #
    # **Off by default**, because warmup changes the trajectory of every first-order
    # arm and plain Adam is where every published first-order number here comes from.
    # Turn it on for an arm that needs it -- AdEMAMix is warmed on `alpha` and `b3`
    # regardless, which is a different warmup and not optional.
    lr_warmup: bool = False

    # Emit a checkpoint every N first-order iterations, so ONE run yields a whole
    # budget ladder instead of one run per rung. `polish_checkpoints` does this for the
    # quasi-Newton stage; with `lbfgs_iters = 0` that stage never runs and a pure
    # first-order arm produced no checkpoints at all -- 10 rungs meant 10 runs.
    # 0 disables it, which is every published number here.
    adam_checkpoint_every: int = 0

    # Collocation counts per stage, and how often the polish redraws.
    #
    # `n_colloc` alone drove BOTH stages, which meant Adam ran full batch -- every step
    # evaluating the same count the quasi-Newton stage uses. That is not how Adam is run
    # anywhere: the literature uses small batches and many steps (JAX-PI: 4096 points,
    # 200000 steps), and a full-batch first-order method is a different algorithm.
    #
    # `polish_refresh` redraws the quasi-Newton set every N iterations, restarting the
    # optimiser so its curvature history stays consistent WITHIN a block. arXiv:2605.24278
    # runs "20000 BFGS iterations in blocks of 1000" for exactly this reason: a fixed set
    # is what makes curvature meaningful and also what the polish can overfit. 0 keeps the
    # single fixed set, which is every published number here.
    #
    # None means "use n_colloc", so the defaults are unchanged.
    adam_colloc: int | None = None
    polish_colloc: int | None = None
    polish_refresh: int = 0

    # Hold the Fourier->trunk projection FIXED during the quasi-Newton stage.
    # That layer is an encoder -- it selects which embedded frequencies are used, a
    # representation choice Adam's fresh-sample stream suits.
    #
    # The DETERMINACY argument this comment used to make was wrong and is withdrawn:
    # freezing takes the trainable count from 17029 to 16965, a 0.4% change, because the
    # projection was never fitting capacity in the first place (sec 7.5.37a). What
    # freezing actually changes is the CURVATURE DIMENSION -- the space L-BFGS builds its
    # pairs in drops from 49797 to 16965 at f256, and from 25221 to 16965 at f64. That is
    # a conditioning argument, and it predicts the gain should scale with the embedding
    # width, which is testable and is what `freeze_after` is for. Off by default.
    freeze_encoder: bool = False

    # Freeze that projection PART WAY THROUGH the quasi-Newton stage rather than for all
    # of it. `freeze_after = k` runs k iterations with everything trainable and the
    # remaining `lbfgs_iters - k` with the encoder held, on the SAME collocation set.
    #
    # sec 7.5.32 measured the all-or-nothing form and found it worse at either Adam
    # budget: the projection evidently still has work to do. This asks whether it has a
    # FINITE amount -- if the representation settles early and only the trunk keeps
    # improving, the late iterations are paying for 33% (f64) or 66% (f256) of a
    # curvature space that has stopped moving.
    #
    # Zero disables it. Incompatible with `polish_refresh`, which restarts the optimiser
    # on its own schedule; setting both raises rather than silently picking one.
    freeze_after: int = 0

    # Cumulative quasi-Newton iteration counts at which the model is handed to a
    # caller-supplied callback, so ONE run can be scored at several budgets instead of
    # being re-run once per budget. `(30000, 40000, 50000)` with `lbfgs_iters = 50000`
    # gives three scored states from one 50000-iteration solve.
    #
    # A checkpoint does not perturb the run. The optimiser state is carried across the
    # segment boundary rather than rebuilt, so the trajectory is the one an uninterrupted
    # solve would take -- if it were restarted instead, every intermediate row would be
    # measuring the checkpointing rather than the budget, and sec 7.5.37 measured that
    # restart at 1.5x worse. Empty disables it.
    polish_checkpoints: tuple[int, ...] = ()

    # Trainable multi-resolution Fourier feature pyramid, arXiv:2605.24278 ("beignet").
    # OFF by default (`levels = 0`), so no published number moves when it lands.
    #
    # The paper's claim is that this ARCHITECTURE, not a change of optimiser, lets Adam
    # reach accuracy previously needing higher-order methods. It is therefore the only
    # honest way to test "Adam replaces the quasi-Newton stage" here: section 7.5.11
    # measured our unmodified Adam flat over two decades, so running it longer tests a
    # strawman. Note the paper's own Table 2 has MLP+BFGS at 7.11e-20 against
    # beignet+Adam's 6.63e-19, so the claim is that Adam REACHES the regime, not that it
    # wins it.
    #
    # `beignet_pad` is a registered deviation: Fourier interpolation is periodic and this
    # channel is not, so `zeta` is mapped into the interior of one period.
    beignet_levels: int = 0
    beignet_features: int = 14
    beignet_base: int = 2
    beignet_noise: float = 0.1
    beignet_pad: float = 0.25

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

    # Onset head: two trainable scalars `(zeta*, t*)` and the two tangency
    # residuals that pin them (`onset_head`). Off by default.
    #
    # Onset is the first instant the field TOUCHES saturation, so at that instant
    # the peak is the contact point and two conditions hold together:
    #     T_c(zeta*, t*) = T_sat + dT_sup      and      d T_c/d zeta (zeta*, t*) = 0
    # Reading the height off a threshold crossing instead asks a flat function
    # where it crosses a value: near the peak the error law is
    # `sqrt(2 eps / kappa)` -- a SQUARE ROOT. Stationarity is `delta(slope)/kappa`,
    # linear and divided by a curvature of ~1066 K per unit zeta squared.
    #
    # It also puts onset in the OBJECTIVE. Every onset number in this project so
    # far was read off a trained field afterwards; nothing ever optimised for it,
    # which is the first reason M4 never moved.
    onset_head: bool = False

    # Curvature pairs kept by the quasi-Newton stage. Explicit and shared, because
    # it was neither: torch passed `history_size=50` while the JAX default path
    # called `optax.lbfgs()` bare, whose default `memory_size` is **10**. Measured
    # on an identical objective from identical weights, that one argument IS the
    # whole torch/JAX gap -- torch at 10 degrades to JAX's curve (1.71x at 300
    # iterations), and optax at 50 matches torch's to within 2%.
    #
    # 50 is not tuned; it is what torch was already using, kept so the fix moves
    # JAX onto the published torch behaviour rather than moving both somewhere new.
    lbfgs_history: int = 50

    # Laplace embedding (REPORT-01 section C.7, docs/axial_nn.md section 7.5.18).
    # Physical decay rates in 1/s; the embedding uses `exp(-s_k * t_end * t_hat)`.
    # `()` is off and is the control.
    #
    # A Fourier basis is oscillatory and this transient is built out of DECAY:
    # coast-down at 1/tau_pump and six precursor groups spanning 0.0124 to 3.01
    # per second. Approximating exp(-0.2 t) over the window out of sines costs many
    # terms and still misses the tail; one exponential does it exactly. The split
    # is not arbitrary either -- the oscillatory structure is in `zeta` and the
    # exponential structure is in `t`, which is the anisotropy section 7.5.12
    # measured on the bandwidth, reached from the physics instead of a sweep.
    laplace_rates: tuple[float, ...] = ()
    # How the two bases combine. "alone" drops Fourier entirely (the known-shape
    # case: a fit, not a basis). "sum" concatenates the blocks, which is right when
    # the solution is a superposition. "product" modulates each Fourier group by one
    # rate, giving damped sinusoids -- right when the two are coupled, which is what
    # a transient excursion is rather than a sum of one of each.
    laplace_mode: str = "sum"

    # Two-encoder "modified MLP" [Wang, Teng & Perdikaris 2021], the architecture
    # jaxpi uses by default; multiplicative interactions carry the inputs to every
    # layer instead of letting them wash out with depth.
    modified_mlp: bool = False

    # Pseudo-time stepping against spurious solutions
    # [Wang, Koohy, Lu & Perdikaris, arXiv:2604.23528]. 0 disables.
    pts_every: int = 0
    pts_dtau: float = 1.0
    pts_growth: float = 1.5

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

    device: str = "cpu"
    seed: int = 0
    log_every: int = 1000
