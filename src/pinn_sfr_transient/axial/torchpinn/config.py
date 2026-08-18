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
    # This does not use the reference in the loss. The
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

    # Variable scaling: divide each residual block by its own characteristic rate
    # so all blocks are O(1) [Ko & Park, JCP 529 113860 (2025)]. The natural rates
    # span 813x here (`physics.residual_scales`), almost all of it the void, which
    # is 8x beyond what `weight_max_ratio` can undo — so the fixed part has to
    # come out analytically. D39 measured this as a no-op, correctly,
    # back when the adaptive weights were unbounded and cancelled it.
    residual_scaling: bool = True

    # Eliminate the void algebraically instead of solving it (deviation D-TH-3).
    # `alpha` fills a node in 0.71 ms against a 0.113 s transport time, so it is a
    # fast variable slaved to `T_c` -- the same elimination D-KIN-1 makes for the
    # prompt neutron mode. Removes a residual block whose normalised rate is 8.5e4
    # and lets the front appear analytically where `T_c` crosses saturation.
    void_closure: bool = True

    # The first-order stage's algorithm. "adam" is every published number here.
    # "ademamix" adds a slow gradient EMA (arXiv:2409.03137), measured ~2x better than
    # Adam beyond 8000 iterations and ~260x WORSE at 3000, because the slow
    # EMA has not warmed up -- so it is gated on budget, never swapped in blind.
    #
    # JAX only, and `training.py` RAISES on the other values rather than substituting
    # Adam. It did substitute, silently, so a torch arm labelled `ademamix` in a study
    # was Adam and nothing in the run said otherwise. This comment used to declare the
    # non-implementation and protect nothing.
    first_order: str = "adam"

    # Warmup as a FRACTION of the first-order budget. A fraction rather than a step
    # count so it scales with the budget instead of silently becoming the whole run at
    # a short one.
    #
    # It names two things: the "schedulefree" arm's internal warmup, and the length of
    # `lr_warmup`'s ramp. Both backends use it for both.
    sf_warmup_frac: float = 0.1

    # Linear warmup in front of the cosine decay, over `sf_warmup_frac` of the
    # first-order budget. `optax.warmup_cosine_decay_schedule` on the JAX side;
    # `LinearLR` into `CosineAnnealingLR` through `SequentialLR` here, which is torch's
    # shipped equivalent. The two agree to 9.4e-11 after the ramp and differ by at most
    # `lr / warmup` during it, because optax starts at exactly zero and torch's
    # `start_factor` must be positive.
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

    # Run the loss under `torch.compile` -- the torch analogue of the JAX twin's
    # `eqx.filter_jit`, which is always on there. **Both stages**, and the quasi-Newton
    # one matters more: its closure is evaluated several times per iteration through the
    # line search, and it used to call `causal_loss` directly, so this knob did nothing
    # for it and nothing at all for an `adam_iters = 0` arm. Steady state at f256 with 500
    # collocation points, 8 threads pinned with `taskset`, machine otherwise idle, from
    # `tools/backend_smoke.py --compile` (best of three warm budget-pairs, so neither
    # compilation nor a single hiccup is in the figure):
    #
    #     quasi-Newton   104.58 -> 16.30 ms/iteration     6.4x
    #
    # and for the first-order loop, over four runs:
    #
    #     eager      90.2 - 99.7 ms/iteration     (10.0 - 11.1 it/s)
    #     compiled    6.6 -  8.9 ms/iteration    (112.9 - 152.7 it/s)
    #     speedup    10.7x - 15.1x
    #
    # A RANGE and not a point. The eager arm is steady to 1.12x within a run; the
    # compiled arm is the variable one, up to 1.81x, because at 8 ms an iteration any
    # interference is a large fraction of it. The third digit of a speedup here is not
    # a real quantity -- "over 10x" is the claim that survives a re-run.
    #
    # The gain is fusion of elementwise work, not matmul: a step issues ~800 `aten::mul`
    # and ~230 `aten::add` against 96 `aten::mm`, and the forward-mode passes in
    # `state_and_grads` decompose into hundreds more `prims::` ops that eager dispatches
    # one at a time. `residual_blocks` alone goes 70.4 ms -> 6.7 ms.
    #
    # It does not move a number. `uv run python tools/backend_smoke.py --compile` trains
    # 200 Adam iterations both ways from one seed and reports
    # `||dparams|| / ||params|| = 3.6e-16` -- a few ulp -- so this is the same kind of
    # change as `foreach=True`, not a thread-count change.
    #
    # **Off by default anyway, because compilation is not free**: 12-40 s per input
    # shape, and RAR grows the collocation set by `rar_add` until `rar_cap`, so a
    # default run pays for up to 20 shapes. At ~85 ms saved per iteration that is worth
    # having above roughly 10 000 first-order iterations and a loss below it -- and the
    # default budget is 30. Long runs set it; the test suite must not.
    compile: bool = False

    device: str = "cpu"
    seed: int = 0
    log_every: int = 1000
