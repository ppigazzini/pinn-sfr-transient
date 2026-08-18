# Axial PINN — formulation, recipe, and measured results

The neural-network side of the 1D axial boiling model. The physics, and every
deviation from the SAS4A/SASSYS-1 manual, are in
[`axial_physics.md`](axial_physics.md); this document covers the ansatz, the
training recipe, the two backends, and **what has actually been measured** —
including the results that came out badly, which are most of them.

---

## 0. Status quo

Every number is three seeds against an `n_axial = 160` reference on **both
backends**, and every table is reproducible by a sub-command of
[`tools/axial_study.py`](../tools/axial_study.py). Complete: `ruler`, `horizon`,
`budget`, `optimizer`, `parity`, `plan-a`, `combo`, `regime`, `regime-sign`,
`default`, `scaling`, `levelset`, and `margin` through f1024 (the ladder's measured
endpoint is f512, §7.5.8). Running: `grid`, which crosses Adam against the
quasi-Newton stage independently at fixed f128 — the question §7.5.3 and §7.5.5 each
answered under a constraint. Committed but unrun: `frontfrac`,
`capacity-optimiser`.

### 0.1 Where the accuracy stands

| configuration | `T_f` | `T_s` | `L_void` | worst-seed margin | front |
|---|---|---|---|---|---|
| **shipped default** — `300/3000`, f256 | 0.0941 | **0.0282** | **0.2834** | **+24.4 K** | **every seed** |
| **documented best** — `300/3000`, f512 | **0.0710** | **0.0216** | **0.3012** | **+34.6 K** | **every seed** |
| *previous default* (`8000/500`, f0) | 0.1243 | 0.0434 | 0.0367 | **−2.3 K** | **on no seed** |
| *published-table budget* (`3000/300`, f0) | 0.1386 | 0.0765 | 0.1529 | +12.5 K | every seed |
| reference | — | — | 0.3812 | — | — |
| **acceptance bar** | 0.01 | 0.01 | — | — | — |

torch, three seeds. §0.6 gives the JAX figures and the cost of each.

**The bar is missed by 2.8× at the default and 2.2× at the documented best**,
against 4.3× before this study. `L_void` is at **74% of the reference** by default
and 79% at best, against the previous default's 10%. The best single f512 run
reaches `T_s` **0.0148** — 1.5× off the bar — but the f512 seed range is 1.71×, so
the worst seed is the number to quote.

**The previous default formed no boiling front on any seed of either backend**
(§7.2.9) — the repository failing to produce its own headline result. Both causes,
the training horizon and the iteration budget, are fixed and pinned by tests.

The ruler is not the limit. At `n = 160` the reference's own error is 1.1–1.6e-3
(§6.5), so the 1% bar sits 6–9× above it and the failure is **20–45× the ruler**.

### 0.2 What is settled

| | |
|---|---|
| **Hard constraints** | exact, for any weights: IC `0.0`, inlet `T_c(0,t) = T_in` to `0.0`, `α ∈ [0,1)` by construction, `c(0) = 1`, `P(0) = 1.000000` under Plan A |
| **One set of equations** | the residual calls the same `continuous_derivatives` the reference discretises; bit-equality is tested |
| **Backend parity** | the two backends' residuals agree to **1e-14** at identical parameters. The 16.8% accuracy gap is the framework L-BFGS and nothing else: `jax/torch` is 1.168 with each framework's own and **0.999** with a shared implementation (§7.3.2) |
| **The ruler** | converged and quantified at every mesh (§6.5). Onset time is mesh-independent; onset location is converged to one cell |
| **Objective 2** | answered — the positive void coefficient drives an excursion to 5.3× nominal, governed by `zeta_sign`, not by the void worth ([`axial_physics.md`](axial_physics.md) §10) |

### 0.3 What is understood but not fixed

**The front is one inequality.** Under D-TH-3, `α` is a function of `T_c` alone, so
"the front forms" means `max T_c > 1169.0 K`. The shipped default clears it by
**20.5 K out of a ~590 K range**. `max α` is a saturating function of that margin
and carries no independent information (§7.2.8).

**The mean and the extremum are near-independent**, which is why a change can
improve every temperature score and switch the front off. That cost this project
nine remedies scored on the mean while M4 asks for the peak.

**Plan A's failure is systematic, not variance.** `L2(P)` = 0.1110 at three seeds
(7.7% spread) and `min ρ/β` = −0.1503 against the reference's −0.2052 — the network
finds **26–28% less negative feedback on every seed**, with a 2.6% spread on that
quantity (§7.4.1).

**Two constraints bind, and they are independent.**

*The measure.* The loss is a mean over the domain and the front occupies a few
percent of it, so the front contributes a few percent of the objective however long
training runs. More optimisation therefore converges more precisely to a minimiser
whose peak is wrong — measured directly in §7.5.5, where 3k → 8k iterations improve
`T_s` by **47%** and destroy the front on both backends, and 16k does not recover
it. This is why nine remedies scored on the mean did nothing for the front.

*Capacity.* Fourier features 32 → 512 improve `T_s` by **39%** on torch and 20% on
JAX, `L_void` from 0.2270 to 0.3012, and the worst-seed margin from +7.6 K to
+34.6 K — monotonically, on both backends (§7.5.8). The model was
under-parameterised, and every one of the nine refuted remedies was ablated on a
network an order of magnitude too small. That does not make those ablations wrong;
it makes them ablations of a capacity-limited model, which is D59's rule at a scale
nobody checked.

The two are not alternatives. The measure decides *where the loss puts its weight*;
capacity decides *how sharp a peak is representable at all*.

### 0.4 What is open

| | |
|---|---|
| **M4 acceptance** | onset within 0.5 s and one cell — **time met at three seeds** (worst 0.0181 s, §7.5.16a); the height half is a tautology on a monotone `T_c` and cannot be failed, so M4 no longer discriminates |
| **The 1% bar** | **not met**, by 3.5× at best |
| **`L_void`** | best 0.2270 against the reference's 0.3812 — 40% short |
| **M6 acceptance** | **failed**, 11.1% on `L2(P)` against a 1% bar |
| **D-TH-2** | `z`-dependent flow after voiding: implemented, and Radau cannot step it |
| **D-FB-3** | five feedback mechanisms omitted; the model is **non-conservative** in that direction |

### 0.5 Why the best known configuration is not the default

`adam_iters=300, lbfgs_iters=3000, fourier_features=512` beats the default on every
metric, on both backends, at three seeds — and forms the boiling front on every seed
where the **default forms it on none** (§7.2.9, six runs, both backends). It is
nonetheless **not** shipped as the default, for three reasons:

* **Every published table in this document was measured on the default.** Moving it
  invalidates all of them at once, and this project has just spent a revision
  recovering from a default that silently disagreed with its own tables (§7.2.7).
* **It is more wall-clock** — 52% for the budget change alone, and `f128` adds
  further cost on top of that.
* **`AGENTS.md` requires new behaviour to land off by default** so no published
  number moves when it does.

The correct sequence is to re-measure the document against it and then move both
together. That is a compute task, not a development one, and
`tools/axial_study.py default` is the arm that does it — it runs with **no
overrides at all**, which is the only configuration that can catch a
default/documentation mismatch. Two such mismatches have been found this way
(§7.2.7, §7.2.9), and both were invisible to every other arm because every other
arm passes its knobs explicitly.

**The case for moving it is now stronger than "better numbers".** The shipped
default does not produce the repository's headline result: it forms no boiling
front, on either backend, on every seed measured. A default that cannot reproduce
the documentation is a defect regardless of what the alternative scores.

### 0.6 Which configuration to use

**Use JAX for the long runs**, on accuracy grounds and on inertia: it is within
**1.08×** of PyTorch at f512 (§7.5.10), and every measurement on the shelf was taken
with it. For most of this project's life it looked slower *and* weaker on one axis,
and both readings were artefacts (§7.5.17, §7.5.19).

**The speed argument for that choice no longer holds.** The 4.4× in §7.5.19 was
measured against an eager PyTorch loop. With `compile=True` the ordering reverses —
at 500 collocation points and f256 on 8 pinned cores, best of three repeats per run,
four runs:

| | ms/iteration | spread within a run |
|---|---|---|
| torch, compiled | 7.85 – 8.94 | 1.05× – 1.21× |
| JAX, jitted | 15.36 – 16.61 | 1.03× – 1.14× |

The ranges do not overlap and the gap, **1.78× to 1.96× in PyTorch's favour**, is well
outside the within-backend spread. `uv run python tools/backend_smoke.py --timing`
reproduces it. This is one width, one thread count, one machine, and it is a timing
and not an accuracy result, so it does not on its own move the default — but the
sentence "JAX is faster" is now false as written.

PyTorch remains a first-class arm: two independent implementations agreeing is the
strongest check here, and it is what caught the defect.

All figures below are three seeds. Wall-clocks are **contended** — several jobs at
`OMP_NUM_THREADS=8` — so treat them as ratios, not benchmarks; accuracy is
unaffected (§7.2.8).

| purpose | configuration | `T_s` | `L_void` (% of ref) | worst margin |
|---|---|---|---|---|
| **best front fidelity** | `300/3000`, f256, `fourier_bands=(1,4,16)`, **jax** | 0.0233 | **0.3793 (99.5%)** | +28.1 K |
| **best mean accuracy** | `300/3000`, f256, `fourier_bands=(0.25,1,4,16)`, **jax** | **0.0203** | 0.2932 (77%) | +24.4 K |
| **best without new knobs** | `300/3000`, f512, **torch** | 0.0216 | 0.3012 (79%) | +34.6 K |
| shipped default | `300/3000`, f256, **jax** | 0.0292 | 0.2440 (64%) | +25.8 K |

`fourier_bands` is **off by default** and stays off until it has been measured
under the same discipline as everything else it would displace — but it is the
first configuration to reach the reference's voided length, and §7.5.14 explains
why: the high band resolves the front, the low bands keep the bulk smooth, and at
a fixed feature total the split between them is a choice rather than an accident.

**Onset is a question about *time*, not height.** `T_c` is monotone in `ζ`, so its
maximum — and therefore onset — is always at the outlet, and any height metric is
reporting the mesh rather than the network (§7.5.16, retracted there).
`onset_by_tangency` is still the better *time* readout, because the threshold time
is quantised by the 0.25 s scoring grid; it says 0.62–0.84 s against a 0.5 s
criterion. `onset_head` is measured harmful and stays off.

### 0.7 Method notes that changed the answers

* **Three seeds, or say the sample size in the sentence.** Every two-seed claim made
  during this study was wrong at three: the budget sweep's "monotonic" front, the
  backend gap's disappearance, and a 0.1% variance that was two coincident samples.
* **Do not pair seed indices across backends.** `seed=1` seeds two different RNGs;
  index-paired the backend ratios read 1.167/0.997/1.366, as distributions they read
  1.158/1.166/1.177.
* **A retraction needs the same evidence as the claim.** §7.3.2 was withdrawn on one
  seed and re-confirmed at three.
* **A control arm that reproduces a published configuration** is what caught D67 —
  the default that produced no front.

---

## 1. What the network solves

A map `(ζ, t) → (T_f, T_cl, T_s, T_c, α)`, trained on the Chapter 3 residuals
alone. By default `α` is *computed* from `T_c` rather than learned (D-TH-3), so
the network's free outputs are the four temperatures and the residual carries
four field blocks. No reference data enters the loss; the reference is used only at test time,
the protocol the 0D model already follows ([`neural_network.md`](neural_network.md) §7).

**One set of equations.** The residual calls
`axial.physics.continuous_derivatives` — the same function, and therefore the same
flux and boiling expressions, that the M2 reference solver discretises. A test
rebuilds the residual by hand from it and asserts bit-equality, so the network and
its ground truth cannot drift apart. This is the axial counterpart of the 0D
model's `tests/test_consistency.py`.

Two modes:

| | power | collocation |
|---|---|---|
| **Plan B** (M3) | prescribed, `P(t)/P_0 = 1` | scattered `(ζ, t̂)` |
| **Plan A** (M6) | an **output** of the prompt-jump closure | time only, against the fixed axial quadrature |

Plan A has to use a tensor grid because reactivity is an axial *integral*
(Eq. 4.5-3, Eq. 4.5-25): one power amplitude couples every node at a given time to
every other, and an integral cannot be evaluated on a random cloud of points.
Field residuals are then reduced over that quadrature so all six blocks are one
value per time — which is also the shape causal weighting wants. RAR is disabled
under feedback, since it adds arbitrary points and a quadrature rule cannot absorb
them.

## 2. The ansatz — every constraint hard, none in the loss

> **`α` is no longer a network output.** Under the default `void_closure = True`
> it is a function of the network's own `T_c` (D-TH-3), so the residual has
> **four** field blocks, not five, and the void-free initial and inlet conditions
> fall out of the closure rather than being imposed by a gate. The gated-sigmoid
> form below is what `void_closure = False` still selects, and what every number
> in §5 and §7.1–§7.2.2 was measured on.

```math
\theta_k(\zeta, \hat t) = \theta_{k,0}(\zeta)\,\exp\!\big(\hat t\, N_k(\zeta, \hat t)\big),
\qquad
\alpha = 1 - \big(1 - b(T_c)\big)^3,
\qquad
c_i = \exp\!\big(\hat t\, N_{c,i}(\hat t)\big)
```

| Constraint | Mechanism | Measured |
|---|---|---|
| Initial condition | `exp(0) = 1` | exact, `0.0` |
| **Positivity `T ≥ T_in`** | `θ₀ ≥ 0` times a positive exponential | exact under ×50 adversarial weights |
| Coolant inlet `T_c(0,t) = T_in` | `θ_c0(0) = 0`, so it falls out of the same form | exact, `0.0`, **no separate gate** |
| Void `α ∈ [0,1)`, void-free start, none at inlet | **the algebraic closure itself** (D-TH-3) | exact by construction |
| Precursors `c(0) = 1`, `c > 0` | bounded exponential | exact |

Nothing is penalised, so the objective is pure physics and there is one fewer
competing term for the weighting to balance.

**The multiplicative form replaced an additive one, and that was a formulation fix,
not a refactor.** With `θ = θ₀ + t̂·N` nothing bounded the temperatures below.
Logging `min(T_f)` through training shows the optimiser walking

```
722 K  →  441 K  →  170 K  →  −1 K      over 115 iterations, while the loss FELL
```

after which `log(T_f/T_f0)` in the logarithmic Doppler (Eq. 4.5-3) returns NaN and
Plan A dies. The residual was perfectly content in that region: the
spurious-solution failure mode of
[arXiv:2604.23528](https://arxiv.org/abs/2604.23528), and exactly what REPORT-01
§5.2 item 8 says to parameterise away — advice this model had applied to the void
and precursors but not to the temperatures. Constraining the ansatz to the
physical manifold removes the region rather than penalising it.

The exponent is bounded, `exp(S·tanh(x/S))` with `S = 4`, so it cannot overflow;
the reference needs at most a 9.6× growth in excess temperature against the `e⁴ ≈ 55×`
allowed, so the ceiling never binds and `tanh` stays in its linear region.

**The void half of the ansatz carries the mirror-image asymmetry — and the
obvious cure was measured and rejected.** The temperatures start *exactly* on the
steady profile, because `exp(0) = 1`. The void starts at `sigmoid(raw) ≈ 0.5` at
every interior point, because the output layer is initialised `U(±1/√width)` —
while the reference `α` is identically zero over ~96% of the channel and over all
of `t < 10.8 s`. Iteration zero therefore begins roughly 50× too voided
everywhere, and `α` is not an inert output: it degrades `film_coefficient`
(halving `h_ec` and `h_sc`), shifts `α_D` between its flooded and voided values,
and enters Eq. 4.5-25.

The asymmetry is real. The one-constant cure, `sigmoid(raw − 4)` so that
`α ≈ 0.018` at initialisation, is not: see §7.1 for the three-seed table. It is
**reverted**, and this paragraph is what survives, because the measurement is
worth more than the change would have been. This is the sixth remedy in this
document to be proposed on a sound argument and then fail its ablation, and the
first one proposed by an audit rather than by the literature — which does not
make it different.

**Admissibility, checked before adopting**: `θ₀ ≥ 0` for all four fields, zero only
at `ζ = 0` for the coolant and structure — exactly where the boundary condition
pins them. The structure being pinned to `T_in` at the inlet is correct, not a
side effect: its only coupling is to the coolant, held at `T_in` there.

## 3. Training recipe

Adam with cosine decay → L-BFGS polish, plus causal temporal weighting
[Wang, Sankaran & Perdikaris 2024], gradient-norm adaptive block weights
[Wang, Teng & Perdikaris 2021], and residual-adaptive refinement
[Wu et al. 2023]. float64 throughout. Init is `U(±1/√fan_in)` on weights *and*
biases in both backends — [`neural_network.md`](neural_network.md) §9 records what
happens when the two disagree.

### 3.1 Knobs and their measured defaults

| knob | default | why |
|---|---|---|
| `residual_scaling` | `True` | per-block variable scaling; §7.2.3 |
| `void_closure` | `True` | the algebraic void, D-TH-3; §7.2.3–§7.2.4 |
| `weight_max_ratio` | `1.0` (off) | adaptive block weights measured harmful; §7.2.1 |
| `causal_eps` | `0.0` (off) | causal weighting measured harmful; §7.2.4 |
| `t_train_frac` | `0.275` | the model's validity horizon, 16.5 s of 60 s. **Was `1.0`, which forms no front at all**; §7.2.4, §7.2.7 |
| `front_net` | `False` | front-position network, measured worse on every metric |
| `n_windows` | `1` (off) | neutral in the re-ablation; §7.2.5 |
| `fourier_features` | `0` (off) | **now measured better** (−11.1% at 3 seeds); not adopted — see below |
| `modified_mlp` | `False` (off) | **now measured better** (−16.1% at 3 seeds); not adopted — see below |
| `pts_every` | `0` (off) | pseudo-time stepping measured harmful; §7.2.5 |
| `optimizer` | `"lbfgs"` | the quasi-Newton stage. `"ssbfgs"` selects limited-memory self-scaled BFGS; §7.5 |

**Two defaults are open questions rather than settled ones.** §7.2.5 shows
Fourier features and the modified MLP both improve every temperature under the
current formulation, reversing D38. Neither is adopted, because the modified MLP
also gives the worst voided length of the working arms — it trades the front for
the fields — and because their **combination is worse than either alone** (§7.2.6).

## 4. Two backends, and why

> **Both are packages, not modules.** Each is split after
> [jaxpi2](https://github.com/sifanexisted/jaxpi2) into config / architectures /
> ansatz / residuals / weighting / sampling / training / evaluation, with the
> dependency graph a DAG in that order, and `axial.pinn_jax` / `axial.pinn_torch`
> re-exporting the public surface. What it buys is that an ablation — a different
> architecture, weighting or sampler — is a config change rather than an edit to
> one long file, and that `evaluate` never being imported by `training` makes
> "the reference never enters the loss" structural instead of a convention.
>
> Two modules do not mirror each other, both because of torch's idiom rather than
> a design choice: `nn.Module` owns its parameters and its forward pass, so the
> ansatz and the residuals share `torchpinn.model`; and the sampler needs the
> model to place points on the front while the loop needs mutable optimiser
> state, so both share `Trainer` in `torchpinn.training`.

> **Parity is a tested property, not an intention.** Both backends expose the
> same knobs with the same defaults, asserted field by field. The only
> asymmetries are framework-imposed and deliberate: JAX sets its device globally
> where torch takes `device`, and JAX's RAR keeps a **fixed-size** set so `jit`
> never recompiles where torch grows a reservoir (`rar_keep` against
> `rar_add`/`rar_cap`).
>
> This has failed twice and both failures were silent. The algebraic void closure
> landed in torch first, which made a published parity table a comparison of two
> different models. And `front_frac` was declared in the JAX config and read by
> nothing, so setting it did nothing and reported nothing — the same class of
> defect as the causal weighting that variable scaling switched off. There are now
> three tests: the knob *sets* must match up to the documented exceptions, the
> shared defaults must be equal, and each ported knob must actually train and
> preserve every hard constraint.

`pinn_torch.py` is object-oriented and eager; `pinn_jax.py` is functional
(Equinox + Optax). They share the residual functions and differ only in training
mechanics. That is the point: a disagreement between them is informative.

It has already paid twice. It showed the pre-fix failure was **not**
backend-specific, so the formulation was implicated rather than PyTorch. Then,
after the positivity fix, torch improved 2.5–8.7× and JAX did not move at all —
which exposed a real bug in the JAX port: it trained on a frozen collocation set
between RAR refreshes, where the 0D twin (and torch) resample every step. Fixing
that did **not** close the accuracy gap, so the gap remains open.

**Record a number before a refactor, not after.** Splitting the JAX backend into
its package dropped the `jax_enable_x64` call, and the whole backend trained in
float32 — with all 241 tests green, because no test asserted a dtype. It was caught
by a four-digit accuracy baseline taken before the split and re-checked after. The
torch split was then done in that order deliberately: confirm the backend is
bit-reproducible run to run at a pinned `OMP_NUM_THREADS`, lock the baseline
(`seed=0`, 500 iterations, `T_f` 0.2343, `T_cl` 0.3104, `T_s` 0.2106, `T_c` 0.2111),
refactor, re-check. It reproduced exactly. A passing suite is not a regression test
for a refactor that can silently change precision.

## 5. Measured results

> **Provenance (Annex A, N8).** Every number in §5 and §7.1–§7.2.2 was measured on
> the **pre-D-TH-3 formulation**: the void solved as a differential unknown, block
> weights unbounded, no per-block residual scaling, and the full 60 s horizon.
> All four have since changed. Treat §5 and §7.1–§7.2.2 as a record of how the
> model got here, not as current accuracy. The current numbers are §7.2.3
> (variable scaling), §7.2.4 (the phantom void), §7.2.5 (the re-ablation, which
> **retracts half of D38**) and §7.4 (Plan A). Where a §5 conclusion has been
> overturned the later section says so explicitly.


All at **3000 Adam + 300 L-BFGS**, seed 0, relative `L2` against the reference,
unless stated. The acceptance bar is `1e-2`.

### 5.1 The positivity fix

| | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| additive ansatz | 0.156 | 0.374 | 0.278 | 0.177 |
| **multiplicative** | **0.062** | **0.120** | **0.032** | **0.034** |

2.5–8.7× better, and Plan A stopped producing NaN. This is the only change that
has ever produced a directional improvement, and it came from tracing an actual
failure rather than applying a recipe.

### 5.2 Seed robustness (additive ansatz, five seeds)

`T_f` 0.14–0.21, `T_cl` 0.19–0.37, `T_s` 0.07–0.28, `T_c` 0.03–0.23. No lucky
seed; the failure was systematic, not variance.

### 5.3 Budget — **retracted; see §7.5.3 and §7.5.5**

> **This section's conclusion was wrong, and it closed the question for the rest of
> the project.** Two points, one seed, on the pre-D-TH-3 formulation. Every study
> for the following nine sections then ran at a single fixed budget without asking
> again — including every remedy ablation, every backend comparison and every
> parity table. The budget was the one axis nobody varied, on the strength of the
> sentence below.

| Adam iters | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| 3000 | **0.062** | 0.120 | **0.032** | 0.034 |
| 8000 | 0.246 | 0.095 | 0.075 | 0.028 |

The original claim: *"`T_f` gets 4× worse with more training. Non-monotonic in
budget means the optimiser wanders between minima rather than converging slowly, so
more iterations will not fix this."*

**Measured against the current formulation, the opposite holds.** §7.5.3 moves
budget into the quasi-Newton stage and every temperature improves — `T_s` by 41%,
`T_c` by 39% — with **non-overlapping seed ranges** across three seeds. §7.2.9 then
found the shipped 8000/500 default has a *better* mean than every published table.
The model is optimisation-limited from three independent directions (§0.3), which
is precisely what "more iterations will not fix this" denied.

This is the fourth conclusion in this document overturned by D59's rule — *an
ablation is a statement about the formulation it was run on* — and the most
expensive, because unlike D38 and D39 it was never re-tested. A negative result
that closes an axis deserves more evidence than one that opens one, and this had
two points and one seed.

### 5.4 Remedy ablation — **every one made it worse**

Worst relative `L2` across the four temperature fields:

| configuration | worst |
|---|---|
| **baseline** | **0.120** |
| time windows = 5 | 0.188 |
| time windows = 10 | 0.201 |
| Fourier features = 32 | 0.255 |
| modified MLP | 0.397 |

The modified MLP — jaxpi's default architecture — is **3.3× worse** here than a
plain MLP. Five techniques that are standard for this class of problem all degrade
it. That is itself the result: when the accepted remedies for "the PINN will not
converge" consistently hurt, the problem is not the training recipe, and stacking
published methods on top is noise around whatever is actually wrong.

### 5.5 Per-equation residual scaling — a provable no-op

The blocks' time constants are 0.58 s (fuel), 0.025 s (cladding), 0.107 s
(structure), 0.113 s (coolant transit) against a 60 s horizon, so scaling all four
by `t_end` leaves `t_end/τ` spanning 104 to 2378. Rescaling each block by its own
time constant — the standard advice for stiff multiscale PINNs — reproduced the
previous result **to every digit**. Two structural reasons: Adam is scale-invariant,
and the gradient-norm weighting sets `λ_k = mean(g)/g_k`, cancelling any fixed
per-block constant exactly. `physics.residual_scales` keeps the numbers as a
diagnostic and is deliberately not wired into the residual.

## 6. Bug hunt — what is actually wrong

### 6.1 The ansatz can represent the solution

Fitted to the reference by supervised regression (no physics): **rel `L2` ≈ 2.3e-3
on all four fields.** So architecture, width, depth and hard constraints are all
adequate, and the entire "try a better architecture" family of remedies is ruled
out — consistent with §5.4.

### 6.2 A measurement artefact, corrected

Evaluating the physics residual at that fit gave 1e8–1e9 for `T_f` and `T_cl`,
which looked like proof that the true solution is not a minimum of the loss. It was
not: those were **interpolation errors of the fit**, measured at random points while
the fit was only enforced on the reference grid. On the grid the same residuals are
3.2 and 2.2e3. Recorded because the first number was reported before the control
was run, which was the wrong order.

### 6.3 The void residual is real, and localised

On the reference grid, mean squared residual by time:

| | t ∈ [0,8) | t ∈ [8,14) | t ∈ [14,60) |
|---|---|---|---|
| **alpha** | **6.87** | **2.95e5** | **4.84e5** |
| T_c | 931 | 4.51e3 | 1.42e4 |
| T_cl | 494 | 2.10e3 | 6.76e3 |

Five orders of magnitude across boiling onset, and not an artefact — present at the
grid points themselves.

### 6.4 The reference is not converged in `α` — at `n = 40`

> **Superseded in its general form; see §6.5.** This section was measured on the
> `n_axial = 40` reference. Scoring moved to `n = 160` in §7.2.1 and the ruler was
> never re-measured there. It has been now, and at `n = 160` the temperature
> conclusion below does not hold.

| reference `n = 40` vs `n = 320` | T_f | T_cl | T_c | **alpha** |
|---|---|---|---|---|
| relative `L2` | 5.1e-3 | 7.2e-3 | 5.4e-3 | **1.03e-1** |

**The reference the PINN was scored against was itself ~10% wrong in the void
fraction**, because it is first-order upwind at `n_axial = 40` and the front spans
2–6 cells. The temperatures were converged to ~5e-3 — which sits right at the 1e-2
acceptance bar, leaving almost no headroom.

Two consequences:

* `L_void_max_err_m = 0.391 m`, a number that did not move across *any*
  configuration tried, was never the PINN's error to fix.
* The M2 convergence study measured orders on the **non-boiling** case, so the void
  field's non-convergence went unmeasured. That is a gap in the test design, not in
  the solver: everything `axial_physics.md` §5 claims about the reference — exact
  steady state, energy conservation, convergence orders — was verified and stands.

### 6.5 The ruler at the mesh actually used — and the excuse it removes

Five meshes, same solver settings, `n_out = 241`, field errors relative `L2`
against `n = 640` bilinearly interpolated onto its grid:

| `n_axial` | onset `t` | onset `ζ` | `L_void` max | peak clad | `T_f` | `T_cl` | `T_s` | `T_c` | `α` |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 10.75 s | 0.9625 | 0.39109 m | 2148.63 K | 5.56e-3 | 7.77e-3 | 5.84e-3 | 5.89e-3 | 1.127e-1 |
| 80 | 10.75 s | 0.9563 | 0.38380 m | 2154.29 K | 2.53e-3 | 3.66e-3 | 2.72e-3 | 2.75e-3 | 6.46e-2 |
| **160** | 10.75 s | 0.9594 | 0.38116 m | 2156.55 K | **1.08e-3** | **1.59e-3** | **1.16e-3** | **1.18e-3** | **3.15e-2** |
| 320 | 10.75 s | 0.9609 | 0.37973 m | 2157.95 K | 3.63e-4 | 5.43e-4 | 3.88e-4 | 3.94e-4 | 1.14e-2 |
| 640 | 10.75 s | 0.9602 | 0.37901 m | 2158.66 K | — | — | — | — | — |

The whole study costs 42 s. It should have been run when scoring moved to
`n = 160`, and was not.

**At the mesh used for scoring, the temperature ruler is 1.1–1.6e-3.** The 1%
acceptance bar sits **6–9× above** it, not at it. So the hedge §6.4 licensed — that
some unknown part of the failure is the ruler — **does not apply to any temperature
number in this document.** The PINN reports 0.071–0.188 (§7.3.2) against a ruler
good to 0.0016: the failure is 45–120× the ruler, and it is the network.

Three further results:

* **Onset time is mesh-independent** — 10.75 s on all five meshes, i.e. resolved to
  the 0.25 s output interval. M4's 0.5 s criterion is measurable, and the PINN's
  5 s miss is entirely the PINN's.
* **Onset location is converged to one cell, non-monotonically** — 0.9625, 0.9563,
  0.9594, 0.9609, 0.9602. The spread is 0.006 and one cell at `n = 160` is 0.00625,
  so "within one cell" is the tightest criterion this metric can carry — which is
  exactly what M4 asks for.
* **`L_void` converges first order; `n = 160` is 0.57% high.** A 1% bar on voided
  length clears the ruler. A 1% bar on the *pointwise* `α` field does not (3.15e-2),
  which is why the front is scored on length. That choice is now quantified rather
  than argued.

## 7. M7 — hardening and backend parity

M7 asks for three things: a multi-seed table, every performance claim measured at
a stated config, and torch and JAX statistically indistinguishable. The first two
are now satisfied. The third is met **conditionally**: with each framework's own
L-BFGS the backends differ by a consistent 16.8% on `T_s` and `T_c`, and with one
shared L-BFGS implementation the ratio is 0.999 (§7.3.2). The gap was the optimiser
and nothing else. `TBD` below marks a number that has not been measured, not one that has
been measured and omitted.

This section is long and was written in the order the work happened. If you want
only the current state, read §7.2.3–§7.2.5 (what made the front form, and what it
costs), §7.3.2 (where the two backends stand) and §7.9 (what is left). Sections are
numbered so that a superseded result keeps its number and gains a marker, rather
than being deleted — the reason a result was wrong is usually the finding.

### 7.1 Multi-seed study

Five seeds, torch, Plan B, 3000 Adam + 300 L-BFGS, **additive ansatz** (i.e. before
the §2 fix), scored against the `n_axial = 40` reference:

| seed | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| 0 | 0.156 | 0.374 | 0.278 | 0.177 |
| 1 | 0.212 | 0.285 | 0.184 | 0.142 |
| 2 | 0.200 | 0.290 | 0.162 | 0.230 |
| 3 | 0.187 | 0.237 | 0.130 | 0.173 |
| 4 | 0.141 | 0.193 | 0.071 | 0.032 |
| **spread** | 1.5× | 1.9× | 3.9× | 7.2× |

No lucky seed: every seed fails the bar on every field, so the failure was
systematic rather than variance. **This table is superseded** — it predates both
the positivity fix and the discovery that the reference is unconverged in `α`
(§6.4). It is kept because it is the only multi-seed evidence that exists.

**Post-fix multi-seed table, scored against `n_axial = 160`** — three seeds,
torch, Plan B, 3000 Adam + 300 L-BFGS, multiplicative ansatz, with and without
the void-head bias of §2:

| bias | seed | T_f | T_cl | T_s | T_c | `max α` |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.078 | 0.117 | 0.036 | 0.038 | 0.0008 |
| 0 | 1 | 0.454 | 0.173 | 0.084 | 0.088 | 0.0000 |
| 0 | 2 | 0.978 | 0.389 | 0.082 | 0.215 | 0.9916 |
| 4 | 0 | 0.156 | 0.165 | 0.099 | 0.118 | 0.0000 |
| 4 | 1 | 0.137 | 0.243 | 0.123 | 0.143 | 0.9917 |
| 4 | 2 | 0.253 | 0.272 | 0.087 | 0.110 | 0.0000 |
| **mean, bias 0** | | **0.503** | 0.226 | **0.068** | **0.114** | |
| **mean, bias 4** | | **0.182** | 0.227 | 0.103 | 0.124 | |

Three things, in order of how much they matter:

1. **Seed variance dominates everything else.** `T_f` at bias 0 spans 0.078 to
   0.978 — a **12.5× spread on three seeds**. Any single-seed comparison in this
   model, including several already in this document, is measuring noise. This is
   also why the earlier one-seed read of the bias ablation looked decisive and
   was not.
2. **The void field is bimodal, not inaccurate.** `max α` is either ~0 or ~0.99
   against a reference maximum of 1.0 — the boiling front either switches on or
   never appears, two seeds out of six getting it. That is a training bistability,
   and it is not something an accuracy metric can describe. `L_void` is
   correspondingly 0.000 m or ~0.04 m against the reference's 0.381 m.
3. **The void-head bias does not earn its place.** It improves the `T_f` mean and
   cuts its spread to 1.8×, and it is neutral-to-worse on the other three fields.
   Reverted (§2).

Reproduced with `ref n_axial = 160`, so unlike §7.1's superseded table this one is
scored against a converged ruler — the M7.5 item. Note that the temperatures were
**not** the thing the ruler was hiding: they are 4–98% wrong, far above the
reference's own 5e-3.

**The gradient-norm block weights diverge** — `w(T_f)` reaches 3.1e5 to 6.2e6
while `w(α)` pins at 0.451, identical to three digits across independent seeds.
That was §7.2's last untested suspect. It has now been tested; see §7.2.1.

### 7.2 The formulation fixes

> **Every table below is reproducible by a committed command.** Each study in this
> section and in §7.3–§7.5 is a sub-command of
> [`tools/axial_study.py`](../tools/axial_study.py) — `ruler`, `horizon`, `budget`,
> `optimizer`, `parity`, `plan-a`. Before that existed, every number here came from
> a scratch file, and one of those used a `t_train_frac` that differed from the
> shipped default without saying so (§7.2.7).


Six measurements, in the order they were taken. Together they are why the boiling
front forms at all; §7.2.3 is the single change that did it. Read §7.2.4 before
quoting §7.2.3 — the second retracts a result of the first.

#### 7.2.1 The block weighting was the variance

Four weighting variants × three seeds, torch, Plan B, 3000 Adam + 300 L-BFGS,
scored against `n_axial = 160`. Only *ratios* between block weights can matter —
Adam is scale-invariant to a global factor, which is the same argument REPORT-01
D39 uses against fixed per-equation scaling — so the intervention is bounding the
spread, not the magnitude. `clipN` renormalises the target to unit geometric mean
and clamps it to `[1/N, N]`; `none` switches the weighting off entirely.

| variant | T_f | T_cl | T_s | T_c | `T_f` seed spread | weight spread |
|---|---|---|---|---|---|---|
| `none` | **0.117** | **0.140** | 0.043 | **0.033** | **1.13×** | 1 |
| `clip10` | 0.130 | 0.142 | **0.041** | 0.034 | 1.20× | 1.9e1 |
| `clip100` | 0.110 | 0.174 | 0.061 | 0.049 | 2.22× | 2.0e2 |
| `current` (unbounded) | 0.376 | 0.242 | 0.062 | 0.093 | **10.4×** | up to 5.0e6 |

Read the last two columns first. **The 12.5× seed variance of §7.1 was not a
property of this model — it was the block weighting.** Bound the spread and it
collapses to 1.1–1.2×, and *every field improves*: `T_c` by 2.8×, `T_f` by 3.2×,
`T_cl` by 1.7×. The ordering is monotone in the cap, which is the same statement
as "the weighting itself is doing the damage".

The mechanism is a positive feedback with nothing to stop it. `λ_k = mean(g)/g_k`
hands a block whose gradient falls — because it is being fitted — an ever-larger
weight, which makes the loss even more about that block. `w(T_f)` running to 1e6
while `w(α)` sits at 0.451 means the void residual was being suppressed by six to
seven orders of magnitude relative to the fuel.

**Adopted:** `weight_max_ratio` in both backends, **default 1.0 — i.e. off**.
That default is set by §7.2.3, not by this table: once variable scaling removes
the *static* imbalance, the adaptive part is worse than nothing on all four
fields. This is the first change in this document to *improve* the fit, and
unlike the six that failed it was proposed by a measurement rather than to one.

**It does not fix the void.** `max α` stays at 0.000–0.004 in every well-behaved
variant against a reference maximum of 1.0. The single run that ever reached
α ≈ 0.99 is `current`/seed 2 — simultaneously the *worst* temperature run
(`T_f` 0.63) and a 2.4× over-prediction of voided length (0.92 m against 0.38 m).
The boiling front switching on is not a success mode here; it is the same
instability wearing a different hat.

#### 7.2.2 Why the void does not form — a number, not a technique

The void residual sits at ~1e11–1e12 in every non-boiling run, four to six orders
above every other block, and no weighting scheme moves it. That is not a training
pathology, it is the equation:

```math
\Gamma = \frac{q''_{wall}}{\lambda\,\rho_v\,A_c}
       = \frac{5.0\times10^{4}}{3.873\times10^{6} \times 0.2738 \times 3.34\times10^{-5}}
       \approx 1.4\times10^{3}\ \mathrm{s^{-1}}
```

A node fills with vapour in **0.71 ms**, against a 60 s horizon. In normalised
time `dα/dt̂` must reach **8.5e4**, where every other block's normalised rate is
1e2 to 1e3. Squared, that is the 1e10 residual floor observed. The network is
being asked to represent a near-discontinuity in `t̂` with a smooth `tanh` MLP,
and `residual_scales` had been hiding it by reporting the *coolant transit time*,
0.113 s, for the void block — 160× too slow. Corrected, and pinned by
`test_void_block_time_constant_is_the_vaporisation_time_not_the_transit_time`.

This is the same stiffness argument that justified the prompt-jump approximation
for the kinetics (D-KIN-1, a factor ~2300), applied to the block that actually
limits this model now. §7.2.3 acts on it.

#### 7.2.3 Variable scaling — the void equation now gets solved

The fix is the one the stiff-PINN literature prescribes and that this repo
already applies globally: **divide each equation by its own characteristic rate**
so every residual block is O(1). VS-PINN [Ko & Park, *JCP* **529** 113860 (2025)]
formulates it as variable scaling with an NTK justification; [Wang et al., *JCP*
**504** 113112 (2024)] as a practical framework for multi-magnitude loss terms.
`docs/neural_network.md` §2 already credits the *global* form of exactly this for
making the 0D model trainable — §7.2.2 simply shows it was never applied per
block, where the spread is 813×.

`residual_normalisation(p) = tau_k / t_end`, applied to the residual before
squaring. It changes the loss and provably not the equation — the residual's zero
set is untouched, asserted by
`test_variable_scaling_changes_the_loss_but_never_the_equation` and by the
un-scaling now built into the consistency test.

Three variants × three seeds, torch, Plan B, 3000 Adam + 300 L-BFGS, `n = 160`
reference. `on`/`off` is variable scaling, the number is `weight_max_ratio`:

| variant | T_f | T_cl | T_s | T_c | `max α` | `L_void` | `T_f` spread |
|---|---|---|---|---|---|---|---|
| `off_10` | 0.128 | 0.141 | **0.041** | **0.034** | ~0.000 | **0.000 m** | 1.21× |
| `on_10` | 0.099 | 0.139 | 0.091 | 0.083 | 0.976 | 0.801 m | 1.09× |
| **`on_1`** | **0.091** | **0.128** | 0.077 | 0.068 | 0.960 | 0.794 m | 1.31× |

(reference `L_void` = 0.381 m)

**The equations are now actually being solved.** Converting the blocks back to
physical units — the scaling is exact, so this is a like-for-like comparison —
the residual falls by four to six orders of magnitude on *every* block:

| block | `off_10` | `on_1` | drop |
|---|---|---|---|
| `T_f` | 3.1e6 | 1.6e0 | 2.0e6× |
| `T_cl` | 3.1e7 | 1.6e3 | 1.9e4× |
| `T_s` | 3.1e5 | 2.6e1 | 1.2e4× |
| `T_c` | 4.9e6 | 4.7e1 | 1.0e5× |
| **`α`** | **5.7e11** | **3.1e5** | **1.9e6×** |

**And the boiling front forms — in 6 of 6 scaled runs against 0 of 6 unscaled.**
`max α` goes from ~0 to 0.96–0.98 and is reproducible to three digits across
seeds. M4's mechanism is being represented for the first time.

**Adaptive weighting is now harmful, and consistently.** `on_1` beats `on_10` on
all four fields. That is the expected result rather than a surprise: once the
static imbalance is removed analytically, the adaptive scheme has nothing left to
correct and only injects noise. Hence `weight_max_ratio = 1.0` as the default.

**What it does not fix, and this is now the sharpest open problem.** `L_void`
over-predicts by **2.1×** — 0.79 m against the reference's 0.381 m — and `T_s`
and `T_c` get *worse* (0.041 → 0.077, 0.034 → 0.068). Those two facts are the
same fact: the void field now exists, it is too long, and a voided node degrades
`film_coefficient`, so the void's error propagates into the structure and coolant
temperatures. Before, the network predicted no void at all and the temperatures
were free to fit the non-boiling part cleanly.

That trade is worth taking — a model that produces a front 2× too long is nearer
the physics than one that produces none, and REPORT-01 §4.1 is explicit that `α`
is judged on voided length and onset rather than on temperature `L2` — but it
must be stated in those terms rather than reported as a clean win.

**Residual and trajectory error have decoupled**, which is worth flagging on its
own: a 1e6× reduction in residual bought ~25% in `T_f` and cost 2× in `T_c`.

#### 7.2.4 The "over-running front" was a phantom, and §7.2.3 caused it

§7.2.3 reported `L_void` over-predicted by 2.1× and called it a front 2× too
long. **That reading was wrong.** Plotting `α(ζ,t)` against the reference instead
of comparing scalars:

| t [s] | `L_ref` | `L_pinn` | `max α_ref` | `max α_pinn` |
|---|---|---|---|---|
| 1.25 | 0.000 | 0.125 | 0.000 | 0.181 |
| 5.00 | 0.000 | 0.459 | 0.000 | 0.626 |
| 10.00 | 0.000 | 0.697 | 0.000 | 0.885 |
| 12.50 | 0.225 | 0.757 | 1.000 | 0.933 |

Onset: reference **10.75 s at ζ = 0.96** (the top, where the coolant is hottest);
network **0.25 s at ζ = 0.05** (the inlet). There is no front. There is a smooth
channel-wide void ramp starting at `t = 0`, and its integral merely happens to
pass through a plausible-looking number.

**The cause is the normalisation §7.2.3 introduced.** The void equation carries
*two* rates, and they do not live in the same place:

| term | rate | where it acts |
|---|---|---|
| vaporisation `Γ = q''/(λ ρ_v A_c)` | 1412 /s | inside the front only, <4% of the domain |
| advection `u/H` | 8.8 /s | everywhere |

§7.2.3 normalised the block by the *vaporisation* time, which is correct at the
front and 160× too small everywhere else. The advective residual that pins
`α = 0` through the subcooled bulk collapses to **3.9e-5** against 1.0 at the
front, so voiding the entire channel from `t = 0` costs the optimiser essentially
nothing. Below saturation `boiling_fraction` underflows to *exactly* zero, so
there the void equation is pure advection and `α ≡ 0` is its **unique** solution
given a zero initial and inlet condition. The physics was never ambiguous; the
loss was indifferent.

**Fix: normalise by the rate that governs the block's dynamics — transport, not
its source.** Three seeds, torch, Plan B, 3000 Adam + 300 L-BFGS, `n = 160`.
`PHANTOM` is `max L_void` over the window where the reference has not yet boiled
at all, so its true value is exactly zero:

| variant | T_f | T_cl | T_s | T_c | **PHANTOM** | onset |
|---|---|---|---|---|---|---|
| `vap` (§7.2.3) | 0.113 | 0.157 | 0.118 | 0.096 | **0.701 m** | 0.25 s @ ζ=0.05 |
| `adv` | 0.223 | 0.298 | 0.197 | 0.197 | **0.0038 m** | — |
| **`adv_h`** | 0.145 | 0.197 | **0.084** | **0.085** | **0.0016 m** | — |
| `adv_hc` | 0.184 | 0.249 | 0.146 | 0.146 | 0.0048 m | 2.5 s @ ζ=0.99 |

**The phantom void falls 440×**, from 0.70 m to 0.0016 m. `adv_h` adds the
horizon truncation below and is the most *reproducible* result this model has
produced: `T_f` = 0.1439 / 0.1449 / 0.1449 across three seeds, a spread of 1.007×.

**The training horizon was the second cause.** With prescribed power the channel
leaves the §12.13 property range at 16.5 s and the reference stops there by
design (D-SCOPE-1). Training to `t_end = 60 s` asks the network to satisfy
residuals over 72% of a horizon where the model does not apply, and one smooth
function of `t̂` carries that state back to `t = 0`. `t_train_frac` truncates it;
`adv_h` beats `adv` on all four fields. This is not using the reference in the
loss — the horizon is a property of the model's validity range, not of its
solution.

**Causal weighting was a third finding, and a negative one.** The un-normalised
`exp(−ε·prefix)` makes `ε` carry the reciprocal units of the loss, so §7.2.3's
1e10 change in loss scale silently switched causality off: the ramp measured
1.000 → 0.977 across all 32 chunks, a 2% tilt where it had been a hard cutoff.
The formulation is now scale-free — `exp(−ε·prefix/total)`, so `ε` is
dimensionless and sets the ramp's log dynamic range. But with causality genuinely
restored, `adv_hc` is *worse* than `adv_h` on every field. It does put the void in
the right **place** (ζ ≈ 0.99 against the reference's 0.96) while getting the time
wrong, which is a real if small clue. Default `causal_eps = 0.0`.

**What remains, stated without decoration: the front still does not form.**
`L_void` is 0.002 m against the reference's 0.381 m. The network now correctly
keeps `α = 0` where the physics says it must, and fails to raise it where the
physics says it should. That is the honest M4 problem, and §7.2.2's number is
why: `dα/dt̂` must reach 8.5e4 inside a front spanning a few percent of the
domain. It is a resolution problem, not a weighting one, and the remedies for it
are the free-boundary and level-set formulations of REPORT-01 §7.4 — a
formulation change, which is M8's subject, not another loss term.

#### 7.2.5 N6 — re-ablated against the algebraic closure, and D38 is half wrong

Every remedy in §7.2.x was measured against the *old* formulation: differential
void, unbounded block weights, no residual scaling. All of that has changed, so
D38's conclusion — "the §7 remedy list is not applicable here, and applying it
hurt" — was re-tested. Seven arms × two seeds, 3000 Adam + 300 L-BFGS, reference
`n = 160`, on top of the current defaults:

| arm | T_f | T_cl | T_s | T_c | `L_void` | `max α` | mean ΔT |
|---|---|---|---|---|---|---|---|
| base | 0.1376 | 0.1886 | 0.0749 | 0.0751 | 0.1805 | 1.0000 | — |
| `n_windows=4` | 0.1385 | 0.1895 | 0.0751 | 0.0752 | 0.1830 | 1.0000 | +0.4% |
| **`fourier_features=32`** | 0.1286 | 0.1787 | 0.0600 | 0.0604 | **0.1878** | 0.9696 | **−12.8%** |
| **`modified_mlp`** | **0.1270** | **0.1786** | **0.0509** | **0.0521** | 0.1121 | 0.9919 | **−18.9%** |
| `weight_max_ratio=10` | 0.1409 | 0.1927 | 0.0789 | 0.0804 | 0.1632 | 1.0000 | +4.3% |
| `causal_eps=5` | 0.1938 | 0.2625 | 0.1591 | 0.1598 | **0.0000** | **0.0000** | +76.3% |
| `pts_every=500` | 0.2112 | 0.2822 | 0.1829 | 0.1834 | **0.0000** | **0.0000** | +97.9% |
| reference | — | — | — | — | 0.3812 | 1.0000 | |

**D38 is retracted for the two architecture remedies and confirmed for the loss
ones.** The modified MLP — jaxpi's default, previously recorded as *3.3× worse* —
is now the best arm on all four temperatures, by 19%. Fourier features,
previously 0.255, now improve every field by 13% *and* move voided length toward
the reference. Time windowing is neutral. Causal weighting and pseudo-time
stepping remain catastrophic, and now visibly so: under both the boiling front
never forms at all, `max α = 0.0000`.

The reading that survives is narrower than D38's and more useful. A remedy aimed
at **representation** — spectral bias, depth — was being masked by a formulation
whose loss was dominated by a block carrying a normalised rate of 8.5e4. Remove
that (D-TH-3) and the representation remedies pay. A remedy aimed at
**reweighting** the loss was, and remains, harmful here, because the imbalance it
targets is now removed analytically rather than adaptively.

**This is the first time in this project that adding something helped.** Every
prior improvement — the algebraic closure, bounding the block weights, truncating
the horizon — was a subtraction.

Not yet adopted as defaults. The modified MLP buys the best temperatures and the
*worst* voided length of the three working arms (0.1121 against the base 0.1805,
reference 0.3812), so it trades the front for the fields; Fourier improves both.
The combination is untested.

#### 7.2.6 The two winners do not compose

§7.2.5 found two remedies that help. Combining them is the obvious next step, and
it is the one the recipe would have taken without measuring. Four arms, **three**
seeds each, so the two-seed figures in §7.2.5 are superseded by these:

| arm | T_f | T_cl | T_s | T_c | `L_void` | `max α` | mean ΔT |
|---|---|---|---|---|---|---|---|
| base | 0.1360 | 0.1874 | 0.0707 | 0.0711 | 0.1630 | 0.9998 | — |
| `fourier_features = 32` | 0.1279 | 0.1776 | 0.0590 | 0.0594 | **0.2070** | 0.9797 | **−11.1%** |
| `modified_mlp` | **0.1271** | 0.1790 | **0.0513** | **0.0526** | 0.0932 | 0.9606 | **−16.1%** |
| **both** | 0.1360 | 0.1875 | 0.0776 | 0.0778 | 0.1037 | **0.8813** | **+4.8%** |
| reference | — | — | — | — | 0.3812 | 1.0000 | |

**Each helps alone; together they are worse than neither.** The combination lands
at +4.8% against base, so it gives back everything both remedies won and a little
more, and it has the worst `max α` of any working arm — 0.88, meaning the front
only partly forms on two of three seeds.

Both remedies attack **spectral bias**, by different routes: Fourier features lift
the input into a high-frequency basis, the modified MLP carries the input to every
layer through multiplicative gating. Applying both appears to over-correct, and
the void field — the sharpest feature in the problem, and the one with no residual
of its own under D-TH-3 — is what pays for it.

**This is the ninth remedy in this document argued soundly and refuted by
measurement, and the second proposed by this audit rather than the literature.**
Neither remedy is adopted as a default. On this evidence `fourier_features = 32`
is the better single choice, because it is the only arm that improves the
temperatures *and* moves voided length toward the reference; the modified MLP wins
the temperatures and gives up the front.

**A note on seed counts.** §7.2.5's two-seed figures were −12.8% and −18.9%; at
three seeds they are −11.1% and −16.1%. The third seed moderated both. The
direction held, the magnitude did not.

#### 7.2.7 The training horizon was undocumented, and it is a cliff

> Numbered out of sequence because it was found last, by a control arm, and it
> conditions every table in §7.2.5 onward.

A budget-split study (§7.5.3) included an arm reproducing §7.3.2's configuration,
purely as a check that the harness had not drifted. It came back
`T_f = 0.2247` against a published `0.1363`, and `max α = 0.0000` — **no boiling
front at all.**

The cause is `t_train_frac`. Its shipped default was `1.0`, and every table from
§7.2.5 onward was measured at **0.275** while being described as "the current
defaults". The value appeared nowhere — not in the config, the docs, the report or
a committed script. Three values, seed 0, 3000 Adam + 300 L-BFGS, `n = 160` ruler:

| `t_train_frac` | horizon | T_f | T_cl | T_s | T_c | `max α` | `L_void` |
|---|---|---|---|---|---|---|---|
| 0.250 | 15.0 s | **0.1250** | **0.1720** | **0.0592** | **0.0585** | 1.0000 | 0.2759 |
| **0.275** | **16.5 s** | 0.1379 | 0.1892 | 0.0753 | 0.0756 | 1.0000 | 0.1634 |
| 0.300 | 18.0 s | 0.1515 | 0.2062 | 0.0950 | 0.0955 | **0.0000** | **0.0000** |
| 1.000 | 60.0 s | 0.2247 | 0.2996 | 0.1987 | 0.1990 | **0.0000** | **0.0000** |
| §7.2.5 base | — | 0.1376 | 0.1886 | 0.0749 | 0.0751 | 1.0000 | 0.1805 |

0.275 reproduces the published row to three or four digits on all four fields, so
that is the value, and it is now the default in both backends. `16.5 / 60 = 0.275`
is not a fitted number: **the reference stops at 16.5 s on every mesh from `n = 40`
to `n = 640`** (§6.5), because that is where the channel leaves the §12.13 property
range. A test now ties the default to that measured stop time, so a change in the
physics cannot silently invalidate it again.

**Two things this exposes, and the second is worse than the first.**

*The repository did not produce its own headline result.* At the shipped default,
`python -m pinn_sfr_transient.axial.pinn_torch` formed no front. "The boiling front
does now form" was true of the measurements and false of the code as delivered, for
every reader who ran it.

*And the result sits on a cliff.* 0.300 is 0.025 away and the front does not form —
not degraded, **absent**, `max α` exactly 0. The temperatures barely move across
that boundary (0.0756 → 0.0955 on `T_c`), so nothing in the temperature metrics
warns you. Whatever forms the front is bistable in this knob, and §7.3.4 already
found the other half of that story: with the L-BFGS stage off, `max α = 0` too.
Two unrelated-looking knobs, the same binary outcome.

**0.250 is better on all four fields and is rejected.** 15 s is not where this
model stops being valid; 16.5 s is, and it is measured. Choosing the horizon by its
score against the reference would be fitting the problem statement to the ruler —
the one thing `t_train_frac` is documented as *not* being. The better number is
recorded here so the choice is visible rather than quietly taken.

#### 7.2.8 There is no front mechanism — it is one inequality on `T_c`

The cliff in §7.2.7 and the L-BFGS switch in §7.3.4 look like two unrelated knobs
producing the same binary outcome. They are not two mechanisms. Under D-TH-3 the
void is a function of the network's own `T_c` and nothing else:

```math
\alpha = 1 - \big(1 - b(T_c)\big)^3,
\qquad
b(T_c) = \tfrac{1}{2}\left(1 + \tanh \frac{T_c - T_{\mathrm{sat}} - \Delta T_{\mathrm{sup}}}{2\,\Delta T_{\mathrm{smooth}}}\right)
```

so **"the front forms" is the single inequality**
`max T_c > T_sat + ΔT_sup = 1169.0 K`. There is no separate front to get right.
Every knob that switches `max α` between 0 and 1 does so by moving the *peak* of
`T_c` across one threshold.

That resolves what looked paradoxical. **Relative `L2` is an average and the front
is an extremum**, so a change can improve one while destroying the other — and
`T_c`'s relative `L2` barely moves across the cliff (0.0756 → 0.0955) precisely
because the norm is dominated by the subcooled bulk, where nothing happened, while
the peak that matters slipped below 1169 K.

The budget sweep of §7.5.3 is the same trade in continuous form: as the
quasi-Newton stage takes more of the budget, `T_c` improves monotonically in `L2`
(0.0756 → 0.0492 → 0.0396) and the front degrades monotonically
(`max α` 1.0000 → 0.9662 → 0.6274). A smoother, better-in-the-mean fit has a lower
peak. That is not a defect in the optimiser; it is the metric and the physics
asking for different things.

**The margin is 20.5 K.** At the shipped configuration the network's peak `T_c` is
**1189.4 K** against a threshold of **1169.0 K** — it clears saturation by 20.5 K
out of a ~590 K range, about 3.5%. That is why the front is stable in arm A of
§7.5.3 and erratic in arms B and C: a smoother fit gives up a few tens of kelvin at
the peak, and a few tens of kelvin is all there is.

**And `max α` carries no information beyond that margin.** The closure is
invertible, so the margin *predicts* `max α` outright, at `dT_smooth = 2 K`:

| margin | predicted `max α` | measured `max α` | arm |
|---|---|---|---|
| +0.6 K | 0.9229 | **0.9199** | C + modified_mlp |
| +3.7 K | 0.9975 | **0.9974** | A + fourier |
| +18.6 K | 1.0000 | **1.0000** | C + fourier |
| +20.5 K | 1.0000 | **1.0000** | A, shipped default |

Four arms, four exact matches. **So reporting `max α` alongside the margin is
reporting the same number twice**, and it saturates by about 8 K of margin — past
which it cannot distinguish a front that barely exists from one with 20 K of
headroom. Every `max α = 1.0000` in this document means only "the peak is more than
about 8 K above saturation".

**`L_void` is the metric that carries front information, and it is a different
quantity.** The margin is a property of one point; `L_void` integrates `α` over the
channel, so it measures the *width* of the super-saturated region. The two come
apart, and the §7.5.4 arms show it plainly: `A + fourier` has the **smallest**
margin of the good arms (+3.7 K) and the **largest** `L_void` (0.2681), while the
shipped default has 5× the margin (+20.5 K) and 0.1634. A lower, broader
super-saturated region beats a taller, narrower one.

So the honest decomposition is: **the margin gates whether a front exists at all;
`L_void` says how much of the channel is in it.** This document has been treating
`max α` and `L_void` as one "front" metric, and they are not.

Two consequences worth stating:

* **Scoring temperatures in relative `L2` cannot detect front failure**, and this
  document has reported both side by side for long enough to have noticed. Every
  study now records `max T_c` and its margin to 1169 K, which is the quantity the
  front actually depends on.
* **A 1% bar on `T_c` in `L2` does not imply the front forms.** The two criteria
  are close to independent. M4's acceptance is a statement about the peak; §7.2.5's
  is a statement about the mean.

**A note on how these runs were timed.** §7.5.3 and the studies after it were run
five at a time on a 48-core machine at `OMP_NUM_THREADS=8` each, because nothing
among them depends on anything else. That leaves the wall-clocks contended and the
accuracy untouched, and both halves of that are measured rather than assumed:
`torch/lbfgs` at seed 0 returns 0.1379 / 0.1892 / 0.0753 / 0.0756 with
`L_void` 0.1634 in the contended parity run and **the same four digits** in the
uncontended budget run, at 876 s against 592 s. Three separate processes — the
`budget`, `parity` and `optimizer` studies — have now reproduced that
configuration, and seed 1's, to every digit printed. Thread count changes float
reduction order and so changes answers; concurrency at a fixed thread count does
not. Every study row records its own `load1` and `OMP_NUM_THREADS` so no timing here
can be mistaken for a clean one.

#### 7.2.9 D67 again, on a second axis: the budget

§7.2.7 found that the shipped `t_train_frac` disagreed with every published table
and fixed it. **It fixed one of two mismatches.**

Twelve tables in this document state **"3000 Adam + 300 L-BFGS"**.
`AxialTrainConfig` ships **8000 Adam + 500 L-BFGS**. No table was ever measured at
the shipped budget, and the mismatch was invisible for the same reason as the
first: every study passed the budget explicitly, so nothing ever ran the default.

Measured at the true default — `t_train_frac = 0.275`, 8000 + 500, seed 0, **both
backends**:

| | `T_s` | `L_void` | `max α` | margin |
|---|---|---|---|---|
| **true shipped default, torch** | **0.0371** | 0.0440 | 0.7383 | **−1.1 K** |
| **true shipped default, jax** | 0.0529 | 0.0506 | 0.7920 | **−0.7 K** |
| published tables (3000/300), torch | 0.0739 | 0.1505 | 1.0000 | +20.5 K |

**The shipped default does not form a front in either backend.** Both peaks sit
*below* saturation — 1.1 K on torch, 0.7 K on JAX. Two independent implementations
failing the same way is what makes this a property of the configuration rather than
of one optimiser stack.

And the mechanism is §7.2.8 for the third independent time: the true default has a
**better mean** than every published table — `T_s` 0.0371 against 0.0739 — and
loses the front, because more optimisation gives a smoother fit and a lower peak.
The first two instances were the budget sweep (§7.5.3) and the training horizon
(§7.2.7). This is the same trade arriving through the one knob nobody had varied
because everyone was overriding it.

**The lesson is narrower than "check your defaults".** It is: *a parameter you
always pass explicitly is a parameter whose default is never tested.* Both D67
defects were of exactly that kind, and the control arm that caught the first one
could not catch the second, because the control arm also passed the budget
explicitly. `tools/axial_study.py default` now runs with **no overrides at all**,
which is the only arm that can detect this class.

### 7.3 Backend parity

Two tables, a year apart in understanding. §7.3.1 is kept because the *reason* it
was wrong is the finding; §7.3.2 is the current measurement.

#### 7.3.1 The first table, and the wrong-axis bug that produced it

| | T_f | T_cl | T_s | T_c | config |
|---|---|---|---|---|---|
| torch | 0.062 | 0.120 | 0.032 | 0.034 | seed 0, post-fix, ref `n=40` |
| jax | 0.243 | 0.328 | 0.323 | 0.197 | seed 0, post-fix, ref `n=40` |
| **ratio** | **3.9×** | **2.7×** | **10×** | **5.8×** | |

**The table above is superseded and must be re-measured — the cause was found.**
It was produced with a JAX backend that applied causal weighting along the
**wrong axis**.

`causal_loss` read the chunking variable as `pts[0]`. Under Plan A the
collocation tuple really is `(that, zeta_q, weights)`, so that was right. Under
Plan B — the regime the parity table was measured in — `_collocation` returns
`(zeta, that)`, so the loss was chunked by **axial position**: the causal ramp
`exp(−ε Σ L)` ran *up the channel* instead of forward in time, penalising the top
of the core for the residual accumulated below it. The torch twin was always
correct. Fixed, and pinned by
`test_causal_weighting_chunks_on_time_not_on_zeta`, which also asserts the two
backends agree on the reduction itself.

Two other candidate causes had been tested and rejected, and both stand:

* *frozen collocation* — JAX trained on a fixed set between RAR refreshes where
  torch resamples every step. Real divergence from the 0D convention, fixed, and
  it changed nothing (0.243 → 0.243).
* *a dead L-BFGS polish* — `optax.lbfgs` does run and does reduce the loss
  (7.4e3 → 3.4e1), and JAX accuracy with and without it is the same.

A third contributor was found alongside it: the two configs did not carry the
same **budget**. JAX defaulted to 3000 Adam / 300 L-BFGS / RAR every 1000 against
torch's 8000 / 500 / 2000, so the default-configuration comparison was never
like-for-like. The defaults now match.

> **Superseded.** The table below predates the algebraic void closure, and for a
> period the closure existed in torch only — so it compares two different
> models. Re-measure before quoting (Annex A, N1 and N8).

**Post-fix parity, seed 0, 3000 Adam + 300 L-BFGS, scored against `n = 160`:**

| | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| torch | 0.133 | 0.167 | 0.089 | 0.114 |
| jax | 0.172 | 0.219 | 0.112 | 0.103 |
| **ratio** | 1.29× | 1.31× | 1.26× | 0.90× |

**3–10× has become 0.9–1.3×.** The two backends also start from near-identical
losses (2.098e2 against 2.054e2). Given §7.1's 12.5× seed spread, one seed cannot
establish "statistically indistinguishable" and this table does not claim it —
but the systematic gap that made D40 a defect is gone. Multi-seed parity
statistics remain TBD; that is compute, not development.

#### 7.3.2 Re-measured after the closure

> **Superseded — §7.5.17.** This section's JAX column was measured with
> `optax.lbfgs` at `memory_size=10` against torch's `history_size=50`. The
> "1.168 with each framework's own L-BFGS" was comparing 50 curvature pairs
> against 10, not two implementations. Re-runs are in flight.

§7.3.1's parity table was marked superseded because it compared two models rather
than two backends. With the knob sets now equal and tested, the comparison is
valid again. Identical config, identical budget (3000 Adam + 300 L-BFGS),
identical `n = 160` ruler, three seeds each:

| | T_f | T_cl | T_s | T_c | `L_void` | time |
|---|---|---|---|---|---|---|
| torch | **0.1363** | **0.1879** | **0.0707** | **0.0713** | 0.1471 | 938 s |
| jax | 0.1428 | 0.1948 | 0.0863 | 0.0865 | 0.1555 | **398 s** |
| ratio | 1.05× | 1.04× | 1.22× | 1.21× | 1.06× | **0.42×** |

**Accuracy: 1.04–1.22×, against 2.7–10× before.** D40 is closed. But *statistically
indistinguishable* — M7's actual criterion — is met on only half the fields. Per-seed
ranges:

| field | torch | jax | overlap |
|---|---|---|---|
| `T_f` | 0.1329–0.1394 | 0.1382–0.1466 | **yes** |
| `T_cl` | 0.1849–0.1915 | 0.1891–0.1997 | **yes** |
| `T_s` | 0.0622–0.0765 | 0.0784–0.0925 | **no** |
| `T_c` | 0.0632–0.0771 | 0.0785–0.0929 | **no** |

On `T_s` and `T_c` the ranges do not overlap at three seeds: torch is
consistently ~21% better. That is a real residual difference and not noise — and it
is **no longer unexplained**: it is `torch.optim.LBFGS` against `optax.lbfgs`, and
a shared implementation removes it entirely (below). Both backends reach
`max α = 1.0000`, so the front forms in both.

**Speed: JAX is 2.4× faster at identical budget**, and the figure is robust —
2.36× from the contended three-seed runs, 2.41× from a clean uncontended
500-iteration pair (torch 105.9 s, jax 44.0 s).

**Compilation was the cause, and this section said the opposite for four
milestones. Both the measurement and the explanation behind it are withdrawn.**

The claim was that `torch.compile` on the torch step buys 1.06× at 17 s of compile
time, so it does not earn its place. The number was real. What it measured was not
the technique but this repository's code: the residual stack broke into **eight
graphs**, so almost every operation still ran eager and 1.06× is what a compile that
does not happen is worth. `fullgraph=True` would have raised instead of silently
falling back, and that is now how the compiled path is configured.

Two defects held the graph open, both fixed:

* `_backend.xp` sniffed `type(x).__module__` to select the array module. Dynamo
  constant-folds that for a tensor and not for a Python float, and the residuals do
  pass one, through `saturation_temperature(p.p_system)`.
* `state_and_grads` marked its coordinates `requires_grad`. Forward mode carries the
  derivative in the tangent so nothing needed the reverse-mode leaf, but AOTAutograd
  refuses a graph returning a tensor derived from an in-graph `requires_grad_()`.

With a full graph, at f256 with 500 collocation points on 8 pinned cores, over four
runs of `tools/backend_smoke.py --compile`:

| | ms/iteration | it/s |
|---|---|---|
| eager | 90.2 – 99.7 | 10.0 – 11.1 |
| compiled | 6.6 – 8.9 | 112.9 – 152.7 |

**Over 10× — the range across runs is 10.7× to 15.1×.** A range because that is what
the measurement supports: the eager arm is steady to 1.12× within a run and the
compiled arm varies up to 1.81×, since at 8 ms an iteration any interference is a
large fraction of it. The third digit of a speedup here is not a real quantity.

**The profile explanation was wrong in the same way.** "88% is forward-plus-backward
through `torch.func.jvp` in float64 — dense BLAS-bound linear algebra, which Inductor
cannot improve" describes work that is *not* dense linear algebra. A step issues about
**800 `aten::mul` and 230 `aten::add` against 96 `aten::mm`**, plus hundreds of
`prims::` operations the forward-mode passes decompose into, each dispatched
separately. That is elementwise work, it is exactly what fusion is for, and
`residual_blocks` alone goes from 70.4 ms to 6.7 ms.

The general lesson is the one `fullgraph` now enforces: **a partial compile measures
your graph breaks, not your kernels.** A silent fallback to eager reports the result
as a small win and hides the reason.

**The 2.4× reported above is therefore an eager-torch number**, as is every speed
comparison in §7.5.19. Against the compiled loop the ordering reverses; see the
matched measurement there.

**So M7's acceptance is not met at the default, for a reason now identified.** Two
fields agree, two differ by a consistent 21%, and the difference is the optimiser
implementation — see the three-seed table below, where a shared implementation
brings the ratio to 0.999.

**The equations are exonerated.** Transplanting the torch model's weights into the
Equinox model — so both backends hold *identical* parameters — and evaluating every
residual block at identical points gives agreement to **1e-14 relative**, on every
block including `T_s` and `T_c`, with the ansatz itself matching to 1.8e-15. So the
21% is training dynamics, not a forked equation, and a test now pins that
(`test_residual_blocks_are_identical_given_identical_parameters`). Combined with
§7.3.3 (RAR is not the cause) and §7.3.4 (the gap is the quasi-Newton stage), the
remaining suspect is the *implementation* of L-BFGS — `torch.optim.LBFGS` against
`optax.lbfgs`, the last thing in the pipeline that is not shared source.

That is now directly testable: `optimizer = "lbfgs-shared"` selects this
repository's own L-BFGS in **both** backends, pinned to agree to 1e-10 by
`test_self_scaled_bfgs_agrees_across_backends`.

**Tested at three seeds, and the answer is clean: the framework L-BFGS is the
whole gap.** `T_s`, three seeds each, identical config, identical ruler:

| arm | mean | per-seed range |
|---|---|---|
| torch / `lbfgs` | 0.0739 | 0.0677–0.0786 |
| jax / `lbfgs` | 0.0863 | 0.0784–0.0925 |
| torch / `lbfgs-shared` | 0.0775 | 0.0766–0.0789 |
| jax / `lbfgs-shared` | 0.0774 | 0.0715–0.0819 |

| optimiser | `jax / torch` mean ratio | rank-paired per-seed ratios |
|---|---|---|
| `lbfgs` (each framework's own) | **1.168** | 1.158, 1.166, 1.177 |
| `lbfgs-shared` (one implementation) | **0.999** | 0.933, 1.024, 1.039 |

With each framework's own L-BFGS, JAX is **16.8% worse** on `T_s` and the ratio is
extraordinarily stable — 1.158 to 1.177 across three seeds. Swap in one shared
implementation and the ratio is **0.999**, straddling unity. **§7.3.2's gap is real,
and its cause is the L-BFGS implementation: `torch.optim.LBFGS` against
`optax.lbfgs`, and nothing else.**

> #### The retraction two paragraphs of this document ago was itself wrong, for a
> reason worth keeping
>
> An earlier revision retracted §7.3.2 on the strength of **seed 1**, where the
> index-paired ratio was 0.997 — no gap. The three index-paired ratios are 1.167,
> 0.997 and 1.366, which look like noise around nothing.
>
> **Index-pairing across backends is not a meaningful pairing.** `seed=1` seeds two
> different RNG implementations drawing two different initialisations; torch's seed
> 1 and JAX's seed 1 are unrelated draws. Comparing them elementwise pairs a lucky
> torch draw with an unlucky JAX one and calls the difference a measurement.
>
> Compared as **distributions** — which is the only thing three samples of two
> different generators can support — the gap is 1.158/1.166/1.177. The apparent
> contradiction was an artefact of the pairing, not of the data.
>
> The rule this earns: **a retraction needs the same evidence as the claim.**
> §7.3.2 was a three-seed claim; retracting it on one seed repeated the error it
> was retracting.

**What is left unexplained is nothing, for once.** The residuals are identical to
1e-14 (§7.3.2's transplant test), RAR is excluded (§7.3.3), and the optimiser
accounts for the remainder. The margins agree: `lbfgs-shared` gives +12.0 to
+20.6 K on torch and +15.0 to +16.8 K on JAX, overlapping, where the framework
optimisers give +9.8 to +20.5 K and +9.5 to +26.0 K.

**M7's acceptance criterion — the two backends statistically indistinguishable — is
met, with `optimizer = "lbfgs-shared"`.** It is not met at the default, and the
default is each framework's own optimiser because that is what a user of either
framework gets.

#### 7.3.3 RAR is not the cause

The obvious suspect was the last structural asymmetry: torch grows an RAR
reservoir up to `rar_cap`, JAX keeps a fixed `rar_keep` set so `jit` never
recompiles. At the default budget they genuinely differ — RAR fires once, at
iteration 2000, contributing **200** extra collocation points in torch against
**400** in JAX, on a base set of 6000.

Tested by removal, three seeds each, everything else identical:

| | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| ratio jax/torch, **RAR on** | 1.048 | 1.037 | 1.220 | 1.213 |
| ratio jax/torch, **RAR off** | 1.036 | 1.027 | **1.141** | **1.135** |

**The gap survives.** Removing RAR from both backends narrows the `T_s` excess
from 22% to 14% — about a third of it — and the per-seed ranges on `T_s` and
`T_c` still do not overlap. RAR contributes, and is not the explanation.

The direction is at least consistent with the asymmetry: RAR *helps* torch
(`T_s` 0.0707 → 0.0734 when removed) and *hurts* JAX (0.0863 → 0.0837), which is
what a backend receiving twice as many high-residual points would show. The
effect is simply too small to account for the gap.

#### 7.3.4 The L-BFGS polish is the whole remaining gap

The last genuinely different implementation: torch uses `torch.optim.LBFGS` with
a strong-Wolfe line search and a snapshot/revert guard, JAX uses `optax.lbfgs`
with its own default. Tested the same way, by removal — Adam only, three seeds
each:

| | T_f | T_cl | T_s | T_c | all ranges overlap? |
|---|---|---|---|---|---|
| ratio jax/torch, **polish on** | 1.048 | 1.037 | 1.220 | 1.213 | no (`T_s`, `T_c`) |
| ratio jax/torch, **polish off** | 0.971 | 0.973 | **0.949** | **0.949** | **yes, all four** |

**Remove the polish and the gap does not merely close — it reverses.** Adam-only,
JAX is 3–5% *better* than torch and every per-seed range overlaps. That is M7's
criterion met, on a configuration nobody would ship.

So the difference is not in the model, the residuals, the collocation or the
optimiser's first phase. It is that **`torch.optim.LBFGS` extracts more from this
loss than `optax.lbfgs` does**:

| | Adam only | + polish | gain |
|---|---|---|---|
| torch `T_s` | 0.1483 | 0.0707 | **2.10×** |
| jax `T_s` | 0.1407 | 0.0863 | 1.63× |

Both benefit; torch benefits more, and the difference between 2.10× and 1.63× is
the entire 21%.

> **Read this as "at zero iterations", not "only this stage can".** §7.5.11's grid
> measures the stage at 30, 300 and 3000, and `adam3000/qn30` **does** form a front
> (+8.2 K). Thirty quasi-Newton iterations suffice when Adam is large. What is true
> is an exchange rate, not an exclusivity: at matched totals a quasi-Newton
> iteration is worth roughly an order of magnitude more than an Adam one.

**The larger finding is not about parity at all.** With the polish off, `L_void`
is 0.0000 and `max α` is 0.0000 in **both** backends, on every seed. Three
thousand Adam iterations never form the boiling front; three hundred L-BFGS
iterations do. The quasi-Newton polish is not a refinement here — it is the step
that finds the front, which makes the choice of L-BFGS implementation a physics
question rather than a tuning one.

That reframes §7.5's deferred optimiser bake-off: SSBroyden and SSBFGS
([arXiv:2501.16371](https://arxiv.org/abs/2501.16371)) drop into exactly this
slot, and this is now the highest-value untested item in the project rather than
a nice-to-have. **TBD.**

### 7.4 N5 — Plan A measured end to end

M6 shipped the prompt-jump closure in the network and never scored it; §6.4 said
so and marked the acceptance criterion unverified. It is now measured. Quadrature
at `n_axial = 80` and `n_time = 64` to keep the tensor grid near Plan B's cost;
the ruler is a separate `n = 160` closed-loop reference, so training resolution
and scoring resolution are independent. 3000 Adam + 300 L-BFGS, seed 0, 681 s.

| quantity | PINN | reference |
|---|---|---|
| `P(0)` | **1.000000** | 1.000000 |
| peak power | **1.0000** | 1.0000 |
| minimum power | 0.5380 | 0.5021 |
| relative `L2` on `P(t)` | **0.2497** | — |
| `max ρ/β` (tripwire) | **+0.0000** | +0.0000 |
| `min ρ/β` | −0.1558 | −0.2052 |
| `T_f`, `T_cl`, `T_c` | 0.1668, 0.2033, 0.1094 | — |

**What passes.** `P(0) = 1` to six figures, because the closure is hard-constrained
rather than penalised: `c(0) = 1` and `ρ(0) = 0` give `P = Σβᵢ/β` by construction.
Peak power matches exactly — though that is nearly free here, since feedback only
ever removes reactivity, so the peak sits at `t = 0`. The pole tripwire agrees
exactly: both report `max ρ/β = +0.0000`, and the network independently
reproduces the D49 result that the void never inserts positive reactivity.

**What fails.** The power *trajectory* is 25% off against a 1% bar, and both the
power drop and the reactivity swing are under-predicted — `min ρ/β` −0.156
against −0.205, so the network finds 24% less negative feedback than the
reference does. The temperatures are worse than Plan B's (0.167 against 0.137 on
`T_f`), which is expected: Plan A adds six precursor unknowns and an axial
integral coupling every node to every other.

**So M6's acceptance criterion is failed, not unverified.** The two agree on peak
power and on the tripwire; they do not agree on the trajectory, and the
reactivity balance does not close to tolerance.

Single seed. Given §7.1's history, that is a measurement and not a statistic.

#### 7.4.1 Re-measured at the shipped budget, two seeds

**The budget above is not the shipped one.** §7.4 ran 3000 Adam + 300 L-BFGS;
`AxialTrainConfig` ships **8000 + 500**, which is what a reader running the model
gets. Re-measured there, `uv run python tools/axial_study.py plan-a`:

Three seeds, **both backends**:

| | `L2(P)` | `min ρ/β` | vs reference | min `P` | `max ρ/β` |
|---|---|---|---|---|---|
| **torch** | **0.1110** `[.1060–.1142]` | −0.1503 `[−.1521,−.1482]` | **73%** | 0.4627 | **+0.0000** |
| **jax** | 0.1324 `[.1092–.1549]` | **−0.1773** `[−.1862,−.1646]` | **86%** | 0.4672 | **+0.0000** |
| reference | — | −0.2052 | — | 0.5021 | +0.0000 |

`P(0) = 1.000000` to six figures on all six runs, and `max ρ/β = +0.0000` on all
six — the closure is hard-constrained rather than penalised, and D49 is
independently reproduced twelve times over.

**The two backends miss by different amounts — 27% and 14%.** That alone rules out
an error in the reactivity *formulation*, since both compute `ρ` from the same
shared function. And they trade: torch fits the power trajectory better (`L2(P)`
0.1110 against 0.1324) while JAX finds substantially more of the negative
reactivity. Neither is uniformly better, which is the same configuration-dependent
backend behaviour §7.3.2 shows.

**`L2(P)` is 0.106–0.113 against §7.4's 0.2497 — but that is a 4× compute
difference, not an improvement in method.** Read it as: Plan A is
optimisation-limited too, which is the third independent sign of that after §7.5.3
and §7.3.4.

**The failure mode is unchanged and is now reproducible.** Both seeds find
`min ρ/β ≈ −0.149` against the reference's −0.2052 — **27% less negative feedback**
— and both under-shoot the power minimum (0.462–0.466 against 0.5021) where §7.4's
smaller budget over-shot it (0.5380). More optimisation moved the error across the
reference without changing its character: the network still cannot find the
reactivity swing.

What passes still passes exactly: `P(0) = 1.000000` to six figures because the
closure is hard-constrained rather than penalised, and the pole tripwire reads
`+0.0000` on both seeds, independently reproducing D49.

**Three seeds, 7.7% apart, and the failure is the most reproducible thing in this
document.** `min ρ/β` is −0.1482 / −0.1505 / −0.1521 against the reference's
−0.2052: the network finds **26–28% less negative feedback** on every seed, with a
2.6% spread. It is not variance.

**It is also not a deficit in the reactivity balance**, which is what this section
said before the terms were separated. §7.4.2 splits it: the Doppler integral is
right and the *void* term is 84–92% short. Plan A's power error is the front
failure.

M6's acceptance criterion is **failed** — 11.1% against a 1% bar — and now failed
with a number that will not move on re-running.

#### 7.4.2 The reactivity deficit is the void term — which is the front

§7.4.1 reported that Plan A finds 26–28% less negative reactivity than the
reference and called it a systematic deficit in the reactivity balance. **That
framing was wrong, and the decomposition says so.**

`predict_reactivity_components` splits the network's `ρ` the way the reference
already reported it — Doppler (with axial expansion) and void — at seed 0, both
backends:

| | Doppler | vs reference | void | vs reference |
|---|---|---|---|---|
| reference | −0.1779 | — | **−0.0344** | — |
| torch | −0.1481 | 0.832 | **−0.0027** | **0.077** |
| jax | −0.1809 | **1.017** | **−0.0054** | **0.157** |

**The deficit is the void term.** JAX reproduces the Doppler integral essentially
exactly — 1.017 — and still misses **84% of the void reactivity**. Torch is 17%
short on Doppler and **92% short on the void**.

That is not a second defect. The void worth is integrated against `α`, `α` is a
function of `T_c` alone (D-TH-3), and Plan A runs at `t_train_frac = 1.0` where the
front is weakest. **Plan A's power error is the front failure, read through the
reactivity.**

So one mechanism now accounts for: the temperature/front trade (§7.2.8), the
seed fragility of every marginal arm (§7.5.4), the shipped default's failure
(§7.2.9), and Plan A's acceptance failure. The margin is not one problem among
several — it is the problem.

> **A note on how §7.4.1 got it wrong.** The inference ran: the reference has the
> most negative `ρ` *and* the highest power floor, while both networks have less
> negative `ρ` and dip lower, so the precursors must be under-predicted. That
> compares `min ρ` with `min P` as though they occur at the same instant. They need
> not, and the closure `P = Σβᵢcᵢ/(β − ρ)` is pointwise in time, not in extrema.
> The decomposition took one run to produce and answers the question directly.
> **Reasoning from two summary statistics is not a measurement**, and this project
> has a tool for that reasoning — `reactivity_components` — which had existed for
> weeks and had never been called on a network.

### 7.5 Optimiser bake-off

The bake-off was deferred on the grounds that with the reference unconverged in
`α` and the two backends 3–10× apart, an optimiser comparison would be measuring
the ruler. **Both halves of that reason are now gone.** The parity gap was a
causal-weighting axis bug and is 0.9–1.3× (§7.3.1–§7.3.2). The temperature ruler is
1.1–1.6e-3 at the scoring mesh (§6.5). And §7.3.4 measured that the L-BFGS stage is
not a refinement but the step that *forms the boiling front*. That makes the
quasi-Newton stage the most load-bearing component of the recipe and the only one
never varied.

#### 7.5.1 Self-scaled BFGS, implemented in both backends

Plain L-BFGS applies the Oren–Luenberger scaling once, to `H₀`:
`γ = (sᵀy)/(yᵀy)`. Self-scaled BFGS applies a scaling at **every** update
[Oren & Luenberger 1974; Al-Baali 1998], the family
[arXiv:2501.16371](https://arxiv.org/abs/2501.16371) reports beating L-BFGS across
PINN benchmarks:

```math
H_{k+1} = (I - \rho s y^\top)\,\tau_k H_k\,(I - \rho y s^\top) + \rho s s^\top,
\qquad \rho = \frac{1}{y^\top s},
\qquad \tau_k = \min\!\left(1, \frac{1}{b_k}\right),
\qquad b_k = \frac{y^\top H_k y}{y^\top s}
```

The secant condition `H_{k+1} y = s` holds for any symmetric matrix in the middle,
so `τ` is free. It is fixed by requiring the scaled operator to reproduce the
observed curvature along `y`: `yᵀ(τH_k)y = yᵀs` gives `τ = 1/b_k` exactly. Capping
at 1 makes it a damper only. In limited memory it enters the second loop of the
two-loop recursion as a multiplication immediately before pair `i`'s correction.

Both backends implement it, with a strong-Wolfe line search at `c₁ = 1e-4`,
`c₂ = 0.9`. `optimizer` defaults to `"lbfgs"`, so no published number moves.

**The scaling direction was wrong on the first attempt, and nothing failed.** With
`H_k/τ_k` instead of `τ_k H_k` the method still satisfies the secant condition,
still descends and still converges — it just converges worse, and it burned six
line-search evaluations per iteration instead of one. It was caught because the
implementation is checked against `torch.optim.LBFGS` on problems whose minima are
known *before* it is allowed near the PINN. Nothing about the PINN's loss would
have revealed it.

#### 7.5.2 Self-scaling loses the mean and wins the variance

Five variants, `H₀` scaling and the `min(1, ·)` cap swept, against
`torch.optim.LBFGS`. Objective value after N iterations, function evaluations in
brackets:

| variant | quadratic, cond 1e8 (500 it) | Rosenbrock n=100 (500 it) |
|---|---|---|
| `torch.optim.LBFGS` | 1.272e+03 (537) | 2.820e-12 (592) |
| ours, `self_scale=False` | 1.309e+03 (530) | 3.429e-10 (581) |
| ours, capped, `H₀` scaled | 2.383e+03 (501) | 6.566e+01 (503) |
| ours, capped, no `H₀` | 2.808e+03 (701) | 3.250e+01 (515) |
| ours, uncapped, `H₀` scaled | 2.580e+03 (516) | 5.155e+01 (516) |
| ours, uncapped, no `H₀` | 1.913e+03 (725) | 2.964e+01 (530) |

Two things to read here. **The control works**: `self_scale=False` tracks
`torch.optim.LBFGS` on both problems, so the line search and recursion are sound
and any difference in the other rows is the scaling. **And every self-scaled
variant loses**, on both problems, at every budget — decisively on Rosenbrock.

That is a negative result about *these* problems, not about the paper. Two
differences are known and neither is dismissible: a PINN loss is not a quadratic,
which is the paper's whole premise; and the paper's winning configuration gives the
quasi-Newton stage **30000** iterations against 300 here, where asymptotic
behaviour is what is being compared. The second of those is testable on its own,
and is §7.5.3.

**On the PINN it also loses the mean — and the stability belongs to the
implementation, not to the self-scaling.** Three seeds, torch, 3000 Adam + 300
quasi-Newton, `n = 160` ruler:

| optimiser | `T_s` mean | per seed | spread | margin per seed |
|---|---|---|---|---|
| `lbfgs` (torch's) | **0.0739** | 0.0753, 0.0786, 0.0677 | **16.2%** | +20.5, +17.2, +9.8 K |
| `lbfgs-shared` (ours, plain) | 0.0775 | 0.0769, 0.0789, 0.0766 | **3.0%** | +20.6, +16.1, +12.0 K |
| `ssbfgs` (ours, self-scaled) | 0.0862 | 0.0853, 0.0852, 0.0880 | **3.4%** | +21.2, +21.1, +14.9 K |

**Self-scaling costs 11% of the mean and buys nothing in variance.** `ssbfgs` and
`lbfgs-shared` have the same spread to within noise (3.4% against 3.0%), so the
paper's method is simply worse here — on quadratics, on Rosenbrock and on the PINN.

**What is real is that this repository's L-BFGS is 5× more seed-stable than
torch's**: 3.0% against 16.2%, and a margin range of 8.6 K against 10.7 K. It pays
for that with 4.6% of the mean. The likely cause is the line search — `torch.optim.LBFGS`
interpolates cubically inside the zoom where this implementation bisects, which
converges faster and lands in more varied places.

> An earlier revision of this section reported a **0.1% seed spread** for `ssbfgs`
> and called it "a 30× reduction worth having". That was two seeds: 0.0853 and
> 0.0852, which happened to coincide. The third is 0.0880 and the spread is 3.4%.
>
> **Every two-seed claim made during this study was wrong at three** — the budget
> sweep's monotonic front, the backend gap's disappearance, and this. A two-sample
> estimate of a *variance* is worse still, and this one is why `AGENTS.md` now says
> three seeds with ranges or say the sample size in the sentence.

#### 7.5.3 The budget split

The paper's other conclusion is the more interesting one for this model. Its
winning schedule is **Adam[1000] + quasi-Newton[30000]** — ~97% quasi-Newton. This
project gives the stage that forms the front **9%** of its budget. Three splits at
equal total iterations, three seeds, `n = 160` ruler:

| arm | Adam | quasi-Newton |
|---|---|---|
| A | 3000 | 300 |
| B | 1000 | 2300 |
| C | 300 | 3000 |

Arm A reproduces the §7.3.2 configuration, so it doubles as a check that the
harness has not drifted — and that is how §7.2.7 was found.

**Three seeds, `n = 160` ruler, means with per-seed ranges:**

| arm | T_f | T_cl | T_s | T_c | `max α` | `L_void` | time |
|---|---|---|---|---|---|---|---|
| A 3000/300 | 0.1373 | 0.1890 | 0.0739 | 0.0742 | **1.0000** | **0.1505** | 597 s |
| | 0.1348–0.1392 | 0.1870–0.1908 | 0.0677–0.0786 | 0.0684–0.0786 | 1.0000–1.0000 | 0.1212–0.1670 | |
| B 1000/2300 | 0.1267 | 0.1784 | 0.0485 | 0.0499 | 0.9567 | 0.0853 | 792 s |
| | 0.1260–0.1278 | 0.1778–0.1792 | 0.0463–0.0515 | 0.0479–0.0527 | 0.9040–1.0000 | 0.0571–0.1208 | |
| C 300/3000 | **0.1247** | **0.1761** | **0.0434** | **0.0450** | 0.8702 | 0.0724 | 905 s |
| | 0.1222–0.1260 | 0.1732–0.1779 | 0.0379–0.0464 | 0.0396–0.0480 | **0.6274–0.9998** | 0.0495–0.1068 | |
| reference | — | — | — | — | 1.0000 | 0.3812 | — |

**The external result is confirmed for the mean, and the seed ranges do not
overlap.** Moving budget into the quasi-Newton stage improves every temperature —
`T_s` by **41%** and `T_c` by **39%** from A to C — and `T_f`, `T_cl`, `T_s` and
`T_c` all separate cleanly arm by arm. It costs 52% more wall-clock at equal
iteration count, because a quasi-Newton iteration carries a line search and an Adam
iteration does not.

**The front tells the opposite story, and it is a spread rather than a trend.**
`max α` is **1.0000 on all three seeds** in arm A and ranges **0.6274–0.9998** in
arm C. The mean falls monotonically, but a mean over a near-binary quantity is a
poor summary: what arm C actually does is make the front *unreliable*, not
uniformly worse.

§7.2.8 says why, and §7.2.8's own measurement is the number to hold on to: at arm A
the peak `T_c` clears saturation by **20.5 K**. The temperature scores are averages
and the front is the extremum `max T_c > T_boil`; a smoother, better-in-the-mean
fit gives up a few tens of kelvin at the peak, and a few tens of kelvin is the whole
margin. Arm C sits on the threshold and its front outcome is decided by the seed.

**Neither arm is adopted.** Arm A is what every published table was measured on, and
arm C trades the criterion M4 asks for against the criterion this project has been
optimising. §7.5.4 is the attempt to stop trading.

#### 7.5.4 Pairing a mean-winner with a peak-winner

§7.2.6 measured that the two representation remedies do not compose, and both were
mean-winners. §7.2.8 changes what "compose" should mean here: the mean and the
peak are close to independent, so the pairing worth trying is one remedy for each.

Fourier features are the only change measured to improve **both**, and §7.2.6's
three-seed table is the stronger evidence: `L_void` **0.2070 against a 0.1630
base** while also taking −11.1% off the mean. The modified MLP beats it on the mean
(−16.1%) and *halves* `L_void` to 0.0932. One remedy raises the peak, the other
lowers it — which is what reducing spectral bias should do to an extremum, and what
smoothing should do against it. So:

| arm | what it tests |
|---|---|
| C + `fourier_features=32` | the mean-winning budget with the peak-winning representation |
| C + `modified_mlp` | control: a stronger mean-winner that *costs* the peak |
| A + `fourier_features=32` | separates the Fourier effect from the budget effect |

**Three seeds, and it holds:**

| arm | T_f | T_s | T_c | `L_void` | `max α` | margin per seed |
|---|---|---|---|---|---|---|
| A (shipped default) | 0.1373 | 0.0739 `[.0677–.0786]` | 0.0742 | 0.1505 `[.1212–.1670]` | 1.0000 | +20.5, +17.2, +9.8 K |
| C (budget only) | 0.1247 | 0.0434 `[.0379–.0464]` | 0.0450 | 0.0724 `[.0495–.1068]` | 0.8702 | — |
| **C + fourier** | **0.1143** `[.1084–.1201]` | **0.0353** `[.0315–.0380]` | **0.0364** `[.0324–.0393]` | **0.2270** `[.2010–.2455]` | **1.0000** | +18.6, +13.5, +7.6 K |
| C + modified_mlp | 0.1248 `[.1244–.1251]` | 0.0447 `[.0438–.0460]` | 0.0462 | 0.0734 `[.0500–.0928]` | 0.9260 | **+0.6, +2.4, −0.0 K** |
| A + fourier | 0.1277 `[.1253–.1314]` | 0.0578 `[.0519–.0646]` | 0.0588 | 0.1883 `[.0865–.2681]` | 0.9425 | **+3.7, −0.4, +12.7 K** |
| reference | — | — | — | 0.3812 | 1.0000 | — |

**`C + fourier` is the best result this model has produced, on every metric at
once, and the seed ranges are disjoint from the default's on all four.** `T_s` and
`T_c` are **52% better**, `T_f` 17% better, and `L_void` **51% better** — 0.2270
against 0.1505, where the reference is 0.3812. `max α = 1.0000` on all three seeds,
with 7.6–18.6 K of margin.

**Both controls behaved as designed, and both are seed-fragile in exactly the way
the margin predicts.**

`C + modified_mlp` pairs the same mean-winning budget with a peak-*lowering*
remedy. The prediction was that the mean would stall and the margin collapse; it
did both, on all three seeds — temperatures no better than plain arm C, and margins
of **+0.6, +2.4 and −0.0 K**. The front sits exactly on the threshold every time,
which is about as clean a confirmation as a three-seed control can give.

`A + fourier` isolates the Fourier effect and is the sharpest illustration of
§7.2.8 in the document. Its margins are **+3.7, −0.4 and +12.7 K** and its `L_void`
tracks them precisely: **0.2681, 0.0865, 0.2103**. One seed in three lands below
saturation and the voided length collapses by 3×. That is not a defect in the
remedy — it is what §7.2.8 says a threshold crossing must look like when the peak
sits on it. The seed spread on `L_void` is **3.1×**, against 1.22× for
`C + fourier`.

**`C + fourier` is the only arm with margin to spare on every seed** — its worst is
+7.6 K where the two controls reach −0.0 and −0.4 K — and it is the only one whose
`L_void` range is disjoint from the default's. Both effects are real and they add:
Fourier widens the super-saturated region, the quasi-Newton budget sharpens the
mean, and **neither alone is stable**. `A + fourier` has the higher single-seed
`L_void` and a 3.1× seed spread; `C` alone has the mean and loses the front.

That is §7.2.8 predicting a three-arm comparison in advance — which arm keeps its
margin, which spends it, and which is decided by the seed. Nothing in this document
had produced a prediction before; everything else here was explanation after the
fact. It also settles §7.2.6 in retrospect: Fourier and the modified MLP "did not
compose" because one *raises* the peak and the other *lowers* it. They were
competing for the margin, not for the mean — invisible until the margin was
measured, which happened only because the front failure forced the diagnostic.

**Not adopted as the default**, for the reasons in §0.5: every published table in
this document was measured on the default, moving it invalidates all of them at
once, and it costs 52% more wall-clock. The re-measurement is compute, not
development, and it is the single highest-value thing left to do here.

#### 7.5.4b `lbfgs_iters` means the same thing in both backends

The two quasi-Newton loops are different constructs and the parity test asserts only
that the *number* matches. JAX runs `jax.lax.fori_loop(0, n, …)` — exactly `n`,
unconditionally. torch runs `torch.optim.LBFGS(max_iter=n)` with
`tolerance_grad = 1e-12` and `tolerance_change = 1e-14`, which runs **at most** `n`.
If torch stopped early the shared knob would not be a shared budget, and every
cross-backend number measured at that setting would carry an uncontrolled variable.

Measured:

| requested | torch `n_iter` | `func_evals` |
|---|---|---|
| 300 | **300** | 371 (1.24 per iteration) |
| 3000 | **3000** | 3496 (1.17 per iteration) |

**torch runs the full count at both budgets** — the tolerances are never reached on
this problem. The knob is honest, and §7.3.2, §7.5.8 and the combo study are clean.
The ~1.2 evaluations per iteration is the strong-Wolfe line search behaving
normally, so the wall-clocks in these tables reflect work rather than search
thrashing.

Pinned by `test_lbfgs_iters_means_the_same_thing_in_both_backends`, so a change in
torch's defaults surfaces as a failing test rather than as a quiet asymmetry inside
a parity table.

#### 7.5.5 How many epochs does it need? — the axis nobody varied

§5.3 closed this question in one sentence on two points and one seed, and nine
sections then ran at a single fixed budget. Reopened properly: a ladder at a fixed
10:1 Adam-to-quasi-Newton ratio, three seeds, both backends,
`uv run python tools/axial_study.py scaling`.

| budget | `T_s` torch | range | margin min | front | `T_s` jax | margin min | front |
|---|---|---|---|---|---|---|---|
| **3k/300** (published) | 0.0765 | .0697–.0808 | **+12.5 K** | **every seed** | 0.0863 | **+9.5 K** | **every seed** |
| 8k/500 (shipped) | **0.0405** | .0364–.0488 | −1.8 K | **lost** | 0.0497 | −2.4 K | **lost** |
| 16k/1000 | 0.0438 | .0410–.0468 | −1.8 K | **lost** | **0.0432** | −2.1 K | **lost** |

**Two answers, and they point opposite ways.**

*The mean improves and then stops.* 3k → 8k is a **47% gain on torch** and 42% on
JAX — the single largest budget effect measured here. 8k → 16k then **reverses on
torch** (0.0405 → 0.0438) and adds 13% on JAX. So the mean saturates somewhere
between 8k and 16k, and the exact point is backend-dependent.

*The front is destroyed and never recovers.* Both large budgets lose it on at least
one seed, on **both backends**, with negative worst-seed margins. **The published
3k/300 budget is the only rung in this ladder that forms a front on every seed —
and it has the worst mean of the three.**

**So "more epochs make it worse" is half right and the important half is not the
usual one.** Convergence does not degrade: the loss keeps falling and the
temperature scores improve until they plateau. What degrades is a *derived
threshold quantity* the loss never asked about. This is §7.2.8 measured on the one
axis nobody had varied, and it is the sharpest statement of Annex C's measure bug in
the document: **the objective is a mean over the domain, the front is a few percent
of that domain, so more optimisation converges more precisely to a minimiser whose
peak is wrong.**

It also settles §5.3's claim. *"Non-monotonic in budget means the optimiser wanders
between minima rather than converging slowly, so more iterations will not fix this"*
is wrong on both clauses: the mean improves 47% with more iterations, and the
non-monotonicity is not wandering but two quantities moving in opposite directions.

#### 7.5.6 Level-set collocation — the measure fix, and why it is not enough alone

The fix §7.5.5 and Annex C imply: sample collocation where the network's own `T_c`
is nearest saturation, so the objective stops under-weighting the few percent of the
domain the front occupies. Importance sampling by the front indicator rather than by
residual magnitude — which is what RAR does, and what cannot work here, since the
residual is small everywhere once the void is closed algebraically.

Run at **8k/500**, the budget that loses the front entirely (§7.5.5), so the question
is direct. Three seeds, both backends,
`uv run python tools/axial_study.py levelset`:

| 8k/500 arm | `T_s` | `L_void` | `max α` | worst margin | front |
|---|---|---|---|---|---|
| plain, torch | **0.0405** | 0.0463 | 0.713 | −1.8 K | **no seed** |
| +levelset, torch | 0.0541 | 0.0424 | 0.821 | −1.8 K | **no seed** |
| **+levelset+f128, torch** | 0.0483 | **0.1610** | **0.998** | **+3.3 K** | **every seed** |
| plain, jax | 0.0497 | 0.0410 | 0.692 | −2.4 K | **no seed** |
| +levelset, jax | 0.0598 | 0.0198 | 0.582 | −2.5 K | **no seed** |
| **+levelset+f128, jax** | 0.0498 | **0.1673** | **0.989** | **+2.4 K** | **every seed** |

**The measure fix alone fails, on both backends.** The margin stays negative, and
the mean gets *worse* — 0.0405 → 0.0541 on torch, 0.0497 → 0.0598 on JAX. Diverting
a quarter of the collocation to a peak the network cannot represent costs the bulk
and buys nothing.

**Measure plus capacity works, on both backends.** The margin turns positive,
`L_void` goes up 3.5×, `max α` reaches ~0.99. Neither ingredient is sufficient by
itself and both are necessary at this budget.

That is the two-constraint picture of §0.3 at its sharpest, and it is a stronger
statement than either study alone could make: **the measure decides where the loss
puts its weight; capacity decides whether there is anything there to find.** Point
the loss at an unrepresentable peak and you lose twice.

**What it does not show.** Even the working arm — `T_s` 0.0483, `L_void` 0.1610 — is
well behind the best-known route of §7.5.8 (300/3000 + f512: `T_s` 0.0216, `L_void`
0.3012). So front-aware sampling **validates the diagnosis without being the best
path**: rescuing a large Adam budget is worse than not spending one. `front_frac`
was fixed at 0.25 throughout and has never been swept
(`tools/axial_study.py frontfrac`), so the arm is a demonstration of the mechanism
rather than a tuned configuration.

#### 7.5.7 M4, scored for the first time

M4's acceptance is *onset within 0.5 s and one cell* — 0.5 s and 0.00625 at the
`n = 160` ruler. §6.5 established that both are measurable; nothing had ever
reported them. Three seeds, both backends,
`uv run python tools/axial_study.py default`:

| | `T_s` | `L_void` | `max α` | onset `t` error | onset `ζ` error |
|---|---|---|---|---|---|
| shipped default, torch | 0.0434 | 0.0367 | 0.685 | 2.50–4.00 s | — |
| shipped default, jax | 0.0497 | 0.0410 | 0.692 | 3.25–4.25 s | — |
| **C+fourier, torch** | **0.0353** | **0.2270** | **1.0000** | **0.50–0.75 s** | 0.013–0.025 |
| C+fourier, jax | 0.0386 | 0.2064 | 0.9987 | 1.00–1.50 s | 0.013–0.025 |
| **bar** | — | — | — | **≤ 0.5 s** | **≤ 0.00625** |

**M4 fails on every configuration.** `C+fourier` is **3–5× closer on onset time**,
with one run landing exactly at the bar; onset location misses by 2–4 cells
everywhere.

> **A metric artefact, caught and fixed.** The first version of this table showed
> the *shipped default* passing onset location — 0.00000 on one seed, better than
> the arm that forms a front. It is not a result. With `max α = 0.685` and
> `L_void = 0.037` against the reference's 0.381, the default's front is
> **vestigial**, and "the first point where `α > 0.01`" is well defined for a trace
> of void and lands anywhere. The metric was rewarding the absence of the thing it
> measures.
>
> `scoring.py` now reports onset as `NaN` below `max α = 0.9`, which is about 2 K
> of margin — the point at which there is a front rather than a trace of one. The
> dashes above are that guard firing. It is the same rule that already made onset
> `NaN` when the network never boils: **a position is only meaningful if the thing
> has a position.**

#### 7.5.8 Raising the margin deliberately — the first thing aimed at it

§7.5.4 found that Fourier features raise the saturation margin as a side effect.
Nothing had ever aimed at the margin. This does: `fourier_features` swept 32 → 64 →
128 on top of the quasi-Newton budget, three seeds, both backends,
`uv run python tools/axial_study.py margin`.

Success is `margin_K` at **every** seed, not the mean — a margin large on average
and negative once is the `A + fourier` failure of §7.5.4.

| | `T_s` torch | range | `T_s` jax | **jax/torch** | `L_void` torch | margin min torch |
|---|---|---|---|---|---|---|
| f32 | 0.0353 | .0315–.0380 | 0.0386 | 1.09× | 0.2270 | +7.6 K |
| f64 | 0.0340 | .0300–.0369 | 0.0375 | 1.10× | 0.2397 | +13.8 K |
| f128 | 0.0314 | .0285–.0344 | 0.0363 | 1.16× | 0.2424 | +17.5 K |
| f256 | 0.0282 | .0251–.0301 | 0.0358 | 1.27× | 0.2834 | +24.4 K |
| **f512** | **0.0216** | **.0148–.0253** | 0.0310 | **1.44×** | **0.3012** | **+34.6 K** |
| f1024 | 0.0234 | .0205–.0269 | 0.0304 | 1.30× | 0.3027 | +31.6 K |
| shipped default | 0.0434 | — | 0.0497 | 1.15× | 0.0367 | **−1.1 K** |
| reference | — | — | — | — | 0.3812 | — |

Three seeds per cell, both backends. Every arm holds a **positive margin on every
seed**; the shipped default holds a negative one on every seed.

**The ladder ends at f512.** `f1024` is not better: the mean is slightly worse on
torch (0.0234 against 0.0216), the worst seed is worse (0.0269 against 0.0253), and
`L_void` and the margin are flat. The ranges overlap heavily, so at three seeds the
two rungs are indistinguishable and doubling the parameters again buys nothing. It
cost 2.5 h per run to establish that, which is the price of a measured endpoint
rather than an assumed one.

So **capacity binds up to ~f512 and not beyond**, and the accelerating trend that
looked open-ended two rungs earlier did close — one rung later than it appeared it
might.

**Monotone in `T_s`, in `L_void` and in margin-minimum on both backends, positive
on every seed of every arm.** Targeting the margin raised it and improved the mean
at the same time — the only lever in this document that moves the average and the
extremum the same way, which is what §7.2.8 predicts of a change that relieves
spectral bias.

`f512` on torch reaches `T_s` **0.0216** and `L_void` **0.3012** — 79% of the
reference, against the shipped default's 10% — with a worst-seed margin of
**+34.6 K**, against the 20.5 K the whole "the front forms" claim rested on before
this study.

> **The JAX column here is superseded — §7.5.17.** It was measured at
> `memory_size=10`, and the headline below is the artefact: more parameters need
> more curvature pairs, so a fixed 10 degrades faster with capacity. The torch
> column stands.

**Both backends improve; torch improves faster, and the gap grows monotonically.**
Across f32 → f512, torch gains **39%** on `T_s` and JAX **20%**, so `jax/torch`
climbs **1.09× → 1.44×**. The architectures and residuals are identical — verified
to 1e-14 (§7.3.2) — so something in the JAX stack converts capacity into accuracy
less efficiently.

**Variance grows with capacity, and it is the more useful signal.** The `T_s` seed
range spans 1.20× at f256 and **1.71× at f512** (0.0148–0.0253). The best f512 seed
reaches 0.0148 — the 1% bar 1.5× away — and the worst 0.0253. So the honest metric
for this ladder is the **worst** seed, exactly as the margin already is: by that
measure f512 gives 0.0253 against f256's 0.0301, which is a real but much smaller
step than the means suggest.

> An earlier revision of this section reported "JAX has saturated at ~f128" from the
> f256 rung. JAX then gained a further 14% at f512. It was flat over **one** rung,
> which is not saturation — the sixth partial-ladder claim in this study to need
> revising within the hour. The rule now applied: report a ladder when its rungs are
> complete, means with ranges, and let the interpretation follow the table.

**The obvious suspect is the optimiser, and §7.3.2 already implicates it.** The
framework L-BFGS was the *entire* backend gap at the shipped configuration — 1.168
own against 0.999 shared. If `optax.lbfgs` cannot exploit added capacity while
`torch.optim.LBFGS` can, one arm settles it: `axial_study.py capacity-optimiser`
runs f512 under both optimisers on both backends. **TBD.**

That also puts §7.3.2's own number in its place. "The framework L-BFGS is the whole
gap" was measured at one configuration, and the gap runs 1.09× to 2.27× across this
ladder — so it described the point it was taken at, not the backends. Third
configuration-bound conclusion in this document, after D67 and §5.3.

> **A two-seed claim, retracted.** An earlier revision reported JAX as
> *non-monotone* — "f32 +16.9 K, f64 +4.5 K, f128 +11.6 K" — and called the effect
> torch-specific. That was two seeds. At three, JAX is monotone in both `T_s` and
> the margin minimum. **Fifth two-seed claim overturned at three in this study**,
> after the budget sweep's monotone front, the backend gap's disappearance,
> `ssbfgs`'s 0.1% variance, and the vestigial-front onset artefact of §7.5.7.

#### 7.5.9–7.5.11 The epoch surface, and the two sweeps that were never run

`frontfrac` and `capacity-optimiser` had been committed with their designs fixed
and never executed — the weak form of D67, a study that exists only as an intention.
Both are now measured.

**§7.5.9, `frontfrac` — diverting collocation to the front does not help.** Five
splits, three seeds, at the 8k/500 budget §7.5.6 used, with today's f256 default:

| `front_frac` | 0.0 | 0.05 | 0.10 | 0.25 | 0.50 |
|---|---|---|---|---|---|
| `T_s` [jax] | **0.0400** | 0.0407 | 0.0448 | 0.0497 | 0.0518 |
| worst margin | +3.6 K | +4.4 K | +2.9 K | +3.2 K | +2.0 K |

`T_s` degrades **monotonically** with front sampling and the margin is flat inside
its noise. 25% — the value §7.5.6 used, chosen as plausible rather than measured —
is 1.24× worse on the mean than not doing it at all.

This sharpens Annex C rather than refuting it. The measure is a real constraint, but
**re-weighting the measure is not the remedy**: moving points onto the front does not
make the front easier, it starves the bulk, and at f256 the front forms anyway.
§7.5.14 is the same diagnosis treated correctly — supply the basis the front needs
instead of re-weighting the points it is scored on.

> **Not comparable to §7.5.6's arms.** Those ran with `fourier_features = 0` — the
> third arm adds f128 explicitly, and their `L_void` of 0.046 is the no-Fourier
> signature. This study inherits today's f256 default. An earlier reading of the
> control as "the front now forms where §7.5.6 lost it, so the memory fix rescued
> it" was wrong: the difference is capacity, not memory.

**§7.5.10, `capacity-optimiser` — answered, and the answer is §7.5.17.** f512,
three seeds:

| | `T_s` | `L_void` | worst margin |
|---|---|---|---|
| jax, own `optax.lbfgs` | 0.0233 | 0.2978 | +35.7 K |
| jax, shared implementation | 0.0226 | 0.2982 | +35.6 K |

**3% apart.** At `memory_size = 10` the same two arms read 0.0336 and 0.0265. So
this section's premise — that JAX converts capacity into accuracy worse because of
`optax.lbfgs` — was measuring one unset argument. Once the memory matches, the two
implementations agree, and **JAX f512 against torch f512 is 1.08×** rather than the
1.44× this document published.

> **The 27 JAX runs of the grid below are superseded — §7.5.17** (`memory_size=10`).
> The torch half stands, and the three conclusions hold on torch alone.

**§7.5.11, the grid — complete.** Adam ∈ {30, 300, 3000} × quasi-Newton ∈ {30, 300,
3000} at f128, **all 54 runs: nine cells, three seeds, both backends**. Mean `T_s`,
worst-seed margin:

| | qn30 | qn300 | qn3000 |
|---|---|---|---|
| **adam30** [torch] | 0.2314 · **no front** | 0.0657 · +5.1 K | **0.0303** · +19.0 K |
| **adam300** [torch] | 0.1906 · **no front** | 0.0603 · +6.3 K | 0.0314 · +17.5 K |
| **adam3000** [torch] | 0.0707 · +0.7 K | 0.0455 · +3.6 K | 0.0305 · +14.4 K |
| **adam30** [jax] | 0.2336 · **no front** | 0.0960 · +3.4 K | **0.0350** · +9.8 K |
| **adam300** [jax] | 0.1935 · **no front** | 0.0880 · +2.6 K | 0.0363 · +9.5 K |
| **adam3000** [jax] | 0.0762 · +2.4 K | 0.0545 · +3.7 K | 0.0358 · +5.3 K |

> **Read with §7.5.38.** This surface is measured at `n_colloc = 4000` — 6000 points once
> the early-time cluster is counted — which is **1.41 residuals per parameter** against the
> 17 029-parameter body (§7.5.37a), so the system is mildly *over*determined. A quasi-Newton
> iteration is full-batch and therefore linear in the point count, so the affordability of
> a 3000- or 30000-iteration quasi-Newton stage is itself a consequence of that count. The
> conclusion below holds at this count; it is not a general claim about first- against
> second-order methods. §7.5.38 measures the axis directly: flat above 1.41, and a cliff
> below the determined point.
> (An earlier version of this note said 0.48 and "2.07× underdetermined", counting the
> encoder as capacity — see §7.5.31a's retraction.)

At three seeds on both backends:

- **The quasi-Newton axis is monotone across two decades**, in `T_s` and in margin,
  in every row. No interior optimum in the range swept.
- **Once `qn3000` is set the Adam axis is flat** — 0.0303 / 0.0314 / 0.0305 on
  torch, 0.0350 / 0.0363 / 0.0358 on JAX. The differences are inside a single
  cell's seed spread. **The axis the shipped default was tuned on is the axis that
  matters least**, and `adam30` is not distinguishable from `adam300` at a tenth of
  the cost.
- **A starved quasi-Newton stage cannot be bought back with Adam.** `qn30` forms no
  front at `adam30` or `adam300` on either backend. At `adam3000` it forms one with
  **+0.7 K** of worst-seed margin — 0.7 K out of a 590 K range — for `T_s` 0.0707
  against `adam30/qn3000`'s 0.0303 at twice the wall-clock.

This replaces the `budget` study, which measured three points on a *diagonal* and
ranked them on a mean it predates `front_metrics` and could not use to see the
front. **The default's Adam budget is not doing measurable work**; the quasi-Newton
budget is, which raises the priority of the optimiser bake-off (§8, item 1) rather
than answering it.

> **Superseded, and left here only as a marker.** A one-seed version of §7.5.10 stood
> at this point, reporting `jax` f512 at 0.0336 against torch's 0.0153 and concluding
> that JAX-with-shared "still sits 1.73× from torch at f512", so §7.3.2's parity result
> "does not survive the move to f512". Its closing line was "one seed; the remaining two
> decide it". They did, above: at three seeds and matched curvature memory the two arms
> are **3% apart** and the backend ratio at f512 is **1.08×**. The 1.73× was
> `memory_size = 10` (§7.5.17) read as a framework difference, and §7.3.2 needs no
> rescuing. Deleting the block outright would erase a claim this document made; keeping
> it live would leave two contradictory §7.5.10s in one section.

#### 7.5.12–7.5.14 Three embeddings, at three seeds on both backends

All three moved exactly one knob from the shipped default with a control arm that
reproduces it. The **four control arms of §7.5.12–§7.5.14 and §7.5.16 agree to every
digit** on JAX — `T_s` 0.0292 [0.0286–0.0297], `L_void` 0.2440, margin +25.8 K, from
four separately launched processes — so the runs are deterministic at a fixed thread
budget and every new knob is inert when off, end to end rather than only in a unit
test.

JAX arms are three seeds at `lbfgs_history = 50` (§7.5.17); torch arms are two seeds
and were never affected by that defect.

##### 7.5.14 Multi-scale Fourier bands — the front, essentially solved

| bands | `T_s` torch | `T_s` jax | `L_void` torch | `L_void` jax | margin torch | margin jax |
|---|---|---|---|---|---|---|
| single (control) | 0.0278 | 0.0292 | 0.2834 | 0.2440 | +24.4 K | +25.8 K |
| (1, 4) | **0.0206** | 0.0225 | 0.2705 | 0.2536 | **+15.3 K** | **+16.6 K** |
| (1, 4, 16) | 0.0270 | 0.0233 | 0.3334 | **0.3793** | **+51.3 K** | +28.1 K |
| (0.25, 1, 4, 16) | 0.0226 | **0.0203** | **0.3521** | 0.2932 | +41.4 K | +24.4 K |

**`L_void = 0.3793` against the reference's 0.3812 is 99.5%** — the voided length is
essentially exact, at three seeds, at the shipped feature count and wall-clock,
while `T_s` and the margin also beat the control. The published best before this was
79% of the reference (§7.5.8), and the shipped default is 64%.

> ### This is a statement about `qn3000`, and it does not survive the funded budget
>
> Every arm in this section ran at the old `qn3000`. §7.5.24's 2×2 measured the same
> embedding at `qn30000` and it **reverses on every column**: `T_s` 5.6× worse, voided
> length 92.4% against a single band's 99.3%, margin +31.3 K against +67.6 K, three
> seeds, non-overlapping.
>
> So the mechanism described below is real but it is *compensating for an
> under-converged optimiser*. A multi-band basis is an implicit preconditioner
> (SAFE-NET, arXiv:2502.07209); once the quasi-Newton stage is funded it accumulates
> that curvature itself, the preconditioning is redundant, and splitting 256 features
> across three bands is left as pure capacity loss. **The budget alone reaches 99.3%**,
> which is the result this section claimed for the bands.
>
> `fourier_bands` is off permanently. The paragraph below about "the first controllable
> dial between the mean and the extremum" stands only at a starved budget, where the
> dial exists because the optimiser has not done its job yet.

**The mechanism is legible, and both backends agree on it.** Two bands win the
*mean* and lose the *front* — (1,4) has the lowest margin in the set on both
backends. Adding the 16× band reverses that: `L_void` and the margin jump, `T_s`
gives some back. The high band resolves the near-discontinuity, the low bands keep
the bulk smooth, and at a fixed feature total the split between them is a choice.

That is what makes this different from every other lever in this document. §7.2.8
and Annex C say the loss is a **mean** and the front is an **extremum**, and that the
two move independently; every previous remedy hit one or the other by accident.
**This is the first controllable dial between them.**

##### 7.5.12 Anisotropic bandwidth — the same idea, cruder

| `zeta_scale` | `T_s` torch | `T_s` jax | margin torch | margin jax |
|---|---|---|---|---|
| 1.0 (control) | 0.0278 | 0.0292 | +24.4 K | +25.8 K |
| 2.0 | 0.0290 | 0.0294 | +22.5 K | +21.6 K |
| 4.0 | 0.0269 | 0.0271 | +18.4 K | +14.7 K |
| 8.0 | **0.0194** | **0.0244** | +23.9 K | +26.5 K |

Real and consistent — 1.43× on torch, 1.20× on JAX at `zeta_scale = 8`, with the
front neutral. But it **shifts** the whole spatial band rather than **adding** one,
so it buys front resolution by giving up bulk resolution. §7.5.14 dominates it: same
physics, better instrument. The earlier reading of the first three torch arms as
"monotone degradation, likely a negative result" was three points and is retracted.

##### 7.5.13 The level-set coordinate — inert, and now explicable

`T_s` 0.0292 → 0.0284 on JAX (three seeds), 0.0278 → 0.0257 on torch, at **1.95×
the cost**. It cannot help, and the reason is structural rather than empirical:
`φ = (T_c − T_sat − ΔT_sup)/ΔT` is a monotone function of `T_c`, which the network
already computes. Feeding it back is a **re-parameterisation, not information**. The
front sits at `φ = 0` by construction — but placing it still requires getting `T_c`
right, which is the original problem.

**Tenth remedy argued soundly and refuted by measurement**, and the most expensive.

#### 7.5.15 A defect the level-set arm exposed: JAX arms were scored under the defaults

`tools/axial_study.py`'s `train_jax` discarded the config it had just trained under
and called `pj.predict(model, p, zeta, t)` with none. `predict` falls back to
`AxialTrainConfig()` when given no config, so **every JAX arm was trained under its
arm's settings and scored under the shipped defaults**.

The torch path cannot do this. Its model carries its own `cfg`, so the evaluator is
structurally unable to disagree with the training. The JAX twin is functional and
must be *handed* the config, which makes forgetting it a silent operation rather
than an impossible one — a real asymmetry between the two backends, and one worth
recording rather than quietly patching.

It surfaced as a **crash** only because `level_set_input` changes an array *shape*:
the model was built with three inputs and the evaluator fed it two. `predict` also
reads `t_train_frac` through `horizon()`, and that changes a *value* — an arm
sweeping it would have returned a plausible, wrong score in complete silence. That
is D67's failure mode exactly, in a different place.

**Scope, checked rather than assumed.** `predict` reads `t_train_frac`,
`level_set_input`, `front_net` and `void_closure`. No completed JAX study varied any
of them from the defaults, so **no published JAX number moves**. The one arm that
did vary one is the arm that crashed. Fixed, with a test that asserts the *raise* —
and says why, so nobody later loosens it to a tolerance.

#### 7.5.16 Onset, put in the objective and read by tangency

M4 asks for onset within 0.5 s **and** one cell, and it has never moved. Two
reasons, and only one is the network's.

**It was never in the objective.** Onset was read off a trained field afterwards.
Every knob this document sweeps was ranked on a mean (relative `L2`) and later on an
extremum (the saturation margin). **Neither is a position.** Nothing has ever
optimised for where the front starts.

**And the readout was square-root conditioned.** Onset happens at the *maximum* of
`T_c`, so near it `T_c ≈ T_boil − κ(ζ−ζ*)²/2`, and recovering a position from a
*value* error there goes as `√(2ε/κ)` — the worst possible law for small errors.
Measured against this model's reference:

| | |
|---|---|
| onset location | `ζ = 0.9875` |
| `∂T_c/∂ζ` there | 64.8 K per unit `ζ` — **11.6× flatter** than the profile's steepest point |
| `∂²T_c/∂ζ²` there | 1066 K per unit `ζ` squared |
| one cell | **0.405 K** of `T_c` |
| `ε` at `T_s = 0.0216` | 3.4 K → `√(2ε/κ)` = **13 cells** |

The front forms near the channel top, where the cosine power shape has run out and
`T_c` is nearly flat — the worst place to locate a level set.

**The fix is the tangency pair.** Onset is the first instant the field *touches*
saturation, so at that instant the peak *is* the contact point:

```math
T_c(\zeta^*, t^*) = T_{\mathrm{sat}} + \Delta T_{\mathrm{sup}},
\qquad
\frac{\partial T_c}{\partial \zeta}(\zeta^*, t^*) = 0
```

Solving for the height then costs `δζ ~ δ(slope)/κ` — **linear**, and divided by a
curvature that is *large* exactly where the gradient is small. A slope error of
3.8 K per unit `ζ` moves the answer 0.6 cells.

It also explains an asymmetry the measurements showed and nothing accounted for.
**Onset time already passes**: `δt ~ ε/|∂T_c/∂t| = 3.4/43.0 = 0.079 s` against a
0.5 s criterion, because `T_c` rises steeply in **time** while being flat in
**space**. One coordinate was always well posed; only the other was read by
thresholding.

Two pieces, both landed in both backends:

- **`onset_by_tangency`** in the shared scorer, reported **alongside** the threshold
  readout rather than replacing it. Every published onset number was measured the
  old way, and a metric that changes definition silently makes its own history
  unreadable — the comparison is itself the measurement.
- **`onset_head`**, two trainable scalars through a sigmoid so `(ζ*, t*)` stay in the
  domain by construction, with the two residuals as a loss block. A parameter rather
  than a network, because onset is two numbers at fixed parameters; a network is only
  needed to make onset a function of `void_worth_net`/`tau_pump` for the M9 sweep.

Tests assert the *properties*: equal block counts and equal initialisation across
backends, that the gradient reaches the **field network** and not only the head
(otherwise the head chases a field it cannot influence and onset is still a
read-off), that the readout finds a parabola vertex placed deliberately between grid
points, and that a field never reaching saturation returns `nan` rather than zero
error.

> ## RETRACTED — the tangency readout does not locate anything
>
> **`T_c` is monotone increasing in `ζ`.** Coolant heats as it rises, so the maximum
> of `T_c` is *always the last node*, and the tangency condition `∂T_c/∂ζ = 0` has
> **no interior solution**. The derivation above assumed onset sits at an interior
> maximum where the profile turns over. It does not.
>
> So `onset_by_tangency` returns the outlet **by construction**. The `m4_bar` ladder
> makes it unmissable: its onset height is *exactly* the last cell centre at every
> mesh — 0.98750 = 1 − 1/80, 0.99687 = 1 − 1/320, 0.99961 = 1 − 1/2560 — converging
> to `ζ = 1` rather than to a physical location.
>
> **Therefore "0.00 cells on every seed of every arm" measured nothing.** Both the
> network and the reference put the peak at the last cell because both fields are
> monotone. Zero error there is a tautology, and it was written up as the headline
> result of this section across §0.6, §7.9 and the README. Twelfth retraction, and
> the most confidently stated.

**What survives.** The three-seed arms themselves stand; it is the *interpretation*
of the height column that does not.

| arm | `T_s` | `L_void` | worst margin | height err, threshold |
|---|---|---|---|---|
| head off | 0.0292 | 0.2440 | +25.8 K | 2.67 cells |
| head on | 0.0346 | 0.2283 | +13.2 K | 2.67 cells |

**Onset *time* is the quantity that carries information**, and it is worse than this
document reported. Per seed, head off: threshold 0.50 / 0.25 / 0.50 s against
tangency **0.84 / 0.68 / 0.62 s**. The threshold number was flattered by **grid
quantisation** — the scoring grid is 0.25 s, so that metric can only report multiples
of 0.25. The tangency *time* is still meaningful even though the tangency *height*
is not: it interpolates when the peak reaches saturation, which is a real crossing,
and it says 0.62–0.84 s against a 0.5 s criterion.

Since the height answer is "the top of the channel" whatever the network does, **M4
turns entirely on the time**, and at *this* budget the time fails.

> ### SUPERSEDED — at the funded budget the time criterion is met
>
> The 0.62–0.84 s above was measured at `qn3000`, where `T_s ≈ 0.029`. At the shipped
> default (the `adam30/qn30000` rung, `T_s = 0.0017`) it is **0.0006–0.0181 s**. The
> paragraph stands as a statement about the arm it was run on; it is not a statement
> about the model. See §7.5.16a.

#### 7.5.16a Onset time at the funded budget — three seeds, and the criterion is met

`tools/onset_conditioning.py`, JAX, three seeds, at the shipped default
(`adam_iters = 30`, `lbfgs_iters = 30000`, `t_train_frac = 0.275`, f256 — recorded in
each row rather than inferred from the config). Onset is located by **root-finding on
the network's own dense output**, bracketing the crossing and refining with `brentq`,
which is how `scipy.integrate.solve_ivp` locates events. That removes the grid
quantisation the paragraph above complains about, from both sides of the comparison.

**The reference onset is 10.9784 s, not 10.75 s.** 10.75 is a point on the 0.25 s
output grid; the crossing is 0.23 s later. Every onset error quoted before this
section was measured against a grid point, and roughly a quarter of a second of the
"miss" was the ruler's own quantisation.

| seed | `t*` network | `\|Δt\|` | outlet `L∞`, ±2 s window | `L2` on `T_c` |
|---|---|---|---|---|
| 0 | 10.9848 s | 0.0064 s | 1.511 K | 0.001648 |
| 1 | 10.9789 s | 0.0006 s | 1.502 K | 0.001692 |
| 2 | 10.9965 s | 0.0181 s | 1.591 K | 0.001777 |

**Worst seed 0.0181 s against a 0.5 s criterion, on three seeds.** M4's *time* half is
met, and it is met by the budget alone — no onset residual, no tangency head, no
sampling change. The same axis that took `T_s` from 0.0258 to 0.0017 took the onset
error with it.

**Two things are stated here rather than in a caveat below.**

The **seed spread is 32×**, 0.0006 to 0.0181 s. The criterion holds on every seed, so
the verdict is safe — but this is the widest spread of any quantity in this document,
and no ranking of any arm may be built on it at fewer than three seeds.

And **the result sits at a test uncertainty ratio of 2.0 against the ruler**
(§7.5.22). The reference's own onset is uncertain by 0.009 s at the scoring mesh
(§7.5.21), so the worst seed is twice that and the best seed is *below* it. The bar
itself is sound at a ratio of 56. But "28× inside the criterion" is not a statement
this reference can support, and it is not made: **met** is what the measurement
carries, and how far inside is now the ruler's question, exactly as for the
temperature bar at 1.06.

**The conditioning hypothesis this tool was built to test is refuted, in the useful
direction.** Annex E.6 asked whether onset time is an *amplifier* of field error,
`δt ≈ ‖δT‖∞ / |Ṫ_out(t*)|`. Measured: `Ṫ_out = +25.57` K/s at the crossing, an
amplification of 0.0391 s/K, so the outlet `L∞` of ~1.5 K predicts `δt ≈ 0.059` s.
The measured errors are **0.11 / 0.01 / 0.29 of that**. The bound is loose by roughly
an order of magnitude, not violated: `L∞` over a ±2 s window bounds the error
*somewhere* in the window, and the error *at the crossing* is much smaller. So onset
is not amplifying field accuracy — it is converting it at better than the first-order
rate, and buying timing through field accuracy is efficient rather than wasteful.

**What this leaves of M4.** The time half is met. The height half remains what the
retraction above says it is: `T_c` is monotone in `ζ`, so both the network and the
reference put the peak at the last node and any height error is a tautology. M4 as
written cannot be failed on height and is now passed on time, which means **M4 no
longer discriminates between formulations** and should be replaced rather than
chased. §8 is written on the assumption that onset is the open problem; it is not.

**The head is confirmed harmful at three seeds** — `T_s` +18%, worst margin −49%, and
tangency `t_err` **2.4× worse** (0.71 s → 1.81 s mean). That last number is the
diagnosis confirming itself: `t*` is the coordinate the head parks in the wrong
place, and the time error is exactly what degrades.

The degeneracy is structural. **The two conditions do not pin the point to the
field; they pin the field to the point** — bending `T_c` until it is tangent to
`T_boil` at the wrong place is cheaper than moving `(ζ*, t*)` to the right one. And
they carry no new information, being a consequence of the PDE rather than additional
physics, so as residuals they can only distort. `onset_head` stays off and no further
work is planned on it.

Isolated, and here that matters more than usual: this is the **fourth** distinct use
of the level set in this model, after §7.5.6's sampling measure, the front network's
interface parameterisation, and §7.5.13's input coordinate. Overlapping mechanisms
are how one collects another's credit.

#### 7.5.17 The cross-backend gap was one unset argument

Every accuracy comparison between the two backends in this document is wrong, in
one direction, for one reason. `jaxpinn/training.py` called

```python
opt = optax.lbfgs()
```

bare. `optax.lbfgs`'s default `memory_size` is **10**. The torch twin passed
`history_size=50`, and so did both shared-implementation paths. Three of the four
quasi-Newton paths kept 50 curvature pairs; **the JAX default path — the one every
published JAX number was measured with — kept 10.**

**The completed grid says the gap lives in that stage and nowhere else.** The
`jax/torch` ratio on `T_s`, from the 54-run surface, is a pure function of the
quasi-Newton axis and independent of Adam:

| | qn30 | qn300 | qn3000 |
|---|---|---|---|
| adam30 | **1.01×** | 1.46× | 1.15× |
| adam300 | **1.01×** | 1.46× | 1.16× |
| adam3000 | 1.08× | 1.20× | 1.18× |

At `qn30` the backends agree to 1%. The ansatz, the residuals, the initialisation,
Adam, the sampling and the causal weighting are identical in every cell of that
table, so they are all exonerated: the divergence appears only when L-BFGS does
work. And the shape is what a memory difference predicts — zero when neither
optimiser has filled ten pairs, maximal when torch has fifty and JAX has ten,
partially recovering when extra iterations compensate for a worse Hessian model.

**Isolated, the causation is direct.** Torch weights were copied into the JAX
module, the same collocation points used, and the objective checked first: `torch =
3.3506486090e+00`, `jax = 3.3506486090e+00`, **relative difference exactly 0** at
the small size and 1.2e-16 — one ulp — at f512. Only the optimiser then varied:

| optimiser | it100 | it300 | it1000 |
|---|---|---|---|
| torch `history=50` | 7.18e-03 | 1.50e-03 | 4.22e-04 |
| **torch `history=10`** | 8.08e-03 | 2.57e-03 (1.71×) | 7.98e-04 (1.89×) |
| **optax `memory=10`** (shipped) | 8.24e-03 | 2.50e-03 (1.66×) | 7.74e-04 (1.83×) |
| **optax `memory=50`** | 6.22e-03 | 1.48e-03 (0.98×) | 4.32e-04 (1.02×) |

**Torch at memory 10 becomes JAX** (within 3%) and **optax at memory 50 becomes
torch** (within 2%). The two L-BFGS implementations are equivalent at equal memory.
It is not the zoom line search, not `scale_init_precond`, not the framework.

##### 7.5.17a The ladder, and why its first reading was wrong

Swept further at f512 the memory kept paying — at **equal iterations**:

| memory | loss @600 it | vs mem50 | ms/it | curvature MB |
|---|---|---|---|---|
| 50 | 6.4257e-04 | 1.000× | 195.3 | 67 |
| 100 | 5.4730e-04 | 0.852× | 225.0 | 134 |
| 200 | 4.7590e-04 | 0.741× | 282.1 | 267 |
| 300 | 4.2898e-04 | **0.668×** | 339.6 | 401 |

Monotone, no turning point, mem300 half again as good as mem50. On that table the
obvious conclusion is "push higher".

**It is the wrong axis.** More memory costs time — mem300 runs at 1.74× the ms/it
of mem50 — so the question a default turns on is which is better for a fixed
*compute* budget. Re-run at **equal wall-clock**, 200 s per arm, each running as
many iterations as it fits:

| memory | loss @50 s | @100 s | @200 s | iterations done |
|---|---|---|---|---|
| 50 | **1.4778e-03** | 7.5809e-04 | 3.9447e-04 | 1100 |
| 100 | 1.4964e-03 | **7.1074e-04** | **3.5687e-04 (0.905×)** | 950 |
| 200 | 2.2655e-03 | 8.2093e-04 | 3.9030e-04 (0.989×) | 750 |
| 300 | 3.0198e-03 | 1.0229e-03 | 4.2898e-04 (**1.087×**) | 625 |

**The ranking reverses.** mem300 goes from 1.50× *better* to 1.087× *worse*. The
ladder turns at 100, and at short budgets the shipped 50 beats everything — a large
memory starts hundreds of iterations behind and has to earn that back.

The crossover also moves with the budget: mem200 runs 1.533× → 1.083× → 0.989×
across the three marks, still improving. The shipped recipe spends **3000**
quasi-Newton iterations, roughly 550 s, which is 2.75× the largest budget measured
here — so where the optimum sits at the budget this model actually uses is **not
measured**, and extrapolating the trend would repeat the error this subsection
exists to record.

##### 7.5.17b What was changed, and what it costs

`lbfgs_history` is now an explicit knob in both configs, threaded into all four
quasi-Newton paths, with a `--lbfgs-history` flag on `axial_study.py` so a sweep is
a command rather than an edited default.

**It stays at 50**, deliberately, though 100 is the best measured value at 200 s.
Fifty is what torch has always used, so the fix lands JAX on the *published* torch
behaviour instead of moving both backends somewhere new — and the ~9.5% that 100
buys is small next to the 1.8× that 10 was costing.

Timing, measured back to back at constant contention: JAX pays **11–13%** for
50 over 10. Torch gets **5–9% faster**, which is not a paradox — the `O(m·n)`
two-loop recursion is negligible beside a residual evaluation that differentiates
the network, so a better search direction wins by making the strong-Wolfe line
search accept `α = 1` more often, and that saves whole function evaluations.

##### 7.5.17c What is now superseded

**Every JAX accuracy number in this document was measured at memory 10.** The torch
numbers are untouched — torch always passed 50. Specifically:

| | status |
|---|---|
| §7.3.2, the parity claim | **superseded.** 1.168 with each framework's own L-BFGS was measuring 50 against 10, not two implementations |
| §7.5.8, the capacity ladder's JAX column | **superseded**, including "the gap grows with capacity" — more parameters need more curvature pairs, so a fixed 10 degrades faster |
| §7.5.10, `capacity-optimiser` | **partly explained.** `lbfgs-shared` improved JAX 1.27× because that path already passed 50; part of what looked like a better algorithm was more memory |
| §7.5.11, the grid's JAX half | **superseded** — 27 of its 54 runs |
| §7.5.12–§7.5.16, the JAX arms | **superseded**; being re-run |

The re-runs use `--only "[jax]" --lbfgs-history 50`, into separate files, so each
study keeps a **paired** memory-10/memory-50 comparison at the same arms and seeds.
That pairing measures at training scale what the isolated bake-off could only
measure on the loss.

#### 7.5.18 The Laplace embedding — measured, and it fails

A Fourier basis is oscillatory and this transient is built out of **decay**:
coast-down at `1/τ_pump = 0.2` s⁻¹ and six precursor groups spanning 0.0124 to
3.01 s⁻¹, a 243× range. Approximating `exp(-0.2t)` over a 60 s window out of sines
costs many terms and still misses the tail; `exp(-s_k t)` does it in one. The split
is not arbitrary either — the oscillatory structure is in `ζ` and the exponential
structure is in `t`, which is §7.5.12's anisotropy reached from the physics rather
than from a sweep.

Rates taken straight from the manual, three combination modes, three seeds, both
backends:

| mode | `T_s` jax | `T_s` torch | `L_void` jax | margin jax | margin torch |
|---|---|---|---|---|---|
| **off** (control) | **0.0292** | **0.0283** | **0.2440** | **+25.8 K** | **+24.4 K** |
| `sum` — concatenated blocks | 0.0303 | 0.0290 | 0.2381 | +23.7 K | +21.6 K |
| `product` — damped sinusoids | 0.0357 | 0.0353 | 0.2111 | +9.6 K | +9.9 K |
| `alone` — rates only, no Fourier | 0.0652 | 0.0683 | 0.1438 | +3.2 K | +4.7 K |

**Nothing beats the control, on either backend.** `sum` is inside the seed noise at
2–4% worse; `product` costs 22% on the mean and more than halves the margin;
`alone` is 2.3× worse, which is what dropping the spatial resolution the front needs
should cost. Two independent implementations agreeing at three seeds each is as
decisive as this document gets.

**The study predicted its own result, and that is the part worth keeping.** Its
docstring recorded the prior before the arms ran: the ansatz is already
multiplicative, `θ = θ₀ exp(t̂ N)`, so a decaying mode was *always* representable and
the embedding could only make it **easier**, not newly possible. It does not.

That places it exactly in §8's dead category — a re-parameterisation that adds no
information — alongside §7.5.13's level-set coordinate, which failed for the same
reason and was also a monotone function of something the network already computes.
**Thirteenth remedy argued soundly and refuted by measurement.**

One design risk is retired rather than mitigated: `exp(-s t̂)` was expected to
underflow at the fastest precursor rate. It reaches `2.7e-22` at the end of the
trained window — small, and exactly representable in float64 — so no clipping is
needed. What did matter is that the rates enter scaled by the **trained** horizon
rather than `t_end`, since `t_train_frac` shortens the window and a wrongly scaled
rate would decay the basis far too fast while still training happily. That is
asserted against the physics rather than against itself.

#### 7.5.19 The speed comparison, made properly for the first time

Every torch-vs-JAX wall-clock before this one ran torch at `OMP_NUM_THREADS=8`
against JAX using **~230 threads** — XLA's CPU backend sizes its own Eigen pool
from `hardware_concurrency()` and ignores the OpenMP variable torch obeys. That is
a thread-count comparison wearing a backend comparison's clothes, and `AGENTS.md`'s
"a wall-clock needs a stated thread budget" was being satisfied on paper while being
violated in fact.

Both backends given all 48 cores, run **sequentially** on an idle machine, identical
weights, identical points, objective verified identical (`2.3148662743e-01`, 1.2e-16
apart), f512 with 6000 points:

| | Adam | quasi-Newton | total |
|---|---|---|---|
| torch | 146.20 s (731.0 ms/it) | 99.03 s (990.3 ms/it) | 245.23 s |
| **jax** | 34.64 s (173.2 ms/it) | 20.80 s (208.0 ms/it) | **55.44 s** |
| jax/torch | **0.24×** | **0.21×** | **0.23×** |

**Against an eager torch loop JAX is 4.4× faster, and the published 2.4× understates
it** — on the quasi-Newton phase alone, the phase §7.5.11 shows does all the work, it
is **4.8×**. Read "eager" as load-bearing: the compiled loop reverses this, below.

**But the thread asymmetry did not invalidate the old ratios**, and an earlier
revision of this document claimed it did. At 8 threads the same benchmark gives
2270 vs 530 ms/it — ratio **0.23**. At 48 threads, 990 vs 208 — ratio **0.21**. The
two backends scale almost identically with thread count (torch 2.3×, JAX 2.5× from
8 to 48 threads), so the ratio is robust. What was wrong was the absolute numbers
and the claim of a pinned budget, not the comparison.

**And JAX's answers depend on the thread count**, which no timing caveat covers.
`OMP_NUM_THREADS` binds torch and is ignored by XLA's CPU backend, so a JAX arm
nominally at 8 threads was measured creating **291**. Thread count changes float
reduction order, so this is a *correctness* issue and not only a timing one:

| affinity | threads | `sum T_c` |
|---|---|---|
| 48 cores | 291 | 31802.5076120401**35** |
| 8 cores | 56 | 31802.5076120401**57** |

About 3 ulp — numerically harmless, and fatal to §4's "reproduces run to run to four
digits", which was only ever true of torch.

**Affinity is what JAX obeys, and the core *count* is what matters.** Pinning to 8
cores reproduces bitwise on repeat, and a *different* block of 8 gives the identical
answer — so concurrent studies can take different blocks and stay comparable.
`axial_study.py --cpu-block K` does this, every row now records the affinity
alongside `OMP_NUM_THREADS`, and a row without it cannot be compared on wall-clock.

**Both scale badly, which is a planning fact.** Six times the threads buys 2.3–2.5×.
These networks are small and the step is `jvp`-bound, so past roughly 8 threads most
of the machine idles — running six studies at 8 threads each is closer to optimal
than one at 48.

##### And the speed half has now reversed again, for the same class of reason

Every number above compares against **eager** PyTorch. That was the only PyTorch
there was: the compiled path broke into eight graphs and bought 1.06×, which §7.3.2
mistook for a property of `torch.compile` rather than of this repository's code. With
those defects fixed and `fullgraph=True` enforcing the whole graph, at 500 collocation
points and f256 on 8 pinned cores — best of three repeats within a run, four runs:

| | ms/iteration | it/s | spread within a run |
|---|---|---|---|
| torch, compiled | 7.85 – 8.94 | 111.8 – 127.5 | 1.05× – 1.21× |
| JAX, jitted | 15.36 – 16.61 | 60.2 – 65.1 | 1.03× – 1.14× |

```bash
uv run python tools/backend_smoke.py --timing
```

**1.78× to 1.96× in PyTorch's favour**, ranges not overlapping and the gap well
outside the within-backend spread.

This is a *timing*, at one width, one thread count, one machine, and it says nothing
about accuracy — so it does not by itself move the default, and `§0.6` still leads
with JAX on accuracy and on the weight of measurement already taken with it. What it
does retire is the sentence "JAX is faster", which is now true only of the eager
comparison it was measured on.

**The estimator needed fixing before any of this could be said.** A per-iteration cost
here is the difference of two wall-clocks, and one sample per run gave 6.36, 7.79,
8.10, 8.45, 10.27 and 10.53 ms for identical work — from which the cross-backend ratio
came out 1.35×, 0.99× and 1.25× on three consecutive tries, reversing. Wall-clock noise
is one-sided, so the check now takes the minimum over three repeats and prints the
spread; and when the cross-backend gap is smaller than the within-backend spread it
says so and refuses the headline.

#### 7.5.21 Is M4's criterion attainable? — the ruler check

§7.5.16 argued that one cell is 0.405 K of `T_c`, a relative `L2` of 0.0026, only
1.6–2.3× above the reference's own error — D35's failure mode, an acceptance bar
sitting at the ruler's precision. **That worry is now measured, and it is wrong.**

`tools/m4_bar.py` refines the reference against itself. No network is involved:
solve at `n_axial` 40 → 1280 and watch how far the reference's *own* onset moves.

| `n_axial` | `Δt` threshold | `Δζ` threshold | `Δt` tangency | `Δζ` tangency |
|---|---|---|---|---|
| 40 | 0.000 s | 0.44 cells | 0.050 s | 1.94 cells |
| 80 | 0.000 s | 0.56 cells | 0.019 s | 0.94 cells |
| **160** (scoring) | **0.000 s** | **0.06 cells** | **0.009 s** | **0.44 cells** |
| 320 | 0.000 s | 0.19 cells | 0.004 s | 0.19 cells |
| 640 | 0.000 s | 0.06 cells | 0.001 s | 0.06 cells |

**At the scoring mesh the reference's own onset is uncertain by 0.06 cells and
0.009 s.** A one-cell, half-second criterion therefore sits an order of magnitude
*above* the ruler, not inside it. **M4's criterion is sound and the target is
attainable** — which means the failure is genuinely the network's, and chasing it is
worthwhile rather than chasing discretisation error.

The earlier worry confused two quantities: the *temperature* error (1.1–1.6e-3
relative `L2`) is not the *onset* error, and onset is far better converged than a
pointwise temperature because it is a threshold crossing of a monotone field.

Two things the ladder also settles:

**The threshold onset time is quantised, not converged.** It reads `0.000 s` at every
mesh because the 0.25 s output grid puts every answer in the same bin. That is the
same artefact §7.5.16 found in the network's scores, appearing in the reference.

**The tangency height converges to `ζ = 1`**, exactly the last cell centre at every
mesh, which is what proves it is reporting the outlet rather than locating a front.

#### 7.5.22 Every bar now carries the ruler's uncertainty beside it

Two acceptance bars in this project have been wrong in the same way — D35's void bar,
withdrawn because the reference's own error there is `3.2e-2`, and §7.5.16's onset-height
worry, which turned out to be unfounded only after §7.5.21 measured it. Both were
resolved case by case. Calibration practice has a rule for this, and adopting it removes
the judgement call.

**The test uncertainty ratio.** A tolerance is meaningful only if it sits at least **four
times** above the uncertainty of the instrument measuring it — standard since
MIL-STD-45662A (1988) and carried into ANSI/NCSL Z540. Applied here:

| quantity | reference's own uncertainty | bar | ratio | verdict |
|---|---|---|---|---|
| temperatures, relative `L2` | 1.1–1.6e-3 | 0.01 | ≈ 6.3 | **sound** |
| onset time | 0.009 s | 0.5 s | ≈ 56 | **sound** |
| onset height | 0.06 cells | 1 cell | ≈ 17 | **sound** |
| pointwise `α` | 3.2e-2 | 0.01 | ≈ 0.3 | **withdrawn, correctly** |

So the bars are fine. **The result is what has run out of room.** `T_s = 0.0017` against a
ruler of 1.1–1.6e-3 is a ratio of about **1.06** — the network's error is now the same
size as the reference's own. §0's headline is therefore stated as "meets the bar and has
reached the reference's resolution", not as a factor by which it beats it: below a ratio
of one, the comparison measures the ruler.

That has a consequence for what to do next. Further accuracy on the temperature fields
**cannot be demonstrated against this reference**, whatever the network does. Either the
reference is refined — it converges, so this is available and merely expensive — or the
claim moves to a quantity where headroom remains, which is onset time at a ratio of 56.
The second is free and is the direction §8 already points.

`tools/m4_bar.py` measures the reference's own uncertainty for onset. The same treatment
for every published bar belongs in `axial_study.py`, so that a bar and its ratio are
emitted together and neither can be quoted alone.

#### 7.5.23 Plan A's 84–92% miss is not a sampling problem

§7.4 measured the closed reactivity loop missing 84–92% of the **void** integral while
the Doppler integral — same fields, same network, non-cancelling weight — is right to
1.017. Every remedy proposed for it since has been a *sampling* remedy: put more
collocation points on the boiling front, where the void is.

Dual-weighted-residual theory answers that without training anything. The error in a
functional is `<R(u_theta), z*>` to leading order (Becker & Rannacher, *Acta Numerica*
10), where `z*` solves the adjoint problem with the functional's derivative as source.
The coolant equation is advective, so its adjoint runs **backwards** in $\zeta$ and `z*`
accumulates the void worth from the outlet downwards. `tools/plan_a_adjoint.py` computes
it in closed form at the time the functional peaks.

| quantity | value |
|---|---|
| `J+` (positive worth) | +4.656e-04 |
| `J-` (negative worth) | −1.695e-04 |
| cancellation ratio | 0.4663 |
| `\|z*\|` at the inlet | 6.09e-06 |
| `\|z*\|` at the outlet | 0.0 |
| support | 72% of the channel, up to $\zeta = 0.7156$ |

**The adjoint is a step function**, and more sharply than the hypothesis predicted: the
void slope underflows to *exactly* zero wherever the coolant is subcooled, so the adjoint
source lives only on the boiling band, and accumulating from the top makes `z*` zero above
the band and constant below it.

Two consequences. Every point in the lower channel carries **equal** sensitivity and every
point above the front carries **none** — so residual-magnitude sampling, which
concentrates at the front, aims at the region the functional is insensitive to and would
be expected to hurt. And a uniform sampler is already near-optimal here, which means the
miss is not about where the points go at all: it is the field's accuracy in the lower
channel, weighted by a constant.

The cancellation ratio is the other half of the answer. At 0.4663 a relative error of
$\epsilon$ on each half becomes 2.1 $\epsilon$ on the sum, so **`J+` and `J-` are reported
separately from here on**. A single near-cancelling number can be right by accident, and
when it is wrong it does not say which half failed.

#### 7.5.25 M4 is retired, and what has to replace it

M4 asks for onset within 0.5 s **and** one cell. Both halves are now dead, for opposite
reasons, and neither death is the network's fault.

- **The height half cannot be failed.** `T_c` is monotone in `ζ`, so the peak is the last
  node for the network and for the reference alike (§7.5.16's retraction). Any height
  error is a tautology.
- **The time half is passed**, at three seeds, by the budget alone (§7.5.16a).

So M4 no longer separates two formulations, and an acceptance criterion that cannot
separate two formulations is not a criterion. Chasing it further would be measuring
nothing.

**What a replacement has to clear.** Two hurdles, not one, and this project has only
ever checked the first:

1. its **bar** must sit at least 4× above the reference's own error on that quantity —
   the test uncertainty ratio of §7.5.22, whose absence retracted D35; and
2. the **network's current error** must itself sit well above that same reference error,
   or there is no headroom left to rank anything in.

Hurdle 2 is new here and it is what kills most of the obvious candidates. Refining the
reference against itself at the scoring mesh (`tools/m4_bar.py`, `n_axial` 160 against
640):

| quantity | reference's own error | network now | TUR of the result |
|---|---|---|---|
| temperatures, relative `L2` | 1.1–1.6e-3 | 1.7e-3 | 1.06 |
| onset time | 0.009 s | 0.0181 s | 2.0 |
| peak voided length | 0.569% | 0.7% | 1.2 |
| pointwise `α` | 3.15e-2 | — | 0.3 (withdrawn) |
| **void `J+`** | **1.742%** | not split yet | — |
| **void `J-`** | **0.053%** | not split yet | — |
| **void `J` (sum)** | **2.735%** | **84–92%** | **≈31** |

**Everything this project currently scores on has run out of room.** The temperatures,
the onset time and the voided length are all within a factor of 2 of the reference's own
error. That is not a complaint about the reference — it converges, and refining it is
available and merely expensive — it is a statement that these three quantities can no
longer rank two formulations against *this* ruler.

**The closed-loop void reactivity is the exception, by a factor of thirty.** The
reference knows its own void functional to 2.7% at the scoring mesh, and Plan A misses it
by 84–92% (§7.4). A bar at, say, 20% would sit 7× above the ruler — hurdle 1 cleared —
while leaving a gap the network is nowhere near — hurdle 2 cleared. It is also the
quantity the model exists for: void feedback is what drives the ULOF excursion, and
§7.5.23 has already shown the miss is a field-accuracy problem in the lower channel
rather than a sampling one, so it is attackable.

**A proposed M4′ scoring `J+` and `J-` separately, each to 20%, has been measured and
half of it is dead.** `tools/plan_a_adjoint.py --network` trains at the shipped default
and splits the network's own functional. Three seeds, JAX:

| seed | `J+` error | `J-` error |
|---|---|---|
| 0 | 1.66% | **0.0000%** |
| 1 | 2.10% | **0.0000%** |
| 2 | 2.66% | **0.0000%** |

**`J-` is bit-identical to the reference on every seed** — `-1.694643e-04` on both
sides. That is not accuracy. At the time the functional peaks the negative-worth region
is *fully voided*, `α = 1`, in the network and in the reference alike, so
`J- = Σ w·1·dζ` is fixed by the geometry and cannot be got wrong by any network that
boils the top of the channel at all. It is the height half of M4 again: a criterion that
cannot be failed. **`J-` is withdrawn as a criterion**, one section after being proposed.

And `J+` has no room either. 1.66–2.66% against a ruler of 1.742% is a test uncertainty
ratio of about **1.2** — inside the ruler, exactly like the temperatures, the onset time
and the voided length.

**So the open-loop split is exhausted too, and that locates the remaining headroom
precisely.** §7.4's 84–92% is a miss on the *closed* loop, where `ρ_void` feeds back
into the kinetics and the error compounds; the open-loop functional evaluated on the
network's own `α` is right to within the ruler. The two are different measurements and
conflating them would have set a bar on the wrong one. **M4′ must be a closed-loop
criterion** — Plan A's reactivity, not the functional evaluated on a field — and that is
the measurement to design next.

One thing survives from the analysis above and is worth keeping: if a closed-loop bar is
set, the scoring mesh should probably move to `n_axial = 320` for it, where the
reference's own error on `J` falls from 2.735% to 0.922% and a 5% bar would clear the 4:1
ratio. The temperature fields do not need that; this functional would.

**The functional peaks at t = 16.50 s** — the end of the valid window, where
`exp(t̂ N)` has its largest excursion and the network is least constrained. So M4′ is a
hard target as well as a live one, which is what M4 stopped being.

#### 7.5.26 Gauss-Newton, at equal wall-clock, loses by 33x

Roadmap D.4, measured. `tools/gauss_newton_experiment.py --solver dual`, the shipped
default configuration, 9000 s against the default budget's measured 9060 s — equal
wall-clock, not equal iterations, for the reason §7.5.17a gives.

The step is solved in **residual space**: with `m = 24000` residuals against
`n = 50309` parameters the dual system is the smaller one, so a subsample of 3000 rows
is formed densely and Cholesky-solved, which removes the sketch rank, the CG tolerance
and the preconditioner from the surface in one move (arXiv:2505.21404).

| | dual Gauss-Newton | shipped default |
|---|---|---|
| wall-clock | 9271 s | 9060 s |
| steps | 37 | — |
| training loss | 6.22e-2 → 1.12e-4 | — |
| `T_s` | **0.0569** | **0.0017** |
| `L_void` | 58% of reference | 99.3% |
| worst margin | +9.6 K | +67.6 K |

**33× worse than L-BFGS at the same price**, and it *stalled*: the last four steps moved
the loss by nothing while the damping climbed to `λ = 2.1e7`, which is Levenberg-Marquardt
collapsing the step towards gradient descent. Seed 0, one sample — stated, though at 33×
the seed is not the question.

Note the loss fell 560× while `T_s` stayed poor. That is the measure bug of Annex C in a
new place: the training loss is a mean over collocation points and the score is a field
norm against the reference, and a method can drive one a long way without the other.

**D.4 is refuted on its own terms.** The premise was that ill-conditioning — measured at
8.1e7 in the probe — makes a curvature-aware step worth its cost. It is not: L-BFGS is
already a curvature-aware method, it accumulates that curvature for free across
iterations, and 37 exact Gauss-Newton steps buy less than 30000 quasi-Newton ones.

#### 7.5.27 The optimiser bake-off, funded at last — and plain L-BFGS wins

`optimizer` ran at 3000 Adam / 300 quasi-Newton, the starved diagonal §7.5.11 shows is
the regime where the quasi-Newton stage does not matter, and §7.5.22's audit found the
*funded* bake-off had never been run at all. It has now, at 30 Adam / 3000 quasi-Newton,
three seeds, JAX:

| optimiser | `T_s` | worst margin | sec |
|---|---|---|---|
| **`lbfgs`** | **0.0258 [.0246–.0271]** | **+30.0 K** | 1033 |
| `lbfgs-shared` | 0.0265 [.0258–.0269] | +31.4 K | 1449 |
| `ssbfgs` | 0.0421 [.0392–.0453] | +6.5 K | 1701 |

**Plain L-BFGS wins on both columns.** Self-scaled BFGS is 63% worse on the mean and
loses 79% of the margin at 1.65× the wall-clock — and the margin is the column that says
whether there is a boiling front at all. `lbfgs-shared`, this repository's own
implementation, tracks the framework one to 3%, which is the parity check that makes the
comparison readable.

This is the regime-dependent negative §7.5.2 predicted from first principles: L-BFGS
already applies the Oren–Luenberger scaling to `H₀` every iteration, so self-scaling is
largely redundant against an L-BFGS baseline while supplying something real against the
unscaled full-memory BFGS the contrary papers use.

> **`ssbroyden` was dropped, for cost, and the cost is the finding.** Its arm ran **8.2
> hours on a single seed** against `lbfgs`'s 1033 s — 28× and still going — and was
> stopped. That is structural rather than unlucky: the Broyden replay is `O(m²n)` per
> iteration against L-BFGS's `O(mn)`, so at `history = 50` it is ~50× the work per step,
> and the 4.3× on record was measured at `qn300` where far fewer curvature pairs
> accumulate. A method whose per-iteration cost scales with the square of the memory is
> not competitive at the budget where the quasi-Newton stage actually matters. Stated as
> a dropped arm, not omitted.

#### 7.5.28 beignet implemented — the only honest test of "Adam replaces Newton"

arXiv:2605.24278 reports a **trainable multi-resolution Fourier feature pyramid** reaching
"an accuracy regime previously attained only by using computationally expensive
higher-order optimizers", *using Adam*. The claim is about the architecture, not the
optimiser, which is what makes it testable here — and what makes running our own Adam
longer a strawman, since §7.5.11 measured that axis flat and every study on disk has it
saturating near 0.04–0.05 whatever is spent.

Implemented in both backends. Each level holds a learnable periodic grid queried by the
bandlimited interpolant of that grid, `g(u) = Re Σ_k DFT(θ)[k] exp(2πi k u) / N`. The
grids are **trainable** — which is the whole mechanism, and the one thing that separates
this from our existing multi-band arms, whose `B` is frozen under `stop_gradient`.

Correctness asserted as properties rather than numbers: the interpolant reproduces its own
grid at the nodes to 1e-16 in both backends, interpolates between nodes rather than
stepping, takes gradient on every grid, and the two backends agree to 4.4e-16 from the
same grid. Short both-backend pass: torch 40.8 s, JAX 22.3 s, identical parameter counts.

**One registered deviation, and it is the main scientific risk.** Fourier interpolation is
periodic and every benchmark in the paper is a periodic problem; this channel is not,
since `T_c` rises monotonically from inlet to outlet. `beignet_pad` maps `ζ` into the
interior of one period. If that is insufficient the failure will appear as error
concentrated at the inlet and outlet rather than as a uniformly worse field, and that
distinction decides whether a negative is about the paper or about the port.

Measured cost: **1.94× a Fourier-embedding Adam step** (1613 against 831 ms/iter).
`axial_study.py adamonly` runs it against the frozen-Fourier embedding at the same 30000
Adam iterations and the same `lr = 1e-3`, one knob apart. Not yet measured.

> The paper's own Table 2 puts MLP + BFGS at 7.11e-20 against beignet + Adam's 6.63e-19.
> Even there, Adam *reaches* the higher-order regime rather than winning it.

#### 7.5.24 Bands and budget are not two gains — the budget subsumes the embedding

The 2×2 nobody had published. `uv run python tools/axial_study.py bandsbudget`, JAX,
three seeds per cell, `fourier_bands ∈ {single, (1,4,16)}` × `lbfgs_iters ∈ {3000, 30000}`.

| cell | `T_s` | seed range | worst margin | `L_void` | % of reference |
|---|---|---|---|---|---|
| `single/qn3000` | 0.0258 | .0246–.0271 | +30.0 K | 0.2973 | 78.0% |
| `bands/qn3000` | 0.0309 | .0257–.0358 | +34.3 K | 0.3638 | 95.4% |
| **`single/qn30000`** | **0.0017** | .0016–.0017 | **+67.6 K** | **0.3784** | **99.3%** |
| `bands/qn30000` | 0.0096 | .0073–.0139 | +31.3 K | 0.3523 | 92.4% |

**The control passed first.** `single/qn30000` reproduces the `qn30000` rung exactly — 0.0017, range
1.06, margin +67.6 K — so the harness had not moved and the other three cells are
readable.

**At `qn3000` bands do what §7.5.14 credited them with.** They trade the mean for the
front: 1.20× worse on `T_s`, but voided length 95.4% against 78.0% and a better
worst-seed margin. The mechanism is as advertised — the high band resolves the
near-discontinuity at the cost of splitting a fixed feature total three ways.

**At `qn30000` it reverses on every column.** `T_s` is **5.6× worse**, voided length
falls to 92.4%, and the margin more than halves. Per seed the bands cell is
0.0139 / 0.0077 / 0.0073 against 0.0016 / 0.0016 / 0.0017 — the two distributions do not
come close to overlapping, so this is not a seed artefact.

**Interaction 0.21.** Bands buy 0.83× at `qn3000` and 0.17× at `qn30000`; if the two
axes were independent that ratio would be near 1.

**The mechanism is the one Annex E predicted, and it is stronger than "redundant".**
SAFE-NET (arXiv:2502.07209) frames Fourier features as an *implicit preconditioner*.
While the optimiser is under-converged that preconditioning is worth having. Once L-BFGS
has accumulated enough curvature pairs to precondition itself it is redundant — and what
remains is the **capacity loss**: each of three bands gets a third of the 256 features and
is therefore coarser than one 256-feature band. So the embedding is not merely subsumed,
it is a net cost past the point where the budget does the same job.

**And the budget alone delivers the front result the bands were credited with** — 99.3%
of the reference voided length against the bands' best 95.4%.

Two consequences, both acted on:

- **`fourier_bands` stays off permanently**, not "pending the 2×2". The Annex D roadmap
  item proposing it as a default is closed against.
- **§7.5.14's headline is a statement about a starved budget** and is corrected there.
  That is now the **third** published conclusion in this document that was an artefact of
  measuring at `qn3000` — after §7.5.11's Adam axis and §7.5.27's optimiser bake-off. The
  pattern is general enough to state as a rule: *an architectural comparison run at an
  under-converged budget measures which architecture reaches the under-converged state
  faster, which is a different question from which is more accurate.*

#### 7.5.29 The embedding is representation, not preconditioning — and Newton is the cheap one

Two questions answered together, JAX, seed 0, at the shipped configuration except where
the row says otherwise.

**Is the Fourier embedding just a preconditioner?** §7.5.24 showed the *multi-band*
embedding is subsumed by a funded budget. If the embedding *itself* were also only
preconditioning, then removing it and funding the quasi-Newton stage fully should recover
the same answer. It does not:

| arm | embedding | budget | `T_s` | margin | sec |
|---|---|---|---|---|---|
| **shipped default** | f256 | adam30 / qn30000 | **0.0017** | **+67.6 K** | 9060 |
| **no embedding, funded** | none | adam30 / qn30000 | **0.0397** | **+0.8 K** | 2980 |
| no embedding, Adam only | none | adam30000 / qn0 | 0.0443 | **−1.6 K** | 3233 |
| no embedding, AdEMAMix | none | adam30000 / qn0 | 0.0460 | +0.9 K | 3465 |
| f256, Adam only | f256 | adam30000 / qn0 | 0.0329 | +21.6 K | 10149 |

**23× worse with the front gone.** +0.8 K of margin against the reference's +69.2 K, out
of a 590 K range. So the two embedding results are opposite and both stand: *splitting* a
fixed feature budget across bands is preconditioning, which the funded optimiser
duplicates; *having* the embedding at all is **representation** — high-frequency capacity
a raw-coordinate MLP cannot synthesise, and no amount of curvature buys it.

The sharpest line in the table is that **without the embedding nothing forms a front** —
−1.6 K, +0.8 K, +0.9 K, all at or below the noise floor whatever the optimiser. The
Fourier features are what make the front *representable*; the quasi-Newton budget is what
makes it *accurate*. Neither substitutes for the other.

##### The first-order methods buy nearly nothing, and they are not cheaper

Adam-only at 30000 iterations reaches 0.0329 with the embedding against the funded
stage's 0.0017 at comparable wall-clock — **19× worse** — and lands in the 0.04–0.05 band
every Adam arm on disk saturates in. AdEMAMix, at `optax.contrib` defaults, is *worse*
than Adam here (0.0460 against 0.0443), which does not reproduce `REPORT-01`'s 0D note of
~2× better beyond 8000 iterations. One seed, at unswept `b3`/`alpha`, so it is stated as
a sample and not as a refutation of that note.

**And the cost argument for preferring them is backwards in this repository.** Wall-clock
per nominal iteration, measured at matched configuration and matched iteration count:

| configuration | L-BFGS | Adam | AdEMAMix |
|---|---|---|---|
| no embedding | **99.3 ms** | 107.8 ms | 115.5 ms |
| f256 | **302.0 ms** | 338.3 ms | — |

**The quasi-Newton stage is the cheapest of the three, in both configurations** — 1.09×
to 1.16× cheaper per iteration than the first-order methods it is supposed to be an
expensive alternative to. The mechanism is implementation, not theory: the quasi-Newton
stage is a jitted `fori_loop` while the Adam stage steps through Python, and at this
problem size the per-step cost is dominated by the residual gradient, which both pay
identically. Adam and AdEMAMix are also indistinguishable from each other at the step
level — measured back to back at identical configuration, `ademamix/adam = 0.994`.

That removes the usual justification for the Adam→L-BFGS split. The literature calls
higher-order methods "computationally expensive" (arXiv:2605.24278's own phrasing); here
the higher-order stage is **faster per iteration and 19× more accurate**, and the
first-order stage is very nearly free to delete — the `qnladder` sweep measured `adam0/qn30000` at
0.0018 against `adam30`'s 0.0017, with overlapping seed ranges.

> **The cost result is hardware-scoped; the accuracy results are not.** Two different
> kinds of claim sit in this section and they do not travel together.
>
> *The clock is ours.* 50 309 parameters, full batch, float64, eight CPU cores. On a
> datacenter GPU an Adam step over a large collocation batch is embarrassingly parallel
> and close to free, while a strong-Wolfe line search is **sequential** — several
> function evaluations that cannot overlap. FP64 runs at 9.7–34 TFLOPS on A100/H100
> against roughly 1/32 to 1/64 of FP32 on consumer parts (§7.7), so the papers reporting
> higher-order methods as "computationally expensive" may be stating something literally
> true of their regime and false of ours. **Our 99.3 ms against 107.8 ms could invert on
> their hardware**, and nothing here contradicts them on cost.
>
> *The accuracy is not ours alone.* `adam0/qn30000` at 0.0018 against `adam30`'s 0.0017,
> Adam saturating at 0.04–0.05 across a 500× budget range, and the front failing to form
> without the embedding are properties of the optimisation trajectory in float64. Faster
> hardware reaches the same numbers sooner. The measurement is not that Adam is a bad
> optimiser; it is that nothing here needs what Adam provides.

> **The two stages do not minimise the same objective, and the comparison should say so.**
> The Adam stage draws a **fresh collocation set every step** — about 1.2e8 distinct
> points over 30 000 iterations — so it minimises the *expected* residual over the
> domain. The quasi-Newton stage runs on **one fixed set of `n_colloc` points** for its
> whole run, because curvature pairs are meaningless if the objective moves underneath
> them. So "Adam 30000 against qn30000" pits a noisy population objective against an
> exact finite-sample one, and part of Adam's deficit is that it is solving the harder
> problem.
>
> That does not rescue it, for a reason that is checked rather than argued: with
> `n_colloc = 4000` the quasi-Newton stage has roughly 16 000 residual constraints
> against 50 309 parameters — **underdetermined by 3×** — so it could drive its fixed set
> to zero and be arbitrarily wrong between the points. It is not. `T_s` is scored against
> the Radau reference on a *different* grid, and 0.0017 with `L_void` at 99.3% is a
> generalisation result, not a training-set one. `REPORT-01` §5.1 raised exactly this
> worry — "it can overfit that set; always report trajectory error, not just loss" — and
> the trajectory error is what this document reports.

#### 7.5.30 What the quasi-Newton stage is actually handed — and how little the Adam loop does

Two facts about the recipe that no table in this document records, both found by reading
the training loop rather than by measuring anything.

**The quasi-Newton stage trains on one arbitrary sample, and the two backends choose it
differently.** In JAX the polish receives `pts` as it stands when the Adam loop exits —
**the last Adam iteration's draw**, merged with whatever RAR points have accumulated.
Nobody chose that set; it is an artefact of loop structure. In torch, `_lbfgs` calls
`self.collocation()` and draws its **own** fresh set.

**The samplers themselves are identical**, and that was checked rather than assumed:
both produce `n_colloc` uniform points, plus an early-time cluster of `n_colloc // 2`
over the first 40% of the window, plus the RAR reservoir when it is non-empty, plus front
points when `front_frac` is active. So the difference is the *provenance of the draw* and
nothing else — two independent samples from one sampler. No published comparison is
affected.

> An earlier revision of this paragraph said only "both are single uniform draws", which
> reads as though torch might not carry the RAR reservoir into its polish. It does
> (`parts = [pts, early, *([self.rar] if self.rar.numel() else [])]`). Recorded because
> the sentence implied a cross-backend divergence that does not exist, which is the kind
> of claim this document is supposed to check before making.

At `n_colloc = 4000` that set carries roughly **16 000 residual constraints against
50 309 parameters**, underdetermined by 3.1×. The stage that does all the work in this
recipe is fitting an underdetermined system on 4000 arbitrary points and still reaching
`T_s = 0.0017` against a reference on a *different* grid. That is the ansatz and the
embedding constraining the function space, not the point count.

`n_colloc` and `lr` appeared in **no study row on disk** and have never been swept. Rows
now record both, plus `first_order` — a row that cannot state its own configuration is
the defect AGENTS.md already names for budgets, and it applied here the whole time.

**At the shipped default the Adam loop does almost nothing.** With `adam_iters = 30`,
every mechanism the loop exists to run is unreachable:

| feature | fires at `adam30`? | why |
|---|---|---|
| RAR refinement | **no** | first trigger at `it = 2000` |
| adaptive block weights | **no** | `weight_max_ratio = 1.0`, guard is `> 1.0` |
| time-window curriculum | **no** | `n_windows = 1`, so `t_max = 1.0` throughout |
| pseudo-time anchors | **no** | `pts_every = 0` |

So 30 iterations buys 30 Adam steps and 30 redraws of the collocation set, and nothing
else. That is the mechanical reason `adam0` and `adam30` are indistinguishable
— 0.0018 [.0017–.0022] against 0.0017 [.0016–.0017], overlapping — since the two
configurations differ only in a PRNG key and thirty warm-up steps. It is not that Adam's
contribution is small; **at this budget Adam is barely in the recipe.**

It also means RAR, adaptive weighting and the curriculum are live only at budgets nothing
currently ships at. `rar_every = 2000` needs `adam_iters > 2000`, which only the old
`adam3000` studies ever had — so those three features are, in effect, untested at the
configuration every recent number was measured at.

#### 7.5.31a The budget split is a consequence of our collocation count, not only of the optimisers

A caveat on §7.5.11 and on the `qnladder` sweep, argued rather than measured, and recorded because it
is the kind of thing this project otherwise rediscovers as a retraction.

> ## Retracted: the premise is an arithmetic error
>
> **This section's central claim — that we run an underdetermined collocation set —
> is wrong, and it is wrong because it counted the encoder as fitting capacity.** The
> correct count is in §7.5.37a; the shipped configuration runs at **1.41**, comfortably
> on the *right* side of the literature's line, and always did. Everything below is kept
> for the record. What survives is the cost argument — a quasi-Newton iteration is
> full-batch and linear in the point count — and §7.5.38's threshold, which is a real
> effect that this section's numbers merely mislocated.

**The literature prescribes an overdetermined collocation set and we run a 3.1×
underdetermined one.** arXiv:2605.30910 — *PINNs Failure Modes are Overfitting* — argues
that PINN failures are overfitting rather than architectural or optimiser deficiencies,
and prescribes overdetermining the system: collocation points substantially exceeding
parameters. Other work sets point counts per sub-problem specifically to hold the
residuals-to-parameters ratio fixed.

This model's numbers are on the wrong side of that line. The sampler draws
`n_colloc` uniform points **plus `n_colloc // 2`** in an early-time cluster, so the
shipped `n_colloc = 4000` is **6000 points** per step; at four residual blocks each and
49 797 *trainable* parameters (50 309 includes the 512 frozen `B` entries):

| `n_colloc` | points drawn | residuals | residuals / parameters | `qn30000` cost |
|---|---|---|---|---|
| **4000 (shipped)** | 6000 | 24 000 | **0.482** | 2.5 h |
| 8300 | 12 450 | 49 800 | 1.00 (break-even) | 5.2 h |
| 10 540 | 15 810 | 63 240 | 1.27 | 6.6 h |

> An earlier revision of this table said 4000 points, 16 000 residuals and a ratio of
> 0.32 — it forgot the early-time cluster, which is half as many points again. The system
> is underdetermined by 2.07×, not 3.1×. The argument is unaffected; the arithmetic was
> wrong by 1.5× and is corrected here rather than quietly.
>
> **And the corrected figure was still wrong**, by a larger factor and for a different
> reason: 49 797 counts 32 768 parameters of encoder read-out. The ratio is 1.409. Two
> corrections to one table, neither of which questioned the denominator — see §7.5.37a.

**And our own training loop already cites that paper — for the other half of its
argument.** The JAX Adam loop resamples every step and the comment says a frozen set is
"the collocation-overfitting mode of arXiv:2605.30910". We took the resampling lesson and
not the counting one. The resampling fix applies only to the *Adam* stage; the
quasi-Newton stage is fixed-set **and** underdetermined, which is squarely the regime that
paper describes.

**The consequence for the headline.** §7.5.11 and the `qnladder` sweep conclude that the quasi-Newton
axis does all the work and the Adam axis none. That is measured and stands *at this
collocation count* — but the count is what makes the measurement affordable. A
quasi-Newton iteration is full-batch, so its cost is linear in the point count; an Adam
step over the same points parallelises. At an overdetermined set the same 30 000
quasi-Newton iterations cost ~10 h rather than 2.5 h, and the split a paper can buy shifts
hard toward Adam. **`adam 10^5 / qn 10^3` may be the rational choice at their ratio and
`adam 30 / qn 30000` the rational choice at ours**, with no disagreement about the
optimisers at all.

##### And Adam is run FULL BATCH, which is the larger confound

Every Adam step in this project evaluates **all 6000 points**. That is a full-batch
gradient with momentum and per-coordinate scaling — it is not stochastic minibatch Adam,
and there is no separate knob for an Adam batch size: one sampler feeds both stages.

That explains §7.5.29's cost measurement entirely. Adam at 107.8 ms per iteration against
L-BFGS at 99.3 ms looks like a tie because **both evaluate 6000 residuals per step**, and
that evaluation is essentially the whole cost. Adam's advantage in the literature comes
from *not* doing that — 128 or 256 points per step is 20–45× cheaper, so 20–45× more
steps fit in the same wall-clock.

So "Adam is 19× worse at comparable wall-clock" is properly read as **"full-batch Adam is
19× worse than L-BFGS on the same full batch"**. That is a far narrower claim, and an
unsurprising one: a full-batch first-order method against a full-batch second-order method
on an ill-conditioned problem is the comparison quasi-Newton is supposed to win. **The
property that makes Adam worth using was never given to it here.**

So the finding should be read as *"at 0.48 residuals per parameter, with Adam run full
batch, the quasi-Newton axis dominates"*, not as a general claim about first- against
second-order methods. Whether
0.0017 is a ceiling imposed by the underdetermined set, or achieved despite it, is
unmeasured — and §7.5.30 notes the ansatz and embedding are evidently doing the
constraining, since the fixed 4000-point solution generalises to a reference on a
different grid.

#### 7.5.31 The capacity ladder is flat at a funded budget — f32 matches f256

`uv run python tools/axial_study.py fourierbudget`, JAX, `adam10000 / qn30000`, three
seeds per rung.

| `fourier_features` | `T_s` | seed range | `L_void` | worst margin | sec | arrays | capacity |
|---|---|---|---|---|---|---|---|
| **f32** | 0.0026 | .0021–.0032 | 0.3748 | +65.4 K | **5110** | 21 125 | 17 029 |
| f64 | 0.0025 | .0018–.0031 | 0.3729 | +65.7 K | 6122 | 25 221 | 17 029 |
| **f128** | **0.0019** | .0018–.0022 | 0.3769 | +67.2 K | 7208 | 33 413 | 17 029 |
| f256 (shipped) | 0.0024 | .0019–.0029 | 0.3687 | +65.2 K | **12 289** | 49 797 | 17 029 |

**Every rung's seed range overlaps every other rung's.** f32 spans .0021–.0032 against
f256's .0019–.0029, and f256 is *nominally worse than f128* while costing 1.7× more.
`L_void` moves 2% across an eightfold change in embedding width; the worst-seed margin
moves 2 K.

**f32 matches f256 at 42% of the wall-clock**, so nothing above f32 is worth paying for at
this budget.

> **This section's title was wrong, and the last column is why.** It is not a *capacity*
> ladder: the fitting capacity is **17 029 at every rung** and only the encoder read-out
> grows (§7.5.37a). §7.5.42 re-measures the same axis on the recommended configuration —
> `adam0`, 5000 points, a funded polish — and finds it flat there too, so the default
> stays f64. A flat result across a column that does not vary is not a finding
> about capacity — it is the arithmetic working. What the ladder does measure, and
> measures usefully, is that **a 32× wider random-Fourier basis buys nothing**, which is a
> statement about the embedding and is the one §7.5.29 makes on other grounds. The
> original text read the flatness as "capacity above f32 buys nothing measurable"; that
> claim has never been tested here, because no arm in this document has changed the body.

§7.5.12's ladder — which is where the shipped `fourier_features = 256` comes from, as its
config comment says — was measured entirely at `adam300 / qn3000`. Every row of
`margin.json`, `margin256`, `margin512` and `margin1024` carries that budget. It found
f32 → f512 monotone; at a funded quasi-Newton stage the same axis is flat.

> ### This is the fourth conclusion invalidated by `qn3000`, and that is now the finding
>
> §7.5.11's Adam axis, §7.5.27's optimiser bake-off, §7.5.24's multi-scale bands, and now
> §7.5.12's capacity ladder were all drawn from measurements at a starved quasi-Newton
> stage, and all four dissolve or reverse when the stage is funded. Four is enough to stop
> treating them as separate corrections:
>
> **Every architectural conclusion this project drew was an artefact of an under-converged
> optimiser.** An architectural comparison run at an under-converged budget measures which
> architecture reaches the under-converged state faster, which is a different question
> from which is more accurate — and this document answered the wrong one four times.
>
> The corollary is uncomfortable and worth stating: the remaining architectural choices
> here — width, depth, `fourier_scale` — were never measured at all (§7.5.30), so they are
> not even in the position the four above were. They are simply unexamined.

#### 7.5.32 Freezing the encoder makes the polish worse, at either Adam budget

§7.5.30 showed the first Linear — the Fourier-to-trunk projection — is 66% of the model,
and §7.5.31a that letting the quasi-Newton stage move it is what leaves that stage 2.07×
underdetermined. Holding it fixed makes the polish overdetermined (24 000 residual entries
against 16 965 weights, 1.41), which is the side of the line the literature prescribes.

Measured, seed 0, against a byte-identical control:

| arm | `T_s` | `L_void` | margin | sec |
|---|---|---|---|---|
| control, `freeze=False` | **0.0019** | 0.3772 | +67.3 K | 11 584 |
| `freeze=True` after `adam10000` | 0.0141 | 0.3297 | +50.9 K | 10 061 |

**7.2× worse**, at 0.87× the cost.

The obvious objection is that 10 000 full-batch Adam steps had barely moved the encoder,
so the frozen layer was near its initialisation and the arm tested a random-feature
method. §7.5.33 removes that objection and the answer does not change.

**So determinacy is not the binding constraint.** Making the polish overdetermined by
removing parameters is strictly worse than leaving it underdetermined with the encoder
free. That agrees with §7.5.29 from the opposite direction: the embedding is
*representation*, and the polish needs to keep adjusting it.

#### 7.5.33 The deep-learning schedule — small-batch Adam does not rescue it either

Every Adam measurement in this document was made at **full batch** (§7.5.31a): the Adam
stage evaluating the same 6000 points the quasi-Newton stage uses. That is not how a
first-order method is run anywhere — JAX-PI takes 200 000 steps of 4096 points, and the
cost advantage that makes Adam attractive comes entirely from the small batch. So "Adam
buys nothing here" had only ever been tested against an algorithm nobody uses.

`axial_study.py dlstyle` runs the protocol properly: **60 000 Adam steps at 1000 points**,
then the encoder frozen, then **30 000 quasi-Newton iterations at 6000 points redrawn
every 1000** — blocked restarts as in arXiv:2605.24278, so curvature stays consistent
within a block while the stage as a whole cannot overfit one draw.

| arm | `T_s` | `L_void` | margin | sec |
|---|---|---|---|---|
| **dlstyle** | **0.0324** | 62% | +24.2 K | 11 229 |
| `f256 adam10000/qn30000`, full batch | **0.0019** | 99% | +67.3 K | 11 584 |

**17× worse at the same wall-clock**, and it lands in the same 0.03–0.05 band every
Adam-only arm has occupied regardless of batch size, step count, embedding or optimiser
variant (§7.5.29).

So the full-batch objection was a real methodological gap and is now measured **not** to
be the explanation. Small-batch Adam, in the regime and at the step count the literature
uses, does not close the gap on this problem.

**It changes three things at once, deliberately — it is a schedule, not an ablation** —
so it cannot attribute the failure between small-batch Adam, the freeze, and the blocked
polish. Given §7.5.32 measured the freeze alone at 7.2× worse, the freeze is the prime
suspect, but that is inference and is labelled as such.

#### 7.5.31b The design of that sweep, and the risk it carried

§7.5.29 established the embedding is *necessary*. It did not establish how much of it is.
`uv run python tools/axial_study.py fourierbudget` sweeps `fourier_features ∈ {32, 64,
128, 256}` at `adam10000 / qn30000`. The embedding is the widest layer in the network, so
each rung is a cheaper step than the last, and the question is whether the extra
iterations a cheap embedding buys within a wall-clock make up for the capacity it gives
away.

§7.5.12's capacity ladder found f32 → f512 monotone — **measured at `qn3000`**. That puts
every capacity conclusion here in exactly the position §7.5.14's bands were in before
§7.5.24 refuted them: a statement about a starved optimiser. If the ladder flattens at a
funded budget, the shipped f256 is over-specified.

`adam_iters = 10000` is above `rar_every`, so **RAR is active in these arms** and it is
not in any recent table. That is a live difference in the recipe, not only a budget
change — so the f-ladder above is internally consistent but is not directly comparable to
the shipped `adam30/qn30000` row, where RAR never fires.

#### 7.5.34 Removing the Adam stage makes the axial model *better* — 2×2, three seeds

The `qnladder` sweep measured `adam0/qn30000` at 0.0018 against `adam30`'s 0.0017 and called them
indistinguishable, which made Adam look merely useless. §7.5.30 then showed why that was a
weak test: at 30 iterations the Adam loop does nothing at all, so it compared no Adam
against almost no Adam.

`axial_study.py adamcheck` compares **no Adam** against **10 000 full-batch Adam steps** —
a budget where RAR fires and the loop does real work — at two embedding widths, with an
identical polish: 50 000 quasi-Newton iterations on 6000 points **redrawn every 1000**
(the blocked-restart protocol of arXiv:2605.24278). Three seeds per cell.

| | `adam0` | `adam10000` | ratio |
|---|---|---|---|
| **f32** | 0.0038 [.0021–.0056] | 0.0074 [.0042–.0118] | 1.9× |
| **f64** | **0.0024 [.0023–.0025]** | 0.0067 [.0030–.0124] | **2.8×** |

**f64's ranges do not overlap** — .0023–.0025 against .0030–.0124 — so at that width
removing the first-order stage entirely makes the model **2.8× more accurate**. f32's do
overlap (.0042–.0056 is common), so f32 shows the direction without establishing it.

`f64 adam0` is the best-behaved arm in this document: `L_void` 0.3784 (99.3% of
reference), worst-seed margin +66.5 K, and a seed spread of **1.09×** — the tightest
measured here. Both `adam10000` cells carry the widest spreads in the study, 2.8× and
4.1×, so the Adam stage adds variance as well as error.

**So the first-order stage is not merely removable, it is harmful at this budget.** That
does not contradict that; it explains it. Thirty Adam iterations do nothing and cost
nothing; ten thousand do something, and what they do is move the parameters somewhere a
curvature-based method does worse from.

> **And it is not our schedule's fault — §7.5.40.** The obvious objection to this section
> is that Adam did not plateau, our cosine decay ran out. Schedule-free AdamW, which has
> no schedule to end, is 33× worse than the quasi-Newton stage alone and produces **no
> boiling front at all**; as a warm start it makes the polish 13.6× worse. The reading
> above survives the strongest available version of the objection.

> **A retraction of my own reading, recorded because the pattern is the one this document
> keeps repeating.** On seed 0 alone `f32 adam0` read 0.0021 with 99.9% voided length, and
> that was reported as "the best arm we have measured" and compared against three-seed
> numbers. At three seeds it is 0.0038 [.0021–.0056] — the single seed was the *best* of a
> 2.7× spread. The hedge "one seed" was stated and was useless, exactly as AGENTS.md says
> a hedge under a confident headline always is. **f64, not f32, is the arm that survived.**

#### 7.5.35 The 0D model does the opposite, which is the cleanest evidence that this is formulation-dependent

`REPORT-01` §5.1 states that "L-BFGS from scratch on a PINN loss stalls", citing Rathore
et al. — but no row in this repository measured it, and §7.5.34 contradicts it outright on
the axial model. The 0D model, under the **same** optimiser implementation, the same JAX
backend and the same `memory = 50`:

| 0D arm | power relative `L2` | sec |
|---|---|---|
| `adam0/qn0` (untrained network) | 0.2083 | 6.1 |
| `adam0/qn100` | 0.1949 | 7.6 |
| **`adam0/qn1000`** | **0.4466** | 6.5 |
| **`adam100/qn100`** | **0.3997** | 13.9 |

**More quasi-Newton makes the 0D model worse than not training it at all.** 100 iterations
move the power error 6%; 1000 iterations take it to 2.1× *worse than the untrained
network*. That is not a stall, it is divergence in the trajectory.

It is also the exact failure `REPORT-01` §5.1 warns about two sentences later: the polish
"runs full-batch on a *fixed* collocation set — that is also how it can overfit that set.
Always report trajectory error, not just loss." The 0D polish drives the collocation
residual down while the trajectory goes the other way. The divergence guard does not catch
it because the guard is on the training loss.

**So both behaviours exist in this repository, under one implementation.** On the axial
model a pure quasi-Newton solve is the best configuration measured; on the 0D model it is
worse than an untrained network. The Adam-then-L-BFGS consensus is therefore neither
right nor wrong in general — it is a statement about a formulation, and the two
formulations here differ in three respects at once (multiplicative hard-constraint ansatz
with no penalty terms, a Fourier embedding, and advection with a near-discontinuous
front), so this measurement does not say *which* of them is decisive.

> **And a wrong diagnosis of my own, corrected.** A 50 000-iteration 0D probe ran for
> nearly three hours producing nothing, and I attributed it to XLA compilation growing
> with the `fori_loop` bound. That was wrong: `qn1000` compiles and runs in 6.5 s. The
> three hours are unexplained and the probe was killed; nothing here rests on it.

#### 7.5.36 The 0D backend had the same bare `optax.lbfgs()` — never fixed

Found while setting up §7.5.35. `pinn_jax.py` called `optax.lbfgs()` **bare**, defaulting
to `memory_size = 10`, while the torch twin passed `history_size = 50`. That is the
identical defect §7.5.17 traced in the axial model, where it accounted for the *entire*
cross-backend accuracy gap and was read as a framework difference for four milestones.

The axial model was fixed. **The 0D model was not, and nobody looked.** So every 0D
cross-backend comparison in this project is affected — including `REPORT-01` §5.1's table
reporting JAX's polish improving −74% / −40% against torch's −82% / −54%. That gap has
exactly the shape the memory defect produces, and it is sitting in the report as a
framework observation.

Both 0D configs now carry `lbfgs_history = 50`, so the two cannot drift again. The §5.1
table should be re-measured or marked as measured at `memory_size = 10` on the JAX side.

#### 7.5.37 The blocked restart hurts — and the best configuration in this project has no Adam stage

§7.5.30 posed a question it could not answer: the polish runs on one fixed collocation
set, which is **both** what makes a curvature estimate meaningful — the pairs are
meaningless if the objective moves underneath them — **and** what a 2× underdetermined
stage can overfit. arXiv:2605.24278 runs its BFGS baseline in blocks of 1000 for the
second reason. Every published number here used the fixed set; every arm in §7.5.34 used
the refresh; nobody had separated them.

One knob, `f64 adam0/qn50000` on 6000 points, three seeds:

| polish set | `T_s` | range | `L_void` | worst margin | sec |
|---|---|---|---|---|---|
| **one fixed set** | **0.0016** | **.0016–.0017** | **0.3790** | **+67.8 K** | 5859 |
| redrawn every 1000 | 0.0024 | .0023–.0025 | 0.3784 | +66.5 K | 8950 |

**The ranges do not overlap: redrawing is 1.5× worse.** Curvature consistency beats
overfitting protection here, decisively. And the overfitting the refresh was guarding
against does not appear — the fixed-set arm is scored against the reference on a
*different* grid and is the most accurate arm in this document, so whatever it fits on
4000 sampled points generalises.

> The wall-clock column is **not** a clean comparison. The refresh arms ran six-to-eight
> concurrent, the fixed arms three, and §7.3.2's rule is that a time from a loaded machine
> is not comparable to one from an idle one. The 1.53× is mostly contention; the two
> configurations do identical arithmetic apart from 50 optimiser restarts. **The accuracy
> comparison is clean; the timing one is not**, and no conclusion below rests on it.

##### The recommended configuration changes

| | shipped default | **best measured** |
|---|---|---|
| embedding | f256 | **f64** |
| Adam | 30 | **0** |
| quasi-Newton | 30 000, fixed set | **50 000, fixed set** |
| trainable arrays | 49 797 | **25 221** |
| *of which* fitting capacity | 17 029 | **17 029** — unchanged |
| `T_s` (3 seeds) | 0.0017 [.0016–.0017] | **0.0016 [.0016–.0017]** |
| `L_void` | 0.3784 | **0.3790** |
| worst margin | +67.6 K | **+67.8 K** |

**Equal or better on every column, with a quarter of the encoder and no first-order stage
at all.** The two `T_s` ranges are identical to four digits, so this is not a claim of
improved accuracy — both are at the reference's own resolution (§7.5.22) and neither can
be distinguished from the other. The claim is that **the same result is reachable with a
much smaller embedding and none of the Adam machinery**.

> **Corrected.** This table read "trainable parameters 49 797 → 25 221" and the sentence
> claimed "half the model". It is not half the model: §7.5.37a shows the **fitting
> capacity is 17 029 in both**, and what f64 halves is the encoder read-out. The
> measurement is untouched and the recommendation stands — a smaller embedding at equal
> accuracy is still worth taking, and it is 2× cheaper per iteration — but "half the
> model" overstated what changed.

> **The point count is superseded by §7.5.38; the budget survives.** Quoting only
> configurations that have been run:
>
> | | **recommended** | this section |
> |---|---|---|
> | embedding | f64 | f64 |
> | Adam | 0 | 0 |
> | polish set | **5000 points** | 6000 points |
> | quasi-Newton | 50 000 | 50 000 |
> | `T_s` (3 seeds) | 0.0016 | 0.0016 |
> | onset error | 0.0057 s | 0.0058 s |
> | cost | **~8000 s** | ~9600 s |
>
> Identical on every measured quantity for **83%** of the machine time. The 50 000
> iterations were the right call for the wrong reason — §7.5.39 shows the axis is
> monotone, so nothing below it is better at anything, and only the cost was ever in
> question. The 6000 points were not: they are 1.2× more than the recommendation and were
> inherited, never measured.
>
> A **4000-point, `qn30000`** corner would be cheaper still at a plausible ~3900 s, and is
> **unmeasured** — the two axes were swept as an L and never as a rectangle, so that
> corner has never been run and is not recommended from here.

That also retires a great deal of apparatus. At `adam_iters = 0` the RAR reservoir, the
adaptive block weighting, the time-window curriculum and the pseudo-time anchors are not
merely unreachable (§7.5.30) — they are absent. The recipe is: a hard-constraint
multiplicative ansatz, a frozen Fourier embedding, and one long quasi-Newton solve on a
fixed collocation set.

> **This also confounds §7.5.33.** The `dlstyle` arm carried the refresh, so its 17× deficit
> mixes small-batch Adam, the frozen encoder *and* a polish protocol now measured to cost
> 1.5×. Its conclusion — that small-batch Adam does not rescue the first-order stage —
> survives, because 17× is far larger than 1.5×, but the figure is not attributable and
> should not be quoted as if it were.

#### 7.5.37a The Fourier embedding is an encoder, and counting it as capacity broke three sections

**The fitting capacity of this model is 17 029 parameters at every embedding width.** Not
25 221, not 49 797. Those figures count the read-out projection, whose width is set by the
encoder's output and not by anything the network can express:

| | all arrays | frozen `B` | encoder read-out | **body** |
|---|---|---|---|---|
| f32 | 21 189 | 64 | 4 096 | **17 029** |
| f64 | 25 349 | 128 | 8 192 | **17 029** |
| f128 | 33 669 | 256 | 16 384 | **17 029** |
| f256 (shipped) | 50 309 | 512 | 32 768 | **17 029** |
| f1024 | 150 149 | 2 048 | 131 072 | **17 029** |

`B` is drawn once and held under `stop_gradient` in both backends, so it was already
excluded. The error was the next column: `mlp.layers[0].weight` has shape
`(width, 2 * n_features)`, so it grows 32× from f32 to f1024 while the five layers behind
it — 16 965 parameters of hidden weights, biases and the output head — never move.

**This is the explanation of §7.5.31, not merely a correction to it.** That section
measured the capacity ladder as *flat* from f32 to f256 and called it a surprising result
about capacity. It is not surprising and it is not about capacity: **the two arms have
identical capacity**, and the ladder was measuring the width of an encoder read-out. The
measurement stands; its title was wrong.

The rule, because this document will otherwise redo it: **a ratio needs a denominator
that varies when the model's expressive power varies.** The test of a candidate
denominator is whether changing it changes what the network can represent. Here, changing
the embedding width by 32× changes 128 000 parameters and nothing about the answer, which
is the evidence that those parameters are not capacity. `tests/axial/test_axial_pinn.py`
now pins the body count across widths and across both backends, so this cannot drift back.

#### 7.5.38 The polish set has a threshold, and it is not where determinacy predicts

§7.5.31a is the one open argument in this document that was **argued and never measured**:
the literature (arXiv:2605.30910) prescribes a collocation set that overdetermines the
system, and it claimed every number here came from an underdetermined one. This sweep
measures the axis directly, moving one knob — the size of the fixed polish set — over a
20× range. `f64 adam0/qn50000`, fixed set, `polish_colloc` the only difference.

| points | ratio | seeds | `T_s` | range | `L_void` | worst margin | onset err | sec |
|---|---|---|---|---|---|---|---|---|
| 1 000 | 0.235 | 1 | **0.0332** | — | 0.3500 | +79.4 K | **0.753 s** | 3136 |
| 2 000 | 0.470 | 1 | **0.0309** | — | 0.3397 | +28.0 K | 0.193 s | 4200 |
| **3 000** | **0.705** | 3 | **0.0088** | **[.0018–.0159]** | 0.3372 | +42.2 K | 0.076 s | 5122 |
| **4 000** | **0.940** | 3 | **0.0017** | [.0016–.0017] | 0.3792 | +67.7 K | 0.0112 s | 6227 |
| 5 000 | 1.174 | 3 | 0.0016 | [.0016–.0016] | 0.3788 | +67.4 K | 0.0057 s | 6973 |
| 6 000 (shipped) | 1.409 | 3 | 0.0016 | [.0016–.0017] | 0.3790 | +67.8 K | 0.0058 s | 5859 |
| 10 000 | 2.349 | 3 | 0.0016 | [.0016–.0016] | 0.3790 | +67.7 K | 0.0049 s | 16 191 |
| 19 999 | 4.698 | 3 | 0.0016 | [.0016–.0016] | 0.3792 | +67.8 K | 0.0033 s | 29 488 |

Reference: `L_void` 0.38116, margin +69.24 K. Rulers (§7.5.22): temperatures 1.1 to
1.6e-3, onset 0.009 s. Ratios are against the 17 029-parameter body (§7.5.37a);
`polish_colloc` is `n` and the sampler adds `n // 2` early-time points, so the rungs are
labelled by point count, which is the physical quantity.

**The axis is a threshold with a flat top**, and the transition occupies one rung rather
than falling between two. Below it the failure is an absence rather than a degradation:
the boiling margin is wrong in *both* directions (+79.4 K and +28.0 K against +69.2 K),
and the 1000-point arm misses the onset bar outright at 0.753 s against a 0.5 s tolerance
— the only arm on this ladder whose error the reference can resolve at all, at 83× the
ruler. Above it, six-fold more points move nothing outside the reference's own
uncertainty.

##### The transition rung is bistable, which is why one seed could not place it

**3000 points does not have an accuracy; it has two.** Per seed: 0.0018, 0.0159, 0.0088 —
an **8.8× spread**, against 1.06× at 4000 and 1.00× at 5000. The mean of 0.0088 describes
none of the three runs. What the seeds differ in is whether the front is found at all:
`L_void` averages 0.3372 against the reference's 0.3811, and the worst margin is +42.2 K
against +69.2 K, so the poor seeds are not less accurate versions of the good one — they
have a front in the wrong place.

That makes the rung a **statement about the initialisation**, not about the point count,
and it is the reason a single sample could not locate the edge. Seed 0 drew the good
basin and read 0.0018, which is indistinguishable from the converged shelf.

##### Where the edge is, and what this ladder cannot say about it

The transition is bracketed at $0.71 < r^* \le 0.94$: 3000 points is unreliable at three
seeds and 4000 is reliable at three. That is as far as the data goes, and the limit is
worth stating rather than reading through:

**0.940 sits 6% below the determined point, and the ladder steps by 1.33×.** So this
sweep cannot separate "the edge is at 0.94" from "the edge is at 1.0". Determinacy —
arXiv:2605.30910's prescription that residuals should exceed parameters — is *consistent*
with the bracket, and so is any other mechanism inside it, including the resolution
argument that roughly 3500 points in $(t, \zeta)$ is what a sub-cell interface moving
through the channel needs before the residual can see it. Distinguishing them needs rungs
between 3000 and 4000 at three seeds each, and a way to vary the two candidates
independently — which the body count now makes possible, since capacity can be changed
without touching the embedding (§7.5.37a).

Recorded as **unresolved**, with a stated bracket.

> **Three readings of this axis have now been withdrawn, and the sequence is the lesson.**
>
> 1. *"Absolute point count governs, determinacy does not"* — reasoned from f256 at 6000
>    points (then miscounted as ratio 0.48) against f64 at 2000 (0.32): two close ratios,
>    18× apart in accuracy. §7.5.37a corrected the denominator to 1.41 and 0.47, on
>    opposite sides of one, and the contradiction evaporated.
> 2. *"The threshold sits at the determined point"* — written while the 3000- and
>    4000-point rungs were still running, and refuted when they returned 0.0018 and 0.0017.
> 3. *"The edge is below the determined point, so determinacy is not the mechanism"* —
>    rested entirely on 3000 points reaching 0.0018 at **one seed**. Two more seeds put
>    that rung at 0.0159 and 0.0088; it is bistable, not converged, and the claim requires
>    it to be reliable.
>
> The third is the instructive one, because it was written *after* the rule against
> reading an axis mid-flight was added to `AGENTS.md`, and it obeyed the letter of that
> rule — every arm had returned. What it did instead was rest a conclusion on the one rung
> of eight that had a single sample, on the reasoning that a 17× step is far outside any
> seed spread. The step was; the rung's **position** was not, and the position was the
> whole claim. **The seed count has to be sufficient on the rung the conclusion turns on,
> not on the ladder as a whole.**

##### Above the threshold: nothing measurable, in either direction

6000 against 10 000 points, three seeds each. Same backend and same seeds throughout, so
unlike a cross-backend comparison these seeds are **legitimately paired** — identical
initialisations, one knob apart.

| | 6000 (1.41) | 10 000 (2.35) | change |
|---|---|---|---|
| `T_s` | 0.001641 [.001629–.001661] | 0.001631 [.001621–.001646] | **−0.7%** |
| `T_c` | 0.001700 [.001675–.001721] | 0.001693 [.001678–.001714] | −0.4% |
| `T_f` | 0.002560 [.002478–.002686] | 0.002578 [.002494–.002661] | +0.7% |
| `T_cl` | 0.003691 [.003568–.003851] | 0.003722 [.003587–.003846] | +0.9% |
| `L_void` | 0.3790 | 0.3790 | none |
| worst margin | +67.83 K | +67.74 K | −0.09 K |
| onset time error | 0.0058 s [.0010–.0095] | 0.0049 s [.0043–.0052] | −15% |
| sec (contended) | 5859 | 16 191 | +176% |

**Every one of those differences is below the resolution of the instrument.** `T_s`
improves in all three paired seeds — −0.81%, −0.86%, −0.31% — which suggests the effect is
real rather than noise, and it is also **thirty times smaller than the reference's own
uncertainty** of 1.1 to 1.6e-3 (§7.5.22, where the temperature bar's ratio is 1.06). The
same holds everywhere else: the onset errors sit at 0.005 s against a reference uncertain
by 0.009 s, and the void error at 3e-3 against a ruler of 3.2e-2. **Buying more
overdetermination than 1.41 buys nothing**, and the 20 000-point arms confirm it at 4.70:
`T_s` 0.0016 on all three seeds, `L_void` 0.3792, worst margin +67.8 K — the 6000-point
numbers to four digits, for **5× the wall-clock**.

**One signal is not flat: the seed spread.** Onset time scatters over 0.0010 to 0.0095 s at
6000 points and over 0.0043 to 0.0052 s at 10 000 — an **8.5× narrower** spread — and the
margin spread falls from 0.86 K to 0.22 K. More points averaging the sampling noise is the
obvious explanation and a mechanism that needs no new physics. At three seeds a variance
claim is weak evidence, so it is recorded as an observation to confirm, not a finding; but
it is the only column where the extra points bought anything at all.

##### Where the extra points still buy something: onset, not temperature

`T_s` saturates at **4000** points — every rung from there up sits at or below the ruler,
so they cannot be ordered. **Onset time can be**, and §7.5.22 already identified it as the
quantity with headroom left (ratio 56 against its bar). Three seeds per rung:

| points | onset error | against the 0.009 s ruler |
|---|---|---|
| 3 000 | 0.076 s | 8.4× — but the rung is bistable, so this is a mean of two regimes |
| 4 000 | 0.0112 s | **1.2× — still resolvable** |
| 5 000 | 0.0057 s | below |
| 20 000 | 0.0033 s | below |

So the smallest sufficient set **depends on the headline**: 4000 points if the claim is
about temperatures, **5000** if it is about onset timing. That is the one place on this
axis where a decision is still available, and onset is the axis §8 says the project should
be moving onto — so 5000 is the defensible default and 4000 is the floor.

**The shipped 6000 is 1.5× what even the onset claim needs**, and 20 000 costs 5× the
wall-clock of 6000 for nothing measurable on any field.

> An earlier revision of this subsection put the temperature floor at 3000 and quoted its
> onset error as 0.0129 s. Both were seed 0 of a bistable rung; at three seeds the rung
> does not converge at all. The floor is 4000.

##### The cheap direction is not cheap

1000 points cost 3136 s against 6000 points at 5859 s — 1.87× for a sixth of the data,
because 50 000 quasi-Newton iterations carry a per-iteration overhead that dominates once
the batch is small. Accuracy cannot be traded back for wall-clock by shrinking the set:
below the threshold the run costs most of what it cost before and returns a solution with
no front in the right place.

> The `sec` column is **not** clean and no conclusion rests on it. The rungs ran at loads
> from 5.4 to 34.5 as the ladder filled and drained — the 6000-point arms at 14.5, the
> 10 000 at 25–34.5 — so the wall-clock mixes the point count with contention. That is why
> 6000 appears *cheaper* than 5000 here, which it is not. Point count is a lower bound on
> relative cost; the measured figures are not attributable, and only the accuracy columns
> carry an argument.

Reproduced by, one arm per invocation — the rung is the `--only` token and the seed count
follows the table:

```bash
uv run python tools/axial_study.py adamcheck --seeds 0 --cpu-block 0 \
    --only 'f64 adam0/qn50000@3k-fixed' --out __DEV/studies/adamcheck_f64_3k_s0.json
```

Rungs are `@1k`, `@2k`, `@3k`, `@4k`, `@5k`, `@6k`, `@10k`, `@20k`, all suffixed
`-fixed`. Every rung from 3000 up carries **three seeds**; 1000 and 2000 carry one,
which is enough for them because they fail by 20× and the question there is only whether
they fail. Three seeds are not a formality on the transition rungs — they are what
revealed that 3000 is bistable, and a single sample there produced a conclusion that
survived two hours.

#### 7.5.39 The quasi-Newton budget, measured at last on the collocation count it should be

Every budget conclusion in this document was measured at the wrong point count. §7.5.11,
The `qnladder` sweep and §7.5.31 ran at 6000 points, which §7.5.38 shows is 1.5× more than even the
onset claim needs; §7.5.29's axis ran at 3000, which is the bistable rung. The two axes
have been confounded throughout, and this separates them: **`f64 adam0`, 5000 points, the
budget as the only knob, three seeds per rung.**

5000 points and not 4000 because 4000 is the temperature floor while 5000 is where onset
drops below its own ruler, and measuring a budget against a saturated metric answers
nothing.

| `lbfgs_iters` | `T_s` | range | spread | `L_void` | worst margin | onset err | sec | load |
|---|---|---|---|---|---|---|---|---|
| 10 000 | 0.0129 | [.0104–.0155] | 1.49× | 0.3420 | +49.3 K | 0.194 s | 1633 | 30.0 |
| 20 000 | 0.0030 | [.0026–.0033] | 1.29× | 0.3728 | +65.9 K | 0.034 s | 3207 | 29.4 |
| **30 000** | **0.0018** | [.0018–.0019] | 1.06× | 0.3781 | +67.7 K | 0.0165 s | 4827 | 29.7 |
| **40 000** | **0.0017** | [.0016–.0017] | 1.03× | 0.3786 | +67.8 K | 0.0092 s | 6399 | 32.9 |
| 50 000 | 0.0016 | [.0016–.0016] | 1.02× | 0.3788 | +67.4 K | 0.0057 s | 6973 | **9.6** |

**This axis saturates smoothly; it is not the cliff the collocation axis is.** `T_s` falls
4.3× from 10 000 to 20 000, then 1.7×, then 1.06×, then 1.06× — a monotone approach with
no rung where the solution is simply absent. Contrast §7.5.38, where one 1.33× step in
points moved the answer 17× and the rung below it was bistable. **The two axes fail
differently**, which is worth knowing before reading either: too few points and there is
no front, too few iterations and the front is there but imprecise. Every rung here from
20 000 up has a boiling front on every seed.

##### The budget also buys reproducibility, which no other axis in this document does

The seed spread contracts monotonically: **1.49× → 1.29× → 1.06× → 1.03× → 1.02×**. That
is the same 12.5×-spread problem §7.1 records, dissolving as a function of one knob. It
also says something about every under-funded arm on the shelf: a wide seed range at
`qn3000` is not evidence that the configuration is unstable, it is evidence that the
optimiser stopped early, and §7.5.31b's worry about ranking arms by single seeds is
sharpest exactly where the budget is smallest.

##### Cost is exactly linear, and this is the one clean timing in the document

| `lbfgs_iters` | sec | per iteration |
|---|---|---|
| 10 000 | 1633 | 163.3 ms |
| 20 000 | 3207 | 160.3 ms |
| 30 000 | 4827 | 160.9 ms |
| 40 000 | 6399 | 160.0 ms |

**160 ms per iteration, constant to 2% across a 4× range in budget.** These four rungs ran
concurrently at loads 29.4 to 32.9 — matched, by construction, because the arms were
packed two-deep per CPU block so that long and short jobs shared the machine rather than
following one another onto an emptying one. §7.3.2's rule is usually invoked to void a
timing comparison; here it is satisfied, and the linearity is a real measurement rather
than a hope.

> **The 50 000 row's 6973 s is not part of that.** It was measured in §7.5.38 at **load
> 9.6**, a third of the others, so it appears cheaper than 40 000 while doing 25% more
> work. At the 160.5 ms/iteration these four establish, 50 000 iterations cost **~8000 s**
> at matched load. The accuracy column is unaffected.

##### What to fund: more is simply better, and the question is only what is demonstrable

**There is no trade-off on this axis.** `qn50000` is weakly better than every rung below
it on every column — `T_s`, `L_void`, onset, and seed spread — so nothing is given up by
funding it. An earlier revision of this section presented the choice as splitting by
claim, "`qn30000` for temperatures and `qn50000` for onset", which implied `qn30000` is
*better for temperatures*. It is not. It is **cheaper and indistinguishable**.

The real question is where the extra 1.66× buys something the reference can resolve:

| | `T_s` vs its 1.1–1.6e-3 ruler | onset vs its 0.009 s ruler |
|---|---|---|
| `qn30000` | 0.0018 — **at the ruler** | 0.0165 s — 1.8×, resolvable |
| `qn40000` | 0.0017 — at the ruler | 0.0092 s — 1.0×, marginal |
| `qn50000` | 0.0016 — at the ruler | 0.0057 s — **below the ruler** |

So `qn30000` and `qn50000` cannot be distinguished on temperatures **by this reference**,
and can be on onset. That makes `qn30000` a defensible economy for a temperature-only
claim — 4827 s against ~8000 s, buying nothing demonstrable — and `qn50000` the right
default, because it is better everywhere and provably better where the ruler still has
room.

| | measured | cost |
|---|---|---|
| **default** | 5000 points, `qn50000` | ~8000 s, load-corrected |
| economy, temperature claims only | 5000 points, `qn30000` | **4827 s**, measured |

> **Both rows hold the point count at 5000, and that is deliberate.** The measured grid is
> an **L**, not a rectangle: every point count was run at `qn50000`, and every budget was
> run at 5000 points. The corner where both are dialled down — 4000 points *and*
> `qn30000` — **has never been run**, and it is exactly where an interaction would show,
> since each knob is one rung above its own transition (§7.5.38's 4000-point spread is
> 1.06×, and `qn30000`'s is 1.06× here). A revision of this section quoted that corner at
> "~3900 s" from a linear cost model. No study row contains it; the claim is withdrawn and
> the recommendation stays on the measured cell.

So §7.5.37's `qn50000` was the right call, and the shipped `qn30000` is not wrong so much
as **unfalsifiable against this reference** on the quantity the paper currently leads
with. The 1.66× buys accuracy that only becomes visible on onset — which is the same
conclusion the collocation axis reached from the other direction. The two now agree:
**temperatures are done, and every remaining decision on this model is an onset
decision.**

That also says what the next reference-side work is worth. A refined reference would move
the temperature ruler below 1.1e-3 and make these three rungs orderable again; until then,
any budget from 30 000 up is a cost decision and not an accuracy one.

Reproduced by, one arm per invocation:

```bash
uv run python tools/axial_study.py adamcheck --seeds 0 --cpu-block 0 \
    --only 'f64 adam0/qn30000@5k-fixed' --out __DEV/studies/adamcheck_f64_qn30000_5k_s0.json
```

#### 7.5.40 Schedule-free AdamW: the first-order stage fails with no schedule to blame

Every "is the first-order stage enough?" measurement in this document ran Adam under our
own cosine decay (`optax.cosine_decay_schedule`, `alpha = 0.1`). So each of them — §7.5.11,
the `qnladder` sweep, §7.5.29, §7.5.34 — has been open to the same objection: *Adam did not plateau,
our schedule ran out.* The objection is reasonable and it has never been tested.

**Schedule-free AdamW** (arXiv:2405.15682, `optax.contrib.schedule_free_adamw`) is the
version of the method whose entire claim is that no schedule is needed. It **replaces**
the cosine rather than composing with it — a constant `lr = 1e-3`, its own warmup at 10%
of the budget, `weight_decay = 0.0` so the only difference from Adam is the schedule-free
averaging. Seed 0, 5000 points, f64, JAX only (this lives in `optax.contrib`, the same
standing divergence as `ademamix`).

| arm | `T_s` | `L_void` | worst margin | max `α` | sec | load |
|---|---|---|---|---|---|---|
| sf-AdamW 30 000, **no polish** | **0.0603** | 0.1057 | **−0.2 K** | 0.857 | 4201 | 10.6 |
| sf-AdamW 10 000 → `qn30000` | **0.0245** | 0.2666 | +39.9 K | 1.000 | 4627 | 4.9 |
| `qn30000` alone (§7.5.39) | **0.0018** | 0.3781 | +67.7 K | 1.000 | 4827 | 29.7 |

**Alone it does not produce the transient at all.** The margin is *negative*: the channel
never reaches saturation, so there is no boiling front, `L_void` is 0.106 against the
reference's 0.381, and the onset error is undefined because there is no onset. This is not
a less accurate solution of the problem — it is a solution with the phenomenon missing.
33× worse on `T_s` than the quasi-Newton stage at the same 30 000 iterations.

**And as a warm start it makes the polish 13.6× worse**, while spending 40 000 iterations
against the winner's 30 000. It recovers a front, but a badly placed one: `L_void` 0.267
and a margin of +39.9 K against the reference's +69.2 K.

##### What this closes

The schedule objection is answered, and answered in the direction the existing results
already pointed. §7.5.34's "removing the Adam stage makes the model better" was open to
being read as a statement about *our* Adam — our decay, our learning rate, our schedule
ending at the wrong moment. A first-order stage **with no schedule to end** does the same
damage. The first-order stage is not being mis-scheduled; it moves the parameters
somewhere the quasi-Newton stage is worse off starting from.

> **Two caveats, both stated because they cut against the conclusion.** These are single
> seeds, and both schedule-free arms ran on a nearly idle machine — loads 4.9 and 10.6
> against the control's 29.7 — so their wall-clocks *understate* their true cost and the
> comparison is more favourable to them than reality. Neither changes the reading: a
> negative boiling margin is not a seed effect, and a 33× gap is not a timing artefact.

**A note on cost, because it bears on §7.5.31a.** A schedule-free step runs at ~140 ms
against L-BFGS's 160 ms at the same 5000 points — within 20%. The literature's argument
for first-order methods is that their steps are 20–45× cheaper, which depends on
minibatching; here **both optimisers evaluate the same 5000 residuals per step**, so the
30 000-against-30 000 comparison is close to equal wall-clock as well as equal iterations.

The reported iterate is the averaged `x`, never the optimiser's `y`. That distinction is
the standard way to get a wrong number out of this family — it trains, it converges, and
nothing indicates the wrong sequence was read — so the conversion happens before anything
downstream sees the model, including the polish, and
`tests/axial/test_schedule_free.py::test_the_polish_starts_from_x_not_y` pins it.

#### 7.5.41 Freezing the encoder part-way: 2.5x worse, and 22% faster

§7.5.32 measured the all-or-nothing freeze and found it worse. This asks whether the
encoder's work is merely **front-loaded**: 10 000 iterations with the Fourier read-out
trainable, then held for the remaining 40 000, on the same fixed 5000-point set. If the
representation settles early, the late iterations are carrying a curvature space that has
stopped moving.

One solve per seed, scored at three budgets by `polish_checkpoints` — the optimiser state
is carried across each stop, so these are the trajectory the run took rather than three
independent short runs. Three seeds:

| | `qn30000` | `qn40000` | `qn50000` | seed range at 50k | sec |
|---|---|---|---|---|---|
| **f64 frozen** | 0.0078 | 0.0051 | **0.0041** | [.0027–.0050] | 6592 |
| f64 **unfrozen** (§7.5.39) | 0.0018 | 0.0017 | **0.0016** | [.0016–.0016] | 6973 |
| **f256 frozen** | 0.0052 | 0.0045 | **0.0038** | [.0024–.0063] | 11 882 |
| f256 unfrozen | — | — | — | | *running* |

**At f64 the answer is no: freezing costs 2.5×**, and the gap narrows with budget
(4.3× → 3.0× → 2.5×) without closing. The frozen runs are still converging at 50 000 while
the unfrozen ones finished by 30 000, and their seed spread stays wide — 0.0027 to 0.0050
against an unfrozen arm that is 0.0016 on all three seeds. The encoder's work is not
front-loaded; ten thousand iterations of freedom is not enough for it.

##### It is genuinely faster, and that is still not a reason to do it

Backing out the 10 000 free iterations at the known unfrozen rate, **the frozen iterations
run at 124.8 ms against 160.0 ms — a 22% saving**, at comparable load (27.5 against 29.7
and 32.9). That is far more than the arithmetic predicts: the two-loop recursion is ~1% of
an iteration, and the saving is better explained by memory traffic, the L-BFGS history
falling from 20 MB to 13.6 MB at f64 with six processes competing for bandwidth.

It buys nothing usable. 22% cheaper per iteration for 2.5× the error is a bad trade, and
at equal wall-clock it is worse still: unfrozen `qn40000` costs *less* than frozen
`qn50000` and is 3× more accurate.

##### The conditioning hypothesis is refuted — the control landed

Freezing barely changes fitting capacity, 17 029 to 16 965 (§7.5.37a), so any *gain* has to
come from the **curvature dimension**, and that falls by different factors at the two
widths. The prediction is sharp and one-sided: a conditioning effect must be **larger where
the saving is larger**.

With the unfrozen control now measured at both widths, three seeds each, same 5000 points:

| | frozen | unfrozen | penalty | curvature dimension |
|---|---|---|---|---|
| f64 | 0.0041 | 0.0016 | **2.53×** | 25 221 → 16 965 (**1.5×**) |
| f256 | 0.0038 | 0.0016 | **2.38×** | 49 797 → 16 965 (**2.9×**) |

**The penalties are the same to 6% while the curvature saving differs by two.** Freezing
costs as much where it saves most, which is what the hypothesis forbids. It is refuted, not
merely unsupported, and the difference matters: the earlier revision of this subsection
could only say the two *frozen* arms coincided, which is compatible with conditioning if
f256 unfrozen were worse. It is not — f256 unfrozen is 0.0016, exactly where f64 is.

What remains is the reading §7.5.32 already had: freezing removes work the encoder is still
doing. The curvature space it saves is not the mechanism, at either width.

#### 7.5.42 The embedding width, measured at last on the recommended configuration

Every rung of §7.5.31's ladder ran at 6000 points under a 10 000-iteration Adam stage, and
§7.5.37a showed that ladder never varied capacity at all. This is the width axis at the
configuration the project now recommends — `adam0`, a fixed 5000-point set, a funded
polish — across three budgets and three seeds, 27 runs.

**Fitting capacity is 17 029 at every rung.** What changes is the read-out: 8192 at f64,
16 384 at f128, 32 768 at f256.

| budget | width | fuel | clad | film | coolant | `L_void` | worst margin | onset | onset spread |
|---|---|---|---|---|---|---|---|---|---|
| qn30000 | f64 | 0.0024 | 0.0035 | 0.0018 | 0.0019 | 99.2% | **+67.7 K** | 0.0165 s (1.8×) | **15.7×** |
| | f128 | 0.0024 | 0.0035 | 0.0017 | 0.0018 | 99.4% | +66.9 K | 0.0101 s (1.1×) | 2.0× |
| | f256 | 0.0025 | 0.0036 | 0.0017 | 0.0017 | 99.4% | +66.7 K | **0.0091 s (1.0×)** | 7.0× |
| qn40000 | f64 | 0.0025 | 0.0036 | 0.0017 | 0.0017 | 99.3% | **+67.8 K** | 0.0092 s | 1.5× |
| | f128 | 0.0025 | 0.0037 | 0.0016 | 0.0017 | 99.4% | +66.2 K | **0.0042 s** | 3.7× |
| | f256 | 0.0026 | 0.0038 | 0.0016 | 0.0017 | 99.4% | +67.2 K | 0.0050 s | 4.6× |
| **qn50000** | **f64** | 0.0026 | 0.0037 | 0.0016 | 0.0017 | 99.4% | +67.4 K | 0.0057 s | 1.9× |
| | f128 | 0.0026 | 0.0037 | 0.0016 | 0.0017 | 99.4% | +66.3 K | **0.0026 s** | 3.2× |
| | f256 | 0.0026 | 0.0038 | 0.0016 | 0.0017 | 99.4% | +67.2 K | 0.0033 s | 12.6× |

Rulers: temperatures 1.1 to 1.6e-3, onset 0.009 s (the bracketed multiples), reference
`L_void` 0.38116 and margin +69.24 K.

##### At the recommended budget the three widths are indistinguishable

At `qn50000` every width gives `T_s` 0.0016, `L_void` **99.4%**, and an onset error
**below the reference's own uncertainty**. Onset *height* error is exactly 0.0000 in all
nine runs at that budget — which is the tautology §5.2 of the paper records rather than a
result: the coolant heats monotonically, so boiling always begins at the outlet.

To five decimals a pattern does appear, and it is a **trade rather than an improvement**:

| | film | coolant | fuel | clad |
|---|---|---|---|---|
| f64 | 0.00163 | 0.00169 | **0.00258** | **0.00372** |
| f128 | 0.00162 | 0.00168 | **0.00257** | **0.00371** |
| f256 | **0.00161** | **0.00167** | 0.00263 | 0.00380 |

**f256 buys film and coolant accuracy by giving up fuel and cladding**, and the ordering
holds at all three budgets, so it is not a seed effect. f128 is the only width at-or-better
than f64 on all four. All of it sits below the temperature ruler, so none of it is
demonstrable — a consistent direction, not a measurable difference.

##### Where width does buy something: the starved budget

At `qn30000` the onset differences are **at or above the ruler** and therefore real: 1.8×
for f64 against 1.1× and 1.0×. More striking is the spread — f64's onset scatters
**15.7×** across three seeds there, against f128's 2.0×. The narrow embedding is not
merely less accurate at a starved budget, it is *unstable*, and that is the honest argument
for widening.

By `qn40000` the argument is gone: f64 reaches the ruler exactly (1.0×) and every width is
below it at `qn50000`.

##### Cost, and why the raw seconds must not be ranked

| | ms per iteration | at load | normalised |
|---|---|---|---|
| f64 | 139.5 | **9.6** | 1.00 |
| f128 | 200.1 | 29.8 | ~1.25 |
| f256 | 277.6 | 14.6 | ~1.9 |

The f64 arms ran on a machine three times quieter than the f128 arms, so the wall-clocks
are not comparable as measured. Normalising through §7.5.39's anchor — f64 at 160.5 ms at
load ~30 — gives roughly **1 : 1.25 : 1.9**, which tracks the read-out width rather than
anything the network can represent.

##### The default stays f64

At the recommended `qn50000` the widths cannot be told apart by this reference, and f64 is
**about half the cost of f256**. Widening is worth considering only at a starved budget,
where it buys onset accuracy and, more importantly, onset *stability* — and the project's
answer to a starved budget is to fund it, not to widen the embedding.

### 7.6 Pseudo-time stepping

Implemented (`pts_every`, `pts_dtau`, `pts_growth`), and **measured harmful** in
the §7.2.5 re-ablation — it is off by default and no further work is planned. An
earlier ablation run was killed before finishing, which is why an interim version
of this section recorded the accuracy as TBD.

### 7.7 GPU timing

**Not a goal, rather than a gap.** CPU is the target: these networks are far too
small to saturate a device, and the float64 the problem needs is throttled to
roughly 1/32–1/64 of FP32 on consumer NVIDIA hardware. The axial model has never
been benchmarked on a GPU and there is no plan to. What *does* need pinning before
any timing here is quoted is `OMP_NUM_THREADS` — the default is every core, so two
concurrent runs oversubscribe, and thread count changes float reduction order. With
it pinned the torch backend reproduces run to run to four digits, which is what made
the post-refactor regression check in §4 meaningful. §7.3.2 states which of its
wall-clocks were contended and which were not, for exactly this reason.

The one timing question that *is* open is a CPU one: the 2.4× JAX advantage in
§7.3.2, which remains unattributed.

### 7.8 What M7 did deliver

The JAX twin itself (`axial/pinn_jax.py`), sharing the residual functions with the
torch backend and satisfying every hard constraint exactly. It has already earned
its cost twice: it showed the pre-fix failure was *not* backend-specific, which
implicated the formulation; and the post-fix divergence exposed the frozen
collocation bug. Both are findings a single backend could not have produced.

### 7.9 What is still open

| topic | status |
|---|---|
| Fourier + modified MLP combined | **measured, and it fails** — §7.2.6 |
| Plan A, multiple seeds | **TBD** — one seed measured (§7.4) |
| Backend parity, post-closure | **closed, for a third time and for the right reason** — §7.5.17. The gap was `optax.lbfgs`'s default `memory_size=10` against torch's 50. At matched memory the two implementations agree to 2–3% on an identical objective, and JAX f512 is 1.08× torch rather than 1.44× |
| The JAX speed advantage | **reversed once PyTorch compiles** — §0.6, §7.3.2. The 4.4× of §7.5.19 was against an eager torch loop; with `compile=True` torch is 1.78×–1.96× faster at matched settings, four runs, non-overlapping ranges. The old attribution ("`torch.compile` accounts for 1.06× and is not the answer") measured eight graph breaks, not the technique |
| Optimiser bake-off (SSBroyden / SSBFGS) | **TBD — not started**, and §7.5.11 makes it the highest-value remaining item: at three seeds on both backends the quasi-Newton axis is the *only* one that moves the front |
| How many epochs it needs | **answered — 54 runs, nine cells, three seeds, both backends** — §7.5.11. Quasi-Newton monotone over two decades; the Adam axis flat once `qn3000` is set, so the default's Adam budget does no measurable work |
| Three front-aimed embeddings | **measured, three seeds** — §7.5.12–§7.5.14. `fourier_bands=(1,4,16)` reaches 99.5% of the reference voided length; `zeta_scale=8` is real but cruder; `level_set_input` is inert at 1.95× the cost |
| The Laplace embedding | **measured, and it fails** — §7.5.18. No mode beats the control on either backend at three seeds; the ansatz is already multiplicative, so it could only make a decaying mode easier and does not |
| How much collocation goes to the front | **measured, and it fails** — §7.5.9. `T_s` degrades monotonically from 0% to 50%; re-weighting the measure is not the remedy |
| M4 acceptance: onset within 0.5 s and one cell | **the time half is met at three seeds** — §7.5.16a: 0.0006 / 0.0064 / 0.0181 s against 0.5 s at the funded default, root-found rather than read off the 0.25 s grid. The 0.62–0.84 s figure was measured at `qn3000` and is superseded. The height answer is "the outlet" whatever the network does, because `T_c` is monotone in `ζ`, so **M4 as written no longer discriminates between formulations** |
| Is M4's criterion sound? | **yes — measured, §7.5.21.** At the scoring mesh the reference's own onset is uncertain by 0.06 cells and 0.009 s, so the criterion sits an order of magnitude above the ruler. The failure is the network's and the target is worth chasing |
| Is Adam needed at all? | **answered: no, and it is harmful.** `adam0` beats `adam10000` by 2.8× (§7.5.34), the best arm in this document has no first-order stage (§7.5.37), and the objection that our *schedule* was at fault is closed by §7.5.40 — schedule-free AdamW forms no boiling front at all and degrades the polish 13.6× as a warm start |
| Where the quasi-Newton axis ends | **not measured** — §7.5.11 is monotone over two decades with no interior optimum, and by §7.5.8's own rule an unterminated monotone trend is an extrapolation. Kiyani et al. run 30000 quasi-Newton iterations against this model's 3000 |
| Where the memory optimum sits at the real budget | **not measured** — §7.5.17a. The iso-time optimum is 100 at a 200 s budget; the recipe spends ~550 s and the crossover moves with the budget |
| The 1% bar on temperatures | **not met** — see §7.2.5 for the current figures |

## 8. What to do next

**The thirteen remedies now make a pattern, and it is sharper than any of them.**
Sorting every isolated arm by what it changed:

| what it changed | outcome |
|---|---|
| **the function space** — Fourier capacity (§7.5.8), multi-scale bands (§7.5.14), anisotropic bandwidth (§7.5.12) | **having an embedding is decisive; its size is not.** Without one, no optimiser forms a front (§7.5.29). With one, f32 matches f256 at 42% of the cost — the capacity ladder is flat at a funded budget (§7.5.31). Bands and anisotropic bandwidth helped *only* at a starved one (§7.5.24) |
| **the optimiser** — quasi-Newton budget (§7.5.11), curvature memory (§7.5.17) | **decisive**; the only axis that forms the front at all |
| **the first-order stage** — Adam, AdEMAMix (§7.5.29), schedule-free AdamW (§7.5.40) | **harmful, and not cheaper.** `adam0` beats `adam10000` by 2.8× (§7.5.34); Adam-only is 19× worse; per iteration L-BFGS is the *cheapest* of the three (99.3 ms against 107.8 and 115.5). Removing the schedule does not rescue it: schedule-free AdamW produces **no front**, and warm-starting from it costs 13.6× |
| **the loss measure** — level-set sampling (§7.5.6), front fraction (§7.5.9), block and causal weighting | **failed or inert**; `frontfrac` degrades monotonically |
| **extra residuals** — onset head (§7.5.16), front network, pseudo-time | **harmful**; a consequence of the PDE carries no information as a constraint |
| **re-parameterisation of the *output*** — level-set coordinate (§7.5.13), Laplace embedding (§7.5.18) | **inert**; both are functions of things the network already computes, so neither adds information. **Not** the same as a trainable change of *input* coordinate, which is a change of function space and is untested here — see the narrowing below |

**Change the function space or change the optimiser. Do not reweight the loss, and
do not add residuals the PDE already implies.** That is not a hunch — it is thirteen
measurements, and it is what the roadmap below is ordered against. The full version,
with the 2026 literature it draws on, is `__DEV/REPORT-01-MILESTONES.md` Annex D, and the
revised plan after a five-topic literature sweep is Annex E.

> **The rule has since been narrowed, and it matters.** §7.5.13 fed the network a
> coordinate built from its own *output* and measured it inert; that was generalised to
> "re-parameterisations add no information". The model-order-reduction literature says the
> object with theory behind it is different: for transport-dominated problems with moving
> discontinuities every fixed linear basis has slowly decaying Kolmogorov n-width, and the
> published remedy is a registration map on the **input** coordinate, `x − s(t)` with `s`
> trainable — not a feature derived from the output. Only one of those was tested. **A
> function of the network's own output adds nothing; a trainable change of input coordinate
> is a change of function space**, and belongs in the row above rather than this one.


1. **Spend on the quasi-Newton axis, and find out where it ends** — §7.5.11. The
   completed 54-run surface says the Adam budget does no measurable work and the
   quasi-Newton budget does all of it. Two cells of that statement are **not
   measured** and both are cheap:

   - **`adam = 0`.** The grid's floor is 30. "30 is as good as 3000" is measured;
     "Adam is unnecessary" is not. A short warm start may still matter, because
     quasi-Newton from a random init with an identity inverse-Hessian can take a bad
     first step — and 30 iterations is cheap insurance. But it has never been tested.
   - **`qn > 3000`.** The axis is **monotone with no interior optimum in the range
     swept**, and this document's own rule — applied to the Fourier ladder in §7.5.8
     — is that a monotone trend with no measured end is an untested extrapolation.
     Kiyani et al. run Adam[1000] + SSBroyden[**30000**], a 1:30 ratio; this model
     runs 300 + 3000 and the 300 is doing nothing.

   If the axis keeps improving, the current `T_s` of 0.0216 is a **budget** limit
   rather than the model's, and the 1% bar may be reachable by spending on the right
   axis instead of by adding a method.

   **Why quasi-Newton should dominate here is not mysterious.** Adam is built for
   noisy stochastic gradients over large datasets; this loss is full-batch and
   essentially deterministic, so Adam's variance machinery buys nothing and what
   remains is a *diagonal* preconditioner. Meanwhile the residual contains
   derivatives of the network, and differentiating amplifies exactly the
   high-frequency content spectral bias suppresses — so the Hessian spectrum spans
   orders of magnitude with strong off-diagonal structure, which a diagonal scaling
   cannot touch. That is Rathore et al.'s loss-landscape argument, and it predicts
   the surface measured here.

2. **The optimiser bake-off** — §7.5. Follows directly: SSBroyden/SSBFGS drop into
   the quasi-Newton slot with no new machinery, and §7.5.11 has just established
   that this is the slot that matters.
3. ~~**Explain the 21% on `T_s` and `T_c`.**~~ **Done, then reopened — §7.5.10.**
   §7.3.2 found the L-BFGS implementation: 1.168 with each framework's own, 0.999
   with one shared. At f512 the shared optimiser closes only part of it and 1.73×
   remains, so that answer was configuration-bound like its own caveat warned.
4. ~~**Converge the ruler in `α`, then re-derive the acceptance bar.**~~ **Done —
   §6.5.** The bar stands at 1% for the temperatures, where the ruler is 1.1–1.6e-3,
   and for `L_void`, where it is 0.57%. The pointwise `α` field cannot carry a 1%
   bar (3.15e-2) and is not scored on one. This removed an excuse rather than a
   problem: the temperature failure is 45–120× the ruler.
5. **Plan A at more than one seed** — §7.4. Given §7.1's 12.5× seed spread, the
   single Plan A measurement is an observation and is labelled as one.

6. **The weak form** — `__DEV/…` Annex D.3, and the highest-value item beyond the
   budget questions. It is the only candidate that addresses Annex C's measure bug
   *without* reweighting: integrating the residual against locally supported test
   functions makes each test function a **separate equation**, so the front stops
   competing for a share of one average and gets its own rows. Integration by parts
   also lowers the derivative order, which attacks the ill-conditioning from the
   other end. `hp`-VPINN and Petrov–Galerkin VPINN are the current forms, and a
   boiling front in a 590 K field is a singular perturbation, which is the regime
   they are written for.
7. **Gauss-Newton / energy natural gradient, preconditioned** — Annex D.4. The
   principled version of what L-BFGS approximates, now affordable via randomized
   Nyström preconditioning of the Gramian. Follows the bake-off rather than
   replacing it: SSBroyden sets the bar it has to clear.
8. **Least squares on frozen random features** — Annex D.5. The limit of "Adam
   contributes nothing and the linear algebra contributes everything": freeze the
   hidden layer and the residual is linear in the output weights. This model already
   uses random Fourier features, so it is closer than it sounds — the obstacle is
   that our residual is *not* linear in the output (log Doppler, cubed `tanh`
   closure), so it needs an outer linearisation, which is Gauss-Newton again.
9. **Shock fitting and partition-of-unity decomposition, borrowed from CFD** —
   Annex D.6. Shock fitting makes the front an **unknown of the formulation** rather
   than a residual bolted onto a fixed parameterisation, which is exactly where
   §7.5.16's onset head went wrong. FBPINN's reported mechanism — localisation turns
   global high frequencies into local low ones — is §7.5.14's band result stated in
   space rather than in frequency, which argues both that it will work and that the
   two may be redundant.

Items 1–5 are measurable with the code as it stands and need only compute. Items
6–9 need real development, and are ordered so that the cheap answers arrive first —
if the quasi-Newton budget alone closes the gap, most of the rest is unnecessary. No accuracy number from
this model should be quoted outside this document, and none is.
