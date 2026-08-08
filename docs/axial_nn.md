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
`default`, `scaling`, and `margin` through f512. Running: `levelset` and the f1024
rung.

### 0.1 Where the accuracy stands

| configuration | `T_f` | `T_s` | `L_void` | worst-seed margin | front |
|---|---|---|---|---|---|
| **shipped default** (8k/500) | 0.1243 | 0.0434 | 0.0367 | **−1.1 K** | **on no seed** |
| published-table budget (3k/300) | 0.1386 | 0.0765 | 0.1529 | +12.5 K | every seed |
| f128 + quasi-Newton budget | 0.1024 | 0.0314 | 0.2424 | +17.5 K | every seed |
| **best known — f512** | **0.0710** | **0.0216** | **0.3012** | **+34.6 K** | **every seed** |
| reference | — | — | 0.3812 | — | — |
| **acceptance bar** | 0.01 | 0.01 | — | — | — |

**The bar is missed by 2.2×**, down from 4.3× at the shipped default. `L_void` is
at **79% of the reference**, against the default's 10%. The best single run reaches
`T_s` **0.0148** — 1.5× off the bar — but the f512 seed range is 1.71×, so the worst
seed is the number to quote.

**The shipped default forms no boiling front on any seed of either backend**
(§7.2.9). That is the repository failing to produce its own headline result, and it
outranks every accuracy figure here.

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
| **M4 acceptance** | onset within 0.5 s and one cell — **not met**; both criteria are now known to be measurable (§6.5) |
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

### 0.6 Method notes that changed the answers

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

**The cause is not compilation, contrary to what this section previously claimed.**
The obvious explanation was that `eqx.filter_jit` compiles the whole step while
torch runs eager. Measured, it is wrong. `torch.compile` on the torch step buys
**1.06×**, at 17 s of compile time — for a 3000-iteration run that is ~19 s saved
against 17 s spent, so it does not earn its place and is not adopted.

The profile says why. Of a 211 ms torch step: forward 116 ms, backward 71 ms,
optimiser 25 ms. **88% is forward-plus-backward through `torch.func.jvp` in
float64** — dense BLAS-bound linear algebra, which Inductor cannot improve. The
optimiser is 12%, and `foreach` is indistinguishable from noise here (198.5 ms
against 199.9 ms with it off) because the model has 12 parameter tensors and
17k parameters, so there is almost nothing to fuse.

**The 2.4× is therefore unattributed.** The remaining candidates are how XLA fuses
`vmap`-of-`jvp` against `torch.func.jvp`, and the float64 CPU kernels each stack
dispatches to. Neither has been measured. **TBD** — and until it is, the number
should be reported as an observation, not explained.

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
| Backend parity, post-closure | **closed** — §7.3.2. The gap is the framework L-BFGS: 1.168 with each framework's own, **0.999** with one shared implementation, three seeds each. M7's criterion is met with `optimizer = "lbfgs-shared"` |
| The 2.4× JAX speed advantage | **unattributed** — §7.3.2. `torch.compile` accounts for 1.06× of it and is not the answer |
| Optimiser bake-off (SSBroyden / SSBFGS) | **TBD — not started**, and §7.3.4 makes it the highest-value remaining item |
| GPU timing | **not a goal** — §7.7. CPU is the target |
| Pseudo-time stepping accuracy | measured harmful (§7.2.5); no further work planned |
| M4 acceptance: onset within 0.5 s and one cell | **not met.** The front now forms, but onset is late. Under D-TH-3 the front is the level set `T_c = T_sat + ΔT_sup`, so this is bounded by `T_c` accuracy |
| The 1% bar on temperatures | **not met** — see §7.2.5 for the current figures |

## 8. What to do next

In order, and none of it is "add another method". Nine remedies have now been
argued soundly and refuted by measurement (§5.4, §7.2.1, §7.2.5, §7.2.6); the tenth
would not be different.

1. **The optimiser bake-off** — §7.5. §7.3.4 measured that L-BFGS is not polishing
   anything here: it is the step that forms the boiling front, and Adam alone leaves
   `max α = 0` in both backends. That makes the quasi-Newton stage the most
   load-bearing component of the recipe and the least examined. SSBroyden/SSBFGS
   drop into the same slot with no new machinery.
2. ~~**Explain the 21% on `T_s` and `T_c`.**~~ **Done — §7.3.2.** It is the L-BFGS
   implementation: 1.168 with each framework's own, 0.999 with one shared. As with
   every previous backend disagreement in this document, it was bug-shaped and it
   was a bug — this time in a dependency rather than in this code.
3. ~~**Converge the ruler in `α`, then re-derive the acceptance bar.**~~ **Done —
   §6.5.** The bar stands at 1% for the temperatures, where the ruler is 1.1–1.6e-3,
   and for `L_void`, where it is 0.57%. The pointwise `α` field cannot carry a 1%
   bar (3.15e-2) and is not scored on one. This removed an excuse rather than a
   problem: the temperature failure is 45–120× the ruler.
4. **Plan A at more than one seed** — §7.4. Given §7.1's 12.5× seed spread, the
   single Plan A measurement is an observation and is labelled as one.

Everything above is measurable with the code as it stands; none of it needs new
development, only compute and a converged void reference. No accuracy number from
this model should be quoted outside this document, and none is.
