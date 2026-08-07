# Axial PINN — formulation, recipe, and measured results

The neural-network side of the 1D axial boiling model. The physics, and every
deviation from the SAS4A/SASSYS-1 manual, are in
[`axial_physics.md`](axial_physics.md); this document covers the ansatz, the
training recipe, the two backends, and **what has actually been measured** —
including the results that came out badly, which are most of them.

> **Status.** The network trains, satisfies every hard constraint exactly, and
> does **not** meet the 1% bar. Current Plan B: `T_f` 0.137, `T_cl` 0.189,
> `T_s` 0.075, `T_c` 0.075 (§7.2.5); current Plan A: 0.250 relative `L2` on
> `P(t)` (§7.5). The boiling front does now form, `max α = 1.0000` against the
> reference's 1.0000, since the void was eliminated algebraically (§7.2.3–§7.2.4).
> None of this is presented as a working result.

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
> in §5 and §7.1–§7.2 was measured on.

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
| `t_train_frac` | `1.0` | set below 1 for Plan B, whose validity window ends before `t_end`; §7.2.4 |
| `front_net` | `False` | front-position network, measured worse on every metric |
| `n_windows` | `1` (off) | neutral in the re-ablation; §7.2.5 |
| `fourier_features` | `0` (off) | **now measured better** (−11.1% at 3 seeds); not adopted — see below |
| `modified_mlp` | `False` (off) | **now measured better** (−16.1% at 3 seeds); not adopted — see below |
| `pts_every` | `0` (off) | pseudo-time stepping measured harmful; §7.2.5 |

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

## 5. Measured results

> **Provenance (Annex A, N8).** Every number in §5 and §7.1–§7.2 was measured on
> the **pre-D-TH-3 formulation**: the void solved as a differential unknown, block
> weights unbounded, no per-block residual scaling, and the full 60 s horizon.
> All four have since changed. Treat §5 and §7.1–§7.2 as a record of how the
> model got here, not as current accuracy. The current numbers are §7.2.3
> (variable scaling), §7.2.4 (the phantom void), §7.2.5 (the re-ablation, which
> **retracts half of D38**) and §7.5 (Plan A). Where a §5 conclusion has been
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

### 5.3 Budget — **non-monotonic**

| Adam iters | T_f | T_cl | T_s | T_c |
|---|---|---|---|---|
| 3000 | **0.062** | 0.120 | **0.032** | 0.034 |
| 8000 | 0.246 | 0.095 | 0.075 | 0.028 |

`T_f` gets **4× worse** with more training. Non-monotonic in budget means the
optimiser wanders between minima rather than converging slowly, so more iterations
will not fix this.

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

### 6.4 The reference is not converged in `α`

| reference `n = 40` vs `n = 320` | T_f | T_cl | T_c | **alpha** |
|---|---|---|---|---|
| relative `L2` | 5.1e-3 | 7.2e-3 | 5.4e-3 | **1.03e-1** |

**The reference the PINN has been scored against is itself ~10% wrong in the void
fraction**, because it is first-order upwind at `n_axial = 40` and the front spans
2–6 cells. The temperatures are converged to ~5e-3 — which sits right at the 1e-2
acceptance bar, leaving almost no headroom.

Two consequences:

* `L_void_max_err_m = 0.391 m`, a number that did not move across *any*
  configuration tried, was never the PINN's error to fix.
* The M2 convergence study measured orders on the **non-boiling** case, so the void
  field's non-convergence went unmeasured. That is a gap in the test design, not in
  the solver: everything `axial_physics.md` §5 claims about the reference — exact
  steady state, energy conservation, convergence orders — was verified and stands.

## 7. M7 — hardening and backend parity

M7 asks for three things: a multi-seed table, every performance claim measured at
a stated config, and torch and JAX statistically indistinguishable. **None of the
three is satisfied.** What exists is below; `TBD` marks a number that has not been
measured, not one that has been measured and omitted.

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

### 7.2.1 The block weighting was the variance

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

### 7.2.2 Why the void does not form — a number, not a technique

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

### 7.2.3 Variable scaling — the void equation now gets solved

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

### 7.2.4 The "over-running front" was a phantom, and §7.2.3 caused it

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

### 7.2 Backend parity

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

### 7.3 Optimiser bake-off

**TBD — not started.** The plan called for SSBroyden/SSBFGS
([arXiv:2501.16371](https://arxiv.org/abs/2501.16371)) first, since they drop into
the same schedule slot as L-BFGS with no new machinery, then NysNewton-CG
([ICML 2024](https://proceedings.mlr.press/v235/rathore24a.html)). Deferred
deliberately: with the reference unconverged in `α` and the two backends 3–10×
apart, an optimiser comparison would be measuring the ruler.

### 7.4 Pseudo-time stepping

Implemented (`pts_every`, `pts_dtau`, `pts_growth`) and smoke-tested; **accuracy
TBD** — the ablation run was killed before its three configurations finished.

### 7.5 GPU timing

**TBD — not started.**

### 7.6 What M7 did deliver

The JAX twin itself (`axial/pinn_jax.py`), sharing the residual functions with the
torch backend and satisfying every hard constraint exactly. It has already earned
its cost twice: it showed the pre-fix failure was *not* backend-specific, which
implicated the formulation; and the post-fix divergence exposed the frozen
collocation bug. Both are findings a single backend could not have produced.

## 7.6 What is still open

| topic | status |
|---|---|
| Fourier + modified MLP combined | **measured, and it fails** — §7.2.6 |
| Plan A, multiple seeds | **TBD** — one seed measured (§7.5) |
| Backend parity, post-closure | **TBD** — the structure is at parity and tested; the accuracy comparison has not been re-run |
| Optimiser bake-off (SSBroyden / SSBFGS) | **TBD — not started** |
| GPU timing | **TBD — not started** |
| Pseudo-time stepping accuracy | measured harmful (§7.2.5); no further work planned |
| M4 acceptance: onset within 0.5 s and one cell | **not met.** The front now forms, but onset is late. Under D-TH-3 the front is the level set `T_c = T_sat + ΔT_sup`, so this is bounded by `T_c` accuracy |
| The 1% bar on temperatures | **not met** — see §7.2.5 for the current figures |

## 8. What to do next

In order, and none of it is "add another method":

1. **Fix the measurement first.** Score against `n_axial ≥ 160`, and re-derive the
   acceptance bar so it sits above the reference's own error. Some unknown part of
   the "PINN failure" is the ruler.
2. **Score `α` on a metric a front can satisfy** — voided length, onset time and
   location — rather than a pointwise norm across an unconverged discontinuity.
3. **Only then** re-run the ablation. Remedies aimed at a moving front cannot be
   judged against a reference that resolves the front to 10%.

Until (1) and (2) are done, no accuracy number from this model should be quoted,
and none is quoted in this repository outside this document.

Everything marked `TBD` above is measurable with the code as it stands; none of it
needs new development, only compute and a corrected reference.

### 7.2.5 N6 — re-ablated against the algebraic closure, and D38 is half wrong

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

### 7.2.6 The two winners do not compose

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

## 7.4 Backend parity, re-measured after the closure

§7.2's parity table was marked superseded because it compared two models rather
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
consistently ~21% better. That is a real residual difference, not noise, and it is
unexplained. Both backends reach `max α = 1.0000`, so the front forms in both.

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

**So M7's acceptance is still not met, for a much narrower reason.** Two fields
agree, two differ by a consistent 21%.

### 7.4.1 RAR is not the cause

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

### 7.4.2 The L-BFGS polish is the whole remaining gap

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

**The larger finding is not about parity at all.** With the polish off, `L_void`
is 0.0000 and `max α` is 0.0000 in **both** backends, on every seed. Three
thousand Adam iterations never form the boiling front; three hundred L-BFGS
iterations do. The quasi-Newton polish is not a refinement here — it is the step
that finds the front, which makes the choice of L-BFGS implementation a physics
question rather than a tuning one.

That reframes §7.3's deferred optimiser bake-off: SSBroyden and SSBFGS
([arXiv:2501.16371](https://arxiv.org/abs/2501.16371)) drop into exactly this
slot, and this is now the highest-value untested item in the project rather than
a nice-to-have. **TBD.**

## 7.5 N5 — Plan A measured end to end

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
