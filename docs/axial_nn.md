# Axial PINN — formulation, recipe, and measured results

The neural-network side of the 1D axial boiling model. The physics, and every
deviation from the SAS4A/SASSYS-1 manual, are in
[`axial_physics.md`](axial_physics.md); this document covers the ansatz, the
training recipe, the two backends, and **what has actually been measured** —
including the results that came out badly, which are most of them.

> **Status.** The network trains, satisfies every hard constraint exactly, and
> reproduces the reference to **3–12% relative `L2`** against an acceptance bar of
> 1%. It does **not** meet that bar. Section 6 documents a bug hunt that found the
> bar itself is partly unsound, and section 5 documents five standard remedies
> that all made the fit *worse*. None of this is presented as a working result.

---

## 1. What the network solves

A map `(ζ, t) → (T_f, T_cl, T_s, T_c, α)`, trained on the Chapter 3 residuals
alone. No reference data enters the loss; the reference is used only at test time,
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

```math
\theta_k(\zeta, \hat t) = \theta_{k,0}(\zeta)\,\exp\!\big(\hat t\, N_k(\zeta, \hat t)\big),
\qquad
\alpha = \tanh(a\hat t)\,\tanh(a\zeta)\,\sigma(N_\alpha),
\qquad
c_i = \exp\!\big(\hat t\, N_{c,i}(\hat t)\big)
```

| Constraint | Mechanism | Measured |
|---|---|---|
| Initial condition | `exp(0) = 1` | exact, `0.0` |
| **Positivity `T ≥ T_in`** | `θ₀ ≥ 0` times a positive exponential | exact under ×50 adversarial weights |
| Coolant inlet `T_c(0,t) = T_in` | `θ_c0(0) = 0`, so it falls out of the same form | exact, `0.0`, **no separate gate** |
| Void `α ∈ [0,1)`, void-free start, none at inlet | gated, **biased** sigmoid | exact by construction |
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

Available and **off by default**, because the ablation in §5 says they hurt:
time-window curriculum (`n_windows`), random Fourier features
(`fourier_features`), the two-encoder modified MLP (`modified_mlp`), and
pseudo-time stepping (`pts_every`).

## 4. Two backends, and why

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
own: a 1e6× reduction in residual bought ~25% in `T_f` and cost 2× in `T_c`. That
is the signature the 2026 spurious-solution and overfitting work describes
([arXiv:2604.23528](https://arxiv.org/abs/2604.23528),
[arXiv:2605.30910](https://arxiv.org/abs/2605.30910)) and that REPORT-01 §5.2
items 9–10 anticipate. The next diagnosis belongs there — specifically, why the
front over-runs — and not in another architecture.

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
