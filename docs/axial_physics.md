# Axial boiling model — physics and deviation register

The second model in this repository: a **1D axially resolved SFR channel** with
sodium boiling, following the SAS4A/SASSYS-1 manual
([ANL/NSE-SAS/5.8.1](https://sas-doc.nse.anl.gov/latest/)). It sits alongside the
lumped 0D model of [`physics_theory.md`](physics_theory.md), which it does not
replace — the 0D model remains the fast regression harness and the pedagogical
entry point.

> **Status: M0–M7 built; M8 attempted, one half adopted.** Parameters, mesh and
> shapes (M0); the thirteen §12.13 sodium correlations (M1); the Chapter 3 energy
> balance and its stiff reference solver (M2); the PINN with prescribed power
> (M3); §12.4 boiling onset with a mixture void field (M4); §12.5.1 film
> degradation and dryout (M5); the prompt-jump kinetics closure in both the
> reference and the PINN (M6); the JAX twin (M7).
>
> **M8 was attempted from two directions.** Eliminating the void algebraically
> (D-TH-3) is adopted and is the change that made the boiling front form at all.
> A front-position network was built, measured worse on every metric, and is off
> by default. M9 (parametric sweep) is not started.
>
> **Four mechanisms have since been added, all off by default and all
> registered**: condensation (D-TH-4), decay heat (D-KIN-3), axial fuel expansion
> (D-FB-4) and the vapour-expansion flow term (D-TH-2, implemented but not
> usable). Defaults are unchanged, so no previously published number moves.
>
> **The reference solver is verified — with one measured exception in §6.5, where
> the void fraction is not mesh-converged at the default resolution. The PINN is
> not converged**; [`axial_nn.md`](axial_nn.md) records every measurement,
> including the negative ones.

---

## 1. Why a second model

The 0D model uses a *demonstration* void surrogate — a `tanh` ramp centred at
820 K, flagged in `physics_theory.md` §6 as not the ~1156 K sodium saturation
temperature. It cannot represent where boiling starts, how far the void spreads,
or the fact that the sodium void worth **changes sign** near the top of the core.
Those are the quantities a loss-of-flow safety argument actually turns on.

This model resolves the channel axially and takes its thermophysics, boiling
onset and feedback laws from the manual.

## 2. What the manual specifies

| Physics | Manual location | Status here |
|---|---|---|
| Fuel and cladding conduction | Eq. 3.3-1 | **done**, radially lumped |
| Fuel-to-cladding flux (gap conductance **+ radiation**) | Eq. 3.3-4 | **done** |
| Coolant energy, three sources `Q_c + Q_ec + Q_sc` | Eq. 3.3-5 | **done** |
| Direct neutron/gamma heating in the coolant, `γ_c` | Eq. 3.3-6 | **done** (`gamma_c`) |
| Pre-boiling momentum, `w = w(t)` independent of `z` | Eq. 3.9-1 | **done** |
| Point kinetics (prompt jump) | Eq. 4.2-4 | **done** (`prompt_jump_power`) |
| Delayed-neutron precursors | Eq. 4.3-1 | **done** |
| Decay heat, `ψ_t = ψ_f + ψ_h`, ANS standard | Eq. 4.2-2, §4.4 | **done**, three groups, off by default (D-KIN-3) |
| Condensation as the negative branch of the film heat flow | §12.5 | **done**, off by default (D-TH-4) |
| Axial fuel expansion feedback | §4.5.4 | **done**, off by default (D-FB-4) |
| Vapour expansion accelerating the flow | §12.2, §12.6 | **implemented, not usable** (D-TH-2) |
| **Doppler, logarithmic**, flooded↔voided interpolation | Eq. 4.5-2, 4.5-3 | **done** (`alpha_D`, `reactivity`) |
| **Coolant density + void as one worth sum** | Eq. 4.5-25 | **done** (`void_worth`, `reactivity`) |
| Boiling onset: saturation + superheat | §12.4 | **done** (`boiling_fraction`) |
| Cladding/structure → vapour heat path | §12.5.1 | **done** (`film_coefficient`) |
| Sodium properties | Eq. 12.13-1 … 12.13-13 | **done** (`axial/sodium.py`) |

The chapters are mirrored for offline reading; regenerate with
`uv run python tools/fetch_sas_manual.py docs/sas4a`.

## 3. Deviation register

Every departure from the manual, with the equation it departs from and why.
**This register is the contract**: a deviation that is not listed here is a bug,
not a simplification. Each entry is either *settled* (decided at M0) or *open*
(needs the owner).

### D-KIN-1 — Prompt jump approximation *(settled)*

**Departs from:** §4.6.1, Eq. 4.6-1/4.6-2. **Status: adopted.**

The manual solves the **full** point kinetics equations: the precursor equations
are integrated analytically over a step, the power amplitude is expanded
quadratically, and precision is held by comparing against a half-step solution.
The phrase "prompt jump" appears **nowhere in the 216-page manual** — SAS4A
deliberately keeps the prompt term.

We set `dP/dt = 0` and close on `P = Σβᵢcᵢ/(β − ρ)` instead. **The justification
is PINN trainability, not fidelity**: it removes the `Λ ≈ 5e-7 s` mode that
`neural_network.md` §2 identifies as the documented failure driver for stiff
PINNs, which is the same quasi-steady-state manoeuvre Stiff-PINN applies to
chemical kinetics. Two consequences that must be reported, not buried:

- The closure has a **pole at prompt criticality** (`ρ → β`). Every run must log
  `max_t ρ/β`; the 0D model's measured peak is `+0.17` dollars, i.e. 83 % margin,
  but raising the void worth spends exactly that margin.
- It is valid for `ρ ≪ β` **and** for reactivity varying slowly against the
  prompt mode `Λ/β ≈ 1.4e-4 s`. A ULOF driven by a ~5 s coast-down satisfies the
  second condition by four orders of magnitude — the stronger of the two
  arguments.

### D-GEOM-1 — Radially lumped materials *(settled)*

**Departs from:** §3.2.2.1, Eq. 3.3-1. **Status: adopted.**

SAS4A resolves the pin radially: 4–11 nodes in the fuel, 3 in the cladding, 1 in
the coolant, 2 in the structure. We use **one node per material per axial
position**, so radial conduction becomes a resistance chain.

The fuel case is defensible rather than merely convenient: the manual evaluates
Doppler on the **radial mass-averaged** fuel temperature (Eq. 4.5-3), which is
precisely what a single lumped node approximates. The cost is the radial
temperature *profile* — centreline vs surface — which matters for melting
(§3.3.5) and is out of scope here.

### D-GEOM-2 — Structure node retained, but lumped *(settled)*

**Departs from:** §3.2.2.1, §12.5.1. **Status: retained (`gamma_2`).**

The requirement names three materials (fuel, cladding, sodium). The manual has
**four** — the structure (duct wall) exchanges heat with the coolant (`Q_sc` in
Eq. 3.3-5) and with the vapour after voiding, weighted by
`γ₂ = (structure surface area)/(cladding surface area)` (§12.5.1).

Dropping it would make the §12.5.1 heat path un-implementable and would
**under**-predict vapour generation — a non-conservative error. It is therefore
kept as `gamma_2`, with `gamma_2 = 0.0` available to disable it explicitly.

### D-FB-1 — One void-worth distribution, not two coefficients *(settled)*

**Departs from:** nothing — this **removes** a deviation. **Status: adopted.**

The 0D model splits coolant feedback into a linear `α_c (T_c − T_c0)` term plus a
separate `α_void φ(T_c)`. The manual uses **one** mechanism (Eq. 4.5-25):

```math
\delta k_{Na} = \epsilon_d \sum_i \sum_j (\rho_c)_{ij}\,\alpha_{ij}
```

a per-segment worth `(ρ_c)_j` times the segment void fraction, where the void
fraction covers *both* liquid density change and boiling-induced voiding. The
axial model follows the manual: a single `void_worth(ζ)` profile. This also
removes a double-counting risk at onset, where both of the 0D model's terms are
active simultaneously.

`void_worth(ζ)` is **positive** through the core and turns **negative** above
`zeta_sign` — the leakage reversal near the axial blanket. A uniform positive
coefficient cannot represent it and overstates the excursion.

The manual's optional second-order term (Eq. 4.5-26, `VOIDRA2`) is **not**
implemented: the manual states it is incompatible with the sodium boiling model.

### D-FB-2 — Logarithmic Doppler *(settled)*

**Departs from:** nothing — this **removes** a deviation. **Status: adopted.**

The 0D model uses linear `α_f (T_f − T_f0)`. The manual defines the coefficient
by `T_f d(δk_D)/dT_f = α_D` (Eq. 4.5-2), integrating to

```math
\delta k_D(t) = \alpha_D \ln\!\left[\frac{T_f(t)}{T_f(0)}\right]
```

(Eq. 4.5-3). Over a ULOF fuel-temperature swing these are not interchangeable:
the linear form keeps growing where the true one saturates, and the error lands
on the feedback that has to turn the excursion over. The axial model uses the
logarithmic form. It costs one logarithm and stays smooth for autodiff.

`α_D` is additionally **interpolated between flooded (`ADOP`) and voided
(`BDOP`) values** (§4.5.3), so voiding modulates Doppler. That is implemented as
`AxialParams.alpha_D(void_fraction)`; setting the two coefficients equal
disables the coupling.

### D-FLOW-1 — Prescribed flow rather than prescribed head *(settled)*

**Departs from:** Eq. 5.3-61. **Status: adopted, narrowed.**

PRIMAR's table-look-up pump prescribes **head**, `H(t) = H_r f(t)` with
`f(0) = 1`, and lets the loop momentum equation produce the flow; centrifugal
pumps (§5.3.4.2) additionally carry rotor inertia, which makes a physical
coast-down nearer hyperbolic than exponential.

We prescribe **flow** directly, `g(t) = f_nc + (1 − f_nc)e^{−t/τ}`. Same shape,
one equation instead of a loop model, and no primary circuit. The deviation is
narrow and named; it is not an invention.

### D-TH-1 — Eulerian mixture void field, not Lagrangian slug tracking *(settled)*

**Departs from:** §12.2, §12.3, §12.6. **Status: adopted — the load-bearing one.**

Chapter 12 is a **multiple-bubble slug-ejection** model: up to nine bubbles,
created, merged and collapsed as discrete events, with liquid-slug momentum
integrated between moving interfaces. A PINN represents the solution as one
smooth function of `(ζ, t)`; topology changes have no smooth representation and
no usable autodiff gradient.

We keep the manual's thermophysics and onset criterion and replace the slug
bookkeeping with a **continuous void fraction field** `α(ζ, t) ∈ [0, 1)` on the
fixed axial mesh, transported by a mixture formulation. Taken verbatim from
Chapter 12: the §12.13 property correlations, the §12.4 saturation-plus-superheat
onset criterion (`DTS ≈ 10 K` first bubble, `DTSI ≈ 3–4 K` after), and the
§12.5.1 cladding/structure→vapour heat path.

This is the deviation a reviewer will challenge first. It is deliberate, and M8
proposes a single-bubble two-interface case as the direct comparison against
§12.5 where Chapter 12 *is* tractable.

### D-TH-3 — The void eliminated algebraically *(settled)*

**Departs from:** §12.4/§12.5 as a *differential* void. **Status: adopted,
`void_closure = True`.**

The vapour source fills a node in 0.71 ms against a 0.113 s transport time, so
`α` is a fast variable slaved to the temperature field. It is eliminated
algebraically — `α = 1 − (1 − b)³`, with `b` the superheat switch — and its
residual removed from the PINN loss, which is the manoeuvre Stiff-PINN applies to
fast chemical species and D-KIN-1 already applies to the prompt neutron mode.

The reference solver still integrates the differential void; the closure is a
**PINN-side** approximation, validated against it at 1.2% on maximum voided
length, inside the reference's own 2.6% mesh error. See
[`axial_nn.md`](axial_nn.md) §7.2.3.

Two exact properties make it sound rather than convenient. Below saturation `b`
underflows to **exactly** zero, so the void equation is pure advection there and
`α = 0` is its unique solution under a zero initial and inlet condition. And the
source is non-negative everywhere — there is no condensation at the default
settings — so the superheated set is contained in the voided set.

The closure is what made the boiling front form at all. It also supplies the
void-free initial and inlet conditions for free, so the gate and sigmoid head the
ansatz previously used are unnecessary.

### D-TH-4 — Condensation present but off by default *(settled — experimental)*

**Departs from:** §12.5. **Status: implemented, disabled.**

The manual determines the rate of vapour *formation and condensation* from one
film heat flow, condensation being its negative branch — "Condensation in cooler
regions can cause a vapor bubble to shrink" — and carries an adjustable
condensation coefficient. Omitting it was an **unregistered** deviation until
Rev 6; `AxialParams.condensation` now implements it, defaulting to 0.0.

**Measured inert in this scenario, and the reason is structural.** Condensation
needs the film heat flow to reverse. Vapour only ever exists where dryout has
already driven the cladding *hotter* than the coolant, so `q_ec > 0` everywhere
`α > 0` (minimum +1168 W/m) and the net wall heat never changes sign. At
`condensation = 1.0` the voided length moves 0.3% and `α` stays monotone along a
characteristic to 1e-6. Energy conservation is unaffected (5.1e-5).

Condensation is what the multi-bubble slug-ejection model of D-TH-1 needs, where
bubbles expand into cooler regions above and below the boiling zone. In a single
Eulerian channel that terminates at dryout six seconds after onset, there is no
cooler region for the vapour to reach.

### D-KIN-3 — Decay heat, three groups not twenty-three *(settled)*

**Departs from:** §4.4. **Status: implemented, `decay_fraction = 0.0` by default.**

`ψ_t = ψ_f + ψ_h` (Eq. 4.2-2). The fission channel carries `1 − decay_fraction`,
so the nominal total is exactly one and no criticality retuning is needed. Three
groups spanning seconds to hours reproduce the *shape* of the ANS standard rather
than its full table.

**This removes the zero-power attractor.** At `decay_fraction = 0` the homogeneous
kinetics have no source and `P = c = 0` is an exact solution of the whole coupled
system — the collapse mode REPORT-01 §5.2 exists to diagnose. Non-zero, `ψ_h` does
not vanish when `ψ_f` does: measured, `total_power(0, h) = 0.065`.

### D-FB-4 — Axial fuel expansion *(settled)*

**Departs from:** §4.5.4. **Status: implemented, `alpha_expansion = 0.0` by default.**

Fuel lengthening lowers the axial power density and leaks more neutrons, so the
coefficient is negative and the mechanism stabilising. Omitting it was
**non-conservative** — the model over-predicts the excursion (D-FB-3). Linear in
`T_f − T_f0` on the same radial mass-average as Doppler, so it vanishes at nominal
and needs no offset. Measured at `−1e-6 /K` and a 30 % fuel-temperature rise, it
adds 0.083 `β` of negative reactivity: −0.4498 `β` becomes −0.5329 `β`.

Radial core expansion and control-rod-drive expansion remain unimplemented, so
D-FB-3 narrows rather than closes.

### D-TH-2 — Axially uniform flow after voiding *(open — implemented, not usable)*

**Annex A, N7b: a kinematic closure was built and measured.** Mixture continuity
with both phases incompressible gives `du/dz = Γ_α (1 − ρ_v/ρ_l)`, integrated
upward from the inlet — `physics.expansion_velocity`, behind
`AxialParams.flow_expansion`, default `False`.

It is **not usable as formulated.** Sizing it on the baseline solution the extra
velocity is 1.6 m/s against a base 1.6 m/s at t = 16.5 s, i.e. a 2× CFL
reduction, which should be affordable. It is not: with the term active the Radau
solve fails to complete a 14 s transient at `n_axial = 40` inside 110 s, where the
same case without it finishes in seconds.

The reason is a feedback loop the CFL estimate does not see — expansion
accelerates the flow, which sweeps more wall heat into the boiling region, which
makes more vapour, which accelerates the flow. That loop *is* slug ejection, and
it is why Chapter 12 carries a dedicated implicit slug-momentum solver rather
than adding an expansion term to the energy equation. A kinematic `du/dz` cannot
substitute for `Fc`, the condensation momentum loss, and the interface momentum
balance of §12.6.

**So D-TH-2 stays open, and now with its cost measured rather than assumed.**
Closing it is the Lagrangian slug model, i.e. a scope change back toward what
D-TH-1 replaced — not a coefficient and not a closure.


**Departs from:** §12.2. **Status: open.**

Eq. 3.9-1 states `w = w(t)`, independent of `z`, for pre-boiling incompressible
flow — so uniform flow is **correct and documented** in the Plan B regime and M3
needs no caveat. Once vapour is generated the mixture accelerates and `w` becomes
`z`-dependent; that acceleration *is* the slug-ejection mechanism. M4 must either
carry `w(ζ, t)` from mixture continuity or state the assumption and accept
under-predicted void growth.

### D-FB-3 — Five feedback mechanisms omitted *(open — needs the owner)*

**Departs from:** §4.5.4, §4.5.6, §4.5.7, §4.5.8. **Status: open.**

The manual carries eight feedback mechanisms. This model implements Doppler,
coolant/void, and an external insertion. **Omitted: fuel/cladding/structure axial
expansion, radial (core) expansion, control-rod-drive expansion, and
fuel/cladding relocation.**

Axial and radial expansion are **negative** and are among the principal reasons
real metal-fuel SFRs are inherently self-limiting under loss of flow. Omitting
them is therefore **non-conservative**: this model will over-predict the
excursion. That is acceptable for a PINN methodology study and unacceptable in a
safety claim. It must be stated in those words wherever results are reported.

### D-KIN-2 — Decay heat *(closed by D-KIN-3)*

**Superseded.** Decay heat was open and is now implemented; the live entry is
D-KIN-3. The identifier is retained because earlier revisions of the milestone
report cite it.

### D-SCOPE-1 — No pin failure or material relocation *(settled)*

**Departs from:** Chapters 8–11, 13–16. **Status: out of scope.**

This model stops at the **onset and growth of voiding**. DEFORM-4 pin mechanics,
CLAP cladding motion, PLUTO2/PINACLE/LEVITATE fuel motion are not modelled. A
corollary that must be checked, not assumed: results are only self-consistent
while the transient stays bounded and cladding stays intact. If a parameter sweep
pushes a case past that, the case is outside the model, not a prediction.

## 4. Sodium properties (M1)

`axial/sodium.py` implements all thirteen numbered relations of §12.13. Two
things are worth knowing before using them.

**Validity is 590–2270 K and nothing enforces it.** The manual stops the
polynomial fits at ~90 % of the critical point (2503.3 K) to avoid fitting the
rapid near-critical variation, and `C_l`, `β_s` and `α_p` all contain
`1/(T_c − T)`, so they diverge there. The module deliberately **neither clamps
nor raises**: a hard guard inside a residual would break autodiff and abort a
training run the moment a transient overshot. `sodium.in_range(T)` is provided so
callers can diagnose it instead; M2 should assert it on the reference solution.

**All three backends, from the first module — and that is a correctness
mechanism.** One implementation serves numpy, torch and JAX by dispatching on
argument type, so all three evaluate the same expression tree. `neural_network.md`
§9 records why this matters here specifically: the PyTorch initialisation bug
that left the power trajectory at `L2 ≈ 0.3` was identified *because* the JAX
twin fit well at the same budget. A cross-backend discrepancy is a signal, and
two backends cannot tell you which one is wrong.

Measured agreement: the pure-polynomial correlations are **bit-identical** across
all three, while those using `exp`, `log` or division agree to **~1 ULP** — the
backends call different libm implementations, and IEEE-754 does not require
correctly-rounded transcendentals. Both bounds are asserted, because a looser
tolerance would hide genuine transcription drift and a tighter one fails on
rounding alone. Torch and JAX are additionally checked against *each other*, and
their autodiff gradients are checked to agree to 1e-12.

**JAX must run in float64.** `jax.config.update("jax_enable_x64", True)`, as
`pinn_jax` does at import. At the default float32 the `Ts(Ps(T))` round-trip
degrades from 1e-11 K to ~1e-2 K; a test asserts the gap so the requirement
cannot quietly lapse.

Verification at the normal boiling point, against values independent of the
manual (a self-consistent but mistyped coefficient would otherwise pass):

| Quantity | This module at 1154 K | Literature |
|---|---|---|
| `Ts` at 1 atm | 1159 K | 1154–1156 K |
| Latent heat | 3.88e6 J/kg | ~3.9e6 |
| Liquid density | 743 kg/m³ | 740–780 |
| Vapour density | 0.263 kg/m³ | ~0.28 |
| Liquid heat capacity | 1272 J/kg-K | 1260–1300 |
| Liquid conductivity | 52.1 W/m-K | 46–50 |
| Liquid viscosity | 1.65e-4 Pa-s | ~1.8e-4 |

Round-trip `Ts(Ps(T)) − T` over the full range: **1.6e-11 K**, as expected from
Eq. 12.13-4 being the exact analytic inverse of Eq. 12.13-2.

**This closes the headline caveat of the 0D model.** `physics_theory.md` §6 flags
its 820 K void onset as a demonstration threshold rather than the sodium
saturation temperature. Boiling onset now comes from `Ts(Ps)` plus the §12.4
superheat, at the right temperature.

## 5. The M2 reference solver

Method of lines: Chapter 3's energy equations discretised in `z`, advanced by an
implicit Radau integrator. Four verification handles, none of which assumes the
solver is correct:

| Check | Result |
|---|---|
| `rhs(0, y_steady)` — steady state is an exact oracle | **3.4e-11 K/s** |
| Steady rise equals the telescoped nodal sources | exact to 1e-12 relative |
| Constant flow (`f_nc = 1`) holds the steady state | `< 1e-6 K` over 30 s |
| Energy balance closure | **1.4e-5** at `n_out = 241`, converging to zero as `Δt²` |
| Mesh convergence, transient (L2-in-time of `T_out`) | ratio **≈ 2** — first order |
| Mesh convergence, quasi-steady (`T_out` at `t_end`) | ratio **≈ 4** — second order |

The two convergence orders differ *for a reason*, and that is itself the check.
First-order upwind advection dominates during the coast-down. But in steady state
the upwind stencil telescopes to the **exact** integral of the nodal sources and
contributes no truncation error at all, leaving only the second-order midpoint
quadrature of the axial power shape. Seeing 1 and 2 in the right places is much
stronger evidence than seeing one number twice.

Upwind is deliberate rather than convenient: it is monotone, so it will not
oscillate across the void front that M4 introduces. First order is the price.

**Radau needs the Jacobian sparsity pattern.** Every coupling is local — a node
talks to its material neighbours and, for the coolant, its upwind neighbour — so
of the `(4n)²` entries only `O(n)` are non-zero. Supplying that pattern turned
the `n = 640` convergence study from intractable into 1.8 s. A test finite-
differences the true Jacobian and asserts the pattern covers every real coupling,
because a missing entry would silently degrade Radau's Newton solve rather than
fail loudly.

**Scope limit, asserted rather than hidden.** M2 is single-phase by construction,
but with the default coast-down the coolant passes the sodium saturation
temperature (~1159 K at 1 atm) at about **t = 11 s**. From there the run is
non-physical: it is a *solver verification vehicle* until M4 adds boiling. A test
asserts the crossing happens, so the limitation cannot quietly disappear.

## 6. The M3 PINN and the M4 void field

### 6.1 M3 — the network

A network of `(ζ, t)` trained on the Chapter 3 residuals alone; no reference data
enters the loss. Power is prescribed, so this is the plan's **Plan B**: the
thermal-hydraulics is validated before the kinetics feedback is closed at M6.

**One set of equations.** The residual calls `continuous_derivatives` — the same
function, and therefore the same flux and boiling expressions, that the M2
reference discretises. A test rebuilds the residual by hand from it and asserts
bit-equality, so the network and its ground truth cannot drift apart. That test
is this model's answer to the 0D `tests/test_consistency.py`.

**Three hard constraints, none of them in the loss:**

| Constraint | Mechanism | Measured |
|---|---|---|
| Initial condition | `θ = θ₀(ζ) + t̂·N` | exact, `0.0` error |
| Coolant inlet `T_c(0,t) = T_in` | extra `ζ` factor on that column | exact, `0.0` error |
| Void `α ∈ [0,1)`, void-free start, no void at inlet | `tanh(a t̂)·tanh(a ζ)·σ(N)` | exact by construction |

Eq. 3.9-1 admits exactly one upstream condition and the void equation likewise,
so those are the only boundary conditions imposed — no more, no fewer.

The steady profile is implemented **twice**, once in numpy and once in torch,
because `jvp` cannot trace a numpy detour and the ansatz must differentiate with
respect to `ζ`. A test asserts they agree to 1e-9, so there is still one
definition being checked rather than two being trusted.

### 6.2 M4 — boiling onset and the void field

The §12.4 criterion `T_c > T_sat + DTS` becomes a logistic of width `dT_smooth`,
now around a **physical** saturation temperature from Eq. 12.13-4 (~1159 K at
1 atm) rather than the 0D model's 820 K demonstration value. Above it, the wall
heat stops raising the coolant temperature and starts making vapour.

Measured on the reference:

| Quantity | Result |
|---|---|
| Boiling onset | **t = 10.8 s at ζ = 0.96** — the top of the channel, the hottest point |
| Void bounds | `α ∈ [0, 1]` to round-off, by the `(1−α)` shutoff |
| Voided length | 0.50 m, reached by t ≈ 40 s |
| Energy closure **with** latent heat | 3.6e-6, converging to zero as `Δt²` |
| Control (`p_system` raised so `T_sat` ≈ 2280 K) | void identically zero |

**Voiding is explosive, and that is physical.** Filling one node with vapour
takes about 1 J while the wall delivers about 1 kW, so the front runs away within
seconds of onset. This is exactly why Chapter 12 is a slug-*ejection* model.

**Two conservation defects were caught by the energy-balance test, not by
inspection** — see §7.

### 6.3 M5 — film degradation and dryout

Section 12.5.1 puts the liquid film and the vapour **in series** in the
wall-to-coolant resistance. M5 implements that as a void-weighted blend,
`h_eff = (1−α)·h_wet + α·h_vapour`, with the vapour value two to three orders
smaller. This is what turns boiling from a temperature *plateau* into a cladding
*excursion*, and it is why M4 alone was not enough — before M5 a boiling run and
one that could not boil agreed to ~1 K.

| Quantity | Result |
|---|---|
| Peak cladding, boiling | 2149 K |
| Peak cladding, cannot boil | 1685 K |
| **Dryout excursion** | **+464 K** |
| Energy closure | 5.0e-5, still converging as `Δt²` |

**The run now stops at the validity limit.** Once dryout drives temperatures
toward the 2270 K ceiling of the §12.13 fits, integration terminates and
`stopped_early` records it. With the default parameters that is **t = 16.5 s**,
5.7 s after onset. This is not a numerical convenience: past dryout this model
has no melting, no cladding motion and no fuel relocation (Chapters 8–16), and
the sodium correlations stop at 2270 K, so integrating on would extrapolate three
models at once and present the result as a prediction. Stopping makes the model
state in its own output exactly where it ceases to apply.

**One thing M5 deliberately does *not* do.** The mixture heat capacity
`(1−α)ρ_l c_l + α ρ_v c_g` is implemented (`coolant_capacity`) and correct, but is
**not used in the residuals**. Substituting it into the *temperature*-form energy
equation breaks conservation — a changing capacity needs an enthalpy
formulation — and measurably so: the closure degrades from 3.6e-6 to ~2e-2. M5
therefore degrades the film coefficient, which is the mechanism §12.5.1 actually
describes, and leaves the capacity to a future enthalpy-form revision. A test
pins the expression and records why it is unused.

### 6.4 M6 — the kinetics closure (reference side)

`solve_reference(..., feedback=True)` makes power an **output**: six precursor
ODEs (Eq. 4.3-1) closed by the prompt jump, `P = Σβᵢcᵢ/(β−ρ)`, with reactivity
from two axial integrals — logarithmic Doppler (Eq. 4.5-3, with the flooded↔voided
`α_D`) and the merged coolant/void worth (Eq. 4.5-25).

**The nominal state is exactly critical with no offset.** At nominal
`ln(T_f/T_f0) = 0` and `α = 0`, so both integrals vanish identically and
`P = Σβᵢ/β = 1` to the last bit. The 0D model has to absorb a residual into
`rho_ext` to get the same thing; the logarithmic form gives it for free.

| Quantity | Plan A | Plan B (prescribed) |
|---|---|---|
| `P(0)` | 1.000000000000 | — |
| Peak power | **1.0000 at t = 0** — feedback only removes reactivity | — |
| Final power | 0.4984 | — |
| **max ρ/β (pole tripwire)** | **+0.0000** (bar: < 0.5) | — |
| min ρ/β | −0.207 | — |
| Boiling onset | 15.3 s | 10.8 s |
| Peak cladding | **1893 K** | 2149 K |
| Reaches the validity limit? | **no**, completes 60 s | yes, stops at 16.5 s |

Closing the loop is **stabilising**: less power, later onset, cooler cladding, and
the run no longer leaves its own validity range.

**But `max ρ/β = 0.0000` is an artefact of *where* the channel boils, not a
statement about stability.** The void worth changes sign at `zeta_sign = 0.80`
(§3, D-FB-1) and boiling starts at ζ ≈ 0.96–0.99, because that is where the
coolant is hottest. The mixture void field advects **upward only**, so the voided
region can never reach the positive-worth part of the core. Measured over the
whole closed-loop transient:

| component | min [ρ/β] | max [ρ/β] |
|---|---|---|
| Doppler | −0.181 | 0.000 |
| coolant density / void | −0.036 | **+8.8e-23** (i.e. zero) |

Every void contribution is negative, so the net reactivity can only fall and the
pole is unreachable *by construction rather than by physics*. The consequence is
sharp: **the positive sodium-void feedback this project exists to examine is
never exercised at the shipped defaults**, and Objective 2 — "does tuning
`α_void` capture both regimes?" — cannot be answered by scaling
`void_worth_net`, because that scales a term which is identically ≤ 0.

`AxialTrajectory.rho_doppler`, `AxialTrajectory.rho_void` and
`AxialTrajectory.void_worth_is_exercised` now report this on every run, so a null
result can no longer be read as a stabilising one. Exercising the positive branch
needs `zeta_sign` above the onset location, or a void field that can propagate
downward — which is slug ejection, i.e. D-TH-1. **That is a parameter and scope
decision for the owner, not one taken here.**

**The stopwatch test passes.** The report's §5.2 predicted that under the
prompt-jump closure no delayed-neutron tail can decay faster than
`1/λ₁ = 80.6 s`, at any reactivity. Measured tail: **260 s**. A faster tail would
have meant the closure was violated or bypassed rather than that the physics was
severe.

**Not done: the PINN side of M6.** `pinn_torch.py` still runs Plan B with power
prescribed. Extending it needs a second network for the precursors (functions of
`t` alone) and the axial quadrature of §3.5a inside the residual. The M6
acceptance criterion "reference and PINN agree on peak power and time" is
therefore **unverified**.

### 6.5 The reference is not converged in the void fraction

Found during the M7 bug hunt, and it changes how every PINN number must be read:

| reference `n = 40` vs `n = 320` | T_f | T_cl | T_c | **alpha** |
|---|---|---|---|---|
| relative `L2` | 5.1e-3 | 7.2e-3 | 5.4e-3 | **1.03e-1** |

The temperatures are converged to ~5e-3, but the **void fraction is ~10% wrong at
`n_axial = 40`** — first-order upwind with a front spanning 2–6 cells. The M2
convergence study measured its orders on the **non-boiling** case, so this went
unmeasured; that is a gap in the test design, not in the solver. Everything §5
claims about the reference — exact steady state, energy conservation, convergence
orders — was verified on that basis and stands.

Consequence: score the PINN against `n_axial >= 160`, and judge `α` on voided
length and onset rather than a pointwise norm across an unconverged front. See
[`axial_nn.md`](axial_nn.md) §6.

## 7. Known gaps and honest limits

**Two energy-conservation defects, both found by the balance check:**

1. The model diverted `b·q_wall` from the coolant but vaporised only
   `b·(1−α)·q_wall`, so energy vanished as a node approached dryout. Fixed by
   using the same `b(1−α)` on both sides.
2. The balance itself omitted the **latent heat convected out of the top with the
   vapour** — about 45 W against 50 kW, i.e. 9e-4, which is exactly where the
   closure floored. Adding it recovered clean `Δt²` convergence to zero.

Neither was visible in the trajectories, which looked entirely plausible
throughout. This is the argument for conservation checks over eyeball validation.

**A third, in the check itself.** `energy_balance` defaulted its amplitude to
one. That is correct for Plan B and wrong for Plan A, where feedback drives the
power down to 0.498 of nominal: the closure read **0.382** against a true
1.5e-5. No test exercised the feedback trajectory, so a diagnostic built to
detect conservation defects was itself reporting one. It now takes the amplitude
from `traj.power`, which is the delivered power in both plans.

**And one in the parameter container.** `AxialParams.steady_precursors` returned
the *absolute* `C_i = β_i P / (Λ λ_i)` — the 0D model's convention — where this
model's state is the normalised `c_i = C_i / C_{i,0}`, about 5e4 times smaller.
Dormant, because `reference.steady_state` writes `np.ones(N_GROUPS)` directly,
and the only test asserted proportionality in `P`, which holds either way. The
two definitions are now pinned against each other.

## 8. Parameter provenance

**The defaults in `AxialParams` are representative placeholders**, order-of-
magnitude correct for an oxide-fuelled SFR pin cell and not taken from any
specific reactor. Chapter 2 of the manual (the input-deck reference) is the place
to source a consistent realistic set; that is M2 work. Until then, nothing from
this model should be quoted as a physical prediction.

## 9. Milestone status

| M | Deliverable | State |
|---|---|---|
| **M0** | Package scaffolding, `AxialParams`, axial shapes, this register | **done** |
| **M1** | Sodium properties (§12.13) from the manual | **done** |
| **M2** | Method-of-lines reference solver, held-out truth | **done** |
| **M3** | Plan B PINN — prescribed power, no feedback | **done** |
| **M4** | Boiling onset and void field | **done** |
| **M5** | Film / dryout heat path (§12.5.1) | **done** |
| **M6** | Prompt-jump kinetics closure — Plan B → Plan A | reference done; PINN measured and **failed** — `L2(P)` 0.1110 at three seeds against a 1% bar, with 26–28% too little negative feedback on every seed ([`axial_nn.md`](axial_nn.md) §7.4.1) |
| **M7** | Hardening, seeds, JAX port | **done** — backend parity closed: residuals agree to 1e-14 and the 16.8% accuracy gap is the framework L-BFGS, 0.999 with a shared implementation ([`axial_nn.md`](axial_nn.md) §7.3.2) |
| M8 | single-bubble Chapter 12 comparison | superseded — the void is now closed algebraically (D-TH-3), and the front-position network it would have used measured worse |
| M9 | parametric sweep / regime map | **reference half done — §10. The parametric PINN is not attempted**, and that is a decision: the single-point network misses its bar by 7–19×, so extending it over a parameter family would measure nothing |

---

## 10. M9 — the Objective 2 regime map, from the reference

M9 asks for a parametric PINN on `(ζ, t, α_void, τ_pump)` whose regime
*classification* matches a reference sweep. **Only the reference half is
delivered.** The single-point network misses its accuracy bar by 7–19×
([`axial_nn.md`](axial_nn.md) §7.3.2), and a parametric extension of a model that
fails at one parameter value measures nothing. This project has refused nine
remedies on measurement; refusing a stretch milestone on the same grounds is the
same standard.

The reference half stands alone, because Objective 2 is a physics question:
*does the positive sodium void coefficient drive a power excursion?*

### 10.1 The sweep, and the answer

Thirty closed-loop points, `n_axial = 80`, `void_worth_net` from 0 to
1.6e-2 (8× the default) against `tau_pump` from 1 to 20 s. Reproduce with
`uv run python tools/axial_study.py regime`.

| result | value |
|---|---|
| points solved | 30 / 30 |
| **positive void worth exercised** | **0 / 30** |
| **peak power** | **1.0000 at every point** |
| **max `ρ/β`** | **+0.0000 at every point** |
| regimes found | `boiling-bounded` 24, `no-boiling` 6 |
| regimes *not* found | `power-excursion`, `prompt-critical` |

**There is no excursion anywhere in this family, and there cannot be.** Boiling
starts at `ζ ≈ 0.96` and the void worth changes sign at `zeta_sign = 0.80`, so the
voided region lies entirely inside the **negative** lobe. Scaling `void_worth_net`
scales a term that is only ever evaluated where it is negative.

The signature of that is in the table and is worth stating plainly: at fixed
`tau_pump`, **raising the void worth *reduces* the voided length** — at
`tau_pump = 2.5 s`, `L_void` falls 0.2428 → 0.1454 as the worth goes 0 → 1.6e-2.
More sodium void worth makes this transient *safer*. That is the correct behaviour
for a void that only voids where its worth is negative, and it is the opposite of
the mechanism the project exists to study.

Onset time depends only on `tau_pump` (4.50, 8.25, 15.25, 32.50 s, then no boiling
at 20 s) and not at all on the void worth, which is the same fact seen from the
other side: the worth never reaches the power.

### 10.2 What this closes, and what it does not

**It closes D49 and issue I6 in the stronger form.** D49 recorded that the
*default* parameter set never samples the void positive. This shows that is not a
property of the default but of the whole `(void_worth_net, tau_pump)` family:
`max ρ/β = 0` describes the geometry, not the transient, and **Objective 2 cannot
be answered by scaling `void_worth_net`** — the conclusion D49 reached by argument,
now measured across 30 points.

**It does not close Objective 2.** The parameter that governs it is `zeta_sign`,
the sign-change height, which must sit *above* where boiling starts for the
positive branch to be sampled at all. `AxialParams.with_positive_void_worth()`
exists for exactly this and sets `zeta_sign = 0.995`. §10.3 sweeps it.

The regime map above is therefore a **negative control** — it establishes that the
benign outcome is robust to the two knobs a reader would reach for first, which is
worth knowing before any positive result is claimed from the third.

### 10.3 `zeta_sign` — and Objective 2, answered

Eighteen closed-loop points, `zeta_sign` from 0.80 to 0.995 against `void_worth_net`
at 2e-3, 8e-3 and 1.6e-2. Reproduce with
`uv run python tools/axial_study.py regime-sign`.

| `zeta_sign` | worth | regime | peak `P` | `max ρ/β` | exercised | `L_void` |
|---|---|---|---|---|---|---|
| 0.800 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | no | 0.1465 |
| 0.800 | 8.0e-3 | boiling-bounded | 1.0000 | +0.0000 | no | 0.1092 |
| 0.800 | 1.6e-2 | boiling-bounded | 1.0000 | +0.0000 | no | 0.0895 |
| 0.900 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | yes | 0.1785 |
| 0.900 | 8.0e-3 | boiling-bounded | 1.0000 | +0.0516 | yes | 0.3091 |
| 0.900 | 1.6e-2 | boiling-bounded | 1.0000 | +0.0000 | **no** | 0.1019 |
| 0.950 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | yes | 0.2009 |
| 0.950 | 8.0e-3 | **power-excursion** | **1.8601** | +0.4473 | yes | 0.4841 |
| 0.950 | 1.6e-2 | **power-excursion** | **2.4960** | +0.6171 | yes | 0.3301 |
| 0.970 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | yes | 0.2057 |
| 0.970 | 8.0e-3 | **power-excursion** | **1.9423** | +0.4623 | yes | 0.4869 |
| 0.970 | 1.6e-2 | **power-excursion** | **3.5897** | +0.7168 | yes | 0.3508 |
| 0.990 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | yes | 0.2081 |
| 0.990 | 8.0e-3 | **power-excursion** | **2.1432** | +0.4956 | yes | 0.5037 |
| 0.990 | 1.6e-2 | **power-excursion** | **4.0264** | +0.7424 | yes | 0.3554 |
| 0.995 | 2.0e-3 | boiling-bounded | 1.0000 | +0.0000 | yes | 0.2084 |
| 0.995 | 8.0e-3 | **power-excursion** | **2.2119** | +0.5045 | yes | 0.5101 |
| 0.995 | 1.6e-2 | **power-excursion** | **5.2957** | +0.7958 | yes | 0.3702 |

**Objective 2 is answerable, and the answer is yes.** The positive sodium void
coefficient does drive a power excursion in this model — to **5.3× nominal** at
`zeta_sign = 0.995`, `void_worth_net = 1.6e-2` — and the transient is no longer
self-limiting on Doppler alone. Eight of eighteen points are excursions; the
positive branch is exercised at fourteen.

**It is governed by `zeta_sign`, not by the void worth.** At `zeta_sign = 0.80` the
branch is never sampled at any worth (§10.1 showed that across 30 more points). An
excursion needs **both** `zeta_sign ≳ 0.95` *and* `void_worth_net ≳ 8e-3`: at the
default worth of 2e-3 the positive branch is sampled from `zeta_sign = 0.90`
upward and still never overcomes Doppler, peak power staying at 1.0000 to four
figures.

> ### The strongest points are near the closure's pole and are **not predictions**
>
> D-KIN-1's prompt-jump approximation has a pole at `ρ = β`. The register requires
> every run to report `max ρ/β` and states that anything approaching 1 is *outside
> the approximation, not a result*. The four largest excursions sit at
> **+0.62 to +0.80**, i.e. 62–80% of the way to the pole, where the closure
> degrades and the neglected prompt dynamics stop being negligible.
>
> Read those rows as **the model saying an excursion begins**, not as a prediction
> of its size. A peak of 5.3× at `ρ/β = 0.80` is a tripwire, and quoting it as a
> power level would be exactly the error D-KIN-1 was registered to prevent.
> Resolving them needs the full point kinetics, which is a deviation-register
> change, not a parameter change.

**One non-monotonicity, and it is physical.** At `zeta_sign = 0.90`, raising the
worth from 8e-3 to 1.6e-2 turns the positive sampling **off** — `exercised` goes
yes → no and `L_void` collapses 0.3091 → 0.1019. More void worth means more
*negative* reactivity from the lobe below the sign change, which cuts the power,
which cuts the heating, so the void never climbs high enough to reach the positive
lobe at all. The system is bistable there: moderate worth lets the front into the
positive region, large worth suppresses the front before it arrives.

That is the same mechanism as §10.1's "more worth is safer", seen at the boundary
where it stops holding.

### 10.4 What M9 delivers, and what it does not

**Delivered.** The regime map, both axes, 48 closed-loop points, four regime
classes of which two are populated on each axis. Every point reports `max ρ/β`, as
M9's acceptance requires. The controlling parameter is identified and the boundary
is located.

**Not delivered.** The parametric PINN, and with it M9's remaining acceptance
criteria — that a trained model's classification match this map, and that it beat
re-solving on wall-clock. Not attempted, for the reason in §10: the single-point
network misses its accuracy bar by 7–19×. A surrogate is worth building over a
family a solver finds expensive; here one point costs 6 s, so the reference sweep
above cost about four minutes of wall-clock, and there is nothing for a surrogate
to beat until the base model is accurate.
