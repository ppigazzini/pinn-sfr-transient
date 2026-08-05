# Axial boiling model — physics and deviation register

The second model in this repository: a **1D axially resolved SFR channel** with
sodium boiling, following the SAS4A/SASSYS-1 manual
([ANL/NSE-SAS/5.8.1](https://sas-doc.nse.anl.gov/latest/)). It sits alongside the
lumped 0D model of [`physics_theory.md`](physics_theory.md), which it does not
replace — the 0D model remains the fast regression harness and the pedagogical
entry point.

> **Status: milestone M4.** Implemented: `axial/config.py` — parameters, mesh,
> shapes, Doppler interpolation (M0); `axial/sodium.py` — the thirteen §12.13
> correlations (M1); `axial/physics.py` + `axial/reference.py` — the Chapter 3
> energy balance, its stiff reference solver (M2) and the §12.4 boiling onset
> with a mixture void field (M4); `axial/pinn_torch.py` — the PINN trained on
> those residuals (M3). The film and dryout heat path (M5) and the kinetics
> closure (M6) are **not implemented**, so power is still prescribed and the
> post-dryout response is out of scope — see §7.

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
| Point kinetics | Eq. 4.2-4 | M6 |
| Delayed-neutron precursors | Eq. 4.3-1 | M6 (`beta_i`, `lambda_i`) |
| Decay heat, `ψ_t = ψ_f + ψ_h`, ANS standard | Eq. 4.2-2, §4.4 | M6 (decision pending) |
| **Doppler, logarithmic**, flooded↔voided interpolation | Eq. 4.5-2, 4.5-3 | M0 (`alpha_D`) |
| **Coolant density + void as one worth sum** | Eq. 4.5-25 | M0 (`void_worth`) |
| Boiling onset: saturation + superheat | §12.4 | **done** (`boiling_fraction`) |
| Cladding/structure → vapour heat path | §12.5.1 | M5 |
| Sodium properties | Eq. 12.13-1 … 12.13-13 | **done** (`axial/sodium.py`) |

The chapters are mirrored for offline reading; regenerate with
`python __DEV/sasdoc_fetch.py <outdir>`.

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

### D-TH-2 — Axially uniform flow after voiding *(open — M4)*

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

### D-KIN-2 — Decay heat *(open — needs the owner)*

**Departs from:** Eq. 4.2-2, §4.4. **Status: open.**

The manual splits the power amplitude as `ψ_t = ψ_f + ψ_h`, fission plus decay,
with `ψ_h` from the ANS decay heat power standard. This model currently has only
`ψ_f`.

The omission is structural, not cosmetic: with no decay-heat term the homogeneous
kinetics have **no source**, so `P = c = 0` is an attractor and a sustained
negative reactivity drives power to zero with nothing to stop it. Real fission
product decay heat is ~6–7 % of nominal immediately after shutdown. If decay heat
is adopted, follow §4.4 rather than inventing a two-group model.

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

## 7. Known gaps and honest limits

**Post-dryout response is out of scope (M5).** Once a node reaches `α = 1` the
latent sink correctly switches off — there is no liquid left to boil — and the
wall heat returns to sensible heating. But this model still gives that node
*liquid* heat capacity and liquid advection, so its temperature afterwards is not
physical. Measured symptom: past dryout, the boiling run and a run that cannot
boil at all agree to ~1 K. M5's film and dryout heat path is what closes this,
and a test asserts the symptom so it cannot be mistaken for a result.

**Two energy-conservation defects, both found by the balance check:**

1. The model diverted `b·q_wall` from the coolant but vaporised only
   `b·(1−α)·q_wall`, so energy vanished as a node approached dryout. Fixed by
   using the same `b(1−α)` on both sides.
2. The balance itself omitted the **latent heat convected out of the top with the
   vapour** — about 45 W against 50 kW, i.e. 9e-4, which is exactly where the
   closure floored. Adding it recovered clean `Δt²` convergence to zero.

Neither was visible in the trajectories, which looked entirely plausible
throughout. This is the argument for conservation checks over eyeball validation.

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
| M5 | Film / dryout heat path (§12.5.1) | not started |
| M6 | Prompt-jump kinetics closure — Plan B → Plan A | not started |
| M7–M9 | Hardening, Chapter 12 comparison, parametric sweep | not started |
