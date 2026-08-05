# Axial boiling model — physics and deviation register

The second model in this repository: a **1D axially resolved SFR channel** with
sodium boiling, following the SAS4A/SASSYS-1 manual
([ANL/NSE-SAS/5.8.1](https://sas-doc.nse.anl.gov/latest/)). It sits alongside the
lumped 0D model of [`physics_theory.md`](physics_theory.md), which it does not
replace — the 0D model remains the fast regression harness and the pedagogical
entry point.

> **Status: milestone M0 (scaffolding).** Implemented so far:
> `src/pinn_sfr_transient/axial/config.py` — the parameter container, the axial
> mesh, the power and void-worth shapes, and the Doppler coefficient
> interpolation. The reference solver (M2), boiling onset (M4) and the kinetics
> closure (M6) are **not implemented**. Nothing here is validated against a
> reference yet; do not quote numbers from it.

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
| Fuel and cladding conduction | Eq. 3.3-1 | M2 |
| Fuel-to-cladding flux (gap conductance **+ radiation**) | Eq. 3.3-4 | M2 |
| Coolant energy, three sources `Q_c + Q_ec + Q_sc` | Eq. 3.3-5 | M2 |
| Direct neutron/gamma heating in the coolant, `γ_c` | Eq. 3.3-6 | M2 (`gamma_c`) |
| Pre-boiling momentum, `w = w(t)` independent of `z` | Eq. 3.9-1 | M2/M3 |
| Point kinetics | Eq. 4.2-4 | M6 |
| Delayed-neutron precursors | Eq. 4.3-1 | M6 (`beta_i`, `lambda_i`) |
| Decay heat, `ψ_t = ψ_f + ψ_h`, ANS standard | Eq. 4.2-2, §4.4 | M6 (decision pending) |
| **Doppler, logarithmic**, flooded↔voided interpolation | Eq. 4.5-2, 4.5-3 | M0 (`alpha_D`) |
| **Coolant density + void as one worth sum** | Eq. 4.5-25 | M0 (`void_worth`) |
| Boiling onset: saturation + superheat | §12.4 | M4 (`dT_superheat`) |
| Cladding/structure → vapour heat path | §12.5.1 | M5 |
| Sodium properties | Eq. 12.13-1 … 12.13-13 | M1 |

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

## 4. Parameter provenance

**The defaults in `AxialParams` are representative placeholders**, order-of-
magnitude correct for an oxide-fuelled SFR pin cell and not taken from any
specific reactor. Chapter 2 of the manual (the input-deck reference) is the place
to source a consistent realistic set; that is M2 work. Until then, nothing from
this model should be quoted as a physical prediction.

## 5. Milestone status

| M | Deliverable | State |
|---|---|---|
| **M0** | Package scaffolding, `AxialParams`, axial shapes, this register | **done** |
| M1 | Sodium properties (§12.13) from the manual | not started |
| M2 | Method-of-lines reference solver, held-out truth | not started |
| M3 | Plan B PINN — prescribed power, no feedback | not started |
| M4 | Boiling onset and void field | not started |
| M5 | Film / dryout heat path (§12.5.1) | not started |
| M6 | Prompt-jump kinetics closure — Plan B → Plan A | not started |
| M7–M9 | Hardening, Chapter 12 comparison, parametric sweep | not started |
