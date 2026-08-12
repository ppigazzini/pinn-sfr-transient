# A neural surrogate for the sodium-boiling phase of an SFR unprotected loss-of-flow transient: what it reproduces, and what it does not

**P Pigazzini**

Independent researcher — <pasquale.pigazzini@gmail.com>

> **Draft.** Markdown working copy; `paper.typ` is the typeset version. Every number
> here is traceable to a row in `__DEV/studies/`, and the full measurement record —
> including the negative results, the retractions and the machine-learning methodology
> this paper deliberately does not dwell on — is in [`docs/`](../docs/).

## Abstract

We solve the sodium-boiling phase of an unprotected loss-of-flow (ULOF) transient in a
sodium-cooled fast reactor with a physics-informed neural surrogate, and report it as a
reactor-physics result rather than a machine-learning one. One coolant channel is
resolved axially with four material fields — fuel, cladding, film and coolant temperature
— and a sodium void fraction, coupled to six-group point kinetics through a logarithmic
Doppler integral and a void-worth integral whose weight changes sign near the top of the
core. Thermophysics, the saturation-plus-superheat boiling criterion and both feedback
laws are taken from the SAS4A/SASSYS-1 manual; four deviations from it are registered
with their justification, of which two are load-bearing. Against a verified stiff Radau
reference the surrogate reproduces the temperature fields to a relative $L_2$ of
$1.7 \times 10^{-3}$, places boiling onset within 0.018 s of the reference's 10.9784 s,
and reproduces 99.3% of the peak voided length with a saturation margin of +67.6 K
against +69.2 K. Both of the first two now sit at the reference solver's own resolution,
so we report them as met rather than as measured accuracies. What the surrogate does
**not** yet deliver is the closed reactivity loop: the void-worth integral is a
near-cancellation of two large opposite contributions, and driving the kinetics from the
learned fields recovers only 8–16% of it, while the non-cancelling Doppler integral over
the same fields is correct to 1.017. We show by a dual-weighted-residual argument that
this is not a sampling deficiency, quantify the cancellation, and identify the closed-loop
void reactivity as the outstanding physics problem.

## 1. Introduction

In an unprotected loss-of-flow accident the primary pumps coast down while the reactor
protection system fails to scram. Coolant flow decays, the sodium outlet temperature
rises, and if saturation plus the required superheat is reached the coolant boils. In a
sodium-cooled fast reactor the resulting voiding carries a **positive** reactivity
contribution over most of the core height, and it competes with the prompt negative
Doppler feedback. Whether the excursion terminates benignly is decided by the sign and
timing of that competition, which makes the *onset* and *axial extent* of boiling the
quantities of engineering interest, not merely a field norm.

Whole-plant analysis of this sequence is the province of system codes — SAS4A/SASSYS-1
being the reference of record. A differentiable surrogate for the boiling phase is
attractive for uncertainty propagation, for parameter inference from limited
instrumentation, and for embedding a channel model inside an optimisation loop, all of
which need many evaluations and derivatives with respect to inputs.

This paper reports such a surrogate and is explicit about the division of labour. The
machine-learning system is described only as far as needed to reproduce it (§4); the
detail is spent on the physics being solved (§2), on the reference against which it is
judged and the resolution that reference actually has (§3), on what the surrogate
reproduces (§5), and on the one coupling it does not yet close (§6). The complete
methodological record — architecture sweeps, optimiser comparisons, seed statistics,
cross-implementation checks and the negative results — is published alongside in the
repository documentation and is cited rather than reproduced here.

## 2. Physical model

### 2.1 Geometry and state

A single coolant channel is resolved along its axis, $\zeta \in [0, 1]$, with radially
lumped materials. The state at each height is four temperatures — fuel $T_f$, cladding
$T_{cl}$, film $T_s$ and coolant $T_c$ — together with a void fraction $\alpha$. Radial
lumping (D-GEOM-1) and retention of a lumped structure node (D-GEOM-2) follow the
manual's own reduced treatment.

### 2.2 Governing equations

Fuel and cladding conduction follow manual Eq. 3.3-1, with the fuel-to-cladding flux of
Eq. 3.3-4 carrying **both** gap conductance and radiation. The coolant energy equation is
Eq. 3.3-5 with its three sources $Q_c + Q_{ec} + Q_{sc}$, and includes direct
neutron and gamma heating $\gamma_c$ (Eq. 3.3-6). Coolant transport is advective with a
prescribed flow decay,

```math
f(t) = f_{\mathrm{nc}} + (1 - f_{\mathrm{nc}})\, e^{-t / \tau_{\mathrm{pump}}},
```

with $f_{\mathrm{nc}}$ the natural-circulation floor. Pre-boiling momentum uses the
manual's Eq. 3.9-1 result that $w = w(t)$ is independent of height.

### 2.3 Sodium properties and the boiling criterion

All thirteen numbered relations of manual §12.13 are implemented. Their stated validity
is **590–2270 K**, the fits stopping near 90% of the critical point (2503.3 K) because
$C_l$, $\beta_s$ and $\alpha_p$ all contain $1/(T_c - T)$ and diverge there. The
implementation neither clamps nor raises — a hard guard inside a residual would abort a
differentiable solve the moment a transient overshot — and instead exposes a range check
that the reference solution asserts.

Boiling onset is the saturation-plus-superheat criterion of §12.4: the coolant boils
where $T_c > T_{\mathrm{sat}}(p) + \Delta T_{\mathrm{sup}}$. The cladding-and-structure to
vapour heat path follows §12.5.1.

### 2.4 Neutronics and feedback

Power follows point kinetics with six delayed-neutron precursor groups (Eq. 4.3-1) closed
by the prompt-jump approximation (Eq. 4.2-4). Two feedbacks couple the thermal fields back
to the power, both as **axial integrals**:

- **Doppler**, logarithmic in fuel temperature and interpolated between flooded and voided
  states (Eq. 4.5-2, 4.5-3);
- **coolant density and void as a single worth distribution** (Eq. 4.5-25),

```math
\rho_{\mathrm{void}}(t) = \int_0^1 w(\zeta)\, \alpha(\zeta, t)\, \mathrm{d}\zeta ,
```

where the worth $w$ is **positive over most of the core and negative near the top**. That
sign change is the physical origin of the difficulty reported in §6.

### 2.5 Registered deviations from the manual

Every departure from SAS4A is registered with its justification; four matter here.

**D-KIN-1, prompt jump.** The prompt neutron mode is eliminated algebraically. Standard
practice, and it removes the stiffest timescale from the system.

**D-TH-3, the void slaved to temperature.** Rather than tracking vapour dynamics, the void
fraction is closed algebraically on the coolant temperature,

```math
\alpha = 1 - \bigl(1 - b(T_c)\bigr)^3 ,
```

with $b$ the superheat switch. Justified by timescale separation — vapour fills a node in
0.71 ms against an advective 0.113 s — and selected from six candidate closures on maximum
voided-length error, at 1.2% against the reference's own 2.6% mesh error. The cubic form
is chosen for its finite slope at $b = 0$; a square-root closure has unbounded slope there.
**This deviation is what makes the boiling front form at all** in a differentiable
formulation.

**D-TH-1, Eulerian mixture void** rather than Lagrangian slug tracking, and **D-FLOW-1,
prescribed flow** rather than prescribed pump head.

Registered but **off by default**, so no result here depends on them: condensation
(D-TH-4), three-group decay heat (D-KIN-3), axial fuel expansion (D-FB-4). Vapour
expansion accelerating the flow (D-TH-2) is implemented but not usable. Five further
feedback mechanisms are omitted and recorded as such (D-FB-3), and there is no pin failure
or material relocation (D-SCOPE-1) — which bounds the transient to its pre-failure phase.

## 3. Reference solution and its resolution

The reference is a stiff Radau integration on an axial mesh, and it is the instrument
every claim below is measured against, so its own resolution bounds what can be claimed.

It reproduces the analytic steady state to $3.4 \times 10^{-11}$ K/s and conserves energy
at second order in the step. Refining it against itself:

| quantity | reference's own error at the scoring mesh |
|---|---|
| temperatures, relative $L_2$ | 1.1–1.6 × 10⁻³ |
| boiling onset time | 0.009 s |
| onset height | 0.06 cells |
| peak voided length | 0.57% |
| **pointwise void fraction** | **3.2 × 10⁻²** |

Two consequences are load-bearing. First, **an acceptance bar on the pointwise void
fraction is not supportable** and was withdrawn: the reference does not know that quantity
to better than 3%. Second, calibration practice requires a tolerance to sit at least four
times above the uncertainty of the instrument measuring it, so the 1% temperature bar is
sound at a ratio of about 6 and the 0.5 s onset criterion at about 56 — but a *result*
below that uncertainty is measuring the instrument.

The reference terminates when any temperature reaches the top of the §12.13 property fits,
which defines the validity horizon; the surrogate is trained only over that horizon.

**Onset time is located by root-finding, not by grid inspection.** The reference's own
onset is **10.9784 s**; read off the 0.25 s output grid it appears as 10.75 s. A quarter
of a second of every onset error previously reported for this model was that quantisation.

## 4. The surrogate

Described here only as far as reproduction requires; the design study is in the
repository documentation.

The state is written multiplicatively,
$\theta(\zeta, \hat{t}) = \theta_0(\zeta)\exp(\hat{t} N(\zeta, \hat{t}))$, with $\theta_0$
the analytic steady profile. This makes the initial condition exact, keeps every
temperature at or above the inlet, and pins the single upstream boundary condition the
advection equation admits — all without penalty terms. An additive ansatz, tried first,
allowed the optimiser to drive the fuel temperature negative, at which point the
logarithmic Doppler term is undefined.

Inputs pass through a random Fourier embedding
$x \mapsto [\sin(2\pi Bx), \cos(2\pi Bx)]$ with $B$ frozen. The trunk is a 64-wide,
5-layer tanh MLP with five outputs, carrying 17 029 fitted parameters; the embedding's
read-out adds a further 32 768 that scale with the embedding width and not with what the
network can represent.
Training minimises the squared PDE residuals at collocation points drawn fresh each step,
with a short Adam stage followed by a long L-BFGS stage with strong-Wolfe line search. All
arithmetic is float64 on CPU — curvature pairs are meaningless at float32 residual
magnitudes.

Two independent implementations exist, in PyTorch and JAX/Equinox, required by test to
expose identical knobs and defaults and sharing only the numpy definition of the physics.
Results below are from the JAX implementation; the cross-implementation agreement and its
limits are documented in the repository.

## 5. What the surrogate reproduces

All results at three seeds unless stated, against the reference at its scoring mesh.

### 5.1 Temperature fields

The relative $L_2$ error on the film temperature is $1.7 \times 10^{-3}$ (range
1.6–1.7 × 10⁻³), against a 1% acceptance bar. Because the reference's own error is
1.1–1.6 × 10⁻³, the surrogate has reached the resolution of the instrument judging it: the
bar is met, and *how far inside* it is not a question this reference can answer.

### 5.2 The boiling front

The engineering quantities are reproduced:

| quantity | surrogate | reference |
|---|---|---|
| peak voided length | **99.3%** of reference | — |
| worst-seed saturation margin | **+67.6 K** | +69.2 K |
| boiling onset time | **0.0006 / 0.0064 / 0.0181 s** error | 10.9784 s |

The onset criterion of 0.5 s is met on every seed, with the worst at 0.018 s. Two caveats
belong in the same sentence as the result: the seed spread on onset is 32×, the widest of
any quantity measured here, and the reference's own onset uncertainty is 0.009 s, so the
worst seed sits at twice the instrument's resolution and the best below it. Met is what
the measurement supports.

**Onset height is not a discriminating quantity in this model.** The coolant heats
monotonically up the channel, so the hottest point — and therefore where boiling begins —
is always the outlet, for the surrogate and the reference alike. A height criterion here
measures the mesh, not the model. An earlier revision of this work reported that quantity
as solved exactly; it had merely been restated as a tautology.

## 6. What is not yet solved: the closed void-reactivity loop

With the thermal fields prescribed, the surrogate is accurate. Driving the **kinetics**
from the learned fields is a different matter, and it fails in a specific and instructive
way.

Over the same fields and the same network, the Doppler integral is reproduced to a factor
of **1.017**. The void integral recovers only **8–16%** of the reference's value. The two
differ in one respect: the Doppler weight has one sign, and the void worth changes sign
near the top of the core.

Splitting the void functional at the sign change, at the instant it peaks:

| | value |
|---|---|
| positive-worth region, `J+` | +4.656 × 10⁻⁴ |
| negative-worth region, `J-` | −1.695 × 10⁻⁴ |
| their sum, `J` | +2.962 × 10⁻⁴ |
| cancellation ratio, `|J| / (|J+| + |J-|)` | **0.466** |

A relative error $\epsilon$ on each half therefore becomes $2.1 \epsilon$ on the sum. The
functional is an ill-conditioned target by construction, and reporting it as one number
hides which half is wrong; we report the halves separately from here on.

**It is not a sampling problem.** Dual-weighted-residual theory gives the error in a
functional as $\langle R(u_\theta), z^*\rangle$ to leading order, with $z^*$ the solution
of the adjoint problem sourced by the functional's derivative. For this advective coolant
operator the adjoint runs backwards in $\zeta$, and it can be evaluated in closed form.
The result is a **step**: the void slope underflows to exactly zero wherever the coolant is
subcooled, so $z^*$ is constant over the lower 72% of the channel and zero above it. Every
point below the front carries equal sensitivity and every point above it carries none.
Residual-magnitude sampling concentrates points *at* the front — precisely where the
functional is insensitive — so it cannot help, and a uniform sampler is already near
optimal.

Evaluated open-loop on the surrogate's own field, the functional is in fact accurate:
the positive half to 1.66–2.66% and the negative half exactly, the latter because the negative-worth region is
fully voided in surrogate and reference alike and the integral is then fixed by geometry.
**The deficit therefore arises in the closed loop**, where $\rho_{\mathrm{void}}$ feeds
back into the kinetics and the error compounds — not in the evaluation of the functional
on a given field. That localises the remaining work precisely.

## 7. Discussion

The surrogate reproduces the thermal-hydraulic phase of this transient to the resolution
of the reference solver, including the safety-relevant front quantities: when boiling
starts, how far it extends, and by what margin. For applications that consume those
fields — uncertainty propagation over thermophysical parameters, inference of boundary
conditions from limited instrumentation, gradient-based studies of the coastdown — the
model is usable now, and its derivatives with respect to inputs are exact by construction.

For applications that require the **feedback loop closed**, it is not. The obstruction is
not accuracy in any ordinary sense — the fields are right and the open-loop functional is
right — but the conditioning of a near-cancelling integral inside a feedback path. That is
a property of the physics: a positive void worth over most of the core against a negative
worth near the top is precisely what makes SFR void feedback interesting, and it is
precisely what makes it a hard target for a surrogate. Any neural approach to sodium void
feedback will meet the same 2.1× amplification.

Two consequences for practice. First, **a surrogate should be qualified against the
functionals it will be used for, not only against field norms**: an $L_2$ of
$1.7\times10^{-3}$ on temperature coexists here with an 84–92% miss on a reactivity
integral built from the same field. Second, **the reference's own resolution must be
quantified before an acceptance bar is set**. Two bars in this work were set without that
check; one was withdrawn, and the temperature result has since reached the point where the
instrument, not the model, is the limit.

## 8. Conclusions

A physics-informed neural surrogate for the sodium-boiling phase of an SFR unprotected
loss-of-flow transient, built on SAS4A/SASSYS-1 thermophysics and feedback laws with four
registered deviations, reproduces the reference solution's temperature fields to
$1.7 \times 10^{-3}$ relative $L_2$, its boiling onset to within 0.018 s of 10.9784 s, and
99.3% of its peak voided length with a saturation margin of +67.6 K against +69.2 K. The
first two are at the reference's own resolution and are reported as met rather than as
measured accuracies.

The closed void-reactivity loop is not reproduced, recovering 8–16% of the reference
integral while the non-cancelling Doppler integral over the same fields is correct to
1.017. The cause is quantified — a cancellation ratio of 0.466, amplifying any error on
either half by 2.1× — and shown by an adjoint argument not to be a sampling deficiency.
Closing that loop is the outstanding problem for this model, and we expect it to be the
outstanding problem for neural surrogates of sodium void feedback generally.

## Acknowledgments

The SAS4A/SASSYS-1 documentation maintained by Argonne National Laboratory was the sole
source for the thermophysics and feedback laws used here.

## References

1. Fanning T H (ed) 2017 *The SAS4A/SASSYS-1 Safety Analysis Code System* ANL/NE-16/19
   (Argonne, IL: Argonne National Laboratory)
2. Raissi M, Perdikaris P and Karniadakis G E 2019 Physics-informed neural networks: a
   deep learning framework for solving forward and inverse problems involving nonlinear
   partial differential equations *J. Comput. Phys.* **378** 686
3. Becker R and Rannacher R 2001 An optimal control approach to a posteriori error
   estimation in finite element methods *Acta Numerica* **10** 1
4. Tancik M *et al* 2020 Fourier features let networks learn high frequency functions in
   low dimensional domains *Adv. Neural Inf. Process. Syst.* **33** 7537
5. Nocedal J and Wright S J 2006 *Numerical Optimization* 2nd edn (New York: Springer)
6. Taylor B N and Kuyatt C E 1994 *Guidelines for Evaluating and Expressing the
   Uncertainty of NIST Measurement Results* NIST Technical Note 1297
7. Jakeman J D, Barba L A, Martins J R R A and O'Leary-Roseberry T 2026 Verification and
   validation for trustworthy scientific machine learning *Mach. Learn.: Sci. Technol.*
   **7** 025055
8. Eça L and Hoekstra M 2014 A procedure for the estimation of the numerical uncertainty
   of CFD calculations based on grid refinement studies *J. Comput. Phys.* **262** 104
