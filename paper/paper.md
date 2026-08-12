# A neural surrogate for the sodium-boiling phase of an SFR unprotected loss-of-flow transient: what it reproduces, and what it does not

**P Pigazzini**

Independent researcher — <pasquale.pigazzini@gmail.com>

> **Draft.** Markdown working copy; `paper.typ` is the typeset version. Every number
> here is traceable to a row in `__DEV/studies/`, and the full measurement record —
> including the negative results, the retractions and the machine-learning methodology
> this paper deliberately does not dwell on — is in [`docs/`](../docs/).

## Abstract

We solve the sodium-boiling phase of an unprotected loss-of-flow transient in a
sodium-cooled fast reactor with a physics-informed neural surrogate. One coolant channel is
resolved axially with four material fields and a sodium void fraction, coupled to six-group
point kinetics through Doppler and void-worth integrals, the latter changing sign near the
top of the core. Thermophysics, the boiling criterion and both feedback laws follow the
SAS4A/SASSYS-1 manual, with four registered deviations. Against a verified stiff Radau
reference the surrogate reaches a relative $L_2$ of $1.6\times10^{-3}$ on the temperature
fields, places boiling onset within 0.008 s of 10.9784 s, and reproduces 99.4% of the
peak voided length at a saturation margin of $+67.4$ K against $+69.2$ K. The first two
sit at the reference's own resolution, so we report them as met rather than as measured
accuracies. What the surrogate does not yet deliver is the closed reactivity loop: the
void-worth integral nearly cancels between two large opposite contributions, and driving
the kinetics from the learned fields recovers only 8–16% of it while the non-cancelling
Doppler integral is correct to 1.017. We quantify that cancellation and show by an adjoint
argument that it is not a sampling deficiency.

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
times above the uncertainty of the instrument measuring it [16], so the 1% temperature bar is
sound at a ratio of about 6 and the 0.5 s onset criterion at about 56 — but a *result*
below that uncertainty is measuring the instrument.

The reference terminates when any temperature reaches the top of the §12.13 property fits,
which defines the validity horizon; the surrogate is trained only over that horizon.

![What drives the transient: the pumps coast down to the natural-circulation floor while
the power is still near nominal, so the outlet temperature rises until it meets saturation
plus superheat. The dashed line marks boiling onset.](../docs/img/charts/boundary_conditions.png)

**Onset time is located by root-finding, not by grid inspection.** The reference's own
onset is **10.9784 s**; read off the 0.25 s output grid it appears as 10.75 s. A quarter
of a second of every onset error previously reported for this model was that quantisation.

## 4. Physics-informed neural networks

The surrogate is a physics-informed neural network (PINN), and because this paper is
addressed to reactor physicists rather than to the machine-learning literature, this
section states what that means, what it does not mean, and which of its known weaknesses
govern the design in §5.

### 4.1 The idea

A PINN represents the solution of a differential equation directly as a neural network
$u_\theta$ of the independent variables, and fits it by driving the equation's own
residual to zero [2]. For a system written as $\mathcal{R}[u] = 0$ over the domain, the training objective
is

```math
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}
\bigl\lVert \mathcal{R}[u_\theta](x_i, t_i)\bigr\rVert^2 ,
```

evaluated at $N$ **collocation points** scattered through the domain. The derivatives
$\partial_t u_\theta$ and $\partial_x u_\theta$ that $\mathcal{R}$ needs come from
automatic differentiation of the network itself, so they are exact to machine precision
rather than differenced on a mesh.

Three properties follow, and they are why the method is of interest here.

**No solution data is required.** The objective contains no measured or simulated values
of $u$. The network is fitted to the *equations*, not to a dataset, which distinguishes a
PINN from a regression surrogate trained on solver output. The reference solution of §3 is
used only to **score** the result, never to train it.

**The result is mesh-free and differentiable.** $u_\theta$ is a closed-form function that
can be evaluated anywhere in the domain and differentiated with respect to its inputs —
and, with the same machinery, with respect to physical parameters. That is what makes the
method attractive for uncertainty propagation and parameter inference, which need many
evaluations and gradients rather than one accurate trajectory [3, 4].

**Constraints can be imposed exactly or by penalty.** Initial and boundary conditions are
commonly added as extra penalty terms whose weights must then be chosen. The alternative,
used here, is to build them into the functional form of $u_\theta$ so they hold
identically for any parameters [4].

### 4.2 Why they are hard to train, and what follows for the design

PINNs are not function approximation with a different loss. The objective is a composition
of the network with a differential operator, which makes the optimisation problem
substantially worse conditioned than a supervised one. Four failure modes are established
in the literature [5] and each is visible in this problem.

**Ill-conditioning, and the choice of optimiser.** The Hessian of a PINN objective is
severely ill-conditioned, and first-order stochastic methods stall on it far short of what
the architecture can represent [10, 12]. Quasi-Newton methods, which build curvature from
successive gradients [14], reach accuracies first-order methods do not, and comparative
optimiser studies for PINNs find the same [11, 13]. The consequence here is §5.5: the
optimisation budget, not the architecture, decides whether the boiling front forms at all.

**Spectral bias.** Plain multilayer perceptrons learn low-frequency structure long before
high-frequency structure, which is fatal for a solution containing a sharp moving front.
Mapping the inputs through fixed sinusoidal features removes the bias by presenting the
high frequencies directly to the first layer [6]. Without it no optimiser tested here
forms a front.

**Competing loss terms.** When initial conditions, boundary conditions and the residual
are summed as penalties their gradients compete, and the balance shifts during training
[7, 8]. The remedy adopted here is to remove the competition rather than tune it, by
making the constraints exact (§5.1).

**Where the points go.** The residual is minimised only where it is sampled, so the
collocation distribution is part of the method [9]. For a front problem the instinct is to
concentrate points on the front; §7 shows by an adjoint argument that for the functional
this paper cares about, that instinct is wrong.

### 4.3 What a PINN result must be compared against

A recurring criticism of the machine-learning-for-PDEs literature is that reported
speed-ups and accuracies are measured against weak baselines and reported selectively
[19]. This paper follows the opposite convention: the baseline is a verified stiff solver
of the same equations, its own discretisation error is quantified first (§3), and results
at or below that error are reported as **met** rather than as measured accuracies.

## 5. The surrogate

### 5.1 Hard-constraint ansatz

The state is not the network output. It is written multiplicatively,

```math
\theta(\zeta,\hat{t}) = \theta_0(\zeta)\exp\bigl(\hat{t}\,N(\zeta,\hat{t})\bigr),
```

where $\theta_0$ is the analytic steady profile of §2, $\hat{t} = t/t_{\mathrm{end}}$, and
$N$ is the network. Three constraints then hold for *any* weights, exactly and without a
penalty term:

- at $\hat{t} = 0$ the exponent vanishes and $\theta = \theta_0$, so the initial condition
  is the steady state to machine precision;
- the exponential is positive, so no temperature can fall below the inlet — which matters
  because the Doppler feedback is logarithmic in $T_f$ and undefined for non-positive
  arguments;
- multiplying by a factor vanishing at $\zeta = 0$ pins the single upstream boundary
  condition the advection equation admits.

This is not a stylistic preference. An additive ansatz was tried first; the optimiser drove
the fuel temperature negative while the loss fell, at which point the logarithmic Doppler
term is undefined. Exact constraints also remove the loss-balancing problem of §4.2
entirely, since there are no constraint penalties to balance against the residual.

### 5.2 Input embedding

The two inputs $(\zeta, \hat{t})$ pass through a random Fourier feature map [6],
$x \mapsto [\sin(2\pi Bx), \cos(2\pi Bx)]$, with $B$ drawn once from a Gaussian and held
fixed — a change of input coordinates, not a fitted layer. The default uses **64
features**, giving a 128-component embedding.

Wider embeddings were measured against this default at the same budget, three seeds each.
**128 and 256 features are indistinguishable from 64 on every quantity the reference can
resolve** — the same relative $L_2$ to four digits, the same 99.4% of peak voided length,
and onset errors all below the reference's own uncertainty — while costing roughly 1.25 and
1.9 times as much per iteration. The narrowest of the three is therefore the default. A
wider embedding does help at a *starved* optimisation budget, where it both improves onset
accuracy and sharply reduces its scatter across seeds; at the funded budget of §5.5 that
advantage has gone.

### 5.3 Network

The trunk is a 64-wide, 5-layer tanh multilayer perceptron with five outputs, one per
field. Its parameter count is worth stating carefully, because two different numbers can
be quoted:

| component | parameters |
|---|---|
| trunk (5 layers, 64 wide, 5 outputs) | **17 029** |
| read-out from the embedding ($64 \times 128$) | 8 192 |
| frozen Fourier matrix $B$ | 128 |
| total arrays | 25 349 |

The trunk's 17 029 parameters are invariant to the embedding width; a 256-feature
embedding changes only the read-out, from 8192 to 32 768. That column is not fitting
capacity, which is why widening it changes nothing measurable.

### 5.4 Residuals and collocation

The loss applies the objective of §4.1 to the four field equations of §2 together with the
void closure, each block scaled to a common magnitude so no field dominates the sum by
virtue of its units. The collocation set is drawn once over
$(\zeta, \hat{t}) \in [0,1]^2$ and held fixed for the whole solve; the default is **5000
points**.

The count is a measured choice. At 2000 points a front still forms but is badly placed —
onset 0.19 s late, and a saturation margin of +28 K against the reference's +69 K — while
5000 is where the onset error falls below the reference's own resolution. Beyond it
nothing measurable changes.

### 5.5 Training

Training is **50 000 L-BFGS iterations** [14] with a strong-Wolfe line search, and **no
first-order stage at all**. All arithmetic is in double precision; curvature pairs are
meaningless at single-precision residual magnitudes.

The absence of a first-order stage is deliberate and measured. A short Adam warm start
neither helps nor is free: removing it improves the converged result, and a schedule-free
first-order method run alone for the same 30 000 iterations produces no boiling front at
all. Redrawing the collocation set during the solve costs a factor of 1.5 against holding
it fixed, because the curvature pairs a quasi-Newton method accumulates are only
meaningful while the objective is unchanged. The supporting measurements are in the
repository documentation and are not reproduced here.

## 6. What the surrogate reproduces

All results at three seeds unless stated, against the reference at its scoring mesh.

### 6.1 Temperature fields

The relative $L_2$ error on the film temperature is $1.6 \times 10^{-3}$ (range
1.62–1.65 × 10⁻³), against a 1% acceptance bar. Because the reference's own error is
1.1–1.6 × 10⁻³, the surrogate has reached the resolution of the instrument judging it: the
bar is met, and *how far inside* it is not a question this reference can answer. The other
fields follow at 1.7 × 10⁻³ (coolant), 2.6 × 10⁻³ (fuel) and 3.7 × 10⁻³ (cladding).

### 6.2 The boiling front

![Coolant temperature over space and time. The cyan contour is the boiling criterion,
which under D-TH-3 *is* the front rather than a separately computed curve; the star marks
onset, located by tangency.](../docs/img/charts/temperature_map.png)

![Void fraction over space and time, with axial profiles at five instants and at onset.
The front forms at the outlet and propagates downward.](../docs/img/charts/vapor_fraction.png)

![Front height and voided length. The saturation level set and the $\alpha > 0.5$ contour
do not coincide; the gap is the partially voided region the worth integral is most
sensitive to.](../docs/img/charts/front_height.png)

The engineering quantities are reproduced:

| quantity | surrogate | reference |
|---|---|---|
| peak voided length | **99.4%** of reference | — |
| worst-seed saturation margin | **+67.4 K** | +69.2 K |
| boiling onset time | **0.0042 / 0.0046 / 0.0082 s** error | 10.9784 s |

The onset criterion of 0.5 s is met on every seed, with the worst at 0.008 s. **Every seed
now sits below the reference's own onset uncertainty of 0.009 s**, so onset is reported as
met and is no longer separable from the instrument either; the seed spread has fallen to
2.0× from the 32× of the earlier configuration. Met is what the measurement supports.

**Onset height is not a discriminating quantity in this model.** The coolant heats
monotonically up the channel, so the hottest point — and therefore where boiling begins —
is always the outlet, for the surrogate and the reference alike. A height criterion here
measures the mesh, not the model. An earlier revision of this work reported that quantity
as solved exactly; it had merely been restated as a tautology.

## 7. What is not yet solved: the closed void-reactivity loop

With the thermal fields prescribed, the surrogate is accurate. Driving the **kinetics**
from the learned fields is a different matter, and it fails in a specific and instructive
way.

> The figures in this section were measured on the earlier default configuration
> (256 Fourier features, a short Adam stage, 30 000 quasi-Newton iterations on 6000
> points) and have not been re-measured on the configuration of §5. The cancellation
> ratio is a property of the worth distribution and does not depend on the surrogate; the
> recovered fractions may move.

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

![The void worth $w(\zeta)$, shaded by sign — positive over most of the core and negative
near the top, so the reactivity functional is a difference of two large contributions
rather than a sum of small ones.](../docs/img/charts/void_worth_split.png)

![Reactivity split by mechanism, in units of $\beta_{eff}$. Reporting only the net would
hide which component carries the error.](../docs/img/charts/reactivity.png)

A relative error $\epsilon$ on each half therefore becomes $2.1 \epsilon$ on the sum. The
functional is an ill-conditioned target by construction, and reporting it as one number
hides which half is wrong; we report the halves separately from here on.

**It is not a sampling problem.** Dual-weighted-residual theory [15] gives the error in a
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

## 8. Discussion

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
$1.6\times10^{-3}$ on temperature coexists here with an 84–92% miss on a reactivity
integral built from the same field. Second, **the reference's own resolution must be
quantified before an acceptance bar is set** [17, 18]. Two bars in this work were set without that
check; one was withdrawn, and the temperature result has since reached the point where the
instrument, not the model, is the limit.

## 9. Conclusions

A physics-informed neural surrogate for the sodium-boiling phase of an SFR unprotected
loss-of-flow transient, built on SAS4A/SASSYS-1 thermophysics and feedback laws with four
registered deviations, reproduces the reference solution's temperature fields to
$1.6 \times 10^{-3}$ relative $L_2$, its boiling onset to within 0.008 s of 10.9784 s, and
99.4% of its peak voided length with a saturation margin of +67.4 K against +69.2 K. The
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
3. Karniadakis G E, Kevrekidis I G, Lu L, Perdikaris P, Wang S and Yang L 2021
   Physics-informed machine learning *Nat. Rev. Phys.* **3** 422
4. Lu L, Meng X, Mao Z and Karniadakis G E 2021 DeepXDE: a deep learning library for
   solving differential equations *SIAM Rev.* **63** 208
5. Krishnapriyan A S, Gholami A, Zhe S, Kirby R M and Mahoney M W 2021 Characterizing
   possible failure modes in physics-informed neural networks *Adv. Neural Inf. Process.
   Syst.* **34** 26548
6. Tancik M, Srinivasan P P, Mildenhall B, Fridovich-Keil S, Raghavan N, Singhal U,
   Ramamoorthi R, Barron J T and Ng R 2020 Fourier features let networks learn high
   frequency functions in low dimensional domains *Adv. Neural Inf. Process. Syst.*
   **33** 7537
7. Wang S, Teng Y and Perdikaris P 2021 Understanding and mitigating gradient flow
   pathologies in physics-informed neural networks *SIAM J. Sci. Comput.* **43** A3055
8. Wang S, Sankaran S and Perdikaris P 2024 Respecting causality for training
   physics-informed neural networks *Comput. Methods Appl. Mech. Eng.* **421** 116813
9. Wu C, Zhu M, Tan Q, Kartha Y and Lu L 2023 A comprehensive study of non-adaptive and
   residual-based adaptive sampling for physics-informed neural networks *Comput. Methods
   Appl. Mech. Eng.* **403** 115671
10. Rathore P, Lei W, Frangella Z, Lu L and Udell M 2024 Challenges in training PINNs: a
    loss landscape perspective *Proc. 41st Int. Conf. on Machine Learning*
11. Kiyani E, Shukla K, Urbán J F, Darbon J and Karniadakis G E 2025 Which optimizer works
    best for physics-informed neural networks and Kolmogorov-Arnold networks? Preprint
    arXiv:2501.16371
12. Urbán J F, Stefanou P and Pons J A 2025 Unveiling the optimization process of
    physics-informed neural networks *J. Comput. Phys.* **523** 113656
13. Müller J and Zeinhofer M 2023 Achieving high accuracy with PINNs via energy natural
    gradient descent *Proc. 40th Int. Conf. on Machine Learning* **202** 25471
14. Nocedal J and Wright S J 2006 *Numerical Optimization* 2nd edn (New York: Springer)
15. Becker R and Rannacher R 2001 An optimal control approach to a posteriori error
    estimation in finite element methods *Acta Numerica* **10** 1
16. Taylor B N and Kuyatt C E 1994 *Guidelines for Evaluating and Expressing the
    Uncertainty of NIST Measurement Results* NIST Technical Note 1297
17. Jakeman J D, Barba L A, Martins J R R A and O'Leary-Roseberry T 2026 Verification and
    validation for trustworthy scientific machine learning *Mach. Learn.: Sci. Technol.*
    **7** 025055
18. Eça L and Hoekstra M 2014 A procedure for the estimation of the numerical uncertainty
    of CFD calculations based on grid refinement studies *J. Comput. Phys.* **262** 104
19. McGreivy N and Hakim A 2024 Weak baselines and reporting biases lead to overoptimism
    in machine learning for fluid-related partial differential equations *Nat. Mach.
    Intell.* **6** 1256
