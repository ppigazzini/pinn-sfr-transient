// Typeset copy, laid out to __DEV/paper_template.docx via iop-jpcs.typ.
//
//   typst compile paper/paper.typ
//
// Kept in step with paper.md and paper.tex by hand. Where they disagree,
// docs/axial_nn.md holds the measurement record and all three are wrong until
// reconciled against it.

#import "iop-jpcs.typ": jpcs, references

#show: jpcs.with(
  title: [A neural surrogate for the sodium-boiling phase of an SFR unprotected
  loss-of-flow transient: what it reproduces, and what it does not],
  authors: ((name: "P Pigazzini", aff: 1),),
  affiliations: ("Independent researcher",),
  email: "pasquale.pigazzini@gmail.com",
  abstract: [We solve the sodium-boiling phase of an unprotected loss-of-flow (ULOF)
  transient in a sodium-cooled fast reactor with a physics-informed neural surrogate, and
  report it as a reactor-physics result rather than a machine-learning one. One coolant
  channel is resolved axially with four material fields — fuel, cladding, film and coolant
  temperature — and a sodium void fraction, coupled to six-group point kinetics through a
  logarithmic Doppler integral and a void-worth integral whose weight changes sign near the
  top of the core. Thermophysics, the saturation-plus-superheat boiling criterion and both
  feedback laws are taken from the SAS4A/SASSYS-1 manual; four deviations from it are
  registered with their justification, of which two are load-bearing. Against a verified
  stiff Radau reference the surrogate reproduces the temperature fields to a relative $L_2$
  of $1.6 times 10^(-3)$, places boiling onset within 0.008 s of the reference's 10.9784 s,
  and reproduces 99.4% of the peak voided length with a saturation margin of $+67.4$ K
  against $+69.2$ K. Both of the first two now sit at the reference solver's own
  resolution, so we report them as met rather than as measured accuracies. What the
  surrogate does _not_ yet deliver is the closed reactivity loop: the void-worth integral
  is a near-cancellation of two large opposite contributions, and driving the kinetics from
  the learned fields recovers only 8–16% of it, while the non-cancelling Doppler integral
  over the same fields is correct to 1.017. We show by a dual-weighted-residual argument
  that this is not a sampling deficiency, quantify the cancellation, and identify the
  closed-loop void reactivity as the outstanding physics problem.],
)

= Introduction

In an unprotected loss-of-flow accident the primary pumps coast down while the reactor
protection system fails to scram. Coolant flow decays, the sodium outlet temperature
rises, and if saturation plus the required superheat is reached the coolant boils. In a
sodium-cooled fast reactor the resulting voiding carries a _positive_ reactivity
contribution over most of the core height, and it competes with the prompt negative
Doppler feedback. Whether the excursion terminates benignly is decided by the sign and
timing of that competition, which makes the _onset_ and _axial extent_ of boiling the
quantities of engineering interest, not merely a field norm.

Whole-plant analysis of this sequence is the province of system codes — SAS4A/SASSYS-1
being the reference of record [1]. A differentiable surrogate for the boiling phase is
attractive for uncertainty propagation, for parameter inference from limited
instrumentation, and for embedding a channel model inside an optimisation loop, all of
which need many evaluations and derivatives with respect to inputs.

This paper reports such a surrogate and is explicit about the division of labour. The
machine-learning system is described only as far as needed to reproduce it (@sec:surrogate);
the detail is spent on the physics being solved (@sec:physics), on the reference against
which it is judged and the resolution that reference actually has (@sec:reference), on what
the surrogate reproduces (@sec:results), and on the one coupling it does not yet close
(@sec:openloop). The complete methodological record — architecture sweeps, optimiser
comparisons, seed statistics, cross-implementation checks and the negative results — is
published alongside in the repository documentation and is cited rather than reproduced
here.

= Physical model <sec:physics>

== Geometry and state

A single coolant channel is resolved along its axis, $zeta in [0, 1]$, with radially
lumped materials. The state at each height is four temperatures — fuel $T_f$, cladding
$T_"cl"$, film $T_s$ and coolant $T_c$ — together with a void fraction $alpha$. Radial
lumping (D-GEOM-1) and retention of a lumped structure node (D-GEOM-2) follow the manual's
own reduced treatment.

== Governing equations

Fuel and cladding conduction follow manual Eq. 3.3-1, with the fuel-to-cladding flux of
Eq. 3.3-4 carrying _both_ gap conductance and radiation. The coolant energy equation is
Eq. 3.3-5 with its three sources $Q_c + Q_"ec" + Q_"sc"$, and includes direct neutron and
gamma heating $gamma_c$ (Eq. 3.3-6). Coolant transport is advective with a prescribed flow
decay,

$ f(t) = f_"nc" + (1 - f_"nc") e^(-t \/ tau_"pump") , $

with $f_"nc"$ the natural-circulation floor. Pre-boiling momentum uses the manual's
Eq. 3.9-1 result that $w = w(t)$ is independent of height.

== Sodium properties and the boiling criterion

All thirteen numbered relations of manual §12.13 are implemented. Their stated validity is
*590–2270 K*, the fits stopping near 90% of the critical point (2503.3 K) because $C_l$,
$beta_s$ and $alpha_p$ all contain $1 \/ (T_c - T)$ and diverge there. The implementation
neither clamps nor raises — a hard guard inside a residual would abort a differentiable
solve the moment a transient overshot — and instead exposes a range check that the
reference solution asserts.

Boiling onset is the saturation-plus-superheat criterion of §12.4: the coolant boils where
$T_c > T_"sat" (p) + Delta T_"sup"$. The cladding-and-structure to vapour heat path follows
§12.5.1.

== Neutronics and feedback

Power follows point kinetics with six delayed-neutron precursor groups (Eq. 4.3-1) closed
by the prompt-jump approximation (Eq. 4.2-4). Two feedbacks couple the thermal fields back
to the power, both as _axial integrals_: *Doppler*, logarithmic in fuel temperature and
interpolated between flooded and voided states (Eqs. 4.5-2, 4.5-3); and *coolant density
and void as a single worth distribution* (Eq. 4.5-25),

$ rho_"void" (t) = integral_0^1 w(zeta) alpha(zeta, t) dif zeta , $ <eq:void>

where the worth $w$ is *positive over most of the core and negative near the top*. That
sign change is the physical origin of the difficulty reported in @sec:openloop.

== Registered deviations from the manual

Every departure from SAS4A is registered with its justification; four matter here.

=== D-KIN-1, prompt jump
The prompt neutron mode is eliminated algebraically. Standard practice, and it removes the
stiffest timescale from the system.

=== D-TH-3, the void slaved to temperature
Rather than tracking vapour dynamics, the void fraction is closed algebraically on the
coolant temperature,

$ alpha = 1 - (1 - b(T_c))^3 , $ <eq:closure>

with $b$ the superheat switch. Justified by timescale separation — vapour fills a node in
0.71 ms against an advective 0.113 s — and selected from six candidate closures on maximum
voided-length error, at 1.2% against the reference's own 2.6% mesh error. The cubic form is
chosen for its finite slope at $b = 0$; a square-root closure has unbounded slope there.
*This deviation is what makes the boiling front form at all* in a differentiable
formulation.

=== D-TH-1 and D-FLOW-1
Eulerian mixture void rather than Lagrangian slug tracking, and prescribed flow rather than
prescribed pump head.

Registered but _off by default_, so no result here depends on them: condensation (D-TH-4),
three-group decay heat (D-KIN-3), axial fuel expansion (D-FB-4). Vapour expansion
accelerating the flow (D-TH-2) is implemented but not usable. Five further feedback
mechanisms are omitted and recorded as such (D-FB-3), and there is no pin failure or
material relocation (D-SCOPE-1) — which bounds the transient to its pre-failure phase.

= Reference solution and its resolution <sec:reference>

The reference is a stiff Radau integration on an axial mesh, and it is the instrument every
claim below is measured against, so its own resolution bounds what can be claimed.

It reproduces the analytic steady state to $3.4 times 10^(-11)$ K/s and conserves energy at
second order in the step. Refining it against itself gives @tab:ruler.

#figure(
  caption: [The reference solution's own error at the scoring mesh. These bound what any
  comparison against it can claim.],
  table(
    columns: 2,
    align: (left, center),
    stroke: none,
    table.hline(),
    [Quantity], [Reference's own error],
    table.hline(),
    [Temperatures, relative $L_2$], [1.1–$1.6 times 10^(-3)$],
    [Boiling onset time], [0.009 s],
    [Onset height], [0.06 cells],
    [Peak voided length], [0.57%],
    [Pointwise void fraction], [$3.2 times 10^(-2)$],
    table.hline(),
  ),
) <tab:ruler>

Two consequences are load-bearing. First, *an acceptance bar on the pointwise void fraction
is not supportable* and was withdrawn: the reference does not know that quantity to better
than 3%. Second, calibration practice requires a tolerance to sit at least four times above
the uncertainty of the instrument measuring it [6], so the 1% temperature bar is sound at a
ratio of about 6 and the 0.5 s onset criterion at about 56 — but a _result_ below that
uncertainty is measuring the instrument.

The reference terminates when any temperature reaches the top of the §12.13 property fits,
which defines the validity horizon; the surrogate is trained only over that horizon.

#figure(
  image("../docs/img/charts/boundary_conditions.png", width: 100%),
  caption: [What drives the transient. The pumps coast down to the natural-circulation
  floor while the power is still near nominal, so the outlet temperature rises until it
  meets saturation plus superheat. The dashed line marks boiling onset.],
) <fig:bc>

*Onset time is located by root-finding, not by grid inspection.* The reference's own onset
is *10.9784 s*; read off the 0.25 s output grid it appears as 10.75 s. A quarter of a
second of every onset error previously reported for this model was that quantisation.

= The surrogate <sec:surrogate>

Described here only as far as reproduction requires; the design study is in the repository
documentation.

The state is written multiplicatively,

$ theta(zeta, hat(t)) = theta_0 (zeta) exp(hat(t) N(zeta, hat(t))) , $

with $theta_0$ the analytic steady profile. This makes the initial condition exact, keeps
every temperature at or above the inlet, and pins the single upstream boundary condition
the advection equation admits — all without penalty terms. An additive ansatz, tried first,
allowed the optimiser to drive the fuel temperature negative, at which point the
logarithmic Doppler term is undefined.

Inputs pass through a random Fourier embedding [4],
$x arrow.r [sin(2 pi B x), cos(2 pi B x)]$ with $B$ frozen, at *64 features*. The trunk is a
64-wide, 5-layer tanh MLP with five outputs, carrying *17 029 fitted parameters*; the
embedding contributes a further 8192 in the read-out layer, whose width follows the
embedding rather than anything the network can represent, and 128 frozen entries in $B$.

Training minimises the squared PDE residuals [2] on *one fixed set of 5000 collocation
points*, with *50 000 L-BFGS iterations* [5] under a strong-Wolfe line search and *no
first-order stage at all*. Each of those three choices is measured rather than inherited: a
first-order warm start degrades the result rather than helping it, a set redrawn during the
solve costs a factor of 1.5 against a fixed one, and 5000 points is where the onset error
falls below the reference's own resolution. At 2000 points a front still forms but is badly
placed: onset is 0.19 s late and the saturation margin is $+28$ K against the reference's
$+69$ K. All
arithmetic is float64 on CPU; curvature pairs are meaningless at float32 residual
magnitudes.

Two independent implementations exist, in PyTorch and JAX/Equinox, required by test to
expose identical knobs and defaults and sharing only the numpy definition of the physics.
Results below are from the JAX implementation; the cross-implementation agreement and its
limits are documented in the repository.

= What the surrogate reproduces <sec:results>

All results at three seeds unless stated, against the reference at its scoring mesh.

== Temperature fields

The relative $L_2$ error on the film temperature is $1.6 times 10^(-3)$ (range
1.62–$1.65 times 10^(-3)$), against a 1% acceptance bar. Because the reference's own error
is 1.1–$1.6 times 10^(-3)$, the surrogate has reached the resolution of the instrument
judging it: the bar is met, and _how far inside_ it is not a question this reference can
answer. The other fields follow at $1.7 times 10^(-3)$ (coolant), $2.6 times 10^(-3)$
(fuel) and $3.7 times 10^(-3)$ (cladding).

== The boiling front

#figure(
  image("../docs/img/charts/temperature_map.png", width: 100%),
  caption: [Coolant temperature over space and time. The cyan contour is
  $T_"sat" + Delta T_"sup"$, which under D-TH-3 _is_ the boiling front rather than a
  separately computed curve; the star marks onset, located by tangency.],
) <fig:tmap>

#figure(
  image("../docs/img/charts/vapor_fraction.png", width: 100%),
  caption: [Void fraction over space and time, with axial profiles at five instants and
  at onset. The front forms at the outlet and propagates downward.],
) <fig:void>

#figure(
  image("../docs/img/charts/front_height.png", width: 100%),
  caption: [Front height and voided length. The saturation level set and the
  $alpha > 0.5$ contour do not coincide: the gap between them is the partially voided
  region the worth integral is most sensitive to.],
) <fig:front>

The engineering quantities are reproduced (@tab:front).

#figure(
  caption: [Safety-relevant front quantities against the reference.],
  table(
    columns: 3,
    align: (left, center, center),
    stroke: none,
    table.hline(),
    [Quantity], [Surrogate], [Reference],
    table.hline(),
    [Peak voided length], [*99.4%* of reference], [—],
    [Worst-seed saturation margin], [*$+67.4$ K*], [$+69.2$ K],
    [Boiling onset time], [*0.0042 / 0.0046 / 0.0082 s* error], [10.9784 s],
    table.hline(),
  ),
) <tab:front>

The onset criterion of 0.5 s is met on every seed, with the worst at 0.008 s. *Every seed
now sits below the reference's own onset uncertainty of 0.009 s*, so onset is reported as
met and is no longer separable from the instrument either; the seed spread has fallen to
2.0× from the 32× of the earlier configuration. Met is what the measurement supports.

*Onset height is not a discriminating quantity in this model.* The coolant heats
monotonically up the channel, so the hottest point — and therefore where boiling begins —
is always the outlet, for the surrogate and the reference alike. A height criterion here
measures the mesh, not the model. An earlier revision of this work reported that quantity
as solved exactly; it had merely been restated as a tautology.

= What is not yet solved: the closed void-reactivity loop <sec:openloop>

With the thermal fields prescribed, the surrogate is accurate. Driving the _kinetics_ from
the learned fields is a different matter, and it fails in a specific and instructive way.

The figures in this section were measured on the earlier default configuration (256 Fourier
features, a short Adam stage, 30 000 quasi-Newton iterations on 6000 points) and have not
been re-measured on the configuration of @sec:surrogate. The cancellation ratio is a
property of the worth distribution and does not depend on the surrogate; the recovered
fractions may move.

Over the same fields and the same network, the Doppler integral is reproduced to a factor
of *1.017*. The void integral recovers only *8–16%* of the reference's value. The two
differ in one respect: the Doppler weight has one sign, and the void worth changes sign
near the top of the core.

Splitting the void functional of @eq:void at the sign change, at the instant it peaks,
gives @tab:cancel.

#figure(
  caption: [The void functional split at the sign change of the worth, at peak. The two
  halves are of comparable size and opposite sign.],
  table(
    columns: 2,
    align: (left, center),
    stroke: none,
    table.hline(),
    [Contribution], [Value],
    table.hline(),
    [Positive-worth region, $J^+$], [$+4.656 times 10^(-4)$],
    [Negative-worth region, $J^-$], [$-1.695 times 10^(-4)$],
    [Their sum, $J$], [$+2.962 times 10^(-4)$],
    [Cancellation ratio, $abs(J) \/ (abs(J^+) + abs(J^-))$], [*0.466*],
    table.hline(),
  ),
) <tab:cancel>

#figure(
  image("../docs/img/charts/void_worth_split.png", width: 78%),
  caption: [The void worth $w(zeta)$, shaded by sign. Positive over most of the core and
  negative near the top, so the reactivity functional of @eq:void is a difference of two
  large contributions rather than a sum of small ones.],
) <fig:worth>

#figure(
  image("../docs/img/charts/reactivity.png", width: 78%),
  caption: [Reactivity split by mechanism, in units of $beta_"eff"$. Reporting only the
  net would hide which component carries the error.],
) <fig:rho>

A relative error $epsilon$ on each half therefore becomes $2.1 epsilon$ on the sum. The
functional is an ill-conditioned target by construction, and reporting it as one number
hides which half is wrong; we report the halves separately from here on.

*It is not a sampling problem.* Dual-weighted-residual theory [3] gives the error in a
functional as $lr(angle.l R(u_theta), z^* angle.r)$ to leading order, with $z^*$ the
solution of the adjoint problem sourced by the functional's derivative. For this advective
coolant operator the adjoint runs backwards in $zeta$, and it can be evaluated in closed
form. The result is a *step*: the void slope underflows to exactly zero wherever the
coolant is subcooled, so $z^*$ is constant over the lower 72% of the channel and zero above
it. Every point below the front carries equal sensitivity and every point above it carries
none. Residual-magnitude sampling concentrates points _at_ the front — precisely where the
functional is insensitive — so it cannot help, and a uniform sampler is already near
optimal.

Evaluated open-loop on the surrogate's own field, the functional is in fact accurate: the
positive half to 1.66–2.66% and the negative half exactly, the latter because the
negative-worth region is fully voided in surrogate and reference alike and the integral is
then fixed by geometry. *The deficit therefore arises in the closed loop*, where
$rho_"void"$ feeds back into the kinetics and the error compounds — not in the evaluation of
the functional on a given field. That localises the remaining work precisely.

= Discussion

The surrogate reproduces the thermal-hydraulic phase of this transient to the resolution of
the reference solver, including the safety-relevant front quantities: when boiling starts,
how far it extends, and by what margin. For applications that consume those fields —
uncertainty propagation over thermophysical parameters, inference of boundary conditions
from limited instrumentation, gradient-based studies of the coastdown — the model is usable
now, and its derivatives with respect to inputs are exact by construction.

For applications that require the *feedback loop closed*, it is not. The obstruction is not
accuracy in any ordinary sense — the fields are right and the open-loop functional is right
— but the conditioning of a near-cancelling integral inside a feedback path. That is a
property of the physics: a positive void worth over most of the core against a negative
worth near the top is precisely what makes SFR void feedback interesting, and it is
precisely what makes it a hard target for a surrogate. Any neural approach to sodium void
feedback will meet the same 2.1× amplification.

Two consequences for practice. First, *a surrogate should be qualified against the
functionals it will be used for, not only against field norms*: an $L_2$ of
$1.6 times 10^(-3)$ on temperature coexists here with an 84–92% miss on a reactivity
integral built from the same field. Second, *the reference's own resolution must be
quantified before an acceptance bar is set* [7, 8]. Two bars in this work were set without
that check; one was withdrawn, and the temperature result has since reached the point where
the instrument, not the model, is the limit.

= Conclusions

A physics-informed neural surrogate for the sodium-boiling phase of an SFR unprotected
loss-of-flow transient, built on SAS4A/SASSYS-1 thermophysics and feedback laws with four
registered deviations, reproduces the reference solution's temperature fields to
$1.6 times 10^(-3)$ relative $L_2$, its boiling onset to within 0.008 s of 10.9784 s, and
99.4% of its peak voided length with a saturation margin of $+67.4$ K against $+69.2$ K.
The first two are at the reference's own resolution and are reported as met rather than as
measured accuracies.

The closed void-reactivity loop is not reproduced, recovering 8–16% of the reference
integral while the non-cancelling Doppler integral over the same fields is correct to
1.017. The cause is quantified — a cancellation ratio of 0.466, amplifying any error on
either half by 2.1× — and shown by an adjoint argument not to be a sampling deficiency.
Closing that loop is the outstanding problem for this model, and we expect it to be the
outstanding problem for neural surrogates of sodium void feedback generally.

#heading(numbering: none)[Acknowledgments]

The SAS4A/SASSYS-1 documentation maintained by Argonne National Laboratory was the sole
source for the thermophysics and feedback laws used here.

#references((
  [Fanning T H (ed) 2017 _The SAS4A/SASSYS-1 Safety Analysis Code System_ ANL/NE-16/19 (Argonne, IL: Argonne National Laboratory)],
  [Raissi M, Perdikaris P and Karniadakis G E 2019 Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations _J. Comput. Phys._ *378* 686],
  [Becker R and Rannacher R 2001 An optimal control approach to a posteriori error estimation in finite element methods _Acta Numerica_ *10* 1],
  [Tancik M, Srinivasan P P, Mildenhall B, Fridovich-Keil S, Raghavan N, Singhal U, Ramamoorthi R, Barron J T and Ng R 2020 Fourier features let networks learn high frequency functions in low dimensional domains _Adv. Neural Inf. Process. Syst._ *33* 7537],
  [Nocedal J and Wright S J 2006 _Numerical Optimization_ 2nd edn (New York: Springer)],
  [Taylor B N and Kuyatt C E 1994 _Guidelines for Evaluating and Expressing the Uncertainty of NIST Measurement Results_ NIST Technical Note 1297 (Gaithersburg, MD: National Institute of Standards and Technology)],
  [Jakeman J D, Barba L A, Martins J R R A and O'Leary-Roseberry T 2026 Verification and validation for trustworthy scientific machine learning _Mach. Learn.: Sci. Technol._ *7* 025055],
  [Eça L and Hoekstra M 2014 A procedure for the estimation of the numerical uncertainty of CFD calculations based on grid refinement studies _J. Comput. Phys._ *262* 104],
))
