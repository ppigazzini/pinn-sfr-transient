# Optimisation budget, not architecture, governs accuracy in a physics-informed neural network for sodium boiling

**P Pigazzini**

Independent researcher — <pasquale.pigazzini@gmail.com>

> **Draft.** This is the Markdown working copy of the paper. `paper.typ` is the
> typeset version against the IOP JPCS layout; when the two disagree, the numbers in
> [`docs/axial_nn.md`](../docs/axial_nn.md) are the source and both are wrong until
> reconciled. See [README.md](README.md) for how to build the PDF.

## Abstract

We train a physics-informed neural network (PINN) on the unprotected loss-of-flow
transient of a sodium-cooled fast reactor, resolving one coolant channel axially with
four material fields and a sodium-void closure taken from the SAS4A/SASSYS-1 manual.
Against a verified stiff reference solver the network reaches a relative $L_2$ error of
$1.7 \times 10^{-3}$ on the cladding-to-coolant film temperature and reproduces 99.3% of
the reference peak voided length. That error is comparable to the reference solver's own
discretisation error of 1.1–1.6 × 10⁻³: the 1% acceptance bar is met, but by the 4:1
test-uncertainty-ratio convention the residual accuracy is no longer resolvable against
this reference, and we report it as such rather than as a measured accuracy. The result
is obtained without any change of architecture. In a 54-run factorial sweep over Adam and
quasi-Newton iteration counts, replicated at three seeds in two independent
implementations, the Adam axis is flat across two decades once the quasi-Newton stage is
funded, and Adam alone never produces a boiling front; extending L-BFGS from 3000 to
30 000 iterations improves the error fifteenfold, monotonically, on every seed. Thirteen
architectural and loss-side remedies were tested in isolation against the same control:
those that changed the function space or the optimiser helped, while every remedy that
reweighted the loss, added residuals already implied by the governing equations, or
re-parameterised the inputs was inert or harmful. We also report a defect class that a
single implementation cannot detect, in which one unset library default accounted for an
entire apparent cross-framework accuracy gap. We conclude that reported PINN comparisons
are not interpretable without a stated iteration budget and a stated optimiser
configuration.

## 1. Introduction

Physics-informed neural networks solve partial differential equations by minimising the
residual of the governing equations at collocation points, using no solution data [1].
The literature is dominated by architectural and loss-side proposals: adaptive weighting,
resampling, embeddings and network variants. Comparisons between them are usually
reported at a fixed and often unstated optimisation budget.

We report a study in which the optimisation budget dominates every architectural choice
we tested. The vehicle is a safety-relevant transient rather than a manufactured
benchmark: the unprotected loss-of-flow (ULOF) accident in a sodium-cooled fast reactor,
in which coolant flow decays, sodium approaches saturation, and a positive void
reactivity coefficient competes with negative Doppler feedback. The quantity of interest
is not only a field norm but the formation, extent and timing of a boiling front.

Three contributions follow. First, a factorial measurement showing that the quasi-Newton
iteration count is the only training axis that changes the physics of the solution, and
that extending it alone takes the model from missing its acceptance bar by a factor of
two to meeting it. Second, a classification of thirteen remedies tested under identical
isolation, which separates interventions that change the function space or the optimiser
from those that only reweight or re-parameterise. Third, a methodological result: two
independent implementations of the same formulation disagreed by up to 44%, and the
entire disagreement was one unset library default.

## 2. Physical model and reference solution

### 2.1 Governing equations

One coolant channel is resolved axially in $\zeta \in [0, 1]$ with four material fields —
fuel, cladding, film and coolant temperature — and a void fraction $\alpha$. Coolant
enthalpy transport is advective with a prescribed flow decay

```math
f(t) = f_{\mathrm{nc}} + (1 - f_{\mathrm{nc}})\, e^{-t / \tau_{\mathrm{pump}}},
```

and the sodium properties, saturation temperature and superheat criterion are taken from
the SAS4A/SASSYS-1 manual [2]. Reactivity feedback couples the fields to six-group point
kinetics through two axial integrals, a Doppler term weighted by the power shape and a
void term weighted by a worth that changes sign near the top of the core.

Two deviations from the manual are load-bearing and are registered as such. The prompt
neutron mode is eliminated by the prompt-jump approximation, and the void fraction is
slaved algebraically to the coolant temperature,

```math
\alpha = 1 - \bigl(1 - b(T_c)\bigr)^3,
```

with $b$ the superheat switch. The second is justified by timescale separation — vapour
fills a node in 0.71 ms against an advective 0.113 s — and was selected from six candidate
closures on maximum voided-length error, at 1.2% against the reference's own 2.6% mesh
error. The cubic form is chosen for its finite slope at $b = 0$; a square-root closure has
an unbounded slope there and returns NaN on the first backward pass.

### 2.2 Reference solver and its uncertainty

The reference is a stiff Radau integration on an axial mesh. It reproduces the analytic
steady state to $3.4 \times 10^{-11}$ K/s, conserves energy at second order in the step,
and places boiling onset independently of mesh over $n_{\mathrm{axial}} = 40$ to 640, at
10.9784 s when the crossing is root-found (10.75 s if read off the 0.25 s output grid,
which is how it was reported before we located events properly).

Because acceptance bars are stated against this solver, its own uncertainty bounds what
any bar can mean. Refining the reference against itself, the onset time and height at the
scoring mesh $n_{\mathrm{axial}} = 160$ are converged to 0.009 s and 0.06 cells
respectively, and the temperature fields to a relative $L_2$ of 1.1–1.6 × 10⁻³. The 1%
temperature bar therefore sits 6–9 times above the ruler, and the onset criterion an
order of magnitude above it. An earlier acceptance bar on the pointwise void fraction did
not survive this test and was withdrawn: the reference's own error there is
$3.2 \times 10^{-2}$.

## 3. Network and training

The state is written multiplicatively,

```math
\theta(\zeta, \hat{t}) = \theta_0(\zeta)\, \exp\bigl(\hat{t}\, N(\zeta, \hat{t})\bigr),
```

with $\theta_0$ the analytic steady profile. This makes the initial condition exact, keeps
every temperature at or above the inlet, and pins the single upstream boundary condition
admitted by the advection equation, all without penalty terms. An additive ansatz, tried
first, allowed the optimiser to drive the fuel temperature negative while the loss fell,
at which point the logarithmic Doppler term returns NaN.

Inputs pass through a random Fourier embedding,
$x \mapsto [\sin(2 \pi B x), \cos(2 \pi B x)]$. Training is Adam followed by an L-BFGS
stage with a strong-Wolfe line search. All arithmetic is float64 on CPU.

Two independent implementations were written, one in PyTorch and one in JAX/Equinox, and
are required by test to expose identical knobs with identical defaults. They share only
the numpy definitions of the physics.

## 4. Protocol

Every result below is the mean of three seeds, reported with the seed range, and every
comparative claim is required to hold in both implementations. Single-seed conclusions in
this project have been overturned by the next seed on eleven occasions, and the practice
of stating a sample size in the sentence that states the result was adopted after the
fourth.

Thread count is pinned, because it changes floating-point reduction order and therefore
changes answers rather than only timings. `OMP_NUM_THREADS` binds PyTorch but not JAX,
whose CPU backend sizes its own pool: an arm nominally at eight threads was measured
creating 291. Pinning CPU affinity binds both, and the core count — not which cores —
determines the result bitwise.

## 5. Results

### 5.1 The optimisation budget

**Table 1.** Relative $L_2$ on the film temperature $T_s$ over a factorial sweep of Adam
against quasi-Newton iterations, at 128 Fourier features. Three seeds per cell, both
implementations. Cells marked † produced no boiling front on any seed.

| Adam | PyTorch qn 30 | PyTorch qn 300 | PyTorch qn 3000 | JAX qn 30 | JAX qn 300 | JAX qn 3000 |
|---|---|---|---|---|---|---|
| 30   | 0.2314† | 0.0657 | **0.0303** | 0.2336† | 0.0960 | **0.0350** |
| 300  | 0.1906† | 0.0603 | 0.0314     | 0.1935† | 0.0880 | 0.0363     |
| 3000 | 0.0707  | 0.0455 | 0.0305     | 0.0762  | 0.0562 | 0.0358     |

Table 1 is the central measurement. The quasi-Newton axis is monotone across two decades
in every row of both implementations. The Adam axis, once the quasi-Newton stage is funded
at 3000 iterations, is flat: 0.0303, 0.0314 and 0.0305 in PyTorch across a hundredfold
change in Adam iterations, a spread smaller than the seed range of a single cell. Adam
alone, at 30 quasi-Newton iterations, produces no boiling front at all, and raising Adam a
hundredfold does not recover one.

The axis had never been pushed past 3000. Table 2 extends it.

**Table 2.** Extending the quasi-Newton stage at the shipped configuration, 256 Fourier
features, three seeds. $L_{\mathrm{void}}$ is the peak voided length as a percentage of
the reference's; the margin is the worst-seed excess of peak coolant temperature over
saturation plus superheat, against the reference's +69.2 K.

| Adam / qn  | $T_s$      | seed range | $L_{\mathrm{void}}$ | margin   | s per run |
|---|---|---|---|---|---|
| 30 / 3000   | 0.0258     | 1.10×      | 78%        | +30.0 K  | 905  |
| 30 / 10 000 | 0.0044     | 1.67×      | 97%        | +62.0 K  | 3089 |
| 30 / 30 000 | **0.0017** | 1.06×      | **99.3%**  | +67.6 K  | 9060 |

Extending the quasi-Newton stage tenfold improves the error fifteenfold, monotonically on
every seed, and takes the model from missing its 1% bar by a factor of two to meeting it.

How far inside the bar is a question the reference cannot answer. Metrological practice
requires a tolerance to sit at least four times above the uncertainty of the instrument
measuring it — the test uncertainty ratio, standard in calibration since MIL-STD-45662A
and carried into ANSI/NCSL Z540. Against a reference error of 1.1–1.6 × 10⁻³ the 1% bar
has a ratio of about 6, so the bar itself is sound. The measured $1.7 \times 10^{-3}$ has
a ratio of about 1.06. We therefore state that the network has reached the reference's own
resolution, and decline to quote a factor by which it beats the bar: below a ratio of one
the comparison is measuring the ruler. The peak voided length reaches 99.3% of the
reference and the saturation margin +67.6 K against the reference's +69.2 K, so the front
is quantitatively right rather than merely present. The seed range contracts from 1.10 to
1.06, which reverses an assumption held throughout this work: the seed sensitivity that
forced three-seed replication was itself a symptom of an under-converged optimisation.

That the cost is tenfold is not incidental. But it is a cost that can simply be paid,
which no architectural remedy in the next section could be.

### 5.2 Thirteen remedies under isolation

Each remedy below moved exactly one setting from a common control and was scored on the
same reference. Compound arms were avoided after an early combined arm failed without
being attributable to either half.

**Table 3.** Remedies tested in isolation, grouped by what each changes. Three seeds and
both implementations except where noted.

| What it changes | Remedies | Outcome |
|---|---|---|
| Function space | Fourier capacity 32→512; multi-scale bands; anisotropic bandwidth | All helped. Bands reached 99.5% of reference voided length while also improving the mean |
| Optimiser | Quasi-Newton budget; curvature memory | Decisive; the only axis that forms the front |
| Loss measure | Level-set sampling; front-fraction sampling; block and causal weighting | Failed or inert. Front sampling degrades the mean monotonically from 0% to 50% |
| Extra residuals | Onset-tangency head; front-position network; pseudo-time | Harmful. A condition implied by the governing equations carries no information as a constraint |
| Re-parameterisation | Level-set input coordinate; Laplace decay embedding | Inert on both implementations |

The grouping in Table 3 is not post-hoc tidying: it was written down after twelve remedies
and then correctly predicted the thirteenth. A Laplace embedding was proposed on the sound
argument that the transient is built from exponential decay — pump coast-down and six
precursor groups spanning a 243-fold range of rates — which a sinusoidal basis represents
poorly. It was classified in advance as a re-parameterisation and predicted inert, because
the multiplicative ansatz already represents a decaying mode. It measured inert, on both
implementations at three seeds.

The two failures worth stating positively are the loss-side ones. Diverting collocation
onto the boiling front does not make the front easier to resolve; it starves the bulk, and
the mean error degrades monotonically with the diverted fraction. Adding the tangency
conditions that define onset as residuals is worse than adding nothing: because they are a
consequence of the governing equations rather than independent information, the optimiser
satisfies them by deforming the field toward the wrong point, and the onset time error
grows by a factor of 2.4.

### 5.3 A defect one implementation cannot find

The two implementations disagreed on accuracy by a margin that grew with model size, from
1.09 to 1.44 across a capacity ladder, on residuals verified identical to $10^{-14}$.

The cause was a single unset argument. `optax.lbfgs()` defaults to retaining ten curvature
pairs; the PyTorch side passed fifty. Copying one implementation's weights into the other,
verifying the objective agreed to the last bit, and varying only the optimiser resolved
it: PyTorch at ten pairs reproduces the JAX loss curve to 3%, and JAX at fifty reproduces
the PyTorch curve to 2%. The two implementations of L-BFGS are equivalent at equal memory.

The apparent framework effect was therefore an artefact, and so was a second one: every
timing comparison had bound PyTorch's threads while leaving JAX's unbound. At matched
thread count JAX is 4.4 times faster, not the 2.4 previously recorded.

Neither defect is detectable from within one implementation, because within one
implementation nothing disagrees.

## 6. Discussion

The practical conclusion is narrow and, we think, general. In this problem the
optimisation budget is not a nuisance parameter to be fixed while architectures are
compared; it is the dominant variable, and it dominates by more than an order of magnitude
in final error. A comparison of two PINN architectures at 3000 quasi-Newton iterations
would have ranked them on which reaches an under-converged state faster, which is a
different question from which is more accurate.

This has a direct bearing on reported results. Self-scaled quasi-Newton methods have been
reported to improve PINN accuracy by one to three orders of magnitude over BFGS [7, 14].
**We are not in a position to confirm or contradict that, and say so rather than quoting
the comparison we have.** Our optimiser comparison exists only at 3000 Adam / 300
quasi-Newton — the starved diagonal that Table 1 shows is precisely the regime where the
quasi-Newton stage does not yet matter — and at that budget the four methods do not order
consistently across our two implementations. They also cannot: those runs predate the
curvature-memory defect of §5.3, so the two sides were compared at different memory. The
comparison at a funded quasi-Newton stage is designed and has not been run.

We report that as a gap rather than as a result, because an optimiser comparison made in
an under-converged regime is the precise error this paper is about, and we are not exempt
from it.

What can be said is why little should be expected. Limited-memory BFGS already rescales
its initial inverse-Hessian approximation by the Oren–Luenberger factor

```math
\gamma_k = \frac{s_k^{\mathsf{T}} y_k}{y_k^{\mathsf{T}} y_k}
```

at every iteration, and this is the default in both frameworks used here [10]. Self-scaling
should therefore be largely redundant against an L-BFGS baseline, while against the
unscaled full-memory BFGS with $H_0 = I$ used in the studies above it supplies a scaling
that is otherwise absent. Those studies also use networks of ~10³ parameters, roughly
twenty times smaller than ours, and report single runs without seed statistics; a third
study finds self-scaled Broyden losing to plain BFGS on two of four equations [15]. That
is a prediction we intend to test, not a finding.

The general point stands independently of who is right, and it is the one we can support:
an optimiser comparison is only interpretable alongside its baseline's configuration, its
curvature memory, its network size, and its seed count. Three of those four have already
inverted a conclusion in this work.

Two reporting conventions from the wider computational-science literature would have caught
our own errors earlier, and we adopt them. Recent verification-and-validation guidance for
scientific machine learning requires that reference-data fidelity be documented and its
effect on the surrogate assessed, and that error variability over seeds be reported rather
than a single run [12]; a survey of machine learning for fluid-related PDEs found that 60
of 76 papers claiming to beat a numerical method compared against a weak baseline [13]. A
stiff Radau reference with a quantified discretisation error is a strong baseline, and
quantifying it is what told us our own headline number had run out of resolution.

The classification in Table 3 suggests a discipline for proposals. An intervention that
changes what the network can represent, or how the optimiser explores it, is worth
testing. An intervention that reweights an objective, or re-expresses inputs the network
already computes, has no mechanism by which to add information, and in this study none of
the five such interventions did.

Timing behaves the same way, and this is the sharpest form of the paper's claim. Boiling
onset was thought to be the model's outstanding failure at 0.62–0.84 s against a 0.5 s
criterion. That figure was measured at 3000 quasi-Newton iterations. At 30 000 the error
is **0.0006, 0.0064 and 0.0181 s** on three seeds — the criterion is met, by the
optimisation budget alone, with no residual, sampling or architectural change aimed at
onset at any point. A second correction is metrological rather than numerical: locating
the crossing by root-finding rather than by scanning the 0.25 s output grid moves the
*reference's* own onset from 10.75 s to 10.9784 s, so a quarter of a second of the
original miss was the ruler's quantisation.

We report it as met and decline to say by how much. The reference's own onset is
uncertain by 0.009 s at the scoring mesh, so the worst seed sits at a test uncertainty
ratio of 2.0 and the best seed below 1 — the same exhaustion of resolution as the
temperature bar. The seed spread, 32×, is the widest of any quantity we measure; the
criterion holds on every seed, but no comparison between formulations should be built on
this scalar at fewer than three.

One negative worth stating: we had expected onset time to *amplify* field error, since a
threshold crossing converts a temperature error into a time error at the reciprocal of
the heating rate. Measured, that first-order bound over-predicts the error by between 3.4
and 100 times. Onset converts field accuracy at better than the first-order rate, so the
derived scalar is easier than the field it derives from, not harder.

## 7. Conclusions

A physics-informed neural network for a sodium-cooled fast reactor boiling transient meets
a 1% accuracy bar, reaching $1.7 \times 10^{-3}$ relative $L_2$ and 99.3% of the reference
peak voided length, through optimisation budget alone — and in doing so reaches the
resolution limit of the reference it is scored against, at a test uncertainty ratio of
about one. Across a 54-run factorial sweep at three seeds and two implementations, the
quasi-Newton iteration count is the only training axis that changes the physics of the
solution; the Adam count is flat across two decades. Of thirteen remedies tested in
isolation, those that changed the function space or the optimiser helped and those that
reweighted the loss, added implied residuals, or re-parameterised inputs did not. We
recommend that PINN comparisons report the optimiser configuration and the iteration
budget as primary experimental variables, and that architectural claims be demonstrated at
a converged budget.

## Acknowledgments

The SAS4A/SASSYS-1 documentation maintained by Argonne National Laboratory was the sole
source for the thermophysics and feedback laws used here.

## References

1. Raissi M, Perdikaris P and Karniadakis G E 2019 Physics-informed neural networks: a
   deep learning framework for solving forward and inverse problems involving nonlinear
   partial differential equations *J. Comput. Phys.* **378** 686
2. Fanning T H (ed) 2017 *The SAS4A/SASSYS-1 Safety Analysis Code System* ANL/NE-16/19
   (Argonne, IL: Argonne National Laboratory)
3. Wang S, Sankaran S and Perdikaris P 2024 Respecting causality for training
   physics-informed neural networks *Comput. Methods Appl. Mech. Eng.* **421** 116813
4. Wang S, Teng Y and Perdikaris P 2021 Understanding and mitigating gradient flow
   pathologies in physics-informed neural networks *SIAM J. Sci. Comput.* **43** A3055
5. Tancik M, Srinivasan P P, Mildenhall B, Fridovich-Keil S, Raghavan N, Singhal U,
   Ramamoorthi R, Barron J T and Ng R 2020 Fourier features let networks learn high
   frequency functions in low dimensional domains *Adv. Neural Inf. Process. Syst.*
   **33** 7537
6. Rathore P, Lei W, Frangella Z, Lu L and Udell M 2024 Challenges in training PINNs: a
   loss landscape perspective *Proc. 41st Int. Conf. on Machine Learning*
7. Kiyani E, Shukla K, Urbán J F, Darbon J and Karniadakis G E 2025 Which optimizer works
   best for physics-informed neural networks and Kolmogorov-Arnold networks? Preprint
   arXiv:2501.16371
8. Müller J and Zeinhofer M 2023 Achieving high accuracy with PINNs via energy natural
   gradient descent *Proc. 40th Int. Conf. on Machine Learning* **202** 25471
9. Wu C, Zhu M, Tan Q, Kartha Y and Lu L 2023 A comprehensive study of non-adaptive and
   residual-based adaptive sampling for physics-informed neural networks *Comput. Methods
   Appl. Mech. Eng.* **403** 115671
10. Nocedal J and Wright S J 2006 *Numerical Optimization* 2nd edn (New York: Springer)
11. Taylor B N and Kuyatt C E 1994 *Guidelines for Evaluating and Expressing the
    Uncertainty of NIST Measurement Results* NIST Technical Note 1297 (Gaithersburg, MD:
    National Institute of Standards and Technology)
12. Jakeman J D, Barba L A, Martins J R R A and O'Leary-Roseberry T 2026 Verification and
    validation for trustworthy scientific machine learning *Mach. Learn.: Sci. Technol.*
    **7** 025055
13. McGreivy N and Hakim A 2024 Weak baselines and reporting biases lead to overoptimism
    in machine learning for fluid-related partial differential equations *Nat. Mach.
    Intell.* **6** 1256
14. Eça L and Hoekstra M 2014 A procedure for the estimation of the numerical uncertainty
    of CFD calculations based on grid refinement studies *J. Comput. Phys.* **262** 104
15. Urbán J F, Stefanou P and Pons J A 2025 Unveiling the optimization process of
    physics-informed neural networks *J. Comput. Phys.* **523** 113656
16. Si K, An H, Ma Z and Yan X 2026 From non-convex self-concordant regularization to
    scalable quasi-Newton training of PINNs Preprint arXiv:2608.04206
