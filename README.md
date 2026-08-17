# pinn-sfr-transient

Physics-Informed Neural Networks for the **Unprotected Loss of Flow (ULOF)**
transient in a Generation-IV **Sodium-cooled Fast Reactor (SFR)**. No experimental
data is used for training — the physics residuals are the teacher; a stiff `scipy`
integrator is the held-out reference.

The repository holds **two models**, and they are at very different stages:

| | [0D lumped](docs/physics_theory.md) | [1D axial boiling](docs/axial_physics.md) |
|---|---|---|
| state | 6-group point kinetics + two thermal nodes | four material fields on an axial mesh + sodium void |
| void | a `tanh` demonstration surrogate at 820 K | saturation + superheat from the SAS4A manual, ~1156 K |
| reference solver | verified | verified, except the void fraction (`axial_physics.md` §6.5) |
| PINN | **meets its bar** — a few 1e-3 relative L2 | **meets its bar** — 0.0017 against 0.01, and 99.3% of the reference voided length; at the reference's own resolution |
| backends | PyTorch, JAX, DeepXDE | **JAX** (default), PyTorch |

This README is a map. The physics, the neural-network methodology, and the usage
details live in [`docs/`](docs/) — see [Documentation](#documentation).

## The 0D model — a solved problem

Loss of flow drives the coolant past the void-onset temperature; the positive
void coefficient pushes power to **1.38× nominal at ≈ 23 s**, then negative
Doppler feedback dominates and the power turns over, settling to ≈ 0.69× — a
bounded, self-limiting transient. Trained on residuals alone, the PINN recovers
the whole trajectory ([`docs/neural_network.md`](docs/neural_network.md) §7).

## The 1D axial model — the current work

Because the 0D void surrogate cannot say *where* boiling starts, how far the void
spreads, or that the sodium void worth **changes sign** near the top of the core —
which is what a loss-of-flow safety argument turns on — a second model resolves the
channel axially and takes its thermophysics, boiling onset and feedback laws from the
[SAS4A/SASSYS-1 manual](https://sas-doc.nse.anl.gov/latest/) (ANL/NSE-SAS/5.8.1).

**Status, stated plainly.** The reference solver is verified and the physics
question is answered: the positive sodium void coefficient **does** drive a power
excursion, to 5.3× nominal, governed by the height at which the void worth changes
sign rather than by its magnitude ([`docs/axial_physics.md`](docs/axial_physics.md)
§10).

The PINN trains, satisfies every hard constraint exactly, and **meets its 1%
accuracy bar** — `T_s = 0.0017` relative `L2` at three seeds, with 99.3% of the
reference's peak voided length and a saturation margin of +67.6 K against the
reference's +69.2 K. It got there **by optimisation budget alone**: extending the
quasi-Newton stage from 3000 to 30 000 iterations improved the error fifteenfold,
monotonically, on every seed, with no change of architecture.

**And it has run out of room to be measured in.** The reference's own error is
1.1–1.6e-3, so 0.0017 sits at a test uncertainty ratio of about **1.06** — the
network's error is now the same size as the ruler's. Calibration practice wants a
tolerance four times above its instrument, so the bar itself is sound at a ratio of
6, but *how far inside* the bar we are is no longer a question this reference can
answer. Further accuracy on the temperature fields cannot be demonstrated against it
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.22).

**The boiling front is essentially solved too — by the same budget.** The shipped
default reaches **99.3% of the reference's peak voided length** at three seeds, against
64% for the old budget.

An earlier revision of this file credited that to the *embedding*: giving the Fourier
basis several frequency bands at once reached 99.5% at the old `qn3000` budget. A 2×2
has since measured the two together, and they are **not** two gains. At the funded
budget the multi-band embedding is 5.6× worse on the mean and loses half the saturation
margin. A multi-band basis is an implicit preconditioner; once the quasi-Newton stage is
funded it accumulates that curvature itself, and splitting a fixed feature budget across
bands is left as pure capacity loss. The bands are off
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.24).

**Onset turns out to be a question about *time*, not place.** The coolant heats
monotonically up the channel, so the hottest point — and therefore where boiling
starts — is always the outlet. A height criterion measures the mesh, not the
network; an earlier revision of the documentation claimed that quantity had been
solved exactly, and it had merely been restated as a tautology (§7.5.16).

Onset *time* is the quantity that carries information, and **it is now met** —
0.0006, 0.0064 and 0.0181 s against a 0.5 s criterion, three seeds. The earlier
0.62–0.84 s was measured at the old quasi-Newton budget; the same axis that took
`T_s` to 0.0017 took onset with it, with no residual, sampling or architectural
change ever aimed at onset. Two corrections came with it: the *reference's* own
onset is 10.9784 s rather than 10.75 s once the crossing is root-found instead of
read off the 0.25 s grid, so a quarter of a second of every onset error this project
published was the ruler's quantisation; and onset turns out to convert field accuracy
at *better* than the first-order rate, not worse (§7.5.16a).

Here too the room has gone: the reference's own onset is uncertain by 0.009 s, so the
worst seed sits at a ratio of 2.0 and the best below 1. Met is what the measurement
carries; a factor is not. Since the height half cannot be failed on a monotone field
and the time half is now passed, **M4 no longer discriminates between formulations**
and is being replaced (§7.5.25).

**Architecture turned out not to matter here — and finding that out took four
corrections.** The Adam iteration count, the choice of quasi-Newton variant, a multi-scale
Fourier basis and the size of the Fourier embedding were each measured, concluded on, and
then overturned once the quasi-Newton stage was properly funded. The embedding *width* is
the clearest case: a 32-feature embedding matches the shipped 256 at 42% of the wall-clock
and 42% of the parameters, with every seed range overlapping. The original ladder that
chose 256 was measured at a tenth of the final budget. An architectural comparison made at
an under-converged budget measures which architecture reaches the under-converged state
faster, which is a different question from which is more accurate — and this project
answered the wrong one four times ([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.31).

**The Fourier embedding and the optimisation budget do different jobs, and neither
substitutes for the other.** Strip the embedding entirely and give the quasi-Newton stage
its full funded budget: the error goes from 0.0017 to 0.0397 and the boiling front
disappears — +0.8 K of saturation margin against the reference's +69.2 K. Without the
embedding *nothing* forms a front, on any optimiser tried. The features make the front
representable; the budget makes it accurate.

**The best configuration we have measured has no Adam stage and half the network.**
A 64-feature embedding, no first-order stage at all, and one long quasi-Newton solve on a
fixed collocation set reaches `T_s = 0.0016` [.0016–.0017] with 99.4% of the reference
voided length and a +67.8 K margin — equal or better than the shipped default on every
column, with **25 221 trainable parameters against 49 797**. Both sit at the reference's
own resolution, so this is not a claim of more accuracy; it is the same result from half
the model and none of the Adam machinery
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.37).

**The Adam stage turned out to be worse than unnecessary — on this model.** Removing it
entirely, and spending the whole budget on the quasi-Newton stage with its collocation set
redrawn periodically, makes the axial model **2.8× more accurate** at three seeds with
non-overlapping ranges — and gives the tightest seed spread we have measured (1.09×).

That is emphatically *not* a general claim about first-order methods, and the counter-example
is in this repository. The 0D lumped model, under the same optimiser, the same backend and
the same settings, **diverges** without an Adam warm start: a pure quasi-Newton solve reaches
a power error of 0.45 against an untrained network's 0.21. So the standard Adam→L-BFGS recipe
is right for one of our two models and wrong for the other, and which one you are on is
decided by the formulation rather than by the optimiser
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.34–35).

**And the first-order stage buys nearly nothing here — nor is it cheaper.** Adam alone,
at the same iteration count and comparable wall-clock, is 19× worse than spending that
budget on the quasi-Newton stage, and AdEMAMix is worse still. Per *iteration*, measured
at matched configuration, L-BFGS is the **cheapest** of the three (99.3 ms against Adam's
107.8 and AdEMAMix's 115.5) — so in this implementation the higher-order stage is faster
per iteration *and* an order of magnitude more accurate, and dropping Adam altogether
costs nothing measurable.

Both claims come with a scope that matters more than either. **Adam here is run
full batch** — every step evaluates all 6000 collocation points, the same set the
quasi-Newton stage uses — so it is a first-order method with momentum, not the stochastic
minibatch Adam the literature runs. That alone explains the cost tie: both optimisers pay
for 6000 residual evaluations per step, which is essentially the whole cost, while Adam's
real advantage comes from batches of 128–256 and 20–45× more steps per second. So "Adam
is 19× worse" is properly *"full-batch Adam is 19× worse than L-BFGS on the same full
batch"* — the comparison a second-order method is supposed to win.

The collocation count compounds it: 24 000 residuals against 49 797 trainable parameters
is 0.48, a 2.07× *underdetermined* system, against a literature that prescribes
overdetermining. A quasi-Newton iteration costs linearly in the point count while a
minibatch Adam step does not, so affording 30 000 quasi-Newton iterations is partly a
consequence of our small set — and `adam 10⁵ / qn 10³` may be the rational split at an
overdetermined ratio rather than a mistake
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.31a).

That cost comparison is **hardware-scoped and we say so**: eight CPU cores, float64,
50 309 parameters. On a datacenter GPU an Adam step over a large batch is nearly free
while a strong-Wolfe line search is sequential, so the literature's "computationally
expensive higher-order optimizers" may be true of that regime and false of ours. The
*accuracy* results do not depend on the hardware
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.29).

**How many epochs it needs was never asked until now.** A 54-run sweep of Adam
against quasi-Newton iterations, three seeds on both backends, finds that **the
quasi-Newton budget is the axis that forms the front** — the Adam axis is flat over
two decades once it is funded, and Adam alone never produces a front at all. That is
awkward, because the Adam count is what the shipped default was tuned on. It is also
what the loss landscape predicts: the residual differentiates the network, so its
curvature is badly conditioned and strongly off-diagonal, which a first-order method
with a diagonal preconditioner cannot address (§7.5.11).

Which means **the axis that has never been pushed is the one that matters**. It is
monotone with no measured end at 3000 iterations, while the literature this recipe
comes from runs 30000 — so `T_s = 0.02` may be a *budget* limit rather than the
model's. That experiment is running.

**JAX is now the default backend, and the reason is a two-part correction.** PyTorch
had been beating JAX in every experiment, by a margin that grew with model size — on
identical residuals, which for deterministic mathematics should not happen. It was
one unset argument: `optax.lbfgs()` defaults to keeping **10** curvature pairs and
the PyTorch side was passing **50**. Copy one backend's weights into the other,
verify the objective matches to the last bit, and vary only the optimiser: PyTorch
at 10 reproduces JAX's curve, and JAX at 50 reproduces PyTorch's. Fixed, with the
memory now an explicit shared setting — and **every JAX accuracy number was
superseded and re-measured** (§7.5.17). With that fixed the two agree to 1.08× at
the largest capacity measured.

The second half is speed. Every timing comparison had PyTorch pinned to 8 threads
while JAX quietly used every core, because JAX's CPU backend ignores the variable
PyTorch obeys and sizes its own pool — 291 threads where 8 were asked for. Given the
machine equally and run one at a time, **JAX is 4.4× faster**, and 4.8× on the
quasi-Newton stage, which is the stage that does all the work (§7.5.19). So a
backend that looked slower *and* weaker was neither; both readings were artefacts of
unequal settings. PyTorch stays a first-class arm — two independent implementations
agreeing is the strongest check here, and it is what caught the defect.

[`docs/axial_nn.md`](docs/axial_nn.md) **§0 is the status quo** — accuracy, what is
settled, what is open, and **§0.6 says which configuration to use**. §5–§7 carry
every measurement, including the negative results, which outnumber the positive
ones, and including which of that document's own conclusions have been retracted —
twelve so far, including a whole column of that document's JAX results and a headline of its own that turned out to be a tautology. Thirteen remedies have been argued soundly and refuted by measurement; the negative results outnumber the positive ones by four to one.

**Every deviation from the manual is registered** in
[`docs/axial_physics.md`](docs/axial_physics.md) §3 with its equation number. That
register is a contract, not a commentary: an unregistered deviation is a bug, and
new physics lands **off by default** so no published number moves when it does.

## Quick start

```bash
uv sync                             # create .venv from pyproject + lockfile
uv run pinn-sfr reference           # 0D stiff reference -> results/ (held-out data)
uv run pinn-sfr figures             # draw the 0D transients -> results/figures/ (gitignored)
uv run pinn-sfr axial reference     # 1D axial channel, prescribed power
uv run pinn-sfr axial reference --feedback   # ... with the prompt-jump kinetics closed
uv run pinn-sfr axial figures       # -> results/figures/axial_*.png (gitignored)
uv run pytest                       # the suite; backend tests skip without their extra
```

Both models are solved by **two equally first-class PINN backends** — PyTorch and
JAX — on the same residuals. Each is an optional extra:

```bash
uv sync --extra torch-cpu  && uv run python -m pinn_sfr_transient.axial.pinn_torch
uv sync --extra jax-cpu    && uv run python -m pinn_sfr_transient.axial.pinn_jax
uv sync --extra deepxde --extra torch-cpu && uv run python -m pinn_sfr_transient.pinn_deepxde
```

**CPU is the target.** These are small float64 networks; float64 is throttled to
roughly 1/32–1/64 of FP32 on consumer NVIDIA hardware, which is most of the reason a
GPU does not pay for itself here. CUDA builds exist (swap any `-cpu` extra for
`-gpu`) but are neither required nor benchmarked on the axial model. Google Colab is
**not** a target: the axial PINN needs tens of minutes of CPU per run.

**Pin the thread budget before quoting any timing — or any reproducibility claim.**
Thread count changes float reduction order, so it changes answers and not only
speed. `OMP_NUM_THREADS` binds PyTorch and is **ignored by JAX**, whose CPU backend
sizes its own pool from the core count; use `tools/axial_study.py --cpu-block K`,
which sets CPU affinity. Measured: the same run gives `…040135` on 48 cores and
`…040157` on 8, and pinning makes it bitwise reproducible. The core *count* binds
the answer, *which* cores does not — so concurrent jobs can take different blocks
and stay comparable.

## Documentation

- [`docs/physics_theory.md`](docs/physics_theory.md) — the 0D model: point
  kinetics, lumped thermal-hydraulics, reactivity feedback (Doppler + positive
  sodium void), the ULOF transient, non-dimensionalisation, parameters, caveats.
- [`docs/neural_network.md`](docs/neural_network.md) — the 0D PINN: normalized-state
  formulation, hard-IC ansatz, architecture, Adam→L-BFGS training, the adaptive
  recipe (causal weighting, gradient-norm loss weights, residual-adaptive sampling,
  forward-mode autodiff), and a JAX-vs-PyTorch comparison (§9).
- [`docs/axial_physics.md`](docs/axial_physics.md) — the 1D model: four material
  fields, real sodium properties, saturation-plus-superheat boiling onset, film
  dryout, the prompt-jump kinetics closure, and **the deviation register**.
- [`docs/axial_nn.md`](docs/axial_nn.md) — the 1D PINN: ansatz, hard constraints,
  training recipe, both backends, and every measured result.
- [`docs/sas4a/`](docs/sas4a/) — a local text mirror of the SAS4A/SASSYS-1 manual,
  so an equation citation can be checked without a network round-trip. Fetched by
  [`tools/fetch_sas_manual.py`](tools/fetch_sas_manual.py).
- [`tools/axial_study.py`](tools/axial_study.py) — one sub-command per published
  axial study: `ruler`, `horizon`, `budget`, `grid`, `qnladder`, `optimizer`,
  `parity`, `plan-a`, `combo`, `margin`, `scaling`, `levelset`, `frontfrac`,
  `capacity-optimiser`, `default`, `aniso`, `bands`, `lsinput`, `onset`, `laplace`,
  `regime`, `regime-sign`. Every table in `axial_nn.md` and
  `axial_physics.md` §10 is reproducible by one of them. This exists because it
  once did not: a published configuration differed from the shipped default and
  nobody could tell, since the measurement lived in an uncommitted scratch file.
- [`docs/usage.md`](docs/usage.md) — install, run, train, use as a library, compute
  requirements, troubleshooting.
- [`docs/references.md`](docs/references.md) — annotated bibliography
  ([`docs/references.bib`](docs/references.bib) for LaTeX).
- [`notebooks/01_ulof_walkthrough.ipynb`](notebooks/01_ulof_walkthrough.ipynb) —
  interactive end-to-end walkthrough of the 0D model.
- [`notebooks/02_safety_map.ipynb`](notebooks/02_safety_map.ipynb) —
  parameter-space safety study (peak-power map, phase portraits; numpy/scipy only).

## Layout

```
pinn-sfr-transient/
├── src/pinn_sfr_transient/
│   ├── config.py        # SFRParams (typed; derived steady state)
│   ├── physics.py       # reactivity, void, flow, RHS (numpy)
│   ├── reference.py     # stiff Radau solver -> Trajectory
│   ├── pinn_torch.py    # 0D PyTorch PINN (OO/eager; recipe + RAR)
│   ├── pinn_jax.py      # 0D JAX PINN (functional; Equinox + Optax)
│   ├── pinn_deepxde.py  # 0D DeepXDE variant (same residuals, vanilla loop)
│   ├── plotting.py      # 4-panel reference figure
│   ├── figures.py       # draws the 0D transients into results/ (never committed)
│   ├── cli.py           # `pinn-sfr` (reference | figures | axial …)
│   └── axial/
│       ├── config.py    # AxialParams (mesh, geometry, feedback switches)
│       ├── sodium.py    # the §12.13 sodium property correlations
│       ├── physics.py   # the residuals — ONE definition, shared by all three
│       ├── scoring.py   # ONE scorer, numpy-only, never imported by a loss
│       ├── reference.py # stiff axial solver -> AxialTrajectory
│       ├── figures.py   # draws the axial transients into results/ (never committed)
│       ├── torchpinn/   # PyTorch backend, split config/archs/ansatz/model/
│       │                #   weighting/training/evaluate
│       ├── jaxpinn/     # JAX backend, the same split plus residuals/samplers
│       ├── pinn_torch.py# facade re-exporting torchpinn's public surface
│       └── pinn_jax.py  # facade re-exporting jaxpinn's public surface
├── tests/               # pytest: consistency, physics, CLI, both PINNs, parity,
│                        #   and test_hostile_audit.py — the algebra checked against
│                        #   dense matrices, scipy and sympy rather than against
│                        #   this project's own other half
├── docs/                # theory, usage, references — TEXT AND TABLES ONLY, no
│                        #   images anywhere in this repository;
│                        #   sas4a/ mirrors the manual
├── tools/               # axial_study.py       — every published axial table, one
│                        #   sub-command per study, because a number is reproducible
│                        #   when its configuration is in the repository
│                        # check_published_accuracy.py — the 0D claim, from defaults
│                        # check_markdown.py    — the AGENTS.md rendering rules
│                        # fetch_sas_manual.py  — mirrors the SAS4A manual
├── notebooks/           # guided walkthroughs (outputs stripped; run to reproduce)
└── results/             # held-out reference .npz files (gitignored)
```

The axial backends are **packages, not modules**, split after
[jaxpi2](https://github.com/sifanexisted/jaxpi2) so that an ablation is a config
change rather than an edit to one long file — and so that `evaluate` never being
imported by `training` makes "the reference never enters the loss" a structural
property instead of a convention. They expose the same knobs with the same
defaults, asserted field by field by a test, because that has silently broken twice
([`docs/axial_nn.md`](docs/axial_nn.md) §4).

Tooling: **uv** (project + envs), **ruff** (lint + format, `select = ["ALL"]`),
**ty** (type check), **pre-commit** (local quality gate), **pytest**, and a GitHub
Actions workflow ([`.github/workflows/test.yml`](.github/workflows/test.yml)) with
three jobs: the CPU torch backend on Python 3.14 with an 85% coverage gate, the JAX
backend on 3.13, and a core-only job on 3.13 that proves the optional-import guards
hold. `src/` layout, fully type-hinted, PEP 561 (`py.typed`).

## License

[Blue Oak Model License 1.0.0](LICENSE) (SPDX: `BlueOak-1.0.0`).
