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
| PINN | **meets its bar** — a few 1e-3 relative L2 | **0.020 against a 0.01 bar**; the *front* is solved to 99.5% |
| backends | PyTorch, JAX, DeepXDE | **JAX** (default), PyTorch |

This README is a map. The physics, the neural-network methodology, and the usage
details live in [`docs/`](docs/) — see [Documentation](#documentation).

## The 0D model — a solved problem

![ULOF reference transient — power, temperatures, sodium void fraction, and reactivity/flow](docs/img/ulof_reference.png)

Loss of flow drives the coolant past the void-onset temperature; the positive
void coefficient pushes power to **1.38× nominal at ≈ 23 s**, then negative
Doppler feedback dominates and the power turns over, settling to ≈ 0.69× — a
bounded, self-limiting transient. Trained on residuals alone, the PINN recovers
the whole trajectory ([`docs/neural_network.md`](docs/neural_network.md) §7).

## The 1D axial model — the current work

![Axial boiling — voided length against time, and the saturation level set that defines the front](docs/img/axial_front.png)

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

The PINN trains, satisfies every hard constraint exactly, and **does not meet its
1% accuracy bar** — 0.029 relative L2 on `T_s` at the shipped defaults, **0.020** at
the best measured configuration. The reference's own error is 1.1–1.6e-3, so the
bar sits 6–9× above the ruler and the failure is the network's, not the ruler's.

**The boiling front, though, is essentially solved.** Giving the Fourier embedding
several frequency bands at once — a low band for the smooth bulk, a high one for the
near-discontinuous front, at the same total feature count and the same wall-clock —
reaches **99.5% of the reference's voided length** at three seeds, against 64% for
the shipped default, while *also* improving the mean error and the saturation
margin. Every other lever in this project trades one against the other; this is the
first that moves both, and the mechanism is legible enough to say why
([`docs/axial_nn.md`](docs/axial_nn.md) §7.5.14).

**And onset location is exact.** Boiling starts at the *maximum* of the coolant
temperature, where the axial profile is 11.6× flatter than at its steepest and one
mesh cell is 0.4 K — so reading a *position* off a *value* threshold there scales as
a square root of the field error, which is the worst possible law. Solving the
tangency conditions instead (the field touches saturation, and touches it
tangentially) puts onset within **0.00 cells on every seed of every arm**, against
2.7–4.0 cells for thresholding.

That fixed the wrong half of the problem, which is the interesting part: onset
*time* was passing only because the scoring grid is 0.25 s and quantised it
favourably. Measured without quantisation it is 0.62–0.84 s against a 0.5 s
criterion — so M4's binding constraint has flipped from *where* to *when* (§7.5.16).
The same arithmetic questions the criterion itself: one cell is only 1.6–2.3× above
the *reference solver's own* error.

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
eleven so far, and one of them was a whole column of that document's JAX results.

**Every deviation from the manual is registered** in
[`docs/axial_physics.md`](docs/axial_physics.md) §3 with its equation number. That
register is a contract, not a commentary: an unregistered deviation is a bug, and
new physics lands **off by default** so no published number moves when it does.

## Quick start

```bash
uv sync                             # create .venv from pyproject + lockfile
uv run pinn-sfr reference           # 0D stiff reference -> results/ (held-out data)
uv run pinn-sfr figures             # (re)generate the 0D figures -> docs/img/
uv run pinn-sfr axial reference     # 1D axial channel, prescribed power
uv run pinn-sfr axial reference --feedback   # ... with the prompt-jump kinetics closed
uv run pinn-sfr axial figures       # -> docs/img/axial_*.png
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
│   ├── figures.py       # regenerates the 0D docs/img/ figures
│   ├── cli.py           # `pinn-sfr` (reference | figures | axial …)
│   └── axial/
│       ├── config.py    # AxialParams (mesh, geometry, feedback switches)
│       ├── sodium.py    # the §12.13 sodium property correlations
│       ├── physics.py   # the residuals — ONE definition, shared by all three
│       ├── scoring.py   # ONE scorer, numpy-only, never imported by a loss
│       ├── reference.py # stiff axial solver -> AxialTrajectory
│       ├── figures.py   # regenerates the axial docs/img/ figures
│       ├── torchpinn/   # PyTorch backend, split config/archs/ansatz/model/
│       │                #   weighting/training/evaluate
│       ├── jaxpinn/     # JAX backend, the same split plus residuals/samplers
│       ├── pinn_torch.py# facade re-exporting torchpinn's public surface
│       └── pinn_jax.py  # facade re-exporting jaxpinn's public surface
├── tests/               # pytest: consistency, physics, CLI, both PINNs, parity,
│                        #   and test_hostile_audit.py — the algebra checked against
│                        #   dense matrices, scipy and sympy rather than against
│                        #   this project's own other half
├── docs/                # theory, usage, references; img/ holds ALL figures;
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
