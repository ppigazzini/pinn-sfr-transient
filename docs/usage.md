# Usage guide

Everything a reader needs to install, run, train, extend, and troubleshoot
`pinn-sfr-transient`. For the science see [`physics_theory.md`](physics_theory.md)
and [`neural_network.md`](neural_network.md) (the 0D model) or
[`axial_physics.md`](axial_physics.md) and [`axial_nn.md`](axial_nn.md) (the 1D
axial boiling model); for citations see [`references.md`](references.md).

---

## 1. Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | ≥ 3.13 | everything (uv can install it for you) |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.6 | project & environment management |
| git | any recent | cloning / version control |
| PyTorch | ≥ 2.13 | the PyTorch PINN backends — optional extra |
| JAX + Equinox + Optax | ≥ 0.11 / ≥ 0.13.8 / ≥ 0.2.8 | the JAX PINN backends — optional extra |
| DeepXDE | ≥ 1.15 | the DeepXDE 0D variant — optional extra |

Google Colab is **not** a target. The axial PINN needs tens of minutes of CPU per
run, so a hosted runtime was never where it would be trained, and holding the
Python floor at Colab's version cost a compatibility shim for no benefit.

Install uv (one line; nothing else is required globally):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv provisions the right Python automatically, so you do **not** need to install
Python yourself.

---

## 2. Install

```bash
git clone https://github.com/ppigazzini/pinn-sfr-transient.git
cd pinn-sfr-transient

uv sync            # creates .venv with core + dev deps (numpy, scipy, ruff, ty, pytest…)
```

`uv sync` reads `pyproject.toml` (+ `uv.lock` if present) and builds an isolated
`.venv/`. The package installs in editable mode, so `import pinn_sfr_transient`
and the `pinn-sfr` command work immediately.

**Optional deep-learning backends** (install only if you train a PINN). Each
framework has a `-cpu` and a `-gpu` build; the two builds of one framework are
**mutually exclusive**, but PyTorch and JAX can be installed together:

```bash
uv sync --extra torch-cpu                  # PyTorch ≥ 2.13, CPU-only wheel (small)
uv sync --extra jax-cpu                    # JAX (Equinox + Optax), CPU-only
uv sync --extra torch-gpu                  # CUDA PyTorch (large; see §7 before you do)
uv sync --extra deepxde --extra torch-cpu  # DeepXDE + a torch backend
```

The `torch-cpu` wheel comes from the dedicated PyTorch CPU index (configured in
`pyproject.toml` under `[tool.uv.sources]`), so it skips the multi-GB CUDA stack;
JAX's CPU/CUDA wheels are plain PyPI packages.

> Optional: commit a lockfile for reproducible installs across machines:
> `uv lock && git add uv.lock`.

---

## 3. Run the reference simulation (no PyTorch needed)

This integrates the stiff ULOF system and writes the held-out reference
trajectory (data only — the `.npz` the PINN trainers validate against). Figures
are produced separately by `pinn-sfr figures` (§3.1), so PNGs only ever live in
`docs/img/`.

```bash
uv run pinn-sfr reference
```

Options:

```bash
uv run pinn-sfr reference --t-end 60 --n-out 2000 --outdir results
```

| Flag | Default | Meaning |
|---|---|---|
| `--t-end` | `60` | transient horizon in seconds |
| `--n-out` | `2000` | number of output time samples |
| `--outdir` | `results` | where to write outputs |

Output written to `results/` (gitignored, regenerable):

* `ulof_reference.npz` — the trajectory (`t, P, C, Tf, Tc`), used **only** for
  test-time PINN validation.

Console prints peak power, peak temperatures, and peak void fraction.

### 3.1 Regenerate the documentation figures

Every figure shown in the README and docs is rebuilt from the model — one command
per model:

```bash
uv run pinn-sfr figures              # 0D  -> docs/img/*.png
uv run pinn-sfr figures --no-pinn    # ... skipping the optional PINN overlay
uv run pinn-sfr axial figures        # 1D  -> docs/img/axial_*.png
```

The 0D command writes the reference transient, the reactivity decomposition, the
phase portrait, the void-coefficient sweep and the peak-power safety map; with the
`torch` extra installed it also trains a short PINN and adds `pinn_overlay.png`. The
axial command writes the four material fields, the boiling front (voided length and
the saturation level set) and the closed-loop reactivity split. Figures are always
regenerated from `figures.py` — `src/pinn_sfr_transient/figures.py` and
`src/pinn_sfr_transient/axial/figures.py` — and never committed from notebook output.

### 3.2 Interactive notebook (recommended for a first read)

A guided Jupyter notebook walks through the 0D model end to end — reference simulation,
the four-panel plot, the feedback decomposition, the normalized-residual
verification, a void-coefficient parameter sweep, and a short PINN demo:

```bash
uv sync --extra notebook          # adds JupyterLab + ipykernel
uv run jupyter lab notebooks/01_ulof_walkthrough.ipynb
```

Sections 1–6 need only numpy/scipy; the PINN cell is guarded and is skipped
cleanly if `torch` is absent (add `--extra torch-cpu` to enable it).

---

## 4. Run the tests and quality checks

```bash
uv run pytest                     # the full suite (236 tests with both extras)
uv run pytest --no-cov -k physics # a quick subset, no coverage
uv run ruff check .               # lint
uv run ruff format --check .      # formatting
uv run ty check                   # type check (src only)
```

Backend tests use `pytest.importorskip`, so they **skip** rather than fail when
that extra is absent; the numpy/scipy core runs on its own.

The suite verifies that the nominal state is an exact fixed point, that each
reference trajectory satisfies its own equations, and — the load-bearing part —
that every backend's residuals are algebraically identical to the numpy `physics.py`
they claim to solve. `tests/test_consistency.py` does this for the 0D model and
`tests/axial/test_axial_pinn.py` for the axial one, so the deep-learning maths is
validated even with no framework installed. A third test asserts that the two axial
backends expose equal block counts, equal input widths and field-by-field equal
defaults, because a feature landing in one backend and not the other forks the model
silently and makes every cross-backend number a comparison of two different things.

Enable the git hooks (run ruff/ty/pytest automatically on every commit):

```bash
uv run pre-commit install
```

---

## 5. Train the PINN

Two from-scratch backends solve the *same* normalized residuals (no data —
physics only), then print the relative $L_2$ error against the held-out reference.
Generate the reference first (`uv run pinn-sfr reference`). Both backends train on
the *same* optimisation budget and fit comparably.
[`neural_network.md`](neural_network.md) §9 compares the two (they are equally
first-class). §5.3 below covers the axial model, which has its own two backends and
its own accuracy story.

### 5.1 PyTorch

```bash
uv sync --extra torch-cpu          # or --extra torch-gpu for a CUDA build
uv run python -m pinn_sfr_transient.pinn_torch
```

Object-oriented / eager implementation. All knobs live in `TrainConfig`
(`src/pinn_sfr_transient/pinn_torch.py`):

```python
from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.pinn_torch import TrainConfig, train, predict, relative_l2
import numpy as np

cfg = TrainConfig(
    width=64, depth=5,          # MLP size
    n_colloc=4000,              # collocation points per step
    adam_iters=15000,           # Adam iterations
    lbfgs_iters=600,            # L-BFGS polish iterations
    causal_eps=1.0,             # causal-weighting strength (Wang et al. 2024)
    causal_chunks=32,           # time chunks for causal weighting
    weight_update_every=250,    # grad-norm block-weight rebalancing cadence
    rar_every=2000,             # residual-adaptive resampling cadence (Wu et al. 2023)
    rar_add=200, rar_cap=4000,  # RAR points added / reservoir cap
    jacobian="forward",         # "forward" (torch.func) or "reverse" (autograd)
    device="cpu",               # "cpu" | "cuda" | "mps"
    seed=0,
)

model = train(SFRParams(), cfg)
pinn = predict(model)

ref = dict(np.load("results/ulof_reference.npz"))
print(relative_l2(pinn, ref))   # {'P': ..., 'Tf': ..., 'Tc': ...}
```

What the adaptive pieces do (see `docs/neural_network.md` §4):

* **Causal weighting** — fits earlier times before later ones across the void
  front. Raise `causal_eps` for stricter causality (slower, more stable).
* **Adaptive block weights** — auto-balances the four residual blocks by gradient
  norm; nothing to tune by hand.
* **RAR** — adds high-residual collocation points over time; increase `rar_add`
  for sharper fronts.
* **`jacobian="forward"`** — fast `torch.func` time-derivative; falls back to
  reverse mode automatically if a build can't compose the transforms.

### 5.2 JAX (Equinox + Optax)

```bash
uv sync --extra jax-cpu          # or --extra jax-gpu for a CUDA build
uv run python -m pinn_sfr_transient.pinn_jax
```

Functional implementation — an Equinox model (immutable PyTree) trained with
Optax (`optax.adam` then `optax.lbfgs`), same recipe *and same budget* as §5.1.
CPU wall-clock varies a lot by machine. If you do reach for an accelerator, use a
**GPU, not a TPU**: TPUs lack the float64 this stiff problem needs, and JAX silently
falls back to CPU on a TPU runtime.

### 5.3 The axial boiling model

The 1D axially resolved channel has its own sub-commands. It needs only
numpy/scipy/matplotlib for the reference and the figures:

```bash
uv run pinn-sfr axial reference                 # prescribed power -> results/axial_reference.npz
uv run pinn-sfr axial reference --feedback      # closed prompt-jump kinetics
uv run pinn-sfr axial reference --n-axial 320   # 160 or more is mesh-converged
uv run pinn-sfr axial figures                   # -> docs/img/axial_*.png
```

The reference sub-command prints boiling onset time and location, peak cladding
temperature, voided length, the energy-balance closure and — under `--feedback` —
the `ρ/β` pole tripwire together with the Doppler and void components separately.
That split matters: the net reactivity is not small because the two mechanisms
cancel. See [`axial_physics.md`](axial_physics.md).

Training the axial PINN is a Python entry point rather than a CLI sub-command,
because it takes tens of minutes on CPU:

```bash
OMP_NUM_THREADS=8 uv run python -m pinn_sfr_transient.axial.pinn_torch
OMP_NUM_THREADS=8 uv run python -m pinn_sfr_transient.axial.pinn_jax
```

**Pin the thread count.** `OMP_NUM_THREADS` defaults to every core, so two
concurrent runs oversubscribe and each reports a wall-clock that says more about
the other run than about the code. Thread count also changes float reduction order,
so a run is only reproducible against a stated thread budget — with one pinned, the
torch backend reproduces to four digits run to run.

Each backend is a **package**, and the entry-point module above is a facade over it:

| | PyTorch | JAX |
|---|---|---|
| package | `axial/torchpinn/` | `axial/jaxpinn/` |
| modules | `config`, `archs`, `ansatz`, `model`, `weighting`, `training`, `evaluate` | `config`, `archs`, `ansatz`, `residuals`, `weighting`, `samplers`, `training`, `evaluate` |
| knobs | `TrainConfig` — the same fields, with the same defaults, in both | |

Two modules do not mirror each other, and that is torch's idiom rather than a
design choice: `nn.Module` owns its parameters *and* its forward pass, so the ansatz
and the residuals share `torchpinn.model`; and the sampler needs the model to place
points on the predicted front while the loop needs mutable optimiser state, so both
share `Trainer` in `torchpinn.training`.

Import from either the facade or the package — they are the same objects:

```python
from pinn_sfr_transient.axial.pinn_torch import TrainConfig, train
from pinn_sfr_transient.axial.torchpinn import TrainConfig, train  # identical
```

**The training horizon is a scope decision, not a tuning knob.** `t_train_frac`
defaults to `0.275` — 16.5 s of the 60 s horizon — because that is where the
channel leaves the §12.13 sodium property range and the reference stops, on every
mesh from `n = 40` to `n = 640`. It defaulted to `1.0` until recently, which trains
over 72% of a horizon where the model does not apply and **forms no boiling front
at all**; see [`axial_nn.md`](axial_nn.md) §7.2.7. Plan A (`feedback=True`) needs no
truncation and uses the full horizon.

**Every published table is reproducible by a command.** Each study in
[`axial_nn.md`](axial_nn.md) is a sub-command of
[`../tools/axial_study.py`](../tools/axial_study.py):

```bash
OMP_NUM_THREADS=8 uv run python tools/axial_study.py ruler      # reference mesh convergence
OMP_NUM_THREADS=8 uv run python tools/axial_study.py horizon    # the training horizon
OMP_NUM_THREADS=8 uv run python tools/axial_study.py budget     # Adam vs quasi-Newton split
OMP_NUM_THREADS=8 uv run python tools/axial_study.py optimizer  # L-BFGS vs self-scaled BFGS
OMP_NUM_THREADS=8 uv run python tools/axial_study.py parity     # torch vs JAX
OMP_NUM_THREADS=8 uv run python tools/axial_study.py plan-a     # closed-loop power
```

The training studies take tens of minutes per arm. `ruler` is 42 seconds and needs
no deep-learning extra at all.

**Accuracy: do not quote it from here.** The axial PINN does **not** meet its 1%
bar. [`axial_nn.md`](axial_nn.md) §5–§7 carries every measurement, including which
of them are superseded, and it is the only place in this repository where an axial
accuracy number is quoted.

### 5.4 DeepXDE variant

```bash
uv sync --extra deepxde --extra torch-cpu
DDE_BACKEND=pytorch uv run python -m pinn_sfr_transient.pinn_deepxde
```

Same normalized residuals, but a *vanilla* framework-driven training loop — no
causal weighting, RAR, or gradient-norm balancing. On this stiff problem it
**under-fits**: the residual reaches ~1e-6 while the power is still ~28% off. It is
a baseline that shows the same physics in a high-level library (and why the §4
recipe matters), **not a recommended solver here** — use the PyTorch or JAX
backend. See `docs/neural_network.md` §6.

---

## 6. Use it as a library

The public API (`pinn_sfr_transient/__init__.py`) is import-ready:

```python
from pinn_sfr_transient import SFRParams, solve_reference, void_fraction

p = SFRParams(t_end=40.0, alpha_void=8e-3)   # override any parameter
traj = solve_reference(p, n_out=500)
print(traj.P.max(), traj.Tc.max())
print(void_fraction(traj.Tc.max(), p))
```

`SFRParams` (`src/pinn_sfr_transient/config.py`) exposes every physical
parameter; derived constants (`UA`, `W0`, `Cf`, `Cc`, `beta_i`) and the
criticality offset are recomputed automatically, so the nominal steady state
stays exact whatever you change.

### Physical vs demonstration parameters
The default `T_onset = 820 K` is a *demonstration* hot-channel boiling threshold,
not the ~1156 K sodium saturation at 1 atm. For a more physical run, raise
`T_onset` and shorten `tau_pump` (more aggressive coast-down) so the transient
still reaches voiding.

---

## 7. Compute requirements

**CPU is the target, and no GPU is required.**

| workload | cost |
|---|---|
| reference solvers, tests, lint, type-check | seconds to a few minutes, CPU-only |
| 0D PINN, either backend | minutes on CPU (~5 to ~13 min for the same workload across machines) |
| axial PINN, either backend | tens of minutes on CPU |

Both PINNs are small MLPs trained in **float64** — the 0D one is ~17k parameters
over a 1-D input with ~4k-point batches — and memory use stays well under 1 GB.

**Why a GPU does not pay for itself here.** float64 is throttled to roughly
1/32–1/64 of FP32 on consumer NVIDIA hardware, and these networks are too small to
saturate a device in the first place, so the kernel-launch and transfer overheads eat
a large share of the step. A strong desktop CPU stays competitive with a gaming GPU.
The axial model has never been benchmarked on a GPU
([`axial_nn.md`](axial_nn.md) records that as a deliberate non-goal, not a gap).

CUDA builds do exist if you want to try: `uv sync --extra torch-gpu` or
`--extra jax-gpu` (each framework's `-cpu` and `-gpu` extras are mutually exclusive;
the two frameworks can coexist), then set `device="cuda"` — or `"mps"` on Apple
Silicon — in `TrainConfig`. Measure before you believe it, and exclude the first
iteration, which is compilation rather than computation.

A GPU would pay off if the problem were scaled up: much larger networks, or the
parametric / operator-learning extension (DeepONet / FNO over many ULOF scenarios in
float32).

---

## 8. Reproducibility

* `seed` in `TrainConfig` fixes the RNG, and both backends seed **before** module
  construction, because `nn.init` draws from the global RNG. The reference solvers
  are deterministic.
* **Pin `OMP_NUM_THREADS`.** Thread count changes float reduction order, so a
  "reproducible" result that was never pinned to a thread budget is not
  reproducible. With one pinned, the torch axial backend reproduces to four digits.
* Record a regression number *before* a refactor, not after. Splitting the JAX
  backend into modules once dropped `jax_enable_x64` and ran the whole thing in
  float32 with the entire suite green; only a number taken beforehand caught it.
* Optionally commit `uv.lock` and use `uv sync --frozen` to pin exact dependency
  versions across machines.
* Quality is enforced locally by pre-commit (`uv run pre-commit install`):
  ruff (lint + format), ty, pytest, and file-hygiene hooks.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PyTorch >= 2.13 is required` on import | run `uv sync --extra torch-cpu` (or `--extra torch-gpu`) |
| `JAX backend requires …` on import | run `uv sync --extra jax-cpu` (or `--extra jax-gpu`) |
| `DeepXDE is required` | run `uv sync --extra deepxde --extra torch-cpu`; set `DDE_BACKEND=pytorch` |
| `uv sync --frozen` errors | no committed lockfile → `uv lock` (or use plain `uv sync`) |
| `[pinn] forward-mode autodiff unavailable…` | harmless; it auto-falls back to reverse mode. To force it, set `jacobian="reverse"` |
| Validation step says "Run `pinn-sfr reference` first" | generate the reference `.npz` before training |
| `ty` flags `torch`/`deepxde` symbols | expected; the optional backends are excluded from `ty` in `pyproject.toml` precisely so the gate does not flip on whether an extra is installed |
| Training loss plateaus / diverges | raise `causal_eps`, lower `lr`, or increase `lbfgs_iters`; check `device`/precision |
| An axial run takes far longer than the last one | another run is oversubscribing the cores; pin `OMP_NUM_THREADS` on both |
| Two axial runs disagree at the same seed | check the thread budget matches before suspecting the code |

---

## 10. Where to go next

* [`physics_theory.md`](physics_theory.md) — the 0D model, equations, parameters.
* [`neural_network.md`](neural_network.md) — the 0D PINN architecture and recipe.
* [`axial_physics.md`](axial_physics.md) — the 1D axial model and the deviation
  register against the SAS4A/SASSYS-1 manual.
* [`axial_nn.md`](axial_nn.md) — the axial PINN and every measurement taken on it.
* [`sas4a/`](sas4a/) — the local text mirror of the manual, so an equation citation
  can be checked without a network round-trip.
* [`references.md`](references.md) / [`references.bib`](references.bib) — annotated
  bibliography.
