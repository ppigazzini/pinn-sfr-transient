# Usage guide

Everything a reader needs to install, run, train, extend, and troubleshoot
`pinn-sfr-transient`. For the science see `docs/physics_theory.md` and
`docs/neural_network.md`; for citations see `docs/references.md`.

---

## 1. Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | ≥ 3.12 | everything (uv can install it for you); 3.12 matches Colab |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.6 | project & environment management |
| git | any recent | cloning / version control |
| PyTorch | ≥ 2.12 | only the PINN (`pinn_torch`) — optional extra |
| DeepXDE | ≥ 1.11 | only the DeepXDE variant — optional extra |

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
uv sync --extra torch-cpu                  # PyTorch ≥ 2.12, CPU-only wheel (small)
uv sync --extra jax-cpu                    # JAX (Equinox + Optax), CPU-only
uv sync --extra torch-gpu                  # CUDA PyTorch; both -gpu builds train ~5x faster on a Colab T4
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

Every figure shown in the README and docs is rebuilt from the model by:

```bash
uv run pinn-sfr figures              # -> docs/img/*.png
uv run pinn-sfr figures --no-pinn    # skip the optional PINN overlay
```

This writes the reference transient, the reactivity decomposition, the phase
portrait, the void-coefficient sweep and the peak-power safety map to
`docs/img/`. With the `torch` extra installed it also trains a short PINN and
adds `pinn_overlay.png`. Figures are always regenerated from
`src/pinn_sfr_transient/figures.py`, never committed from notebook output.

### 3.2 Interactive notebook (recommended for a first read)

A guided Jupyter notebook walks through the whole model — reference simulation,
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
uv run pytest                     # 4 consistency tests (numpy/scipy only)
uv run ruff check .               # lint
uv run ruff format --check .      # formatting
uv run ty check                   # type check
```

The tests verify the nominal state is an exact fixed point, that the reference
satisfies the ODEs, and crucially that the PINN's *normalized* residuals equal
the physical ODEs to machine precision — so the deep-learning math is validated
even without PyTorch installed.

Enable the git hooks (run ruff/ty/pytest automatically on every commit):

```bash
uv run pre-commit install
```

---

## 5. Train the PINN

Two from-scratch backends solve the *same* normalized residuals (no data —
physics only), then print the relative-L2 error against the held-out reference.
Generate the reference first (`uv run pinn-sfr reference`). Both backends train on
the *same* optimisation budget and fit comparably; a GPU speeds them up ~5× on a
Colab T4 (about a minute, vs several on CPU — varies a lot by instance).
`docs/neural_network.md` §9 compares the two (they are equally first-class).

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
    compile=False,              # set True to try torch.compile
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
Optax (`optax.adam` then `optax.lbfgs`), same recipe *and same budget* as §5.1. On
a GPU runtime XLA compiles and runs the whole step on the GPU, several times faster
than CPU on a Colab T4; the CPU wall-clock varies a lot by Colab instance. (Use a
GPU, not a TPU: TPUs lack the required float64.)

### 5.3 DeepXDE variant

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

**No GPU is required.** This is a small problem:

* The reference solver, tests, lint, and type-check are **CPU-only** and
  finish in seconds.
* The PINN is a tiny MLP (~17k parameters, 1-D input, ~4k-point batches) trained
  in **float64**. Memory use is well under 1 GB.
* A GPU helps here: on a Colab T4 **both** backends train ~5× faster than on its
  (modest) CPU. The margin is smaller than for an fp32 workload — float64 is
  throttled to ≈1/32–1/64 of FP32 on consumer NVIDIA GPUs — so a strong desktop
  CPU can stay competitive with a gaming GPU, but Colab's CPU is not, so the GPU
  wins clearly there.

To use a GPU, set `device="cuda"` (or `"mps"` on Apple Silicon) in `TrainConfig`
and install the CUDA build with `uv sync --extra torch-gpu` (the `torch-cpu` and
`torch-gpu` extras are mutually exclusive; `--extra torch-gpu` pulls the CUDA
`torch` wheel and the nvidia stack from PyPI). The notebook auto-selects the GPU
when one is present.

Beyond this problem a GPU pays off even more if you scale up — much larger
networks, or the parametric / operator-learning extension (DeepONet / FNO over
many ULOF scenarios
in float32).

---

## 8. Reproducibility

* `seed` in `TrainConfig` fixes the PyTorch RNG; the reference solver is
  deterministic.
* Optionally commit `uv.lock` and use `uv sync --frozen` to pin exact dependency
  versions across machines.
* Quality is enforced locally by pre-commit (`uv run pre-commit install`):
  ruff (lint + format), ty, pytest, and file-hygiene hooks.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PyTorch >= 2.12 is required` on import | run `uv sync --extra torch-cpu` (or `--extra torch-gpu`) |
| `DeepXDE is required` | run `uv sync --extra deepxde --extra torch-cpu`; set `DDE_BACKEND=pytorch` |
| `uv sync --frozen` errors | no committed lockfile → `uv lock` (or use plain `uv sync`) |
| `[pinn] forward-mode autodiff unavailable…` | harmless; it auto-falls back to reverse mode. To force it, set `jacobian="reverse"` |
| Validation step says "Run `pinn-sfr reference` first" | generate the reference `.npz` before training |
| `ty` flags `torch`/`deepxde` symbols | expected without the extras; ty treats those optional imports as untyped |
| Training loss plateaus / diverges | raise `causal_eps`, lower `lr`, or increase `lbfgs_iters`; check `device`/precision |

---

## 10. Where to go next

* `docs/physics_theory.md` — the model, equations, and parameters.
* `docs/neural_network.md` — PINN architecture and the training recipe.
* `docs/references.md` / `docs/references.bib` — annotated bibliography.
