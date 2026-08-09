# AGENTS.md

Guidance for AI coding agents working in this repository. Humans should read
[README.md](README.md) and [docs/](docs/) first; this file captures the
conventions and commands an agent needs to make correct, low-friction changes.

## What this is

A Physics-Informed Neural Network (PINN) for the **SFR Unprotected-Loss-of-Flow
(ULOF)** transient: six-group point kinetics + lumped fuel/coolant thermal
hydraulics with a **positive sodium-void** feedback. A stiff `scipy` Radau solve
is the held-out reference; three PINN backends (from-scratch PyTorch, from-scratch
JAX/Equinox+Optax, and DeepXDE) learn the same non-dimensionalised residuals.

## Environment

- **Python >= 3.13**, managed with **uv**. Google Colab is *not* a target (the
  axial PINN needs tens of minutes of CPU per run). Every
  module uses `from __future__ import annotations` for portability across
  3.13-3.14 (e.g. `TYPE_CHECKING`-only names in runtime annotations).
- Core deps (`numpy`, `scipy`, `matplotlib`) are always installed. The three
  deep-learning backends are **optional extras** and import-guarded: importing
  `pinn_torch` / `pinn_jax` / `pinn_deepxde` without the extra raises `SystemExit`,
  it does not crash. Each framework has a `-cpu` and a `-gpu` build (mutually
  exclusive within a framework; the two frameworks can coexist).

```bash
uv sync                                    # core + dev tools (ruff, ty, pytest, pytest-cov)
uv sync --extra torch-cpu                  # PyTorch PINN (CPU wheel; --extra torch-gpu for CUDA)
uv sync --extra jax-cpu                    # JAX PINN, Equinox + Optax (--extra jax-gpu for CUDA)
uv sync --extra deepxde --extra torch-cpu  # DeepXDE variant (needs a torch backend)
```

## Commands

```bash
uv run pinn-sfr reference            # run the stiff reference sim -> results/
uv run pytest                        # tests + coverage (term-missing)
uv run pytest --no-cov -k physics    # quick subset, no coverage
uv run ruff check --fix && uv run ruff format
uv run ty check                      # type check (src only)
```

`results/` is **generated** (figure + `.npz`) and git-ignored — never commit its
contents; the CLI recreates the directory on demand.

## Code style & conventions

- **Ruff is configured with `select = ["ALL"]`.** Before adding a blanket
  `# noqa` or a new global ignore, check `[tool.ruff.lint]` in `pyproject.toml`:
  exclusions are grouped and justified (formatter conflicts, physics notation,
  PEP 649, etc.). Prefer fixing over ignoring; if you must suppress, scope it.
- **Physics notation wins over snake_case.** `P`, `T_f`, `Tc`, `UA`, `dT_void`,
  `R_p` are intentional; the `N8xx` naming rules are disabled for this reason.
- Docstrings follow the **numpy** convention; line length is **100**.
- The numpy `physics.py` RHS and every PINN backend's residuals must stay
  algebraically identical — `tests/test_consistency.py` enforces this for the 0D
  model and `tests/axial/test_axial_pinn.py` for the axial one. If you touch a
  model, update its `physics.py` **and** all its backends together and keep those
  tests green.
- **The axial model has two backends and they must expose the same knobs.** A
  feature that lands in `axial/pinn_torch.py` and not `axial/pinn_jax.py` forks
  the model silently and makes every cross-backend number a comparison of two
  different things; that happened once and cost a published table.
  `tests/axial/test_axial_pinn_jax.py` asserts equal block counts, equal input
  widths and field-by-field equal defaults.
- **Deviations from the manual are a contract, not a comment.** Anything the
  axial model does differently from SAS4A belongs in the `docs/axial_physics.md`
  register with its equation number. An unregistered deviation is a bug. New
  physics goes in **off by default**, so no published number moves when it lands.

## Measurements

The axial model's seed spread has been as large as 12.5x (`docs/axial_nn.md` §7.1),
and single-seed conclusions in this project have been overturned by the next seed
four times: D38, D39, the budget sweep's "monotonic" front degradation, and §7.3.2's
"consistent 21%" backend gap.

- **Never write a comparative headline from one seed.** Three seeds with per-seed
  ranges, or say "seed N, one sample" in the sentence that states the result — not
  in a caveat further down. A hedge below a confident headline does not work.
- **A retraction needs the same evidence as the claim.** A three-seed result is not
  overturned by one contradicting seed. `docs/axial_nn.md` §7.3.2 was retracted on
  seed 1 and then re-confirmed at three, which repeated the error it was retracting.
- **Do not pair seed *indices* across backends.** `seed=1` seeds two different RNG
  implementations drawing two different initialisations, so torch's seed 1 and JAX's
  seed 1 are unrelated draws. Elementwise ratios of them are noise; compare the
  distributions. Index-paired, the backend gap reads 1.167 / 0.997 / 1.366; as
  distributions it reads 1.158 / 1.166 / 1.177.
- **Every published table must be reproducible by a committed command.**
  `tools/axial_study.py` has one sub-command per study. A number measured by an
  uncommitted script is not reproducible, however carefully it was measured: D67 is
  the case where that hid a default which produced no boiling front at all, for four
  milestones.
- **Include a control arm that reproduces something already published**, and check
  it before reading the new arms. That is what caught D67.
- **`OMP_NUM_THREADS` does not bind JAX.** XLA's CPU backend sizes its own pool from
  `hardware_concurrency()`: an arm nominally at 8 threads was measured creating 291.
  Since thread count changes float reduction order, that is a correctness problem,
  not a timing one — the same run gives `...040135` on 48 cores and `...040157` on 8.
  Pin with `axial_study.py --cpu-block K`, which sets CPU affinity. The core *count*
  binds the answer; *which* cores does not, so concurrent studies take different
  blocks and stay comparable.
- **A wall-clock needs a stated thread budget and contention level.**
  `OMP_NUM_THREADS` defaults to every core, and thread count changes float
  reduction order — so it changes answers, not just timings. Concurrency at a
  *fixed* thread count does not (measured: identical digits, 1.48x the time).
- **Measure on the axis the decision is made on.** A curvature-memory sweep was
  monotone at equal *iterations* — 300 pairs looked 1.50x better than 50, with no
  turning point. At equal *wall-clock* the same arms reverse: 300 is 1.09x worse
  and the optimum is 100. A ladder measured on the wrong x-axis is not a weak
  result, it is an inverted one.
- **Two implementations of the same algorithm must be compared at equal
  hyper-parameters, and that has to be checked rather than assumed.** `optax.lbfgs`
  defaults to `memory_size=10`; `torch.optim.LBFGS` was passed `history_size=50`.
  That single unset argument was the entire cross-backend accuracy gap, and it was
  read as a framework difference for four milestones. When two backends disagree,
  diff the *arguments* before theorising about the libraries.
- **An ablation is a statement about the formulation it was run on.** Change the
  formulation and every negative result on the shelf is provisional again (D59).

## Docs & Markdown math (GitHub renders these)

Broken inline LaTeX is a recurring problem — GitHub's renderer is strict. Rules:

- **Never split one token across the math/text boundary.** `$^{238}$U` renders as
  a *dangling superscript* — garbage. Write isotopes as plain text (`U-238`,
  `Pu-240`) or as a single complete span (`${}^{238}\mathrm{U}$`). Likewise no
  `$_{f}$T`; keep the whole symbol in one span (`$T_f$`).
- **No bare sub/superscript spans** (`$^{...}$`, `$_{...}$` with no base).
- **No escaped `\_` or `\^` inside `\text{}`** ("`_` allowed only in math mode") —
  use `n_{\mathrm{in}}`, not `\text{fan\_in}`.
- **No spacing macros (`\!`, `\,`, `\;`) in *inline* `$...$`.** GitHub/IDE inline
  renderers print `\!` as a literal `!` (`$\sim\!10^3$` → "∼!10³"). Fine inside
  ` ```math ` blocks (MathJax), just not inline.
- **Keep a numeric range in ONE inline span; never split it into two `$...$` around a
  dash.** `$10^4$–$10^5$` fails — the second span (a `$` flanked by a dash/digit) is
  not recognised as math and renders literally as `$10^5$`. Write the whole range in one
  span with a word (`$\sim10^4\text{ to }10^5$`) or fully in plain text (`~10⁴–10⁵`).
  Approximations/comparators in prose go in plain text too (`~1.6–2×`, `≳8000`).
- **Don't glue an opening `$` to a preceding `~`, `-`, or `–`.** `~$10^{-3}$` and
  `relative-$L_2$` render literally — the opening `$` needs whitespace (or `(`) before it.
  Put the `~` inside the span (`$\sim10^{-3}$`) and use a space not a hyphen
  (`relative $L_2$`).
- Prefer ` ```math ` fenced blocks for display equations; keep `$` delimiters
  balanced on each line (an odd count breaks the whole line).
- **When wrapping prose, never start a continuation line with `+ `, `- `, or
  `* `** — Markdown turns it into a bullet and shatters the paragraph (e.g. a line
  break before `+ zero-bias`). Move the operator to the end of the previous line or
  reword. (Lines inside a ` ```math ` block are exempt — they're LaTeX.)
- **The scan is a command, not a habit:** `uv run python tools/check_markdown.py`
  checks every tracked `.md` against all of the above plus dead relative links, and
  runs as a pre-commit hook. It strips inline code spans first, so a document may
  quote the broken forms deliberately (this one does).

## Testing

- Tests live in `tests/`. Optional-backend tests use `pytest.importorskip`
  (`torch` for `test_pinn_torch.py`, `jax` for `test_pinn_jax.py`) and **skip**
  (not fail) when that extra is absent.
- Coverage is measured over `pinn_sfr_transient` (the DeepXDE and JAX backends are
  `omit`ted as heavy optional paths). The coverage gate runs with the torch extra;
  a separate CI job exercises the JAX backend.
- Keep new PINN tests tiny (small width/depth, a few iterations) so the suite
  stays sub-second on CPU.

## CI / commits

- `.github/workflows/test.yml` runs the suite on push/PR: a torch job (coverage
  gate, Python 3.14), a JAX job (Python 3.13), and a core-only job (Python 3.13,
  verifies the import guards). Actions are pinned to a full commit SHA with the
  version in a trailing comment — keep that pattern when adding or bumping actions.
- `pre-commit` runs ruff on commit and ty + pytest on push. Run
  `uv run pre-commit run --all-files` before proposing a change.
- Commit/PR only when asked. Do not commit `results/`, `uv.lock` churn unrelated
  to a dependency change, or large binaries.
