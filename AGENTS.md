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

- **Python >= 3.12** (so it installs on Google Colab), managed with **uv**. Every
  module uses `from __future__ import annotations` for portability across
  3.12-3.14 (e.g. `TYPE_CHECKING`-only names in runtime annotations).
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
  algebraically identical — `tests/test_consistency.py` enforces this. If you
  touch the model, update `physics.py` **and** all backends (`pinn_torch`,
  `pinn_jax`, `pinn_deepxde`) together and keep that test green.

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
- After editing any `.md`, scan for these: split tokens (`}$` glued to a letter),
  bare `$^`/`$_`, escaped `\_`/`\^` in `\text`, `\!` (or `\,`/`\;`) in inline `$...$`,
  and odd `$` counts.

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
  gate, Python 3.14), a JAX job (Python 3.12), and a core-only job (Python 3.12,
  verifies the import guards). Actions are pinned to a full commit SHA with the
  version in a trailing comment — keep that pattern when adding or bumping actions.
- `pre-commit` runs ruff on commit and ty + pytest on push. Run
  `uv run pre-commit run --all-files` before proposing a change.
- Commit/PR only when asked. Do not commit `results/`, `uv.lock` churn unrelated
  to a dependency change, or large binaries.
