# The paper

**It is a nuclear engineering paper, not a machine-learning one.** `docs/` and `__DEV/`
collect everything — every ML measurement, every optimiser comparison, every retraction.
This directory does not. The paper describes the default ML system in about a page and
spends its length on the physics: the governing equations as used, where each closure
comes from in the SAS4A manual, the registered deviations and their justification, the
reference solution and the resolution it actually has, what the surrogate reproduces, and
what it does not yet solve. A reader should be able to judge the reactor physics without
knowing what L-BFGS is.


Three copies of the same paper, and they are not redundant:

- **`paper.md`** — the working draft, and the one that carries the current content.
  Reviewable in a pull request, checked by `tools/check_markdown.py` like every other
  document here, and readable without a toolchain. Edit this one while the content is
  still moving.
- **`paper.tex`** — the **submission** format. IOP accepts LaTeX or Word, not Typst, so
  this is what can actually be sent.
- **`paper.typ`** — the local typeset version, one command and no toolchain.

All three are laid out to `__DEV/paper_template.docx` (IOP *Journal of Physics:
Conference Series*), and `paper.tex` and `paper.typ` encode the *same* constants from
it — margins from the guide's Table 1, section styles from its Table 2. Change one and
change the other.

They are kept in sync by hand, so they can drift. When they disagree, neither wins:
`docs/axial_nn.md` holds the measurement record and all three are wrong until reconciled
against it.

> **`paper.typ` has drifted and is currently stale.** It still carries the earlier
> machine-learning framing — a different title, a different abstract, and an optimiser
> bake-off as the headline. `paper.md` was rewritten as a nuclear-engineering paper and
> `paper.tex` was written from `paper.md`, so those two agree and the Typst copy does
> not. Reconcile before building anything from it.

## Why neither is the template's own format

The template is a **layout guide**, not a submission binary: it specifies A4 with
4.0/2.7/2.5/2.5 cm margins, 11 pt Times, a 17 pt bold flush-left title, authors and
abstract indented 25 mm, numbered sections in bold and subsections in italic, captions
below figures and above tables, and IOP numeric references. Both `iop-jpcs.typ` and
`paper.tex`'s preamble encode that specification, so the mapping from the guide to the
output is readable and checkable rather than buried in a binary — and the two encodings
can be diffed against each other.

Plain text is the choice because the paper then lives in the repository the way
everything else does — diffable, reviewable in a pull request, and buildable by one
command. A `.docx` is none of those things, and the numbers in this paper come from JSON
the repository can regenerate.

**`paper.tex` deliberately does not use `iopart.cls`.** The class imposes its own
geometry, which would silently override the guide it is supposed to be following, and it
is not installed everywhere. The body transfers unchanged if IOP ask for the class
specifically — only the preamble is discarded.

## Build

```bash
latexmk -pdf paper/paper.tex          # submission PDF; or pdflatex twice
typst compile paper/paper.typ         # local PDF, no toolchain
```

The Typst copy also builds without the standalone binary:

```bash
uv run --with typst python -c "import typst,pathlib; pathlib.Path('paper.pdf').write_bytes(typst.compile('paper/paper.typ'))"
```

## Where the numbers come from

Every figure in the paper is reproducible by a committed command. The two central
tables are `tools/axial_study.py grid` and `tools/axial_study.py qnladder`; the
reference-uncertainty numbers are `tools/m4_bar.py`; the raw rows are in
`__DEV/studies/`. `docs/axial_nn.md` carries the full measurement record, including
the negative results and the retractions, which the paper only summarises.

## Status

**Draft.** Two gaps, and the first was found by auditing the raw rows after a machine
crash rather than by noticing it in the text:

- **The funded optimiser bake-off has never been run.** Every row in
  `__DEV/studies/optbakeoff.json` and `ssbroyden.json` carries
  `adam_iters = 3000, lbfgs_iters = 300` — the starved diagonal, which §7.5.11 shows is
  the regime where the quasi-Newton stage does not matter. An earlier revision of §6
  quoted "0.0296 against 0.0401, 35% worse" at a *funded* stage; those numbers appear in
  no study file, and the claim has been withdrawn and replaced by a statement of the gap.
  `tools/axial_study.py bakeoff` is the command that would close it.
- **The PyTorch confirmation of the 30 000-iteration ladder is absent.** `qnladder_s{0,1,2}`
  are complete and reproduce Table 2 exactly, but all eighteen rows are JAX.

Nothing here should be submitted before those land.
