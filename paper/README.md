# The paper

Two copies of the same paper, and they are not redundant:

- **`paper.md`** — the working draft. Reviewable in a pull request, checked by
  `tools/check_markdown.py` like every other document here, and readable without a
  toolchain. Edit this one while the content is still moving.
- **`paper.typ`** — the typeset version, laid out to `__DEV/paper_template.docx`
  (IOP *Journal of Physics: Conference Series*). This is what produces the PDF.

They are kept in sync by hand, so they can drift. When they disagree, neither wins:
`docs/axial_nn.md` holds the measurement record and both are wrong until reconciled
against it.

## Why Typst and not the template's own format

The template is a **layout guide**, not a submission binary: it specifies A4 with
4.0/2.7/2.5/2.5 cm margins, 11 pt Times, a 17 pt bold flush-left title, authors and
abstract indented 25 mm, numbered sections in bold and subsections in italic, captions
below figures and above tables, and IOP numeric references. `iop-jpcs.typ` encodes that
specification, so the mapping from the guide to the output is readable and checkable
rather than buried in a binary.

Typst is the choice because the paper then lives in the repository the way everything
else does — plain text, diffable, reviewable in a pull request, and buildable by one
command with no toolchain. A `.docx` is none of those things, and the numbers in this
paper come from JSON the repository can regenerate.

**If you submit to IOP**, they accept LaTeX (`iopart.cls`) or Word, not Typst. Say the
word and I will add a LaTeX variant; the content is format-independent and the
translation is mechanical.

## Build

```bash
uv run --with typst python -c "import typst,pathlib; pathlib.Path('paper.pdf').write_bytes(typst.compile('paper/paper.typ'))"
```

or, with the standalone binary:

```bash
typst compile paper/paper.typ
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
