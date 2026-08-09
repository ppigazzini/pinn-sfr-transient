# The paper

`paper.typ` — the results paper, laid out to `__DEV/paper_template.docx`
(IOP *Journal of Physics: Conference Series*).

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

**Draft.** The optimiser bake-off at the funded budget and the PyTorch confirmation of
the 30 000-iteration ladder were still running when this was written; §5.1 states the
JAX result and says so. Nothing in the paper should be submitted before those land.
