# SAS4A/SASSYS-1 manual mirror

Plain-text mirror of the SAS4A/SASSYS-1 System Analysis Modules manual, the
source of every physical model in `src/pinn_sfr_transient/axial/`.

| | |
|---|---|
| Source | <https://sas-doc.nse.anl.gov/latest/> |
| Edition | ANL/NSE-SAS/5.8.1 |
| Publisher | Argonne National Laboratory, Nuclear Science and Engineering Division |
| Contents | 16 chapters, `Ch01.txt` … `Ch16.txt` |

Argonne National Laboratory holds the manual and its contents. The files here are
a verbatim text rendering retained for offline citation; consult the site above
as the authority.

## Format

Each file opens with a header giving the chapter title, its source URL and its
scope for this project, then the chapter index page followed by every subordinate
page in order. Within a page:

| Marker | Meaning |
|---|---|
| `$$ … $$` | display equation, LaTeX verbatim |
| `[eq 4.5-3]` | the manual's own equation number, following the equation |
| `[anchor: …]` | section anchor, appending to the page URL as `#…` |
| `[page: PartNN/ChNN/name.html]` | start of a subordinate page |
| `$…$` | inline math |

Equations survive as LaTeX because the site renders with MathJax, so
`docs/axial_physics.md` can cite an equation number and a reader can check it
against the same string here and on the site.

## Regenerating

```bash
uv run python tools/fetch_sas_manual.py docs/sas4a
```

Add `--combined` to also write `ALL.txt`, the concatenation of every chapter.
`ALL.txt` is untracked: it duplicates the per-chapter files exactly and exceeds
the repository's 1 MB file limit.

## Chapters this project uses

| Chapter | Use |
|---|---|
| 3 — Core Thermal-Hydraulics | Eq. 3.3-4 gap conductance and radiation, Eq. 3.3-5 coolant energy, Eq. 3.9-1 pre-boiling momentum |
| 4 — Point Kinetics, Decay Heat, Reactivity Feedback | Eq. 4.3-1 precursors, Eq. 4.5-3 logarithmic Doppler, Eq. 4.5-25 coolant/void worth |
| 5 — Primary/Intermediate Loop | Eq. 5.3-61 prescribed pump head |
| 12 — Coolant Voiding | §12.4 superheat onset, §12.5.1 film heat path, §12.13 sodium properties |

The remaining chapters cover the control system, balance of plant, pre-failure
pin mechanics and post-failure material relocation. This project stops at the
onset and growth of voiding, so they are mirrored for completeness and not
implemented — see `docs/axial_physics.md` deviation D-SCOPE-1.
