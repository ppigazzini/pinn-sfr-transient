"""Render the ladder's tables into Markdown, and verify the docs still match them.

Generated rather than transcribed. In the companion repository, copying ninety numbers
by hand into a table put two of them one unit out in the last digit — a defect no amount
of proofreading finds reliably, and one nobody can spot in review because the wrong digit
looks exactly like the right one.

    uv run python tools/axial_study.py ladder-rows                 # print the tables
    uv run python tools/axial_study.py ladder-rows --check         # fail if docs drifted

The checker is the half that matters, and it runs **document to data**. A document quotes
a slice of the ladder inside a fence:

    <!-- ladder: optimizer=lbfgs n_colloc=5000 -->
    | ... generated rows ...
    <!-- /ladder -->

`--check` re-renders each fenced block from its own selector and compares. Change the
measurement without re-rendering and it fails, naming the file and line.

The other direction -- asserting every rendered row appears somewhere in `docs/` -- was
tried first and is wrong: it requires the documents to reproduce all 196 arms or fail.

Markdown, not LaTeX. The companion rendered LaTeX because it had a paper; this repository
publishes only `docs/*.md`, so a LaTeX renderer would have nothing to check against.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

from pinn_sfr_transient.axial.verification import MIN_RATIO

#: Column layout: metric key, heading, scale, decimals. The temperature fields are
#: quoted in units of 1e-3 because that is the magnitude every published table uses and
#: a column of `0.00258` reads worse than one of `2.58`.
COLUMNS: tuple[tuple[str, str, float, int], ...] = (
    ("T_f", "T_f", 1e3, 2),
    ("T_cl", "T_cl", 1e3, 2),
    ("T_s", "T_s", 1e3, 2),
    ("T_c", "T_c", 1e3, 2),
    ("onset", "onset [s]", 1.0, 4),
    ("Lvoid", "L_void [m]", 1.0, 4),
    ("margin", "margin [K]", 1.0, 2),
)

#: Documents the checker scans. Everything published lives in `docs/`.
DOC_GLOB = "docs/*.md"

#: Markers appended to a ratio that does not clear the calibration threshold. Spelled
#: out rather than left to the reader: the whole purpose of the ratio table is to show
#: which side of four a number sits on, and a column of bare decimals makes that a
#: comparison the reader has to perform 1300 times without slipping.
_MARGINAL = " (<4)"
_MEASURES_RULER = " (<1)"


def _cell(entry: dict[str, float], scale: float, dp: int) -> str:
    """One `mean ± half-range` cell, or `—` where no seed produced a finite value.

    **A single sample gets no ± at all.** The half-range of one number is zero, and
    `112.97 ± 0.00` reads as perfect reproducibility when it means "measured once" --
    on a model whose recorded seed spread has reached 12.5x, and where four published
    conclusions have been overturned by the next seed. The `seeds` column already says
    `1`; printing a spread beside it actively contradicts that.
    """
    mean, half = entry.get("mean"), entry.get("half")
    if mean is None or half is None or math.isnan(mean):
        return "—"
    if entry.get("n", 0) < 2:
        return f"{mean * scale:.{dp}f}"
    return f"{mean * scale:.{dp}f} ± {half * scale:.{dp}f}"


def _id_columns(data: dict[str, Any], selector: dict) -> list[str]:
    """Arm-identifying columns a table must carry to be unambiguous.

    The ladder derives which knobs separate its arms and records them as ``arm_fields``.
    A table that omits them is not readable: over the imported corpus nine knobs vary, so
    ``iters`` alone names up to a dozen different arms, and two rows that happen to agree
    on every metric render as the *same string* — which silently weakens
    :func:`check`, since one document line would satisfy several data rows.

    Anything pinned by the selector is dropped: a table already filtered to
    ``optimizer="lbfgs"`` does not need an ``optimizer`` column repeating it.
    """
    return [f for f in data.get("arm_fields", []) if f not in selector]


def _fmt(v: object) -> str:
    """Render an identifying value compactly, without lying about absence."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def rows(data: dict[str, Any], **selector: object) -> list[str]:
    """Render the ladder's arms as Markdown table rows, one string each.

    ``selector`` filters on any arm key, so one table per collocation count or per
    optimiser is a keyword argument rather than a second function.
    """
    out = []
    ids = _id_columns(data, selector)
    arms = [a for a in data["arms"] if all(a.get(k) == v for k, v in selector.items())]
    # `iters` sorts numerically and the identifying knobs sort as text. Folding it into
    # the string key put 1,000,000 between 100,000 and 200,000.
    for arm in sorted(arms, key=lambda a: (*(str(a.get(f)) for f in ids), a.get("iters", 0))):
        cells = [_fmt(arm.get(f)) for f in ids]
        cells.append(f"{arm['iters']:,}")
        cells.append(str(arm["seeds"]))
        cells += [_cell(arm[key], scale, dp) for key, _, scale, dp in COLUMNS]
        out.append("| " + " | ".join(cells) + " |")
    return out


def table(data: dict[str, Any], **selector: object) -> str:
    """Render a complete Markdown table: header, rule, rows.

    The units line states what the temperature columns are scaled by, because a bare
    `2.58` in a relative-`L2` column is ambiguous by three orders of magnitude and that
    ambiguity is exactly how a transcription error hides.
    """
    ids = _id_columns(data, selector)
    metrics = [f"{h} (x1e-3)" if s == 1e3 else h for _, h, s, _ in COLUMNS]
    heads = [*ids, "iters", "seeds", *metrics]
    head = "| " + " | ".join(heads) + " |"
    rule = "|" + "|".join(["---"] * len(heads)) + "|"
    return "\n".join([head, rule, *rows(data, **selector)])


def ratio_table(data: dict[str, Any], **selector: object) -> str:
    """Each arm's error divided by the reference's own uncertainty at the scoring mesh.

    This is the table that says whether a number is a measurement of the model or of the
    ruler. Four is the threshold calibration practice asks for; below one the reference
    is the less accurate of the two things being compared.

    Cells below either bound are **marked**, not left to the reader. The threshold is
    :data:`~pinn_sfr_transient.axial.verification.MIN_RATIO`, so the constant that states
    the rule is the constant the table applies -- rather than a documented four and a
    hard-coded one somewhere else.
    """
    ruler = data.get("ruler") or {}
    if not ruler:
        return "_No reference uncertainty available; run `axial_study.py verify` first._"
    keys = [k for k, _, _, _ in COLUMNS]
    # Same identifying columns as `table`, and for the same reason: without them a row
    # labelled only by its budget names every arm that shares that budget.
    ids = _id_columns(data, selector)
    head = "| " + " | ".join([*ids, "iters", *keys]) + " |"
    rule = "|" + "|".join(["---"] * (len(ids) + len(keys) + 1)) + "|"
    lines = [head, rule]
    arms = [a for a in data["arms"] if all(a.get(k) == v for k, v in selector.items())]
    for arm in sorted(arms, key=lambda a: (*(str(a.get(f)) for f in ids), a.get("iters", 0))):
        cells = [_fmt(arm.get(f)) for f in ids]
        cells.append(f"{arm['iters']:,}")
        for k in keys:
            unc, mean = ruler.get(k), arm[k].get("mean")
            ok = (
                unc is not None
                and mean is not None
                and not math.isnan(unc)
                and not math.isnan(mean)
                and unc > 0
            )
            if not ok:
                cells.append("—")
                continue
            r = mean / unc
            mark = "" if r >= MIN_RATIO else (_MEASURES_RULER if r < 1.0 else _MARGINAL)
            cells.append(f"{r:.2f}{mark}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


#: Fence marking a generated table inside a document. The opening tag may carry a
#: selector, so one document can hold several slices of the ladder:
#:
#:     <!-- ladder: optimizer=lbfgs n_colloc=5000 -->
#:     | ... generated rows ...
#:     <!-- /ladder -->
FENCE_OPEN = re.compile(r"<!--\s*ladder:?\s*(?P<sel>[^>]*?)\s*-->")
FENCE_CLOSE = "<!-- /ladder -->"


def _parse_selector(text: str) -> dict[str, object]:
    """Read ``key=value`` pairs off a fence, coercing to the types the arms hold."""
    out: dict[str, object] = {}
    for token in text.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if v in {"True", "False"}:
            out[k] = v == "True"
        elif v == "None":
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def check(data: dict[str, Any], root: Path = Path()) -> tuple[list[str], int]:
    """Verify every fenced ladder table in ``docs/`` still matches the data file.

    Returns ``(problems, blocks_checked)``.

    **The direction matters and the first version had it backwards.** Asserting that
    every rendered row appears somewhere in the documents means the documents have to
    reproduce the entire corpus — 196 arms — or the check fails. What a document
    actually does is quote a *slice*, and what can go wrong is that the slice stops
    matching the measurement. So the check runs document-to-data: each fenced block is
    re-rendered from its own selector and compared with what is written.

    That catches both directions of drift within a block — a row whose numbers moved,
    and a row that should have appeared or disappeared as arms changed.

    Blocks checked is returned rather than implied. A document that quietly loses its
    fence would otherwise pass by having nothing to check, and "0 problems" from 0
    blocks reads exactly like "0 problems" from 12.
    """
    problems: list[str] = []
    blocks = 0
    for doc in sorted(Path(root).glob(DOC_GLOB)):
        lines = doc.read_text(encoding="utf-8").splitlines()
        for start, line in enumerate(lines):
            m = FENCE_OPEN.search(line)
            if not m or FENCE_CLOSE in line:
                continue
            try:
                end = next(i for i in range(start + 1, len(lines)) if FENCE_CLOSE in lines[i])
            except StopIteration:
                problems.append(f"{doc}:{start + 1}: ladder fence opened and never closed")
                continue
            blocks += 1
            want = table(data, **_parse_selector(m.group("sel"))).splitlines()
            got = [ln for ln in lines[start + 1 : end] if ln.strip()]
            if got != want:
                problems.append(
                    f"{doc}:{start + 1}: table does not match the data "
                    f"({len(got)} lines written, {len(want)} generated). "
                    f"Regenerate with `axial_study.py ladder-rows`."
                )
    return problems, blocks


def load(path: Path) -> dict[str, Any]:
    """Read a ladder data file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
