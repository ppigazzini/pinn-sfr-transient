"""Render the ladder's tables into Markdown, and verify the docs still match them.

Generated rather than transcribed. In the companion repository, copying ninety numbers
by hand into a table put two of them one unit out in the last digit — a defect no amount
of proofreading finds reliably, and one nobody can spot in review because the wrong digit
looks exactly like the right one.

    uv run python tools/axial_study.py ladder-rows                 # print the rows
    uv run python tools/axial_study.py ladder-rows --check         # fail if docs drifted

The checker is the half that matters. It asserts every rendered row appears verbatim in
`docs/`, so a data file and a document cannot drift apart: change the measurement without
re-rendering and the check fails, naming the row.

Markdown, not LaTeX. The companion rendered LaTeX because it had a paper; this repository
publishes only `docs/*.md`, so a LaTeX renderer would have nothing to check against.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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


def _cell(entry: dict[str, float], scale: float, dp: int) -> str:
    """One `mean ± half-range` cell, or `—` where no seed produced a finite value."""
    mean, half = entry.get("mean"), entry.get("half")
    if mean is None or half is None or math.isnan(mean):
        return "—"
    return f"{mean * scale:.{dp}f} ± {half * scale:.{dp}f}"


def rows(data: dict[str, Any], **selector: object) -> list[str]:
    """Render the ladder's arms as Markdown table rows, one string each.

    ``selector`` filters on any arm key, so one table per collocation count or per
    optimiser is a keyword argument rather than a second function.
    """
    out = []
    arms = [a for a in data["arms"] if all(a.get(k) == v for k, v in selector.items())]
    for arm in sorted(arms, key=lambda a: (str(a.get("optimizer")), a.get("iters", 0))):
        cells = [_cell(arm[key], scale, dp) for key, _, scale, dp in COLUMNS]
        out.append(f"| {arm['iters']:,} | {arm['seeds']} | " + " | ".join(cells) + " |")
    return out


def table(data: dict[str, Any], **selector: object) -> str:
    """Render a complete Markdown table: header, rule, rows.

    The units line states what the temperature columns are scaled by, because a bare
    `2.58` in a relative-`L2` column is ambiguous by three orders of magnitude and that
    ambiguity is exactly how a transcription error hides.
    """
    heads = ["iters", "seeds"] + [f"{h} (x1e-3)" if s == 1e3 else h for _, h, s, _ in COLUMNS]
    head = "| " + " | ".join(heads) + " |"
    rule = "|" + "|".join(["---"] * len(heads)) + "|"
    return "\n".join([head, rule, *rows(data, **selector)])


def ratio_table(data: dict[str, Any], **selector: object) -> str:
    """Each arm's error divided by the reference's own uncertainty at the scoring mesh.

    This is the table that says whether a number is a measurement of the model or of the
    ruler. Four is the threshold calibration practice asks for; below one the reference
    is the less accurate of the two things being compared.
    """
    ruler = data.get("ruler") or {}
    if not ruler:
        return "_No reference uncertainty available; run `axial_study.py verify` first._"
    keys = [k for k, _, _, _ in COLUMNS]
    head = "| " + " | ".join(["iters", *keys]) + " |"
    rule = "|" + "|".join(["---"] * (len(keys) + 1)) + "|"
    lines = [head, rule]
    arms = [a for a in data["arms"] if all(a.get(k) == v for k, v in selector.items())]
    for arm in sorted(arms, key=lambda a: a.get("iters", 0)):
        cells = []
        for k in keys:
            unc, mean = ruler.get(k), arm[k].get("mean")
            ok = (
                unc is not None
                and mean is not None
                and not math.isnan(unc)
                and not math.isnan(mean)
                and unc > 0
            )
            cells.append(f"{mean / unc:.2f}" if ok else "—")
        lines.append(f"| {arm['iters']:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def check(data: dict[str, Any], root: Path = Path()) -> list[str]:
    """Return the rendered rows that do **not** appear in any document.

    An empty list means the documents and the data file agree. A non-empty one means one
    of the two moved without the other, and the rows returned say which.
    """
    body = "\n".join(f.read_text(encoding="utf-8") for f in sorted(Path(root).glob(DOC_GLOB)))
    return [line for line in rows(data) if line not in body]


def load(path: Path) -> dict[str, Any]:
    """Read a ladder data file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
