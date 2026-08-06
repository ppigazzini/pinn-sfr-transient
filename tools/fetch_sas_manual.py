"""Mirror the SAS4A/SASSYS-1 manual to plain text with its LaTeX preserved.

Source: https://sas-doc.nse.anl.gov/latest/ — ANL/NSE-SAS/5.8.1.

The manual is a Sphinx site rendered with MathJax, so every equation is LaTeX in
the HTML source. Display equations are emitted as ``$$...$$`` and tagged with the
manual's own equation number as ``[eq 4.5-3]``; section anchors are emitted as
``[anchor: ...]``. Both survive extraction verbatim and are quotable.

Run::

    uv run python tools/fetch_sas_manual.py docs/sas4a
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BASE = "https://sas-doc.nse.anl.gov/latest"
EDITION = "ANL/NSE-SAS/5.8.1"
TIMEOUT_S = 60
MAX_WORKERS = 8
RULE = "=" * 78
SUBRULE = "-" * 78

type Html = str

SCOPE: dict[str, str] = {
    "Ch01": "context: code overview",
    "Ch02": "input-deck reference; source of realistic parameter values",
    "Ch03": "core: single-phase fuel/cladding/coolant/structure model, axial and radial mesh, "
    "pre-boiling transient (3.3), steady state (3.4), boiling-regime temperatures (3.5), "
    "single-phase hydraulics (3.9), feedback coupling (3.12.1), coolant properties (3.19.4)",
    "Ch04": "core: point kinetics (4.2, 4.6.1), delayed neutrons (4.3), ANS decay heat (4.4), "
    "eight reactivity feedback mechanisms (4.5)",
    "Ch05": "core: PRIMAR pumps, coast-down, flow boundary condition, steady-state init (5.9)",
    "Ch06": "out of scope: control system; ULOF is unprotected",
    "Ch07": "out of scope: balance of plant",
    "Ch08": "out of scope: DEFORM-4 pre-failure pin behaviour",
    "Ch09": "out of scope: MFUEL metallic fuel",
    "Ch10": "out of scope: SSCOMP pre-transient characterisation",
    "Ch11": "out of scope: FPIN2",
    "Ch12": "core: coolant voiding, boiling model and sodium properties (12.13)",
    "Ch13": "out of scope: CLAP cladding motion, post-failure",
    "Ch14": "out of scope: PLUTO2 fuel motion, post-failure",
    "Ch15": "out of scope: PINACLE molten fuel relocation, post-failure",
    "Ch16": "out of scope: LEVITATE voided fuel motion, post-failure",
}


@dataclass(frozen=True, slots=True)
class Chapter:
    """One chapter of the manual, located by its part and chapter directory."""

    part: str
    number: str

    @property
    def index_url(self) -> str:
        """URL of the chapter's index page."""
        return f"{BASE}/{self.part}/{self.number}/{self.number}.html"

    def page_url(self, page: str) -> str:
        """URL of ``page`` within this chapter."""
        return f"{BASE}/{self.part}/{self.number}/{page}"


class FetchError(RuntimeError):
    """A page could not be retrieved."""


def fetch(url: str) -> Html:
    """Return the HTML at ``url``.

    Use curl. Cloudflare fronts anl.gov and returns 403 both to ``urllib`` and to
    any request declaring a browser User-Agent; curl's default User-Agent passes.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, URL built from BASE
        ["curl", "-sS", "--fail", "--max-time", str(TIMEOUT_S), url],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = f"{url}: curl exit {result.returncode}: {result.stderr.strip()}"
        raise FetchError(msg)
    return result.stdout


def page_links(index_html: Html) -> list[str]:
    """Return the distinct ``.html`` pages a chapter index references, in order."""
    seen: dict[str, None] = {}
    for href in re.findall(r'class="reference internal" href="([^"]+)"', index_html):
        page = href.split("#")[0]
        if page.endswith(".html") and not page.startswith("..") and "/" not in page:
            seen.setdefault(page, None)
    return list(seen)


def _display_math(match: re.Match[str]) -> str:
    """Render one MathJax display block, keeping its equation number."""
    inner = match.group(1)
    tex = re.search(r"\\\[(.*?)\\\]", inner, re.DOTALL)
    if not tex:
        return ""
    number = re.search(r'<span class="eqno">\(([^)]+)\)', inner)
    tag = f"  [eq {number.group(1)}]" if number else ""
    return f"\n\n$${tex.group(1).strip()}$${tag}\n\n"


def to_text(page_html: Html) -> str:
    """Convert a Sphinx page to plain text, keeping LaTeX and section anchors."""
    body = re.search(r'<div itemprop="articleBody">(.*?)</div>\s*</div>', page_html, re.DOTALL)
    text = body.group(1) if body else page_html

    text = re.sub(r"<(script|style).*?</\1>", "", text, flags=re.DOTALL)
    text = re.sub(
        r'<div class="math notranslate nohighlight"[^>]*>(.*?)</div>',
        _display_math,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'<span class="math notranslate nohighlight">\\\((.*?)\\\)</span>',
        lambda m: f" ${m.group(1).strip()}$ ",
        text,
        flags=re.DOTALL,
    )
    for level in (1, 2, 3, 4):
        text = re.sub(
            rf'<h{level}>(.*?)<a class="headerlink"[^>]*>.*?</a></h{level}>',
            lambda m, lvl=level: f"\n\n{'#' * lvl} {m.group(1).strip()}\n",
            text,
            flags=re.DOTALL,
        )
    text = re.sub(r'<section id="([^"]+)">', r"\n[anchor: \1]\n", text)
    text = re.sub(r"</?(p|li|tr|div|table|dt|dd)[^>]*>", "\n", text)
    text = re.sub(r"</?(td|th)[^>]*>", " | ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def strip_tags(fragment: Html) -> str:
    """Return ``fragment`` with tags removed and entities resolved."""
    return html.unescape(re.sub("<[^>]+>", "", fragment)).strip()


def chapter_list() -> list[Chapter]:
    """Return every chapter listed in the manual's own index."""
    index = fetch(f"{BASE}/index.html")
    rows = re.findall(
        r'class="toctree-l2"><a class="reference internal" href="(Part\d+)/(Ch\d+)/Ch\d+\.html"',
        index,
    )
    return [Chapter(part, number) for part, number in rows]


def render_chapter(chapter: Chapter) -> str:
    """Fetch a chapter and return it as one text document."""
    index = fetch(chapter.index_url)
    heading = re.search(r"<h1>(.*?)<a class=\"headerlink\"", index, re.DOTALL)
    title = strip_tags(heading.group(1)) if heading else chapter.number
    pages = page_links(index)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        bodies = list(pool.map(lambda p: fetch(chapter.page_url(p)), pages))

    chunks = [
        RULE,
        f"CHAPTER {title}",
        f"source: {chapter.index_url}   ({EDITION})",
        f"scope: {SCOPE.get(chapter.number, 'unclassified')}",
        RULE,
        to_text(index),
    ]
    for page, body in zip(pages, bodies, strict=True):
        location = f"{chapter.part}/{chapter.number}/{page}"
        chunks += [f"\n\n{SUBRULE}\n[page: {location}]\n{SUBRULE}", to_text(body)]
    return "\n".join(chunks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="directory to write ChNN.txt into")
    parser.add_argument(
        "--combined",
        action="store_true",
        help="also write ALL.txt, the concatenation of every chapter",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Mirror the manual into ``outdir``. Return a process exit code."""
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    documents: list[str] = []
    failures = 0
    for chapter in chapter_list():
        try:
            text = render_chapter(chapter)
        except FetchError as exc:
            print(f"{chapter.number}: {exc}", file=sys.stderr)
            failures += 1
            continue
        (args.outdir / f"{chapter.number}.txt").write_text(text, encoding="utf-8")
        equations = text.count("$$") // 2
        print(f"{chapter.number}: {len(text) // 1024} KB, ~{equations} display equations")
        documents.append(text)

    if args.combined:
        (args.outdir / "ALL.txt").write_text("\n\n".join(documents), encoding="utf-8")
        print(f"wrote {args.outdir / 'ALL.txt'}")

    print(f"{len(documents)} chapters written to {args.outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
