r"""Check Markdown against the AGENTS.md rendering rules and for dead relative links.

GitHub's inline math renderer is strict, and the failures are silent: the page
still renders, just wrongly. `$^{238}$U` becomes a dangling superscript, `\\!`
inside inline `$...$` prints as a literal `!`, and a range split across two spans
(`$10^4$`-`$10^5$`) renders the second span as source text. AGENTS.md lists the
rules and says to scan for them after every edit. This does the scan.

    uv run python tools/check_markdown.py            # every tracked .md
    uv run python tools/check_markdown.py README.md  # specific files

Exit status is 1 if anything is found, so it works as a pre-commit hook.

Inline code spans are stripped before matching, because a document that
*documents* these rules quotes the broken forms deliberately -- AGENTS.md itself
is full of them.
"""

import re
import subprocess
import sys
from pathlib import Path

# Each rule is (pattern, description). Applied outside fenced blocks and outside
# inline code spans.
RULES: tuple[tuple[str, str], ...] = (
    (r"\$\{?\^", "bare superscript span -- write the whole symbol in one span"),
    (r"\$\{?_", "bare subscript span -- write the whole symbol in one span"),
    (r"\}\$[A-Za-z]", "math span glued to a following letter (split token)"),
    (r"[A-Za-z]\$\{?[A-Za-z^_]", "letter glued to an opening math span"),
    (r"\\text\{[^}]*\\[_^]", r"escaped \_ or \^ inside \text{} -- use n_{\mathrm{in}}"),
    (r"[~\-\u2013]\$", "opening $ glued to a preceding ~, - or en dash"),
    (r"\$[^$\n]*\\[!,;][^$\n]*\$", r"spacing macro (\! \, \;) in inline math"),
)

FENCE = re.compile(r"^\s*(```|~~~)")
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)\s]*)?\)")


def strip_code_spans(line: str) -> str:
    """Blank out inline code spans, by the CommonMark backtick-run rule.

    A run of N backticks opens a span and the next run of *exactly* N closes it.
    A regex cannot express that, and getting it wrong matters here: AGENTS.md
    writes an inline ``` ```math ``` span, whose opening run is one backtick and
    whose first inner run is three. Matching greedily splits it in the middle and
    exposes an unrelated ``$`` further along the line as unbalanced.
    """
    out: list[str] = []
    i = 0
    while i < len(line):
        if line[i] != "`":
            out.append(line[i])
            i += 1
            continue
        n = 0
        while i + n < len(line) and line[i + n] == "`":
            n += 1
        j = i + n
        while j < len(line):
            if line[j] != "`":
                j += 1
                continue
            m = 0
            while j + m < len(line) and line[j + m] == "`":
                m += 1
            if m == n:
                break
            j += m
        if j >= len(line):  # unclosed run: leave the rest as prose
            out.append(line[i:])
            break
        out.append("``")
        i = j + n
    return "".join(out)


def tracked_markdown() -> list[Path]:
    """Return every Markdown file git knows about."""
    out = subprocess.run(
        ["git", "ls-files", "*.md"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in out.stdout.split() if line]


def check(path: Path) -> list[str]:
    """Return one message per problem found in ``path``."""
    problems: list[str] = []
    in_fence = False
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(raw):
            in_fence = not in_fence
            continue

        problems.extend(
            f"{path}:{n}: dead link -> {target}"
            for target in LINK.findall(raw)
            if not target.startswith(("http://", "https://", "mailto:"))
            and not (path.parent / target).exists()
        )

        if in_fence:
            continue
        line = strip_code_spans(raw)
        if line.count("$") % 2:
            problems.append(f"{path}:{n}: odd number of $ -- the whole line stops rendering")
        for pattern, why in RULES:
            if re.search(pattern, line):
                problems.append(f"{path}:{n}: {why}\n    {raw.strip()[:100]}")
    return problems


def main(argv: list[str]) -> int:
    """Scan the named files, or every tracked .md."""
    paths = [Path(a) for a in argv] or tracked_markdown()
    problems = [msg for p in paths if p.exists() for msg in check(p)]
    for msg in problems:
        print(msg)
    print(f"\n{len(problems)} problem(s) in {len(paths)} file(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
