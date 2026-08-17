"""The transient charts: selectable, and cheap when you select the cheap ones.

Rendering every chart is what CI does on a push under ``paper/`` — see
``.github/workflows/paper.yml`` — so this suite does not repeat it. What it pins is the
behaviour a caller relies on and a plotting change could silently break: that ``--only``
names are validated rather than ignored, that a chart needing no trajectory does not
solve one, and that asking for an open-loop chart does not pay for the closed loop.

The last is a real claim in ``generate_all``'s docstring, not a hypothetical: a
closed-loop solve is the expensive one, and ``--only front_height`` would quietly cost it
if the caching were wrong.
"""

import sys

import pytest

from pinn_sfr_transient.axial import charts


def test_every_registered_chart_is_callable_and_declares_its_loop():
    """The registry is what ``--only`` resolves against, so it has to stay well formed."""
    assert charts.CHARTS, "no charts registered"
    for name, (needs_loop, draw) in charts.CHARTS.items():
        assert isinstance(name, str)
        assert name
        assert isinstance(needs_loop, bool)
        assert callable(draw)
    # Power and reactivity are meaningless open-loop: with `feedback=False` the power is
    # prescribed, so plotting it would be a picture of the solver's own input.
    assert charts.CHARTS["power"][0] is True
    assert charts.CHARTS["reactivity"][0] is True


def test_an_unknown_chart_name_is_an_error_not_a_silent_skip(tmp_path):
    """A typo in ``--only`` must fail loudly; producing nothing looks like success."""
    with pytest.raises(ValueError, match="unknown chart"):
        charts.generate_all(tmp_path, only=("front_heigth",))  # codespell:ignore


def test_the_void_worth_chart_needs_no_trajectory(tmp_path, monkeypatch):
    """It is a property of the parameters alone, so it must not trigger a solve.

    Asserted by making a solve fail: if the chart reaches the solver, the test fails
    rather than merely running slower, which is the only way a performance claim stays
    true after someone edits the driver.
    """

    def explode(*_args, **_kwargs):
        msg = "solve_reference must not be called for a parameter-only chart"
        raise AssertionError(msg)

    monkeypatch.setattr(charts, "solve_reference", explode)
    (path,) = charts.generate_all(tmp_path, only=("void_worth",))
    assert path.exists()
    assert path.stat().st_size > 0


def test_an_open_loop_chart_does_not_pay_for_the_closed_loop(tmp_path, monkeypatch):
    """``--only front_height`` must solve once, open-loop, and never with feedback."""
    seen: list[bool] = []
    real = charts.solve_reference

    def spy(p, **kw):
        seen.append(bool(kw.get("feedback", False)))
        return real(p, **kw)

    monkeypatch.setattr(charts, "solve_reference", spy)
    charts.generate_all(tmp_path, n_axial=12, n_out=9, only=("front_height",))
    assert seen == [False], f"expected one open-loop solve, got feedback flags {seen}"


# --- the artefacts themselves -----------------------------------------------------
# Rendered once for the whole module: every chart below asserts a property of the SAME
# set of files, and each reference solve is tens of seconds. This mirrors the fixture
# `tests/test_entrypoints.py` uses for `axial/figures.py`, which exists for the same
# reason -- a plotting module is mostly uncovered until something actually draws it,
# and 274 statements of it took the repository's coverage gate below its floor.
@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    """Every chart, at a mesh small enough to be a test rather than a study."""
    out = tmp_path_factory.mktemp("charts")
    return out, charts.generate_all(out, n_axial=20, n_out=41)


def test_every_chart_writes_a_non_empty_png(drawn):
    """One file per registered chart, each with its own name."""
    _, paths = drawn
    assert len(paths) == len(charts.CHARTS)
    for path in paths:
        assert path.exists(), path
        assert path.suffix == ".png", path
        assert path.stat().st_size > 0, path
    assert len({p.stem for p in paths}) == len(paths), "two charts share a filename"


def test_the_requested_charts_are_all_present(drawn):
    """The set the paper asks for, by name, so a rename cannot silently drop one."""
    _, paths = drawn
    names = {p.stem for p in paths}
    for expected in (
        "temperature_history",
        "final_temperature_profile",
        "vapor_fraction",
        "temperature_map",
        "heat_flux",
        "power",
        "reactivity",
        "front_height",
    ):
        assert expected in names, f"{expected} missing from {sorted(names)}"


def test_explicit_snapshot_times_are_honoured(tmp_path):
    """``--alpha-times`` overrides the per-cent default, including out-of-range asks.

    An out-of-range time must not raise: the nearest column is drawn and labelled with
    the time it was actually drawn at, which is the honest behaviour when someone asks
    for 60 s of a transient that stops at 16.
    """
    (path,) = charts.generate_all(
        tmp_path, n_axial=20, n_out=41, only=("vapor_fraction",), alpha_times=(0.0, 999.0)
    )
    assert path.stat().st_size > 0


def test_module_main_runs(tmp_path, monkeypatch):
    """The documented ``python -m pinn_sfr_transient.axial.charts`` path."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "charts",
            "--outdir",
            str(tmp_path),
            "--n-axial",
            "16",
            "--n-out",
            "21",
            "--only",
            "void_worth,front_height",
        ],
    )
    assert charts.main() == 0
    assert len(list(tmp_path.glob("*.png"))) == 2
