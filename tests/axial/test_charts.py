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

from __future__ import annotations

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
