"""Tests for the figure-generation module and its CLI wiring (numpy/scipy only).

A tiny ``safety_n`` keeps the 2-D sweep fast; the PINN overlay is skipped
(``with_pinn=False``) so these run without the optional torch extra.
"""

import pytest

from pinn_sfr_transient.cli import build_parser, main
from pinn_sfr_transient.figures import generate_all


def test_figures_subcommand_wiring() -> None:
    args = build_parser().parse_args(["figures"])
    assert args.func.__name__ == "_run_figures"
    assert args.no_pinn is False
    assert args.safety_n == 16


def test_generate_all_writes_every_figure(tmp_path) -> None:
    paths = generate_all(tmp_path, with_pinn=False, safety_n=3)

    expected = {
        "ulof_reference.png",
        "feedback_competition.png",
        "void_sweep.png",
        "phase_portrait.png",
        "safety_map.png",
    }
    assert {p.name for p in paths} == expected
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_figures_cli_command(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["pinn-sfr", "figures", "--no-pinn", "--safety-n", "3", "--outdir", str(tmp_path)],
    )
    main()
    assert (tmp_path / "safety_map.png").exists()
