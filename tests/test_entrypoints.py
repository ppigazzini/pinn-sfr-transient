"""Exercise the entry points — CLI, figures, module ``main``s.

These were the largest uncovered surface in the project: `axial/figures.py` sat at
**0%**, and the CLI's axial sub-commands at 67%. That is the same class of gap as
`adam_iters = 0`, which crashed the moment it was first run after months of
existing — code that is shipped, documented in the README, and never executed by
anything.

They are cheap to cover because they are entry points: call them, on the smallest
inputs that are still meaningful, and assert they produce what they claim. What
they cannot tell you is whether the figures are *correct* — only that the code
paths run and write files. That distinction is worth keeping in view: this file
raises coverage, it does not raise confidence in the plots.

Everything runs at a coarse mesh and a short horizon so the suite stays quick.
"""

import runpy
import sys

import matplotlib as mpl
import pytest

mpl.use("Agg")  # no display in CI; must precede any pyplot import


# --- axial figures ---------------------------------------------------------
@pytest.fixture(scope="module")
def axial_figs(tmp_path_factory):
    """Generate the axial figures once; three tests need them and each solve is ~30 s.

    A module-scoped fixture rather than three calls: the tests below assert
    different properties of the *same* artefacts, so regenerating per test triples
    the suite's cost to re-derive identical files.
    """
    from pinn_sfr_transient.axial.figures import generate_all

    out = tmp_path_factory.mktemp("axialfig")
    return out, generate_all(out, n_axial=20)


def test_axial_figures_generate_all_writes_every_figure(axial_figs):
    """`generate_all` must write one non-empty PNG per figure and return the paths."""
    _, paths = axial_figs
    assert len(paths) >= 2
    for path in paths:
        assert path.exists(), path
        assert path.stat().st_size > 0, path
        assert path.suffix == ".png"


def test_axial_figures_cover_the_named_plots(axial_figs):
    """The front and the field plots are separate functions and both must run."""
    _, paths = axial_figs
    names = {p.stem for p in paths}
    assert any("front" in n for n in names), names
    assert len(names) == len(paths), "each figure needs its own filename"


def test_axial_figures_main_runs(tmp_path, monkeypatch):
    """The module's own ``main`` — the documented `python -m ...` path."""
    from pinn_sfr_transient.axial import figures

    monkeypatch.setattr(sys, "argv", ["figures", "--outdir", str(tmp_path), "--n-axial", "16"])
    figures.main()
    assert list(tmp_path.glob("*.png"))


# --- 0D figures ------------------------------------------------------------
def test_zero_d_figures_generate(tmp_path):
    """The 0D figure set, which the README embeds.

    ``safety_n=2`` and ``with_pinn=False`` on purpose. The defaults are a 16x16
    grid of stiff solves plus a PINN train — around ten minutes, which is not a
    test. The grid resolution changes the picture, not the code path.
    """
    from pinn_sfr_transient import figures

    paths = figures.generate_all(tmp_path, with_pinn=False, safety_n=2)
    assert paths
    for path in paths:
        assert path.exists()


def test_zero_d_figures_pinn_overlay_branch(tmp_path, monkeypatch):
    """The ``with_pinn`` branch, with the training stubbed out.

    Stubbing is the honest choice here: the branch under test is *"was an overlay
    produced, and was it appended"*, and training a real PINN to answer that would
    add minutes to the suite while testing something `test_pinn_torch.py` already
    covers. Both outcomes are exercised, since `plot_pinn_overlay` returns `None`
    when the optional backend is absent and that path is equally live.
    """
    from pinn_sfr_transient import figures

    made = tmp_path / "overlay.png"
    made.write_bytes(b"x")
    monkeypatch.setattr(figures, "plot_pinn_overlay", lambda *_a, **_k: made)
    assert made in figures.generate_all(tmp_path, with_pinn=True, safety_n=2)

    monkeypatch.setattr(figures, "plot_pinn_overlay", lambda *_a, **_k: None)
    paths = figures.generate_all(tmp_path, with_pinn=True, safety_n=2)
    assert made not in paths


def test_zero_d_figures_main(tmp_path, monkeypatch, capsys):
    """`python -m pinn_sfr_transient.figures`, with the slow options turned off."""
    from pinn_sfr_transient import figures

    monkeypatch.setattr(
        sys,
        "argv",
        ["figures", "--outdir", str(tmp_path), "--no-pinn", "--safety-n", "2"],
    )
    figures.main()
    assert "figures to" in capsys.readouterr().out


# --- CLI -------------------------------------------------------------------
def test_cli_parser_builds_every_documented_subcommand():
    """Every sub-command the README lists must parse."""
    from pinn_sfr_transient.cli import build_parser

    parser = build_parser()
    for argv in (
        ["reference"],
        ["figures"],
        ["axial", "reference"],
        ["axial", "reference", "--feedback"],
        ["axial", "figures"],
    ):
        args = parser.parse_args(argv)
        assert hasattr(args, "func") or args.command == "axial"


def test_cli_axial_reference_runs_and_reports(tmp_path, capsys):
    """The axial reference sub-command, on a coarse mesh and a short horizon."""
    from pinn_sfr_transient.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "axial",
            "reference",
            "--n-axial",
            "20",
            "--t-end",
            "2.0",
            "--n-out",
            "11",
            "--outdir",
            str(tmp_path),
        ]
    )
    args.func(args)
    out = capsys.readouterr().out
    assert "Axial ULOF" in out
    assert "n_axial" in out


def test_cli_axial_figures_runs(tmp_path, capsys):
    """The axial figures sub-command."""
    from pinn_sfr_transient.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["axial", "figures", "--outdir", str(tmp_path), "--n-axial", "20"])
    args.func(args)
    assert "axial figures" in capsys.readouterr().out


def test_cli_reference_runs(tmp_path, capsys):
    """The 0D reference sub-command."""
    from pinn_sfr_transient.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["reference", "--t-end", "5.0", "--n-out", "50", "--outdir", str(tmp_path)]
    )
    args.func(args)
    assert capsys.readouterr().out


def test_cli_with_no_subcommand_defaults_to_reference(monkeypatch, tmp_path, capsys):
    """No sub-command must run the reference rather than exiting with a usage error.

    That default is a documented behaviour, and the branch implementing it had
    never been executed.
    """
    from pinn_sfr_transient import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pinn-sfr"])
    cli.main()
    assert capsys.readouterr().out


from contextlib import contextmanager


@contextmanager
def _tolerate_clean_exit():
    """Allow ``SystemExit(0)`` from an argparse ``main``, fail on anything else."""
    try:
        yield
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


# --- module __main__ paths -------------------------------------------------
def test_reference_module_main(tmp_path, monkeypatch):
    """`python -m pinn_sfr_transient.reference` — the standalone solve."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reference"])
    runpy.run_module("pinn_sfr_transient.reference", run_name="__main__")


@pytest.mark.parametrize("module", ["pinn_sfr_transient.axial.reference"])
def test_axial_reference_module_main(module, tmp_path, monkeypatch):
    """The axial reference's own ``__main__``, which the docs invoke directly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [module, "--n-axial", "20", "--t-end", "2.0"])
    # argparse may exit 0 on completion; anything else is a failure.
    with pytest.raises((SystemExit, TypeError)) if False else _tolerate_clean_exit():
        runpy.run_module(module, run_name="__main__")
