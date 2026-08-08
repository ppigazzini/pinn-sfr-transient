"""End-to-end tests of the CLI reference pipeline (numpy/scipy only).

Exercises ``cli`` -> ``reference`` by running the command into a temporary
output directory. ``reference`` writes only the held-out trajectory ``.npz``;
figures are produced separately by ``pinn-sfr figures`` (see ``test_figures.py``).
"""

import numpy as np
import pytest

from pinn_sfr_transient.cli import build_parser, main


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args(["reference"])
    assert args.t_end == 60.0
    assert args.n_out == 2000
    assert args.func.__name__ == "_run_reference"


def test_reference_writes_trajectory_only(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["pinn-sfr", "reference", "--t-end", "5", "--n-out", "40", "--outdir", str(tmp_path)],
    )
    main()

    npz = tmp_path / "ulof_reference.npz"
    assert npz.exists()
    # `reference` is data-only: no figure is written (figures -> docs/img/).
    assert not (tmp_path / "ulof_reference.png").exists()

    data = np.load(npz)
    assert set(data.files) == {"t", "P", "C", "Tf", "Tc"}
    assert data["P"].shape == (40,)
    assert data["C"].shape == (6, 40)


def test_no_subcommand_defaults_to_reference(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Bare `pinn-sfr` must fall back to the `reference` sub-command (writes ./results).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["pinn-sfr"])
    main()
    assert (tmp_path / "results" / "ulof_reference.npz").exists()
