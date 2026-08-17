"""Generate the axial model's figures from the model code (Annex A, N2).

Single source of truth for every committed axial figure: run it and the images
under ``results/figures/`` are rebuilt deterministically from the physics, with no
notebook or manual export in the loop.

Run::

    uv run pinn-sfr axial figures
    uv run python -m pinn_sfr_transient.axial.figures --outdir results/figures

Every figure needs only numpy, scipy and matplotlib.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import (
    AxialTrajectory,
    solve_reference,
)

DEFAULT_OUTDIR = Path("results/figures")
_DPI = 140


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def axial_fields(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Four temperature fields and the void as space-time maps."""
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), constrained_layout=True)
    panels = (
        ("$T_f$ [K]", traj.T_f, "inferno"),
        ("$T_{cl}$ [K]", traj.T_cl, "inferno"),
        ("$T_s$ [K]", traj.T_s, "inferno"),
        ("$T_c$ [K]", traj.T_c, "inferno"),
        (r"$\alpha$", traj.alpha, "Blues"),
    )
    for ax, (label, field, cmap) in zip(axes, panels, strict=True):
        im = ax.pcolormesh(traj.t, traj.zeta, field, cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("$t$ [s]")
        ax.set_ylabel(r"$\zeta$")
        ax.set_title(label)
    T_boil = float(sodium.saturation_temperature(p.p_system)) + p.dT_superheat
    fig.suptitle(f"Axial ULOF, prescribed power — saturation + superheat at {T_boil:.0f} K")
    return _save(fig, outdir / "axial_fields.png")


def axial_front(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Voided length and the boiling front against the saturation crossing."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, traj.voided_length, lw=2)
    ax0.set_xlabel("$t$ [s]")
    ax0.set_ylabel(r"$L_{void}$ [m]")
    ax0.set_title("voided length")
    ax0.grid(alpha=0.3)

    T_boil = float(sodium.saturation_temperature(p.p_system)) + p.dT_superheat
    hot = traj.T_c > T_boil
    front = np.where(hot.any(axis=0), traj.zeta[np.argmax(hot, axis=0)], np.nan)
    ax1.plot(traj.t, front, lw=2, label=r"$T_c = T_{sat} + \Delta T_{sup}$")
    voided = traj.alpha > 0.5
    ax1.plot(
        traj.t,
        np.where(voided.any(axis=0), traj.zeta[np.argmax(voided, axis=0)], np.nan),
        "--",
        lw=2,
        label=r"$\alpha > 0.5$",
    )
    ax1.axhline(p.zeta_sign, color="k", ls=":", label=r"void-worth sign change")
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel(r"$\zeta$")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("boiling front is the saturation level set")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    return _save(fig, outdir / "axial_front.png")


def axial_feedback(p: AxialParams, outdir: Path) -> Path:
    """Closed-loop power and the reactivity split, in units of beta.

    The split is the point: the net is not small because the mechanisms cancel,
    but because the void term never goes positive (deviation register, D49).
    """
    traj = solve_reference(p, n_out=241, feedback=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, traj.power, lw=2)
    ax0.set_xlabel("$t$ [s]")
    ax0.set_ylabel("$P/P_0$")
    ax0.set_title("closed-loop power")
    ax0.grid(alpha=0.3)

    b = p.beta_eff
    ax1.plot(traj.t, traj.rho / b, lw=2, label="net")
    ax1.plot(traj.t, traj.rho_doppler / b, lw=1.6, label="Doppler")
    ax1.plot(traj.t, traj.rho_void / b, lw=1.6, label="coolant / void")
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel(r"$\rho/\beta$")
    ax1.set_title(f"reactivity split — void exercised: {traj.void_worth_is_exercised()}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    return _save(fig, outdir / "axial_feedback.png")


def generate_all(outdir: str | Path = DEFAULT_OUTDIR, n_axial: int = 160) -> list[Path]:
    """Regenerate every axial figure. Returns the paths written."""
    out = Path(outdir)
    p = AxialParams(n_axial=n_axial)
    traj = solve_reference(p, n_out=241)
    return [
        axial_fields(traj, p, out),
        axial_front(traj, p, out),
        axial_feedback(p, out),
    ]


def main(argv: list[str] | None = None) -> int:
    """Regenerate the axial figures. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--n-axial", type=int, default=160)
    args = ap.parse_args(argv)
    for path in generate_all(args.outdir, args.n_axial):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
