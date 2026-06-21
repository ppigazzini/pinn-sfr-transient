"""Generate the figures used in the README and ``docs/`` from the model code.

This is the single source of truth for every committed figure: run it and the
images under ``docs/img/`` are rebuilt deterministically from the physics, with
no notebook or manual export in the loop.

Run::

    uv run pinn-sfr figures                 # -> docs/img/*.png
    uv run python -m pinn_sfr_transient.figures --outdir docs/img

The four physics figures need only numpy/scipy/matplotlib. The optional PINN
overlay is rendered only when ``torch`` is importable (``uv sync --extra torch-cpu``)
and is skipped cleanly otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.physics import reactivity, void_fraction
from pinn_sfr_transient.plotting import plot_reference
from pinn_sfr_transient.reference import Trajectory, solve_reference

# Headless-safe backend for writing figure files (no display required).
plt.switch_backend("Agg")

DEFAULT_OUTDIR = Path("docs/img")

# Void coefficients used by the sweep / phase / safety figures.
_VOID_SWEEP = (4.0e-3, 6.0e-3, 8.0e-3, 1.0e-2)
_SWEEP_COLORS = ("#2980b9", "#16a085", "#d35400", "#b22222")


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    """Write ``fig`` to ``outdir/name`` and return the path."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feedback_competition(traj: Trajectory, p: SFRParams, outdir: Path) -> Path:
    """Decompose total reactivity (dollars, rho/beta) into void/Doppler/coolant."""
    t = traj.t
    doppler = p.alpha_f * (traj.Tf - p.Tf0) / p.beta_eff
    coolant = p.alpha_c * (traj.Tc - p.Tc0) / p.beta_eff
    voidrho = p.alpha_void * void_fraction(traj.Tc, p) / p.beta_eff
    total = doppler + coolant + voidrho + p.rho_ext / p.beta_eff

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, voidrho, label="void (+)", color="#8e44ad", lw=2)
    ax.plot(t, doppler, label="Doppler (-)", color="#d35400", lw=2)
    ax.plot(t, coolant, label="coolant (-)", color="#2980b9", lw=2)
    ax.plot(t, total, label="total", color="black", lw=2.5)
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.set(
        title=r"Reactivity components ($\rho/\beta$, dollars)",
        xlabel="t [s]",
        ylabel=r"$\rho/\beta$",
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "feedback_competition.png")


def plot_void_sweep(p: SFRParams, outdir: Path) -> Path:
    """Power excursion for a range of sodium void coefficients."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for av, color in zip(_VOID_SWEEP, _SWEEP_COLORS, strict=True):
        tr = solve_reference(SFRParams(alpha_void=av, t_end=p.t_end))
        label = rf"$\alpha_{{void}}$={av:.0e} (peak {tr.P.max():.2f}x)"
        ax.plot(tr.t, tr.P, color=color, lw=2, label=label)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set(
        title="Effect of the sodium void coefficient on the power excursion",
        xlabel="t [s]",
        ylabel="P / P_nom",
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "void_sweep.png")


def plot_phase_portrait(p: SFRParams, outdir: Path) -> Path:
    """Power vs net reactivity (dollars), parameterised by time, for void worths."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for av, color in zip(_VOID_SWEEP, _SWEEP_COLORS, strict=True):
        pv = SFRParams(alpha_void=av, t_end=p.t_end)
        tr = solve_reference(pv, n_out=1500)
        rho = np.asarray(reactivity(tr.Tf, tr.Tc, pv)) / pv.beta_eff
        label = rf"$\alpha_{{void}}$={av:.0e} (peak {tr.P.max():.2f}x)"
        ax.plot(rho, tr.P, color=color, lw=1.8, label=label)
        ax.plot(rho[0], tr.P[0], "o", color=color, ms=5)
    ax.axvline(0, color="grey", ls=":", lw=0.8)
    ax.axhline(1, color="grey", ls=":", lw=0.8)
    ax.set(
        title="ULOF phase portrait: power vs net reactivity",
        xlabel=r"net reactivity  $\rho/\beta$  (dollars)",
        ylabel="power  P / P_nom",
    )
    ax.legend()
    fig.tight_layout()
    return _save(fig, outdir, "phase_portrait.png")


def plot_safety_map(p: SFRParams, outdir: Path, n: int = 16) -> Path:
    """2-D peak-power map over (sodium void coefficient, pump coast-down time)."""
    av = np.linspace(2e-3, 1.1e-2, n)
    tp = np.linspace(2.0, 14.0, n)
    peak = np.empty((tp.size, av.size))
    for i, tpi in enumerate(tp):
        for j, avj in enumerate(av):
            tr = solve_reference(SFRParams(alpha_void=float(avj), tau_pump=float(tpi)), n_out=400)
            peak[i, j] = tr.P.max()

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(av * 1e3, tp, peak, shading="gouraud", cmap="inferno")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("peak power  P_max / P_nom")
    contour = ax.contour(
        av * 1e3, tp, peak, levels=[1.2, 1.5, 2.0, 2.5], colors="white", linewidths=1
    )
    ax.clabel(contour, fmt="%.1f x", fontsize=9)
    ax.plot(p.alpha_void * 1e3, p.tau_pump, "o", color="cyan", ms=10, mec="k", label="nominal")
    ax.set(
        title="ULOF peak-power safety map",
        xlabel=r"sodium void coeff  $\alpha_\mathrm{void}$  [10$^{-3}$]",
        ylabel=r"pump coast-down  $\tau_\mathrm{pump}$  [s]",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, outdir, "safety_map.png")


def plot_pinn_overlay(traj: Trajectory, p: SFRParams, outdir: Path) -> Path | None:
    """Train a short PINN and overlay it on the reference, if torch is available.

    Returns the figure path, or ``None`` when ``torch`` is not installed.
    """
    # Lazy, guarded import: torch is an optional extra, so figures.py (and the
    # CLI that imports it) must load without it.
    try:
        import torch  # noqa: F401, PLC0415  # ty: ignore[unresolved-import]

        from pinn_sfr_transient.pinn_torch import TrainConfig, predict, train  # noqa: PLC0415
    except ModuleNotFoundError:
        print("torch not installed; skipping the PINN overlay (`uv sync --extra torch-cpu`).")
        return None

    print("Training the PINN for the overlay figure...")
    cfg = TrainConfig(adam_iters=8000, lbfgs_iters=600, device="cpu")
    model = train(p, cfg)
    pinn = predict(model)
    return plot_reference(traj, p, outdir, pinn=pinn, filename="pinn_overlay.png")


def generate_all(
    outdir: Path = DEFAULT_OUTDIR, *, with_pinn: bool = True, safety_n: int = 16
) -> list[Path]:
    """Regenerate every figure used in the README and docs; return the paths.

    ``safety_n`` is the safety-map grid resolution (per axis); lower it for a quick
    coarse map.
    """
    p = SFRParams()
    traj = solve_reference(p)

    paths = [
        plot_reference(traj, p, outdir),
        plot_feedback_competition(traj, p, outdir),
        plot_void_sweep(p, outdir),
        plot_phase_portrait(p, outdir),
        plot_safety_map(p, outdir, n=safety_n),
    ]
    if with_pinn:
        pinn_path = plot_pinn_overlay(traj, p, outdir)
        if pinn_path is not None:
            paths.append(pinn_path)
    return paths


def main() -> None:
    """CLI entry point for ``pinn-sfr figures`` and ``python -m ...figures``."""
    parser = argparse.ArgumentParser(description="Generate README/docs figures from the model.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="output directory")
    parser.add_argument("--no-pinn", action="store_true", help="skip the optional PINN overlay")
    parser.add_argument("--safety-n", type=int, default=16, help="safety-map grid resolution")
    args = parser.parse_args()

    paths = generate_all(args.outdir, with_pinn=not args.no_pinn, safety_n=args.safety_n)
    print(f"\nWrote {len(paths)} figures to {args.outdir}/:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
