"""Figures for the ULOF reference transient (with optional PINN overlay)."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt

from pinn_sfr_transient.config import FloatArray, SFRParams
from pinn_sfr_transient.physics import flow_fraction, reactivity, void_fraction
from pinn_sfr_transient.reference import Trajectory

# Headless-safe backend for writing figure files (no display required).
plt.switch_backend("Agg")


class PinnPrediction(TypedDict):
    """Denormalised PINN prediction for overlay."""

    t: FloatArray
    P: FloatArray
    Tf: FloatArray
    Tc: FloatArray


def plot_reference(
    traj: Trajectory,
    p: SFRParams,
    outdir: str | Path,
    pinn: PinnPrediction | None = None,
    filename: str = "ulof_reference.png",
) -> Path:
    """Render the 4-panel transient summary and return the saved figure path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t = traj.t
    void = void_fraction(traj.Tc, p)
    rho_dollars = reactivity(traj.Tf, traj.Tc, p) / p.beta_eff
    g = flow_fraction(t, p)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))

    ax[0, 0].plot(t, traj.P, color="#b22222", lw=2, label="reference")
    if pinn is not None:
        ax[0, 0].plot(pinn["t"], pinn["P"], "--", color="#1f3a93", lw=1.8, label="PINN")
        ax[0, 0].legend()
    ax[0, 0].axhline(1.0, color="grey", ls=":", lw=0.8)
    ax[0, 0].set(title="Normalised power  P(t)", xlabel="t [s]", ylabel="P / P_nom")

    ax[0, 1].plot(t, traj.Tf, color="#d35400", lw=2, label="T_f (fuel)")
    ax[0, 1].plot(t, traj.Tc, color="#2980b9", lw=2, label="T_c (coolant)")
    ax[0, 1].axhline(p.T_onset, color="green", ls="--", lw=1, label="void onset")
    if pinn is not None:
        ax[0, 1].plot(pinn["t"], pinn["Tf"], "--", color="#d35400", lw=1)
        ax[0, 1].plot(pinn["t"], pinn["Tc"], "--", color="#2980b9", lw=1)
    ax[0, 1].set(title="Temperatures", xlabel="t [s]", ylabel="T [K]")
    ax[0, 1].legend()

    ax[1, 0].plot(t, void, color="#8e44ad", lw=2, label="reference")
    if pinn is not None:
        void_pinn = void_fraction(pinn["Tc"], p)
        ax[1, 0].plot(pinn["t"], void_pinn, "--", color="#1f3a93", lw=1.6, label="PINN")
        ax[1, 0].legend()
    ax[1, 0].set(title=r"Sodium void fraction  $\phi$(t)", xlabel="t [s]", ylabel="void [-]")
    ax[1, 0].set_ylim(-0.02, 1.02)

    ax[1, 1].plot(t, rho_dollars, color="#16a085", lw=2, label=r"total $\rho/\beta$")
    ax[1, 1].plot(t, g, color="#7f8c8d", lw=1.5, ls="--", label="flow g(t)")
    if pinn is not None:
        rho_pinn = reactivity(pinn["Tf"], pinn["Tc"], p) / p.beta_eff
        ax[1, 1].plot(
            pinn["t"], rho_pinn, "--", color="#1f3a93", lw=1.6, label=r"PINN $\rho/\beta$"
        )
    ax[1, 1].axhline(0.0, color="grey", ls=":", lw=0.8)
    ax[1, 1].set(
        title=r"Reactivity ($\rho/\beta$, dollars) and flow fraction",
        xlabel="t [s]",
        ylabel=r"$\rho/\beta$,  g [-]",
    )
    ax[1, 1].legend()

    fig.suptitle(
        "Unprotected Loss of Flow (ULOF) — SFR reference transient",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    path = outdir / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
