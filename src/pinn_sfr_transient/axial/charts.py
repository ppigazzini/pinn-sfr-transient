"""Result charts for the transient: what the channel does, and where boiling starts.

Distinct from :mod:`pinn_sfr_transient.axial.figures`, and the split is deliberate.
``figures.py`` draws the three *explanatory* figures embedded in ``docs/`` — the field
maps, the front, the feedback split — and regenerates them all at once. This module
draws the **result** charts a reader of the paper needs, each one selectable on its own
so a single panel can be rebuilt without paying for the rest.

Every chart is drawn from the reference solution, which is the object the paper's claims
are measured against. Charts that need the reactivity loop closed say so and solve for it
separately: with ``feedback=False`` the power is prescribed and a power chart would be a
picture of its own input.

Run::

    uv run python -m pinn_sfr_transient.axial.charts --outdir results/charts
    uv run python -m pinn_sfr_transient.axial.charts --only front,power

Needs only numpy, scipy and matplotlib.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.physics import clad_coolant_flux, node_geometry
from pinn_sfr_transient.axial.reference import AxialTrajectory, solve_reference
from pinn_sfr_transient.axial.scoring import onset_by_tangency

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_OUTDIR = Path("results/charts")
_DPI = 140

# The four material fields, in the order they are stacked everywhere else.
_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("T_f", "$T_f$ (fuel)", "tab:red"),
    ("T_cl", "$T_{cl}$ (cladding)", "tab:orange"),
    ("T_s", "$T_s$ (film)", "tab:green"),
    ("T_c", "$T_c$ (coolant)", "tab:blue"),
)

# Snapshot times for the void chart, as fractions of the solved horizon.
#
# Read as PERCENTAGES, not seconds. The nominal transient runs to `p.t_end = 60` s but
# the reference stops at the top of the section 12.13 property fits, around 16.7 s, so
# literal times of 20, 40 and 60 s lie outside the model's validity window and do not
# exist to be plotted.
#
# The fractions are NOT 0/20/40/60 either, and the reason is worth stating: onset falls
# at about 67% of the horizon, so every one of those four frames is pre-boiling and the
# panel draws four flat zeros. These span the horizon instead, putting two frames after
# onset where the void actually has structure. `--alpha-times` overrides with seconds.
_ALPHA_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write ``fig`` to ``path`` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _T_boil(p: AxialParams) -> float:
    """Saturation plus the required superheat [K] — the boiling criterion."""
    return float(sodium.saturation_temperature(p.p_system)) + p.dT_superheat


def _onset(traj: AxialTrajectory, p: AxialParams) -> tuple[float, float]:
    """``(time, height)`` of boiling onset, by tangency rather than grid crossing.

    Reading onset off the output grid quantises it to the sampling interval, which was
    a quarter of a second of every onset error this model ever reported.
    """
    return onset_by_tangency(traj.T_c, traj.zeta, traj.t, _T_boil(p))


def _mark_onset(ax: plt.Axes, t_on: float, *, label: bool = True) -> None:
    """Drop a vertical marker at boiling onset on a time axis."""
    if not np.isfinite(t_on):
        return
    ax.axvline(
        t_on,
        color="k",
        ls="--",
        lw=1.0,
        label=f"onset {t_on:.3f} s" if label else None,
    )


# --- the requested charts ---------------------------------------------------------
def temperature_history(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Channel-average and channel-maximum temperature against time.

    Both are shown because they answer different questions: the average is the stored
    energy, the maximum is the safety margin, and in this transient they separate — the
    peak runs away up the channel while the average is still rising smoothly.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    t_on, _ = _onset(traj, p)
    for name, label, colour in _FIELDS:
        field = getattr(traj, name)
        ax0.plot(traj.t, field.mean(axis=0), lw=1.8, color=colour, label=label)
        ax1.plot(traj.t, field.max(axis=0), lw=1.8, color=colour, label=label)
    ax1.axhline(_T_boil(p), color="k", lw=0.9, ls=":", label=r"$T_{sat}+\Delta T_{sup}$")
    for ax, title in ((ax0, "channel average"), (ax1, "channel maximum")):
        _mark_onset(ax, t_on, label=ax is ax1)
        ax.set_xlabel("$t$ [s]")
        ax.set_ylabel("$T$ [K]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")
    ax0.legend(fontsize=8, loc="upper left")
    return _save(fig, outdir / "temperature_history.png")


def final_temperature_profile(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Axial temperature profile at the last solved instant.

    The last instant is where the transient is hardest and where the four fields have
    separated most, so this is the profile that shows the boiling front as a feature
    rather than as a number.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for name, label, colour in _FIELDS:
        ax.plot(traj.zeta, getattr(traj, name)[:, -1], lw=1.8, color=colour, label=label)
    ax.axhline(_T_boil(p), color="k", lw=0.9, ls=":", label=r"$T_{sat}+\Delta T_{sup}$")
    ax.axvline(p.zeta_sign, color="grey", lw=0.9, ls="-.", label="void-worth sign change")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel("$T$ [K]")
    ax.set_title(f"axial temperature profile at $t = {traj.t[-1]:.2f}$ s")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir / "final_temperature_profile.png")


def vapor_fraction(
    traj: AxialTrajectory,
    p: AxialParams,
    outdir: Path,
    times: tuple[float, ...] | None = None,
) -> Path:
    """Void fraction in space and time, with axial snapshots at selected instants.

    The snapshot instants default to 0, 25, 50, 75 and 100 **per cent** of the solved
    horizon plus the onset instant; ``times`` overrides with seconds. See
    ``_ALPHA_FRACTIONS`` for why percentages, and why not the requested 0/20/40/60.
    """
    t_on, z_on = _onset(traj, p)
    if times is None:
        times = tuple(f * float(traj.t[-1]) for f in _ALPHA_FRACTIONS)
    marks = [*times, t_on] if np.isfinite(t_on) else list(times)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
    im = ax0.pcolormesh(traj.t, traj.zeta, traj.alpha, cmap="Blues", shading="auto")
    fig.colorbar(im, ax=ax0, label=r"$\alpha$")
    for t_mark in marks:
        ax0.axvline(t_mark, color="k", lw=0.8, alpha=0.6)
    if np.isfinite(t_on):
        ax0.plot([t_on], [z_on], "r*", ms=11, label="onset")
        ax0.legend(fontsize=8, loc="upper left")
    ax0.set_xlabel("$t$ [s]")
    ax0.set_ylabel(r"$\zeta$")
    ax0.set_title("void fraction over space and time")

    cmap = plt.get_cmap("viridis")
    for i, t_mark in enumerate(marks):
        k = int(np.argmin(np.abs(traj.t - t_mark)))
        pct = 100.0 * traj.t[k] / traj.t[-1]
        is_onset = np.isfinite(t_on) and i == len(marks) - 1 and t_mark == t_on
        ax1.plot(
            traj.zeta,
            traj.alpha[:, k],
            lw=2.2 if is_onset else 1.6,
            ls="--" if is_onset else "-",
            color="crimson" if is_onset else cmap(i / max(len(marks) - 1, 1)),
            # The label states the time the frame was DRAWN AT, which is the nearest
            # grid column, not the time requested -- they differ by up to half a step.
            label=("onset, " if is_onset else "") + f"$t = {traj.t[k]:.2f}$ s ({pct:.0f}%)",
        )
    ax1.axvline(p.zeta_sign, color="grey", lw=0.9, ls="-.")
    ax1.set_xlabel(r"$\zeta$")
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("axial void profile at selected instants")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    return _save(fig, outdir / "vapor_fraction.png")


def temperature_map(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Coolant temperature over space and time, with the saturation contour on it.

    The contour is the boiling front: under D-TH-3 the front *is* the level set
    ``T_c = T_sat + dT_sup``, so drawing it on the field shows the two as one object
    rather than as a field and a separately computed curve.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    im = ax.pcolormesh(traj.t, traj.zeta, traj.T_c, cmap="inferno", shading="auto")
    fig.colorbar(im, ax=ax, label="$T_c$ [K]")
    ax.contour(traj.t, traj.zeta, traj.T_c, levels=[_T_boil(p)], colors="cyan", linewidths=1.6)
    t_on, z_on = _onset(traj, p)
    if np.isfinite(t_on):
        ax.plot([t_on], [z_on], "w*", ms=12)
        ax.annotate(
            f"onset\n{t_on:.3f} s",
            (t_on, z_on),
            textcoords="offset points",
            xytext=(-52, -6),
            color="w",
            fontsize=8,
        )
    ax.axhline(p.zeta_sign, color="w", lw=0.9, ls="-.")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$\zeta$")
    ax.set_title(r"coolant temperature; cyan is $T_{sat}+\Delta T_{sup}$")
    return _save(fig, outdir / "temperature_map.png")


def heat_flux(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Cladding-to-coolant heat flow: total against time, and its axial profile.

    The total is the channel's heat removal and falls as the flow coasts down; the
    axial profile shows where it collapses once the void insulates the wall, which is
    the mechanism that makes the front self-accelerating.
    """
    geo = node_geometry(p)
    q = clad_coolant_flux(traj.T_cl, traj.T_c, geo, p, traj.alpha)
    t_on, _ = _onset(traj, p)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, q.sum(axis=0) / 1e3, lw=1.8)
    _mark_onset(ax0, t_on)
    ax0.set_xlabel("$t$ [s]")
    ax0.set_ylabel("total $Q_{ec}$ [kW]")
    ax0.set_title("cladding-to-coolant heat flow")
    ax0.grid(alpha=0.3)
    if np.isfinite(t_on):
        ax0.legend(fontsize=8)

    for frac, style in ((0.0, ":"), (0.5, "--"), (1.0, "-")):
        k = min(int(frac * (traj.t.size - 1)), traj.t.size - 1)
        ax1.plot(traj.zeta, q[:, k] / 1e3, style, lw=1.6, label=f"$t = {traj.t[k]:.2f}$ s")
    ax1.axvline(p.zeta_sign, color="grey", lw=0.9, ls="-.")
    ax1.set_xlabel(r"$\zeta$")
    ax1.set_ylabel("$Q_{ec}$ per node [kW]")
    ax1.set_title("axial distribution of the flux")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    return _save(fig, outdir / "heat_flux.png")


def power(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Plot normalised power against time, which **requires the closed loop**.

    With ``feedback=False`` the power is prescribed, and plotting it would be a picture
    of the solver's own input; ``generate_all`` therefore hands this chart a trajectory
    solved with ``feedback=True``.
    """
    t_on, _ = _onset(traj, p)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.plot(traj.t, traj.power, lw=2.0)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    _mark_onset(ax, t_on)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$P/P_0$")
    ax.set_title("closed-loop power")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, outdir / "power.png")


def reactivity(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Net reactivity and its Doppler and coolant/void components, in units of beta.

    The split is the chart, not the net: the paper's open problem is that the void term
    is a near-cancellation of two large opposite contributions, and only the components
    show that.
    """
    t_on, _ = _onset(traj, p)
    b = p.beta_eff
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax.plot(traj.t, traj.rho / b, lw=2.0, label="net")
    ax.plot(traj.t, traj.rho_doppler / b, lw=1.6, label="Doppler")
    ax.plot(traj.t, traj.rho_void / b, lw=1.6, label="coolant / void")
    ax.axhline(0.0, color="k", lw=0.8)
    _mark_onset(ax, t_on)
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel(r"$\rho/\beta_{eff}$")
    ax.set_title("reactivity, split by mechanism")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir / "reactivity.png")


def front_height(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Boiling-front height against time, with the voided length beside it.

    Two definitions are drawn because they answer different questions and do not
    coincide: the saturation level set is where boiling *starts*, and the ``alpha > 0.5``
    contour is where the channel is actually voided. The gap between them is the
    partially voided region the worth integral is most sensitive to.
    """
    T_boil = _T_boil(p)
    t_on, z_on = _onset(traj, p)
    hot = traj.T_c > T_boil
    voided = traj.alpha > 0.5
    sat_front = np.where(hot.any(axis=0), traj.zeta[np.argmax(hot, axis=0)], np.nan)
    void_front = np.where(voided.any(axis=0), traj.zeta[np.argmax(voided, axis=0)], np.nan)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, sat_front, lw=2.0, label=r"$T_c = T_{sat}+\Delta T_{sup}$")
    ax0.plot(traj.t, void_front, "--", lw=2.0, label=r"$\alpha > 0.5$")
    ax0.axhline(p.zeta_sign, color="grey", lw=0.9, ls="-.", label="void-worth sign change")
    if np.isfinite(t_on):
        ax0.plot([t_on], [z_on], "r*", ms=11, label=f"onset {t_on:.3f} s")
    ax0.set_ylim(0.0, 1.0)
    ax0.set_ylabel(r"$\zeta$")
    ax0.set_title("front height")

    ax1.plot(traj.t, traj.voided_length, lw=2.0, color="tab:purple")
    ax1.set_ylabel(r"$L_{void}$ [m]")
    ax1.set_title("voided length")
    for ax in (ax0, ax1):
        _mark_onset(ax, t_on, label=False)
        ax.set_xlabel("$t$ [s]")
        ax.grid(alpha=0.3)
    ax0.legend(fontsize=8, loc="upper left")
    return _save(fig, outdir / "front_height.png")


# --- charts the paper wants that were not asked for -------------------------------
def void_worth_split(p: AxialParams, outdir: Path) -> Path:
    """Draw the void-worth distribution shaded by sign -- the paper's open problem.

    The worth is positive over most of the core and negative near the top, so the
    reactivity functional is a difference of two large numbers. Shading the two lobes
    makes the cancellation visible as an area rather than as a ratio in a table.
    """
    zeta = np.linspace(0.0, 1.0, 801)
    w = np.asarray(p.void_worth(zeta), dtype=float)
    pos, neg = np.trapezoid(np.clip(w, 0, None), zeta), np.trapezoid(np.clip(w, None, 0), zeta)
    ratio = abs(pos + neg) / (abs(pos) + abs(neg))

    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    ax.plot(zeta, w, lw=2.0, color="k")
    ax.fill_between(zeta, w, 0, where=w > 0, color="tab:red", alpha=0.3, label="positive worth")
    ax.fill_between(zeta, w, 0, where=w < 0, color="tab:blue", alpha=0.3, label="negative worth")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(p.zeta_sign, color="grey", lw=0.9, ls="-.", label=r"$\zeta_{sign}$")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$w(\zeta)$")
    ax.set_title(f"void worth; fully-voided cancellation ratio {ratio:.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, outdir / "void_worth_split.png")


def saturation_margin(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """How close the channel is to boiling, and the tangency that defines onset.

    The left panel is the safety margin the paper reports. The right is the reason
    onset must be root-found rather than read off a grid: the peak coolant temperature
    approaches the threshold tangentially, so a grid crossing quantises the answer.
    """
    T_boil = _T_boil(p)
    peak = traj.T_c.max(axis=0)
    t_on, _ = _onset(traj, p)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, T_boil - peak, lw=2.0)
    ax0.axhline(0.0, color="k", lw=0.9, ls=":")
    ax0.set_ylabel(r"$T_{sat}+\Delta T_{sup} - \max_\zeta T_c$ [K]")
    ax0.set_title("margin to boiling")

    ax1.plot(traj.t, peak, lw=2.0, label=r"$\max_\zeta T_c$")
    ax1.axhline(T_boil, color="k", lw=0.9, ls=":", label=r"$T_{sat}+\Delta T_{sup}$")
    if np.isfinite(t_on):
        ax1.set_xlim(max(traj.t[0], t_on - 1.0), min(traj.t[-1], t_on + 1.0))
        near = (traj.t > t_on - 1.0) & (traj.t < t_on + 1.0)
        if near.any():
            span = peak[near]
            ax1.set_ylim(span.min() - 1.0, max(span.max(), T_boil) + 1.0)
    ax1.set_ylabel("$T$ [K]")
    ax1.set_title("the tangency that defines onset")
    for ax in (ax0, ax1):
        _mark_onset(ax, t_on, label=ax is ax0)
        ax.set_xlabel("$t$ [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _save(fig, outdir / "saturation_margin.png")


def boundary_conditions(traj: AxialTrajectory, p: AxialParams, outdir: Path) -> Path:
    """Show what drives the transient: the flow coastdown and the outlet response.

    A reader asking "why does it boil?" needs the forcing, not only the response. The
    flow decays to the natural-circulation floor while the power is still near nominal,
    so the outlet temperature rises until it meets saturation.
    """
    t_on, _ = _onset(traj, p)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    ax0.plot(traj.t, traj.flow / traj.flow[0], lw=2.0)
    ax0.set_ylabel("$w(t)/w_0$")
    ax0.set_title("prescribed flow coastdown")
    ax0.set_ylim(0.0, 1.05)

    ax1.plot(traj.t, traj.T_c[0], lw=1.6, label="inlet")
    ax1.plot(traj.t, traj.T_out, lw=2.0, label="outlet")
    ax1.axhline(_T_boil(p), color="k", lw=0.9, ls=":", label=r"$T_{sat}+\Delta T_{sup}$")
    ax1.set_ylabel("$T_c$ [K]")
    ax1.set_title("coolant inlet and outlet")
    for ax in (ax0, ax1):
        _mark_onset(ax, t_on, label=ax is ax0)
        ax.set_xlabel("$t$ [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _save(fig, outdir / "boundary_conditions.png")


# --- driver ------------------------------------------------------------------------
#: Chart name -> (needs the closed loop, drawing function). The registry is what makes
#: `--only` possible and what keeps `generate_all` from being a list of calls to edit
#: in two places.
CHARTS: dict[str, tuple[bool, Callable[..., Path]]] = {
    "temperature_history": (False, temperature_history),
    "final_profile": (False, final_temperature_profile),
    "vapor_fraction": (False, vapor_fraction),
    "temperature_map": (False, temperature_map),
    "heat_flux": (False, heat_flux),
    "front_height": (False, front_height),
    "saturation_margin": (False, saturation_margin),
    "boundary_conditions": (False, boundary_conditions),
    "power": (True, power),
    "reactivity": (True, reactivity),
    "void_worth": (False, void_worth_split),
}


def generate_all(
    outdir: str | Path = DEFAULT_OUTDIR,
    n_axial: int = 160,
    n_out: int = 481,
    only: tuple[str, ...] = (),
    alpha_times: tuple[float, ...] | None = None,
) -> list[Path]:
    """Draw the requested charts. Returns the paths written.

    The open-loop and closed-loop references are solved **at most once each**, and only
    if some selected chart needs them -- a closed-loop solve is the expensive one and
    ``--only front_height`` should not pay for it.
    """
    out = Path(outdir)
    p = AxialParams(n_axial=n_axial)
    wanted = tuple(only) or tuple(CHARTS)
    unknown = [name for name in wanted if name not in CHARTS]
    if unknown:
        msg = f"unknown chart(s) {unknown}; available: {sorted(CHARTS)}"
        raise ValueError(msg)

    cache: dict[bool, AxialTrajectory] = {}

    def traj_for(feedback: bool) -> AxialTrajectory:  # noqa: FBT001
        if feedback not in cache:
            cache[feedback] = solve_reference(p, n_out=n_out, feedback=feedback)
        return cache[feedback]

    paths: list[Path] = []
    for name in wanted:
        needs_loop, draw = CHARTS[name]
        if draw is void_worth_split:
            paths.append(draw(p, out))
        elif draw is vapor_fraction:
            paths.append(draw(traj_for(needs_loop), p, out, alpha_times))
        else:
            paths.append(draw(traj_for(needs_loop), p, out))
    return paths


def main(argv: list[str] | None = None) -> int:
    """Draw the transient charts. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--n-axial", type=int, default=160)
    ap.add_argument("--n-out", type=int, default=481, help="time samples in the reference")
    ap.add_argument(
        "--only",
        default="",
        help=f"comma-separated subset of {sorted(CHARTS)}",
    )
    ap.add_argument(
        "--alpha-times",
        default="",
        help="comma-separated times in SECONDS for the void snapshots; "
        "default is 0/20/40/60 per cent of the solved horizon, plus onset",
    )
    args = ap.parse_args(argv)
    only = tuple(s.strip() for s in args.only.split(",") if s.strip())
    times = tuple(float(s) for s in args.alpha_times.split(",") if s.strip()) or None
    for path in generate_all(args.outdir, args.n_axial, args.n_out, only, times):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
