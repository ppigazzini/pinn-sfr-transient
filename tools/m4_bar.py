"""Is M4's acceptance criterion attainable at all? — the D35 protocol, applied to onset.

M4 asks for boiling onset within **0.5 s and one cell**. `docs/axial_nn.md` §7.5.16
notes that one cell is 0.405 K of coolant temperature, i.e. a relative `L2` of
0.0026 — four times tighter than the 1% bar the temperatures are held to, and only
1.6 to 2.3 times above the *reference solver's own* error.

That is D35's failure mode exactly: an acceptance bar sitting at the ruler's
precision, which was already retracted once for the temperatures (§6.5) and has
never been checked for onset. A bar below the ruler is not a hard target, it is an
unmeasurable one, and a month spent chasing it would be a month spent measuring the
reference's discretisation error.

The test needs no network. **Refine the reference against itself**: solve at
increasing `n_axial` and watch how far its own onset moves. Whatever residual
movement survives at the finest meshes is the floor — no PINN can be scored inside
it, because the quantity it would be scored against is not converged to there.

Reported for both readouts, because §7.5.16 showed they are conditioned differently:
the threshold crossing is `sqrt`-conditioned at a maximum, the tangency solve is
linear, and the criterion should be stated against whichever one is actually used.

    uv run python tools/m4_bar.py
"""

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference
from pinn_sfr_transient.axial.scoring import onset_by_tangency

# The scoring mesh every published number uses, and the ladder around it.
SCORING_N = 160
MESHES = (40, 80, 160, 320, 640, 1280)

# A tolerance must sit at least this far above the uncertainty of the instrument
# measuring it -- MIL-STD-45662A, carried into ANSI/NCSL Z540. Below 4 the bar is
# partly measuring the ruler; below 1 it is measuring nothing else.
MIN_TUR = 4.0


def void_split(traj, p: AxialParams) -> tuple[float, float, float, float]:  # noqa: ANN001
    """Return ``(J+, J-, J, t)`` for the closed-loop void functional at its peak.

    Reported split because the two halves nearly cancel: `docs/axial_nn.md` §7.5.23
    measures a cancellation ratio of 0.4663, so a relative error on each half becomes
    2.1x that on the sum, and a single number cannot say which half moved.
    """
    zeta = np.asarray(traj.zeta)
    w = np.asarray(p.void_worth(zeta))
    dz = float(zeta[1] - zeta[0])
    contrib = w[:, None] * np.asarray(traj.alpha) * dz
    jp = contrib[w > 0].sum(axis=0)
    jn = contrib[w < 0].sum(axis=0)
    k = int(np.argmax(np.abs(jp + jn)))
    return float(jp[k]), float(jn[k]), float(jp[k] + jn[k]), float(traj.t[k])


def main() -> int:
    """Refine the reference and report how far its own onset still moves."""
    p0 = AxialParams()
    thr = float(sodium.saturation_temperature(p0.p_system) + p0.dT_superheat)
    cell = 1.0 / SCORING_N

    print(f"M4 asks for onset within 0.5 s and one cell (dzeta = {cell:.5f}).")
    print("Refining the REFERENCE against itself; no network involved.\n")
    print(f"{'n_axial':>8}{'t_thr':>10}{'z_thr':>10}{'t_tan':>10}{'z_tan':>10}")

    rows = []
    voids = {}
    lvoid = {}
    for n in MESHES:
        traj = solve_reference(AxialParams(n_axial=n), n_out=241)
        t_thr, z_thr = traj.onset()
        t_tan, z_tan = onset_by_tangency(traj.T_c, traj.zeta, traj.t, thr)
        rows.append((n, t_thr, z_thr, t_tan, z_tan))
        voids[n] = void_split(traj, p0)
        lvoid[n] = float(np.max(traj.voided_length))
        print(f"{n:>8}{t_thr:>10.3f}{z_thr:>10.5f}{t_tan:>10.3f}{z_tan:>10.5f}", flush=True)

    fine = rows[-1]
    print(f"\nTaking n_axial = {fine[0]} as truth, error of every coarser mesh:")
    print(
        f"{'n_axial':>8}{'dt_thr [s]':>12}{'dz_thr [cells]':>16}"
        f"{'dt_tan [s]':>12}{'dz_tan [cells]':>16}"
    )
    for n, t_thr, z_thr, t_tan, z_tan in rows[:-1]:
        print(
            f"{n:>8}{abs(t_thr - fine[1]):>12.3f}{abs(z_thr - fine[2]) / cell:>16.2f}"
            f"{abs(t_tan - fine[3]):>12.3f}{abs(z_tan - fine[4]) / cell:>16.2f}"
        )

    at_scoring = next(r for r in rows if r[0] == SCORING_N)
    dz_thr = abs(at_scoring[2] - fine[2]) / cell
    dz_tan = abs(at_scoring[4] - fine[4]) / cell
    dt_thr = abs(at_scoring[1] - fine[1])
    dt_tan = abs(at_scoring[3] - fine[3])

    print(f"\n=== the verdict, at the scoring mesh n_axial = {SCORING_N} ===")
    for label, dt, dz in (("threshold", dt_thr, dz_thr), ("tangency", dt_tan, dz_tan)):
        verdict = "ATTAINABLE" if dz < 1.0 else "BELOW THE RULER"
        print(
            f"  {label:<10} the reference's own onset is uncertain by "
            f"{dt:.3f} s and {dz:.2f} cells  ->  a one-cell criterion is {verdict}"
        )
    print(
        "\nA criterion inside the reference's own uncertainty cannot be met or\n"
        "failed by a network; it can only be met or failed by the mesh."
    )

    # --- which quantity should M4 be replaced with? -------------------------
    # M4 is now un-failable on height (T_c is monotone, so the peak is the last node
    # whatever the network does) and passed on time (§7.5.16a). A replacement has to
    # clear two hurdles at once: its bar must sit >= MIN_TUR above the reference's own
    # error, AND the network's current error must be far enough above that error to
    # leave something to measure. A quantity we already match to within the ruler is
    # useless as a criterion however important it is physically.
    fin = MESHES[-1]
    print(f"\n=== candidate criteria, ruler at n_axial = {SCORING_N} vs {fin} ===")
    print(f"{'quantity':<22}{'ruler':>12}{'network now':>14}{'TUR of result':>15}")

    jp_s, jn_s, j_s, t_peak = voids[SCORING_N]
    jp_f, jn_f, j_f, _ = voids[fin]
    ruler = {
        "void J+": abs(jp_s - jp_f) / abs(jp_f) * 100.0,
        "void J-": abs(jn_s - jn_f) / abs(jn_f) * 100.0,
        "void J (sum)": abs(j_s - j_f) / abs(j_f) * 100.0,
        "peak voided length": abs(lvoid[SCORING_N] - lvoid[fin]) / lvoid[fin] * 100.0,
    }
    # Where the network stands today, from docs/axial_nn.md: §7.4 for the void integral
    # and §7.5.20 for the voided length. `None` means NOT MEASURED -- §7.4's 84-92% is a
    # miss on the SUM, and nobody has ever split the network's error into its two halves.
    # Writing 84% on each row would be inventing two numbers from one, which is the error
    # the split exists to prevent.
    current: dict[str, float | None] = {
        "void J+": None,
        "void J-": None,
        "void J (sum)": 84.0,
        "peak voided length": 0.7,
    }
    for k, r in ruler.items():
        cur = current[k]
        if cur is None:
            print(f"  {k:<20}{r:>11.3f}%{'not measured':>14}{'--':>15}")
        else:
            print(f"  {k:<20}{r:>11.3f}%{cur:>13.1f}%{cur / r:>15.1f}")

    print(
        f"\nThe void functional peaks at t = {t_peak:.2f} s, the END of the valid window,\n"
        "which is where the ansatz exp(t_hat N) has its largest excursion and the\n"
        "network is least constrained -- so it is a hard target as well as a live one."
    )
    print(
        "\nRECOMMENDATION. Score the closed-loop void reactivity, SPLIT into J+ and J-,\n"
        f"and set each bar at >= {MIN_TUR:.0f}x its ruler above. It is the only quantity\n"
        "measured here that the network is still far from: the temperatures, the onset\n"
        "time and the voided length are all now within a factor of ~2 of the reference's\n"
        "own error, so none of them can rank two formulations any more."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
