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

from __future__ import annotations

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference
from pinn_sfr_transient.axial.scoring import onset_by_tangency

# The scoring mesh every published number uses, and the ladder around it.
SCORING_N = 160
MESHES = (40, 80, 160, 320, 640, 1280)


def main() -> int:
    """Refine the reference and report how far its own onset still moves."""
    p0 = AxialParams()
    thr = float(sodium.saturation_temperature(p0.p_system) + p0.dT_superheat)
    cell = 1.0 / SCORING_N

    print(f"M4 asks for onset within 0.5 s and one cell (dzeta = {cell:.5f}).")
    print("Refining the REFERENCE against itself; no network involved.\n")
    print(f"{'n_axial':>8}{'t_thr':>10}{'z_thr':>10}{'t_tan':>10}{'z_tan':>10}")

    rows = []
    for n in MESHES:
        traj = solve_reference(AxialParams(n_axial=n), n_out=241)
        t_thr, z_thr = traj.onset()
        t_tan, z_tan = onset_by_tangency(traj.T_c, traj.zeta, traj.t, thr)
        rows.append((n, t_thr, z_thr, t_tan, z_tan))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
