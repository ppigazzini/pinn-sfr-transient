"""Numerical uncertainty of the axial reference solution, by grid convergence.

The reference is the ruler every accuracy claim is quoted against, so its own error has
to be quantified before any of those claims mean anything. It is a finite-volume
discretisation, so that error is set by the axial mesh and is estimated the standard
way: solve on a sequence of meshes, observe the order of convergence, and extrapolate.

    uv run python tools/axial_study.py verify --out __DEV/studies/verify.json

That command is the *only* source of a reference uncertainty in this repository. Before
it existed, ``paper/paper.tex`` carried a full Richardson table — observed orders, an
extrapolated limit, four solve timings — that no committed command reproduced and no
row of ``__DEV/studies/`` contained. AGENTS.md's reproducibility rule was being enforced
on studies, and a paper is not a study, so nothing caught it. Every uncertainty quoted
anywhere downstream must now be grep-able in this command's output.

Ported from ``verification.py`` in the companion repository ``pinn-ulof``, which shares
this model's physics — ``AxialParams`` is identical field for field, and the reference
solver differs only in prose. Two corrections were made on the way in, both recorded in
the functions concerned:

1. :func:`richardson` rejected non-monotone sequences and returned a finite,
   meaningless limit for a monotone but *diverging* one.
2. The field uncertainties were scaled by a hard-coded factor of two, which assumes
   first-order convergence while the same routine measures the order for the scalars a
   few lines above. The observed field order is now measured and used.
"""

import time
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import energy_balance, solve_reference
from pinn_sfr_transient.axial.scoring import onset_by_tangency

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.reference import AxialTrajectory

#: Meshes solved by default. Three is the minimum that lets the order be *observed*
#: rather than assumed; the extra rungs confirm the order holds over more than one
#: doubling. Doubling is required — :func:`richardson` assumes a refinement ratio of 2.
#:
#: The ladder runs to 2560 because that is the rung the scoring-mesh decision turns on,
#: not because four rungs were not enough to see the order. A mesh has no field
#: uncertainty without a successor, so stopping at 1280 leaves the candidate ruler as
#: the one row with no number against it — and AGENTS.md's rule is that the sample has
#: to be sufficient on the rung the conclusion rests on, not on the ladder as a whole.
#: The five solves cost about 145 s in total, against hours to train what they measure.
MESHES: tuple[int, ...] = (160, 320, 640, 1280, 2560)

#: Time samples. The solve terminates at the validity horizon, so only the first ~67
#: land. Matches ``tools/axial_study.py``, so a verification run and a scoring run
#: sample the reference identically.
N_OUT: int = 241

#: Scalars whose uncertainty is reported, and the unit each is quoted in.
SCALARS: tuple[tuple[str, str], ...] = (
    ("onset", "s"),
    ("zeta", ""),
    ("Lvoid", "m"),
    ("margin", "K"),
)

#: Fields whose relative L2 uncertainty is reported.
FIELDS: tuple[str, ...] = ("T_f", "T_cl", "T_s", "T_c")

#: Refinement ratio between successive entries of :data:`MESHES`.
_RATIO: float = 2.0

#: Ratio of model error to reference uncertainty below which an accuracy claim is not
#: supportable. Calibration practice (MIL-STD-45662A, ANSI/NCSL Z540) requires a
#: tolerance to sit at least four times above the uncertainty of the instrument
#: measuring it; below one, the number is measuring the ruler.
MIN_RATIO: float = 4.0


def solve(n_axial: int, p: AxialParams | None = None, *, feedback: bool = False) -> tuple:
    """Solve the reference on ``n_axial`` nodes. Returns the trajectory and the seconds."""
    p = p or AxialParams()
    t0 = time.perf_counter()
    traj = solve_reference(replace(p, n_axial=n_axial), n_out=N_OUT, feedback=feedback)
    return traj, time.perf_counter() - t0


def scalars(traj: AxialTrajectory, p: AxialParams) -> dict[str, float]:
    """Return the four scalars whose convergence is tracked.

    Onset comes from the tangency readout rather than a threshold crossing, for the
    reason :func:`~pinn_sfr_transient.axial.scoring.onset_by_tangency` gives: near the
    peak the crossing displaces as the square root of the field error, which is the
    worst possible law for a quantity whose mesh convergence is the thing being
    measured.
    """
    T_boil = float(sodium.saturation_temperature(p.p_system)) + p.dT_superheat
    t_on, z_on = onset_by_tangency(traj.T_c, traj.zeta, traj.t, T_boil)
    return {
        "onset": float(t_on),
        "zeta": float(z_on),
        "Lvoid": float(traj.voided_length.max()),
        "margin": float(traj.T_c.max()) - T_boil,
    }


def richardson(coarse: float, medium: float, fine: float) -> tuple[float, float]:
    """Observed order and extrapolated limit from three values on doubling meshes.

    ``f_exact ~ f_h + (f_h - f_2h) / (2^p - 1)``.

    Returns ``(nan, nan)`` unless the sequence is *converging*, which needs two
    conditions and not one:

    - **monotone**, ``d1 / d2 > 0``. Otherwise the quantity is not in the asymptotic
      range and extrapolating it would invent a number.
    - **contracting**, ``abs(d1 / d2) > 1``, i.e. an observed order above zero. The
      companion implementation checked only the first, so a monotone sequence whose
      successive differences *grow* under refinement — a solution getting worse as the
      mesh is refined, which is a discretisation defect and not a convergence rate —
      produced a negative order, a negative ``2^p - 1``, and a finite extrapolated
      limit that means nothing. A number is worse than a ``nan`` here, because
      something downstream will quote it.

    A near-zero order is not rejected: ``margin`` converges at an observed 0.70 and is
    excluded by the caller on the grounds that it derives from a maximum over a
    discrete field rather than from a smooth functional. That is a judgement about the
    quantity, not about the arithmetic, so it belongs where the quantity is known.
    """
    d1, d2 = medium - coarse, fine - medium
    if d2 == 0.0 or d1 / d2 <= 0.0 or abs(d1 / d2) <= 1.0:
        return float("nan"), float("nan")
    order = float(np.log2(abs(d1 / d2)))
    return order, float(fine + d2 / (_RATIO**order - 1.0))


def field_l2(coarse: AxialTrajectory, fine: AxialTrajectory) -> dict[str, float]:
    """Relative L2 of a coarse solution against a fine one, sampled on the fine nodes."""
    n_t = min(len(coarse.t), len(fine.t))
    out = {}
    for name in FIELDS:
        a, b = getattr(coarse, name), getattr(fine, name)
        interp = np.empty((len(fine.zeta), n_t))
        for j in range(n_t):
            interp[:, j] = np.interp(fine.zeta, coarse.zeta, a[:, j])
        out[name] = float(np.linalg.norm(interp - b[:, :n_t]) / np.linalg.norm(b[:, :n_t]))
    return out


def _gap_to_limit(gap: float, order: float) -> float:
    """Error of a mesh from its gap to the next one, at an observed ``order``.

    With ``e_h = C h^p`` the gap between successive doubling meshes is
    ``e_h - e_{h/2} = e_h (1 - 2^-p)``, so ``e_h = gap / (1 - 2^-p)``. At ``p = 1``
    that is the familiar factor of two; the companion implementation hard-coded it,
    which silently asserts first order on fields whose order was never measured.
    """
    if not np.isfinite(order) or order <= 0.0:
        return float("nan")
    return gap / (1.0 - _RATIO**-order)


def _field_orders(gaps: list[dict[str, float]]) -> dict[str, float]:
    """Observed order of each field, from the last two successive-mesh gaps.

    The gaps themselves are the error differences, and they contract by ``2^p`` per
    doubling just as the errors do, so the order reads off the same way. The last pair
    is used rather than an average over all pairs: the asymptotic range is approached
    from below, and the coarse rungs are exactly the ones not in it.
    """
    if len(gaps) < 2:
        return dict.fromkeys(FIELDS, float("nan"))
    prev, last = gaps[-2], gaps[-1]
    out = {}
    for f in FIELDS:
        ratio = last[f] and prev[f] / last[f]
        out[f] = float(np.log2(ratio)) if ratio and ratio > 1.0 else float("nan")
    return out


def _print_solve_header() -> None:
    """Column headings of the per-mesh table."""
    header = " ".join(f"{k:>13s}" for k, _ in SCALARS)
    print(f"{'mesh':>6s} {'solve':>9s} {header} {'energy bal.':>13s}", flush=True)


def _print_solve_row(r: dict) -> None:
    """One mesh: the tracked scalars, the solve cost and the energy closure.

    Printed as each solve returns rather than as a table at the end. The finest rung
    takes minutes, and a command that shows nothing until every mesh is done cannot be
    told apart from one that has hung.
    """
    cells = " ".join(f"{r['scalars'][k]:13.6f}" for k, _ in SCALARS)
    print(f"{r['mesh']:6d} {r['seconds']:8.1f}s {cells} {r['energy_balance']:13.2e}", flush=True)


def _print_orders(orders: dict[str, float], limits: dict[str, float]) -> None:
    """Observed order and extrapolated limit of each scalar."""
    print("\nobserved order and extrapolated limit, from the finest three meshes:")
    for key, unit in SCALARS:
        order, limit = orders[key], limits[key]
        note = "" if np.isfinite(order) else "   (not converging; not extrapolated)"
        print(f"  {key:7s} p={order:5.2f}  limit={limit:.6f} {unit}{note}")


def _print_uncertainty(meshes: tuple[int, ...], unc: dict[int, dict[str, float]]) -> None:
    """Uncertainty of every mesh, scalars then fields."""
    print("\nuncertainty of each mesh:")
    print(
        f"{'mesh':>6s} "
        + " ".join(f"{k:>13s}" for k, _ in SCALARS)
        + "   "
        + " ".join(f"{f:>11s}" for f in FIELDS)
    )
    for n in meshes:
        row = unc[n]
        cells = " ".join(f"{row[k]:13.6f}" for k, _ in SCALARS)
        fields = " ".join(f"{row[f]:11.3e}" if f in row else f"{'--':>11s}" for f in FIELDS)
        print(f"{n:6d} {cells}   {fields}")


def report(
    meshes: tuple[int, ...] = MESHES,
    p: AxialParams | None = None,
    *,
    feedback: bool = False,
) -> dict:
    """Solve the sequence, print the convergence tables, and return everything measured.

    The return value is JSON-serialisable and is what
    ``tools/axial_study.py verify`` writes, so every uncertainty this repository
    quotes is grep-able in one file.

    At least three meshes are required: the observed order of convergence is measured
    from three successive values, not assumed. They must also *double*, because
    :func:`richardson` and :func:`_gap_to_limit` both bake in a refinement ratio of two;
    the default honours that and an argument might not.
    """
    if len(meshes) < 3:
        msg = f"need at least three meshes to observe the order, got {list(meshes)}"
        raise ValueError(msg)
    # Checked here rather than trusted. `richardson` sees three floats and cannot know
    # what meshes produced them, so a sequence like (160, 240, 360) would yield a
    # confident order and a confident limit, both meaningless. An unmeasurable case has
    # to be an error, not a silently wrong number.
    bad = [(a, b) for a, b in pairwise(meshes) if b != 2 * a]
    if bad:
        msg = (
            f"meshes must double: {list(meshes)} breaks at {bad[0]}. "
            f"The extrapolation assumes a refinement ratio of {_RATIO:g}."
        )
        raise ValueError(msg)
    p = p or AxialParams()

    rows, runs = [], {}
    _print_solve_header()
    for n in meshes:
        traj, sec = solve(n, p, feedback=feedback)
        runs[n] = traj
        # `replace(p, n_axial=n)`, not `p`: the geometry must come from the mesh the
        # trajectory was solved on. Pairing a 40-node geometry with a 160-node
        # solution turns a closure of 5e-5 into 0.5.
        rows.append(
            {
                "mesh": n,
                "seconds": sec,
                "scalars": scalars(traj, p),
                "energy_balance": float(energy_balance(traj, replace(p, n_axial=n))),
            }
        )
        _print_solve_row(rows[-1])

    vals = {r["mesh"]: r["scalars"] for r in rows}
    orders, limits = {}, {}
    for key, _ in SCALARS:
        orders[key], limits[key] = richardson(*(vals[n][key] for n in meshes[-3:]))
    _print_orders(orders, limits)

    gaps = [field_l2(runs[a], runs[b]) for a, b in pairwise(meshes)]
    f_orders = _field_orders(gaps)

    unc: dict[int, dict[str, float]] = {}
    for i, n in enumerate(meshes):
        row = {k: abs(vals[n][k] - limits[k]) for k, _ in SCALARS}
        # The finest mesh has no successor, so it has no field estimate here.
        if i < len(gaps):
            row.update({f: _gap_to_limit(gaps[i][f], f_orders[f]) for f in FIELDS})
        unc[n] = row
    _print_uncertainty(meshes, unc)

    return {
        "meshes": list(meshes),
        "n_out": N_OUT,
        "feedback": feedback,
        "rows": rows,
        "scalar_order": orders,
        "scalar_limit": limits,
        "field_order": f_orders,
        "field_gap": [dict(g) for g in gaps],
        "uncertainty": {str(n): unc[n] for n in meshes},
    }
