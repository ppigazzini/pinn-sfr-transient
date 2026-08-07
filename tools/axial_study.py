"""Reproduce every measured table in `docs/axial_nn.md`.

Defect D67: the published axial tables were produced by scratch files that were
never committed, and one of them used a `t_train_frac` that differed from the
shipped default. The numbers were reproducible in principle and not in practice —
at the documented default the model formed no boiling front at all, which nobody
noticed for four milestones.

A number is reproducible because the configuration that produced it is in the
repository. Each sub-command here is one study, and each prints a table in the
form the documentation carries it.

    uv run python tools/axial_study.py ruler        # section 6.5  — reference mesh convergence
    uv run python tools/axial_study.py horizon      # section 7.2.7 — the training horizon
    uv run python tools/axial_study.py budget       # section 7.5.3 — Adam against quasi-Newton
    uv run python tools/axial_study.py optimizer    # section 7.5   — L-BFGS against SSBFGS
    uv run python tools/axial_study.py parity       # section 7.3.2 — torch against JAX
    uv run python tools/axial_study.py plan-a       # section 7.4   — closed-loop power

**Pin `OMP_NUM_THREADS`.** It defaults to every core, so a second job silently
halves the throughput of the first, and thread count changes float reduction
order. A wall-clock without a stated thread budget is not a measurement.

    OMP_NUM_THREADS=8 uv run python tools/axial_study.py budget

Every study writes JSON alongside its table so a result can be re-tabulated
without re-running it. Training studies take tens of minutes per arm on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference

if TYPE_CHECKING:
    from collections.abc import Callable

# The mesh every PINN table is scored against, and the mesh a study converges to.
RULER_N = 160
FINEST_N = 640
SEEDS = (0, 1, 2)
FIELDS = ("T_f", "T_cl", "T_s", "T_c")


# --- shared helpers ---------------------------------------------------------
def ruler(n: int = RULER_N, *, feedback: bool = False) -> Any:  # noqa: ANN401
    """Return the held-out reference the PINN is scored against."""
    return solve_reference(AxialParams(n_axial=n), n_out=241, feedback=feedback)


def score(fields: tuple, traj: Any) -> dict[str, float]:  # noqa: ANN401
    """Relative ``L2`` per temperature, plus the two front metrics."""
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS, fields[:4], ref, strict=True)
    }
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H
    out["max_alpha"] = float(fields[4].max())
    out["L_void_max"] = float((fields[4].sum(axis=0) * dz).max())

    # Under D-TH-3 the void is a function of `T_c` alone, so "the front forms" is
    # not a separate phenomenon: it is the single inequality
    # `max T_c > T_sat + dT_superheat`. Record the margin, because relative `L2`
    # is an average and this is an extremum -- a fit can improve on one while
    # losing the other, which is exactly what the budget sweep shows.
    p_ax = AxialParams()
    threshold = sodium.saturation_temperature(p_ax.p_system) + p_ax.dT_superheat
    out["max_T_c"] = float(fields[3].max())
    out["max_T_c_ref"] = float(traj.T_c.max())
    out["T_boil"] = float(threshold)
    out["margin_K"] = out["max_T_c"] - threshold
    out["margin_K_ref"] = out["max_T_c_ref"] - threshold
    return out


def train_torch(**kw: Any) -> Callable[[Any], tuple]:  # noqa: ANN401
    """Train the torch backend and return a predictor over the ruler's grid."""
    from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

    model = train(AxialParams(), AxialTrainConfig(log_every=10**9, **kw))
    return lambda traj: model.predict(traj.zeta, traj.t)


def train_jax(**kw: Any) -> Callable[[Any], tuple]:  # noqa: ANN401
    """Train the JAX backend and return a predictor over the ruler's grid."""
    from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

    model, p, _ = pj.train(AxialParams(), pj.AxialTrainConfig(log_every=10**9, **kw), verbose=False)
    return lambda traj: pj.predict(model, p, traj.zeta, traj.t)


def run_arm(traj: Any, label: str, backend: str, **kw: Any) -> dict:  # noqa: ANN401
    """One trained arm, timed and scored."""
    t0 = time.perf_counter()
    predict = (train_torch if backend == "torch" else train_jax)(**kw)
    dt = time.perf_counter() - t0
    # Record the load average with every timing. A wall-clock is only a
    # measurement against a stated thread budget AND a stated contention level,
    # and these studies are sometimes run concurrently -- putting it in the row
    # means a later reader cannot mistake a contended time for a clean one.
    row = score(predict(traj), traj) | {
        "arm": label,
        "backend": backend,
        "sec": dt,
        "load1": os.getloadavg()[0],
        "omp": os.environ.get("OMP_NUM_THREADS", "unset"),
        **kw,
    }
    print(
        f"{label:24s} {backend:5s} "
        + " ".join(f"{k}={row[k]:.4f}" for k in FIELDS)
        + f" maxA={row['max_alpha']:.4f} L_void={row['L_void_max']:.4f}"
        + f" maxTc={row['max_T_c']:.1f}K (thr {row['T_boil']:.1f}, margin {row['margin_K']:+.1f})"
        + f" {dt:.0f}s (load {row['load1']:.1f}, OMP {row['omp']})",
        flush=True,
    )
    return row


def write(rows: list[dict], out: Path) -> None:
    """Persist a study's rows so a table can be rebuilt without re-running it."""
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def mean_table(rows: list[dict], key: str = "arm") -> None:
    """Seed-averaged summary, which is what the documentation tables carry."""
    print("\n=== means over seeds ===")
    for label in dict.fromkeys(r[key] for r in rows):
        group = [r for r in rows if r[key] == label]
        m = {k: float(np.mean([r[k] for r in group])) for k in (*FIELDS, "max_alpha", "L_void_max")}
        print(
            f"{label:24s} n={len(group)} "
            + " ".join(f"{k}={m[k]:.4f}" for k in FIELDS)
            + f" maxA={m['max_alpha']:.4f} L_void={m['L_void_max']:.4f}"
        )


def run_all(
    traj: Any,  # noqa: ANN401
    specs: list[tuple[str, dict]],
    out: Path,
    backend: str = "torch",
) -> list[dict]:
    """Run every spec, writing after each so a killed study keeps what it measured.

    These studies run for hours. Collecting rows and writing once at the end means
    a machine reboot, an OOM or a stray kill loses everything -- and this project
    has already lost an ablation that way ("the ablation run was killed before its
    three configurations finished", section 7.6).
    """
    rows: list[dict] = []
    for label, kw in specs:
        rows.append(run_arm(traj, label, kw.pop("backend", backend), **kw))
        write(rows, out)
    return rows


# --- studies ----------------------------------------------------------------
def study_ruler(out: Path) -> None:
    """Measure how wrong the reference is -- section 6.5."""
    meshes = (40, 80, RULER_N, 320, FINEST_N)
    runs = {}
    for n in meshes:
        t0 = time.perf_counter()
        runs[n] = solve_reference(AxialParams(n_axial=n), n_out=241)
        t_on, z_on = runs[n].onset()
        print(
            f"n={n:4d} {time.perf_counter() - t0:6.1f}s onset={t_on:.4f}s @ zeta={z_on:.4f} "
            f"L_void_max={runs[n].voided_length.max():.5f} m "
            f"peak_clad={runs[n].peak_clad:.2f} K",
            flush=True,
        )

    finest = runs[FINEST_N]
    rows = []
    for n in meshes[:-1]:
        errs = {}
        for name in (*FIELDS, "alpha"):
            ref = _interp(finest, getattr(finest, name), finest.zeta, finest.t)
            got = _interp(runs[n], getattr(runs[n], name), finest.zeta, finest.t)
            errs[name] = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        rows.append({"n_axial": n, **errs})
        print(f"n={n:4d} vs {FINEST_N}: " + "  ".join(f"{k}={v:.4e}" for k, v in errs.items()))
    write(rows, out)


def _interp(traj: Any, field: np.ndarray, zeta: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Bilinear interpolation onto the finest mesh's grid."""
    cols = np.array([np.interp(t, traj.t, field[i, :]) for i in range(field.shape[0])])
    return np.array([np.interp(zeta, traj.zeta, cols[:, j]) for j in range(len(t))]).T


def study_horizon(out: Path) -> None:
    """Sweep the training horizon, the knob whose default formed no front -- section 7.2.7."""
    traj = ruler()
    rows = [
        run_arm(
            traj,
            f"t_train_frac={ttf}",
            "torch",
            t_train_frac=ttf,
            seed=0,
            adam_iters=3000,
            lbfgs_iters=300,
        )
        for ttf in (0.25, 0.275, 0.30, 1.0)
    ]
    write(rows, out)


def study_budget(out: Path) -> None:
    """Split the iteration budget between Adam and the quasi-Newton stage -- section 7.5.3.

    arXiv:2501.16371's winning schedule is Adam[1000] + quasi-Newton[30000]; this
    project's default gives the quasi-Newton stage 9%, and section 7.3.4 shows that
    stage is what forms the front.
    """
    traj = ruler()
    arms = (
        ("A adam3000/qn300", 3000, 300),
        ("B adam1000/qn2300", 1000, 2300),
        ("C adam300/qn3000", 300, 3000),
    )
    rows = run_all(
        traj,
        [
            (label, {"adam_iters": adam, "lbfgs_iters": qn, "seed": seed})
            for seed in SEEDS
            for label, adam, qn in arms
        ],
        out,
    )
    mean_table(rows)


def study_optimizer(out: Path) -> None:
    """Compare self-scaled BFGS against L-BFGS at the shipped split -- section 7.5."""
    traj = ruler()
    rows = run_all(
        traj,
        [
            (opt, {"optimizer": opt, "seed": seed, "adam_iters": 3000, "lbfgs_iters": 300})
            for seed in SEEDS
            for opt in ("lbfgs", "lbfgs-shared", "ssbfgs")
        ],
        out,
    )
    mean_table(rows)


def study_parity(out: Path) -> None:
    """Test whether the 21% backend gap on `T_s`/`T_c` is the optimiser -- section 7.3.2.

    `lbfgs-shared` runs this repository's own L-BFGS in both backends, removing the
    last component that is not shared source. The residuals are already known
    identical at identical parameters, so if the gap survives this it is neither
    the equations nor the optimiser implementation.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"{backend}/{opt}",
                {
                    "backend": backend,
                    "optimizer": opt,
                    "seed": seed,
                    "adam_iters": 3000,
                    "lbfgs_iters": 300,
                },
            )
            for seed in SEEDS
            for backend in ("torch", "jax")
            for opt in ("lbfgs", "lbfgs-shared")
        ],
        out,
    )
    mean_table(rows)


def study_plan_a(out: Path) -> None:
    """Measure closed-loop power at three seeds -- section 7.4 has only one."""
    from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

    # Say so before the long silence: the closed-loop reference at n = 160 plus the
    # first Plan A training is ~25 minutes before anything prints, and a study that
    # looks hung is a study someone kills.
    print("solving the closed-loop reference at n_axial=160 ...", flush=True)
    ref = ruler(feedback=True)
    print(
        f"reference: peak={ref.power.max():.4f} min={ref.power.min():.4f} "
        f"max rho/beta={ref.peak_rho_over_beta:+.4f}; training {len(SEEDS)} seeds",
        flush=True,
    )
    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        # Plan A needs no truncation: with feedback the transient is self-limiting
        # and completes 60 s inside the property range.
        model = train(
            AxialParams(),
            AxialTrainConfig(feedback=True, seed=seed, t_train_frac=1.0, log_every=10**9),
        )
        dt = time.perf_counter() - t0
        power, rho = model.predict_power(ref.t)
        row = {
            "seed": seed,
            "L2_P": float(np.linalg.norm(power - ref.power) / np.linalg.norm(ref.power)),
            "P0": float(power[0]),
            "peak_P": float(power.max()),
            "min_P": float(power.min()),
            "max_rho_beta": float(rho.max() / ref._beta),  # noqa: SLF001
            "min_rho_beta": float(rho.min() / ref._beta),  # noqa: SLF001
            "sec": dt,
        }
        rows.append(row)
        print(
            f"seed={seed} L2(P)={row['L2_P']:.4f} P(0)={row['P0']:.6f} "
            f"peak={row['peak_P']:.4f} min={row['min_P']:.4f} "
            f"rho/beta=[{row['min_rho_beta']:.4f},{row['max_rho_beta']:.4f}] {dt:.0f}s",
            flush=True,
        )
    write(rows, out)


def study_combo(out: Path) -> None:
    """Attack the mean and the extremum with different tools -- section 7.5.4.

    Section 7.2.8 established that the two scores are close to independent: the
    front is ``max T_c > T_boil`` and the temperature scores are averages. The
    budget sweep improves the mean and costs the peak, monotonically.

    So pair the arm that wins the mean with the remedy that wins the peak. Section
    7.2.6's three-seed table separates them cleanly: Fourier features give `L_void`
    0.2070 against a 0.1630 base while taking 11.1% off the mean, and the modified
    MLP takes 16.1% off the mean while *halving* `L_void` to 0.0932. One raises the
    peak, the other lowers it -- which is what reducing spectral bias should do to
    an extremum, and what smoothing should do against it.

    This is not the section 7.2.6 combination. That paired two mean-winners and
    they did not compose. This pairs a mean-winner with a peak-winner, and the
    reason to expect anything is mechanical rather than additive.
    """
    traj = ruler()
    arms = (
        ("C+fourier", {"adam_iters": 300, "lbfgs_iters": 3000, "fourier_features": 32}),
        ("C+modified_mlp", {"adam_iters": 300, "lbfgs_iters": 3000, "modified_mlp": True}),
        ("A+fourier", {"adam_iters": 3000, "lbfgs_iters": 300, "fourier_features": 32}),
    )
    rows = [run_arm(traj, label, "torch", seed=seed, **kw) for seed in SEEDS for label, kw in arms]
    write(rows, out)
    mean_table(rows)


def study_regime(out: Path) -> None:
    """M9's reference half: the Objective 2 regime map.

    M9 asks for a parametric PINN on ``(zeta, t, alpha_void, tau_pump)`` whose
    regime *classification* matches a reference sweep. The PINN half is not
    attempted here: the single-point network misses its bar by 7-19x, and a
    parametric extension of a model that fails at one parameter value would be
    measuring nothing.

    The reference half stands on its own, and it attacks D49 directly. At the
    shipped defaults the sodium void worth is never sampled positive, because
    ``zeta_sign = 0.80`` sits below where boiling starts (``zeta`` ~ 0.96) -- so
    ``max rho/beta = 0`` is a statement about the parameter set, not about the
    transient. A sweep says whether **any** point in the family exercises the
    positive branch, which is what Objective 2 turns on.
    """
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    grid = [
        (float(w), float(t))
        for w in (0.0, 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3, 1.6e-2)
        for t in (1.0, 2.5, 5.0, 10.0, 20.0)
    ]
    print(f"{len(grid)} points, n_axial=80, closed loop", flush=True)
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(_regime_point, grid))
    write(rows, out)

    ok = [r for r in rows if "error" not in r]
    print(
        f"\n{'worth':>8s} {'tau':>6s} {'regime':>17s} {'peakP':>7s} "
        f"{'maxrho/b':>9s} {'exercised':>9s} {'onset':>8s} {'L_void':>7s}"
    )
    for r in ok:
        onset = r["onset_t"] if np.isfinite(r["onset_t"]) else float("nan")
        print(
            f"{r['void_worth_net']:8.1e} {r['tau_pump']:6.1f} {r['regime']:>17s} "
            f"{r['peak_power']:7.4f} {r['max_rho_beta']:+9.4f} "
            f"{r['void_exercised']!s:>9s} {onset:8.2f} {r['L_void_max']:7.4f}"
        )
    exercised = [r for r in ok if r["void_exercised"]]
    print(f"\npositive void worth exercised at {len(exercised)}/{len(ok)} points")
    if exercised:
        print(f"max rho/beta over those: {max(r['max_rho_beta'] for r in exercised):+.4f}")


def _regime_point(args: tuple[float, float]) -> dict:
    """Solve one closed-loop point and classify it. Top level, so it can be pickled."""
    worth, tau = args
    p = AxialParams(n_axial=80, void_worth_net=worth, tau_pump=tau)
    try:
        traj = solve_reference(p, n_out=241, feedback=True)
    except Exception as exc:  # noqa: BLE001 - a failed point is data, not a crash
        return {"void_worth_net": worth, "tau_pump": tau, "error": repr(exc)[:200]}
    t_on, z_on = traj.onset()
    max_rb = float(traj.peak_rho_over_beta)
    return {
        "void_worth_net": worth,
        "tau_pump": tau,
        "boils": bool(np.isfinite(t_on)),
        "onset_t": t_on,
        "onset_zeta": z_on,
        "L_void_max": float(traj.voided_length.max()),
        "peak_power": float(traj.power.max()),
        "min_power": float(traj.power.min()),
        "max_rho_beta": max_rb,
        "min_rho_beta": float(traj.rho.min() / traj._beta),  # noqa: SLF001
        "void_exercised": bool(traj.void_worth_is_exercised()),
        "peak_clad": float(traj.peak_clad),
        "t_final": float(traj.t[-1]),
        "regime": (
            "prompt-critical"
            if max_rb >= 1.0
            else "power-excursion"
            if traj.power.max() > 1.01
            else "boiling-bounded"
            if np.isfinite(t_on)
            else "no-boiling"
        ),
    }


def study_regime_sign(out: Path) -> None:
    """M9 part 2: sweep ``zeta_sign``, the parameter Objective 2 actually turns on.

    ``study_regime`` swept ``void_worth_net`` and ``tau_pump`` and found the
    positive void branch exercised at 0 of 30 points. That identifies the wrong
    knob rather than a null result: boiling starts at ``zeta`` ~ 0.96 and the worth
    changes sign at ``zeta_sign = 0.80``, so the voided region lies entirely inside
    the negative lobe and scaling the worth scales a term evaluated only where it
    is negative.

    ``zeta_sign`` is the sign-change height. Sweeping it against the worth is what
    answers Objective 2, and it does: excursions to 5.3x nominal appear.
    """
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    grid = [
        (float(sign), float(worth))
        for sign in (0.80, 0.90, 0.95, 0.97, 0.99, 0.995)
        for worth in (2.0e-3, 8.0e-3, 1.6e-2)
    ]
    print(f"{len(grid)} points, n_axial=80, closed loop", flush=True)
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(_sign_point, grid))
    write(rows, out)

    ok = [r for r in rows if "error" not in r]
    print(
        f"\n{'zeta_sign':>10s} {'worth':>8s} {'regime':>16s} {'peakP':>8s} "
        f"{'maxrho/b':>9s} {'exercised':>9s} {'L_void':>7s}"
    )
    for r in ok:
        print(
            f"{r['zeta_sign']:10.3f} {r['void_worth_net']:8.1e} {r['regime']:>16s} "
            f"{r['peak_power']:8.4f} {r['max_rho_beta']:+9.4f} "
            f"{r['void_exercised']!s:>9s} {r['L_void_max']:7.4f}"
        )
    hot = [r for r in ok if r["max_rho_beta"] > 0.5]
    if hot:
        print(
            f"\n*** {len(hot)} point(s) exceed rho/beta = 0.5. The prompt-jump closure"
            " (D-KIN-1) has a pole at 1; these are near its validity limit and are"
            " warnings, not predictions."
        )


def _sign_point(args: tuple[float, float]) -> dict:
    """Solve one ``(zeta_sign, void_worth_net)`` point. Top level, so it pickles."""
    sign, worth = args
    try:
        p = AxialParams(n_axial=80, zeta_sign=sign, void_worth_net=worth, delta_sign=0.02)
        traj = solve_reference(p, n_out=241, feedback=True)
    except Exception as exc:  # noqa: BLE001 - a failed point is data, not a crash
        return {"zeta_sign": sign, "void_worth_net": worth, "error": repr(exc)[:200]}
    t_on, _ = traj.onset()
    mx = float(traj.peak_rho_over_beta)
    return {
        "zeta_sign": sign,
        "void_worth_net": worth,
        "peak_power": float(traj.power.max()),
        "min_power": float(traj.power.min()),
        "max_rho_beta": mx,
        "void_exercised": bool(traj.void_worth_is_exercised()),
        "onset_t": t_on,
        "L_void_max": float(traj.voided_length.max()),
        "peak_clad": float(traj.peak_clad),
        "regime": (
            "prompt-critical"
            if mx >= 1.0
            else "power-excursion"
            if traj.power.max() > 1.01
            else "boiling-bounded"
            if np.isfinite(t_on)
            else "no-boiling"
        ),
    }


STUDIES = {
    "ruler": study_ruler,
    "horizon": study_horizon,
    "budget": study_budget,
    "optimizer": study_optimizer,
    "parity": study_parity,
    "plan-a": study_plan_a,
    "combo": study_combo,
    "regime": study_regime,
    "regime-sign": study_regime_sign,
}


def main() -> int:
    """Run one study and write its JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = ap.parse_args()
    out = args.out or Path(f"results/axial_study_{args.study}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    STUDIES[args.study](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
