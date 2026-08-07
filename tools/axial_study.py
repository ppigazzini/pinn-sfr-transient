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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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
    row = score(predict(traj), traj) | {"arm": label, "backend": backend, "sec": dt, **kw}
    print(
        f"{label:24s} {backend:5s} "
        + " ".join(f"{k}={row[k]:.4f}" for k in FIELDS)
        + f" maxA={row['max_alpha']:.4f} L_void={row['L_void_max']:.4f} {dt:.0f}s",
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
    rows = [
        run_arm(traj, label, "torch", adam_iters=adam, lbfgs_iters=qn, seed=seed)
        for seed in SEEDS
        for label, adam, qn in arms
    ]
    write(rows, out)
    mean_table(rows)


def study_optimizer(out: Path) -> None:
    """Compare self-scaled BFGS against L-BFGS at the shipped split -- section 7.5."""
    traj = ruler()
    rows = [
        run_arm(traj, opt, "torch", optimizer=opt, seed=seed, adam_iters=3000, lbfgs_iters=300)
        for seed in SEEDS
        for opt in ("lbfgs", "lbfgs-shared", "ssbfgs")
    ]
    write(rows, out)
    mean_table(rows)


def study_parity(out: Path) -> None:
    """Test whether the 21% backend gap on `T_s`/`T_c` is the optimiser -- section 7.3.2.

    `lbfgs-shared` runs this repository's own L-BFGS in both backends, removing the
    last component that is not shared source. The residuals are already known
    identical at identical parameters, so if the gap survives this it is neither
    the equations nor the optimiser implementation.
    """
    traj = ruler()
    rows = [
        run_arm(
            traj,
            f"{backend}/{opt}",
            backend,
            optimizer=opt,
            seed=seed,
            adam_iters=3000,
            lbfgs_iters=300,
        )
        for seed in SEEDS
        for backend in ("torch", "jax")
        for opt in ("lbfgs", "lbfgs-shared")
    ]
    write(rows, out)
    mean_table(rows)


def study_plan_a(out: Path) -> None:
    """Measure closed-loop power at three seeds -- section 7.4 has only one."""
    from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

    ref = ruler(feedback=True)
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
    print(
        f"\nreference: peak={ref.power.max():.4f} min={ref.power.min():.4f} "
        f"max rho/beta={ref.peak_rho_over_beta:+.4f}"
    )
    write(rows, out)


STUDIES = {
    "ruler": study_ruler,
    "horizon": study_horizon,
    "budget": study_budget,
    "optimizer": study_optimizer,
    "parity": study_parity,
    "plan-a": study_plan_a,
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
