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

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference
from pinn_sfr_transient.axial.scoring import relative_l2

if TYPE_CHECKING:
    from collections.abc import Callable

# The mesh every PINN table is scored against, and the mesh a study converges to.
RULER_N = 160
FINEST_N = 640
SEEDS = (0, 1, 2)
# Fourier ladder for the margin study. Extended until the trend turns: 32 -> 256
# improved T_s, L_void and the saturation margin monotonically on both backends,
# and a monotone trend with no measured end is an untested extrapolation.
#
# The embedding is `x -> [sin(2 pi B x), cos(2 pi B x)]`, so `n` features give a
# `2n`-wide input layer. At width 64 the first layer is `2n x 64` and the other
# four are `64 x 64`, so cost is roughly linear in `n` once `2n >> 64`: f1024 is
# about 3x f256. That is the price of finding the end of the ladder.
MARGIN_FEATURES = (32, 64, 128, 256, 512, 1024)
# Share of collocation placed on the saturation level set. 0.0 is the control:
# level-set sampling off, so the arm reduces to plain training at that budget.
FRONT_FRACS = (0.0, 0.05, 0.10, 0.25, 0.50)
# Adam x quasi-Newton grid, decades apart, so each stage's own saturation point
# is visible rather than inferred from a fixed-total split.
GRID_ITERS = (30, 300, 3000)
# Spatial-band multipliers for the anisotropic embedding. None is the control:
# isotropic, i.e. exactly the shipped default.
ANISO_SCALES = (None, 2.0, 4.0, 8.0)
# Idea 2: band multipliers on top of `fourier_scale`. `()` is the shipped single
# band and is the control. The ladder widens the span, it does not shift it -- 1.0
# is in every arm, so a gain is coverage of the extra scales and not a different
# one.
FOURIER_BANDS = ((), (1.0, 4.0), (1.0, 4.0, 16.0), (0.25, 1.0, 4.0, 16.0))
# Every study sweeps both backends. Two independent implementations agreeing is
# the strongest check this project has, and it is the reason the JAX twin exists
# (`docs/axial_nn.md` section 4) -- a result measured on one backend is a result
# about that backend.
BACKENDS = ("torch", "jax")
# Set by --only; filters arms so an extended ladder need not re-run measured points.
_ONLY: str | None = None
FIELDS = ("T_f", "T_cl", "T_s", "T_c")


# --- shared helpers ---------------------------------------------------------
def ruler(n: int = RULER_N, *, feedback: bool = False) -> Any:  # noqa: ANN401
    """Return the held-out reference the PINN is scored against."""
    return solve_reference(AxialParams(n_axial=n), n_out=241, feedback=feedback)


def score(fields: tuple, traj: Any) -> dict[str, float]:  # noqa: ANN401
    """Delegate to the one scorer, so a metric added there appears here too."""
    return relative_l2(fields, traj, AxialParams())


def train_torch(**kw: Any) -> Callable[[Any], tuple]:  # noqa: ANN401
    """Train the torch backend and return a predictor over the ruler's grid."""
    from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

    model = train(AxialParams(), AxialTrainConfig(log_every=10**9, **kw))
    return lambda traj: model.predict(traj.zeta, traj.t)


def train_jax(**kw: Any) -> Callable[[Any], tuple]:  # noqa: ANN401
    """Train the JAX backend and return a predictor over the ruler's grid.

    The **config must be threaded into `predict`**. The torch model carries its
    own `cfg`, so its evaluator cannot desync from its training; the JAX twin is
    functional and `predict(..., cfg=None)` silently falls back to
    `AxialTrainConfig()`. This discarded the cfg with `_`, so a JAX arm was
    trained under its arm's config and then **scored under the defaults** --
    `horizon()` reads `t_train_frac` from it, and the input width depends on
    `level_set_input` and `front_net`.

    It surfaced as a crash rather than as a wrong number only because
    `level_set_input` changes an array *shape*: the model was built with three
    inputs and the evaluator fed it two. A knob that changes a *value* -- which
    `t_train_frac` does -- would have produced a plausible, wrong score in
    silence. That is D67 exactly: a default reasserting itself where a measured
    value was intended.
    """
    from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

    cfg = pj.AxialTrainConfig(log_every=10**9, **kw)
    model, p, cfg = pj.train(AxialParams(), cfg, verbose=False)
    return lambda traj: pj.predict(model, p, traj.zeta, traj.t, cfg)


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
    if _ONLY is not None:
        wanted = [w.strip() for w in _ONLY.split(",") if w.strip()]
        specs = [(label, kw) for label, kw in specs if any(w in label for w in wanted)]
        print(f"--only {wanted}: {len(specs)} arm(s)", flush=True)
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
    rows = run_all(
        traj,
        [
            (
                f"t_train_frac={ttf} [{backend}]",
                {
                    "backend": backend,
                    "t_train_frac": ttf,
                    "seed": 0,
                    "adam_iters": 3000,
                    "lbfgs_iters": 300,
                },
            )
            for backend in BACKENDS
            for ttf in (0.25, 0.275, 0.30, 1.0)
        ],
        out,
    )
    mean_table(rows)


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
            (
                f"{label} [{backend}]",
                {"backend": backend, "adam_iters": adam, "lbfgs_iters": qn, "seed": seed},
            )
            for seed in SEEDS
            for backend in BACKENDS
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
            (
                f"{opt} [{backend}]",
                {
                    "backend": backend,
                    "optimizer": opt,
                    "seed": seed,
                    "adam_iters": 3000,
                    "lbfgs_iters": 300,
                },
            )
            for seed in SEEDS
            for backend in BACKENDS
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
    """Measure closed-loop power at three seeds -- section 7.4 has only one.

    **This runs the SHIPPED budget, not section 7.4's.** That section used 3000 Adam
    + 300 L-BFGS at 681 s; `AxialTrainConfig` ships 8000 + 500, which is what a
    reader running the model actually gets. The two are not comparable and the
    difference is large -- `L2(P)` 0.2497 against 0.1060 at seed 0 -- so any
    improvement here is a budget result until it is measured at a matched budget.

    Plan A takes the full horizon: with feedback the transient is self-limiting and
    completes 60 s inside the property range, so `t_train_frac` stays at 1.0.
    """
    # Say so before the long silence: the closed-loop reference at n = 160 plus the
    # first Plan A training is ~25 minutes before anything prints, and a study that
    # looks hung is a study someone kills.
    print("solving the closed-loop reference at n_axial=160 ...", flush=True)
    ref = ruler(feedback=True)
    print(
        f"reference: peak={ref.power.max():.4f} min={ref.power.min():.4f} "
        f"max rho/beta={ref.peak_rho_over_beta:+.4f}; "
        f"{len(SEEDS)} seeds x {len(BACKENDS)} backends",
        flush=True,
    )
    rows = []
    for seed in SEEDS:
        for backend in BACKENDS:
            t0 = time.perf_counter()
            power, rho = _plan_a_power(backend, seed, ref.t)
            dt = time.perf_counter() - t0
            row = {
                "arm": f"planA [{backend}]",
                "backend": backend,
                "seed": seed,
                "L2_P": float(np.linalg.norm(power - ref.power) / np.linalg.norm(ref.power)),
                "P0": float(power[0]),
                "peak_P": float(power.max()),
                "min_P": float(power.min()),
                "max_rho_beta": float(rho.max() / ref._beta),  # noqa: SLF001
                "min_rho_beta": float(rho.min() / ref._beta),  # noqa: SLF001
                "min_rho_beta_ref": float(ref.rho.min() / ref._beta),  # noqa: SLF001
                "sec": dt,
                "load1": os.getloadavg()[0],
                "omp": os.environ.get("OMP_NUM_THREADS", "unset"),
            }
            rows.append(row)
            print(
                f"[planA {backend:5s}] seed={seed} L2(P)={row['L2_P']:.4f} "
                f"P(0)={row['P0']:.6f} peak={row['peak_P']:.4f} min={row['min_P']:.4f} "
                f"rho/beta=[{row['min_rho_beta']:.4f},{row['max_rho_beta']:.4f}] {dt:.0f}s",
                flush=True,
            )
            write(rows, out)


def _plan_a_power(backend: str, seed: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train Plan A on one backend and return its power and reactivity traces.

    Plan A needs no horizon truncation: with feedback the transient is
    self-limiting and completes 60 s inside the property range.
    """
    if backend == "torch":
        from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

        model = train(
            AxialParams(),
            AxialTrainConfig(feedback=True, seed=seed, t_train_frac=1.0, log_every=10**9),
        )
        return model.predict_power(t)
    from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

    cfg = pj.AxialTrainConfig(feedback=True, seed=seed, t_train_frac=1.0, log_every=10**9)
    model, p_ax, out_cfg = pj.train(AxialParams(), cfg, verbose=False)
    return pj.predict_power(model, p_ax, t, out_cfg)


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


def study_default(out: Path) -> None:
    """Re-baseline: the shipped default against the best known configuration.

    Section 0.5 holds `C + fourier` back from being the default because every
    published table was measured on the old one. This measures both, on both
    backends, at three seeds, with the M4 onset metrics that nothing reported
    before -- which is the re-baseline that unblocks the change.
    """
    traj = ruler()
    arms = (
        ("default", {}),
        ("C+fourier", {"adam_iters": 300, "lbfgs_iters": 3000, "fourier_features": 32}),
    )
    rows = run_all(
        traj,
        [
            (f"{label} [{backend}]", {"backend": backend, "seed": seed, **kw})
            for seed in SEEDS
            for backend in BACKENDS
            for label, kw in arms
        ],
        out,
    )
    mean_table(rows)
    print("\nM4 acceptance: onset within 0.5 s and one cell (0.00625 at n = 160)")
    for r in rows:
        ok_t = r["onset_t_err_s"] <= 0.5
        ok_z = r["onset_zeta_err"] <= 0.00625
        print(
            f"  {r['arm']:24s} seed={r['seed']} onset_t_err={r['onset_t_err_s']:6.2f} s "
            f"{'PASS' if ok_t else 'FAIL':4s}   onset_zeta_err={r['onset_zeta_err']:.5f} "
            f"{'PASS' if ok_z else 'FAIL'}"
        )


def study_margin(out: Path) -> None:
    """Raise the saturation margin deliberately -- nothing has ever targeted it.

    Every fragility in `docs/axial_nn.md` traces to one number: the network's peak
    `T_c` clears saturation by 7.6-20.5 K out of a ~590 K range. Section 7.2.8
    showed the front IS that inequality, and section 7.5.4 showed the arms that
    keep margin are the arms whose `L_void` is stable across seeds.

    Fourier features raised it as a side effect. These arms aim at it:

    * more Fourier features -- if reducing spectral bias raises the peak, more of
      it should raise it further, until the extra capacity costs the mean;
    * a narrower `dT_smooth` -- the logistic width is 2 K, so the closure smears
      the onset over a band comparable to the entire margin. Narrowing it sharpens
      the front but stiffens the gradient, which is the trade section 12.4 names.

    Success is `margin_K` at **every** seed, not the mean -- a margin that is large
    on average and negative once is the `A + fourier` failure of section 7.5.4.
    """
    traj = ruler()
    base = {"adam_iters": 300, "lbfgs_iters": 3000}
    arms = tuple((f"f{n}", {**base, "fourier_features": n}) for n in MARGIN_FEATURES)
    rows = run_all(
        traj,
        [
            (f"{label} [{backend}]", {"backend": backend, "seed": seed, **kw})
            for seed in SEEDS
            for backend in BACKENDS
            for label, kw in arms
        ],
        out,
    )
    mean_table(rows)
    print("\nmargin per arm (the quantity this study targets):")
    for label in dict.fromkeys(r["arm"] for r in rows):
        m = [r["margin_K"] for r in rows if r["arm"] == label]
        print(
            f"  {label:24s} min {min(m):+6.1f} K   mean {sum(m) / len(m):+6.1f} K   "
            f"{'ALL POSITIVE' if min(m) > 0 else 'ONE OR MORE BELOW SATURATION'}"
        )


def study_scaling(out: Path) -> None:
    """Measure whether more optimisation still converges -- section 7.5.5.

    Does the mean keep improving with budget, or has it stopped?

    Twelve published tables use 3000 Adam + 300 L-BFGS; the config ships 8000 + 500
    and nothing had ever run it (section 7.2.9). The first row measured said the
    shipped budget has a **better mean** and **no front**, which raises the obvious
    question: is the mean still improving with budget, or has it converged and the
    extra iterations only smooth the peak away?

    A ladder at a fixed 10:1 Adam-to-quasi-Newton ratio, so budget is the only
    thing varying. Both backends, because a convergence claim about one optimiser
    stack is a claim about that stack.

    Watch `margin_K` alongside `T_s`: section 7.2.8 predicts they move in opposite
    directions, and if they do, "more epochs" has a ceiling set by the front rather
    than by the mean.
    """
    traj = ruler()
    ladder = (
        ("3k/300 (published)", 3000, 300),
        ("8k/500 (shipped)", 8000, 500),
        ("16k/1000", 16000, 1000),
    )
    rows = run_all(
        traj,
        [
            (
                f"{label} [{backend}]",
                {"backend": backend, "adam_iters": adam, "lbfgs_iters": qn, "seed": seed},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for label, adam, qn in ladder
        ],
        out,
    )
    mean_table(rows)
    print("\nconvergence: does the mean keep improving, and what does the peak do?")
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        ts = [r["T_s"] for r in v]
        mg = [r["margin_K"] for r in v]
        print(
            f"  {label:26s} T_s {sum(ts) / len(ts):.4f} [{min(ts):.4f}-{max(ts):.4f}]   "
            f"margin min {min(mg):+6.1f} K   "
            f"{'front on every seed' if min(mg) > 0 else 'FRONT LOST ON >=1 SEED'}"
        )


def study_levelset(out: Path) -> None:
    """Fix the measure: sample collocation on the saturation level set -- section 7.5.6.

    The loss is a **mean over the domain** and the front occupies a few percent of
    the channel, so the front contributes a few percent of the objective however
    long training runs. That is why 8000+500 iterations beat 3000+300 by 47% on
    `T_s` and lose the front entirely (section 7.5.5): more optimisation converges
    more precisely to a minimiser whose peak is wrong.

    RAR cannot fix it -- after the algebraic closure the residual is small
    everywhere, including across the front, so residual-magnitude sampling has no
    signal. Sampling the level set `T_c = T_sat + dT_sup` does, and needs no
    front-position network because under D-TH-3 the front IS that level set.

    Run at the budget that loses the front, so the question is direct: does
    front-aware sampling let more optimisation help the front instead of costing it?
    """
    traj = ruler()
    big = {"adam_iters": 8000, "lbfgs_iters": 500}
    arms = (
        ("8k/500 plain", big),
        ("8k/500 +levelset", {**big, "front_level_set": True, "front_frac": 0.25}),
        (
            "8k/500 +levelset+f128",
            {**big, "front_level_set": True, "front_frac": 0.25, "fourier_features": 128},
        ),
    )
    rows = run_all(
        traj,
        [
            (f"{label} [{backend}]", {"backend": backend, "seed": seed, **kw})
            for seed in SEEDS
            for backend in BACKENDS
            for label, kw in arms
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes front-aware sampling let a large budget keep the front?")
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        mg = [r["margin_K"] for r in v]
        ts = [r["T_s"] for r in v]
        print(
            f"  {label:28s} T_s {sum(ts) / len(ts):.4f}   margin min {min(mg):+6.1f} K   "
            f"{'FRONT ON EVERY SEED' if min(mg) > 0 else 'front lost on >=1 seed'}"
        )


def study_frontfrac(out: Path) -> None:
    """Sweep how much collocation goes to the front -- section 7.5.9.

    Section 7.5.6 showed level-set sampling does what Annex C predicts: at a budget
    that loses the front entirely, diverting collocation to the saturation level set
    brings `max alpha` back from 0.735 to 0.998 and triples `L_void` -- and costs the
    mean, 0.0365 to 0.0482 on `T_s`.

    That is the measure controlling the trade, which is the mechanism. But 25% was
    picked as a plausible number, not measured, and it is the only free parameter in
    the fix. If the trade is smooth in `front_frac` there is a setting that buys the
    front for less mean than 25% does; if it is not, the fix is blunter than the
    diagnosis suggests and that is worth knowing too.

    Judged on both quantities at once: `margin_K` at every seed AND `T_s`. An arm
    that wins the front by giving up the mean is the `A + fourier` trade of section
    7.5.4 again, not progress.
    """
    traj = ruler()
    base = {"adam_iters": 8000, "lbfgs_iters": 500, "front_level_set": True}
    rows = run_all(
        traj,
        [
            (
                f"front_frac={ff} [{backend}]",
                {"backend": backend, "seed": seed, "front_frac": ff, **base},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for ff in FRONT_FRACS
        ],
        out,
    )
    mean_table(rows)
    print("\nthe trade, per arm -- does a smaller share buy the front for less mean?")
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        mg = [r["margin_K"] for r in v]
        ts = [r["T_s"] for r in v]
        lv = [r["L_void_max"] for r in v]
        print(
            f"  {label:28s} T_s {sum(ts) / len(ts):.4f}   L_void {sum(lv) / len(lv):.4f}   "
            f"margin min {min(mg):+6.1f} K"
        )


def study_capacity_optimiser(out: Path) -> None:
    """Test whether the JAX capacity plateau is `optax.lbfgs` -- section 7.5.10.

    Capacity helps torch and does nothing for JAX. `T_s` at seed 0, torch against
    jax: 0.0315/0.0343 at f32, 0.0285/0.0379 at f128, 0.0251/0.0368 at f256, and
    **0.0148/0.0336 at f512** -- a ratio growing 1.09x -> 2.27x. Both backends run
    identical architectures and residuals, verified to 1e-14, so the equations are
    not the cause.

    Section 7.3.2 already found the framework L-BFGS to be the *entire* backend gap
    at the shipped configuration: 1.168 with each framework's own optimiser and
    0.999 with one shared implementation. The hypothesis follows directly --
    `optax.lbfgs` cannot exploit the extra capacity and `torch.optim.LBFGS` can.

    One arm decides it. If JAX tracks torch under the shared optimiser, the plateau
    is the optimiser; if it does not, the gap is in the sampler or the float64
    kernels and is genuinely unexplained.

    This is a sharper question than another ladder rung, which only measures how far
    torch goes -- and torch going further is already known.
    """
    traj = ruler()
    big = {"adam_iters": 300, "lbfgs_iters": 3000, "fourier_features": 512}
    rows = run_all(
        traj,
        [
            (f"f512 {opt} [{backend}]", {"backend": backend, "seed": seed, "optimizer": opt, **big})
            for seed in SEEDS
            for backend in BACKENDS
            for opt in ("lbfgs", "lbfgs-shared")
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes a shared optimiser let JAX use the capacity?")
    for opt in ("lbfgs", "lbfgs-shared"):
        t = [r["T_s"] for r in rows if r["arm"] == f"f512 {opt} [torch]"]
        j = [r["T_s"] for r in rows if r["arm"] == f"f512 {opt} [jax]"]
        if t and j:
            print(
                f"  {opt:14s} torch {sum(t) / len(t):.4f}   jax {sum(j) / len(j):.4f}   "
                f"ratio {sum(j) / len(j) / (sum(t) / len(t)):.2f}x"
            )


def study_grid(out: Path) -> None:
    """Cross Adam against quasi-Newton independently -- section 7.5.11.

    Section 7.5.3 asked "which split of a fixed 3300 iterations", and section 7.5.5
    asked "does a bigger total help" at a fixed 10:1 ratio. Neither asks how many
    iterations each stage actually needs, and the answer to that has been inherited
    rather than measured: `300 Adam + 3000 L-BFGS` reached "best known" by winning a
    fixed-total sweep, and two later studies took it as their base.

    A full cross removes the constraint. `fourier_features` is pinned at 128 so the
    only thing varying is where the iterations go, and the capacity rung is one that
    forms the front on every seed (section 7.5.8).

    Watch two things the earlier sweeps could not separate: whether the quasi-Newton
    stage saturates independently of Adam, and whether the front survives large Adam
    budgets when the quasi-Newton stage is also large -- section 7.5.5 lost the front
    at 8k/500 and could not say which half was responsible.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"adam{a}/qn{q} [{backend}]",
                {
                    "backend": backend,
                    "seed": seed,
                    "adam_iters": a,
                    "lbfgs_iters": q,
                    "fourier_features": 128,
                },
            )
            for seed in SEEDS
            for backend in BACKENDS
            for a in GRID_ITERS
            for q in GRID_ITERS
        ],
        out,
    )
    mean_table(rows)
    print("\nT_s surface (rows Adam, cols quasi-Newton), and the worst-seed margin:")
    for backend in BACKENDS:
        print(f"\n  [{backend}]")
        header = "         " + "".join(f"{q:>12d}" for q in GRID_ITERS)
        print(header)
        for a in GRID_ITERS:
            cells = []
            for q in GRID_ITERS:
                v = [r for r in rows if r["arm"] == f"adam{a}/qn{q} [{backend}]"]
                if not v:
                    cells.append(f"{'--':>12s}")
                    continue
                ts = sum(r["T_s"] for r in v) / len(v)
                mg = min(r["margin_K"] for r in v)
                cells.append(f"{ts:8.4f}{'*' if mg > 0 else '!':>4s}")
            print(f"  {a:6d} " + "".join(cells))
    print("\n  * front on every seed   ! front lost on at least one")


def study_aniso(out: Path) -> None:
    """Idea 1 in isolation: anisotropic Fourier bandwidth -- section 7.5.12.

    The embedding uses one `scale` for every input, which assumes the solution's
    frequency content is isotropic. It is not: the front is a near-discontinuity in
    `zeta` and smooth in `t`. `fourier_scale_zeta` multiplies the spatial band only.

    Isolated: the base is the shipped default and **only** this knob moves, so the
    effect cannot be confounded with capacity or budget. 1.0 is the control and
    reproduces the default exactly.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"zeta_scale={z or 1.0} [{backend}]",
                {"backend": backend, "seed": seed, "fourier_scale_zeta": z},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for z in ANISO_SCALES
        ],
        out,
    )
    mean_table(rows)
    print("\nis a wider spatial band better, and does the margin hold on every seed?")
    _arm_summary(rows)


def _arm_summary(rows: list[dict]) -> None:
    """Per-arm means plus the **worst** margin over seeds.

    The mean of the margin is the wrong statistic: the front is an inequality
    (`max T_c > T_sat + dT_sup`, section 7.2.8), so an arm that forms a front on
    two seeds out of three has not formed a front. Report the minimum.
    """
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        mg = [r["margin_K"] for r in v]
        print(
            f"  {label:30s} T_s {sum(r['T_s'] for r in v) / len(v):.4f}   "
            f"L_void {sum(r['L_void_max'] for r in v) / len(v):.4f}   "
            f"margin min {min(mg):+6.1f} K   "
            f"{'front every seed' if min(mg) > 0 else 'FRONT LOST'}"
        )


def study_lsinput(out: Path) -> None:
    """Idea 3 in isolation: the level-set coordinate as a network input -- section 7.5.13.

    The front is at a fixed value of `phi = (T_c - T_sat - dT_sup) / dT`, not at a
    fixed `zeta`; in `(zeta, t)` its location moves and the network must learn that
    motion. Feeding `phi` gives it a coordinate in which the front is *stationary*,
    which is the same trick as a co-moving frame.

    `phi` is built from the network's own `T_c`, so it comes from a bootstrap pass
    with `phi = 0` -- and **without** `stop_gradient`, so the residual keeps the
    term through `phi`.

    Isolated: only `level_set_input` moves. It is a different mechanism from
    `levelset` (which moves the *sampling* measure, section 7.5.6) and from the
    `front_net` level set (which parameterises the interface); confounding the
    three is how a mechanism gets credit for another's effect.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"level_set_input={on} [{backend}]",
                {"backend": backend, "seed": seed, "level_set_input": on},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for on in (False, True)
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes a front-stationary coordinate buy anything, on every seed?")
    _arm_summary(rows)


def study_bands(out: Path) -> None:
    """Idea 2 in isolation: multi-scale Fourier bands -- section 7.5.14.

    One `fourier_scale` commits to one frequency. The solution has a smooth bulk
    and a near-discontinuous front, so any single band is wrong for one of them:
    too low and the front is smeared (which is exactly how the front is lost),
    too high and the bulk is noisy. Bands cover several at once.

    The feature *total* is held at the default, so each band gets `1/n` of the
    width: this trades resolution within a band for coverage across bands at
    **fixed cost**, and any gain is the trade paying off, not extra capacity. `()`
    is the control and reproduces the shipped default exactly (asserted by test).
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"bands={list(b) or 'single'} [{backend}]",
                {"backend": backend, "seed": seed, "fourier_bands": b},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for b in FOURIER_BANDS
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes covering more scales at fixed width beat picking one?")
    _arm_summary(rows)


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
    "default": study_default,
    "margin": study_margin,
    "scaling": study_scaling,
    "levelset": study_levelset,
    "frontfrac": study_frontfrac,
    "capacity-optimiser": study_capacity_optimiser,
    "grid": study_grid,
    "aniso": study_aniso,
    "lsinput": study_lsinput,
    "bands": study_bands,
}


def main() -> int:
    """Run one study and write its JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    ap.add_argument(
        "--only",
        default=None,
        help="run only arms whose label contains one of these comma-separated "
        "substrings, so a ladder can be extended without re-running what is "
        "already measured",
    )
    args = ap.parse_args()
    out = args.out or Path(f"results/axial_study_{args.study}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    global _ONLY  # noqa: PLW0603 - one filter for the whole run
    _ONLY = args.only
    STUDIES[args.study](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
