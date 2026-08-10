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

**Pin the thread budget on BOTH backends.** `OMP_NUM_THREADS` binds torch and is
ignored by JAX: XLA's CPU backend sizes its own Eigen pool from
`hardware_concurrency()`, so a JAX arm nominally at 8 threads was measured creating
**291**. Thread count changes float reduction order, so this changes answers and not
only timings -- measured, at ~3 ulp:

    48 cores  -> T_c_sum = 31802.507612040135
    8 cores   -> T_c_sum = 31802.507612040157

`--cpu-block` fixes it by pinning CPU affinity, which JAX does obey. The core
*count* is what matters, not which cores: two different blocks of 8 give bitwise
identical results, and repeating a block reproduces exactly. So concurrent studies
take different blocks and stay comparable.

    OMP_NUM_THREADS=8 uv run python tools/axial_study.py budget --cpu-block 0
    OMP_NUM_THREADS=8 uv run python tools/axial_study.py margin --cpu-block 1

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
# Task 1: does the quasi-Newton axis end? Section 7.5.11 is monotone over two
# decades with no interior optimum, and by this project's own rule an unterminated
# monotone trend is an extrapolation. Kiyani et al. run 30000 against this model's
# 3000. Adam is 0 or 30 because the same surface showed the Adam axis flat once the
# quasi-Newton stage is funded -- and 0 has never been run at all.
QN_LADDER = (3000, 10000, 30000)
ADAM_LADDER = (0, 30)
# Adam with the quasi-Newton stage switched OFF entirely, at OUR budget and OUR learning
# rate (1e-3, which is also the paper's base LR) so the arms sit beside everything else
# this project has measured. 30000 Adam iterations is the shipped default's quasi-Newton
# count, so `adam30000/qn0` is the shipped budget spent entirely on Adam.
ADAM_ONLY_ITERS = 30000
# The paper's 1D configuration (its Allen-Cahn row): 9 dyadic levels from 2 to 512, 14
# features each. About 14300 grid parameters, the same order as our f256 embedding.
BEIGNET_1D = {"beignet_levels": 9, "beignet_features": 14, "beignet_base": 2}
# Spatial-band multipliers for the anisotropic embedding. None is the control:
# isotropic, i.e. exactly the shipped default.
ANISO_SCALES = (None, 2.0, 4.0, 8.0)
# Idea 2: band multipliers on top of `fourier_scale`. `()` is the shipped single
# band and is the control. The ladder widens the span, it does not shift it -- 1.0
# is in every arm, so a gain is coverage of the extra scales and not a different
# one.
FOURIER_BANDS = ((), (1.0, 4.0), (1.0, 4.0, 16.0), (0.25, 1.0, 4.0, 16.0))
# Laplace rates straight from the manual, in 1/s: the six delayed-precursor decay
# constants and the pump coast-down. Not tuned and not swept -- this is the
# known-shape case, so the embedding is a fit and the rates are the physics.
LAPLACE_RATES = (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01, 0.2)
# Every study sweeps both backends. Two independent implementations agreeing is
# the strongest check this project has, and it is the reason the JAX twin exists
# (`docs/axial_nn.md` section 4) -- a result measured on one backend is a result
# about that backend.
# JAX first: at matched thread count and matched curvature memory it is 4.4x
# faster (§7.5.19) and within 1.08x on accuracy at f512, so it is the backend a
# sweep should lead with. Torch remains a full first-class arm -- two independent
# implementations agreeing is the strongest check this project has.
BACKENDS = ("jax", "torch")
# Set by --only; filters arms so an extended ladder need not re-run measured points.
_ONLY: str | None = None
# Set by --lbfgs-history; overrides the quasi-Newton curvature memory on every arm.
# It is a flag rather than an edited default because the shipped default is what
# every torch table was measured at, and a re-run at a different memory has to be
# distinguishable from the rows it is being compared against.
_HISTORY: int | None = None
# Set by --adam-iters / --lbfgs-iters; overrides the training budget on arms that do not
# set one themselves. A ladder like `qnladder` sets its own budget per arm and must NOT be
# overridden — that is the whole ladder — so these apply only where the key is absent.
#
# They exist because a sub-command's budget is whatever the config default happens to be
# on the day it runs, and that default has moved: `bakeoff`'s docstring says "the shipped
# 300 / 3000" and the shipped value is now 30 / 30000, which is a hundredfold more
# quasi-Newton and roughly three hundred core-days for its 24 arms. A study whose cost
# silently changed by 100x when an unrelated default moved is not reproducible; naming the
# budget on the command line makes it so, and the rows record it either way.
_ADAM: int | None = None
_QN: int | None = None
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


def pin_cpu_block(block: int, n: int) -> tuple[int, ...]:
    """Pin this process to ``n`` cores starting at ``block * n``, and return them.

    Must run before JAX is imported. `OMP_NUM_THREADS` does not bind XLA's CPU
    backend -- it sizes its own pool from `hardware_concurrency()` -- and affinity
    is what it does obey. Measured: 291 threads unpinned, 56 pinned to 8 cores.

    This is a correctness knob, not a performance one. Thread count changes float
    reduction order, so an unpinned JAX arm is not bitwise reproducible; pinned, it
    is, and two *different* blocks of the same size agree bitwise as well.
    """
    total = os.cpu_count() or 1
    cores = tuple((block * n + i) % total for i in range(n))
    os.sched_setaffinity(0, set(cores))
    return cores


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
        # The affinity size is the budget that actually binds both backends; `omp`
        # binds only torch. A row without this cannot be compared on wall-clock.
        "cpus": len(os.sched_getaffinity(0)),
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


def _selected(label: str, wanted: list[str]) -> bool:
    """Return whether ``label`` is picked by any ``--only`` token, matched at a boundary.

    The first version was ``any(w in label for w in wanted)`` — an unanchored substring
    test — and it silently selected an extra arm whenever one arm's name contained
    another's. `nofourier adam30000` matched `ademamix-nofourier adam30000` too, so a run
    that was meant to be one arm quietly became two, and the fix attempted at the time was
    to rename the arms rather than to fix the test. Renaming makes the collision go away
    for exactly as long as nobody adds a third arm.

    A token now matches only if it is the whole label, the whole label without its
    ``[backend]`` suffix, or a prefix of that ending at a separator — so `nofourier`
    matches `nofourier adam30000/qn0` but never `ademamix-nofourier ...`.
    """
    bare = label.split(" [", maxsplit=1)[0]
    for w in wanted:
        if w in (label, bare):
            return True
        if bare.startswith(w) and (len(bare) == len(w) or bare[len(w)] in " /"):
            return True
    return False


def run_all(
    traj: Any,  # noqa: ANN401
    specs: list[tuple[str, dict]],
    out: Path,
    backend: str = "jax",  # see BACKENDS: faster at equal threads, equal at equal memory
) -> list[dict]:
    """Run every spec, writing after each so a killed study keeps what it measured.

    These studies run for hours. Collecting rows and writing once at the end means
    a machine reboot, an OOM or a stray kill loses everything -- and this project
    has already lost an ablation that way ("the ablation run was killed before its
    three configurations finished", section 7.6).
    """
    if _ONLY is not None:
        wanted = [w.strip() for w in _ONLY.split(",") if w.strip()]
        specs = [(label, kw) for label, kw in specs if _selected(label, wanted)]
        # Print WHICH arms, not just how many. A bare count is why an unanchored
        # substring filter silently ran a second arm for ten minutes before anyone
        # noticed: "1 arm(s)" and "2 arm(s)" look equally plausible in a log.
        print(f"--only {wanted}: {len(specs)} arm(s)", flush=True)
        for label, _ in specs:
            print(f"    selected: {label}", flush=True)
        if not specs:
            print("    (nothing matched -- check the arm names above the filter)", flush=True)
    rows: list[dict] = []
    for label, kw in specs:
        if _HISTORY is not None:
            kw["lbfgs_history"] = _HISTORY
        # setdefault, not assignment: an arm that names its own budget is the study.
        if _ADAM is not None:
            kw.setdefault("adam_iters", _ADAM)
        if _QN is not None:
            kw.setdefault("lbfgs_iters", _QN)
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
            for opt in ("lbfgs", "lbfgs-shared", "ssbfgs", "ssbroyden")
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


def study_onset(out: Path) -> None:
    """Put onset in the objective and read it by tangency -- section 7.5.16.

    M4 asks for onset within 0.5 s and one cell, and it has never moved. Two
    reasons, and only one is the network's.

    **It was never in the objective.** Onset was read off a trained field
    afterwards; every knob this tool sweeps was ranked on a mean (relative `L2`)
    and later on an extremum (the margin). Neither is a *position*.

    **The readout was square-root conditioned.** Onset happens at the maximum of
    `T_c`, so near it `T_c ~ T_boil - kappa (zeta - zeta*)^2 / 2` and a field error
    `eps` displaces a threshold crossing by `sqrt(2 eps / kappa)`. Against this
    model's reference, `kappa = 1066 K` per unit `zeta` squared and `eps = 3.4 K`
    at the best published accuracy.

    The head replaces both with the tangency conditions -- `T_c = T_boil` and
    `dT_c/dzeta = 0` at one point -- whose position sensitivity is
    `delta(slope)/kappa`: linear, and divided by a *large* number.

    Isolated, and the isolation matters more here than usual: this is the FOURTH
    distinct use of the level set in this model, after the sampling measure
    (7.5.6), the front network's interface, and `level_set_input` (7.5.13).
    Overlapping mechanisms are how one collects another's credit.

    Both readouts are scored on every arm, so the control arm alone answers
    whether the *readout* helps, independently of whether the *head* does.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"onset_head={on} [{backend}]",
                {"backend": backend, "seed": seed, "onset_head": on},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for on in (False, True)
        ],
        out,
    )
    mean_table(rows)
    print("\nM4 is onset within 0.5 s AND one cell (0.00625). Threshold vs tangency readout:")
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        thr = [
            r["onset_zeta_err"] / 0.00625 for r in v if r["onset_zeta_err"] == r["onset_zeta_err"]
        ]
        tan = [
            r["onset_zeta_err_tan"] / 0.00625
            for r in v
            if r.get("onset_zeta_err_tan") == r.get("onset_zeta_err_tan")
        ]
        t_err = [r["onset_t_err_s"] for r in v if r["onset_t_err_s"] == r["onset_t_err_s"]]
        print(
            f"  {label:28s} n={len(v)}  "
            f"t_err {(sum(t_err) / len(t_err) if t_err else float('nan')):.2f} s   "
            f"zeta threshold {(sum(thr) / len(thr) if thr else float('nan')):5.1f} cells   "
            f"zeta tangency {(sum(tan) / len(tan) if tan else float('nan')):5.1f} cells"
        )


def study_laplace(out: Path) -> None:
    """Laplace embedding -- alone, summed with Fourier, multiplied by it -- section 7.5.18.

    A Fourier basis is oscillatory; this transient is built out of DECAY. The pump
    coasts down at `1/tau_pump` and six precursor groups span 0.0124 to 3.01 per
    second, a 243x range. Approximating `exp(-0.2 t)` over the window out of sines
    costs many terms and still misses the tail; one exponential does it exactly.

    The complementarity is not arbitrary: the oscillatory structure is in `zeta` and
    the exponential structure is in `t`, which is the anisotropy section 7.5.12
    measured on the bandwidth -- reached here from the physics instead of a sweep.

    Three arms, and the choice between them is the hypothesis:

    * `alone` -- rates fixed from the manual, no Fourier. The known-shape case, where
      the embedding is a **fit** rather than a basis, and should need few features.
    * `sum` -- concatenated blocks, right when the solution is a **superposition** of
      an oscillatory part and a decaying one.
    * `product` -- each Fourier group modulated by one rate, a damped sinusoid. Right
      when the two are **coupled**, which is what a transient excursion is rather
      than a sum of one of each. The feature total is unchanged, so any gain is the
      coupling and not capacity.

    `laplace_rates=()` is the control and reproduces the shipped default exactly.

    The honest prior is that this may be inert: the ansatz is already multiplicative,
    `theta = theta_0 exp(t_hat N)`, so a decaying mode is representable today. The
    claim is that it becomes *easier*, not newly possible -- which is why the control
    arm carries the weight and why 7.5.13, the last re-parameterisation tried, was
    measured inert.
    """
    traj = ruler()
    arms = [
        ("off", {}),
        *[
            (m, {"laplace_rates": LAPLACE_RATES, "laplace_mode": m})
            for m in ("alone", "sum", "product")
        ],
    ]
    rows = run_all(
        traj,
        [
            (f"laplace={label} [{backend}]", {"backend": backend, "seed": seed, **kw})
            for seed in SEEDS
            for backend in BACKENDS
            for label, kw in arms
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes an exponential basis help a transient built out of decay?")
    _arm_summary(rows)


def study_qnladder(out: Path) -> None:
    """Push the quasi-Newton axis until it ends, or show that it does not -- section 7.5.20.

    The most consequential open question in this project. Section 7.5.11 measured the
    quasi-Newton axis monotone across two decades on both backends, with no interior
    optimum, and section 7.5.8's own rule says an unterminated monotone trend is an
    untested extrapolation. Kiyani et al. run Adam[1000] + SSBroyden[**30000**]
    against this model's 300 + 3000.

    **If the axis keeps paying, `T_s = 0.02` is a budget limit and not the model's**,
    and every "the formulation cannot do better" reading in this document is
    premature. If it flattens, the formulation really is the ceiling and the next
    move is the optimiser rather than the budget.

    Adam is 0 or 30 because that surface also showed the Adam axis flat once the
    quasi-Newton stage is funded -- so this asks the sharper question, whether Adam
    is needed **at all**. `adam_iters = 0` had never been run before this study, and
    it crashed the JAX backend when it was: the collocation set was drawn inside the
    Adam loop, so with no loop the polish had no points.

    Cost is the reason this is JAX-only to begin with: at 4.4x torch (section 7.5.19)
    the 30000-iteration arms are hours rather than most of a day. Torch confirms
    whatever this finds.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"adam{a}/qn{q} [{backend}]",
                {"backend": backend, "seed": seed, "adam_iters": a, "lbfgs_iters": q},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for a in ADAM_LADDER
            for q in QN_LADDER
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes more quasi-Newton keep paying, and is Adam needed at all?")
    _arm_summary(rows)


def study_bakeoff(out: Path) -> None:
    """Re-run the optimiser bake-off with the quasi-Newton stage FUNDED -- section 7.5.22.

    `optimizer` runs at 3000 Adam / 300 quasi-Newton, the old diagonal budget, and
    that is the wrong place to compare quasi-Newton methods. Section 7.5.11 measured
    the quasi-Newton axis as the only one that moves the front and the Adam axis as
    flat once it is funded -- so a bake-off at `qn = 300` compares four methods in
    the regime where none of them is given enough iterations to matter. Every arm
    there lands at `T_s ~ 0.05` with a +2 K margin, against the shipped default's
    0.029 and +26 K, which is the symptom.

    This runs the same four at the shipped 300 / 3000. It is a separate sub-command
    rather than an edit to `optimizer`, so the earlier table stays reproducible by
    the command that produced it.

    The prior from the starved run is that plain L-BFGS wins, self-scaling costs 33%
    and the Broyden class costs 44% at 4.3x the wall-clock. If that survives a funded
    quasi-Newton stage it is a real negative result and roadmap item D.4 -- natural
    gradient -- becomes the next thing to try rather than another quasi-Newton
    variant. If it reverses, the earlier bake-off was measured in the wrong regime and
    should be withdrawn.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"{opt} [{backend}]",
                {"backend": backend, "seed": seed, "optimizer": opt},
            )
            for seed in SEEDS
            for backend in BACKENDS
            for opt in ("lbfgs", "lbfgs-shared", "ssbfgs", "ssbroyden")
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes any quasi-Newton variant beat plain L-BFGS once the stage is funded?")
    _arm_summary(rows)


def study_bandsbudget(out: Path) -> None:
    """Are multi-scale bands and a funded quasi-Newton stage the SAME gain -- section 7.5.24.

    Two results sit on the shelf that this project has never run together.

    * Section 7.5.14 measured `fourier_bands = (1, 4, 16)` reaching 99.5% of the reference
      voided length at the old `qn = 3000` budget -- the best embedding number here.
    * Section 7.5.20 measured `qn = 30000` at a single band reaching `T_s = 0.0017`,
      15x better than `qn3000` and monotone on every seed.

    Both were read as independent wins and neither was measured against the other. The
    reason to doubt that reading is mechanistic rather than statistical: a feature
    pyramid **is** a preconditioner. Spreading a fixed feature budget across bands
    equalises the curvature the optimiser sees across scales, which is the same
    ill-conditioning a longer quasi-Newton run works around by accumulating curvature
    pairs. If that is what both are doing, they are one gain bought twice and the 2x2
    interaction cell is flat -- and this project should stop spending on the embedding
    axis. If the cell is not flat they compose, and the combination is the new default.

    A 2x2 is the smallest design that can tell those apart; two one-factor ladders
    cannot, however many seeds each has, because neither ever varies the other factor.

    **The control arm is `single/qn30000`**, which is the shipped default and must
    reproduce section 7.5.20's 0.0017. Read it before reading anything else: that arm is
    the reason D67 was caught, and a study whose control has moved is measuring the
    harness rather than the arms.

    JAX-only, for the same reason as `qnladder` -- at 4.4x torch (section 7.5.19) the
    30000-iteration arms are hours rather than most of a day, and the backend gap is now
    understood (section 7.5.21) rather than being one of the unknowns. That is written
    into the arm list rather than left to the caller: the first version said "JAX-only"
    here and iterated `BACKENDS` below, which is a 24-arm study wearing a 12-arm
    docstring, and its six torch `qn30000` arms alone are most of a week.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"{'bands(1,4,16)' if b else 'single'}/qn{q} [jax]",
                {
                    "backend": "jax",
                    "seed": seed,
                    "fourier_bands": b,
                    "lbfgs_iters": q,
                },
            )
            for seed in SEEDS
            for b in ((), (1.0, 4.0, 16.0))
            for q in (3000, 30000)
        ],
        out,
    )
    mean_table(rows)
    print("\ndo bands and quasi-Newton iterations compose, or are they one gain twice?")
    _arm_summary(rows)
    _interaction(rows)


def study_adamonly(out: Path) -> None:
    """Test whether the ARCHITECTURE lets Adam replace quasi-Newton -- section 7.5.27.

    arXiv:2605.24278 ("beignet") reports a trainable multi-resolution Fourier feature
    pyramid reaching "an accuracy regime previously attained only by using computationally
    expensive higher-order optimizers", **using Adam**. The claim is about the embedding,
    not the optimiser, and that is what makes it testable here.

    **Running our own Adam longer would test a strawman.** Section 7.5.11 measured our
    Adam axis flat across two decades once the quasi-Newton stage is funded, and across
    every study on disk it saturates near 0.04-0.05 whatever is spent -- adam3000/qn30
    0.0762, adam8000/qn500 0.0485, adam16000/qn1000 0.0432 -- while at the same ~1200 s
    the Newton-heavy adam30/qn3000 reaches 0.0302. Scaling Adam 500x moves the answer the
    wrong way. So the question is not "more Adam", it is "a different function space".

    **Two arms, one knob.** Both run `adam_iters = 30000, lbfgs_iters = 0` at our own
    `lr = 1e-3` -- which is also the paper's base LR -- and differ only in the embedding:

    * `fourier` -- the shipped random Fourier features, `B` FROZEN. The Adam-only
      baseline, and the arm that attributes any gain to the architecture rather than to
      spending the budget on Adam.
    * `beignet` -- the paper's trainable pyramid at its own 1D configuration.

    30000 is the shipped default's *quasi-Newton* count, so this is the shipped budget
    spent entirely on Adam and is directly comparable to everything else measured here.

    **The control is cited, not re-run.** `qnladder_s{0,1,2}.json` already carries
    `adam30/qn30000` at three seeds on this backend and configuration -- `T_s = 0.0017`
    in 9060 s -- and re-running it would double this study's cost to reproduce a number
    measured on this tree. That is a stated exception to the control-arm rule, not an
    assumption that it would have passed.

    **What each outcome means.** If `beignet` approaches 0.0017 while `fourier` sits at
    its usual 0.04-0.05, the paper transfers and the quasi-Newton stage is replaceable
    here. If both saturate together, the pyramid buys nothing on this problem and
    section 7.5.11 stands against a much stronger test than the grid alone could give.
    Note the paper's own Table 2 has MLP+BFGS at 7.11e-20 against beignet+Adam's
    6.63e-19, so even its own evidence is that Adam *reaches* the regime rather than
    winning it.

    **The risk to watch first is periodicity.** Fourier interpolation is periodic and
    every benchmark in the paper is a periodic problem; this channel is not. `beignet_pad`
    maps `zeta` into the interior of one period, and if that is insufficient the failure
    will show as error concentrated at the inlet and outlet rather than as a uniformly
    worse field.
    """
    traj = ruler()
    rows = run_all(
        traj,
        [
            (
                f"{name} adam{ADAM_ONLY_ITERS}/qn0 [jax]",
                {
                    "backend": "jax",
                    "seed": seed,
                    "adam_iters": ADAM_ONLY_ITERS,
                    "lbfgs_iters": 0,
                    **extra,
                },
            )
            for seed in SEEDS
            for name, extra in (
                ("fourier", {}),
                ("beignet", BEIGNET_1D),
                # No embedding at all: the coordinate-MLP baseline the beignet paper
                # compares against, and the arm that says how much the frozen Fourier
                # features are worth on their own under Adam.
                ("nofourier", {"fourier_features": 0}),
                # Same, with the slow-EMA first-order method. JAX only: torch has no
                # AdEMAMix and one is deliberately not written for it. Named WITHOUT
                # "nofourier": `--only` is an OR over substrings, so an arm name that
                # contains another arm's name makes the two impossible to select apart.
                ("ademamix", {"fourier_features": 0, "first_order": "ademamix"}),
            )
        ]
        + [
            # THE decisive arm for "is the embedding just a preconditioner?".
            # 7.5.24 showed the multi-band embedding is subsumed by the budget. If the
            # Fourier embedding is ALSO only a preconditioner, then no embedding at all
            # plus a funded quasi-Newton stage should reach the same 0.0017 -- and the
            # entire embedding axis, capacity ladder included, is redundant work.
            # If it does NOT, the embedding buys representation that budget cannot.
            (
                f"nofourier adam30/qn{q} [jax]",
                {
                    "backend": "jax",
                    "seed": seed,
                    "adam_iters": 30,
                    "lbfgs_iters": q,
                    "fourier_features": 0,
                },
            )
            for seed in SEEDS
            for q in (30000,)
        ],
        out,
    )
    mean_table(rows)
    print("\ndoes Adam alone reach the quasi-Newton stage, at comparable wall-clock?")
    _arm_summary(rows)
    _per_second(rows)


def _per_second(rows: list[dict]) -> None:
    """Report each arm against its own wall-clock, which is the axis the decision is on.

    Equal iterations is the wrong comparison between two optimisers with different
    per-step costs, and this project has already had a ladder invert when it was moved
    from iterations to a clock (section 7.5.17a). The control's seconds are the budget
    every Adam arm has to be read against.
    """
    import statistics  # noqa: PLC0415

    print(f"\n{'arm':32s}{'mean sec':>10}{'mean T_s':>10}{'worst margin':>14}")
    for label in dict.fromkeys(r["arm"] for r in rows):
        v = [r for r in rows if r["arm"] == label]
        print(
            f"  {label:30s}{statistics.mean(r['sec'] for r in v):>10.0f}"
            f"{statistics.mean(r['T_s'] for r in v):>10.4f}"
            f"{min(r['margin_K'] for r in v):>13.1f} K"
        )
    print(
        "  (compare each Adam arm against the CONTROL's seconds, not against the other\n"
        "   Adam arms: the question is whether the quasi-Newton stage buys its cost.)"
    )


def _interaction(rows: list[dict]) -> None:
    """Report the 2x2 interaction on `T_s`, which is the number the study exists for.

    Stated as a ratio of ratios: how much bands buy at `qn3000` against how much they buy
    at `qn30000`. A value near 1 means the two axes are independent and additive in the
    log; well below 1 means the second one bought nothing once the first was paid for.

    Per-seed ranges accompany it because a 2x2 read from means alone is exactly the kind
    of single-number comparative headline this project has had to retract four times.
    """
    import statistics  # noqa: PLC0415

    embeddings = ("single", "bands(1,4,16)")
    budgets = (3000, 30000)

    def cell(emb: str, q: int) -> list[float]:
        # Exact match on the arm name, not a prefix: "single/qn3000" is a prefix of
        # "single/qn30000", so `startswith` silently merges two cells of the 2x2 into one.
        tag = f"{emb}/qn{q}"
        return [
            r["T_s"] for r in rows if r["arm"].split(" [")[0] == tag and r.get("T_s") is not None
        ]

    cells = {(e, q): cell(e, q) for e in embeddings for q in budgets}
    if not all(cells.values()):
        print("\ninteraction: incomplete, some cell has no finished run")
        return
    print("\n                    qn3000                  qn30000")
    for e in embeddings:
        cols = "  ".join(
            f"{statistics.mean(cells[e, q]):.4f} [{min(cells[e, q]):.4f}-{max(cells[e, q]):.4f}]"
            for q in budgets
        )
        print(f"  {e:15s} {cols}")
    gain_lo = statistics.mean(cells["single", 3000]) / statistics.mean(cells[embeddings[1], 3000])
    gain_hi = statistics.mean(cells["single", 30000]) / statistics.mean(cells[embeddings[1], 30000])
    print(
        f"\n  bands buy {gain_lo:.2f}x at qn3000 and {gain_hi:.2f}x at qn30000; "
        f"interaction {gain_hi / gain_lo:.2f}"
    )
    print(
        "  (near 1: the axes compose and the combination is the new default. "
        "well below 1: the\n   funded quasi-Newton stage already bought what the "
        "bands were buying -- one gain, twice.)"
    )


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
    "onset": study_onset,
    "laplace": study_laplace,
    "qnladder": study_qnladder,
    "bakeoff": study_bakeoff,
    "bandsbudget": study_bandsbudget,
    "adamonly": study_adamonly,
}


def main() -> int:
    """Run one study and write its JSON."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--out", type=Path, default=None, help="JSON output path")
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds to run instead of all three, so one ladder can be "
        "split across cpu blocks and finish in a third of the wall-clock",
    )
    ap.add_argument(
        "--cpu-block",
        type=int,
        default=None,
        help="pin to OMP_NUM_THREADS cores starting at this block index, so JAX has a "
        "real thread budget and results are bitwise reproducible. Concurrent studies "
        "should take different blocks; the core count is what matters, not which cores",
    )
    ap.add_argument(
        "--lbfgs-history",
        type=int,
        default=None,
        help="override the quasi-Newton curvature memory on every arm. optax.lbfgs "
        "defaults to 10 against torch's 50, and that single argument was the whole "
        "cross-backend accuracy gap (docs/axial_nn.md section 7.5.17)",
    )
    ap.add_argument(
        "--adam-iters",
        type=int,
        default=None,
        help="override Adam iterations on arms that do not set their own. Name it "
        "explicitly for any study whose cost matters: the config default has moved "
        "before and took a sub-command's cost with it",
    )
    ap.add_argument(
        "--lbfgs-iters",
        type=int,
        default=None,
        help="override quasi-Newton iterations on arms that do not set their own",
    )
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
    if args.cpu_block is not None:
        n = int(os.environ.get("OMP_NUM_THREADS", "8"))
        cores = pin_cpu_block(args.cpu_block, n)
        print(f"pinned to {len(cores)} cores {cores[0]}-{cores[-1]}", flush=True)
    if args.seeds:
        global SEEDS  # noqa: PLW0603
        SEEDS = tuple(int(x) for x in args.seeds.split(","))
        print(f"seeds: {SEEDS}", flush=True)
    _ONLY = args.only
    global _HISTORY  # noqa: PLW0603
    _HISTORY = args.lbfgs_history
    global _ADAM, _QN
    _ADAM, _QN = args.adam_iters, args.lbfgs_iters
    if _ADAM is not None or _QN is not None:
        print(f"budget override: adam={_ADAM} qn={_QN} (arms naming their own keep it)")
    STUDIES[args.study](out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
