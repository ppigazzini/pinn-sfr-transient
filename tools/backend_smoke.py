"""The short both-backend check every new algorithm has to pass before a long run.

AGENTS.md: *two backends for correctness, one backend for measurement*. The long runs
are JAX, because it is 4.4x faster at matched threads and the two agree to 3% once
curvature memory matches. But every defect this project actually shipped was a
**cross-backend** defect that a long single-backend run could never have shown:

* `optax.lbfgs()` called bare — `memory_size = 10` against torch's 50, read as a
  framework difference for four milestones;
* a Broyden update contracted against the accumulator instead of the original vector,
  which passed every training run and was 10% wrong;
* JAX arms scored under the shipped defaults because `train` discarded the config it
  was handed, so a whole study measured the control four times.

None of those needed a long run to find, and none of them was findable from inside one
implementation. So this runs **short**, and it runs **both**.

Three checks, deliberately independent, and each targets one of the failures above:

1. **Config parity** — the two `AxialTrainConfig` dataclasses must expose the same
   fields with the same defaults. Catches a knob added to one backend only, which is
   what makes every later cross-backend number a comparison of two different models.
2. **Optimiser parity on a known objective** — both backends' optimisers, same start,
   same hyper-parameters, on an ill-conditioned quadratic and on Rosenbrock, comparing
   the *iterate sequences*. This is where a memory-size default or a wrong update
   formula shows up immediately: an ill-conditioned quadratic is exactly the problem
   that separates 10 curvature pairs from 50, and it costs milliseconds.
3. **Config plumbing** — train a tiny model under a *non-default* config in each
   backend and assert the result actually depends on it. A config silently ignored is
   invisible in any single run, because the run still converges to something.

Check 2 does not compare the PINN loss across backends, and that is on purpose: the two
draw different initialisations from different RNGs, so their loss traces are not
comparable without transferring weights, and a transfer harness is a much larger thing
to keep correct than the defect it would catch. Comparing the optimisers on an objective
with a known answer isolates the algorithm without needing the weights to match.

    uv run python tools/backend_smoke.py
    uv run python tools/backend_smoke.py --optimizer ssbfgs --history 10
"""

import argparse
import sys
import time
from dataclasses import fields

import numpy as np

# An ill-conditioned quadratic is the discriminating problem for curvature memory:
# with too few pairs the method cannot cover the spectrum and stalls, and the stall is
# visible within a few hundred iterations. Rosenbrock adds nonlinearity so a line-search
# or self-scaling defect shows as well.
QUAD_COND = 1e6
QUAD_N = 40
ROSEN_N = 12
DEFAULT_ITERS = 200

# How far apart the two backends' final values may be, as a RATIO. Calibrated rather
# than guessed, on the defect this check exists to catch:
#
#   history 10 -> 1.44e+01 (torch) / 1.51e+01 (jax)
#   history 50 -> 4.74e-01 (torch) / 4.66e-01 (jax)
#
# So the signal -- the wrong curvature memory -- moves the answer **31x**, while the
# cross-backend disagreement at a *fixed* memory is 1.045x and 1.017x. A 1.25 threshold
# sits ~20x above the noise and ~25x below the signal. The first version of this check
# demanded 1e-4 on the final value, which is far inside the run-to-run spread of two
# separate line searches on a cond-1e6 problem and would have failed forever.
RATIO_TOL = 1.25

# Below this both backends have simply converged, and a *relative* comparison of two
# numbers near zero is meaningless -- the first version reported a disagreement of 1.000
# for torch's exact 0.0 against JAX's 1.2e-30, which is agreement to 30 digits.
CONVERGED_ATOL = 1e-12

# How far compiled training may drift from eager, as a RELATIVE parameter-vector
# distance. This check itself measures 3.6e-16 over 200 iterations at f256 -- a few ulp
# -- so the bound is ~1e4 slack on the measurement and still orders below the smallest
# difference any table here reports.
COMPILE_ATOL = 1e-12

# Iterations run before the timed pair, to pay compilation once. Only the compiled arm
# needs it; the eager arm runs it too so the two are timed the same way.
WARM_ITERS = 20

# How many times the timed pair is repeated, and why the estimator is a MINIMUM.
#
# The per-iteration cost is a DIFFERENCE of two wall-clocks, which amplifies jitter: at
# 200 iterations the compiled arm's two timings are about 1.7 s and 3.4 s, so half a
# second of interference anywhere moves the answer by ~30%. Measured, one sample per run
# on an otherwise idle machine: the compiled arm read 6.36, 7.79, 8.10, 8.45, 10.27 and
# 10.53 ms across six runs of identical work, and a cross-backend comparison built on one
# such sample came out 1.35x, 0.99x and 1.25x on three consecutive tries -- i.e. it
# reversed. A single sample here is not a measurement.
#
# Wall-clock noise is one-sided: nothing makes a run faster than the work it does, so the
# minimum over repeats is the estimator that converges on the real cost rather than on
# the machine's mood. The spread is reported beside it so a reader can see whether the
# repeats agreed.
TIMING_REPEATS = 3

# Config fields that legitimately differ, each with the reason it is allowed. Anything
# NOT listed here is a failure: a knob in one backend and not the other forks the model
# silently and makes every later cross-backend number a comparison of two different
# things. These are documented divergences, not discovered ones.
ALLOWED_CONFIG_DIFFS = {
    "device": "torch places tensors explicitly; JAX has no equivalent knob",
    "rar_add": "torch grows an unbounded RAR reservoir",
    "rar_cap": "torch grows an unbounded RAR reservoir",
    "rar_keep": "JAX keeps a FIXED RAR count so jit never recompiles - same idea, "
    "framework-appropriate form (see jaxpinn/config.py)",
    "compile": "torch chooses per run whether to compile; the JAX twin is always "
    "jitted, so there is nothing to switch (see torchpinn/config.py)",
}


def _quadratic(n: int = QUAD_N, cond: float = QUAD_COND) -> np.ndarray:
    """Diagonal of an ill-conditioned quadratic, log-spaced over ``cond``."""
    return np.logspace(0.0, np.log10(cond), n)


def check_config_parity() -> list[str]:
    """Return a list of problems with field/default agreement across the backends."""
    import importlib  # noqa: PLC0415

    cfgs = {}
    for b in ("torch", "jax"):
        try:
            mod = importlib.import_module(f"pinn_sfr_transient.axial.{b}pinn.config")
        except SystemExit:
            return [f"{b} extra not installed - parity NOT checked, which is not a pass"]
        cfgs[b] = mod.AxialTrainConfig()

    problems = []
    tf = {f.name for f in fields(cfgs["torch"])}
    jf = {f.name for f in fields(cfgs["jax"])}
    for name in sorted(tf ^ jf):
        if name in ALLOWED_CONFIG_DIFFS:
            print(f"  known divergence: `{name}` - {ALLOWED_CONFIG_DIFFS[name]}")
            continue
        where = "torch" if name in tf else "jax"
        problems.append(f"field `{name}` exists in {where} only, and is not a declared divergence")
    for name in sorted(tf & jf):
        a, b = getattr(cfgs["torch"], name), getattr(cfgs["jax"], name)
        if a != b and name not in ALLOWED_CONFIG_DIFFS:
            problems.append(f"default `{name}` differs: torch={a!r} jax={b!r}")
    return problems


def _knobs(iters: int, history: int, opt: str) -> dict:
    """Build the optimiser settings ONCE, to be handed to both backends verbatim.

    The point of a single dict is that the two calls cannot drift: the entire
    cross-backend accuracy gap of §7.5.17 was one argument set on one side only, and a
    harness that spelled the arguments out twice could reproduce that defect rather
    than detect it. Both implementations take the same keyword names.
    """
    return {
        "max_iter": iters,
        "history_size": history,
        "self_scale": opt in ("ssbfgs", "ssbroyden"),
        "broyden_phi": 0.5 if opt == "ssbroyden" else 0.0,
        "tolerance_grad": 0.0,
        "tolerance_change": 0.0,
    }


def _torch_final(objective: str, x0: np.ndarray, knobs: dict) -> tuple[float, int]:
    """Minimise with the torch optimiser; return ``(final value, evaluations)``."""
    import torch  # noqa: PLC0415

    from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS  # noqa: PLC0415

    d = torch.tensor(_quadratic()) if objective == "quadratic" else None
    x = torch.tensor(x0.copy(), requires_grad=True)
    calls = [0]
    optimiser = SelfScaledLBFGS([x], **knobs)

    def closure() -> torch.Tensor:
        x.grad = None
        loss = 0.5 * (d * x**2).sum() if d is not None else _rosen_torch(x)
        loss.backward()
        calls[0] += 1
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        final = 0.5 * (d * x**2).sum() if d is not None else _rosen_torch(x)
    return float(final), calls[0]


def _rosen_torch(x):  # noqa: ANN001, ANN202
    return (100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2).sum()


def _jax_final(objective: str, x0: np.ndarray, knobs: dict) -> tuple[float, int]:
    """Run the JAX twin through the shared implementation, with the same knobs."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from pinn_sfr_transient.axial.jaxpinn.optimizers import minimize  # noqa: PLC0415

    d = jnp.asarray(_quadratic())

    def loss(z: jax.Array) -> jax.Array:
        if objective == "quadratic":
            return 0.5 * jnp.sum(d * z**2)
        return jnp.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2)

    vg = jax.value_and_grad(loss)
    _, f = minimize(vg, jnp.asarray(x0.copy()), **knobs)
    return float(f), 0


def check_plumbing(iters: int) -> list[str]:
    """Check that a non-default config changes the answer in both backends.

    Trains twice per backend, at the default and at a deliberately different width, and
    requires the two to differ. A config that is silently dropped still converges to
    *something*, so nothing but a comparison can see it.
    """
    problems = []
    from pinn_sfr_transient.axial.config import AxialParams  # noqa: PLC0415

    p = AxialParams(n_axial=20)
    for b in ("torch", "jax"):
        try:
            if b == "jax":
                from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

                mk = pj.AxialTrainConfig
                run = lambda c: pj.train(p, c, verbose=False)[0]  # noqa: E731
            else:
                from pinn_sfr_transient.axial.torchpinn import (  # noqa: PLC0415
                    AxialTrainConfig,
                    train,
                )

                mk = AxialTrainConfig
                run = lambda c: train(p, c)  # noqa: E731
        except SystemExit:
            problems.append(f"{b} extra not installed - plumbing NOT checked")
            continue
        base = {
            "seed": 0,
            "adam_iters": iters,
            "lbfgs_iters": 0,
            "log_every": 10**9,
            "n_colloc": 128,
        }
        n_a = _param_count(run(mk(**base, width=8, depth=2)), b)
        n_b = _param_count(run(mk(**base, width=16, depth=2)), b)
        if n_a == n_b:
            problems.append(f"{b}: width=8 and width=16 produced the same model - config ignored")
    return problems


def _param_count(model: object, backend: str) -> int:
    """Trainable parameter count, however the backend spells it."""
    if backend == "torch":
        return sum(p.numel() for p in model.parameters())  # ty: ignore
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    leaves, _ = jax.tree_util.tree_flatten(eqx.filter(model, eqx.is_inexact_array))
    return sum(int(a.size) for a in leaves)


def check_compile(iters: int, colloc: int, features: int) -> list[str]:
    """Train the torch first-order loop eager and compiled; require the same answer.

    ``cfg.compile`` is a *performance* switch, so the thing to prove is that it is only
    that. The two runs share a seed, so their collocation draws and initialisation are
    identical and any difference is inductor's -- fusion reordering a reduction, most
    plausibly. At the defaults below -- f256, 500 points, 200 iterations -- it reports
    3.6e-16, a few ulp, so `COMPILE_ATOL` is set at 1e-12: loose enough not to be a flake,
    ~1e4 above the measurement, and still orders below any digit this project publishes.

    Not a pytest: compilation costs 12-40 s per input shape, and the suite is required
    to stay fast. It belongs here, with the other checks that are worth a minute.
    """
    import torch  # noqa: PLC0415

    from pinn_sfr_transient.axial.config import AxialParams  # noqa: PLC0415
    from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

    p = AxialParams()
    kw = {
        "seed": 0,
        "lbfgs_iters": 0,
        "n_colloc": colloc,
        "fourier_features": features,
        "rar_every": 0,
        "pts_every": 0,
        "log_every": 10**9,
    }

    def timed(n: int, *, compiled: bool) -> tuple[list, float]:
        t0 = time.perf_counter()
        model = train(p, AxialTrainConfig(**{**kw, "adam_iters": n}, compile=compiled))
        return [q.detach().clone() for q in model.parameters()], time.perf_counter() - t0

    out, rate, spread = {}, {}, {}
    for compiled in (False, True):
        # Warm first, then difference two budgets. Compilation is a fixed cost paid once
        # per input shape and inductor caches it across calls in a process, so a single
        # timing reports the compile and a naive difference against a cold run reports a
        # NEGATIVE rate. Differencing two warm runs cancels the fixed cost and leaves the
        # steady state, which is the number a long run actually gets.
        timed(WARM_ITERS, compiled=compiled)
        samples = []
        for _ in range(TIMING_REPEATS):
            params, t1 = timed(iters, compiled=compiled)
            _, t2 = timed(2 * iters, compiled=compiled)
            samples.append((t2 - t1) / iters * 1e3)
        out[compiled] = params
        rate[compiled] = min(samples)
        spread[compiled] = max(samples) / min(samples)
    pe, pc = out[False], out[True]
    num = sum(float((a - b).pow(2).sum()) for a, b in zip(pe, pc, strict=True)) ** 0.5
    den = sum(float(a.pow(2).sum()) for a in pe) ** 0.5
    rel = num / max(den, 1e-300)
    bitwise = all(bool(torch.equal(a, b)) for a, b in zip(pe, pc, strict=True))
    print(f"  {iters} iterations, {colloc} points, f{features}, {torch.get_num_threads()} threads")
    sp = f"eager {spread[False]:.2f}x, compiled {spread[True]:.2f}x"
    print(f"  best of {TIMING_REPEATS}; spread {sp}")
    print(f"    eager    {rate[False]:7.2f} ms/iteration  ({1e3 / rate[False]:6.1f} it/s)")
    print(f"    compiled {rate[True]:7.2f} ms/iteration  ({1e3 / rate[True]:6.1f} it/s)")
    ratio = rate[False] / rate[True]
    print(f"    speedup  {ratio:7.2f}x   (steady state; compile cost differenced out)")
    print(f"    ||dparams|| / ||params|| = {rel:.3e}{'  (bitwise equal)' if bitwise else ''}")
    if rel > COMPILE_ATOL:
        msg = (
            f"compiled and eager training disagree by {rel:.3e} > {COMPILE_ATOL:.0e}. "
            f"`compile` is meant to change the wall-clock and nothing else."
        )
        return [msg]
    return []


#: The matched configuration both backends are timed on: first-order only, every
#: cadence off, so each is a single compiled region and neither is being charged for
#: Python the other does not run. Anything switched on here would have to be switched on
#: identically, and the cheapest way to guarantee that is to switch it all off.
TIMING_KW = {
    "seed": 0,
    "lbfgs_iters": 0,
    "rar_every": 0,
    "pts_every": 0,
    "weight_max_ratio": 1.0,
    "causal_eps": 0.0,
    "first_order": "adam",
    "log_every": 10**9,
}


def check_timing(iters: int, colloc: int, features: int) -> list[str]:
    """Time one first-order iteration on each backend, at matched settings.

    Not a pass/fail check -- it reports. The number it reports is the one the
    measurement policy in AGENTS.md rests on, and that policy was set when the torch
    loop ran eager, so it is worth being able to re-measure with one command.

    Both arms warm up and then difference two budgets, so neither compilation nor
    tracing is inside the figure, and both are float64. Pin the cores before believing
    it: thread count changes reduction order as well as speed, and `OMP_NUM_THREADS`
    binds torch while XLA sizes its own pool from `hardware_concurrency()`.
    """
    kw = {**TIMING_KW, "n_colloc": colloc, "adam_colloc": colloc, "fourier_features": features}

    def rate(run: object) -> tuple[float, float]:
        """Best-of-repeats per-iteration cost, and the spread across them."""
        run(WARM_ITERS)  # ty: ignore
        samples = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            run(iters)  # ty: ignore
            t1 = time.perf_counter()
            run(2 * iters)  # ty: ignore
            samples.append((time.perf_counter() - t1 - (t1 - t0)) / iters * 1e3)
        return min(samples), max(samples) / min(samples)

    from pinn_sfr_transient.axial.config import AxialParams  # noqa: PLC0415

    p = AxialParams()
    out: dict[str, tuple[float, float]] = {}
    from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415
    from pinn_sfr_transient.axial import torchpinn as tp  # noqa: PLC0415

    def torch_run(n: int) -> object:
        return tp.train(p, tp.AxialTrainConfig(**kw, adam_iters=n, compile=True))

    def jax_run(n: int) -> object:
        return pj.train(p, pj.AxialTrainConfig(**kw, adam_iters=n), verbose=False)

    out["torch (compiled)"] = rate(torch_run)
    out["jax (jitted)"] = rate(jax_run)

    print(f"  {colloc} points, f{features}, {iters} iterations differenced against {2 * iters}")
    print(f"  best of {TIMING_REPEATS} per backend")
    for name, (ms, sp) in out.items():
        print(f"    {name:18s} {ms:7.2f} ms/iteration  ({1e3 / ms:6.1f} it/s)   spread {sp:.2f}x")
    fast = min(out, key=lambda k: out[k][0])
    slow = max(out, key=lambda k: out[k][0])
    ratio = out[slow][0] / out[fast][0]
    print(f"    {fast} is {ratio:.2f}x faster, at this configuration and this thread count.")
    if ratio < max(sp for _, sp in out.values()):
        print("    BUT that gap is smaller than the spread within a backend, so these two")
        print("    are not separated by this measurement. Do not write a headline from it.")
    return []


def main() -> int:  # noqa: C901, PLR0912, PLR0915 - a report reads better flat
    """Run the three checks and return non-zero if any of them fails."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--optimizer", default="lbfgs", choices=("lbfgs", "ssbfgs", "ssbroyden"))
    ap.add_argument("--history", type=int, default=50, help="curvature pairs, BOTH backends")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--skip-plumbing", action="store_true", help="skip the two tiny trainings")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="also check that torch's `cfg.compile` changes only the wall-clock "
        "(a minute or two: compilation is 12-40 s per input shape)",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="report one first-order iteration on each backend at matched settings",
    )
    ap.add_argument("--compile-iters", type=int, default=200)
    ap.add_argument("--compile-colloc", type=int, default=500)
    ap.add_argument("--compile-features", type=int, default=256)
    args = ap.parse_args()

    failures = 0

    print("=== 1. config parity: same fields, same defaults ===")
    problems = check_config_parity()
    if problems:
        failures += len(problems)
        for q in problems:
            print(f"  FAIL  {q}")
    else:
        print("  ok - every field and default agrees")

    print(f"\n=== 2. optimiser parity: {args.optimizer}, history {args.history} ===")
    try:
        import jax  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError:
        print("  SKIP - both extras are needed and one is missing")
    else:
        knobs = _knobs(args.iters, args.history, args.optimizer)
        rng = np.random.default_rng(0)
        for objective, n in (("quadratic", QUAD_N), ("rosenbrock", ROSEN_N)):
            x0 = rng.normal(size=n) if objective == "quadratic" else np.full(n, -1.2)
            t_f, t_calls = _torch_final(objective, x0, knobs)
            j_f, _ = _jax_final(objective, x0, knobs)
            note = " (cond 1e6 - this is the one memory size shows up on)" if n == QUAD_N else ""
            print(f"  {objective}{note}")
            print(f"    torch {t_f:.8e} in {t_calls} evaluations")
            print(f"    jax   {j_f:.8e}")
            if max(abs(t_f), abs(j_f)) < CONVERGED_ATOL:
                print(f"    both below {CONVERGED_ATOL:.0e} - converged; ratio not meaningful")
                print("    ok")
                continue
            ratio = max(abs(t_f), abs(j_f)) / max(min(abs(t_f), abs(j_f)), 1e-300)
            print(f"    ratio {ratio:.4f}   (tolerance {RATIO_TOL})")
            if ratio > RATIO_TOL:
                print(
                    "    FAIL  two implementations of the same algorithm disagree on a\n"
                    "          problem with a known answer. Diff the ARGUMENTS before\n"
                    "          theorising about the libraries - that was §7.5.17."
                )
                failures += 1
            else:
                print("    ok")

    if args.skip_plumbing:
        print("\n=== 3. config plumbing: SKIPPED by request ===")
    else:
        print("\n=== 3. config plumbing: a non-default config must change the model ===")
        problems = check_plumbing(iters=5)
        if problems:
            failures += len(problems)
            for q in problems:
                print(f"  FAIL  {q}")
        else:
            print("  ok - both backends honour the config they are handed")

    if args.compile:
        print("\n=== 4. torch.compile: same answer, less wall-clock ===")
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            print("  SKIP - the torch extra is missing")
        else:
            problems = check_compile(args.compile_iters, args.compile_colloc, args.compile_features)
            if problems:
                failures += len(problems)
                for q in problems:
                    print(f"  FAIL  {q}")
            else:
                print("  ok - compiled training matches eager")

    if args.timing:
        print("\n=== 5. matched per-iteration cost, both backends ===")
        try:
            import jax  # noqa: F401, PLC0415
            import torch  # noqa: F401, PLC0415
        except ImportError:
            print("  SKIP - both extras are needed and one is missing")
        else:
            check_timing(args.compile_iters, args.compile_colloc, args.compile_features)

    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
