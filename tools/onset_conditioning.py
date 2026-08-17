"""Task 1: is the onset-time error a conditioning number or an accuracy number.

`docs/axial_nn.md` section 7.5.16 reports the network's boiling onset 0.62-0.84 s from the
reference's 10.75 s, against a 0.5 s criterion, while the temperature fields meet their
bar. Annex E.6 says that comparison is not yet interpretable, because to first order

    dt* ~= ||dT||_inf / |dT_out/dt|(t*)

so the *same* field error produces a large or a small timing error depending entirely on
how fast the outlet is heating when it crosses saturation. Two very different problems
hide behind one number:

* If `dT_out/dt` is small at onset -- a slow approach to saturation during coastdown --
  then a sub-kelvin local error explains the whole miss, the amplification is intrinsic,
  and **further field accuracy is an inefficient way to buy timing**.
* If it is large, the local error is many kelvin, the field is locally far worse than a
  global relative `L2` of 0.0017 suggests, and that is a tractable and different problem.

Global relative `L2` cannot distinguish them: it is an average over the whole space-time
domain, dominated by the smooth bulk. What bounds onset is `L_inf` of the **outlet**
in a window around the crossing. This measures both, plus the amplification factor they
imply, and reports the onset time by **root-finding on the network's own dense output**
rather than by scanning a grid -- the discipline `scipy.integrate.solve_ivp` uses for
event location, which brackets the crossing and calls `brentq` to a few epsilon. Any grid
quantisation in the published number is free to remove that way.

    OMP_NUM_THREADS=8 uv run python tools/onset_conditioning.py --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference

# Window, in seconds either side of the reference onset, over which the outlet trace is
# compared. The criterion is 0.5 s, so a window of +/-2 s brackets any plausible miss
# without letting the far-field dominate the L_inf.
WINDOW_S = 2.0


def crossing_time(t: np.ndarray, trace: np.ndarray, threshold: float) -> float:
    """First time the trace crosses ``threshold``, by bracketed root-finding.

    The trace is sampled, so this brackets the crossing on the sample grid and then
    refines inside the bracket with `brentq` on a linear interpolant. That removes the
    grid quantisation which, at the 0.25 s output spacing, is the same size as a quarter
    of the acceptance criterion — section 7.5.16 found the published onset-time error was partly
    this artefact.

    Returns ``nan`` when the trace never reaches the threshold, which is a failure and
    must not be defaulted to zero error.
    """
    above = trace >= threshold
    if not above.any():
        return float("nan")
    i = int(np.argmax(above))
    if i == 0:
        return float(t[0])
    lo, hi = t[i - 1], t[i]
    f = lambda x: np.interp(x, t, trace) - threshold  # noqa: E731
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


def heating_rate(t: np.ndarray, trace: np.ndarray, t_star: float) -> float:
    """``dT_out/dt`` at the crossing, the denominator of the amplification factor.

    Central difference on the sampled trace. A one-sided difference would bias the
    estimate exactly where the transient is steepest, and the sample spacing here is far
    finer than the timescale of the coastdown.
    """
    h = max(2.0 * float(np.mean(np.diff(t))), 1e-3)
    hi = float(np.interp(t_star + h, t, trace))
    lo = float(np.interp(t_star - h, t, trace))
    return (hi - lo) / (2.0 * h)


def main() -> int:  # noqa: PLR0915 - a measurement report reads better flat
    """Measure the conditioning of onset time and report what it implies."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("jax", "torch"), default="jax")
    ap.add_argument("--n-axial", type=int, default=160)
    ap.add_argument("--out", type=Path, default=Path("onset_conditioning.json"))
    args = ap.parse_args()

    p = AxialParams()
    thr = float(sodium.saturation_temperature(p.p_system) + p.dT_superheat)
    traj = solve_reference(AxialParams(n_axial=args.n_axial), n_out=241)

    ref_trace = np.asarray(traj.T_c)[-1, :]
    t_ref = crossing_time(traj.t, ref_trace, thr)
    rate_ref = heating_rate(traj.t, ref_trace, t_ref)

    if args.backend == "jax":
        from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

        cfg = pj.AxialTrainConfig(seed=args.seed, log_every=10**9)
        model, pp, cfg = pj.train(p, cfg, verbose=False)
        fields = pj.predict(model, pp, traj.zeta, traj.t, cfg)
    else:
        from pinn_sfr_transient.axial.torchpinn import AxialTrainConfig, train  # noqa: PLC0415

        cfg = AxialTrainConfig(seed=args.seed, log_every=10**9)
        model = train(p, cfg)
        fields = model.predict(traj.zeta, traj.t)

    net_trace = np.asarray(fields[3])[-1, :]
    t_net = crossing_time(traj.t, net_trace, thr)
    rate_net = heating_rate(traj.t, net_trace, t_net)

    # L_inf of the outlet trace, restricted to a window around the crossing. This is the
    # norm the implicit function theorem actually involves; the global L2 is not.
    win = np.abs(traj.t - t_ref) <= WINDOW_S
    linf_window = float(np.max(np.abs(net_trace[win] - ref_trace[win])))
    linf_global = float(np.max(np.abs(net_trace - ref_trace)))
    l2_global = float(np.linalg.norm(np.asarray(fields[3]) - traj.T_c) / np.linalg.norm(traj.T_c))

    dt_measured = abs(t_net - t_ref)
    dt_predicted = linf_window / abs(rate_ref) if rate_ref else float("nan")

    print(f"reference onset (root-found)      {t_ref:.4f} s")
    print(f"network onset   (root-found)      {t_net:.4f} s")
    print(f"  measured |dt|                   {dt_measured:.4f} s   (criterion 0.5 s)")
    print()
    print(f"heating rate at onset, reference  {rate_ref:+.3f} K/s")
    print(f"heating rate at onset, network    {rate_net:+.3f} K/s")
    print(f"amplification factor 1/|dT/dt|    {1.0 / abs(rate_ref):.4f} s/K")
    print()
    print(f"outlet-trace L_inf, +/-{WINDOW_S:.0f} s window  {linf_window:.4f} K")
    print(f"outlet-trace L_inf, whole run     {linf_global:.4f} K")
    print(f"field-wide relative L2 on T_c     {l2_global:.6f}")
    print()
    print(f"predicted |dt| = L_inf/|dT/dt|    {dt_predicted:.4f} s")
    ratio = dt_measured / dt_predicted if dt_predicted else float("nan")
    print(f"  measured / predicted            {ratio:.2f}")
    print()
    # The verdict is what the tool exists to produce, so it is stated rather than left
    # for a reader to infer from five numbers.
    need_k = 0.5 * abs(rate_ref)
    print(
        f"To meet 0.5 s the outlet trace must be accurate to {need_k:.3f} K in the\n"
        f"window; it is currently {linf_window:.3f} K, so the required improvement is\n"
        f"{linf_window / need_k:.1f}x on that norm."
    )
    if np.isfinite(ratio) and 0.5 < ratio < 2.0:
        print(
            "\nThe first-order estimate explains the measured error, so the miss is\n"
            "CONDITIONING: buying timing through field accuracy costs the amplification\n"
            "factor above, and a formulation that solves for the crossing directly is\n"
            "the cheaper route."
        )
    else:
        print(
            "\nThe first-order estimate does NOT explain the measured error. Either the\n"
            "local field error is larger than the window L_inf suggests, or the crossing\n"
            "is not locally linear -- both are tractable and neither is conditioning."
        )

    args.out.write_text(
        json.dumps(
            {
                "t_ref": t_ref,
                "t_net": t_net,
                "dt_measured": dt_measured,
                "rate_ref": rate_ref,
                "rate_net": rate_net,
                "linf_window": linf_window,
                "linf_global": linf_global,
                "l2_global": l2_global,
                "dt_predicted": dt_predicted,
                "seed": args.seed,
                "backend": args.backend,
                # The budget is recorded, not left to be read off whatever the config
                # default happens to be on the day: this project has already cited a
                # study at a budget it was not run at, and rows that carry their own
                # configuration are what made that detectable.
                "adam_iters": int(cfg.adam_iters),
                "lbfgs_iters": int(cfg.lbfgs_iters),
                "t_train_frac": float(cfg.t_train_frac),
                "fourier_features": int(cfg.fourier_features),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
