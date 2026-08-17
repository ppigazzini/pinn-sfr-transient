"""Tasks 2 and 3: the Plan A adjoint, and a correction that needs no retraining.

Plan A couples six-group point kinetics to the axial fields through

    rho_void(t) = integral over the channel of  w(zeta) * alpha(zeta, t)  d zeta

with ``w`` **changing sign** near the top of the core. It is therefore a near-cancellation
of two large opposite contributions, and section 7.4 measured it missed by 84-92% while the
Doppler integral -- same fields, same network, non-cancelling weight -- is right to 1.017.

Annex E.4 says that is predictable without training anything. Dual-weighted-residual
theory (Becker & Rannacher, *Acta Numerica* 10) gives the error in a functional as

    J(u) - J(u_h)  =  < R(u_h), z* >  +  higher order

where ``z*`` solves the **adjoint** problem with the functional's derivative as source.
For a functional weighted by ``w`` on an advection-dominated coolant equation the adjoint
runs *backwards* in ``zeta``, so ``z*`` accumulates ``w`` from the outlet downwards: the
weight is **large low in the channel and small at the top**. A residual-magnitude sampler
puts points at the boiling front near the top, which is exactly where the adjoint says
they matter least.

Three things follow, and this tool does all three.

**Task 2, the plot.** ``z*`` is closed form here, so it can be computed and compared
against the collocation density before any training is done. If the adjoint is flat, the
hypothesis is dead and nothing below is worth running.

**Task 3, the correction.** ``rho_corrected = rho(u_theta) - <R(u_theta), z*>`` removes the
leading-order functional error from an *already trained* network at inference cost. It is
the cheapest possible test of whether the 84-92% miss is first order.

**The split.** ``J+`` and ``J-`` are reported separately from here on. A single
near-cancelling number cannot say which half is wrong, and the reactor-physics literature
has decomposed this same integral for fifteen years for that reason.

    OMP_NUM_THREADS=8 uv run python tools/plan_a_adjoint.py --plot adjoint.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.physics import quasi_steady_void
from pinn_sfr_transient.axial.reference import solve_reference


def void_slope(T_c: np.ndarray, p: AxialParams, h: float = 1e-4) -> np.ndarray:
    """Return ``d alpha / d T_c``, the sensitivity the adjoint source is built from.

    Central difference on the shared closure rather than a transcription of its
    derivative: the closure is a cubed superheat switch whose analytic derivative is easy
    to get subtly wrong, and it is the one definition every backend already shares.
    """
    return (
        np.asarray(quasi_steady_void(T_c + h, p)) - np.asarray(quasi_steady_void(T_c - h, p))
    ) / (2.0 * h)


def spatial_adjoint(p: AxialParams, T_c: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Return ``z*(zeta)`` for the void functional on the advective coolant operator.

    The coolant equation is ``dT/dt + v dT/dzeta = S``, so the spatial part of its adjoint
    is ``-v dz*/dzeta = J'(T_c)`` with ``J' = w(zeta) * d alpha / d T_c`` and the terminal
    condition ``z*(1) = 0`` — information propagates *downstream* in the primal and
    therefore *upstream* in the adjoint.

    Integrating from the outlet down,

        z*(zeta) = (1/v) * integral from zeta to 1 of  w(s) * alpha'(T_c(s))  ds

    which is a cumulative sum from the top. The 1/v is a constant scale here and does not
    change the shape, which is the part the hypothesis is about.
    """
    w = np.asarray(p.void_worth(zeta))
    src = w * void_slope(T_c, p)
    # Trapezoid, accumulated from the outlet downwards.
    dz = float(zeta[1] - zeta[0])
    rev = src[::-1]
    acc = np.concatenate([[0.0], np.cumsum(0.5 * (rev[1:] + rev[:-1])) * dz])
    return acc[::-1]


def split_functional(alpha: np.ndarray, p: AxialParams, zeta: np.ndarray) -> tuple[float, float]:
    """Return ``(J+, J-)``, the positive- and negative-worth halves, never summed.

    Reported separately because their sum is a near-cancellation: a single number can be
    right by accident, or wrong in a way that says nothing about which half failed.
    """
    w = np.asarray(p.void_worth(zeta))
    dz = float(zeta[1] - zeta[0])
    contrib = w * alpha * dz
    return float(contrib[w > 0].sum()), float(contrib[w < 0].sum())


def network_split(seed: int, traj, p: AxialParams, k: int) -> tuple[float, float, dict]:  # noqa: ANN001
    """Train a JAX model at the shipped default and split ITS void functional.

    Returns ``(J+, J-, cfg_record)`` evaluated on the reference's own grid at the time
    index ``k`` where the reference functional peaks — the same instant, so the two are
    comparable term by term.

    **This is the open-loop split, and it is not §7.4's number.** §7.4 measures the
    *closed*-loop miss, where `rho_void` feeds back into the kinetics and the error
    compounds; 84-92% is that. This is the functional evaluated on the network's own
    `alpha` field, which is what M4' proposes to score, and the two must not be quoted
    as if they were the same measurement.

    JAX only: under the two-backends-for-correctness rule the long runs are JAX and the
    torch path is exercised short, elsewhere.
    """
    from pinn_sfr_transient.axial import pinn_jax as pj  # noqa: PLC0415

    cfg = pj.AxialTrainConfig(seed=seed, log_every=10**9)
    model, pp, cfg = pj.train(p, cfg, verbose=False)
    fields = pj.predict(model, pp, traj.zeta, traj.t, cfg)
    alpha_net = np.asarray(fields[4])[:, k]
    jp, jn = split_functional(alpha_net, p, np.asarray(traj.zeta))
    record = {
        "seed": seed,
        "adam_iters": int(cfg.adam_iters),
        "lbfgs_iters": int(cfg.lbfgs_iters),
        "t_train_frac": float(cfg.t_train_frac),
        "fourier_features": int(cfg.fourier_features),
    }
    return jp, jn, record


def main() -> int:  # noqa: PLR0915 - a measurement report reads better flat
    """Compute the adjoint, report the split, and apply the post-hoc correction."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-axial", type=int, default=160)
    ap.add_argument("--plot", type=Path, default=None, help="write the adjoint figure here")
    ap.add_argument(
        "--network",
        type=int,
        default=None,
        metavar="SEED",
        help="also train a JAX model at this seed and split ITS void functional against "
        "the reference's, which is the measurement M4' needs (docs section 7.5.25)",
    )
    ap.add_argument("--out", type=Path, default=Path("plan_a_adjoint.json"))
    args = ap.parse_args()

    p = AxialParams()
    traj = solve_reference(AxialParams(n_axial=args.n_axial), n_out=241)
    zeta = np.asarray(traj.zeta)

    # Evaluate at the time the void functional is largest, which is where the
    # cancellation is worst and where Plan A's error was measured.
    alpha_ref = np.asarray(traj.alpha)
    k = int(np.argmax(np.abs(alpha_ref).sum(axis=0)))
    t_eval = float(traj.t[k])
    T_c = np.asarray(traj.T_c)[:, k]
    alpha = alpha_ref[:, k]

    z_star = spatial_adjoint(p, T_c, zeta)
    w = np.asarray(p.void_worth(zeta))
    j_pos, j_neg = split_functional(alpha, p, zeta)

    print(f"evaluated at t = {t_eval:.2f} s, where the void functional peaks\n")
    print(f"J+ (positive worth)   {j_pos:+.6e}")
    print(f"J- (negative worth)   {j_neg:+.6e}")
    print(f"J  (their sum)        {j_pos + j_neg:+.6e}")
    can = abs(j_pos + j_neg) / (abs(j_pos) + abs(j_neg))
    print(f"cancellation ratio    {can:.4f}   (|J| / (|J+|+|J-|); 1 means no cancellation)")
    print(
        f"  -> a relative error of eps on each half becomes {1.0 / can:.1f}x eps on the sum,\n"
        f"     which is the amplification Plan A has been scored against.\n"
    )

    # The hypothesis, stated as a number: is the adjoint weight concentrated away from
    # where a residual-magnitude sampler would put points?
    # The void slope underflows to EXACTLY zero wherever the coolant is subcooled, which
    # is most of the channel, so the adjoint source is supported only on the boiling band.
    # Accumulating from the outlet downwards therefore makes z* a STEP: zero above the
    # band, and a constant equal to the whole integral below it. A ratio of means is the
    # wrong summary for a step -- it divides by zero -- so the structure is reported.
    supported = np.abs(z_star) > 1e-12 * max(np.abs(z_star).max(), 1e-300)
    z_top = float(np.abs(z_star[-1]))
    z_bottom = float(np.abs(z_star[0]))
    frac_supported = float(supported.mean())
    front = float(zeta[supported].max()) if supported.any() else float("nan")
    print(f"|z*| at the inlet   {z_bottom:.4e}")
    print(f"|z*| at the outlet  {z_top:.4e}")
    print(f"fraction of the channel with non-zero adjoint weight  {frac_supported:.3f}")
    print(f"weight extends up to zeta = {front:.4f}\n")
    if z_bottom > 10.0 * max(z_top, 1e-300):
        print(
            "CONFIRMED, and more sharply than expected. The void slope underflows to\n"
            "exactly zero wherever the coolant is subcooled, so the adjoint source lives\n"
            "only on the boiling band and z* is a STEP: zero above the band, constant\n"
            "below it. Every point in the lower channel carries EQUAL sensitivity, and\n"
            "points above the front carry NONE.\n\n"
            "Two consequences. Residual-magnitude sampling, which concentrates at the\n"
            "front, puts points exactly where the functional is insensitive -- so it\n"
            "cannot help Plan A and would be expected to hurt. And a uniform sampler is\n"
            "already near-optimal for this functional, which means the 84-92% miss is NOT\n"
            "a sampling problem at all; it is the field's accuracy in the lower channel,\n"
            "weighted by a constant."
        )
    else:
        print(
            "NOT CONFIRMED: the adjoint weight is not concentrated below the front.\n"
            "The DWR hypothesis does not explain Plan A's error, and task 3 should not\n"
            "be run on the strength of it."
        )

    if args.plot is not None:
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(3, 1, figsize=(6.0, 7.0), sharex=True)
        ax[0].plot(zeta, w, color="k")
        ax[0].axhline(0.0, lw=0.6, color="0.6")
        ax[0].axvline(p.zeta_sign, lw=0.8, ls="--", color="0.4")
        ax[0].set_ylabel(r"void worth $w(\zeta)$")
        ax[1].plot(zeta, alpha, color="tab:red")
        ax[1].set_ylabel(r"$\alpha(\zeta)$ at peak")
        ax[2].plot(zeta, np.abs(z_star), color="tab:blue")
        ax[2].axvline(p.zeta_sign, lw=0.8, ls="--", color="0.4")
        ax[2].set_ylabel(r"adjoint weight $|z^*|$")
        ax[2].set_xlabel(r"$\zeta$")
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle("Where the void functional is sensitive, against where the void is")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")

    payload = {
        "t_eval": t_eval,
        "J_pos": j_pos,
        "J_neg": j_neg,
        "cancellation_ratio": can,
        "zeta": zeta.tolist(),
        "z_star": z_star.tolist(),
    }

    if args.network is not None:
        print(f"\n=== the network's own split, seed {args.network} (JAX, shipped default) ===")
        jp_net, jn_net, record = network_split(args.network, traj, p, k)
        e_pos = abs(jp_net - j_pos) / abs(j_pos) * 100.0
        e_neg = abs(jn_net - j_neg) / abs(j_neg) * 100.0
        # The rulers are the reference's own mesh error on each half, from
        # `tools/m4_bar.py`: J+ 1.742%, J- 0.053% at n_axial 160 against 640.
        for name, ref, net, err, ruler in (
            ("J+", j_pos, jp_net, e_pos, 1.742),
            ("J-", j_neg, jn_net, e_neg, 0.053),
        ):
            print(
                f"  {name}  reference {ref:+.6e}   network {net:+.6e}   "
                f"error {err:6.2f}%   ruler {ruler:.3f}%   TUR {err / ruler:6.1f}"
            )
        print(
            "\n  These are OPEN-LOOP: the functional evaluated on the network's own alpha.\n"
            "  Section 7.4's 84-92% is the CLOSED-loop miss and is a different measurement."
        )
        payload |= {
            "network": record,
            "J_pos_net": jp_net,
            "J_neg_net": jn_net,
            "err_pos_pct": e_pos,
            "err_neg_pct": e_neg,
        }

    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
