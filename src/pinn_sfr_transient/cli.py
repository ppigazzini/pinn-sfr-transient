"""Command-line interface: ``pinn-sfr reference`` runs the numpy/scipy pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.figures import DEFAULT_OUTDIR, generate_all
from pinn_sfr_transient.physics import void_fraction
from pinn_sfr_transient.reference import solve_reference


def _run_reference(args: argparse.Namespace) -> None:
    # Produces the held-out reference *data* only (the .npz consumed by the PINN
    # trainers). Figures are the job of `pinn-sfr figures` -> docs/img/, so PNGs
    # live in exactly one place.
    p = SFRParams(t_end=args.t_end)
    traj = solve_reference(p, n_out=args.n_out)

    i_peak = int(np.argmax(traj.P))
    print("ULOF reference transient — summary")
    print(f"  peak power  P_max = {traj.P.max():.3f}  at t = {traj.t[i_peak]:.2f} s")
    print(f"  final power P_end = {traj.P[-1]:.3f}")
    print(f"  peak T_f          = {traj.Tf.max():.1f} K")
    print(f"  peak T_c          = {traj.Tc.max():.1f} K")
    print(f"  peak void         = {void_fraction(traj.Tc.max(), p):.4f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / "ulof_reference.npz"
    np.savez(npz, t=traj.t, P=traj.P, C=traj.C, Tf=traj.Tf, Tc=traj.Tc)
    print(f"  trajectory -> {npz}")
    print("  (figures: run `pinn-sfr figures` -> docs/img/)")


def _run_figures(args: argparse.Namespace) -> None:
    paths = generate_all(args.outdir, with_pinn=not args.no_pinn, safety_n=args.safety_n)
    print(f"Wrote {len(paths)} figures to {args.outdir}/:")
    for path in paths:
        print(f"  {path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``pinn-sfr`` argument parser with its sub-commands."""
    parser = argparse.ArgumentParser(prog="pinn-sfr", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    ref = sub.add_parser("reference", help="run the stiff reference sim -> held-out .npz")
    ref.add_argument("--t-end", type=float, default=60.0, help="transient horizon [s]")
    ref.add_argument("--n-out", type=int, default=2000, help="output samples")
    ref.add_argument("--outdir", type=str, default="results", help="output directory")
    ref.set_defaults(func=_run_reference)

    fig = sub.add_parser("figures", help="regenerate the README/docs figures (-> docs/img/)")
    fig.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="output directory")
    fig.add_argument("--no-pinn", action="store_true", help="skip the optional PINN overlay")
    fig.add_argument("--safety-n", type=int, default=16, help="safety-map grid resolution")
    fig.set_defaults(func=_run_figures)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the selected sub-command."""
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        # Default action when no sub-command is given.
        args = parser.parse_args(["reference"])
    args.func(args)


if __name__ == "__main__":
    main()
