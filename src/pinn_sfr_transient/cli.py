"""Command-line interface for both models.

``pinn-sfr reference`` and ``pinn-sfr figures`` drive the lumped 0D model;
``pinn-sfr axial ...`` drives the 1D axial boiling model.
"""

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


def _run_axial_reference(args: argparse.Namespace) -> None:
    """Solve the axial channel and write the held-out trajectory."""
    from pinn_sfr_transient.axial.config import AxialParams  # noqa: PLC0415
    from pinn_sfr_transient.axial.reference import energy_balance, solve_reference  # noqa: PLC0415

    p = AxialParams(n_axial=args.n_axial, t_end=args.t_end)
    traj = solve_reference(p, n_out=args.n_out, feedback=args.feedback)
    onset_t, onset_z = traj.onset()

    print(f"Axial ULOF — {'closed loop' if args.feedback else 'prescribed power'}")
    print(f"  n_axial            = {p.n_axial}")
    print(f"  boiling onset      = {onset_t:.2f} s at zeta = {onset_z:.3f}")
    print(f"  peak cladding      = {traj.peak_clad:.1f} K")
    print(f"  max voided length  = {traj.voided_length.max():.4f} m")
    print(f"  energy closure     = {energy_balance(traj, p):.2e}")
    print(f"  stopped early      = {traj.stopped_early} (validity limit)")
    if args.feedback:
        b = p.beta_eff
        print(f"  power              = {traj.power.min():.4f} .. {traj.power.max():.4f}")
        print(f"  max rho/beta       = {traj.peak_rho_over_beta:+.4f}   (tripwire, must be < 0.5)")
        print(
            f"  Doppler   [beta]   = {traj.rho_doppler.min() / b:+.4f} .. "
            f"{traj.rho_doppler.max() / b:+.4f}"
        )
        print(
            f"  void      [beta]   = {traj.rho_void.min() / b:+.4f} .. "
            f"{traj.rho_void.max() / b:+.4f}"
        )
        print(f"  void worth exercised = {traj.void_worth_is_exercised()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / "axial_reference.npz"
    np.savez(
        npz,
        t=traj.t,
        zeta=traj.zeta,
        T_f=traj.T_f,
        T_cl=traj.T_cl,
        T_s=traj.T_s,
        T_c=traj.T_c,
        alpha=traj.alpha,
        power=traj.power,
        rho=traj.rho,
        rho_doppler=traj.rho_doppler,
        rho_void=traj.rho_void,
        flow=traj.flow,
    )
    print(f"  trajectory -> {npz}")


def _run_axial_figures(args: argparse.Namespace) -> None:
    """Regenerate the axial figures."""
    from pinn_sfr_transient.axial.figures import generate_all as axial_all  # noqa: PLC0415

    paths = axial_all(args.outdir, n_axial=args.n_axial)
    print(f"Wrote {len(paths)} axial figures to {args.outdir}/:")
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

    axial = sub.add_parser("axial", help="the 1D axial boiling model")
    axial_sub = axial.add_subparsers(dest="axial_command")

    aref = axial_sub.add_parser("reference", help="solve the axial channel -> held-out .npz")
    aref.add_argument("--n-axial", type=int, default=160, help="axial nodes (>=160 is converged)")
    aref.add_argument("--t-end", type=float, default=60.0, help="transient horizon [s]")
    aref.add_argument("--n-out", type=int, default=241, help="output samples")
    aref.add_argument("--feedback", action="store_true", help="close the prompt-jump kinetics")
    aref.add_argument("--outdir", type=str, default="results", help="output directory")
    aref.set_defaults(func=_run_axial_reference)

    afig = axial_sub.add_parser("figures", help="regenerate the axial figures (-> docs/img/)")
    afig.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="output directory")
    afig.add_argument("--n-axial", type=int, default=160, help="axial nodes")
    afig.set_defaults(func=_run_axial_figures)

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
