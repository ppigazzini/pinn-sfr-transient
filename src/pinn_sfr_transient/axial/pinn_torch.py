"""PyTorch axial PINN — public surface.

The implementation lives in :mod:`pinn_sfr_transient.axial.torchpinn`, split into
config, architectures, ansatz, model, weighting, training and evaluation after
`jaxpi2 <https://github.com/sifanexisted/jaxpi2>`_ and mirroring
:mod:`pinn_sfr_transient.axial.jaxpinn`. This module re-exports it so
``from pinn_sfr_transient.axial import pinn_torch`` and
``python -m pinn_sfr_transient.axial.pinn_torch`` keep working, and so the two
backends stay symmetrical at the import level.

Run (after ``uv sync --extra torch-cpu``)::

    uv run python -m pinn_sfr_transient.axial.pinn_torch
"""

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference
from pinn_sfr_transient.axial.torchpinn import *  # noqa: F403
from pinn_sfr_transient.axial.torchpinn import relative_l2, train


def main() -> None:
    """Train, then report the relative L2 error against the M2 reference."""
    p = AxialParams()
    model = train(p)
    traj = solve_reference(p, n_out=201)
    print("\nRelative L2 vs the M2 reference:")
    for k, v in relative_l2(model, traj).items():
        print(f"  {k:4s}: {v:.3e}")


if __name__ == "__main__":
    main()
