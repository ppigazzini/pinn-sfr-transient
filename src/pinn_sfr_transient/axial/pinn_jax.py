"""JAX/Equinox twin of the axial PINN — public surface.

The implementation lives in :mod:`pinn_sfr_transient.axial.jaxpinn`, split into
architecture, ansatz, residuals, weighting, samplers, training and evaluation
after `jaxpi2 <https://github.com/sifanexisted/jaxpi2>`_. This module re-exports
it so ``from pinn_sfr_transient.axial import pinn_jax`` and
``python -m pinn_sfr_transient.axial.pinn_jax`` keep working, and so the two
backends stay symmetrical at the import level.

Run (after ``uv sync --extra jax-cpu``; ``--extra jax-gpu`` for CUDA)::

    uv run python -m pinn_sfr_transient.axial.pinn_jax
"""

from __future__ import annotations

from pinn_sfr_transient.axial.jaxpinn import *  # noqa: F403
from pinn_sfr_transient.axial.jaxpinn import (
    _ALPHA_GATE,  # noqa: F401 - the parity test asserts it equals the torch twin's
    _EXP_BOUND,  # noqa: F401
    _collocation,  # noqa: F401
    _merge,  # noqa: F401
    _power_integral,  # noqa: F401
    _power_shape,  # noqa: F401
    _rar_points,  # noqa: F401
    relative_l2,
    train,
)
from pinn_sfr_transient.axial.reference import solve_reference


def main() -> None:
    """Train and report the relative L2 error against the reference."""
    model, p, cfg = train()
    traj = solve_reference(p, n_out=201, feedback=cfg.feedback)
    print("\nRelative L2 vs the reference:")
    for k, v in relative_l2(model, p, traj, cfg).items():
        print(f"  {k:18s}: {v:.3e}")


if __name__ == "__main__":
    main()
