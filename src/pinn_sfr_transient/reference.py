"""High-accuracy stiff reference integrator for the ULOF transient.

This is the held-out "ground truth" used only for final PINN validation (the
physics teaches the network; reference data is reserved for test-time metrics).
The system is stiff (``Lambda ~ 5e-7`` with precursors ``~1e4-1e5``), so an
implicit solver with per-component absolute tolerances is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from pinn_sfr_transient.config import FloatArray, SFRParams
from pinn_sfr_transient.physics import make_rhs


@dataclass(slots=True)
class Trajectory:
    """Solved transient on a uniform time grid."""

    t: FloatArray
    P: FloatArray
    C: FloatArray  # shape (6, N)
    Tf: FloatArray
    Tc: FloatArray
    y: FloatArray  # full state, shape (9, N)


def solve_reference(
    p: SFRParams,
    n_out: int = 2000,
    method: str = "Radau",
    rtol: float = 1e-8,
) -> Trajectory:
    """Integrate the coupled system from nominal steady state over ``[0, t_end]``."""
    y0 = p.steady_state()

    # Per-component absolute tolerances: precursors are O(1e4-1e5), so a single
    # scalar atol would be either far too loose for P/T or absurd for C_i.
    atol = np.empty(9, dtype=np.float64)
    atol[0] = 1e-9
    atol[1:7] = 1e-3 * np.abs(y0[1:7])
    atol[7:9] = 1e-6

    t_eval = np.linspace(0.0, p.t_end, n_out)
    sol = solve_ivp(
        make_rhs(p),
        (0.0, p.t_end),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=0.5,
    )
    if not sol.success:
        msg = f"Reference integration failed: {sol.message}"
        raise RuntimeError(msg)

    return Trajectory(
        t=sol.t,
        P=sol.y[0],
        C=sol.y[1:7],
        Tf=sol.y[7],
        Tc=sol.y[8],
        y=sol.y,
    )
