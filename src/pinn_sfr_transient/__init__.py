"""Physics-Informed Neural Network for the SFR ULOF transient."""

from __future__ import annotations

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.physics import (
    flow_fraction,
    make_rhs,
    reactivity,
    void_fraction,
)
from pinn_sfr_transient.reference import Trajectory, solve_reference

__version__ = "0.1.0"

__all__ = [
    "SFRParams",
    "Trajectory",
    "flow_fraction",
    "make_rhs",
    "reactivity",
    "solve_reference",
    "void_fraction",
]
