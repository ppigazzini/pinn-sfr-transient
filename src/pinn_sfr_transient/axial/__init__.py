"""1D axial SFR boiling model (SAS4A/SASSYS-1 formulation).

A second, higher-fidelity model living alongside the lumped 0D one in the parent
package. Where the 0D model uses a demonstration void ramp at 820 K, this one
follows the SAS4A/SASSYS-1 manual (ANL/NSE-SAS/5.8.1): axially resolved fuel,
cladding, coolant and structure temperatures, real sodium properties, and the
saturation-plus-superheat boiling criterion.

Every deviation from the manual is registered in ``docs/axial_physics.md`` with
the equation number it departs from. Nothing here changes the 0D model.

Milestone status: **M2**. This package provides the parameter container and
axial shapes (M0), the section 12.13 sodium property correlations (M1), and the
single-phase Chapter 3 thermal-hydraulics with its stiff reference solver (M2).
Boiling onset (M4) and the kinetics closure (M6) are not implemented yet, so
power is still prescribed.
"""

from __future__ import annotations

from pinn_sfr_transient.axial import physics, sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import (
    AxialTrajectory,
    energy_balance,
    solve_reference,
    steady_state,
)

__all__ = [
    "AxialParams",
    "AxialTrajectory",
    "energy_balance",
    "physics",
    "sodium",
    "solve_reference",
    "steady_state",
]
