"""1D axial SFR boiling model (SAS4A/SASSYS-1 formulation).

A second, higher-fidelity model living alongside the lumped 0D one in the parent
package. Where the 0D model uses a demonstration void ramp at 820 K, this one
follows the SAS4A/SASSYS-1 manual (ANL/NSE-SAS/5.8.1): axially resolved fuel,
cladding, coolant and structure temperatures, real sodium properties, and the
saturation-plus-superheat boiling criterion.

Every deviation from the manual is registered in ``docs/axial_physics.md`` with
the equation number it departs from. Nothing here changes the 0D model.

Milestone status: **M7**. Implemented: parameters and axial shapes (M0); the
section 12.13 sodium correlations (M1); the Chapter 3 energy balance and its
stiff reference solver (M2); the PINN with prescribed power (M3); boiling onset
and the mixture void field (M4); film degradation and dryout (M5); the
prompt-jump kinetics closure in both the reference and the PINN (M6); and the
JAX/Equinox twin (M7).

**The reference solver is verified; the PINN is not converged.** It reproduces
the reference to 3-12% relative L2 against a 1% bar, and part of that gap is the
reference itself, which is only ~10% accurate in the void fraction at the default
mesh. ``docs/axial_nn.md`` records every measurement, including the negative ones;
``docs/axial_physics.md`` section 6.5 records the reference limitation. Do not
quote accuracy numbers from this package.
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
