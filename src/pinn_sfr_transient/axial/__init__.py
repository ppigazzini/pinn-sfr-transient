"""1D axial SFR boiling model (SAS4A/SASSYS-1 formulation).

A second, higher-fidelity model living alongside the lumped 0D one in the parent
package. Where the 0D model uses a demonstration void ramp at 820 K, this one
follows the SAS4A/SASSYS-1 manual (ANL/NSE-SAS/5.8.1): axially resolved fuel,
cladding, coolant and structure temperatures, real sodium properties, and the
saturation-plus-superheat boiling criterion.

Every deviation from the manual is registered in ``docs/axial_physics.md`` with
the equation number it departs from. Nothing here changes the 0D model.

Milestone status: **M0 (scaffolding)**. This package currently provides the
parameter container and the axial shape functions only; the physics residuals
(M2), boiling onset (M4) and kinetics closure (M6) are not implemented yet.
"""

from __future__ import annotations

from pinn_sfr_transient.axial.config import AxialParams

__all__ = ["AxialParams"]
