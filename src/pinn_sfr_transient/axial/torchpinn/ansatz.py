"""Closed forms the ansatz is built from.

The axial power shape and its integral, the fuel-temperature inversion, and the
precursor parameterisation. Written against tensors rather than numpy so they
stay inside the autodiff graph.
"""

from typing import TYPE_CHECKING

try:
    import torch  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams

import numpy as np

from pinn_sfr_transient.axial.torchpinn.archs import _bounded_exp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.torchpinn.model import AxialPinn


def _precursors(model: AxialPinn, that: torch.Tensor) -> torch.Tensor:
    """``c_i(t_hat)`` with ``c(0) = 1`` exact and ``c > 0`` guaranteed.

    ``c = exp(t_hat N(t_hat))``, bounded by :func:`_bounded_exp`, does both: the
    ``t_hat`` factor pins the initial condition for any weights, and the
    exponential makes the precursors positive by construction. Positivity is
    what makes
    ``P = sum(beta_i c_i)/(beta - rho)`` unable to reach zero — the collapse mode
    that REPORT-01 section 5.2 spends its length on.
    """
    return _bounded_exp(that * model.kin(that))


def _power_shape(p: AxialParams, zeta: torch.Tensor) -> torch.Tensor:
    """Axial power shape on a tensor (closed form, so autodiff-safe)."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (np.pi * k)) * np.sin(0.5 * np.pi * k)
    return torch.cos(np.pi * k * (zeta - 0.5)) / norm


def _power_integral(p: AxialParams, zeta: torch.Tensor) -> torch.Tensor:
    """Cumulative axial power fraction ``F(zeta)`` on a tensor; ``F(0)=0``, ``F(1)=1``."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    half = 0.5 * np.pi * k
    return (torch.sin(np.pi * k * (zeta - 0.5)) + np.sin(half)) / (2.0 * np.sin(half))


def _fuel_temperature(
    q: torch.Tensor, T_cl: torch.Tensor, area: float, p: AxialParams, iters: int = 5
) -> torch.Tensor:
    """Invert Eq. 3.3-4 for the fuel temperature; radiation makes it nonlinear.

    A fixed unrolled Newton rather than a convergence loop: the iteration count
    must not depend on the data for the graph to be traceable, and 40 steps is
    past convergence for this smooth scalar problem: measured, 4 iterations already
    reach machine precision, so 5 carries a margin of one.
    """
    sigma = 5.670374419e-8
    T_f = T_cl + q / (p.h_gap * area)
    for _ in range(iters):
        f = area * (p.h_gap * (T_f - T_cl) + p.emissivity * sigma * (T_f**4 - T_cl**4)) - q
        T_f = T_f - f / (area * (p.h_gap + 4.0 * p.emissivity * sigma * T_f**3))
    return T_f
