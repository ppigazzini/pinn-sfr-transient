"""Reactivity feedback, void model, flow coast-down and the coupled RHS.

The physics core: reactivity feedback, void model, flow coast-down and the
coupled point-kinetics / thermal-hydraulics right-hand side.

This module is the numpy reference for the model. The ``pinn_torch``,
``pinn_jax`` and ``pinn_deepxde`` backends re-express these same closed-form
residuals in their own frameworks; ``tests/test_consistency`` enforces that the
normalized residuals match the physical ODEs to machine precision.

State vector ``y = [P, C_1, ..., C_6, T_f, T_c]`` (length 9).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pinn_sfr_transient.config import FloatArray, SFRParams

type Scalar = float | FloatArray


def void_fraction(Tc: Scalar, p: SFRParams) -> Scalar:
    """Smooth localized-boiling void fraction in ``[0, 1)``."""
    x = (Tc - p.T_onset) / p.dT_void
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def flow_fraction(t: Scalar, p: SFRParams) -> Scalar:
    """Normalised primary mass-flow during the pump coast-down, ``g(0) = 1``."""
    return p.f_nc + (1.0 - p.f_nc) * np.exp(-t / p.tau_pump)


def reactivity(Tf: Scalar, Tc: Scalar, p: SFRParams) -> Scalar:
    """Total reactivity including temperature and void feedback."""
    phi = void_fraction(Tc, p)
    return p.rho_ext + p.alpha_f * (Tf - p.Tf0) + p.alpha_c * (Tc - p.Tc0) + p.alpha_void * phi


def make_rhs(p: SFRParams) -> Callable[[float, FloatArray], FloatArray]:
    """Build ``f(t, y)`` for :func:`scipy.integrate.solve_ivp`."""
    lam = p.lambda_i
    beta_i = p.beta_i
    beta = p.beta_eff
    Lam = p.Lambda

    def rhs(t: float, y: FloatArray) -> FloatArray:
        P = y[0]
        C = y[1:7]
        Tf = y[7]
        Tc = y[8]

        rho = reactivity(Tf, Tc, p)
        g = flow_fraction(t, p)

        out = np.empty(9, dtype=np.float64)
        out[0] = ((rho - beta) / Lam) * P + float(np.dot(lam, C))
        out[1:7] = (beta_i / Lam) * P - lam * C
        out[7] = (p.P0 / p.Cf) * P - (p.UA / p.Cf) * (Tf - Tc)
        out[8] = (p.UA / p.Cc) * (Tf - Tc) - (p.W0 * g / p.Cc) * (Tc - p.Tin)
        return out

    return rhs
