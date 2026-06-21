"""Parameters for the SFR unprotected-loss-of-flow (ULOF) transient.

All quantities are in a consistent *scaled* unit system: power ``P`` is
normalised to 1.0 at nominal steady state, temperatures are in kelvin,
reactivity is dimensionless (absolute units, with ``beta_eff`` in the same
units), and time is in seconds. Heat capacities and the source scale ``P0`` are
chosen so that the nominal steady state defined here is exact.

The delayed-neutron data use the U-235 six-group *relative* spectrum rescaled to
``beta_eff``; SFR cores have a smaller effective delayed fraction than thermal
U-235 systems, hence ``beta_eff ~ 3.5e-3``. The sodium void coefficient
``alpha_void`` is POSITIVE: the defining, safety-relevant feature of large SFR
cores and the reason ULOF is the limiting unprotected transient for this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

# U-235 six-group reference delayed-neutron data.
_LAMBDA_I: FloatArray = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
_BETA_I_U235: FloatArray = np.array([0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273])


@dataclass(slots=True)
class SFRParams:
    """Configuration for the coupled point-kinetics / thermal-hydraulics model."""

    # --- Neutron kinetics ---
    beta_eff: float = 3.5e-3
    Lambda: float = 5.0e-7
    lambda_i: FloatArray = field(default_factory=_LAMBDA_I.copy)

    # --- Reference steady-state temperatures / power scale ---
    Tf0: float = 1000.0
    Tc0: float = 700.0
    Tin: float = 628.0
    P0: float = 1.0

    # --- Thermal time constants / capacities (some derived in __post_init__) ---
    tau_f: float = 8.0
    Cc_val: float = 0.05

    # --- ULOF pump coast-down ---
    tau_pump: float = 5.0
    f_nc: float = 0.15

    # --- Reactivity feedback coefficients ---
    alpha_f: float = -2.0e-5  # Doppler (fuel), < 0
    alpha_c: float = -1.0e-5  # coolant density/expansion, < 0
    alpha_void: float = 6.0e-3  # sodium void, > 0
    rho_ext: float = 0.0  # external insertion (criticality offset added below)

    # --- Sodium boiling / void onset ---
    T_onset: float = 820.0
    dT_void: float = 25.0

    # --- Time domain ---
    t_end: float = 60.0

    # --- Derived (not constructor arguments) ---
    beta_i: FloatArray = field(init=False)
    UA: float = field(init=False)
    W0: float = field(init=False)
    Cf: float = field(init=False)
    Cc: float = field(init=False)

    def __post_init__(self) -> None:
        # Rescale the U-235 relative spectrum to the requested beta_eff.
        rel = _BETA_I_U235 / _BETA_I_U235.sum()
        self.beta_i = rel * self.beta_eff

        # Derive thermal constants so the nominal steady state is exact:
        #   fuel:    P0 = UA (Tf0 - Tc0)
        #   coolant: UA (Tf0 - Tc0) = W0 (Tc0 - Tin)
        self.UA = self.P0 / (self.Tf0 - self.Tc0)
        self.W0 = self.P0 / (self.Tc0 - self.Tin)
        self.Cf = self.tau_f * self.UA
        self.Cc = self.Cc_val

        # The smooth void model has a small non-zero tail at the nominal coolant
        # temperature; absorb it into the control reactivity so rho(Tf0, Tc0) == 0
        # and the nominal state is a true fixed point (un-cancelled, the 1/Lambda
        # amplification would otherwise dominate dP/dt).
        phi0 = 0.5 * (1.0 + np.tanh(0.5 * (self.Tc0 - self.T_onset) / self.dT_void))
        self.rho_ext -= self.alpha_void * phi0

    def steady_precursors(self, P: float = 1.0) -> FloatArray:
        """Steady-state delayed-precursor concentrations for the given power."""
        return self.beta_i * P / (self.Lambda * self.lambda_i)

    def steady_state(self) -> FloatArray:
        """Nominal steady-state vector ``y0 = [P, C_1..C_6, T_f, T_c]``."""
        y0 = np.empty(9, dtype=np.float64)
        y0[0] = 1.0
        y0[1:7] = self.steady_precursors(1.0)
        y0[7] = self.Tf0
        y0[8] = self.Tc0
        return y0
