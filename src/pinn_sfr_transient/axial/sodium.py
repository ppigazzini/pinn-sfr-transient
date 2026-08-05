"""Sodium thermophysical properties (SAS4A/SASSYS-1 section 12.13).

The eleven property correlations and two derived relations of section 12.13 of
the SAS4A/SASSYS-1 manual (ANL/NSE-SAS/5.8.1,
https://sas-doc.nse.anl.gov/latest/Part04/Ch12/12.13.html), transcribed with
their equation numbers. These replace the 0D model's demonstration void ramp at
820 K with real sodium: at 1 atm the saturation temperature is ~1154 K.

**One implementation, three backends.** Every function dispatches on its
argument type, so numpy arrays, torch tensors and JAX arrays evaluate the *same*
expression tree rather than three transcriptions that could drift apart. What
that buys is exact: the pure-polynomial correlations are **bit-identical** across
all three, since IEEE-754 pins ``+`` and ``*``; the ones using ``exp``, ``log``
or division agree to **~1 ULP**, because the backends call different libm
implementations and IEEE-754 does not require correctly-rounded transcendentals.
Both bounds are asserted in ``tests/axial/test_sodium.py``. All functions are
pure, closed-form and differentiable over the validity range, so they can sit
inside a PINN residual and be differentiated by ``torch.autograd`` or
``jax.grad`` alike.

Carrying all three is a **correctness mechanism**, not bookkeeping:
``docs/neural_network.md`` §9 records that the PyTorch initialisation bug (which
left the power trajectory at ``L2 ~ 0.3``) was identified precisely because the
JAX twin fit well at the same budget. A discrepancy between backends is a signal.

JAX users must enable float64 (``jax.config.update("jax_enable_x64", True)``, as
``pinn_jax`` does at import); at the default float32 these correlations lose
about eight significant digits, and the ``Ts(Ps(T))`` round-trip degrades from
1e-11 K to ~1e-2 K.

**Validity: 590 K to 2270 K** (:data:`T_MIN`, :data:`T_MAX`). The manual stops
the polynomial fits at ~90% of the critical point (:data:`T_CRIT` = 2503.3 K) to
avoid fitting the rapid near-critical variation. Several correlations contain
``1/(T_c - T)`` and diverge at the critical point. **Nothing here clamps or
raises** — a hard guard would break autodiff and abort training on a transient
excursion. Call :func:`in_range` if you need to check, and see
``docs/axial_physics.md``.

Quoted fit accuracies against the underlying Padilla and Fink-Leibowitz
correlations: latent heat 1%, saturation pressure 1.2%, liquid density 0.5%,
vapour density 1.5%, liquid heat capacity 1.5%, vapour heat capacity 1%,
compressibility 0.1%, expansion 0.7%, conductivity 0.5%, viscosity 0.5%.
"""

from __future__ import annotations

from typing import Any

from pinn_sfr_transient.axial._backend import xp as _xp
from pinn_sfr_transient.config import FloatArray

type Scalar = float | FloatArray

# --- Validity range and critical point (manual section 12.13 preamble) ------
T_MIN: float = 590.0
"""Lower bound of the polynomial fits [K]."""

T_MAX: float = 2270.0
"""Upper bound of the polynomial fits [K] (~90% of the critical point)."""

T_CRIT: float = 2503.3
"""Sodium critical temperature [K]. Several correlations diverge here."""

PS_MIN: float = 3.5
"""Lower bound on saturation pressure for Eq. 12.13-4 [Pa]."""

PS_MAX: float = 1.6e7
"""Upper bound on saturation pressure for Eq. 12.13-4 [Pa]."""

# --- Correlation coefficients, verbatim from section 12.13 -----------------
# Heat of vaporization, Eq. 12.13-1 (the manual gives J/kg; the older
# ANL/NE-16/19 printing labelled it J/g, which the coefficients contradict).
_A1, _A2, _A3, _A4 = 5.3139e6, -2.0296e3, 1.0625, -3.3163e-4
# Saturation vapour pressure, Eq. 12.13-2
_A5, _A6, _A7 = 2.169e1, 1.14846e4, 3.41769e5
# Saturation temperature, Eq. 12.13-4 — the analytic inverse of Eq. 12.13-2
_A8 = 2.0 * _A7
_A9 = -_A6
_A10 = _A6**2 + 4.0 * _A5 * _A7
_A11 = -4.0 * _A7
# Liquid density, Eq. 12.13-5
_A12, _A13, _A14 = 1.00423e3, -0.21390, -1.1046e-5
# Vapour density, Eq. 12.13-6
_A15, _A16, _A17 = 4.1444e-3, -7.4461e-6, 1.3768e-8
_A18, _A19, _A20 = -1.0834e-11, 3.8903e-15, -4.922e-19
# Liquid heat capacity, Eq. 12.13-7
_A28, _A29, _A30 = 7.3898e5, 3.1514e5, 1.1340e3
_A31, _A32 = -2.2153e-1, 1.1156e-4
# Vapour heat capacity, Eq. 12.13-8
_A33, _A34, _A35 = 2.1409e3, -2.2401e1, 7.9787e-2
_A36, _A37 = -1.0618e-4, 6.7874e-8
_A38, _A39 = -2.1127e-11, 2.5834e-15
# Liquid adiabatic compressibility, Eq. 12.13-9
_A40, _A41 = -5.4415e-11, 4.7663e-7
# Liquid thermal expansion coefficient, Eq. 12.13-10
_A42, _A43, _A44 = 2.5156e-6, 0.79919, -6.9716e2
_A45, _A46, _A47 = 3.3140e5, -7.0502e7, 5.4920e9
# Liquid thermal conductivity, Eq. 12.13-11
_A48, _A49, _A50, _A51 = 1.1045e2, -6.5112e-2, 1.5430e-5, -2.4617e-9
# Liquid viscosity, Eq. 12.13-12
_A52, _A53, _A54, _A55 = 3.6522e-5, 0.16626, -4.56877e1, 2.8733e4
# Enthalpy of saturated liquid, Eq. 12.13-13
_A56, _A57, _A58, _A59 = -111136.04, 1722.2578, -0.45544483, 1.4692883e-4


def in_range(T: Scalar) -> Any:  # noqa: ANN401 - bool array of the caller's backend
    """Report where ``T`` lies inside the fitted range ``[T_MIN, T_MAX]``.

    Provided so callers can *diagnose* an out-of-range excursion; the property
    functions themselves neither clamp nor raise, because a hard guard would
    break autodiff and abort a training run mid-transient.
    """
    return (T >= T_MIN) & (T <= T_MAX)


# --- Phase change ----------------------------------------------------------
def latent_heat(T: Scalar) -> Scalar:
    """Heat of vaporization [J/kg] — Eq. 12.13-1.

    ``lambda = A1 + A2 T + A3 T^2 + A4 T^3``. Fits Padilla's two recommended
    equations (Golden-Tokar below 1644 K, Miller-Cohen-Dickerman above) to 1%.
    """
    return _A1 + T * (_A2 + T * (_A3 + T * _A4))


def saturation_pressure(T: Scalar) -> Scalar:
    """Saturation vapour pressure [Pa] — Eq. 12.13-2.

    ``ln(Ps) = A5 - A6/T - A7/T^2``. A least-squares fit to Fink and Leibowitz
    (Eq. 12.13-3) chosen because it inverts analytically, which is what
    :func:`saturation_temperature` needs; accurate to 1.2%.
    """
    xp = _xp(T)
    return xp.exp(_A5 - _A6 / T - _A7 / T**2)


def saturation_temperature(Ps: Scalar) -> Scalar:
    """Saturation temperature [K] — Eq. 12.13-4.

    ``Ts = A8 / (A9 + sqrt(A10 + A11 ln(Ps)))``, the exact analytic inverse of
    Eq. 12.13-2 (solve the quadratic in ``1/T``), so ``Ts(Ps(T)) == T`` to
    floating-point precision. Valid for ``PS_MIN <= Ps <= PS_MAX``.
    """
    xp = _xp(Ps)
    return _A8 / (_A9 + (_A10 + _A11 * xp.log(Ps)) ** 0.5)


# --- Densities -------------------------------------------------------------
def liquid_density(T: Scalar) -> Scalar:
    """Saturated liquid density [kg/m^3] — Eq. 12.13-5.

    ``rho_l = A12 + A13 T + A14 T^2``; fits Fink and Leibowitz to 0.5%.
    """
    return _A12 + T * (_A13 + T * _A14)


def vapor_density(T: Scalar) -> Scalar:
    """Saturated vapour density [kg/m^3] — Eq. 12.13-6.

    ``rho_v = Ps (A15/T + A16 + A17 T + A18 T^2 + A19 T^3 + A20 T^4)``, with
    ``Ps`` from Eq. 12.13-2. Replaces Padilla's quasichemical (below 1644 K) and
    Clapeyron (above) forms with one polynomial, accurate to 1.5%.
    """
    poly = _A15 / T + _A16 + T * (_A17 + T * (_A18 + T * (_A19 + T * _A20)))
    return saturation_pressure(T) * poly


# --- Heat capacities -------------------------------------------------------
def liquid_heat_capacity(T: Scalar) -> Scalar:
    """Saturated liquid specific heat [J/kg-K] — Eq. 12.13-7.

    ``C_l = A28/(Tc-T)^2 + A29/(Tc-T) + A30 + A31 (Tc-T) + A32 (Tc-T)^2``.
    Diverges at the critical point; fits Padilla to 1.5%.
    """
    d = T_CRIT - T
    return _A28 / d**2 + _A29 / d + _A30 + d * (_A31 + d * _A32)


def vapor_heat_capacity(T: Scalar) -> Scalar:
    """Saturated vapour specific heat [J/kg-K] — Eq. 12.13-8.

    Sixth-order polynomial ``A33 ... A39``; approximates Padilla's quasichemical
    and Rowlinson's correlations to better than 1%.
    """
    return _A33 + T * (_A34 + T * (_A35 + T * (_A36 + T * (_A37 + T * (_A38 + T * _A39)))))


# --- Mechanical and transport properties -----------------------------------
def liquid_compressibility(T: Scalar) -> Scalar:
    """Liquid adiabatic compressibility [1/Pa] — Eq. 12.13-9.

    ``beta_s = A40 + A41/(Tc - T)``; fits to better than 0.1%.
    """
    return _A40 + _A41 / (T_CRIT - T)


def liquid_expansion(T: Scalar) -> Scalar:
    """Liquid thermal expansion coefficient [1/K] — Eq. 12.13-10.

    Fifth-order expansion in ``1/(Tc - T)``, coefficients ``A42 ... A47``; fits
    Fink and Leibowitz to better than 0.7%.
    """
    d = T_CRIT - T
    return _A42 + (_A43 + (_A44 + (_A45 + (_A46 + _A47 / d) / d) / d) / d) / d


def liquid_conductivity(T: Scalar) -> Scalar:
    """Liquid thermal conductivity [W/m-K] — Eq. 12.13-11.

    ``k_l = A48 + A49 T + A50 T^2 + A51 T^3``; approximates Fink and Leibowitz
    to 0.5%.
    """
    return _A48 + T * (_A49 + T * (_A50 + T * _A51))


def liquid_viscosity(T: Scalar) -> Scalar:
    """Liquid dynamic viscosity [Pa-s] — Eq. 12.13-12.

    ``mu_l = A52 + A53/T + A54/T^2 + A55/T^3``; approximates Fink and Leibowitz
    to 0.5%.
    """
    return _A52 + (_A53 + (_A54 + _A55 / T) / T) / T


def saturated_liquid_enthalpy(T: Scalar) -> Scalar:
    """Enthalpy of saturated liquid [J/kg] — Eq. 12.13-13.

    ``H = A56 + A57 T + A58 T^2 + A59 T^3``. Present in ANL/NSE-SAS/5.8.1 and
    absent from the older ANL/NE-16/19 printing. The manual notes it is used
    only by the PLUTO and LEVITATE fuel-motion models, which are out of scope
    here; it is included for completeness of section 12.13.
    """
    return _A56 + T * (_A57 + T * (_A58 + T * _A59))
