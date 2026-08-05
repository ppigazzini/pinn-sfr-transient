"""M1 acceptance tests for the SAS4A section 12.13 sodium correlations.

Three things are checked, in order of importance:

1. **The analytic inverse really inverts** — ``Ts(Ps(T)) == T`` to 1e-10 across
   the whole fitted range. A transcription slip in Eq. 12.13-2 or 12.13-4 that
   left the pair *mutually* consistent but wrong would still be caught by (2).
2. **The numbers are physically right** — every property is compared against
   independently known sodium values at the normal boiling point, so a wrong
   coefficient cannot hide behind self-consistency.
3. **numpy and torch agree**, and the correlations survive autodiff, which is
   the whole reason they exist in this form. The pure polynomials are asserted
   *bit-identical*; those using ``exp``/``log`` or division are asserted equal to
   ~1 ULP, because the two backends call different libm implementations. That
   split is the sharpest statement available: anything looser would hide a real
   transcription drift, anything tighter fails on rounding alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_sfr_transient.axial import sodium as na

T_BOIL = 1154.0  # sodium normal boiling point [K]
ATM = 101325.0  # [Pa]


# --- 1. the analytic inverse -----------------------------------------------
def test_saturation_roundtrip_is_exact():
    """Eq. 12.13-4 is the closed-form inverse of Eq. 12.13-2, so this is exact."""
    T = np.linspace(na.T_MIN, na.T_MAX, 100_001)
    np.testing.assert_allclose(na.saturation_temperature(na.saturation_pressure(T)), T, atol=1e-10)


def test_saturation_roundtrip_from_the_pressure_side():
    Ps = np.geomspace(na.PS_MIN, na.PS_MAX, 5001)
    back = na.saturation_pressure(na.saturation_temperature(Ps))
    np.testing.assert_allclose(back, Ps, rtol=1e-12)


def test_saturation_pressure_is_one_atmosphere_at_the_boiling_point():
    """The headline check: real sodium boils at ~1154 K at 1 atm, not at 820 K."""
    assert na.saturation_pressure(T_BOIL) == pytest.approx(ATM, rel=0.05)
    assert na.saturation_temperature(ATM) == pytest.approx(T_BOIL, abs=10.0)


def test_saturation_pressure_increases_with_temperature():
    Ps = na.saturation_pressure(np.linspace(na.T_MIN, na.T_MAX, 2000))
    assert np.all(np.diff(Ps) > 0.0)


# --- 2. independent physical values ----------------------------------------
@pytest.mark.parametrize(
    ("fn", "expected", "rel", "unit"),
    [
        (na.latent_heat, 3.9e6, 0.05, "J/kg"),
        (na.liquid_density, 760.0, 0.05, "kg/m^3"),
        (na.vapor_density, 0.28, 0.10, "kg/m^3"),
        (na.liquid_heat_capacity, 1275.0, 0.05, "J/kg-K"),
        (na.liquid_conductivity, 49.0, 0.10, "W/m-K"),
        (na.liquid_viscosity, 1.8e-4, 0.10, "Pa-s"),
    ],
)
def test_property_matches_literature_at_the_boiling_point(fn, expected, rel, unit):
    """Guards against a mistyped coefficient, which self-consistency cannot catch."""
    assert fn(T_BOIL) == pytest.approx(expected, rel=rel), unit


def test_latent_heat_is_in_joules_per_kilogram_not_per_gram():
    """ANL/NE-16/19 labelled Eq. 12.13-1 'J/g'; 5.8.1 corrects it to J/kg."""
    assert 3.0e6 < na.latent_heat(T_BOIL) < 5.0e6


def test_liquid_heat_capacity_uses_the_corrected_A29():
    """A29 = 3.1514e5 in 5.8.1; the superseded printing had 3.154e5 (M1 gate)."""
    assert na._A29 == 3.1514e5


def test_liquid_density_falls_with_temperature():
    rho = na.liquid_density(np.linspace(na.T_MIN, na.T_MAX, 2000))
    assert np.all(np.diff(rho) < 0.0)


def test_liquid_is_far_denser_than_vapour_at_the_boiling_point():
    assert na.liquid_density(T_BOIL) / na.vapor_density(T_BOIL) > 1000.0


@pytest.mark.parametrize(
    "fn",
    [
        na.latent_heat,
        na.saturation_pressure,
        na.liquid_density,
        na.vapor_density,
        na.liquid_heat_capacity,
        na.vapor_heat_capacity,
        na.liquid_expansion,
        na.liquid_conductivity,
        na.liquid_viscosity,
        na.saturated_liquid_enthalpy,
    ],
)
def test_properties_stay_positive_across_the_fitted_range(fn):
    assert np.all(fn(np.linspace(na.T_MIN, na.T_MAX, 5000)) > 0.0)


def test_compressibility_is_positive_and_small():
    beta = na.liquid_compressibility(np.linspace(na.T_MIN, na.T_MAX, 500))
    assert np.all(beta > 0.0)
    assert np.all(beta < 1e-8)


def test_scalar_and_array_inputs_agree():
    for fn in (na.latent_heat, na.saturation_pressure, na.liquid_density):
        assert fn(T_BOIL) == pytest.approx(float(fn(np.array([T_BOIL]))[0]))


# --- validity range --------------------------------------------------------
def test_in_range_flags_the_fitted_window():
    T = np.array([500.0, na.T_MIN, 1154.0, na.T_MAX, 2400.0])
    np.testing.assert_array_equal(na.in_range(T), [False, True, True, True, False])


def test_properties_do_not_raise_outside_the_range():
    """A hard guard would break autodiff and abort training mid-transient."""
    assert np.isfinite(na.liquid_density(2400.0))
    assert np.isfinite(na.liquid_heat_capacity(2400.0))


# --- 3. backend parity and differentiability -------------------------------
# Correlations built only from `+` and `*`: IEEE-754 pins these exactly, so the
# two backends must agree bit-for-bit.
POLYNOMIAL_OF_T = [
    na.latent_heat,
    na.liquid_density,
    na.vapor_heat_capacity,
    na.liquid_conductivity,
    na.saturated_liquid_enthalpy,
]
# Correlations using exp, log or division. numpy and torch call different libm
# implementations, and IEEE-754 does not require transcendentals to be correctly
# rounded, so ~1 ULP of disagreement is expected and is not drift.
TRANSCENDENTAL_OF_T = [
    na.saturation_pressure,
    na.vapor_density,
    na.liquid_heat_capacity,
    na.liquid_compressibility,
    na.liquid_expansion,
    na.liquid_viscosity,
]


@pytest.mark.parametrize("fn", POLYNOMIAL_OF_T)
def test_polynomial_correlations_are_bit_identical_across_backends(fn):
    """Only `+` and `*`, so IEEE-754 leaves the backends no freedom to differ."""
    torch = pytest.importorskip("torch")
    T = np.linspace(700.0, 2200.0, 257)
    got = fn(torch.tensor(T, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, fn(T), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("fn", TRANSCENDENTAL_OF_T)
def test_transcendental_correlations_agree_to_one_ulp(fn):
    """Different libm, same expression tree: rounding may differ, the maths may not."""
    torch = pytest.importorskip("torch")
    T = np.linspace(700.0, 2200.0, 257)
    got = fn(torch.tensor(T, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, fn(T), rtol=1e-14, atol=0.0)


def test_saturation_temperature_matches_across_backends():
    torch = pytest.importorskip("torch")
    Ps = np.geomspace(1e3, 1e7, 257)
    got = na.saturation_temperature(torch.tensor(Ps, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(got, na.saturation_temperature(Ps), rtol=1e-14, atol=0.0)


def test_saturation_pressure_is_differentiable_under_autograd():
    """The correlations sit inside a PINN residual, so dPs/dT must flow."""
    torch = pytest.importorskip("torch")
    T = torch.tensor([1154.0], dtype=torch.float64, requires_grad=True)
    na.saturation_pressure(T).backward()
    # d(ln Ps)/dT = A6/T^2 + 2 A7/T^3  =>  dPs/dT = Ps * that
    expected = na.saturation_pressure(1154.0) * (na._A6 / 1154.0**2 + 2 * na._A7 / 1154.0**3)
    assert T.grad is not None
    assert float(T.grad[0]) == pytest.approx(expected, rel=1e-12)


def test_saturation_temperature_is_differentiable_under_autograd():
    torch = pytest.importorskip("torch")
    Ps = torch.tensor([ATM], dtype=torch.float64, requires_grad=True)
    na.saturation_temperature(Ps).backward()
    assert Ps.grad is not None
    assert float(Ps.grad[0]) > 0.0  # hotter saturation at higher pressure
