"""M0 acceptance tests for the axial model's parameter container.

These check the *contract* the rest of the milestones rely on: a valid mesh, a
unit-integral power shape, a void-worth profile that integrates to the requested
net worth and changes sign where it should, and the manual's flooded-to-voided
Doppler interpolation. No physics is solved here — that starts at M2.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinn_sfr_transient.axial import AxialParams
from pinn_sfr_transient.axial.physics import precursor_derivatives
from pinn_sfr_transient.axial.reference import steady_state
from pinn_sfr_transient.config import SFRParams


def _midpoint_integral(f, n=20001):
    """Integral over [0, 1] by the midpoint rule."""
    zeta = (np.arange(n, dtype=np.float64) + 0.5) / n
    return float(np.mean(f(zeta)))


# --- construction ----------------------------------------------------------
def test_defaults_construct():
    p = AxialParams()
    assert p.n_axial >= 2
    assert p.r_fo < p.r_ci < p.r_co


def test_beta_i_sums_to_beta_eff():
    p = AxialParams()
    assert p.beta_i.shape == (6,)
    assert p.beta_i.sum() == pytest.approx(p.beta_eff, rel=1e-12)


def test_delayed_neutron_data_matches_the_0d_model():
    """The two models must share delayed-neutron data or comparisons are meaningless."""
    axial, lumped = AxialParams(), SFRParams()
    np.testing.assert_allclose(axial.lambda_i, lumped.lambda_i)
    np.testing.assert_allclose(axial.beta_i, lumped.beta_i)


def test_steady_precursors_scale_with_power():
    p = AxialParams()
    np.testing.assert_allclose(p.steady_precursors(2.0), 2.0 * p.steady_precursors(1.0))


def test_steady_precursors_are_normalised_not_absolute():
    """``c_i = C_i / C_{i,0}``, so every group sits at the power, not at ``C_i``.

    Returning the absolute ``beta_i P / (Lambda lambda_i)`` — the 0D model's
    convention — is about 5e4 times too large for this state vector, and would
    put ``P(0)`` nowhere near one if it were ever used as an initial condition.
    """
    p = AxialParams()
    np.testing.assert_allclose(p.steady_precursors(1.0), np.ones(6))
    np.testing.assert_allclose(p.steady_precursors(0.5), np.full(6, 0.5))


def test_steady_precursors_match_the_solver_initial_condition():
    """The config's answer and the one the reference actually integrates agree."""
    p = AxialParams()
    np.testing.assert_allclose(p.steady_precursors(1.0), steady_state(p)[-1])


def test_steady_precursors_annihilate_the_precursor_equation():
    """``dc_i/dt = lambda_i (P - c_i)`` is steady exactly when ``c_i == P``."""
    p = AxialParams()
    d = precursor_derivatives(p.steady_precursors(1.0), 1.0, p)
    np.testing.assert_allclose(d, np.zeros(6), atol=1e-15)


# --- mesh ------------------------------------------------------------------
def test_mesh_shapes_and_bounds():
    p = AxialParams(n_axial=8)
    edges, nodes = p.zeta_edges(), p.zeta_nodes()
    assert edges.shape == (9,)
    assert nodes.shape == (8,)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert np.all(np.diff(nodes) > 0)
    assert p.dz == pytest.approx(p.H / 8)


# --- axial power shape -----------------------------------------------------
def test_power_shape_has_unit_integral():
    """`P(t) * f(zeta)` must deposit exactly `P(t)`; a shape norm bug is silent otherwise.

    The normalisation is analytic, so the residual here is midpoint-rule
    quadrature error (~1e-9 at n=20001), not model error.
    """
    p = AxialParams()
    assert _midpoint_integral(p.power_shape) == pytest.approx(1.0, rel=1e-7)


def test_power_shape_peaks_at_midplane_and_is_positive():
    p = AxialParams()
    zeta = np.linspace(0.0, 1.0, 501)
    f = p.power_shape(zeta)
    assert np.all(f > 0.0)
    assert zeta[int(np.argmax(f))] == pytest.approx(0.5, abs=1e-2)


@pytest.mark.parametrize("extrap", [0.0, 0.05, 0.2, 0.5])
def test_power_shape_unit_integral_across_extrapolation(extrap):
    p = AxialParams(power_extrap=extrap)
    assert _midpoint_integral(p.power_shape) == pytest.approx(1.0, rel=1e-7)


# --- void worth ------------------------------------------------------------
def test_void_worth_integrates_to_requested_net():
    p = AxialParams(void_worth_net=2.0e-3)
    assert _midpoint_integral(p.void_worth) == pytest.approx(2.0e-3, rel=1e-6)


def test_void_worth_changes_sign_at_zeta_sign():
    """Positive through the core, negative near the top -- the leakage reversal."""
    p = AxialParams(zeta_sign=0.8, delta_sign=0.02)
    assert p.void_worth(0.5) > 0.0
    assert p.void_worth(0.95) < 0.0
    assert p.void_worth(np.array([0.8])) == pytest.approx(0.0, abs=1e-12)


def test_void_worth_vanishes_at_both_ends():
    p = AxialParams()
    assert p.void_worth(0.0) == pytest.approx(0.0, abs=1e-12)
    assert p.void_worth(1.0) == pytest.approx(0.0, abs=1e-12)


def test_void_worth_scales_linearly_with_net():
    a = AxialParams(void_worth_net=1.0e-3)
    b = AxialParams(void_worth_net=2.0e-3)
    np.testing.assert_allclose(2.0 * a.void_worth(0.4), b.void_worth(0.4), rtol=1e-12)


# --- Doppler ---------------------------------------------------------------
def test_alpha_D_interpolates_flooded_to_voided():
    """Manual section 4.5.3: alpha_D moves linearly from ADOP to BDOP as the node voids."""
    p = AxialParams(alpha_D_flooded=-6.0e-3, alpha_D_voided=-4.0e-3)
    assert p.alpha_D(0.0) == pytest.approx(-6.0e-3)
    assert p.alpha_D(1.0) == pytest.approx(-4.0e-3)
    assert p.alpha_D(0.5) == pytest.approx(-5.0e-3)


def test_alpha_D_clips_out_of_range_void_fraction():
    p = AxialParams()
    assert p.alpha_D(-1.0) == pytest.approx(p.alpha_D(0.0))
    assert p.alpha_D(2.0) == pytest.approx(p.alpha_D(1.0))


def test_alpha_D_coupling_disabled_when_coefficients_equal():
    p = AxialParams(alpha_D_flooded=-5e-3, alpha_D_voided=-5e-3)
    np.testing.assert_allclose(p.alpha_D(np.linspace(0, 1, 5)), -5e-3)


def test_alpha_D_stays_negative_over_the_void_range():
    p = AxialParams()
    assert np.all(p.alpha_D(np.linspace(0.0, 1.0, 21)) < 0.0)


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"r_fo": 5e-3}, "r_fo < r_ci < r_co"),
        ({"H": 0.0}, "H must be > 0"),
        ({"A_c": -1.0}, "A_c must be > 0"),
        ({"t_end": 0.0}, "t_end must be > 0"),
        ({"beta_eff": 0.0}, "beta_eff must be > 0"),
        ({"dT_smooth": 0.0}, "dT_smooth must be > 0"),
        ({"n_axial": 1}, "n_axial must be >= 2"),
        ({"f_nc": 1.5}, r"f_nc must lie in \[0, 1\]"),
        ({"gamma_c": -0.1}, r"gamma_c must lie in \[0, 1\]"),
        ({"zeta_sign": 0.0}, r"zeta_sign must lie in \(0, 1\)"),
        ({"gamma_2": -1.0}, "gamma_2 must be >= 0"),
        ({"alpha_D_flooded": 1e-3}, "Doppler coefficients must be <= 0"),
        ({"zeta_sign": 0.5, "delta_sign": 100.0}, "nearly integrates to zero"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AxialParams(**kwargs)


def test_structure_node_can_be_disabled():
    """gamma_2 = 0 drops the structure branch of manual Eq. 12.5-5 (deviation D-GEOM-2)."""
    p = AxialParams(gamma_2=0.0)
    assert p.gamma_2 == 0.0
