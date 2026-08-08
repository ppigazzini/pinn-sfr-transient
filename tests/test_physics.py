"""Unit tests for the closed-form physics models (numpy/scipy only)."""

import numpy as np

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.physics import flow_fraction, reactivity, void_fraction


def test_void_fraction_monotone_and_bounded() -> None:
    p = SFRParams()
    Tc = np.linspace(p.Tin, p.T_onset + 200.0, 256)
    phi = void_fraction(Tc, p)
    assert np.all((phi >= 0.0) & (phi < 1.0))
    assert np.all(np.diff(phi) >= 0.0)  # non-decreasing in coolant temperature


def test_flow_fraction_coastdown_limits() -> None:
    p = SFRParams()
    assert flow_fraction(0.0, p) == 1.0  # full forced flow at the onset of ULOF
    assert abs(flow_fraction(1.0e6, p) - p.f_nc) < 1e-12  # -> natural-circulation floor


def test_reactivity_vanishes_at_nominal_state() -> None:
    p = SFRParams()
    # rho_ext is calibrated in __post_init__ so the nominal state is a fixed point.
    assert abs(reactivity(p.Tf0, p.Tc0, p)) < 1e-12


def test_positive_void_feedback_raises_reactivity() -> None:
    p = SFRParams()
    hot = reactivity(p.Tf0, p.T_onset + 4.0 * p.dT_void, p)
    nominal = reactivity(p.Tf0, p.Tc0, p)
    # Doppler/density are negative, but the positive void term dominates once the
    # coolant boils well past onset -- the defining SFR ULOF hazard.
    assert hot > nominal
