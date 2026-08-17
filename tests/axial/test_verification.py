"""Grid-convergence verification of the reference — the arithmetic, not the physics.

:mod:`pinn_sfr_transient.axial.verification` is the only committed source of a
reference uncertainty in this repository, so every number it can emit is quoted
downstream. These tests pin the two things that make such a number trustworthy: that
the extrapolation refuses to produce one when the sequence does not support it, and
that the observed order is recovered when it does.

Deliberately cheap. The convergence *of the reference* is already covered by
``test_axial_reference.py``; solving four meshes up to 1280 nodes takes minutes and
belongs in ``tools/axial_study.py verify``, not in the suite.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from pinn_sfr_transient.axial import verification
from pinn_sfr_transient.axial.verification import _field_orders, _gap_to_limit, richardson


# --- the extrapolation refuses what it cannot support ----------------------
def test_richardson_recovers_a_first_order_sequence():
    """``f_h = L + C h`` on doubling meshes must give ``p = 1`` and ``L`` exactly."""
    limit, c = 10.9784, 0.4
    seq = [limit + c * h for h in (4.0, 2.0, 1.0)]
    order, extrapolated = richardson(*seq)
    assert order == pytest.approx(1.0, abs=1e-12)
    assert extrapolated == pytest.approx(limit, abs=1e-12)


def test_richardson_recovers_a_second_order_sequence():
    """The order is *observed*, not assumed, so a second-order sequence reads as two."""
    limit, c = 0.3792, 0.05
    seq = [limit + c * h**2 for h in (4.0, 2.0, 1.0)]
    order, extrapolated = richardson(*seq)
    assert order == pytest.approx(2.0, abs=1e-12)
    assert extrapolated == pytest.approx(limit, abs=1e-12)


def test_richardson_rejects_a_non_monotone_sequence():
    """Outside the asymptotic range, extrapolating would invent a number."""
    order, limit = richardson(1.0, 2.0, 1.5)
    assert np.isnan(order)
    assert np.isnan(limit)


def test_richardson_rejects_a_monotone_but_diverging_sequence():
    """The case the companion implementation returned a finite number for.

    Differences that *grow* under refinement are a discretisation defect, not a
    convergence rate. Guarding only monotonicity, ``abs(d1 / d2) < 1`` gives a negative
    observed order, a negative ``2^p - 1``, and a finite extrapolated limit that means
    nothing — and a number is worse than a ``nan``, because something downstream quotes
    it. Here ``d1 = 1`` and ``d2 = 2``, which the old guard passed.
    """
    order, limit = richardson(0.0, 1.0, 3.0)
    assert np.isnan(order)
    assert np.isnan(limit)


def test_richardson_rejects_a_stalled_sequence():
    """Two equal values give no information about the order; zero division is not it."""
    order, limit = richardson(1.0, 2.0, 2.0)
    assert np.isnan(order)
    assert np.isnan(limit)


# --- the field estimate uses the order it measured -------------------------
def test_gap_to_limit_is_the_familiar_factor_of_two_at_first_order():
    """At ``p = 1`` the generalised estimate must reproduce what it replaced."""
    assert _gap_to_limit(1e-3, 1.0) == pytest.approx(2e-3)


def test_gap_to_limit_tracks_the_observed_order():
    """At second order the same gap implies a *smaller* error, not the same one.

    Hard-coding the factor of two asserts first order on a field whose order was never
    measured; at ``p = 2`` it overstates the uncertainty by 1.5x, which moves an error
    ratio across the acceptance threshold in the wrong direction.
    """
    assert _gap_to_limit(1e-3, 2.0) == pytest.approx(1e-3 / 0.75)
    assert _gap_to_limit(1e-3, 2.0) < _gap_to_limit(1e-3, 1.0)


def test_gap_to_limit_refuses_a_non_positive_order():
    """A field that is not converging has no error estimate, only a gap."""
    assert np.isnan(_gap_to_limit(1e-3, 0.0))
    assert np.isnan(_gap_to_limit(1e-3, -1.0))
    assert np.isnan(_gap_to_limit(1e-3, float("nan")))


def test_field_orders_read_the_contraction_of_successive_gaps():
    """Gaps contract by ``2^p`` per doubling, exactly as the errors do."""
    gaps = [dict.fromkeys(verification.FIELDS, g) for g in (4e-3, 2e-3, 1e-3)]
    orders = _field_orders(gaps)
    assert set(orders) == set(verification.FIELDS)
    for f in verification.FIELDS:
        assert orders[f] == pytest.approx(1.0, abs=1e-12)


def test_field_orders_refuse_a_single_gap():
    """One gap cannot give an order, and must not be made to look like it does."""
    orders = _field_orders([dict.fromkeys(verification.FIELDS, 1e-3)])
    assert all(np.isnan(v) for v in orders.values())


def test_field_orders_refuse_gaps_that_grow():
    """A field getting worse under refinement gets no order and hence no uncertainty."""
    gaps = [dict.fromkeys(verification.FIELDS, g) for g in (1e-3, 2e-3, 4e-3)]
    assert all(np.isnan(v) for v in _field_orders(gaps).values())


# --- the report's contract -------------------------------------------------
def test_report_requires_three_meshes_to_observe_an_order():
    """Two meshes can only assume the order, which is what this module exists to avoid."""
    with pytest.raises(ValueError, match="at least three meshes"):
        verification.report(meshes=(160, 320))


def test_meshes_double():
    """:func:`richardson` assumes a refinement ratio of two; the default must honour it."""
    assert all(b == 2 * a for a, b in pairwise(verification.MESHES))
