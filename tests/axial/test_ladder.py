"""Grouping, spreading and rendering — the parts that decide what a table says.

No training and no reference solve: these exercise the arithmetic and the grouping on
synthetic rows, which is where the defects that reach a published table actually live.
The end-to-end path (train, save, score, render) is covered by the smoke run in
``tools/axial_study.py ladder``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pinn_sfr_transient.axial import ladder, tables


def _arm(iters, optimizer="lbfgs", **over):
    """One arm as `build` emits it, with every metric present."""
    arm = {
        "optimizer": optimizer,
        "first_order": "adam",
        "n_colloc": 5000,
        "fourier_features": 64,
        "lbfgs_history": 50,
        "iters": iters,
        "seeds": 3,
    }
    for k, _ in ladder.METRICS:
        arm[k] = {"mean": 1e-3, "half": 1e-4, "n": 3}
    for k in ladder.VALUES:
        arm[k] = {"mean": 1.0, "half": 0.1, "n": 3}
    arm.update(over)
    return arm


def _data(*arms, ruler=None):
    return {
        "n_axial": 160,
        "n_out": 241,
        "ruler": ruler or {},
        "reference": {},
        "arms": list(arms),
        "skipped": [],
    }


# --- grouping: the defect the audit found in the companion implementation --
def test_the_arm_key_separates_optimiser_families():
    """Grouping on (points, iters) alone averages two optimisers into one row.

    The companion implementation keyed on the budget and the collocation count only,
    which is correct exactly as long as one optimiser family is in the corpus. Over a
    mixed corpus a quasi-Newton arm and an AdEMAMix arm sharing a budget merge into a
    row describing neither.
    """
    qn = {"optimizer": "lbfgs", "first_order": "adam", "n_colloc": 5000}
    fo = {"optimizer": "adam", "first_order": "ademamix", "n_colloc": 5000}
    assert ladder.arm_key(qn, 50000) != ladder.arm_key(fo, 50000)


def test_the_arm_key_separates_every_declared_knob():
    """A knob in ARM_KEYS that does not change the key is not actually separating arms."""
    base = dict.fromkeys(ladder.ARM_KEYS, 1)
    for key in ladder.ARM_KEYS:
        other = base | {key: 2}
        assert ladder.arm_key(base, 10) != ladder.arm_key(other, 10), key


def test_the_arm_key_separates_budgets():
    cfg = {"optimizer": "lbfgs"}
    assert ladder.arm_key(cfg, 20000) != ladder.arm_key(cfg, 50000)


def test_iters_come_from_the_filename_not_the_configured_total(tmp_path):
    """One run emits several rungs; the config records only the total it was asked for."""
    p = tmp_path / "jax_p5000_i20000_f64_s1_20260101000000-abcd1234.eqx"
    assert ladder.iters_of(p, {"lbfgs_iters": 50000}) == 20000


def test_iters_fall_back_to_the_config_for_a_file_saved_outside_the_hook(tmp_path):
    assert ladder.iters_of(tmp_path / "hand-named.eqx", {"lbfgs_iters": 7000}) == 7000


# --- the spread a table quotes --------------------------------------------
def test_spread_is_mean_and_half_range():
    s = ladder._spread([1.0, 2.0, 3.0])
    assert s["mean"] == pytest.approx(2.0)
    assert s["half"] == pytest.approx(1.0)
    assert s["n"] == 3


def test_spread_reports_the_count_it_actually_used():
    """A rung with one usable seed must not look like a rung with three."""
    s = ladder._spread([1.0, float("nan"), 3.0])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(2.0)


def test_spread_of_nothing_finite_is_nan_not_zero():
    """A run where the front never formed must not read as a perfect score."""
    s = ladder._spread([float("nan"), float("nan")])
    assert np.isnan(s["mean"])
    assert s["n"] == 0


# --- error extraction -------------------------------------------------------
def test_errors_are_distances_and_values_are_quantities():
    """Conflating the two cost a KeyError that was really a units mix-up."""
    m = {
        "T_f": 1e-3,
        "T_cl": 2e-3,
        "T_s": 3e-3,
        "T_c": 4e-3,
        "onset_t_err_tan_s": -0.02,
        "onset_t_tan": 10.99,
        "L_void_max": 0.38,
        "L_void_max_ref": 0.379,
        "margin_K": 68.0,
        "margin_K_ref": 69.0,
    }
    e, v = ladder.errors(m), ladder.values(m)
    assert e["onset"] == pytest.approx(0.02), "an error is a magnitude, never signed"
    assert e["Lvoid"] == pytest.approx(0.001)
    assert e["margin"] == pytest.approx(1.0)
    assert v["onset_t"] == pytest.approx(10.99), "a value is the quantity, not the error"
    assert v["L_void_m"] == pytest.approx(0.38)


# --- rendering and the drift check ----------------------------------------
def test_rows_render_one_line_per_arm():
    rendered = tables.rows(_data(_arm(10000), _arm(20000)))
    assert len(rendered) == 2
    assert all(line.startswith("| ") and line.endswith(" |") for line in rendered)


def test_a_nan_metric_renders_as_a_dash_not_as_a_number():
    """A rung where the front never formed must not print a plausible-looking score."""
    arm = _arm(10000, onset={"mean": float("nan"), "half": float("nan"), "n": 0})
    assert "—" in tables.rows(_data(arm))[0]


def test_the_selector_filters_arms():
    data = _data(_arm(10000), _arm(10000, optimizer="adam"))
    assert len(tables.rows(data, optimizer="lbfgs")) == 1


def test_check_passes_when_the_document_carries_the_row(tmp_path):
    """The point of the checker: docs and the data file cannot drift apart."""
    data = _data(_arm(10000))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text("intro\n\n" + tables.rows(data)[0] + "\n", encoding="utf-8")
    assert tables.check(data, tmp_path) == []


def test_check_names_the_row_that_drifted(tmp_path):
    """Change a measurement without re-rendering and the check must say which row."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "x.md").write_text(tables.rows(_data(_arm(10000)))[0], encoding="utf-8")
    moved = _arm(10000, T_f={"mean": 9e-3, "half": 1e-4, "n": 3})
    missing = tables.check(_data(moved), tmp_path)
    assert len(missing) == 1
    assert "9.00" in missing[0]


def test_check_reports_everything_when_no_document_mentions_the_table(tmp_path):
    (tmp_path / "docs").mkdir()
    assert len(tables.check(_data(_arm(1), _arm(2)), tmp_path)) == 2


# --- the ratio table, which is what says whether a number means anything ---
def test_the_ratio_table_divides_by_the_reference_uncertainty():
    """Four is the threshold; the table exists so a reader can see which side of it."""
    ruler = dict.fromkeys([k for k, _ in ladder.METRICS], 1e-4)
    out = tables.ratio_table(_data(_arm(10000), ruler=ruler))
    assert "10.00" in out, "1e-3 against a 1e-4 ruler is a ratio of ten"


def test_the_ratio_table_says_so_when_there_is_no_ruler():
    """Silence would read as 'no problem'; absence of a ruler is not a passing grade."""
    assert "verify" in tables.ratio_table(_data(_arm(10000)))


def test_the_ratio_table_refuses_to_divide_by_a_missing_uncertainty():
    """The finest mesh has no field estimate, so those cells must stay empty."""
    out = tables.ratio_table(_data(_arm(1), ruler={"T_f": float("nan"), "onset": 1e-3}))
    assert "—" in out


def test_load_round_trips_a_written_ladder(tmp_path):
    path = tmp_path / "ladder.json"
    data = _data(_arm(10000))
    path.write_text(json.dumps(data), encoding="utf-8")
    assert tables.load(path)["arms"][0]["iters"] == 10000
