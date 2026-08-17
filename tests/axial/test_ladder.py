"""Grouping, spreading and rendering — the parts that decide what a table says.

No training and no reference solve: these exercise the arithmetic and the grouping on
synthetic rows, which is where the defects that reach a published table actually live.
The end-to-end path (train, save, score, render) is covered by the smoke run in
``tools/axial_study.py ladder``.
"""

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


def _data(*arms, ruler=None, arm_fields=("optimizer",)):
    return {
        "n_axial": 160,
        "n_out": 241,
        "arm_fields": list(arm_fields),
        "ruler": ruler or {},
        "reference": {},
        "arms": list(arms),
        "skipped": [],
    }


# --- grouping: the defect the audit found in the companion implementation --
def test_arm_fields_are_derived_from_what_actually_varies():
    """A declared list of knobs is wrong the moment a new one is swept.

    Both previous versions got this wrong. The companion keyed on `(points, iters)`,
    merging optimiser families. The first fix here declared five keys, and over the
    imported corpus that still averaged 136 of 334 checkpoints across different
    learning rates -- putting `lr = 0.1`, which diverged, beside `lr = 1e-4`.
    """
    configs = [
        {"optimizer": "adamw", "lr": 1e-4, "width": 64, "seed": 0},
        {"optimizer": "adamw", "lr": 1e-1, "width": 64, "seed": 1},
    ]
    assert ladder.arm_fields(configs) == ("lr",), "only the varying knob separates arms"


def test_arm_fields_pick_up_a_knob_nobody_declared():
    """The property a hard-coded list cannot have."""
    configs = [{"a_knob_invented_tomorrow": 1}, {"a_knob_invented_tomorrow": 2}]
    assert ladder.arm_fields(configs) == ("a_knob_invented_tomorrow",)


def test_arm_fields_ignore_the_seed_and_the_budget():
    """Seeds are what a row averages over; the budget is the ladder's x-axis."""
    configs = [
        {"seed": 0, "lbfgs_iters": 10, "iters": 10, "adam_iters": 1, "log_every": 1, "w": 1},
        {"seed": 1, "lbfgs_iters": 20, "iters": 20, "adam_iters": 2, "log_every": 9, "w": 1},
    ]
    assert ladder.arm_fields(configs) == ()


def test_arm_fields_handle_a_json_list_where_a_tuple_was_stored():
    """Config tuples come back from JSON as lists, which are unhashable in a key."""
    configs = [{"bands": [1.0, 2.0]}, {"bands": [3.0]}]
    assert ladder.arm_fields(configs) == ("bands",)
    assert ladder.arm_key({"bands": [1.0, 2.0]}, 10, ("bands",))


def test_the_arm_key_separates_optimiser_families():
    """A quasi-Newton arm and an AdEMAMix arm sharing a budget are not one row."""
    fields = ("optimizer",)
    qn, fo = {"optimizer": "lbfgs"}, {"optimizer": "adamw"}
    assert ladder.arm_key(qn, 50000, fields) != ladder.arm_key(fo, 50000, fields)


def test_the_arm_key_separates_budgets():
    assert ladder.arm_key({}, 20000, ()) != ladder.arm_key({}, 50000, ())


def test_iters_come_from_the_filename_not_the_configured_total(tmp_path):
    """One run emits several rungs; the config records only the total it was asked for."""
    p = tmp_path / "jax_p5000_i20000_f64_s1_20260101000000-abcd1234.eqx"
    assert ladder.iters_of(p, {"lbfgs_iters": 50000}) == 20000


def test_iters_fall_back_to_the_config_for_a_file_saved_outside_the_hook(tmp_path):
    assert ladder.iters_of(tmp_path / "hand-named.eqx", {"lbfgs_iters": 7000}) == 7000


def test_iters_accept_the_companion_spelling(tmp_path):
    """The imported corpus calls it `iters`; reading only `lbfgs_iters` returned zero.

    Every one of the 55 files in the per-family subdirectories is named `seedN.eqx`,
    so the filename carries no budget and the fallback is all there is. 41 checkpoints
    landed in a bogus `iters = 0` arm before this.
    """
    assert ladder.iters_of(tmp_path / "seed0.eqx", {"iters": 50000}) == 50000
    assert ladder.iters_of(tmp_path / "seed0.eqx", {"adam_iters": 300000}) == 300000


def test_iters_are_zero_only_when_nothing_records_them(tmp_path):
    assert ladder.iters_of(tmp_path / "seed0.eqx", {}) == 0


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
    # Metres, NOT a fraction. Every error here is divided by `verification`'s
    # uncertainty for the same quantity, and that one is in metres; the companion
    # repository published the fraction, which is why its constant is `Lvoid_frac`.
    # Reproducing its ladder row needed exactly this division and nothing else.
    assert e["Lvoid"] == pytest.approx(0.001), "absolute metres, matching the ruler"
    assert e["Lvoid"] != pytest.approx(0.001 / 0.379, rel=1e-3)
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


def test_rows_carry_the_columns_that_identify_the_arm():
    """`iters` alone names up to a dozen arms once several knobs vary."""
    data = _data(_arm(10000, optimizer="lbfgs"), _arm(10000, optimizer="adamw"))
    rendered = tables.rows(data)
    assert any("lbfgs" in r for r in rendered)
    assert any("adamw" in r for r in rendered)


def test_two_arms_differing_only_in_a_knob_do_not_render_identically():
    """Identical strings would let one document line satisfy several data rows.

    That silently weakens `check`: the drift it exists to catch would pass.
    """
    a = _arm(10000, optimizer="lbfgs")
    b = _arm(10000, optimizer="adamw")
    rendered = tables.rows(_data(a, b))
    assert len(set(rendered)) == 2


def test_a_pinned_selector_drops_its_own_column():
    """A table already filtered to one optimiser should not repeat it every row."""
    data = _data(_arm(10000), _arm(20000))
    assert "lbfgs" not in tables.table(data, optimizer="lbfgs")


def test_the_header_names_every_identifying_column():
    data = _data(_arm(1), arm_fields=("optimizer", "lr"))
    head = tables.table(data).splitlines()[0]
    assert "optimizer" in head
    assert "lr" in head


def _doc(tmp_path, body):
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "x.md").write_text(body, encoding="utf-8")
    return tmp_path


def _fenced(data, selector=""):
    sel = f" {selector}" if selector else ""
    body = tables.table(data, **tables._parse_selector(selector))
    return f"<!-- ladder:{sel} -->\n{body}\n<!-- /ladder -->\n"


def test_check_passes_when_the_fenced_table_matches(tmp_path):
    """The point of the checker: a document and the data file cannot drift apart."""
    data = _data(_arm(10000))
    problems, blocks = tables.check(data, _doc(tmp_path, "intro\n\n" + _fenced(data)))
    assert problems == []
    assert blocks == 1


def test_check_names_the_file_and_line_that_drifted(tmp_path):
    """Change a measurement without re-rendering and the check must say where."""
    stale = _data(_arm(10000))
    root = _doc(tmp_path, _fenced(stale))
    moved = _data(_arm(10000, T_f={"mean": 9e-3, "half": 1e-4, "n": 3}))
    problems, blocks = tables.check(moved, root)
    assert blocks == 1
    assert len(problems) == 1
    assert "x.md:1" in problems[0]


def test_check_is_vacuous_but_visible_when_no_document_quotes_the_ladder(tmp_path):
    """Zero problems from zero blocks must be distinguishable from zero from twelve.

    A document that quietly loses its fence would otherwise pass by having nothing left
    to check, which is the fail-open shape this repository keeps getting bitten by.
    """
    problems, blocks = tables.check(_data(_arm(1)), _doc(tmp_path, "no tables here"))
    assert problems == []
    assert blocks == 0


def test_check_catches_an_unclosed_fence(tmp_path):
    """An unterminated fence would otherwise swallow the rest of the document."""
    problems, blocks = tables.check(_data(_arm(1)), _doc(tmp_path, "<!-- ladder: -->\n| x |\n"))
    assert blocks == 0
    assert "never closed" in problems[0]


def test_check_honours_the_selector_on_the_fence(tmp_path):
    """One document can hold several slices, each verified against its own filter."""
    data = _data(_arm(10000, optimizer="lbfgs"), _arm(10000, optimizer="adamw"))
    problems, blocks = tables.check(data, _doc(tmp_path, _fenced(data, "optimizer=lbfgs")))
    assert problems == []
    assert blocks == 1


def test_a_selector_that_stops_matching_is_caught(tmp_path):
    """Rename an optimiser and the slice empties; the document must not stay green."""
    data = _data(_arm(10000, optimizer="lbfgs"))
    root = _doc(tmp_path, _fenced(data, "optimizer=lbfgs"))
    renamed = _data(_arm(10000, optimizer="lbfgs-ssbfgs"))
    problems, _ = tables.check(renamed, root)
    assert problems


def test_the_selector_parser_coerces_types():
    """Arms hold ints, floats and bools; a string selector would match nothing."""
    got = tables._parse_selector("n_colloc=5000 lr=0.0001 cosine=True lr_warmup=False o=adamw")
    assert got == {
        "n_colloc": 5000,
        "lr": 0.0001,
        "cosine": True,
        "lr_warmup": False,
        "o": "adamw",
    }


# --- the ratio table, which is what says whether a number means anything ---
def test_the_ratio_table_divides_by_the_reference_uncertainty():
    """Four is the threshold; the table exists so a reader can see which side of it."""
    ruler = dict.fromkeys([k for k, _ in ladder.METRICS], 1e-4)
    out = tables.ratio_table(_data(_arm(10000), ruler=ruler))
    assert "10.00" in out, "1e-3 against a 1e-4 ruler is a ratio of ten"


def test_a_single_seed_arm_gets_no_plus_minus():
    """`112.97 +/- 0.00` from one sample reads as perfect reproducibility.

    The seed spread on this model has reached 12.5x and four published conclusions have
    been overturned by the next seed. The `seeds` column says 1; a spread beside it
    contradicts that.
    """
    arm = _arm(10000, seeds=1)
    for k, _ in ladder.METRICS:
        arm[k] = {"mean": 1e-3, "half": 0.0, "n": 1}
    row = tables.rows(_data(arm))[0]
    assert "±" not in row
    assert "1.00" in row


def test_the_spread_is_suppressed_per_metric_not_per_row():
    """A metric can be single-seeded while its neighbours are not.

    `_spread` counts the seeds that produced a *finite* value, so a rung where the
    front formed on one seed of three reports `n = 1` for onset and `n = 3` for the
    temperatures. Suppressing per row would hide that; suppressing per metric shows it.
    """
    arm = _arm(10000, onset={"mean": 0.1, "half": 0.0, "n": 1})
    row = tables.rows(_data(arm))[0]
    assert "0.1000 |" in row, "the single-seeded metric loses its spread"
    assert "1.00 ± 0.10" in row, "the three-seeded ones keep theirs"


def test_a_multi_seed_arm_keeps_its_spread():
    arm = _arm(10000, T_f={"mean": 1e-3, "half": 1e-4, "n": 3})
    assert "1.00 ± 0.10" in tables.rows(_data(arm))[0]


def test_budgets_sort_numerically_not_as_text():
    """Folding `iters` into a string key put 1,000,000 between 100,000 and 200,000."""
    data = _data(_arm(1000000), _arm(100000), _arm(200000))
    order = [r.split("|")[2].strip() for r in tables.rows(data)]
    assert order == ["100,000", "200,000", "1,000,000"]


def test_the_ratio_table_identifies_its_arms_too():
    """A ratio row labelled only by budget names every arm sharing that budget."""
    ruler = dict.fromkeys([k for k, _ in ladder.METRICS], 1e-4)
    data = _data(_arm(10000, optimizer="lbfgs"), _arm(10000, optimizer="adamw"), ruler=ruler)
    out = tables.ratio_table(data)
    assert "lbfgs" in out
    assert "adamw" in out
    assert len(set(out.splitlines()[2:])) == 2


def test_ratios_below_the_threshold_are_marked():
    """A column of bare decimals makes the reader do the comparison 1300 times."""
    ruler = dict.fromkeys([k for k, _ in ladder.METRICS], 1e-3)
    resolvable = _arm(1, T_f={"mean": 8e-3, "half": 0.0, "n": 3})
    marginal = _arm(2, T_f={"mean": 2e-3, "half": 0.0, "n": 3})
    ruler_bound = _arm(3, T_f={"mean": 5e-4, "half": 0.0, "n": 3})
    out = tables.ratio_table(_data(resolvable, marginal, ruler_bound, ruler=ruler)).splitlines()
    assert "8.00 |" in out[2], "a ratio clearing four is unmarked"
    assert "2.00 (<4)" in out[3], "a ratio between one and four is marginal"
    assert "0.50 (<1)" in out[4], "a ratio below one is measuring the reference"


def test_the_marking_threshold_is_the_declared_constant():
    """A documented four and a hard-coded four drift apart; there must be one."""
    from pinn_sfr_transient.axial import verification

    assert tables.MIN_RATIO == verification.MIN_RATIO


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
