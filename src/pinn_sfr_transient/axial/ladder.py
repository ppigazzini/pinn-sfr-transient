"""Score a corpus of checkpoints against one reference solve, and emit the data file.

The point of saving models is that a table becomes a query rather than a training run.
This is that query: read a directory of checkpoints, group them by what they actually
are, score every one against a single shared reference, and write the result as JSON.
The published tables are then *rendered* from that file rather than transcribed into
prose, which is what :mod:`pinn_sfr_transient.axial.tables` does.

    uv run python tools/axial_study.py ladder --out __DEV/studies/ladder.json

One reference solve serves the whole corpus, which is the entire economy of the thing:
the solve costs tens of seconds and each checkpoint costs a fraction of that to score.

**Rows are keyed by the header, not the filename.** A mis-named checkpoint groups by
what it is. That also means an arm is identified by its *configuration*, and the
grouping key includes the optimiser family — the companion implementation grouped on
``(points, iters)`` alone, which is correct only as long as one optimiser is in the
corpus. Run that over a mixed corpus and a quasi-Newton arm and an AdEMAMix arm sharing
a budget average into one row that describes neither.

Every error is reported both raw and divided by the reference's own uncertainty at the
scoring mesh, from :mod:`pinn_sfr_transient.axial.verification`. A ratio below four is
not a resolvable difference and a ratio below one is a measurement of the reference —
see ``docs/axial_physics.md`` §6.6, where the shipped configuration sits at 1.05 on the
film field and below one on onset.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pinn_sfr_transient.axial import checkpoint
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.reference import solve_reference

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.reference import AxialTrajectory

#: Axial nodes and time samples the corpus is scored on.
#:
#: 160 is what every published axial table uses, so the ladder is comparable with them
#: by default. It is **not** a mesh at which the temperature fields are resolvable:
#: §6.6 measures the ratio of model error to reference uncertainty at 1.05 on the film
#: field and below one on onset. Pass a finer mesh to `ladder` when the question is how
#: accurate the surrogate is, rather than how it compares with the existing tables.
RULER_N_AXIAL: int = 160
RULER_N_OUT: int = 241

#: Metrics carried by the ladder, and the quantity in `verification`'s uncertainty
#: table each is normalised by. One axis carries all seven once each error is expressed
#: in units of the ruler's own error.
METRICS: tuple[tuple[str, str], ...] = (
    ("T_f", "T_f"),
    ("T_cl", "T_cl"),
    ("T_s", "T_s"),
    ("T_c", "T_c"),
    ("onset", "onset"),
    ("Lvoid", "Lvoid"),
    ("margin", "margin"),
)

#: Quantities carried alongside the errors, in physical units, because a table quotes
#: the value and not only the distance from the reference.
VALUES: tuple[str, ...] = ("onset_t", "L_void_m", "margin_K")

#: Knobs that identify an arm. `optimizer` and `first_order` are in here because a
#: corpus with more than one optimiser family in it is the normal case, not the
#: exception, and averaging across families produces a row describing neither.
ARM_KEYS: tuple[str, ...] = (
    "optimizer",
    "first_order",
    "n_colloc",
    "fourier_features",
    "lbfgs_history",
)


def arm_key(cfg: dict, iters: int) -> tuple:
    """Identify the arm a checkpoint belongs to, from its stored configuration."""
    return (*(cfg.get(k) for k in ARM_KEYS), iters)


def iters_of(path: Path, cfg: dict) -> int:
    """Budget a checkpoint was written at.

    `checkpoint.saver` encodes the *cumulative* quasi-Newton count in the filename,
    because one run emits several rungs and the configuration records only the total it
    was asked for. Falls back to the configured budget for a file saved outside the
    ladder hook.
    """
    for part in Path(path).stem.split("_"):
        if part.startswith("i") and part[1:].isdigit():
            return int(part[1:])
    return int(cfg.get("lbfgs_iters", 0))


def errors(m: dict[str, float]) -> dict[str, float]:
    """Error against the reference for each ladder metric, in the ruler's own units.

    **``Lvoid`` is in metres here and was a fraction in the companion repository**, and
    the difference is deliberate rather than an oversight. Every error in this dict is
    divided by the matching entry of ``verification``'s uncertainty table, and that table
    reports the voided-length uncertainty in metres (0.000186 m at 2560 nodes). A
    fractional error over a metric uncertainty is not a ratio of anything.

    The companion's own constant is named ``Lvoid_frac`` for exactly this reason, and its
    published rows are this number divided by the reference's peak voided length. The
    control arm confirms it: at ``points = 5000, iters = 10000`` scored on 2560 nodes,
    six of the seven metrics reproduce that repository's committed ladder to six
    significant figures, and this one reproduces it after dividing by 0.378454 m.
    """
    return {
        "T_f": m["T_f"],
        "T_cl": m["T_cl"],
        "T_s": m["T_s"],
        "T_c": m["T_c"],
        "onset": abs(m["onset_t_err_tan_s"]),
        "Lvoid": abs(m["L_void_max"] - m["L_void_max_ref"]),
        "margin": abs(m["margin_K"] - m["margin_K_ref"]),
    }


def values(m: dict[str, float]) -> dict[str, float]:
    """Return the front quantities in physical units, as the tables quote them."""
    return {
        "onset_t": m["onset_t_tan"],
        "L_void_m": m["L_void_max"],
        "margin_K": m["margin_K"],
    }


def _spread(vals: list[float]) -> dict[str, float]:
    """Mean and half-range over seeds, plus the count, which the table must show.

    Half-range rather than a standard deviation: at three seeds a standard deviation is
    a worse estimator than the range it is computed from, and the range is what a reader
    can check against the per-seed rows.
    """
    finite = [v for v in vals if np.isfinite(v)]
    if not finite:
        # `n` is the count of seeds that produced a usable number, in both branches.
        # Returning the total here instead made a rung where the front never formed on
        # any seed report the same `n` as a fully converged one.
        return {"mean": float("nan"), "half": float("nan"), "n": 0}
    return {
        "mean": (min(finite) + max(finite)) / 2.0,
        "half": (max(finite) - min(finite)) / 2.0,
        "n": len(finite),
    }


def _ruler(n_axial: int) -> dict[str, float]:
    """Read the reference uncertainty at the scoring mesh, from the verification study.

    Read from `__DEV/studies/verify.json` when it is there and left absent when it is
    not, rather than hard-coded. A hand-maintained table of five constants goes stale
    silently the first time the mesh changes; the companion repository carries exactly
    such a table and a comment asking the next person to remember to update it.
    """
    src = Path("__DEV/studies/verify.json")
    if not src.exists():
        return {}
    data = json.loads(src.read_text())
    row = data.get("uncertainty", {}).get(str(n_axial), {})
    if not row:
        return {}
    return {
        "T_f": row.get("T_f", float("nan")),
        "T_cl": row.get("T_cl", float("nan")),
        "T_s": row.get("T_s", float("nan")),
        "T_c": row.get("T_c", float("nan")),
        "onset": row.get("onset", float("nan")),
        "Lvoid": row.get("Lvoid", float("nan")),
        "margin": row.get("margin", float("nan")),
    }


#: Companion field name -> this repository's name, for the imported corpus.
#:
#: The two repositories call the same two knobs different things, and the ladder groups
#: on them. Translating here rather than in `legacy` keeps that module a verbatim copy
#: of what wrote the files.
_LEGACY_RENAMES: dict[str, str] = {"points": "n_colloc", "iters": "lbfgs_iters"}


def _normalise(cfg: dict) -> dict:
    """Give a legacy header this repository's field names, so one arm key covers both."""
    out = dict(cfg)
    for old, new in _LEGACY_RENAMES.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
    # The corpus predates the optimiser knob; every file without one is quasi-Newton,
    # which is what those runs were. Guessing is wrong here, so it is read from the
    # header where present and defaulted only where the field did not yet exist.
    out.setdefault("optimizer", "lbfgs")
    return out


def _reader(path: Path) -> tuple[dict | None, Any]:
    """Pick the reader for a checkpoint: this repository's format, or the imported one.

    Legacy files carry no ``format`` key — they were written before this repository had
    a checkpoint at all — so its absence is the discriminator. Dispatching on it means
    one corpus directory can hold both, which is what makes the imported 334 comparable
    with anything trained from here on.
    """
    try:
        return checkpoint.header(path), checkpoint.score
    except (ValueError, KeyError):
        pass
    try:
        from pinn_sfr_transient.axial import legacy  # noqa: PLC0415 - optional extra

        return legacy.header(path), legacy.score
    except (ValueError, KeyError, UnicodeDecodeError, ImportError):
        return None, None


def build(
    paths: list[Path],
    out: Path | None = None,
    p: AxialParams | None = None,
    n_axial: int = RULER_N_AXIAL,
) -> dict[str, Any]:
    """Score every checkpoint, group by arm, and return the ladder.

    Skips a checkpoint whose backend extra is not installed rather than failing the
    whole corpus, and says so — the JAX lane should be able to score the JAX half of a
    mixed corpus without torch present.
    """
    p = p or AxialParams()
    scoring_p = replace(p, n_axial=n_axial)
    print(f"reference: {n_axial} axial nodes, {RULER_N_OUT} time samples", flush=True)
    traj: AxialTrajectory = solve_reference(scoring_p, n_out=RULER_N_OUT)

    rows: dict[tuple, list[dict[str, float]]] = {}
    ref: dict[str, float] = {}
    skipped: list[str] = []
    for path in sorted(paths):
        head, scorer = _reader(path)
        if head is None:
            skipped.append(f"{Path(path).name} (unreadable header)")
            continue
        try:
            m = scorer(path, traj, scoring_p)
        except (ImportError, ModuleNotFoundError):
            skipped.append(f"{Path(path).name} ({head.get('backend', 'jax')} not installed)")
            continue
        key = arm_key(_normalise(head["config"]), iters_of(path, head["config"]))
        rows.setdefault(key, []).append(errors(m) | values(m))
        ref = {
            "onset_t": m["onset_t_tan"] - m["onset_t_err_tan_s"],
            "L_void_m": m["L_void_max_ref"],
            "margin_K": m["margin_K_ref"],
        }
        print(f"  scored {Path(path).name}", flush=True)

    arms = []
    for key, seeds in sorted(rows.items(), key=lambda kv: [str(x) for x in kv[0]]):
        arm: dict[str, Any] = dict(zip((*ARM_KEYS, "iters"), key, strict=True))
        arm["seeds"] = len(seeds)
        for name in [k for k, _ in METRICS] + list(VALUES):
            arm[name] = _spread([s[name] for s in seeds])
        arms.append(arm)

    data = {
        "n_axial": n_axial,
        "n_out": RULER_N_OUT,
        "ruler": _ruler(n_axial),
        "reference": ref,
        "arms": arms,
        "skipped": skipped,
    }
    if skipped:
        print(f"\nskipped {len(skipped)}: " + "; ".join(skipped[:5]), flush=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=1) + "\n")
        print(f"\nwrote {out}: {len(arms)} arms over {sum(a['seeds'] for a in arms)} checkpoints")
    return data
