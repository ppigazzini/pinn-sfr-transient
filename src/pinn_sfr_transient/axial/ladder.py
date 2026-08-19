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
what it is.

**And the key is derived, not declared** — see :func:`arm_fields`. Two attempts got this
wrong before the third: the companion grouped on ``(points, iters)``, which merges
optimiser families, and the first version here declared five knobs, which still averaged
136 of 334 checkpoints across learning rates spanning three orders of magnitude. Any
declared list is a list of the knobs someone remembered.

Every error is reported both raw and divided by the reference's own uncertainty at the
scoring mesh, from :mod:`pinn_sfr_transient.axial.verification`. A ratio below four is
not a resolvable difference and a ratio below one is a measurement of the reference —
see ``docs/axial_physics.md`` §6.6, where the shipped configuration sits at 1.05 on the
film field and below one on onset.
"""

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
#: 2560, the mesh at which these fields are actually resolvable. The ratio of model
#: error to reference uncertainty is 4.2 here, 3.1 at 640, 1.9 at 320 and 1.06 at 160 --
#: and a claim needs four. This was 160, which `docs/axial_physics.md` §6.6 had already
#: measured as the reference's error being the size of the model's.
#:
#: The corpus was scored at 160, so a ladder built now is NOT comparable with rows built
#: before this changed. That is the right way round: the old rows were comparable with
#: each other and with nothing real.
RULER_N_AXIAL: int = 2560
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

#: Config fields that do **not** identify an arm.
#:
#: `seed` is the axis a row averages over. The budget fields are the ladder's x-axis and
#: are carried separately. `log_every` changes nothing about the model.
#:
#: Everything else identifies an arm, and that is the point — see :func:`arm_fields`.
NOT_ARM: frozenset[str] = frozenset({"seed", "iters", "lbfgs_iters", "adam_iters", "log_every"})


def _hashable(v: object) -> object:
    """Make a config value usable in a grouping key. JSON gives lists back for tuples."""
    return tuple(v) if isinstance(v, list) else v


def arm_fields(configs: list[dict]) -> tuple[str, ...]:
    """Config fields that actually **vary** across this corpus, and so separate arms.

    Derived from the corpus rather than declared, and that is a correctness property
    rather than a convenience. A hard-coded list is a list of the knobs someone thought
    of, and it is wrong the moment a new one is swept — silently, by averaging two arms
    into a row describing neither.

    This is not hypothetical and it is not only the companion's mistake. That repository
    keyed on ``(points, iters)``, which merges optimiser families. The first fix here
    declared five keys instead, and over the imported corpus that still averaged **136
    of 334 checkpoints** across different learning rates — putting ``lr = 0.1``, which
    diverged, in the same row as ``lr = 1e-4``. The list was wrong in exactly the way a
    declared list is always eventually wrong.

    Deriving it cannot make that mistake: a knob that varies separates arms whether or
    not anyone remembered it.
    """
    seen: dict[str, set] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if k not in NOT_ARM:
                seen.setdefault(k, set()).add(_hashable(v))
    return tuple(sorted(k for k, vals in seen.items() if len(vals) > 1))


def arm_key(cfg: dict, iters: int, fields: tuple[str, ...]) -> tuple:
    """Identify the arm a checkpoint belongs to, from its configuration and budget."""
    return (*(_hashable(cfg.get(k)) for k in fields), iters)


def iters_of(path: Path, cfg: dict) -> int:
    """Budget a checkpoint was written at.

    `checkpoint.saver` encodes the *cumulative* count in the filename, because one run
    emits several rungs while the configuration records only the total it was asked for.
    Falls back to the configuration for a file saved outside that hook.

    **Both budget spellings are accepted.** The imported corpus calls it ``iters``; this
    repository calls it ``lbfgs_iters``. Reading only the second put 41 checkpoints —
    every one whose filename carries no ``iNNNN`` token, which is all 55 in the
    per-family subdirectories — into a bogus ``iters = 0`` arm.
    """
    for part in Path(path).stem.split("_"):
        if part.startswith("i") and part[1:].isdigit():
            return int(part[1:])
    for key in ("lbfgs_iters", "iters", "adam_iters"):
        if cfg.get(key):
            return int(cfg[key])
    return 0


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


#: Where `verify` writes the reference uncertainty the ratio columns divide by.
VERIFY_JSON = Path("__DEV/studies/verify.json")


def _ruler(n_axial: int, src: Path = VERIFY_JSON) -> dict[str, float]:
    """Read the reference uncertainty at the scoring mesh, from the verification study.

    Read from a file rather than hard-coded. A hand-maintained table of five constants
    goes stale silently the first time the mesh changes; the companion repository carries
    exactly such a table with a comment asking the next person to remember.

    Absence is **reported**, not swallowed. This is a relative path, so running from
    another directory finds nothing, and a silent empty ruler means every ratio column
    quietly disappears from the tables — which reads as "no problem" rather than "not
    measured".
    """
    if not Path(src).exists():
        print(f"  no reference uncertainty: {src} not found; ratio columns will be empty")
        return {}
    data = json.loads(Path(src).read_text())
    row = data.get("uncertainty", {}).get(str(n_axial), {})
    if not row:
        have = sorted(data.get("uncertainty", {}))
        print(f"  no reference uncertainty at n_axial={n_axial}; verify.json has {have}")
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
        head = checkpoint.header(path)
    except ValueError, KeyError:
        pass
    else:
        return (head, checkpoint.score) if "config" in head else (None, None)
    try:
        from pinn_sfr_transient.axial import legacy  # noqa: PLC0415 - optional extra

        head = legacy.header(path)
    except ValueError, KeyError, UnicodeDecodeError, ImportError, json.JSONDecodeError:
        return None, None
    # A header without a `config` is not a checkpoint this can group. Returning it
    # anyway raised `KeyError` out of `build` and took the whole corpus run down with
    # one bad file -- the opposite of the per-file skip this function exists to provide.
    return (head, legacy.score) if "config" in head else (None, None)


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

    # Headers first, weights second. A header is one line and needs no backend, so the
    # whole corpus can be indexed before a single reference solve is paid for -- and the
    # arm fields have to be known before anything is grouped.
    index: list[tuple[Path, dict, Any]] = []
    skipped: list[str] = []
    for path in sorted(paths):
        head, scorer = _reader(path)
        if head is None:
            skipped.append(f"{Path(path).name} (unreadable header)")
            continue
        index.append((Path(path), _normalise(head["config"]), scorer))

    fields = arm_fields([cfg for _, cfg, _ in index])
    print(f"{len(index)} checkpoints; arms separated by: {', '.join(fields) or '(budget only)'}")
    print(f"reference: {n_axial} axial nodes, {RULER_N_OUT} time samples", flush=True)
    traj: AxialTrajectory = solve_reference(scoring_p, n_out=RULER_N_OUT)

    rows: dict[tuple, list[dict[str, float]]] = {}
    labels: dict[tuple, dict] = {}
    ref: dict[str, float] = {}
    for path, cfg, scorer in index:
        try:
            m = scorer(path, traj, scoring_p)
        except (ImportError, ModuleNotFoundError) as exc:
            skipped.append(f"{path.name} ({exc})")
            continue
        key = arm_key(cfg, iters_of(path, cfg), fields)
        rows.setdefault(key, []).append(errors(m) | values(m))
        labels[key] = {k: cfg.get(k) for k in fields} | {"iters": iters_of(path, cfg)}
        ref = {
            "onset_t": m["onset_t_tan"] - m["onset_t_err_tan_s"],
            "L_void_m": m["L_void_max_ref"],
            "margin_K": m["margin_K_ref"],
        }
        print(f"  scored {path.name}", flush=True)

    arms = []
    for key, seeds in sorted(rows.items(), key=lambda kv: [str(x) for x in kv[0]]):
        arm: dict[str, Any] = dict(labels[key])
        arm["seeds"] = len(seeds)
        for name in [k for k, _ in METRICS] + list(VALUES):
            arm[name] = _spread([s[name] for s in seeds])
        arms.append(arm)

    data = {
        "n_axial": n_axial,
        "n_out": RULER_N_OUT,
        "arm_fields": list(fields),
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
