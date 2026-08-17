"""Save and reload a trained axial surrogate, in either backend.

Until this module existed, **nothing in this repository saved a trained model**. Every
one of the 169 runs in ``__DEV/studies/`` was scored in-process and then discarded, so a
new metric, a different scoring mesh or a figure nobody had drawn cost another training
run — hours — rather than a re-score, seconds. That is also why the mesh-sensitivity
comparison in ``docs/axial_physics.md`` §6.6 is marked unreproduced: re-scoring three
fixed models against a second reference needs the three models, and they are gone.

The file is one JSON header line followed by the serialised weights. The header holds
the training configuration, because weights alone are not a model — rebuilding the
skeleton needs the embedding width, the trunk shape and the horizon, and inferring those
from array shapes is guesswork that fails silently.

    from pinn_sfr_transient.axial import checkpoint
    path = checkpoint.save(checkpoint.default_path(cfg, "jax"), model, cfg, backend="jax")
    model, cfg, saved = checkpoint.load(path)

**Both backends, one format.** AGENTS.md requires a feature that lands in one axial
backend to land in the other, and a save path for JAX alone would fork them on the first
commit. The header is identical; only the payload differs (``eqx.tree_serialise_leaves``
against ``torch.save`` of the state dict), and :func:`header` reads the metadata of
either without importing either.

Neither ``torch`` nor ``jax`` is imported at module scope, so this imports cleanly in the
``minimal`` lane with no extra installed.

Ported from ``checkpoint.py`` in the companion repository ``pinn-ulof``, with the
backend dispatch, the parameter digest and the format version added here.
"""

import hashlib
import json
import os
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_origin
from uuid import uuid4

import numpy as np

from pinn_sfr_transient.axial.config import AxialParams

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.reference import AxialTrajectory

#: Where training writes by default. Gitignored: a checkpoint is a build product.
DEFAULT_DIR = Path("models")

#: Bumped only when the on-disk layout changes incompatibly. Read on load and refused
#: if unknown, because a silently misread checkpoint is worse than one that will not
#: open.
FORMAT_VERSION: int = 1

#: Backends that can write and read this format.
BACKENDS = ("torch", "jax")

#: Grid resolution used by :func:`matches` to compare two models' fields.
_MATCH_POINTS: int = 17


def run_stamp() -> str:
    """Return a unique run id: ``yyyymmddhhmmss`` in UTC, then eight random hex digits.

    The timestamp alone is not an identifier: two runs launched in the same second share
    it, and :func:`default_path` encodes only a few knobs, so arms differing in any other
    one would contend for a filename and the later writer would win. The timestamp keeps
    names sortable; the random suffix makes them unique whatever two runs have in common.
    """
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]


#: Fields excluded from :func:`params_digest`, because they discretise the *reference*
#: and not the channel.
#:
#: ``n_axial`` is the mesh the stiff solver runs on. The surrogate is a continuous
#: function of ``(zeta, t_hat)`` and never sees it, so a model trained while the
#: reference used 160 nodes is the same model when scored against 2560 — and re-scoring
#: against a finer reference is the main reason to keep a checkpoint at all. Including
#: it made the guard fire on the one workflow the guard exists to enable.
_DIGEST_EXCLUDE: frozenset[str] = frozenset({"n_axial"})


def params_digest(p: AxialParams) -> str:
    """Twelve hex digits fingerprinting the *physical* parameters.

    The configuration in the header describes the training run; it says nothing about
    the channel. A model trained at a non-default ``AxialParams`` and reloaded against
    the defaults would rebuild a skeleton of the right shape and evaluate a different
    physical problem — shapes match, so nothing raises. :func:`load` compares this and
    refuses instead.

    Mesh resolution is deliberately **not** part of the fingerprint; see
    :data:`_DIGEST_EXCLUDE`.
    """
    payload = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in sorted(asdict(p).items())
        if k not in _DIGEST_EXCLUDE
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def default_path(
    cfg: Any,  # noqa: ANN401 - either backend's AxialTrainConfig
    backend: str,
    stamp: str | None = None,
    directory: Path = DEFAULT_DIR,
) -> Path:
    """Name a checkpoint after the run that produced it, and when it was produced.

    The configuration alone does not identify a run: the same configuration re-run after
    a code change produces a different model, and two arms differing only in a knob the
    name omits are indistinguishable. :func:`run_stamp` supplies a unique id, so no two
    runs contend for a filename whatever they have in common.
    """
    stamp = stamp or run_stamp()
    n = getattr(cfg, "n_colloc", 0)
    it = getattr(cfg, "lbfgs_iters", 0)
    f = getattr(cfg, "fourier_features", 0)
    ext = "eqx" if backend == "jax" else "pt"
    return directory / f"{backend}_p{n}_i{it}_f{f}_s{cfg.seed}_{stamp}.{ext}"


def _write(path: Path, head: dict, payload: Any) -> Path:  # noqa: ANN401 - a writer callback
    """Write ``head`` then ``payload(file)`` atomically. Returns the path.

    Write to a sibling and rename. ``Path.replace`` is atomic within a filesystem, so the
    destination is either absent or a complete checkpoint, never a half-written one under
    a name that promises a finished run. JAX dispatches asynchronously, so this function
    is reached before the iterations it saves have run and serialisation blocks on the
    device only when it needs the values; writing in place would leave a zero-byte file
    for the length of a whole block.

    The temp name carries the pid so concurrent writers to one destination each get their
    own. With a fixed suffix they would share a file, the first rename would take it, and
    the second would fail with ``FileNotFoundError``.

    The ``finally`` does not run through a ``SIGKILL``, so a killed process can leave a
    ``.partial`` behind — the *destination* guarantee still holds, and
    :func:`sweep_partials` clears the debris.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("wb") as f:
            f.write((json.dumps(head) + "\n").encode())
            payload(f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def sweep_partials(directory: Path = DEFAULT_DIR) -> list[Path]:
    """Delete ``*.partial`` debris left by killed writers. Returns what was removed."""
    gone = []
    for f in Path(directory).glob("*.partial"):
        f.unlink(missing_ok=True)
        gone.append(f)
    return gone


def save(
    path: Path,
    model: Any,  # noqa: ANN401 - either backend's AxialPinn
    cfg: Any,  # noqa: ANN401 - either backend's AxialTrainConfig
    *,
    backend: str,
    p: AxialParams | None = None,
) -> Path:
    """Write ``model`` and ``cfg`` to ``path``, stamped with the time. Returns the path.

    The timestamp is stored inside the file as well as in the name, so a renamed or
    copied checkpoint still identifies the run that produced it.
    """
    if backend not in BACKENDS:
        msg = f"unknown backend {backend!r}; expected one of {list(BACKENDS)}"
        raise ValueError(msg)
    head = {
        "format": FORMAT_VERSION,
        "backend": backend,
        "saved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "params_digest": params_digest(p or AxialParams()),
        "config": asdict(cfg),
    }
    if backend == "jax":
        import equinox as eqx  # noqa: PLC0415 - optional extra, imported only when used

        return _write(path, head, lambda f: eqx.tree_serialise_leaves(f, model))

    import torch  # noqa: PLC0415 - optional extra, imported only when used

    return _write(path, head, lambda f: torch.save(model.state_dict(), f))


def header(path: Path) -> dict:
    """Read a checkpoint's header without loading it, and without importing a backend.

    This is what makes the corpus self-describing and cheap to index: grouping several
    hundred files by what they actually are costs one line read each, needs neither torch
    nor JAX, and does not depend on the filename. A mis-named checkpoint still groups
    correctly.
    """
    with Path(path).open("rb") as f:
        head = json.loads(f.readline().decode())
    got = head.get("format")
    if got != FORMAT_VERSION:
        msg = f"{path}: checkpoint format {got!r}, this build reads {FORMAT_VERSION}"
        raise ValueError(msg)
    return head


def load(path: Path, p: AxialParams | None = None, *, check_params: bool = True) -> tuple:
    """Read back a model, its configuration and the time it was saved.

    The skeleton is rebuilt from the stored configuration and then filled, which is what
    makes the round trip exact: a configuration disagreeing with the weights raises
    rather than loading something subtly wrong.

    Keys the current ``AxialTrainConfig`` does not define are **dropped**, not raised on.
    A checkpoint written before a knob was retired must keep opening, or a single field
    rename makes an entire corpus unreadable at once. A key the config *requires* and the
    file lacks still raises, which is the real error.
    """
    p = p or AxialParams()
    head = header(path)
    backend = head["backend"]
    if check_params and head.get("params_digest") != params_digest(p):
        msg = (
            f"{path}: saved under AxialParams digest {head.get('params_digest')}, "
            f"loading against {params_digest(p)}. The shapes would match and the physics "
            f"would not; pass the parameters it was trained with, or check_params=False."
        )
        raise ValueError(msg)

    if backend == "jax":
        import equinox as eqx  # noqa: PLC0415 - optional extra, imported only when used
        import jax  # noqa: PLC0415 - optional extra, imported only when used

        from pinn_sfr_transient.axial.jaxpinn import (  # noqa: PLC0415 - optional extra
            AxialPinn,
            AxialTrainConfig,
        )

        cfg = _config_from(AxialTrainConfig, head["config"])
        with Path(path).open("rb") as f:
            f.readline()
            skeleton = AxialPinn(cfg, jax.random.PRNGKey(cfg.seed))
            return eqx.tree_deserialise_leaves(f, skeleton), cfg, head["saved_utc"]

    import torch  # noqa: PLC0415 - optional extra, imported only when used

    from pinn_sfr_transient.axial.torchpinn import (  # noqa: PLC0415 - optional extra
        AxialPinn,
        AxialTrainConfig,
    )

    cfg = _config_from(AxialTrainConfig, head["config"])
    with Path(path).open("rb") as f:
        f.readline()
        model = AxialPinn(p, cfg)
        model.load_state_dict(torch.load(f, weights_only=True))
        model.eval()
        return model, cfg, head["saved_utc"]


def _config_from(cls: Any, stored: dict) -> Any:  # noqa: ANN401 - either backend's config
    """Rebuild a config, dropping keys this build no longer defines.

    Tuple-typed fields round-trip through JSON as lists, and a dataclass with ``slots``
    will happily hold the list — which then compares unequal to the default and breaks
    the cross-backend parity assertion. Restore the declared type rather than trusting
    what JSON gave back.
    """
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for k, v in stored.items():
        if k not in known:
            continue
        # `get_origin`, not a substring of `str(annotation)`. Under PEP 563 --
        # `from __future__ import annotations`, which this package no longer uses --
        # `Field.type` is the *string* "tuple[float, ...]"; under 3.14's PEP 649 it is
        # the type object. Sniffing for "tuple" happens to work on both, and would
        # equally match a field annotated `str` in a module with "tuple" in a comment
        # position that ended up in the annotation. Ask the typing API instead.
        kwargs[k] = tuple(v) if isinstance(v, list) and _is_tuple(known[k].type) else v
    return cls(**kwargs)


def _is_tuple(annotation: object) -> bool:
    """Whether a dataclass field is declared as a tuple, under either annotation regime."""
    if isinstance(annotation, str):  # PEP 563 leftovers, or a hand-written string
        return annotation.startswith(("tuple", "Tuple"))
    return get_origin(annotation) is tuple or annotation is tuple


def matches(a: Any, b: Any, p: AxialParams, cfg: Any, backend: str) -> bool:  # noqa: ANN401
    """Report whether two models agree pointwise, which is what a round trip preserves.

    Compares the **fields**, not the weights: two parameter sets that differ but map every
    input to the same output are the same model, and a serialisation bug that permuted a
    layer would pass an array-by-array comparison of the leaves it permuted.
    """
    grid = np.linspace(0.0, 1.0, _MATCH_POINTS)
    t = np.linspace(0.0, p.t_end * cfg.t_train_frac, _MATCH_POINTS)
    if backend == "jax":
        from pinn_sfr_transient.axial.jaxpinn import predict  # noqa: PLC0415 - optional extra

        fa, fb = predict(a, p, grid, t, cfg), predict(b, p, grid, t, cfg)
    else:
        fa, fb = a.predict(grid, t), b.predict(grid, t)
    return all(np.array_equal(x, y) for x, y in zip(fa, fb, strict=True))


def saver(
    cfg: Any,  # noqa: ANN401 - either backend's AxialTrainConfig
    backend: str,
    p: AxialParams | None = None,
    directory: Path = DEFAULT_DIR,
) -> Any:  # noqa: ANN401 - an on_checkpoint callback
    """Return an ``on_checkpoint`` callback that saves the model at each budget rung.

    Both backends' ``train`` already take ``on_checkpoint(iters, model)`` so one run can
    be *scored* at several budgets instead of re-run once per budget. Saving through the
    same hook means one run also *yields* several checkpoints, which is what turns a
    budget ladder from N training runs into one.

    The stamp is drawn once per run rather than per rung, so every checkpoint from one
    run shares it and the ladder can tell "the same run at 20k and 50k" from "two runs".
    """
    stamp = run_stamp()
    p = p or AxialParams()

    def _save(iters: int, model: Any) -> Path:  # noqa: ANN401 - either backend's AxialPinn
        ext = "eqx" if backend == "jax" else "pt"
        n, f = getattr(cfg, "n_colloc", 0), getattr(cfg, "fourier_features", 0)
        name = f"{backend}_p{n}_i{iters}_f{f}_s{cfg.seed}_{stamp}.{ext}"
        return save(directory / name, model, cfg, backend=backend, p=p)

    return _save


def score(path: Path, traj: AxialTrajectory, p: AxialParams | None = None) -> dict[str, float]:
    """Load a checkpoint and score it against ``traj``, by the one shared scorer.

    Goes through :mod:`pinn_sfr_transient.axial.scoring` rather than either backend's
    own evaluator, so a checkpoint's score has the same definition as every published
    number regardless of which backend wrote it.
    """
    from pinn_sfr_transient.axial.scoring import relative_l2  # noqa: PLC0415 - avoids a cycle

    p = p or AxialParams()
    model, cfg, _ = load(path, p)
    if header(path)["backend"] == "jax":
        from pinn_sfr_transient.axial.jaxpinn import predict  # noqa: PLC0415 - optional extra

        fields_ = predict(model, p, traj.zeta, traj.t, cfg)
    else:
        fields_ = model.predict(traj.zeta, traj.t)
    return relative_l2(fields_, traj, p)
