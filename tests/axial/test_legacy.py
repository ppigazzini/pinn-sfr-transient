"""The frozen reader for the imported checkpoint corpus.

Two kinds of test. The **structural** ones need only ``jax`` and pin the pytree the 334
files were serialised against — those run everywhere and are what stops a future
refactor from silently orphaning the corpus. The **corpus** ones need the files
themselves, are skipped when ``models/`` is absent (it is gitignored), and include the
control arm that validates the whole transfer.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pinn_sfr_transient.axial import legacy
from pinn_sfr_transient.axial.config import AxialParams

MODELS = Path("models")
_corpus = sorted(MODELS.rglob("*.eqx")) if MODELS.is_dir() else []
needs_corpus = pytest.mark.skipif(not _corpus, reason="models/ corpus not present")


# --- the frozen shape, which is the whole contract -------------------------
def test_float64_is_enabled_by_importing_this_module():
    """The corpus was trained in double precision; a float32 skeleton down-casts it.

    Silently: the file opens, the fields look plausible, every score is subtly wrong.
    This module is importable without `jaxpinn`, so it cannot rely on that package's
    flag having been set.
    """
    assert jax.numpy.zeros(1).dtype == np.float64


def test_the_pytree_is_two_fields_in_this_order():
    """`eqx` fills leaves in declaration order; reordering misfills all 334 files.

    This is not a style assertion. The live `jaxpinn.AxialPinn` declares five fields
    starting with `mlp`; if this module ever drifts toward it, every checkpoint either
    raises or -- worse, where shapes happen to agree -- loads scrambled.
    """
    assert [f.name for f in fields(legacy.AxialPinn)] == ["embed", "mlp"]


def test_the_config_keeps_the_companion_field_names():
    """`points` and `iters`, not `n_colloc` and `lbfgs_iters`.

    Renaming them would make every stored header unreadable through the unknown-key
    filter -- silently, and losing exactly the two axes the ladder groups by.
    """
    names = {f.name for f in fields(legacy.TrainConfig)}
    assert {"points", "iters"} <= names
    assert not names & {"n_colloc", "lbfgs_iters"}


def test_the_embedding_takes_four_arguments():
    """The live embedding also takes a scale vector and bands; this one must not."""
    emb = legacy.FourierEmbedding(2, 8, 2.0, jax.random.PRNGKey(0))
    assert emb.B.shape == (2, 8)


def test_the_trunk_width_follows_the_embedding():
    """`in_size = 2 * fourier_features` is what the stored weight shapes assume."""
    cfg = legacy.TrainConfig(fourier_features=16, width=8, depth=2)
    model = legacy.AxialPinn(cfg, jax.random.PRNGKey(0))
    assert model.mlp.layers[0].weight.shape == (8, 32)


def test_the_ansatz_starts_at_the_steady_profile():
    """`exp(0) = 1`, so `t_hat = 0` must return theta_0 exactly, for any weights."""
    p, cfg = AxialParams(), legacy.TrainConfig(fourier_features=8, width=8, depth=2)
    model = legacy.AxialPinn(cfg, jax.random.PRNGKey(0))
    z = jax.numpy.asarray([0.4])
    state = legacy.normalised_state(model, p, z, jax.numpy.zeros((1,)))
    base = legacy.theta0(p, z)
    assert np.allclose(np.asarray(state)[:4], np.asarray(base)[:4])


def test_the_inlet_boundary_condition_is_identically_satisfied():
    """theta_0 vanishes at zeta = 0 for the coolant, pinning the upstream condition."""
    p = AxialParams()
    base = np.asarray(legacy.theta0(p, jax.numpy.zeros((1,))))
    assert base[3] == pytest.approx(0.0, abs=1e-12)


def test_the_horizon_is_the_trained_window_not_the_nominal_one():
    p, cfg = AxialParams(), legacy.TrainConfig(t_train_frac=0.275)
    assert legacy.horizon(p, cfg) == pytest.approx(p.t_end * 0.275)


def test_unknown_header_keys_are_dropped(tmp_path):
    """Retired knobs are still in the stored headers; all 334 must keep opening."""
    import equinox as eqx

    cfg = legacy.TrainConfig(fourier_features=4, width=4, depth=1)
    model = legacy.AxialPinn(cfg, jax.random.PRNGKey(0))
    path = tmp_path / "old.eqx"
    with path.open("wb") as f:
        stale = {
            "saved_utc": "2026-01-01T00:00:00+00:00",
            "config": {
                "points": 5000,
                "iters": 10000,
                "fourier_features": 4,
                "width": 4,
                "depth": 1,
                "drift_points": 12,
                "points_file": "x.npy",
                "residual_scaling": True,
                "uniform_collocation": False,
                "sf_warmup_frac": 0.1,
            },
        }
        import json

        f.write((json.dumps(stale) + "\n").encode())
        eqx.tree_serialise_leaves(f, model)
    back, cfg_back, saved = legacy.load(path)
    assert cfg_back.points == 5000
    assert cfg_back.iters == 10000
    assert saved.startswith("2026")
    assert back is not None


# --- the corpus itself ------------------------------------------------------
@needs_corpus
def test_every_checkpoint_header_parses():
    """A header that will not parse is a file the ladder silently drops."""
    bad = []
    for f in _corpus:
        try:
            head = legacy.header(f)
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            bad.append(f"{f.name}: {exc}")
            continue
        if "config" not in head:
            bad.append(f"{f.name}: no config")
    assert not bad, bad[:5]


@needs_corpus
def test_the_corpus_is_the_expected_size():
    """334 IMPORTED files. A recursive glob finds them; a flat one finds 279 silently.

    Counts only the companion's naming (`pNNNN_iNNNN_...`), because `models/` also
    accumulates checkpoints this repository writes, which carry a `jax_`/`torch_` prefix.
    Asserting on the total made the test fail the moment a local run saved a rung.
    """
    imported = [f for f in _corpus if not f.name.startswith(("jax_", "torch_"))]
    assert len(imported) == 334


@needs_corpus
def test_a_sample_of_the_corpus_actually_loads():
    """Headers parsing is not the same as weights fitting the skeleton."""
    for f in _corpus[:: max(1, len(_corpus) // 8)]:
        model, cfg, _ = legacy.load(f)
        assert cfg.fourier_features > 0
        out = model(jax.numpy.asarray([0.5, 0.5]))
        assert out.shape == (legacy.N_TEMPS + 1,)
        assert np.all(np.isfinite(np.asarray(out)))
