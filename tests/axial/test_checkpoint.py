"""Checkpointing: the header contract, the atomic write, and the round trip.

Split into two halves on purpose. The header, the stamp, the digest and the atomic
write are **backend-free** and are tested with no extra installed, because that is the
lane where a missing-extra import path is actually exercised. The round trips need their
backend and ``importorskip`` out otherwise.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields

import numpy as np
import pytest

from pinn_sfr_transient.axial import checkpoint
from pinn_sfr_transient.axial.config import AxialParams


# --- backend-free: header, naming, digest, atomicity -----------------------
def test_run_stamp_is_unique_across_a_tight_loop():
    """Two arms launched in the same second must not contend for a filename."""
    stamps = {checkpoint.run_stamp() for _ in range(500)}
    assert len(stamps) == 500


def test_run_stamp_sorts_chronologically():
    """The timestamp leads so `ls` orders a directory by when each run finished."""
    s = checkpoint.run_stamp()
    assert len(s.split("-")[0]) == 14
    assert s.split("-")[0].isdigit()


def test_params_digest_moves_with_the_physics():
    """A digest that does not change when the channel changes would guard nothing."""
    base = AxialParams()
    assert checkpoint.params_digest(base) == checkpoint.params_digest(AxialParams())
    assert checkpoint.params_digest(base) != checkpoint.params_digest(AxialParams(T_in=630.0))
    assert checkpoint.params_digest(base) != checkpoint.params_digest(AxialParams(t_end=30.0))


def test_params_digest_ignores_the_reference_mesh():
    """Re-scoring against a finer reference is the main reason to keep a checkpoint.

    The surrogate is a continuous function of `(zeta, t_hat)` and never sees `n_axial`,
    which discretises the reference solver alone. Fingerprinting it made the guard fire
    on exactly the workflow the guard exists to enable -- caught by the first end-to-end
    ladder run, not by any unit test written before it.
    """
    base = AxialParams()
    for n in (20, 320, 2560):
        assert checkpoint.params_digest(base) == checkpoint.params_digest(AxialParams(n_axial=n))


def test_params_digest_survives_the_numpy_fields():
    """`asdict` yields arrays for the six-group data; JSON cannot take them raw."""
    assert len(checkpoint.params_digest(AxialParams())) == 12


def test_write_is_atomic_and_leaves_no_partial(tmp_path):
    """The destination is either absent or complete -- never a truncated file."""
    dest = tmp_path / "m.eqx"
    checkpoint._write(dest, {"format": 1}, lambda f: f.write(b"payload"))
    assert dest.read_bytes().endswith(b"payload")
    assert list(tmp_path.glob("*.partial")) == []


def test_a_failed_write_leaves_the_destination_untouched(tmp_path):
    """A writer that raises must not replace a good checkpoint with a bad one."""
    dest = tmp_path / "m.eqx"
    checkpoint._write(dest, {"format": 1}, lambda f: f.write(b"first"))
    good = dest.read_bytes()

    def boom(_f):
        msg = "device died mid-write"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="device died"):
        checkpoint._write(dest, {"format": 1}, boom)
    assert dest.read_bytes() == good
    assert list(tmp_path.glob("*.partial")) == []


def test_concurrent_writers_get_their_own_temp_file(tmp_path):
    """A fixed temp suffix would make the second writer fail on the first's rename."""
    dest = tmp_path / "m.eqx"
    seen: list[str] = []
    checkpoint._write(dest, {"format": 1}, lambda f: seen.append(f.name))
    assert str(os.getpid()) in seen[0]
    assert seen[0].endswith(".partial")


def test_header_reads_without_a_backend(tmp_path):
    """Indexing a corpus must cost one line read and no optional extra."""
    dest = tmp_path / "m.eqx"
    head = {"format": checkpoint.FORMAT_VERSION, "backend": "jax", "config": {"seed": 3}}
    checkpoint._write(dest, head, lambda f: f.write(b"\x00" * 1024))
    assert checkpoint.header(dest)["config"]["seed"] == 3


def test_header_refuses_an_unknown_format(tmp_path):
    """A silently misread checkpoint is worse than one that will not open."""
    dest = tmp_path / "m.eqx"
    checkpoint._write(dest, {"format": 999, "backend": "jax"}, lambda f: f.write(b""))
    with pytest.raises(ValueError, match="checkpoint format"):
        checkpoint.header(dest)


def test_save_refuses_an_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown backend"):
        checkpoint.save(tmp_path / "m.pt", None, None, backend="tensorflow")


def test_sweep_partials_clears_debris_a_kill_left(tmp_path):
    """`finally` does not run through SIGKILL, so the debris needs a broom."""
    (tmp_path / "a.eqx.123.partial").write_bytes(b"junk")
    (tmp_path / "b.eqx.456.partial").write_bytes(b"junk")
    (tmp_path / "keep.eqx").write_bytes(b"real")
    assert len(checkpoint.sweep_partials(tmp_path)) == 2
    assert (tmp_path / "keep.eqx").exists()


def test_default_path_separates_the_backends(tmp_path):
    """Two backends at one configuration must not collide on a name."""
    jax_cfg = pytest.importorskip("pinn_sfr_transient.axial.jaxpinn").AxialTrainConfig()
    j = checkpoint.default_path(jax_cfg, "jax", stamp="S", directory=tmp_path)
    t = checkpoint.default_path(jax_cfg, "torch", stamp="S", directory=tmp_path)
    assert j != t
    assert j.suffix == ".eqx"
    assert t.suffix == ".pt"


# --- the config filter, which is what keeps an old corpus readable ---------
def test_unknown_keys_are_dropped_and_known_ones_kept():
    """One field rename must not make several hundred files unreadable at once."""
    cfg_mod = pytest.importorskip("pinn_sfr_transient.axial.jaxpinn")
    stored = {"seed": 7, "width": 32, "drift_points": 999, "uniform_collocation": True}
    cfg = checkpoint._config_from(cfg_mod.AxialTrainConfig, stored)
    assert cfg.seed == 7
    assert cfg.width == 32
    assert not hasattr(cfg, "drift_points")


def test_tuple_fields_survive_the_json_round_trip():
    """JSON gives lists back; a list where a tuple is declared breaks config equality."""
    cfg_mod = pytest.importorskip("pinn_sfr_transient.axial.jaxpinn")
    tuple_fields = [f.name for f in fields(cfg_mod.AxialTrainConfig) if "tuple" in str(f.type)]
    assert tuple_fields, "expected at least one tuple-typed knob to guard"
    stored = {name: [1.0, 2.0] for name in tuple_fields}
    cfg = checkpoint._config_from(cfg_mod.AxialTrainConfig, stored)
    for name in tuple_fields:
        assert isinstance(getattr(cfg, name), tuple)


# --- round trips, one per backend ------------------------------------------
def _tiny(cfg_cls):
    return cfg_cls(width=8, depth=2, fourier_features=4, n_colloc=16, seed=0)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_round_trip_preserves_the_fields(tmp_path, backend):
    """The model that comes back must map every input where the saved one did."""
    pytest.importorskip("jax" if backend == "jax" else "torch")
    p = AxialParams()
    if backend == "jax":
        import jax

        from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(cfg, jax.random.PRNGKey(0))
    else:
        from pinn_sfr_transient.axial.torchpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(p, cfg)

    path = checkpoint.save(tmp_path / f"m_{backend}", model, cfg, backend=backend, p=p)
    back, cfg_back, saved = checkpoint.load(path, p)
    assert cfg_back == cfg
    assert saved
    assert checkpoint.matches(model, back, p, cfg, backend)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_load_refuses_the_wrong_physical_parameters(tmp_path, backend):
    """Shapes would match and the physics would not, so nothing else would notice."""
    pytest.importorskip("jax" if backend == "jax" else "torch")
    p = AxialParams()
    if backend == "jax":
        import jax

        from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(cfg, jax.random.PRNGKey(0))
    else:
        from pinn_sfr_transient.axial.torchpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(p, cfg)

    path = checkpoint.save(tmp_path / f"m_{backend}", model, cfg, backend=backend, p=p)
    with pytest.raises(ValueError, match="digest"):
        checkpoint.load(path, AxialParams(T_in=630.0))
    # A different reference mesh is NOT a mismatch -- see the digest test above.
    assert checkpoint.load(path, AxialParams(n_axial=320))
    # The override still exists for a deliberate cross-physics load.
    assert checkpoint.load(path, AxialParams(T_in=630.0), check_params=False)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_header_identifies_the_writer(tmp_path, backend):
    """`ladder` groups by what a file is, which it reads from here."""
    pytest.importorskip("jax" if backend == "jax" else "torch")
    p = AxialParams()
    if backend == "jax":
        import jax

        from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(cfg, jax.random.PRNGKey(0))
    else:
        from pinn_sfr_transient.axial.torchpinn import AxialPinn, AxialTrainConfig

        cfg = _tiny(AxialTrainConfig)
        model = AxialPinn(p, cfg)

    path = checkpoint.save(tmp_path / f"m_{backend}", model, cfg, backend=backend, p=p)
    head = checkpoint.header(path)
    assert head["backend"] == backend
    assert head["config"]["width"] == 8
    assert head["params_digest"] == checkpoint.params_digest(p)


def test_a_renamed_checkpoint_still_identifies_its_run(tmp_path):
    """Nothing depends on the filename; a mis-named file groups by what it is."""
    pytest.importorskip("jax")
    import jax

    from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig

    p, cfg = AxialParams(), _tiny(AxialTrainConfig)
    path = checkpoint.save(
        tmp_path / "a.eqx", AxialPinn(cfg, jax.random.PRNGKey(0)), cfg, backend="jax", p=p
    )
    renamed = path.rename(tmp_path / "totally-wrong-name.eqx")
    assert checkpoint.header(renamed)["config"]["seed"] == cfg.seed
    assert checkpoint.header(renamed)["backend"] == "jax"


def test_the_header_is_one_json_line_then_the_payload(tmp_path):
    """The format is documented as such and other tools read it that way."""
    pytest.importorskip("jax")
    import jax

    from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig

    p, cfg = AxialParams(), _tiny(AxialTrainConfig)
    path = checkpoint.save(
        tmp_path / "a.eqx", AxialPinn(cfg, jax.random.PRNGKey(0)), cfg, backend="jax", p=p
    )
    with path.open("rb") as f:
        head = json.loads(f.readline().decode())
        assert f.read(1) != b""
    assert head["format"] == checkpoint.FORMAT_VERSION


def test_scoring_a_checkpoint_uses_the_shared_scorer(tmp_path):
    """A score must have one definition whichever backend wrote the file."""
    pytest.importorskip("jax")
    import jax

    from pinn_sfr_transient.axial.jaxpinn import AxialPinn, AxialTrainConfig
    from pinn_sfr_transient.axial.reference import solve_reference

    p = AxialParams(n_axial=20)
    cfg = _tiny(AxialTrainConfig)
    path = checkpoint.save(
        tmp_path / "a.eqx", AxialPinn(cfg, jax.random.PRNGKey(0)), cfg, backend="jax", p=p
    )
    traj = solve_reference(p, n_out=9)
    row = checkpoint.score(path, traj, p)
    assert {"T_f", "T_cl", "T_s", "T_c"} <= set(row)
    assert all(np.isfinite(row[k]) for k in ("T_f", "T_cl", "T_s", "T_c"))
