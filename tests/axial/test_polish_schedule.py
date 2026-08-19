"""The staged polish: freezing part-way, and checkpoints that cost nothing.

Two knobs, and one property that carries all the weight. ``polish_checkpoints`` exists so
one run can be scored at several budgets instead of being run once per budget -- but that
economy is only sound if a checkpoint does **not** change the trajectory. If the optimiser
were restarted at each stop, every intermediate row would be measuring the checkpointing
rather than the budget, and `docs/axial_nn.md` section 7.5.37 measured exactly that restart
at 1.5x worse. So the tests below assert the run is unchanged, in both backends, rather
than asserting that the callback fires.

``freeze_after`` is the other half: the encoder is trainable for the first block and held
for the rest. Freezing changes the *curvature dimension*, not the fitting capacity
(section 7.5.37a), so the thing to check is that the projection genuinely stops moving.
"""

import numpy as np
import pytest

CPS = (4, 7, 10)  # includes the final total: checkpoints are EXACTLY what is asked for
ITERS = 10


def _torch_cfg(**kw):
    from pinn_sfr_transient.axial.torchpinn.config import AxialTrainConfig

    return AxialTrainConfig(
        width=8, depth=2, n_colloc=32, adam_iters=0, lbfgs_iters=ITERS, log_every=10**9, **kw
    )


def _jax_cfg(**kw):
    from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig

    return AxialTrainConfig(
        width=8, depth=2, n_colloc=32, adam_iters=0, lbfgs_iters=ITERS, log_every=10**9, **kw
    )


def _jax_fields(model, p, cfg):
    from pinn_sfr_transient.axial import pinn_jax as pj

    zeta = np.linspace(0.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 5)
    return np.stack([np.asarray(f) for f in pj.predict(model, p, zeta, t, cfg)])


def _torch_fields(model):
    zeta = np.linspace(0.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 5)
    return np.stack(model.predict(zeta, t))


def test_torch_checkpoints_do_not_change_the_run():
    """The torch twin, to tolerance rather than bitwise.

    ``torch.optim.LBFGS`` re-evaluates the closure once on entry to every ``.step()``,
    so splitting one call into three does a little more arithmetic in a different order.
    The history is carried, so the *solve* is the same one; the digits are allowed to
    differ in the last few.
    """
    torch = pytest.importorskip("torch")
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.torchpinn.training import train

    torch.manual_seed(0)
    plain = _torch_fields(train(AxialParams(), _torch_cfg()))
    seen: list[int] = []
    torch.manual_seed(0)
    ckpt = _torch_fields(
        train(
            AxialParams(),
            _torch_cfg(polish_checkpoints=CPS),
            on_checkpoint=lambda n, _m: seen.append(n),
        )
    )
    np.testing.assert_allclose(ckpt, plain, rtol=1e-8, atol=1e-8)
    assert seen == list(CPS)


def test_the_checkpoint_callback_sees_the_intermediate_state_not_the_final_one():
    """A checkpoint that handed back the finished model would silently duplicate rows.

    The torch backend mutates parameters in place, so it has to rewind, hand over and
    restore -- the failure mode is real rather than hypothetical, and it would produce a
    study file where every budget scored identically.
    """
    torch = pytest.importorskip("torch")
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.torchpinn.training import train

    torch.manual_seed(0)
    snaps: dict[int, np.ndarray] = {}
    final = train(
        AxialParams(),
        _torch_cfg(polish_checkpoints=(3, ITERS)),
        on_checkpoint=lambda n, m: snaps.__setitem__(n, _torch_fields(m)),
    )
    assert set(snaps) == {3, ITERS}
    assert not np.allclose(snaps[3], _torch_fields(final)), "checkpoint equals the final model"
    # ...and the model handed back to the caller must be the finished one, not a rewind.
    np.testing.assert_array_equal(snaps[ITERS], _torch_fields(final))


def test_first_order_checkpoints_fire_without_a_polish():
    """A pure first-order arm produced no checkpoints at all before this.

    `on_checkpoint` was reachable only from the quasi-Newton stage, so with
    `lbfgs_iters = 0` a ten-rung budget ladder meant ten training runs instead of one.
    """
    jax = pytest.importorskip("jax")
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.jaxpinn import AxialTrainConfig, train

    seen: list[int] = []
    cfg = AxialTrainConfig(
        width=8,
        depth=2,
        fourier_features=8,
        n_colloc=32,
        adam_iters=20,
        lbfgs_iters=0,
        adam_checkpoint_every=5,
        seed=0,
    )
    train(AxialParams(), cfg, verbose=False, on_checkpoint=lambda n, _m: seen.append(n))
    assert seen == [5, 10, 15, 20], "a rung is named by the iterations actually taken"
    assert jax is not None


def test_first_order_checkpoints_are_off_by_default():
    """Zero disables them, which is every published number here."""
    pytest.importorskip("jax")
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.jaxpinn import AxialTrainConfig, train

    seen: list[int] = []
    cfg = AxialTrainConfig(
        width=8,
        depth=2,
        fourier_features=8,
        n_colloc=32,
        adam_iters=10,
        lbfgs_iters=0,
        seed=0,
    )
    train(AxialParams(), cfg, verbose=False, on_checkpoint=lambda n, _m: seen.append(n))
    assert seen == []


def test_rar_every_zero_disables_resampling_rather_than_raising():
    """0 means OFF, as it does for every sibling cadence knob in this config.

    `it % cfg.rar_every` raised ZeroDivisionError, so residual-adaptive resampling could
    not be turned off at all. That mattered: RAR is the one thing this training loop adds
    to a first-order batch that the companion's converging AdEMAMix arm never had, and it
    could not be ablated to find out whether it was the cause.
    """
    pytest.importorskip("jax")
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.jaxpinn import AxialTrainConfig, train

    cfg = AxialTrainConfig(
        width=8,
        depth=2,
        fourier_features=8,
        n_colloc=32,
        adam_iters=12,
        lbfgs_iters=0,
        rar_every=0,
        seed=0,
    )
    model, _, _ = train(AxialParams(), cfg, verbose=False)
    assert model is not None


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_a_polish_that_dies_leaves_its_earlier_rungs_behind(backend):
    """Checkpoints must reach the caller AS THEY ARE REACHED, not after the stage ends.

    Both backends used to collect the polish snapshots in memory and hand them over once
    the stage finished -- and only if the divergence guard approved. Two consequences,
    both silent: a run stopped at 90 000 of 100 000 iterations left NOTHING, and a polish
    that ended badly discarded the good rungs it had already earned along with the bad
    ending, destroying the evidence of where it went wrong.

    The check is the failure itself. The callback raises at the second rung; the first
    must already have been handed over.
    """
    seen: list[int] = []

    class StopError(RuntimeError):
        pass

    def on_checkpoint(n, _model):
        seen.append(n)
        if len(seen) == 2:
            raise StopError

    from pinn_sfr_transient.axial import AxialParams

    p = AxialParams(n_axial=20)
    kw = {
        "seed": 0,
        "fourier_features": 16,
        "width": 8,
        "depth": 2,
        "adam_iters": 0,
        "lbfgs_iters": 40,
        "n_colloc": 64,
        "polish_colloc": 64,
        "rar_every": 0,
        "polish_checkpoints": (10, 20, 30),
    }
    if backend == "jax":
        mod = pytest.importorskip("pinn_sfr_transient.axial.pinn_jax")
        run = lambda: mod.train(  # noqa: E731
            p,
            mod.AxialTrainConfig(log_every=10**9, **kw),
            verbose=False,
            on_checkpoint=on_checkpoint,
        )
    else:
        # Gate on `torch` and NOT on the backend package: `torchpinn` turns a missing
        # extra into `SystemExit`, which is a BaseException rather than an ImportError,
        # so `importorskip` on the package cannot catch it and the test ERRORS instead
        # of skipping. The JAX twin imports `jax` bare and raises ModuleNotFoundError,
        # which is why only this arm broke the two backend-alone lanes.
        pytest.importorskip("torch")
        mod = pytest.importorskip("pinn_sfr_transient.axial.torchpinn")
        run = lambda: mod.train(  # noqa: E731
            p, mod.AxialTrainConfig(log_every=10**9, **kw), on_checkpoint=on_checkpoint
        )
    with pytest.raises(StopError):
        run()
    assert seen == [10, 20], f"{backend}: rungs were buffered, not emitted as reached"
