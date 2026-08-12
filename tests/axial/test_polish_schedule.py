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

from __future__ import annotations

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


@pytest.mark.parametrize("freeze_after", [0, 5])  # 5 is deliberately NOT a checkpoint
def test_jax_checkpoints_do_not_change_the_run(freeze_after):
    """The final model must be what an uninterrupted solve would have produced.

    Bitwise, because the segmented loop runs the identical sequence of updates on the
    identical state -- only the Python-level loop boundary differs. Anything weaker here
    would let a restart hide inside a rounding tolerance.
    """
    pytest.importorskip("jax")
    from pinn_sfr_transient.axial import pinn_jax as pj
    from pinn_sfr_transient.axial.config import AxialParams

    plain = pj.train(AxialParams(), _jax_cfg(freeze_after=freeze_after), verbose=False)
    seen: list[int] = []
    ckpt = pj.train(
        AxialParams(),
        _jax_cfg(freeze_after=freeze_after, polish_checkpoints=CPS),
        verbose=False,
        on_checkpoint=lambda n, _m: seen.append(n),
    )
    np.testing.assert_array_equal(_jax_fields(*ckpt), _jax_fields(*plain))
    # Exactly the requested totals. `freeze_after = 5` splits the polish and its stage
    # boundary must NOT appear here: a boundary is not a checkpoint, and an earlier
    # revision emitted one, adding a row to every study file that nobody asked for.
    assert seen == list(CPS), "checkpoints must fire at the requested totals, and only those"


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


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_freeze_after_actually_stops_the_encoder_moving(backend):
    """After the switch the Fourier read-out must be identically fixed, in both backends.

    Asserted against the *frozen half* rather than against an accuracy: the projection
    moves during the first block and must not move at all during the second.
    """
    pytest.importorskip(backend)
    from pinn_sfr_transient.axial.config import AxialParams

    if backend == "torch":
        import torch

        from pinn_sfr_transient.axial.torchpinn.training import train

        def read_out(m):
            first = next(q for q in m.net.modules() if isinstance(q, torch.nn.Linear))
            return first.weight.detach().numpy().copy()

        torch.manual_seed(0)
        seen: dict[int, np.ndarray] = {}
        train(
            AxialParams(),
            _torch_cfg(freeze_after=5, polish_checkpoints=(4, 7, ITERS)),
            on_checkpoint=lambda n, m: seen.__setitem__(n, read_out(m)),
        )
    else:
        from pinn_sfr_transient.axial import pinn_jax as pj

        def read_out(m):
            return np.asarray(m.mlp.layers[0].weight).copy()

        seen = {}
        pj.train(
            AxialParams(),
            _jax_cfg(freeze_after=5, polish_checkpoints=(4, 7, ITERS)),
            verbose=False,
            on_checkpoint=lambda n, m: seen.__setitem__(n, read_out(m)),
        )

    assert set(seen) == {4, 7, ITERS}
    # 4 is before the switch and 7 after it, so the read-out moves between 4 and 7 but
    # must be identical from 7 onward.
    np.testing.assert_array_equal(seen[ITERS], seen[7])
    assert not np.array_equal(seen[7], seen[4]), "the encoder never moved even while free"


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_freeze_after_refuses_to_share_the_stage_with_polish_refresh(backend):
    """Both knobs schedule the same stage; silently picking one is how defaults hide."""
    pytest.importorskip(backend)
    from pinn_sfr_transient.axial.config import AxialParams

    if backend == "torch":
        from pinn_sfr_transient.axial.torchpinn.training import train

        cfg = _torch_cfg(freeze_after=4, polish_refresh=2)
    else:
        from pinn_sfr_transient.axial import pinn_jax as pj

        train = pj.train
        cfg = _jax_cfg(freeze_after=4, polish_refresh=2)

    with pytest.raises(ValueError, match="same stage"):
        train(AxialParams(), cfg)
