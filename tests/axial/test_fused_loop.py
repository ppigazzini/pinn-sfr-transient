"""The first-order loop runs fused, and fusing moved no cadence event.

The draw and the update used to be three dispatch round-trips per iteration with
Python in between; **15.6% of wall time sat outside the compiled region** and the cores
idled through it. They now run inside one `lax.fori_loop`, with Python only at cadence
boundaries -- RAR, weights, pseudo-time, logging, checkpoints.

That is only safe if every event still fires on the iteration it fired on before, which
is what `_next_boundary` guarantees and what these pin.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.jaxpinn import AxialTrainConfig, train
from pinn_sfr_transient.axial.jaxpinn.training import _next_boundary


def _boundaries(cfg, *, verbose, ckpt):
    it, evs = 0, []
    while it < cfg.adam_iters:
        it += _next_boundary(cfg, it, verbose=verbose, checkpointing=ckpt)
        evs.append(it)
    return evs


def _expected(cfg, *, verbose, ckpt):
    out = {cfg.adam_iters}
    for it in range(1, cfg.adam_iters + 1):
        if cfg.rar_every and it % cfg.rar_every == 0:
            out.add(it)
        if cfg.pts_every and it % cfg.pts_every == 0:
            out.add(it)
        if (
            cfg.weight_max_ratio > 1.0
            and cfg.weight_update_every
            and (it % cfg.weight_update_every == 0)
        ):
            out.add(it)
        if verbose and cfg.log_every and it % cfg.log_every == 0:
            out.add(it)
        if ckpt and cfg.adam_checkpoint_every and it % cfg.adam_checkpoint_every == 0:
            out.add(it)
    return sorted(out)


@pytest.mark.parametrize(
    ("name", "kw", "verbose", "ckpt"),
    [
        (
            "study",
            {
                "adam_iters": 20000,
                "rar_every": 2000,
                "log_every": 1000,
                "adam_checkpoint_every": 5000,
            },
            True,
            True,
        ),
        ("odd periods", {"adam_iters": 1000, "rar_every": 300, "log_every": 70}, True, False),
        (
            "pseudo-time",
            {"adam_iters": 900, "rar_every": 0, "log_every": 0, "pts_every": 250},
            False,
            False,
        ),
    ],
)
def test_no_cadence_event_moves(name, kw, verbose, ckpt):
    """Odd, non-dividing periods are the case a naive block size gets wrong."""
    cfg = AxialTrainConfig(**kw)
    assert _boundaries(cfg, verbose=verbose, ckpt=ckpt) == _expected(
        cfg, verbose=verbose, ckpt=ckpt
    ), name


def test_with_every_cadence_off_the_whole_budget_is_one_compiled_loop():
    """The point of the change: no Python between iterations at all."""
    cfg = AxialTrainConfig(adam_iters=50000, rar_every=0, log_every=0)
    assert _next_boundary(cfg, 0, verbose=False, checkpointing=False) == 50000


def test_a_boundary_is_never_zero():
    """A zero-length block would spin forever."""
    cfg = AxialTrainConfig(adam_iters=100, rar_every=1, log_every=1)
    for it in range(100):
        assert _next_boundary(cfg, it, verbose=True, checkpointing=True) >= 1


def test_the_fused_loop_is_deterministic():
    """Same seed, same machine, same answer -- twice."""
    p = AxialParams()
    cfg = AxialTrainConfig(
        width=8,
        depth=2,
        fourier_features=8,
        n_colloc=64,
        adam_colloc=32,
        adam_iters=30,
        lbfgs_iters=0,
        rar_every=0,
        seed=0,
    )
    from pinn_sfr_transient.axial.jaxpinn import predict

    z, t = np.linspace(0, 1, 5), np.linspace(0, 4, 5)
    a = np.asarray(predict(train(p, cfg, verbose=False)[0], p, z, t, cfg)[0])
    b = np.asarray(predict(train(p, cfg, verbose=False)[0], p, z, t, cfg)[0])
    assert np.array_equal(a, b)


def test_the_fused_loop_trains_and_stays_finite():
    p = AxialParams()
    cfg = AxialTrainConfig(
        width=8,
        depth=2,
        fourier_features=8,
        n_colloc=64,
        adam_colloc=32,
        adam_iters=40,
        lbfgs_iters=0,
        rar_every=10,
        adam_checkpoint_every=20,
        seed=0,
    )
    seen = []
    model, _, _ = train(p, cfg, verbose=False, on_checkpoint=lambda n, _m: seen.append(n))
    assert seen == [20, 40], "checkpoints still land on their own cadence"
    from pinn_sfr_transient.axial.jaxpinn import predict

    f = np.asarray(predict(model, p, np.linspace(0, 1, 5), np.linspace(0, 4, 5), cfg)[0])
    assert np.isfinite(f).all()
