"""Smoke + correctness tests for the from-scratch PyTorch PINN.

Skipped entirely when the optional ``torch`` extra is not installed
(``uv sync --extra torch-cpu``). Configs are deliberately tiny so the whole module
runs in well under a second on CPU.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.pinn_torch import (
    SFRPinn,
    TrainConfig,
    Trainer,
    predict,
    relative_l2,
    train,
)


def _tiny_cfg(**kw: object) -> TrainConfig:
    base = {
        "width": 8,
        "depth": 2,
        "n_colloc": 64,
        "adam_iters": 3,
        "lbfgs_iters": 0,
        "causal_chunks": 4,
        "log_every": 1,
    }
    return TrainConfig(**{**base, **kw})


def test_forward_and_reverse_autodiff_agree() -> None:
    p = SFRParams()
    model = SFRPinn(p, _tiny_cfg())
    t = torch.linspace(0.0, p.t_end, 8, dtype=torch.float64).unsqueeze(1)
    s_fwd, d_fwd = model._deriv_forward(t)
    s_rev, d_rev = model._deriv_reverse(t)
    assert torch.allclose(s_fwd, s_rev, atol=1e-9)
    assert torch.allclose(d_fwd, d_rev, atol=1e-6)


def test_falls_back_when_forward_mode_is_silently_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    # Some torch builds (e.g. on hosted notebooks) miscompute jvp+vmap without
    # raising; the one-time check must catch it and switch to reverse mode.
    p = SFRParams()
    model = SFRPinn(p, _tiny_cfg())
    t = torch.linspace(0.0, p.t_end, 8, dtype=torch.float64).unsqueeze(1)
    _, d_correct = model._deriv_reverse(t)
    monkeypatch.setattr(
        model, "_deriv_forward", lambda tt: (model._state(tt), torch.zeros_like(d_correct))
    )

    _, d = model.state_and_deriv(t)
    assert model.cfg.jacobian == "reverse"  # auto-switched away from broken forward mode
    assert torch.allclose(d, d_correct, atol=1e-6)  # returns the correct reverse-mode result


def test_hard_initial_condition() -> None:
    p = SFRParams()
    model = SFRPinn(p, _tiny_cfg())
    t0 = torch.zeros(1, 1, dtype=torch.float64)
    # s(0) must equal the normalized nominal state s0 = ones(9), by construction.
    assert torch.allclose(model(t0), torch.ones(1, 9, dtype=torch.float64))


def test_tiny_training_runs_and_predicts() -> None:
    p = SFRParams()
    cfg = _tiny_cfg()
    model = Trainer(SFRPinn(p, cfg), p, cfg).train(verbose=False)
    pred = predict(model, n=16)
    assert pred["P"].shape == (16,)
    assert all(np.all(np.isfinite(pred[k])) for k in ("t", "P", "Tf", "Tc"))
    # relative_l2 against an identical reference must be ~0.
    errs = relative_l2(pred, {k: pred[k] for k in ("t", "P", "Tf", "Tc")})
    assert max(errs.values()) < 1e-9


def test_adaptive_components_and_lbfgs() -> None:
    p = SFRParams()
    cfg = _tiny_cfg(lbfgs_iters=2)
    trainer = Trainer(SFRPinn(p, cfg), p, cfg)

    trainer.update_block_weights(trainer.collocation())  # gradient-norm weighting
    assert trainer.block_w.shape == (4,)
    trainer.rar_refine()  # residual-based adaptive refinement
    assert trainer.rar_points.shape[0] > 0
    assert torch.isfinite(trainer.causal_loss(trainer.collocation()))

    model = trainer.train(verbose=False)  # also exercises the L-BFGS polish branch
    assert np.all(np.isfinite(predict(model, n=4)["P"]))


def test_convenience_train_entrypoint() -> None:
    model = train(SFRParams(), _tiny_cfg())
    assert isinstance(model, SFRPinn)
