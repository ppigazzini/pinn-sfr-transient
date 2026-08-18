"""The torch backend must ACT on the config, not merely accept it.

``tools/backend_smoke.py`` compares the two ``AxialTrainConfig`` dataclasses field for
field, which is what stops a knob landing in one backend only. It cannot see the next
failure along: a field present in both, with equal defaults, that one backend reads and
the other silently drops. Six were in that state here --

===================== ==========================================================
``adam_colloc``       ignored; ``n_colloc`` drove both stages, so an arm asking for
                      a small first-order batch ran full batch
``polish_colloc``     ignored, the same way
``polish_refresh``    ignored; the polish never redrew and never restarted
``adam_checkpoint_every`` ignored; a pure first-order arm emitted no checkpoints at
                      all, so a ten-rung ladder cost ten runs
``lr_warmup``         ignored; the schedule was unconditionally plain cosine
``first_order``       ignored; ``"ademamix"`` ran plain Adam under that label
===================== ==========================================================

-- and every one of them is invisible in a single run, because the run still converges
to something. These tests are the missing half of the parity check: they assert the
config reaches the algorithm.
"""

import pytest

torch = pytest.importorskip("torch")

from pinn_sfr_transient.axial import AxialParams
from pinn_sfr_transient.axial.pinn_torch import AxialPinn, AxialTrainConfig, train
from pinn_sfr_transient.axial.torchpinn.training import Trainer

P = AxialParams(n_axial=20)
BASE = {
    "seed": 0,
    "n_colloc": 64,
    "fourier_features": 16,
    "width": 8,
    "depth": 2,
    "rar_every": 0,
    "pts_every": 0,
    "log_every": 10**9,
    "adam_iters": 6,
    "lbfgs_iters": 0,
}


def _cfg(**kw: object) -> AxialTrainConfig:
    return AxialTrainConfig(**(BASE | kw))


@pytest.mark.parametrize(("field", "n"), [("adam_colloc", 7), ("polish_colloc", 11)])
def test_per_stage_collocation_counts_reach_the_draw(field: str, n: int) -> None:
    """Each stage draws its own count, not ``n_colloc``."""
    cfg = _cfg(**{field: n})
    tr = Trainer(AxialPinn(P, cfg), cfg)
    assert tr.collocation(n=getattr(cfg, field))[0].shape[0] == n
    assert tr.collocation()[0].shape[0] == cfg.n_colloc


def test_first_order_checkpoints_fire_on_cadence() -> None:
    """A pure first-order arm emits at every ``adam_checkpoint_every``, 1-indexed.

    Same cadence and same numbering as the JAX twin, which is what lets one run be
    scored at several budgets instead of one run per rung.
    """
    seen: list[int] = []
    cfg = _cfg(adam_iters=25, adam_checkpoint_every=10)
    train(P, cfg, on_checkpoint=lambda n, _: seen.append(n))
    assert seen == [10, 20]


def test_no_first_order_checkpoints_when_the_cadence_is_off() -> None:
    seen: list[int] = []
    train(P, _cfg(adam_iters=25), on_checkpoint=lambda n, _: seen.append(n))
    assert seen == []


@pytest.mark.parametrize("arm", ["ademamix", "schedulefree"])
def test_first_order_arms_run_and_are_not_adam(arm: str) -> None:
    """Each arm trains, and produces a different model from Adam at the same seed.

    `torch.optim` has neither algorithm, so these come from `pytorch_optimizer` --
    a maintained implementation, which is the point: hand-writing one is what AGENTS.md
    forbids, and taking a library is what the JAX twin does with `optax.contrib`.

    The check is that the arm CHANGES something. A `first_order` that silently fell back
    to Adam is exactly the defect this file exists for, and it is invisible in a single
    run because the run still converges.
    """
    pytest.importorskip("pytorch_optimizer")
    ref = train(P, _cfg(first_order="adam"))
    ref_w = [q.detach().clone() for q in ref.parameters()]
    got = train(P, _cfg(first_order=arm))
    assert all(torch.isfinite(q).all() for q in got.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(ref_w, got.parameters(), strict=True)), (
        f"first_order={arm!r} produced Adam's model exactly"
    )


def test_schedule_free_reports_the_averaged_iterate() -> None:
    """Schedule-free must hand back ``x``, not the ``y`` its gradients were taken at.

    The two sequences are not close early in a run, and returning ``y`` gives a
    plausible, worse number with nothing to indicate anything went wrong. The JAX twin
    converts through `schedule_free_eval_params` at the same points; here it is the
    optimiser's own `eval()`.
    """
    pytest.importorskip("pytorch_optimizer")
    cfg = _cfg(first_order="schedulefree")
    tr = Trainer(AxialPinn(P, cfg), cfg)
    opt = tr._first_order()
    for _ in range(20):
        opt.zero_grad()
        tr.causal_loss(*tr.collocation(1.0), 1.0).backward()
        opt.step()
    y = [q.detach().clone() for q in tr.model.parameters()]
    tr._to_reportable(opt)
    x = [q.detach().clone() for q in tr.model.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(y, x, strict=True)), "x and y identical"


def test_schedule_free_refuses_an_external_warmup() -> None:
    """Both schedule the step size; schedule-free warms up internally. Pick one."""
    pytest.importorskip("pytorch_optimizer")
    with pytest.raises(ValueError, match="both schedule the step size"):
        train(P, _cfg(first_order="schedulefree", lr_warmup=True))


def test_unknown_first_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown first_order"):
        train(P, _cfg(first_order="nonsense"))


def test_warmup_ramps_then_decays_and_plain_cosine_does_not() -> None:
    """``lr_warmup`` puts a linear ramp in front of the cosine; off, it is absent."""
    cfg = _cfg(adam_iters=100, lr=1e-3, lr_warmup=True)
    tr = Trainer(AxialPinn(P, cfg), cfg)
    opt = torch.optim.Adam(tr.model.parameters(), lr=cfg.lr)
    sched = tr._lr_schedule(opt)
    lrs = []
    for _ in range(cfg.adam_iters):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    warm = max(1, int(cfg.sf_warmup_frac * cfg.adam_iters))
    assert lrs[0] == pytest.approx(cfg.lr / warm)  # starts at the first rung, not the peak
    assert lrs[:warm] == sorted(lrs[:warm])  # ramps
    assert max(lrs) == pytest.approx(cfg.lr)  # reaches the peak, does not exceed it
    assert lrs[-1] < lrs[warm]  # then decays
    # ...to eta_min. The last SAMPLE is one step short of it, the schedule being read
    # before each step rather than after, so this is 1e-2 and not tighter.
    assert lrs[-1] == pytest.approx(cfg.lr * 0.1, rel=1e-2)

    plain = _cfg(adam_iters=100, lr=1e-3)
    tr2 = Trainer(AxialPinn(P, plain), plain)
    opt2 = torch.optim.Adam(tr2.model.parameters(), lr=plain.lr)
    assert isinstance(tr2._lr_schedule(opt2), torch.optim.lr_scheduler.CosineAnnealingLR)


def test_warmup_cosine_agrees_with_the_optax_schedule_after_the_ramp() -> None:
    """The torch composition is the same curve optax's twin produces.

    They cannot agree *during* the ramp -- optax starts at exactly zero and torch's
    ``start_factor`` must be positive -- so the check is that they agree after it, which
    is where the two backends would otherwise be running different schedules.
    """
    optax = pytest.importorskip("optax")
    cfg = _cfg(adam_iters=100, lr=1e-3, lr_warmup=True)
    tr = Trainer(AxialPinn(P, cfg), cfg)
    opt = torch.optim.Adam(tr.model.parameters(), lr=cfg.lr)
    sched = tr._lr_schedule(opt)
    lrs = []
    for _ in range(cfg.adam_iters):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    warm = max(1, int(cfg.sf_warmup_frac * cfg.adam_iters))
    ref = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warm, cfg.adam_iters, cfg.lr * 0.1)
    worst = max(abs(a - float(ref(i))) for i, a in enumerate(lrs) if i >= warm)
    assert worst < 1e-9


def test_polish_refresh_redraws_and_restarts() -> None:
    """``polish_refresh`` runs the polish in blocks, each on a fresh set.

    Checked by counting optimiser constructions: one per block, because a history that
    spanned two draws would span two objectives.
    """
    cfg = _cfg(adam_iters=2, lbfgs_iters=20, polish_refresh=5)
    tr = Trainer(AxialPinn(P, cfg), cfg)
    made = []
    inner = tr._make_opt
    tr._make_opt = lambda n: (made.append(n), inner(n))[1]
    tr.train(verbose=False)
    assert made == [5, 5, 5, 5]


def test_polish_refresh_and_freeze_after_are_mutually_exclusive() -> None:
    """Both schedule the same stage; the JAX twin raises here too."""
    cfg = _cfg(adam_iters=2, lbfgs_iters=20, polish_refresh=5, freeze_after=10)
    with pytest.raises(ValueError, match="same stage"):
        train(P, cfg)


def test_compile_is_requested_fullgraph_and_static(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `compile` is on, the loss is compiled with both settings that matter.

    Neither is tuning. ``fullgraph=True`` because a partial compile here is worth close
    to nothing and hides the reason -- the residual stack broke into eight graphs until
    two defects were fixed, and a silent fallback to eager would have shown that as a 3%
    regression rather than as an error. ``dynamic=False`` because automatic dynamic
    shapes switch on at the second distinct collocation count -- RAR produces one every
    ``rar_every`` -- and ``torch._make_dual`` then fails on a symbolic size, so
    forward-mode AD and dynamic shapes cannot be combined in 2.13.

    The equivalence of the compiled and eager answers is checked by
    ``uv run python tools/backend_smoke.py --compile``, which is a minute of
    compilation and does not belong in this suite.
    """
    calls: list[dict] = []

    def fake_compile(fn: object, **kw: object) -> object:
        calls.append(kw)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    train(P, _cfg(compile=True))
    assert calls == [{"fullgraph": True, "dynamic": False}]


def test_compile_off_by_default_does_not_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default must stay eager: the suite cannot afford 12-40 s per input shape."""
    calls: list[dict] = []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: (calls.append(kw), fn)[1])
    train(P, _cfg())
    assert calls == []
    assert AxialTrainConfig().compile is False
