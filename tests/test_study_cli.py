"""Every CLI override must reach the config the model is built from.

An override that is parsed, stored in a module global and then never read is invisible:
the run completes, the header records the default, and the arm silently measured
something other than what was asked for.

That happened. `--rar-every 0` was dropped because a `str.replace` anchor stopped
matching after `ruff --fix` rewrote the line it anchored on, so three 200k arms ran with
`rar_every = 2000` -- exactly the setting they existed to ablate -- and destabilised on
schedule, looking like evidence when they were a bug.

These capture the config `study_ademamix` actually builds rather than trusting the
plumbing, which is the same check `tools/backend_smoke.py` applies to the backends.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _study():
    """Import tools/axial_study.py as a module."""
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location("axial_study", ROOT / "tools" / "axial_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["axial_study"] = mod
    spec.loader.exec_module(mod)
    return mod


def _capture_cfg(mod, monkeypatch, **globals_):
    """Run the arm with training stubbed out, and return the config it built."""
    seen = {}

    def fake_train(p, cfg, **_kw):
        seen["cfg"] = cfg
        return None, p, cfg

    def fake_saver(*_a, **_k):
        return lambda *_args: Path("x")

    for k, v in globals_.items():
        monkeypatch.setattr(mod, k, v, raising=False)
    monkeypatch.setattr(mod, "SEEDS", (0,))
    monkeypatch.setattr(mod, "write", lambda *_a, **_k: None)

    # `train` and `checkpoint` are imported INSIDE `study_ademamix`, so they are not
    # attributes of the study module; patch them where they actually live.
    import pinn_sfr_transient.axial.jaxpinn as jp
    from pinn_sfr_transient.axial import checkpoint

    monkeypatch.setattr(jp, "train", fake_train)
    monkeypatch.setattr(checkpoint, "saver", fake_saver)
    mod.study_ademamix(Path("/dev/null"))
    return seen["cfg"]


def test_lr_override_reaches_the_config(monkeypatch):
    assert _capture_cfg(_study(), monkeypatch, _LR=7e-6, _ADAM=8).lr == 7e-6


def test_warmup_override_reaches_the_config(monkeypatch):
    cfg = _capture_cfg(_study(), monkeypatch, _WARMUP_FRAC=0.42, _ADAM=8)
    assert cfg.sf_warmup_frac == 0.42


def test_rar_every_zero_reaches_the_config(monkeypatch):
    """0 must survive. `x if x else default` would silently discard it, and did."""
    assert _capture_cfg(_study(), monkeypatch, _RAR=0, _ADAM=8).rar_every == 0


def test_defaults_are_the_published_arm(monkeypatch):
    cfg = _capture_cfg(_study(), monkeypatch, _ADAM=8)
    assert cfg.lr == 1e-4
    assert cfg.sf_warmup_frac == 0.1
    assert cfg.rar_every == 2000
    assert cfg.first_order == "ademamix"
    assert cfg.fourier_features == 256
    assert cfg.adam_colloc == 500
    assert cfg.n_colloc == 10000
    assert cfg.lr_warmup is True
    assert cfg.lbfgs_iters == 0
