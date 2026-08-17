"""Verify each model reproduces its published accuracy from the shipped defaults.

Defect D67: the axial model's published tables were measured at a `t_train_frac`
that was not the shipped default, and at the default the model produced no boiling
front at all. Nobody noticed for four milestones, because nothing ever ran the
documented command and compared the result to the documented claim.

That is a defect in practice, not in a model, so this checks both.

    uv run pinn-sfr reference                                  # the 0D held-out data
    OMP_NUM_THREADS=4 uv run python tools/check_published_accuracy.py

Nothing here is tuned: `TrainConfig()` and `SFRParams()` exactly as delivered. The
axial equivalent is `tools/axial_study.py budget`, whose arm A is the shipped
configuration.
"""

import time
from pathlib import Path

import numpy as np

from pinn_sfr_transient.config import SFRParams
from pinn_sfr_transient.pinn_torch import TrainConfig, predict, relative_l2, train

# `docs/neural_network.md` section 9: "a few 1e-3 relative L2 on each field".
CLAIM = 1e-2


def main() -> int:
    """Train the 0D PINN at its defaults and check it against the published claim."""
    ref = Path("results/ulof_reference.npz")
    if not ref.exists():
        print(f"missing {ref} -- run `uv run pinn-sfr reference` first")
        return 2

    cfg = TrainConfig()
    print(
        f"defaults: adam={cfg.adam_iters} lbfgs={cfg.lbfgs_iters} width={cfg.width} "
        f"depth={cfg.depth} causal_eps={cfg.causal_eps} seed={cfg.seed}",
        flush=True,
    )
    t0 = time.perf_counter()
    model = train(SFRParams(), cfg)
    dt = time.perf_counter() - t0

    err = relative_l2(predict(model), dict(np.load(ref)))
    print(f"\ntrained in {dt:.0f}s")
    for k, v in err.items():
        print(f"  {k:3s}: {v:.3e}")

    worst = max(err.values())
    ok = worst < CLAIM
    print(f"\nworst field: {worst:.3e}")
    print("published claim of 'a few 1e-3 on each field':", "HOLDS" if ok else "DOES NOT HOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
