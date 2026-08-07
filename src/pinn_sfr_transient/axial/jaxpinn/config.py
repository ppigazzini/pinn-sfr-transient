"""Training hyper-parameters for the JAX axial PINN.

Separated from the networks, the residuals and the samplers so a knob can be
read without importing a model, and so the dependency graph stays a DAG.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AxialTrainConfig:
    """Hyper-parameters for the JAX axial PINN. Mirrors the torch config."""

    width: int = 64
    depth: int = 5
    n_colloc: int = 4000
    # Budget knobs match the torch twin exactly. They did not — 3000/300/1000
    # against 8000/500/2000 — so the default-configuration comparison the parity
    # study rests on was never a like-for-like one. Anything measured across
    # backends must be at an identical budget or it is measuring the schedule.
    adam_iters: int = 8000
    lbfgs_iters: int = 500
    lr: float = 1e-3

    # Dimensionless: the prefix sum is normalised by the total. See the torch twin.
    causal_eps: float = 0.0
    causal_chunks: int = 32
    # Fraction of `p.t_end` trained over — a scope decision; see the torch twin.
    t_train_frac: float = 1.0
    weight_update_every: int = 250
    weight_momentum: float = 0.9
    # Bound on the block-weight spread; off by default, see the torch twin.
    weight_max_ratio: float = 1.0
    # Variable scaling per residual block; see the torch twin.
    residual_scaling: bool = True
    # Eliminate the void algebraically (D-TH-3); see the torch twin.
    void_closure: bool = True
    # Front-position network (M8 option 2). Measured worse on every metric; see
    # the torch twin for the table.
    front_net: bool = False
    front_frac: float = 0.25  # share of collocation drawn near the predicted front

    # --- remedies for the moving front; every one measured in docs/axial_nn.md --
    # Time-window curriculum: train [0, w t_end] for growing w. Neutral in the
    # section 7.2.5 re-ablation.
    n_windows: int = 1
    # Random Fourier features against spectral bias [Tancik et al. 2020]. Measured
    # -11.1% at three seeds (section 7.2.6) but NOT adopted: it does not compose
    # with `modified_mlp`. 0 disables.
    fourier_features: int = 0
    fourier_scale: float = 2.0
    # Two-encoder "modified MLP" [Wang, Teng & Perdikaris 2021]. Measured -16.1%,
    # and likewise not adopted -- see section 7.2.6.
    modified_mlp: bool = False
    # Pseudo-time stepping [Wang, Koohy, Lu & Perdikaris, arXiv:2604.23528].
    # Measured harmful: under it the boiling front does not form at all. 0 disables.
    pts_every: int = 0
    pts_dtau: float = 1.0
    pts_growth: float = 1.5

    # RAR keeps a FIXED count so `jit` never recompiles (the torch twin grows an
    # unbounded reservoir instead — same idea, framework-appropriate form).
    rar_every: int = 2000
    rar_pool: int = 20000
    rar_keep: int = 400

    feedback: bool = False
    n_time: int = 128
    seed: int = 0
    log_every: int = 1000
