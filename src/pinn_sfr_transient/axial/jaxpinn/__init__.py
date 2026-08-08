"""JAX/Equinox axial PINN, split into independent modules.

The split follows `jaxpi2 <https://github.com/sifanexisted/jaxpi2>`_, whose
structural point is that the architecture, the physics, the sampler and the
evaluator should not know about each other:

======================  ===================================================
:mod:`config`           training hyper-parameters; no dependencies
:mod:`archs`            network architectures; no physics
:mod:`ansatz`           the hard constraints and the analytic steady profile
:mod:`residuals`        the equations, from the shared ``axial.physics``
:mod:`weighting`        how blocks are combined into one scalar
:mod:`samplers`         where the residual is evaluated
:mod:`training`         orchestration only
:mod:`evaluate`         scoring; never imported by training
======================  ===================================================

The dependency graph is a DAG in that order. What this buys is that an ablation —
a different architecture, a different weighting, a different sampler — is a
config change rather than a rewrite of one long module.

``axial.pinn_jax`` re-exports this package's public surface, so existing imports
and the ``python -m`` entry point are unaffected.
"""

from __future__ import annotations

import jax

# float64 BEFORE any array is created, and before any submodule is imported.
# This is a correctness setting, not a performance one: at the default float32
# the section 12.13 sodium correlations lose about eight significant digits and
# the saturation round-trip degrades from 1e-11 K to ~1e-2 K.
#
# It lives here rather than in a submodule because importing any submodule
# imports this package first, so this is the one place that cannot be bypassed.
# Splitting this module into a package once dropped this line, and the whole
# backend silently ran in float32 -- caught by a regression number, not by a
# test, which is why `test_float64_is_enabled` now exists.
jax.config.update("jax_enable_x64", True)

from pinn_sfr_transient.axial.jaxpinn.ansatz import (
    _power_integral,
    _power_shape,
    front_position,
    horizon,
    normalised_state,
    precursors,
    state_and_grads,
    theta0,
)
from pinn_sfr_transient.axial.jaxpinn.archs import (
    _ALPHA_GATE,
    _EXP_BOUND,
    FIELDS,
    N_TEMPS,
    AxialPinn,
    FourierEmbedding,
    ModifiedMLP,
)
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.jaxpinn.evaluate import (
    predict,
    predict_power,
    predict_reactivity_components,
    relative_l2,
)
from pinn_sfr_transient.axial.jaxpinn.residuals import (
    closed_loop_blocks,
    front_residual,
    n_field_blocks,
    onset_point,
    onset_residual,
    residual_blocks,
    uses_front,
)
from pinn_sfr_transient.axial.jaxpinn.samplers import _collocation, _merge, _rar_points
from pinn_sfr_transient.axial.jaxpinn.training import train
from pinn_sfr_transient.axial.jaxpinn.weighting import (
    bounded_weights,
    causal_loss,
    causal_weights,
    pts_penalty,
)

__all__ = [
    "FIELDS",
    "N_TEMPS",
    "_ALPHA_GATE",
    "_EXP_BOUND",
    "AxialPinn",
    "AxialTrainConfig",
    "FourierEmbedding",
    "ModifiedMLP",
    "_collocation",
    "_merge",
    "_power_integral",
    "_power_shape",
    "_rar_points",
    "bounded_weights",
    "causal_loss",
    "causal_weights",
    "closed_loop_blocks",
    "front_position",
    "front_residual",
    "horizon",
    "n_field_blocks",
    "normalised_state",
    "onset_point",
    "onset_residual",
    "precursors",
    "predict",
    "predict_power",
    "predict_reactivity_components",
    "pts_penalty",
    "relative_l2",
    "residual_blocks",
    "state_and_grads",
    "theta0",
    "train",
    "uses_front",
]
