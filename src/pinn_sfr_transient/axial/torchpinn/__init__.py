"""PyTorch axial PINN, split into independent modules.

The split follows `jaxpi2 <https://github.com/sifanexisted/jaxpi2>`_ and mirrors
:mod:`pinn_sfr_transient.axial.jaxpinn`, so the two backends can be read against
each other module by module:

==================  =========================================================
:mod:`config`       training hyper-parameters; no dependencies
:mod:`archs`        network architectures and ansatz constants; no physics
:mod:`ansatz`       the closed forms the ansatz is built from
:mod:`model`        the ansatz and the residuals
:mod:`weighting`    causal ramp and block-weight bound
:mod:`training`     sampling and the training loop
:mod:`evaluate`     scoring; never imported by training
==================  =========================================================

**Two places do not mirror the JAX twin, and both are torch's idiom rather than a
design choice.** ``nn.Module`` owns its parameters and its forward pass, so the
ansatz and the residuals share :mod:`model`; and the sampler needs the model to
place points on the predicted front while the loop needs mutable optimiser state,
so both share ``Trainer`` in :mod:`training`.

``axial.pinn_torch`` re-exports this package, so existing imports and the
``python -m`` entry point are unaffected.
"""

from pinn_sfr_transient.axial.torchpinn.ansatz import (
    _fuel_temperature,
    _power_integral,
    _power_shape,
    _precursors,
)
from pinn_sfr_transient.axial.torchpinn.archs import (
    _EXP_BOUND,
    FIELDS,
    MLP,
    N_TEMPS,
    FourierEmbedding,
    _bounded_exp,
)
from pinn_sfr_transient.axial.torchpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.torchpinn.evaluate import relative_l2
from pinn_sfr_transient.axial.torchpinn.model import AxialPinn
from pinn_sfr_transient.axial.torchpinn.training import Trainer, train

__all__ = [
    "FIELDS",
    "MLP",
    "N_TEMPS",
    "_EXP_BOUND",
    "AxialPinn",
    "AxialTrainConfig",
    "FourierEmbedding",
    "Trainer",
    "_bounded_exp",
    "_fuel_temperature",
    "_power_integral",
    "_power_shape",
    "_precursors",
    "relative_l2",
    "train",
]
