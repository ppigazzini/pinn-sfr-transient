"""Array-module dispatch shared by the axial model's numeric code.

One transcription of every correlation and residual, evaluated by whichever
backend the caller passes. Carrying numpy, torch and JAX together is a
correctness mechanism rather than bookkeeping: ``docs/neural_network.md`` §9
records that the PyTorch initialisation bug was identified because the JAX twin
fit well at the same budget, and two backends cannot tell you which of them is
wrong.
"""

from typing import Any

import numpy as np


def xp(x: Any) -> Any:  # noqa: ANN401 - deliberately backend-agnostic
    """Return the array module (``numpy``, ``torch`` or ``jax.numpy``) matching ``x``.

    ``torch`` and ``jax`` are optional extras and are imported only when their
    array type is actually passed, so the axial package imports with neither
    installed. Both JAX flavours resolve here -- concrete arrays
    (``jaxlib._jax.ArrayImpl``) and tracers (``jax._src...``) -- because both
    module paths start with ``jax``.
    """
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch  # noqa: PLC0415 - optional extra, imported only when used

        return torch
    if mod.startswith("jax"):
        import jax.numpy as jnp  # noqa: PLC0415 - optional extra, imported only when used

        return jnp
    return np


def like(x: Any, values: Any) -> Any:  # noqa: ANN401 - deliberately backend-agnostic
    """Convert a numpy constant into ``x``'s backend, dtype and device.

    Model constants (``beta_i``, ``lambda_i``) live in :class:`AxialParams` as
    numpy arrays, but the residuals that use them may be evaluated on torch or
    JAX arrays. ``numpy_array * tensor`` raises, so the constant has to be moved
    rather than the field.
    """
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch  # noqa: PLC0415 - optional extra, imported only when used

        return torch.as_tensor(values, dtype=x.dtype, device=x.device)
    if mod.startswith("jax"):
        import jax.numpy as jnp  # noqa: PLC0415 - optional extra, imported only when used

        return jnp.asarray(values, dtype=x.dtype)
    return np.asarray(values)
