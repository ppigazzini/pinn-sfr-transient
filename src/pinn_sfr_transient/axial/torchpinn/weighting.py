"""Loss weighting: the causal ramp and the block-weight bound.

Pure functions of the per-chunk losses and the gradient norms, so a weighting
scheme can be swapped or removed without touching the physics.
"""

try:
    import torch  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc


def _causal_weights(losses: torch.Tensor, eps: float) -> torch.Tensor:
    """Causal temporal weights ``exp(-eps * prefix / total)`` [Wang, Sankaran & Perdikaris 2024].

    A chunk is down-weighted by how much residual is still outstanding *before*
    it, so the network is not rewarded for fitting late times until the early
    ones are right.

    **The prefix sum is normalised by the total, which the original formulation
    does not do, and here it is not optional.** ``exp(-eps * prefix)`` treats
    ``eps`` as carrying the reciprocal units of the loss, so the ramp is only
    meaningful while the loss stays near the scale ``eps`` was tuned at. Variable
    scaling (``residual_scaling``) moved that scale by about ten orders of
    magnitude, and the weighting silently went inert: measured, the ramp spanned
    1.000 to 0.977 across all 32 chunks — a 2% tilt where it had been a hard
    cutoff. The network was then free to fit late times first, which is exactly
    what it did.

    Normalised, ``eps`` is dimensionless and directly interpretable: the last
    chunk starts at about ``exp(-eps)`` relative to the first, whatever the loss
    magnitude, and the ramp relaxes on its own as the early residuals fall.
    """
    prefix = torch.cumsum(losses, 0) - losses
    return torch.exp(-eps * prefix / (losses.sum() + 1e-30))


def _bounded_weights(target: torch.Tensor, cap: float) -> torch.Tensor:
    """Renormalise block weights to unit geometric mean, then clamp to ``[1/cap, cap]``.

    ``cap = 1`` collapses every weight to one, i.e. no weighting at all.
    """
    if cap <= 1.0:
        return torch.ones_like(target)
    return (target / torch.exp(torch.log(target).mean())).clamp(1.0 / cap, cap)
