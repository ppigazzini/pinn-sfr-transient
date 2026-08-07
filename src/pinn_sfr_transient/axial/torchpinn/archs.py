"""Network architectures and the ansatz's shared constants.

Function approximators only: no physics, no residuals, no sampling. The
separation follows jaxpi2 and mirrors the JAX twin's ``archs`` module.
"""

from __future__ import annotations

try:
    import torch  # ty: ignore
    from torch import nn  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "PyTorch >= 2.13 is required: `uv sync --extra torch-cpu` (or `--extra torch-gpu`)"
    raise SystemExit(msg) from exc


import numpy as np

FIELDS: tuple[str, ...] = ("T_f", "T_cl", "T_s", "T_c", "alpha")
N_TEMPS: int = 4


_FRONT_MAX: float = 1.25
"""Upper bound on the predicted front height; above 1.0 means "no front in the channel"."""

_ALPHA_GATE: float = 10.0
"""Sharpness of the void gate; large enough to saturate away from the boundaries."""

"""**On the void head's initialisation — a real asymmetry, and a cure that failed.**

The output layer is initialised ``U(+/-1/sqrt(width))``, so ``sigmoid(raw)`` sits
at ~0.5 at every interior point on iteration zero, while the reference void field
is identically zero over ~96% of the channel and over all of ``t < 10.8 s``. The
void is not an inert output either — it degrades ``film_coefficient``, shifts
``alpha_D`` between its flooded and voided values, and enters Eq. 4.5-25. The
temperatures, by contrast, start *exactly* on the steady profile because their
ansatz is multiplicative with ``exp(0) = 1``. The asymmetry is genuine.

The obvious cure — ``sigmoid(raw - 4)``, so the head starts at 0.018 — was
implemented and then **measured, at three seeds against an ``n_axial = 160``
reference, and it does not earn its place**: mean relative ``L2`` went 0.226 to
0.227 on ``T_cl``, 0.068 to 0.103 on ``T_s`` and 0.114 to 0.124 on ``T_c``. It
did cut the ``T_f`` seed spread from 12.5x to 1.8x, which is interesting and not
sufficient. Reverted, and recorded in ``docs/axial_nn.md`` section 7.1 with the
table, because the measurement is worth more than the change would have been:
seed variance dominates this model, and the void's real blocker is the block
weighting, not the initialisation.
"""

_EXP_BOUND: float = 4.0
"""Bound on the exponent of the multiplicative ansatz.

``exp(S tanh(x/S))`` equals ``exp(x)`` for small ``x`` but saturates at
``exp(+/-S)``, so the ansatz cannot overflow however hard the optimiser pushes.
``S = 4`` allows a factor ``e^4 ~ 55`` on the steady excess temperature; the
reference needs at most 9.6, so the bound never binds and ``tanh`` stays in its
linear region where the gradient is healthy.
"""


def _bounded_exp(x: torch.Tensor) -> torch.Tensor:
    """``exp`` with a smooth ceiling and floor — see :data:`_EXP_BOUND`."""
    return torch.exp(_EXP_BOUND * torch.tanh(x / _EXP_BOUND))


class FourierEmbedding(nn.Module):
    """Random Fourier features, ``x -> [sin(2 pi B x), cos(2 pi B x)]``.

    A plain MLP is spectrally biased toward smooth functions, which is the wrong
    prior for a boiling front that moves through the domain. Lifting the inputs
    into a random Fourier basis restores the high frequencies
    [Tancik et al. 2020; Wang, Wang & Perdikaris 2021].
    """

    def __init__(self, n_in: int, n_features: int, scale: float) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(n_in, n_features, dtype=torch.float64) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ModifiedMLP(nn.Module):
    """Two-encoder MLP of [Wang, Teng & Perdikaris 2021], as used by jaxpi.

    Two encoders ``U`` and ``V`` are computed once from the input and mixed into
    every hidden layer, ``h <- (1 - z) * U + z * V``. The inputs therefore reach
    the last layer undiminished, which a plain feed-forward stack cannot manage
    at depth.
    """

    def __init__(self, n_in: int, n_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.u = nn.Linear(n_in, width)
        self.v = nn.Linear(n_in, width)
        self.first = nn.Linear(n_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, n_out)
        for m in [self.u, self.v, self.first, self.out, *self.hidden]:
            k = m.in_features**-0.5
            nn.init.uniform_(m.weight, -k, k)
            nn.init.uniform_(m.bias, -k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.tanh(self.u(x)), torch.tanh(self.v(x))
        h = torch.tanh(self.first(x))
        for layer in self.hidden:
            z = torch.tanh(layer(h))
            h = (1.0 - z) * u + z * v
        return self.out(h)


class MLP(nn.Module):
    """Plain tanh MLP, ``n_in -> n_out``.

    Weights *and* biases from ``U(-k, k)`` with ``k = 1/sqrt(fan_in)``, matching
    Equinox. That is not cosmetic: ``docs/neural_network.md`` §9 records that the
    0D model's earlier ``xavier_normal`` + zero-bias init left the gradient-norm
    scheme unable to lift the stiff block's weight, and the fit was an order of
    magnitude worse. Same recipe, same init.
    """

    def __init__(self, n_out: int = 4, width: int = 64, depth: int = 5, n_in: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                k = m.in_features**-0.5
                nn.init.uniform_(m.weight, -k, k)
                nn.init.uniform_(m.bias, -k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
