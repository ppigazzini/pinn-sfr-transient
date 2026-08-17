"""The beignet pyramid: spectral interpolation, and the same object in both backends.

`BeignetPyramid` implements the trainable multi-resolution Fourier feature pyramid of
arXiv:2605.24278, whose claim is that this *architecture* lets Adam reach accuracy
previously needing higher-order optimisers. Testing that claim here is only meaningful
if the architecture is the paper's, so these assert the properties the maths guarantees
rather than numbers a run happened to produce:

* the bandlimited interpolant reproduces its own grid **exactly** at the grid nodes,
  which is the defining property of Eq. 6 and the one a plausible-but-wrong
  transcription would miss;
* it genuinely interpolates between nodes rather than stepping;
* the grids are **trainable** -- that is the entire mechanism, and it is the one thing
  that distinguishes this from `FourierEmbedding`, whose `B` is held under
  `stop_gradient`;
* both backends produce the same features from the same grid, to float64 precision.
"""

import numpy as np
import pytest

N_NODES = 8
N_FEATURES = 3


def _grid_and_nodes():
    """A grid, and the coordinates of its own nodes under ``pad = 0``."""
    return np.arange(N_NODES) / N_NODES


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_interpolant_reproduces_its_grid_at_the_nodes(backend):
    """``g(j/N)`` must equal ``theta[j]``, which is what makes it an interpolant.

    A transcription that got the DFT normalisation or the frequency ordering wrong still
    produces smooth, plausible features and would train perfectly happily -- so this is
    asserted against the definition rather than against a converged loss.
    """
    pytest.importorskip(backend)
    u = _grid_and_nodes()

    if backend == "torch":
        import torch

        from pinn_sfr_transient.axial.torchpinn.archs import BeignetPyramid

        pyr = BeignetPyramid(1, N_FEATURES, N_NODES, 1.0, 0.0)
        theta = pyr.grids[0].detach().numpy()
        x = torch.tensor(np.stack([u, np.zeros(N_NODES)], axis=1))
        got = pyr(x).detach().numpy()[:, 1:]
    else:
        import jax
        import jax.numpy as jnp

        from pinn_sfr_transient.axial.jaxpinn.archs import BeignetPyramid

        pyr = BeignetPyramid(1, N_FEATURES, N_NODES, 1.0, 0.0, jax.random.PRNGKey(0))
        theta = np.asarray(pyr.grids[0])
        got = np.stack([np.asarray(pyr(jnp.asarray([q, 0.0])))[1:] for q in u])

    np.testing.assert_allclose(got, theta, rtol=0, atol=1e-12)


def test_it_interpolates_between_nodes_rather_than_stepping():
    """Halfway between nodes the value must differ from both neighbours.

    Guards the degenerate implementation that rounds to the nearest grid cell, which
    would pass the node test above and be a lookup table rather than a bandlimited field.
    """
    torch = pytest.importorskip("torch")

    from pinn_sfr_transient.axial.torchpinn.archs import BeignetPyramid

    pyr = BeignetPyramid(1, N_FEATURES, N_NODES, 1.0, 0.0)
    theta = pyr.grids[0].detach().numpy()
    mid = (np.arange(N_NODES) + 0.5) / N_NODES
    got = pyr(torch.tensor(np.stack([mid, np.zeros(N_NODES)], axis=1))).detach().numpy()[:, 1:]
    assert np.abs(got - theta).min() > 1e-6


def test_the_grids_are_trainable_which_is_the_whole_mechanism():
    """The paper's contribution is a *learned* pyramid, so gradients must reach the grid.

    `FourierEmbedding.B` is deliberately frozen in both backends. If these grids were
    frozen the same way, the arm would measure a fixed multi-band basis -- which this
    project has already measured -- and would say nothing about arXiv:2605.24278.
    """
    torch = pytest.importorskip("torch")

    from pinn_sfr_transient.axial.torchpinn.archs import BeignetPyramid

    pyr = BeignetPyramid(2, N_FEATURES, 4, 1.0, 0.25)
    x = torch.tensor(np.stack([np.linspace(0.0, 1.0, 5), np.zeros(5)], axis=1))
    pyr(x).sum().backward()
    for g in pyr.grids:
        assert g.grad is not None, "no gradient reached a pyramid grid"
        assert float(g.grad.abs().max()) > 0.0, "gradient reached the grid but is zero"


def test_both_backends_give_the_same_features_from_the_same_grid():
    """Parity of the object itself, before any parity of results is claimed."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from pinn_sfr_transient.axial.jaxpinn.archs import BeignetPyramid as JaxPyramid
    from pinn_sfr_transient.axial.torchpinn.archs import BeignetPyramid as TorchPyramid

    tp = TorchPyramid(1, N_FEATURES, N_NODES, 1.0, 0.0)
    theta = tp.grids[0].detach().numpy()
    jp = JaxPyramid(1, N_FEATURES, N_NODES, 1.0, 0.0, jax.random.PRNGKey(0))
    jp = jax.tree_util.tree_map(lambda a: jnp.asarray(theta) if a.shape == theta.shape else a, jp)

    u = _grid_and_nodes()
    got_t = tp(torch.tensor(np.stack([u, np.zeros(N_NODES)], axis=1))).detach().numpy()
    got_j = np.stack([np.asarray(jp(jnp.asarray([q, 0.0]))) for q in u])
    np.testing.assert_allclose(got_t, got_j, rtol=0, atol=1e-12)
