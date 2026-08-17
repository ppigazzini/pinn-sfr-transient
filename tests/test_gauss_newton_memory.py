"""The chunked dual Jacobian: same matrix as `jacrev`, bounded memory unlike it.

`tools/gauss_newton_experiment.py` built its dual Gramian with a single
`jax.jacrev` over 3000 residual rows. Reverse mode holds one copy of the forward
tape per cotangent in flight, so that asked for ~113 GB (measured, and linear in
the row count) and took a 64 GB host down. The fix builds the same matrix in row
blocks of `jac_chunk`.

Two things therefore have to hold, and only the first is about arithmetic:

* the blocked build returns what the unblocked one did to machine precision,
  including when the row count is not a multiple of the chunk — the tail is
  padded with a repeated index and trimmed, which is the step most likely to be
  off by one;
* peak memory scales with the **chunk**, not with the number of rows. That is the
  whole point of the change, and it is a property of the compiled program rather
  than of the numbers it produces, so no accuracy test can catch its regression.

Deliberately tiny: a 3-parameter model and a handful of rows. The defect was
never about scale, it was about which axis the allocation grew along.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# `tools/` is scripts, not an installed package, so it is not on the path the way
# `pinn_sfr_transient` is. Done here rather than in a shared conftest because this is the
# only test that reaches into it, and a repo-wide path hack to serve one file is worse.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.gauss_newton_experiment import (
    compiled_bytes,
    make_dual_jacobian,
)


def _toy_residuals(n_rows: int = 37):
    """A nonlinear residual map with a genuine Jacobian, small enough to be instant."""
    a = jnp.asarray(np.linspace(0.3, 1.7, n_rows))[:, None]
    b = jnp.asarray(np.linspace(-1.0, 1.0, n_rows))[:, None]

    def residuals(x: jax.Array) -> jax.Array:
        return jnp.sin(a * x[0] + b * x[1]) + jnp.tanh(x[2] * a).ravel()[:, None] @ jnp.ones((1, 1))

    return lambda x: residuals(x).ravel(), n_rows


@pytest.mark.parametrize("chunk", [1, 4, 8, 64])
def test_blocked_jacobian_equals_jacrev(chunk):
    """The blocked build must reproduce `jax.jacrev` to machine precision.

    37 rows is deliberately coprime with every chunk here, so the padded-and-trimmed
    tail is exercised in each case rather than only when it happens to divide.
    """
    residuals, n_rows = _toy_residuals()
    x = jnp.asarray([0.4, -0.9, 1.3])
    idx = jnp.arange(n_rows)

    _, dual_gram = make_dual_jacobian(residuals, chunk)
    j_blocked, gram = dual_gram(x, idx)
    j_ref = jax.jacrev(lambda z: residuals(z)[idx])(x)

    # To machine precision, not bit-exactly: the blocked build reduces in a different
    # order from `jacrev`, which is worth ~1 ulp and is not a defect.
    np.testing.assert_allclose(np.asarray(j_blocked), np.asarray(j_ref), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(np.asarray(gram), np.asarray(j_ref @ j_ref.T), rtol=1e-12, atol=0)


def test_a_subsample_takes_the_rows_it_was_given():
    """A strided subsample must select rows, not silently re-index them."""
    residuals, n_rows = _toy_residuals()
    x = jnp.asarray([0.4, -0.9, 1.3])
    idx = jnp.arange(0, n_rows, 5)

    _, dual_gram = make_dual_jacobian(residuals, 4)
    j_blocked, _ = dual_gram(x, idx)
    j_full = jax.jacrev(residuals)(x)

    np.testing.assert_allclose(
        np.asarray(j_blocked), np.asarray(j_full)[idx, :], rtol=1e-14, atol=1e-16
    )


def test_peak_memory_scales_with_the_chunk_and_not_with_the_row_count():
    """The property the fix exists for, asserted on the compiled program.

    Doubling the chunk must roughly double the reverse sweep's buffers; doubling the
    number of *rows* at a fixed chunk must not. The original code had these the other
    way round — memory grew with the rows, because every row was differentiated at once
    — and no test of the numbers could have seen it.

    Tolerances are loose because a fixed overhead sits under the linear term; the
    assertion is about which quantity the growth tracks, not its constant.
    """
    residuals, _ = _toy_residuals(n_rows=512)
    x = jnp.asarray([0.4, -0.9, 1.3])

    def peak(n_rows: int, chunk: int) -> float:
        jac_rows, _ = make_dual_jacobian(residuals, chunk)
        return compiled_bytes(jac_rows, x, jnp.arange(n_rows), jnp.arange(chunk))

    small, large = peak(512, 8), peak(512, 64)
    if not np.isfinite(small) or not np.isfinite(large):
        pytest.skip("this backend reports no memory analysis")

    # 8x the chunk must cost several times the memory: the sweep is the dominant term.
    assert large > 3.0 * small, f"chunk 8 -> 64 grew only {large / small:.2f}x"

    # ...while 4x the rows at a fixed chunk must not, which is the regression that
    # matters. `idx` itself grows, so this is not asserted to be exactly flat.
    few, many = peak(128, 8), peak(512, 8)
    assert many < 2.0 * few, f"rows 128 -> 512 grew {many / few:.2f}x at a fixed chunk"
