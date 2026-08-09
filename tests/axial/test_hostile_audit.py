"""Hostile audit: check this project's algebra with tooling that is not this project.

Every other test file checks the code against itself — a torch path against a JAX
path, a closure against its own docstring, a config against a config. That catches
divergence between two transcriptions and is blind to a shared mistake, which is
exactly the failure mode this project keeps hitting: an inverted self-scaling that
both smoke tests accepted, a Broyden update that passed every training run and was
10% wrong, an `optax.lbfgs` default that agreed with itself for four milestones.

So the rules here are deliberately different:

* **Never verify a routine with the routine's own machinery.** The quasi-Newton
  operators are checked against dense matrices built from the textbook formulas;
  autodiff is checked against finite differences; a closure is checked against a
  `sympy` transcription written from the docstring rather than from the code.
* **Prefer an outside implementation where one exists** — `scipy.optimize` for the
  optimiser, `scipy.integrate` for the quadrature.
* **Assert the property the maths guarantees**, not the number the code produced.

These are slower than the rest of the suite and they are meant to be. They are the
only tests here that could catch an error both backends make.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, optimize

from pinn_sfr_transient.axial import AxialParams

# ---------------------------------------------------------------------------
# 1. the quasi-Newton operators, against dense matrices and against scipy
# ---------------------------------------------------------------------------


def _dense_broyden(pairs: list[tuple], gamma: float, phi: float) -> np.ndarray:
    """Build ``H`` explicitly from the textbook Broyden-class update.

    Deliberately the slow, obvious form — full ``n x n`` matrices, one update at a
    time, no recursion and no limited-memory trick:

        H <- tau H
        H <- H - (H y s' + s y' H)/(s'y) + (1 + y'Hy/(s'y)) s s'/(s'y)   [BFGS]
        H <- H + phi (y'Hy) v v',  v = s/(s'y) - Hy/(y'Hy)               [family]

    Nothing here is shared with the implementations it audits: they apply an
    operator to a vector without ever forming it, by a two-loop recursion (BFGS)
    or a sequential replay (Broyden). If both agree with this, the recursions are
    right; if they agree only with each other, they are consistent and possibly
    both wrong.
    """
    n = len(pairs[0][0])
    h = gamma * np.eye(n)
    for s_raw, y_raw, rho, tau in pairs:
        s = np.asarray(s_raw, dtype=np.float64).reshape(-1)
        y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
        if tau != 1.0:
            h = h * tau
        hy = h @ y
        yhy = float(y @ hy)
        h = h - (np.outer(hy, s) + np.outer(s, hy)) * rho + (1.0 + yhy * rho) * rho * np.outer(s, s)
        if phi != 0.0 and yhy > 0.0:
            v = s * rho - hy / yhy
            h = h + phi * yhy * np.outer(v, v)
    return h


def _random_pairs(n: int = 24, m: int = 5, tau: float = 0.8) -> list[tuple]:
    """Curvature pairs with ``s'y > 0``, which every quasi-Newton update assumes."""
    rng = np.random.default_rng(0)
    out = []
    for _ in range(m):
        s = rng.normal(size=n)
        y = rng.normal(size=n) + 3.0 * s  # force positive curvature
        out.append((s, y, 1.0 / float(s @ y), tau))
    return out


@pytest.mark.parametrize("phi", [0.0, 0.35, 1.0])
def test_torch_quasi_newton_operator_matches_a_dense_reference(phi):
    """The recursions must reproduce an operator built the obvious way."""
    torch = pytest.importorskip("torch")

    from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS

    pairs = _random_pairs()
    gamma = 1.7
    rng = np.random.default_rng(1)
    v = rng.normal(size=len(pairs[0][0]))
    want = _dense_broyden(pairs, gamma, phi) @ v

    opt = SelfScaledLBFGS([torch.zeros(1, requires_grad=True)])
    tp = [(torch.tensor(s), torch.tensor(y), rho, tau) for s, y, rho, tau in pairs]
    tv = torch.tensor(v)
    # `phi = 0` must also match through the two-loop, which is a different algorithm
    # again -- so at that point three independent implementations have to agree.
    opt.broyden_phi = phi
    got = opt._apply_broyden(tv, tp, gamma) if phi else opt._apply_H(tv, tp, gamma)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.35, 1.0])
def test_jax_quasi_newton_operator_matches_a_dense_reference(phi):
    """The JAX twin, against the same dense reference rather than against torch."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from pinn_sfr_transient.axial.jaxpinn.optimizers import _apply_broyden, _apply_H

    pairs = _random_pairs()
    gamma = 1.7
    rng = np.random.default_rng(1)
    v = rng.normal(size=len(pairs[0][0]))
    want = _dense_broyden(pairs, gamma, phi) @ v

    jp = [(jnp.asarray(s), jnp.asarray(y), rho, tau) for s, y, rho, tau in pairs]
    jv = jnp.asarray(v)
    got = _apply_broyden(jv, jp, gamma, phi) if phi else _apply_H(jv, jp, gamma)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10, atol=1e-12)


def test_self_scaling_shrinks_the_operator_in_the_direction_it_claims():
    """``tau < 1`` must SHRINK ``H``, which is the sign this file once had backwards.

    An inverted self-scaling (`H/tau` where `tau H` was meant) converged anyway and
    was invisible for a while, so the direction is asserted against the dense
    reference rather than inferred from convergence.
    """
    pairs_small = _random_pairs(tau=0.5)
    pairs_unit = _random_pairs(tau=1.0)
    rng = np.random.default_rng(2)
    v = rng.normal(size=24)
    shrunk = _dense_broyden(pairs_small, 1.0, 0.0) @ v
    plain = _dense_broyden(pairs_unit, 1.0, 0.0) @ v
    assert np.linalg.norm(shrunk) < np.linalg.norm(plain)


@pytest.mark.parametrize("phi", [0.0, 0.5])
def test_optimiser_finds_the_same_minimiser_as_scipy(phi):
    """An outside optimiser, on the same function, must reach the same point.

    `scipy.optimize.minimize` shares no code with this project. Comparing the
    *minimiser* rather than the loss is deliberate: a broken operator can still
    drive a loss down (the inverted self-scaling did) while converging to a
    different place or far more slowly.
    """
    torch = pytest.importorskip("torch")

    from pinn_sfr_transient.axial.torchpinn.optimizers import SelfScaledLBFGS

    def f_np(z):
        return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2))

    x0 = np.full(8, -1.2)
    ref = optimize.minimize(f_np, x0, method="BFGS", tol=1e-14)

    x = torch.tensor(x0, requires_grad=True)
    opt = SelfScaledLBFGS(
        [x],
        max_iter=400,
        history_size=20,
        broyden_phi=phi,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
    )

    def closure():
        x.grad = None
        loss = (100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    np.testing.assert_allclose(x.detach().numpy(), ref.x, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(x.detach().numpy(), np.ones(8), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. autodiff, against finite differences
# ---------------------------------------------------------------------------


def test_ansatz_derivatives_match_central_differences():
    """``state_and_grads`` is a ``jvp``+``vmap`` composition; check it numerically.

    The residual is built entirely from these derivatives, so an error here is an
    error in every number this project has published — and it would be invisible to
    the cross-backend tests if both backends compose the transforms the same wrong
    way. Central differences share nothing with autodiff.
    """
    torch = pytest.importorskip("torch")

    from pinn_sfr_transient.axial import pinn_torch as pt

    p = AxialParams()
    m = pt.AxialPinn(p, pt.AxialTrainConfig(width=16, depth=2, fourier_features=8, seed=0))
    rng = np.random.default_rng(0)
    zeta = torch.tensor(rng.uniform(0.1, 0.9, size=(6, 1)))
    that = torch.tensor(rng.uniform(0.1, 0.9, size=(6, 1)))
    _, d_dt, d_dz = m.state_and_grads(zeta, that)

    h = 1e-6
    with torch.no_grad():
        dt_fd = (m.normalised_state(zeta, that + h) - m.normalised_state(zeta, that - h)) / (2 * h)
        dz_fd = (m.normalised_state(zeta + h, that) - m.normalised_state(zeta - h, that)) / (2 * h)
    np.testing.assert_allclose(d_dt.detach().numpy(), dt_fd.numpy(), rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(d_dz.detach().numpy(), dz_fd.numpy(), rtol=2e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. the void closure, against a symbolic transcription of its documented form
# ---------------------------------------------------------------------------


def test_void_closure_matches_a_symbolic_transcription_of_its_own_definition():
    """``alpha = 1 - (1 - b)**3`` and its slope, from `sympy` rather than from the code.

    The closure is D-TH-3, a registered deviation, so what it *should* be is written
    down independently of what it *is*. The docstring's two load-bearing claims are
    checked here: the cubic form, and ``d alpha / d b = 3`` at ``b = 0`` — the
    finite slope that makes it usable where ``sqrt(b)`` returns NaN.
    """
    sympy = pytest.importorskip("sympy")

    from pinn_sfr_transient.axial.physics import boiling_fraction, quasi_steady_void

    b = sympy.symbols("b", nonnegative=True)
    alpha_sym = 1 - (1 - b) ** 3
    slope_at_zero = float(sympy.diff(alpha_sym, b).subs(b, 0))
    assert slope_at_zero == 3.0, "the docstring's whole argument for the cubic"

    p = AxialParams()
    T = np.linspace(900.0, 1400.0, 401)
    bf = np.asarray(boiling_fraction(T, p), dtype=np.float64)
    want = np.asarray(
        [float(alpha_sym.subs(b, sympy.Float(float(v), 20))) for v in bf], dtype=np.float64
    )
    np.testing.assert_allclose(np.asarray(quasi_steady_void(T, p)), want, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# 4. the reactivity quadrature, against scipy
# ---------------------------------------------------------------------------


def test_reactivity_weights_converge_at_the_order_a_midpoint_rule_gives():
    """The weights must be a *second-order* quadrature, checked by refining the mesh.

    `kinetics_weights` returns ``shape(zeta) * dz`` on cell centres — a midpoint
    rule — and its docstring claims the sums in `reactivity` are therefore
    quadratures rather than bare sums. The honest test of that is the convergence
    ORDER, not an absolute tolerance: at the shipped ``n_axial = 40`` the Doppler
    weight sums to 1.00017851 against `scipy.quad`'s 1.0, and an error of 1.8e-4 is
    neither a bug nor a pass on its own — it is exactly ``O(h^2)`` for this shape.

    An absolute tolerance here would have failed for the right reason and been
    silenced for the wrong one. Refinement distinguishes a correct second-order
    rule from a subtly wrong one, which no single mesh can.
    """
    from pinn_sfr_transient.axial.physics import kinetics_weights

    exact_dop = integrate.quad(
        lambda z: float(AxialParams().power_shape(np.array([z]))[0]), 0.0, 1.0
    )[0]
    errors = []
    for n in (40, 80, 160, 320):
        p = AxialParams(n_axial=n)
        errors.append(abs(float(np.sum(kinetics_weights(p)[0])) - exact_dop))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(1.8 < o < 2.2 for o in orders), (errors, orders)

    # The void worth is piecewise linear, which a midpoint rule integrates exactly
    # up to round-off -- a different property, and worth asserting separately.
    p = AxialParams()
    exact_void = integrate.quad(lambda z: float(p.void_worth(np.array([z]))[0]), 0.0, 1.0)[0]
    assert abs(float(np.sum(kinetics_weights(p)[1])) - exact_void) < 1e-9


def test_void_worth_changes_sign_where_the_physics_says_it_does():
    """Objective 2 rests on the void worth changing sign; assert it, independently.

    `scipy.optimize.brentq` finds the crossing without using any of this project's
    root-finding, and the location is compared against `zeta_sign` — the parameter
    the whole M9 regime map is organised around.
    """
    p = AxialParams()
    root = optimize.brentq(lambda z: float(p.void_worth(np.array([z]))[0]), 0.05, 0.95)
    assert root == pytest.approx(p.zeta_sign, abs=1e-9)
