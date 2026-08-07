"""Evaluation: predictions and scores against the held-out reference.

Never imported by training, so nothing in the loss can reach the reference by
accident. That separation is the protocol, not a convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.config import FloatArray

import numpy as np

from pinn_sfr_transient.axial.jaxpinn.ansatz import (
    horizon,
    normalised_state,
    precursors,
    theta0,
)
from pinn_sfr_transient.axial.jaxpinn.archs import FIELDS, N_TEMPS, AxialPinn
from pinn_sfr_transient.axial.jaxpinn.config import AxialTrainConfig
from pinn_sfr_transient.axial.physics import (
    kinetics_weights,
    prompt_jump_power,
    reactivity,
)


# --- inference --------------------------------------------------------------
def predict(
    model: AxialPinn,
    p: AxialParams,
    zeta: FloatArray,
    t: FloatArray,
    cfg: AxialTrainConfig | None = None,
) -> tuple[FloatArray, ...]:
    """Evaluate on a ``(zeta, t)`` grid, returning physical fields ``(n_z, n_t)``."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    cfg = cfg or AxialTrainConfig()
    t_end = horizon(p, cfg)
    zz, tt = np.meshgrid(zeta, t, indexing="ij")
    z = jnp.asarray(zz.reshape(-1, 1))
    h = jnp.asarray((tt / t_end).reshape(-1, 1))
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(z, h)
    out = [np.asarray(p.T_in + theta[:, k] * dT).reshape(zz.shape) for k in range(N_TEMPS)]
    out.append(np.asarray(theta[:, N_TEMPS]).reshape(zz.shape))
    return tuple(out)


def predict_power(
    model: AxialPinn, p: AxialParams, t: FloatArray, cfg: AxialTrainConfig | None = None
) -> tuple[FloatArray, FloatArray]:
    """Normalised power and net reactivity on a time grid (Plan A only)."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    cfg = cfg or AxialTrainConfig()
    t_end = horizon(p, cfg)
    zeta_q = jnp.asarray(p.zeta_nodes().reshape(-1, 1))
    w_D, w_void = (jnp.asarray(x) for x in kinetics_weights(p))
    that = jnp.asarray((np.asarray(t) / t_end).reshape(-1, 1))
    n_t, n_z = that.shape[0], zeta_q.shape[0]
    zeta = jnp.tile(zeta_q, (n_t, 1))
    t_rep = jnp.repeat(that, n_z, axis=0)
    theta = jax.vmap(lambda a, b: normalised_state(model, p, a, b, cfg))(zeta, t_rep)
    T_f = p.T_in + theta[:, 0].reshape(n_t, n_z) * dT
    alpha = theta[:, N_TEMPS].reshape(n_t, n_z)
    T_f0 = p.T_in + jax.vmap(lambda z: theta0(p, z))(zeta_q)[:, 0] * dT
    rho = reactivity(T_f, alpha, T_f0[None, :], w_D, w_void, p)
    c = jax.vmap(lambda x: precursors(model, x))(that)
    power = prompt_jump_power(c, rho.reshape(-1, 1), p)
    return np.asarray(power).ravel(), np.asarray(rho).ravel()


def relative_l2(
    model: AxialPinn, p: AxialParams, traj: object, cfg: AxialTrainConfig | None = None
) -> dict[str, float]:
    """Relative ``L2`` per temperature field, plus the absolute voided-length error."""
    fields = predict(model, p, traj.zeta, traj.t, cfg)  # type: ignore[attr-defined]
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)  # type: ignore[attr-defined]
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS[:N_TEMPS], fields[:N_TEMPS], ref, strict=True)
    }
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H  # type: ignore[attr-defined]
    out["L_void_max_err_m"] = float(
        np.max(np.abs(fields[N_TEMPS].sum(axis=0) * dz - traj.voided_length))  # type: ignore[attr-defined]
    )
    return out
