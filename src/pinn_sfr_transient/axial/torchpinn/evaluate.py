"""Scoring against the held-out reference.

Never imported by training, so nothing in the loss can reach the reference by
accident. That separation is the protocol, not a convention.
"""

from __future__ import annotations

import numpy as np

from pinn_sfr_transient.axial import sodium
from pinn_sfr_transient.axial.config import AxialParams
from pinn_sfr_transient.axial.torchpinn.archs import FIELDS, N_TEMPS
from pinn_sfr_transient.axial.torchpinn.model import AxialPinn


def relative_l2(model: AxialPinn, traj: object) -> dict[str, float]:
    """Relative ``L2`` error of every field against the held-out reference."""
    fields = model.predict(traj.zeta, traj.t)  # type: ignore[attr-defined]
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)  # type: ignore[attr-defined]
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS[:N_TEMPS], fields[:N_TEMPS], ref, strict=True)
    }
    # The void is near zero over most of the domain, so a relative L2 there is
    # dominated by its denominator. Report the absolute voided-length error in
    # metres instead -- the quantity M4 is actually judged on.
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H  # type: ignore[attr-defined]
    out["L_void_max_err_m"] = float(
        np.max(np.abs(fields[N_TEMPS].sum(axis=0) * dz - traj.voided_length))  # type: ignore[attr-defined]
    )
    return out | front_metrics(fields, traj, model.p)


def front_metrics(fields: tuple, traj: object, p: AxialParams) -> dict[str, float]:
    """Metrics the front actually depends on, which a relative ``L2`` cannot see.

    Under D-TH-3 the void is a function of ``T_c`` alone, so "the front forms" is
    the single inequality ``max T_c > T_sat + dT_superheat``. That is an
    **extremum**; a relative ``L2`` is an **average**, and the two move
    independently -- a smoother fit scores better in the mean and can drop the peak
    below threshold, switching the front off with no warning in any temperature
    metric (`docs/axial_nn.md` section 7.2.8). Report the margin, so a run that
    loses the front says so.

    ``max_alpha`` is returned for continuity with the published tables, but it is
    **derived from the margin, not independent of it**: the closure is invertible,
    so ``margin -> max_alpha`` is exact and measured so (`axial_nn.md` section
    7.2.8, four arms, four exact matches). It also saturates by about 8 K of
    margin, past which it cannot distinguish a front that barely exists from one
    with 20 K of headroom. The informative pair is ``margin_K`` and
    ``L_void_max_err_m``: the first gates whether a front exists, the second says
    how much of the channel is in it.
    """
    threshold = sodium.saturation_temperature(p.p_system) + p.dT_superheat
    max_T_c = float(fields[3].max())
    return {
        "max_T_c": max_T_c,
        "T_boil": float(threshold),
        # Negative means the network never reaches saturation anywhere, so
        # `alpha` is identically zero and there is no front at all.
        "margin_K": max_T_c - float(threshold),
        "margin_K_ref": float(traj.T_c.max()) - float(threshold),  # type: ignore[attr-defined]
        "max_alpha": float(fields[4].max()),
    }
