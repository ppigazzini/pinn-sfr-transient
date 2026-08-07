"""Scoring against the held-out reference.

Never imported by training, so nothing in the loss can reach the reference by
accident. That separation is the protocol, not a convention.
"""

from __future__ import annotations

import numpy as np

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
    return out
