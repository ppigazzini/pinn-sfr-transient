"""Scoring against the held-out reference.

Never imported by training, so nothing in the loss can reach the reference by
accident. That separation is the protocol, not a convention.

The metrics themselves live in :mod:`pinn_sfr_transient.axial.scoring`, once, and
are shared with the JAX twin and with ``tools/axial_study.py``. They were briefly
defined in three places and the study tool silently stopped reporting two of them.
"""

from __future__ import annotations

from pinn_sfr_transient.axial.scoring import relative_l2 as _score
from pinn_sfr_transient.axial.torchpinn.model import AxialPinn


def relative_l2(model: AxialPinn, traj: object) -> dict[str, float]:
    """Relative ``L2`` per field, the voided-length error, and the front metrics."""
    fields = model.predict(traj.zeta, traj.t)  # type: ignore[attr-defined]
    return _score(fields, traj, model.p)
