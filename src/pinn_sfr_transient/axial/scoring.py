"""Scoring against the held-out reference — one definition, shared by everything.

Both backends' ``predict`` return plain numpy arrays, so nothing here needs torch
or JAX and there is no reason for two copies. There were two, briefly, and then a
third in ``tools/axial_study.py``: the study tool grew its own ``score`` and
silently did not report the M4 onset metrics the evaluators had just gained. That
is the same defect class as D67 — a number whose definition lives in more than one
place drifts, and the drift is invisible.

Never imported by any training module, so nothing in a loss can reach the
reference by accident. That separation is the protocol, not a convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pinn_sfr_transient.axial import sodium

if TYPE_CHECKING:
    from pinn_sfr_transient.axial.config import AxialParams
    from pinn_sfr_transient.axial.reference import AxialTrajectory

FIELDS = ("T_f", "T_cl", "T_s", "T_c")
ONSET_THRESHOLD = 0.01
# Below this peak void fraction the front is vestigial and its position is noise.
# Not a tuning knob: `max alpha` is a saturating function of the saturation
# margin, and 0.9 corresponds to roughly 2 K above threshold -- the point at
# which there is a front rather than a trace of one.
MIN_ALPHA_FOR_ONSET = 0.9


def onset(alpha: np.ndarray, zeta: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """First time and normalised height at which the void exceeds the threshold.

    The same rule :meth:`AxialTrajectory.onset` applies, so the network and the
    reference are scored by one definition rather than two transcriptions of it.
    Returns ``(nan, nan)`` when boiling never starts — which is a **failure**, not
    a missing measurement, and must never be defaulted to zero error.
    """
    hit = alpha > ONSET_THRESHOLD
    if not hit.any():
        return float("nan"), float("nan")
    i = int(np.argmax(hit.any(axis=0)))
    return float(t[i]), float(zeta[int(np.argmax(hit[:, i]))])


def front_metrics(fields: tuple, traj: AxialTrajectory, p: AxialParams) -> dict[str, float]:
    """Metrics the front depends on, which a relative ``L2`` cannot see.

    Under D-TH-3 the void is a function of ``T_c`` alone, so "the front forms" is
    the single inequality ``max T_c > T_sat + dT_superheat``. That is an
    **extremum**; a relative ``L2`` is an **average**, and the two move
    independently — a smoother fit scores better in the mean and can drop the peak
    below threshold, switching the front off with no warning in any temperature
    metric (`docs/axial_nn.md` section 7.2.8).

    ``max_alpha`` is returned for continuity with the published tables, but it is
    **derived from the margin, not independent of it**: the closure is invertible,
    so ``margin -> max_alpha`` is exact and measured so (four arms, four exact
    matches). It also saturates by about 8 K of margin, past which it cannot
    distinguish a front that barely exists from one with 20 K of headroom.

    Onset is reported as ``nan`` unless ``max_alpha`` clears
    :data:`MIN_ALPHA_FOR_ONSET`, because a vestigial front still has a first point
    above the void threshold and that point can land anywhere.

    ``onset_t_err_s`` and ``onset_zeta_err`` are **M4's actual acceptance
    criterion** — onset within 0.5 s and one cell — and until they were added
    nothing reported them, so M4 could be neither passed nor failed. Both are
    measurable: onset time is mesh-independent at 10.75 s and onset location is
    converged to one cell across ``n_axial`` 40 to 640 (section 6.5). One cell at
    the scoring mesh is ``1/160 = 0.00625``.
    """
    threshold = float(sodium.saturation_temperature(p.p_system) + p.dT_superheat)
    max_T_c = float(fields[3].max())
    t_on, z_on = onset(fields[4], traj.zeta, traj.t)
    t_ref, z_ref = traj.onset()
    # A front that barely exists still has a "first point where alpha > 0.01", and
    # that point can land anywhere -- so a configuration with a vestigial front can
    # score WELL on onset location. Measured: the shipped default reaches
    # `max alpha = 0.685` and `L_void = 0.037` against the reference's 0.381, and
    # scored onset_zeta_err = 0.00000 on one seed, better than the arm that
    # actually forms a front. Require a front before reporting where it is.
    if float(fields[4].max()) < MIN_ALPHA_FOR_ONSET:
        t_on = z_on = float("nan")
    return {
        "max_T_c": max_T_c,
        "T_boil": threshold,
        # Negative means the network never reaches saturation anywhere, so `alpha`
        # is identically zero and there is no front at all.
        "margin_K": max_T_c - threshold,
        "margin_K_ref": float(traj.T_c.max()) - threshold,
        "max_alpha": float(fields[4].max()),
        "onset_t": t_on,
        "onset_zeta": z_on,
        "onset_t_err_s": abs(t_on - t_ref),
        "onset_zeta_err": abs(z_on - z_ref),
    }


def relative_l2(fields: tuple, traj: AxialTrajectory, p: AxialParams) -> dict[str, float]:
    """Relative ``L2`` per temperature, the voided-length error, and the front metrics."""
    ref = (traj.T_f, traj.T_cl, traj.T_s, traj.T_c)
    out = {
        name: float(np.linalg.norm(f - r) / np.linalg.norm(r))
        for name, f, r in zip(FIELDS, fields[:4], ref, strict=True)
    }
    # The void is near zero over most of the domain, so a relative L2 there is
    # dominated by its denominator. Report the absolute voided-length error in
    # metres instead -- the quantity M4 is actually judged on.
    dz = (traj.zeta[1] - traj.zeta[0]) * traj.H
    l_void = fields[4].sum(axis=0) * dz
    out["L_void_max_err_m"] = float(np.max(np.abs(l_void - traj.voided_length)))
    # The network's own peak voided length, and the reference's, so a table can
    # quote the quantity rather than only its error. These are different numbers
    # and conflating them cost a KeyError that was really a units mix-up.
    out["L_void_max"] = float(l_void.max())
    out["L_void_max_ref"] = float(traj.voided_length.max())
    return out | front_metrics(fields, traj, p)
