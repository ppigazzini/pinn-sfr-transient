"""DeepXDE implementation of the SFR ULOF PINN (the "fast deployment" path).

Run (after ``uv sync --extra deepxde --extra torch-cpu``)::

    uv run python -m pinn_sfr_transient.pinn_deepxde

Uses the same normalized-state residuals as the from-scratch backends
(:mod:`pinn_sfr_transient.pinn_torch` and :mod:`pinn_sfr_transient.pinn_jax`), so
all three solve identical physics. DeepXDE handles autodiff, collocation sampling
and the training loop; we only declare the time domain, the residuals and the
(hard) initial conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinn_sfr_transient.config import SFRParams

try:
    import deepxde as dde  # ty: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    msg = "DeepXDE is required: `uv sync --extra deepxde --extra torch-cpu`"
    raise SystemExit(msg) from exc


def build_model(p: SFRParams) -> dde.Model:
    """Assemble the DeepXDE model for the normalized ULOF system."""
    lam = p.lambda_i
    beta_i = p.beta_i
    DTf = p.Tf0 - p.Tin
    DTc = p.Tc0 - p.Tin

    geom = dde.geometry.TimeDomain(0.0, p.t_end)

    def ode(t, s):  # backend tensors are untyped
        p_ = s[:, 0:1]
        c = s[:, 1:7]
        th_f = s[:, 7:8]
        th_c = s[:, 8:9]

        dp = dde.grad.jacobian(s, t, i=0)
        dthf = dde.grad.jacobian(s, t, i=7)
        dthc = dde.grad.jacobian(s, t, i=8)
        dcs = [dde.grad.jacobian(s, t, i=k) for k in range(1, 7)]

        Tf = p.Tin + th_f * DTf
        Tc = p.Tin + th_c * DTc

        phi = 0.5 * (1.0 + dde.backend.tanh(0.5 * (Tc - p.T_onset) / p.dT_void))
        rho = p.rho_ext + p.alpha_f * (Tf - p.Tf0) + p.alpha_c * (Tc - p.Tc0) + p.alpha_void * phi
        g = p.f_nc + (1.0 - p.f_nc) * dde.backend.exp(-t / p.tau_pump)

        sum_beta_c = sum(beta_i[k] * c[:, k : k + 1] for k in range(6))
        r_p = p.Lambda * dp - ((rho - p.beta_eff) * p_ + sum_beta_c)
        r_c = [dcs[k] - lam[k] * (p_ - c[:, k : k + 1]) for k in range(6)]
        r_tf = DTf * dthf - ((p.P0 / p.Cf) * p_ - (p.UA / p.Cf) * (Tf - Tc))
        r_tc = DTc * dthc - ((p.UA / p.Cc) * (Tf - Tc) - (p.W0 * g / p.Cc) * (Tc - p.Tin))
        return [r_p, *r_c, r_tf, r_tc]

    net = dde.nn.FNN([1, *([64] * 5), 9], "tanh", "Glorot normal")

    s0 = np.ones(9, dtype=np.float64)

    def out_transform(t, y):  # backend tensors are untyped
        return dde.backend.as_tensor(s0) + (t / p.t_end) * y

    net.apply_output_transform(out_transform)

    data = dde.data.PDE(geom, ode, [], num_domain=4000, num_boundary=2)
    return dde.Model(data, net)


def main() -> None:
    """Train the DeepXDE model and report relative L2 error vs the reference."""
    p = SFRParams()
    model = build_model(p)
    model.compile("adam", lr=1e-3)
    model.train(iterations=15000, display_every=1000)
    model.compile("L-BFGS")
    model.train()

    t = np.linspace(0, p.t_end, 2000)[:, None]
    s = model.predict(t)
    P = s[:, 0]
    Tf = p.Tin + s[:, 7] * (p.Tf0 - p.Tin)
    Tc = p.Tin + s[:, 8] * (p.Tc0 - p.Tin)

    refpath = Path("results/ulof_reference.npz")
    if refpath.exists():
        ref = dict(np.load(refpath))

        def rel(a: np.ndarray, key: str) -> float:
            b = np.interp(t[:, 0], ref["t"], ref[key])
            return float(np.linalg.norm(a - b) / np.linalg.norm(b))

        print(f"rel L2:  P={rel(P, 'P'):.2e}  Tf={rel(Tf, 'Tf'):.2e}  Tc={rel(Tc, 'Tc'):.2e}")


if __name__ == "__main__":
    main()
