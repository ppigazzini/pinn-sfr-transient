"""Parameters for the 1D axial SFR boiling model.

Every field cites the SAS4A/SASSYS-1 manual (ANL/NSE-SAS/5.8.1,
https://sas-doc.nse.anl.gov/latest/) equation or section it comes from, so a
reader can check the formulation rather than trust it. Deviations from the
manual are registered in ``docs/axial_physics.md``.

Units are **SI** throughout (m, kg, s, K, W, Pa) — unlike the 0D
:class:`~pinn_sfr_transient.config.SFRParams`, whose heat capacities live in a
scaled system. Reactivity is dimensionless absolute (same units as
``beta_eff``), never dollars: mixing the two is a factor-``beta`` error, i.e.
about 286x here.

.. warning::
   The **default numeric values are representative placeholders**, chosen to be
   order-of-magnitude correct for an oxide-fuelled SFR pin cell. They are not
   taken from a specific reactor. Chapter 2 of the manual (the input-deck
   reference) is the place to source a consistent, realistic set; that is
   milestone M2 work. Nothing downstream should quote these as physical
   predictions.

Milestone status: **M0**. This module is the parameter container and the axial
shape functions. The derived-exact steady state (M2), the residuals (M2/M4) and
the kinetics closure (M6) are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pinn_sfr_transient.config import _BETA_I_U235, _LAMBDA_I, FloatArray

type Scalar = float | FloatArray


def _midpoints(n: int) -> FloatArray:
    """``n`` midpoint-rule abscissae on ``[0, 1]``."""
    return (np.arange(n, dtype=np.float64) + 0.5) / n


@dataclass(slots=True)
class AxialParams:
    """Configuration for the axially resolved SFR channel.

    The channel carries four materials — fuel, cladding, coolant and structure —
    matching the SAS4A radial mesh (manual section 3.2.2.1: 4-11 nodes in the
    fuel, 3 in the cladding, 1 in the coolant, 2 in the structure). Here each is
    radially lumped into one node per axial position; see
    ``docs/axial_physics.md`` deviation D-GEOM-1 for why that is defensible.
    """

    # --- Geometry (manual section 3.2) -------------------------------------
    H: float = 1.0  # active core height [m]
    n_axial: int = 40  # axial nodes
    r_fo: float = 3.60e-3  # fuel outer radius [m]
    r_ci: float = 3.75e-3  # cladding inner radius [m]
    r_co: float = 4.25e-3  # cladding outer radius [m]
    A_c: float = 3.34e-5  # coolant flow area per pin [m^2]
    D_h: float = 5.00e-3  # hydraulic diameter [m]
    # Structure (duct wall) surface area per unit length, relative to cladding.
    # This is the manual's gamma_2 of section 12.5.1. Set to 0.0 to drop the
    # structure node entirely (see deviation D-GEOM-2).
    gamma_2: float = 0.5
    t_struct: float = 2.0e-3  # structure (duct) wall thickness [m]

    # --- Material properties (constant; sodium is M1's job) ----------------
    rho_f: float = 9.70e3  # fuel density [kg/m^3]
    c_f: float = 330.0  # fuel specific heat [J/kg-K]
    rho_cl: float = 7.80e3  # cladding density [kg/m^3]
    c_cl: float = 550.0  # cladding specific heat [J/kg-K]
    rho_s: float = 7.80e3  # structure density [kg/m^3]
    c_s: float = 550.0  # structure specific heat [J/kg-K]
    # Coolant properties are held CONSTANT at M2 (deviation D-TH-3): the defaults
    # below are the section 12.13 correlations evaluated at `T_ref_props`, which
    # keeps the steady state analytically exact for solver verification. M4
    # replaces them with the temperature-dependent forms from `axial.sodium`.
    T_ref_props: float = 700.0  # reference temperature for the frozen properties [K]
    rho_c: float = 849.09  # = sodium.liquid_density(700 K) [kg/m^3]
    c_c: float = 1272.29  # = sodium.liquid_heat_capacity(700 K) [J/kg-K]

    # --- Heat transfer (manual Eq. 3.3-4, Eq. 3.3-5) -----------------------
    h_gap: float = 1.0e4  # fuel-cladding bond gap conductance [W/m^2-K]
    emissivity: float = 0.70  # fuel thermal emissivity, radiation term of Eq. 3.3-4
    h_clad_coolant: float = 8.0e4  # cladding-coolant film coefficient [W/m^2-K]
    h_struct_coolant: float = 8.0e4  # structure-coolant film coefficient [W/m^2-K]
    # Fraction of total power deposited DIRECTLY in the coolant by neutron and
    # gamma heating (manual Eq. 3.3-6). Small, but it bypasses the fuel and
    # cladding thermal lag entirely, so power reaches the coolant with no delay.
    gamma_c: float = 0.02

    # --- Neutron kinetics (manual Eq. 4.2-4, Eq. 4.3-1) --------------------
    # Same six-group data as the 0D model, rescaled to the SFR beta_eff.
    beta_eff: float = 3.5e-3
    Lambda: float = 5.0e-7  # prompt neutron generation time [s]
    lambda_i: FloatArray = field(default_factory=_LAMBDA_I.copy)

    # --- Reactivity feedback (manual section 4.5) --------------------------
    # Doppler is LOGARITHMIC: delta_k = alpha_D * ln(T_f/T_f0), from
    # T_f d(delta_k)/dT_f = alpha_D (Eq. 4.5-2, Eq. 4.5-3). alpha_D is
    # interpolated between flooded and voided values -- the manual's ADOP and
    # BDOP inputs. Set them equal to switch that coupling off.
    alpha_D_flooded: float = -6.0e-3  # ADOP
    alpha_D_voided: float = -4.0e-3  # BDOP
    # Coolant density AND voiding share ONE worth distribution (Eq. 4.5-25:
    # delta_k = eps_d * sum_j (rho_c)_j * alpha_j). `void_worth_net` is the
    # reactivity inserted if the whole channel voids, i.e. the integral of
    # `void_worth` over the normalised height.
    void_worth_net: float = 2.0e-3
    # Axial worth shape: positive through the core, negative near the top where
    # leakage dominates. `zeta_sign` is the sign-change height.
    zeta_sign: float = 0.80
    delta_sign: float = 0.05
    rho_ext: float = 0.0  # external insertion; criticality offset is M2's job

    # --- Boiling onset (manual section 12.4) -------------------------------
    dT_superheat: float = 10.0  # DTS: superheat for the first bubble [K]
    dT_superheat_later: float = 3.0  # DTSI: subsequent bubbles [K]
    dT_smooth: float = 2.0  # logistic width making the onset differentiable [K]

    # --- Flow (manual Eq. 5.3-61 analogue; see deviation D-FLOW-1) ---------
    tau_pump: float = 5.0  # coast-down time constant [s]
    f_nc: float = 0.15  # natural-circulation floor, fraction of nominal
    w_0: float = 0.25  # nominal coolant mass flow per pin [kg/s]
    T_in: float = 628.0  # core inlet temperature [K]

    # --- Axial power shape -------------------------------------------------
    power_extrap: float = 0.10  # chopped-cosine extrapolation fraction
    P_0: float = 5.0e4  # nominal power per pin [W]

    # --- Time domain -------------------------------------------------------
    t_end: float = 60.0

    # --- Derived (not constructor arguments) -------------------------------
    beta_i: FloatArray = field(init=False)
    void_shape_norm: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate the inputs and derive the group fractions and shape norms."""
        self._validate()
        # Rescale the U-235 relative six-group spectrum to this beta_eff, exactly
        # as the 0D model does, so the two share delayed-neutron data.
        rel = _BETA_I_U235 / _BETA_I_U235.sum()
        self.beta_i = rel * self.beta_eff
        # Normalisation making `void_worth` integrate to `void_worth_net`.
        # Midpoint rule, not `np.trapezoid` (numpy >= 2.0 only) or `np.trapz`
        # (deprecated there): the numpy floor is 1.26 so the package installs on
        # Colab without upgrading its numpy in-kernel.
        self.void_shape_norm = float(np.mean(self._void_shape(_midpoints(20001))))

    def _validate(self) -> None:
        """Reject configurations that are geometrically or physically impossible."""
        if not self.r_fo < self.r_ci < self.r_co:
            radii = f"{self.r_fo}, {self.r_ci}, {self.r_co}"
            msg = f"radii must satisfy r_fo < r_ci < r_co, got {radii}"
            raise ValueError(msg)
        positives = {
            "H": self.H,
            "A_c": self.A_c,
            "D_h": self.D_h,
            "t_end": self.t_end,
            "w_0": self.w_0,
            "P_0": self.P_0,
            "beta_eff": self.beta_eff,
            "Lambda": self.Lambda,
            "dT_smooth": self.dT_smooth,
            "delta_sign": self.delta_sign,
            "t_struct": self.t_struct,
            "rho_c": self.rho_c,
            "c_c": self.c_c,
        }
        for name, value in positives.items():
            if value <= 0.0:
                msg = f"{name} must be > 0, got {value}"
                raise ValueError(msg)
        if self.n_axial < 2:
            msg = f"n_axial must be >= 2, got {self.n_axial}"
            raise ValueError(msg)
        fractions = {"f_nc": self.f_nc, "gamma_c": self.gamma_c, "emissivity": self.emissivity}
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must lie in [0, 1], got {value}"
                raise ValueError(msg)
        if not 0.0 < self.zeta_sign < 1.0:
            msg = f"zeta_sign must lie in (0, 1), got {self.zeta_sign}"
            raise ValueError(msg)
        if self.gamma_2 < 0.0:
            msg = f"gamma_2 must be >= 0, got {self.gamma_2}"
            raise ValueError(msg)
        if self.alpha_D_flooded > 0.0 or self.alpha_D_voided > 0.0:
            msg = "Doppler coefficients must be <= 0 (negative feedback)"
            raise ValueError(msg)

    # -- axial mesh ---------------------------------------------------------
    def zeta_edges(self) -> FloatArray:
        """Normalised heights of the ``n_axial + 1`` axial cell faces."""
        return np.linspace(0.0, 1.0, self.n_axial + 1)

    def zeta_nodes(self) -> FloatArray:
        """Normalised heights of the ``n_axial`` axial cell centres."""
        edges = self.zeta_edges()
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def dz(self) -> float:
        """Axial cell height [m]."""
        return self.H / self.n_axial

    # -- axial shape functions ----------------------------------------------
    def power_shape(self, zeta: Scalar) -> Scalar:
        """Normalised axial power shape, a chopped cosine with unit integral.

        Parameters
        ----------
        zeta
            Normalised height in ``[0, 1]``.

        Returns
        -------
        float or ndarray
            ``f(zeta)`` satisfying ``integral of f over [0, 1] == 1``, so that
            the local volumetric source is ``P(t) * f(zeta)``. Keeping the shape
            fixed in time is the same separable-flux assumption that justifies
            point kinetics (manual Eq. 4.2-1), so it costs nothing extra.
        """
        k = 1.0 / (1.0 + 2.0 * self.power_extrap)
        # integral of cos(pi k (z - 1/2)) over [0, 1] = (2 / (pi k)) sin(pi k / 2)
        norm = (2.0 / (np.pi * k)) * np.sin(0.5 * np.pi * k)
        return np.cos(np.pi * k * (np.asarray(zeta) - 0.5)) / norm

    def _void_shape(self, zeta: Scalar) -> Scalar:
        """Unnormalised axial void-worth shape (positive low, negative high)."""
        z = np.asarray(zeta)
        return np.sin(np.pi * z) * np.tanh((self.zeta_sign - z) / self.delta_sign)

    def void_worth(self, zeta: Scalar) -> Scalar:
        """Axial coolant void reactivity worth, integrating to ``void_worth_net``.

        This is the continuous form of the manual's per-segment worth table
        ``(rho_c)_ij`` in Eq. 4.5-25. It is **positive** through the core
        (spectrum hardening) and turns **negative** above ``zeta_sign``, where
        increased leakage dominates — the sign change that a single uniform
        coefficient cannot represent and that materially bounds the excursion.
        """
        return self.void_worth_net * self._void_shape(zeta) / self.void_shape_norm

    # -- feedback coefficients ----------------------------------------------
    def alpha_D(self, void_fraction: Scalar) -> Scalar:
        """Local Doppler coefficient, interpolated flooded -> voided.

        The manual (section 4.5.3) adjusts the local Doppler coefficient
        linearly between the coolant-in (``ADOP``) and coolant-out (``BDOP``)
        input values "to correct for the effect of coolant voiding on neutron
        leakage". Voiding therefore modulates the very feedback that has to
        bound the excursion. Set ``alpha_D_voided == alpha_D_flooded`` to
        disable the coupling.
        """
        a = np.clip(np.asarray(void_fraction), 0.0, 1.0)
        return self.alpha_D_flooded + (self.alpha_D_voided - self.alpha_D_flooded) * a

    def steady_precursors(self, power: float = 1.0) -> FloatArray:
        """Steady-state normalised precursor concentrations for the given power."""
        return self.beta_i * power / (self.Lambda * self.lambda_i)
