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
   reference) is the place to source a consistent, realistic set. Nothing
   downstream should quote these as physical predictions.

This module is the parameter container and the axial shape functions; the
residuals live in ``physics``, the solver in ``reference``, and the networks in
``pinn_torch`` / ``pinn_jax``.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinn_sfr_transient.axial._backend import xp as _xp
from pinn_sfr_transient.config import _BETA_I_U235, _LAMBDA_I, FloatArray

type Scalar = float | FloatArray

_MIN_VOID_SHAPE_NORM: float = 1e-3
"""Smallest usable mean of the void-worth shape; below it the normalisation blows up."""


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

    # --- Material properties (fuel/cladding/structure constant) ------------
    rho_f: float = 9.70e3  # fuel density [kg/m^3]
    c_f: float = 330.0  # fuel specific heat [J/kg-K]
    rho_cl: float = 7.80e3  # cladding density [kg/m^3]
    c_cl: float = 550.0  # cladding specific heat [J/kg-K]
    rho_s: float = 7.80e3  # structure density [kg/m^3]
    c_s: float = 550.0  # structure specific heat [J/kg-K]
    # Coolant properties are held CONSTANT: the defaults below are the section
    # 12.13 correlations evaluated at `T_ref_props`, which keeps the steady state
    # analytically exact and so usable as a verification oracle. Still constant --
    # `axial.sodium` supplies the temperature-dependent forms, and
    # `physics.coolant_capacity` records why substituting them into this
    # temperature-form energy equation breaks conservation.
    T_ref_props: float = 700.0  # reference temperature for the frozen properties [K]
    rho_c: float = 849.09  # = sodium.liquid_density(700 K) [kg/m^3]
    c_c: float = 1272.29  # = sodium.liquid_heat_capacity(700 K) [J/kg-K]

    # --- Heat transfer (manual Eq. 3.3-4, Eq. 3.3-5) -----------------------
    h_gap: float = 1.0e4  # fuel-cladding bond gap conductance [W/m^2-K]
    emissivity: float = 0.70  # fuel thermal emissivity, radiation term of Eq. 3.3-4
    h_clad_coolant: float = 8.0e4  # cladding-coolant film coefficient, wetted [W/m^2-K]
    h_struct_coolant: float = 8.0e4  # structure-coolant film coefficient, wetted [W/m^2-K]
    # Vapour-phase film coefficient after dryout (manual section 12.5.1: the
    # combined liquid-film and vapour resistance). Two to three ORDERS smaller
    # than the wetted value -- which is why losing the liquid is the safety event.
    h_vapour: float = 5.0e2
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
    # External insertion. No criticality offset is needed: at nominal
    # ln(T_f/T_f0) = 0 and alpha = 0, so both reactivity integrals vanish
    # identically and the reactor is exactly critical -- a free consequence of the
    # logarithmic Doppler that the 0D model has to fake with a fitted offset.
    rho_ext: float = 0.0

    # Axial fuel expansion feedback (manual section 4.5.4). Fuel lengthening
    # lowers the axial power density and leaks more neutrons, so the coefficient
    # is NEGATIVE and the mechanism is stabilising. Omitting it was
    # non-conservative -- the model over-predicts the excursion (D-FB-3).
    # Evaluated on the same radial mass-average as Doppler, per axial segment.
    alpha_expansion: float = 0.0  # [1/K]; -1e-6 is a representative metal-fuel value

    # --- Decay heat (manual Eq. 4.2-2, section 4.4) ------------------------
    # `psi_t = psi_f + psi_h`: total power is fission plus decay. `decay_fraction`
    # is `psi_h` at nominal, so at steady state `psi_t = 1` exactly with no
    # retuning. Three groups spanning seconds to hours, the shape of the ANS
    # standard rather than its full 23-group table (deviation D-KIN-3).
    #
    # **This removes the zero-power attractor.** With `decay_fraction = 0` the
    # homogeneous kinetics have no source and `P = c = 0` is an exact solution of
    # the whole coupled system -- the power-collapse mode. With it non-zero, `psi_h`
    # does not vanish when `psi_f` does. Typical
    # fission-product decay heat immediately after shutdown is 6-7% of nominal.
    decay_fraction: float = 0.0
    decay_lambda: FloatArray = field(
        default_factory=lambda: np.array([1.772e-1, 5.774e-3, 7.671e-5], dtype=np.float64)
    )
    decay_weight: FloatArray = field(
        default_factory=lambda: np.array([0.55, 0.31, 0.14], dtype=np.float64)
    )

    # --- Boiling onset (manual section 12.4) -------------------------------
    p_system: float = 1.01325e5  # system pressure setting T_sat via Eq. 12.13-4 [Pa]
    dT_superheat: float = 10.0  # DTS: superheat for the first bubble [K]
    dT_superheat_later: float = 3.0  # DTSI: subsequent bubbles [K]
    dT_smooth: float = 2.0  # logistic width making the onset differentiable [K]
    # Condensation coefficient (manual section 12.5, "the condensation
    # coefficient"). Section 12.5 determines the rate of vapour *formation and
    # condensation* from one film heat flow, condensation being its negative
    # branch: "Condensation in cooler regions can cause a vapor bubble to shrink."
    #
    # **EXPERIMENTAL, default 0.0 = off.** At zero the vapour source is
    # non-negative, so `alpha` is monotone along a characteristic and slaved to
    # `T_c` -- which is what makes the algebraic closure D-TH-3 valid and the
    # front-position network redundant. Switching it on restores the void as a
    # genuine differential unknown and inverts both of those conclusions.
    condensation: float = 0.0

    # --- Flow (manual Eq. 5.3-61 analogue; see deviation D-FLOW-1) ---------
    # Let vapour generation accelerate the mixture, so the flow acquires a
    # z-dependence after voiding starts (deviation D-TH-2). Eq. 3.9-1 states
    # w = w(t) independent of z, and states its own validity limit: incompressible
    # liquid, one slug filling the channel. That fails the moment vapour forms,
    # because vaporising a kilogram of sodium multiplies its volume by
    # rho_l/rho_v ~ 3100. **EXPERIMENTAL, default False = the Eq. 3.9-1 form.**
    flow_expansion: bool = False
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
        # Midpoint rule on purpose, not `np.trapezoid`: the shape vanishes at
        # both endpoints, where the trapezoid rule spends its two samples, and
        # the midpoint rule is the same order for half the evaluations.
        # Decay-group weights are a partition of `decay_fraction`.
        self.decay_weight = self.decay_weight / self.decay_weight.sum()
        self.void_shape_norm = float(np.mean(self._void_shape(_midpoints(20001))))
        # The shape changes sign, so its mean can be driven arbitrarily close to
        # zero by `zeta_sign` — at which point `void_worth` explodes to keep the
        # requested net. Catch that here rather than in a silently absurd worth.
        if abs(self.void_shape_norm) < _MIN_VOID_SHAPE_NORM:
            msg = (
                f"void worth shape nearly integrates to zero "
                f"(mean {self.void_shape_norm:.2e}) at zeta_sign={self.zeta_sign}; "
                f"the positive and negative lobes cancel, so `void_worth_net` "
                f"cannot be normalised"
            )
            raise ValueError(msg)

    def _validate(self) -> None:
        """Reject configurations that are geometrically or physically impossible."""
        self._validate_geometry()
        self._validate_ranges()

    def _validate_geometry(self) -> None:
        """Reject impossible radii and meshes."""
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
            "p_system": self.p_system,
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

    def _validate_ranges(self) -> None:
        """Reject coefficients outside their physical range."""
        fractions = {"f_nc": self.f_nc, "gamma_c": self.gamma_c, "emissivity": self.emissivity}
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must lie in [0, 1], got {value}"
                raise ValueError(msg)
        if not 0.0 < self.zeta_sign < 1.0:
            msg = f"zeta_sign must lie in (0, 1), got {self.zeta_sign}"
            raise ValueError(msg)
        non_negative = {"gamma_2": self.gamma_2, "condensation": self.condensation}
        for name, value in non_negative.items():
            if value < 0.0:
                msg = f"{name} must be >= 0, got {value}"
                raise ValueError(msg)
        if self.alpha_expansion > 0.0:
            msg = f"alpha_expansion must be <= 0 (stabilising), got {self.alpha_expansion}"
            raise ValueError(msg)
        if not 0.0 <= self.decay_fraction < 1.0:
            msg = f"decay_fraction must lie in [0, 1), got {self.decay_fraction}"
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

    def power_shape_integral(self, zeta: Scalar) -> Scalar:
        """Cumulative axial power fraction, the closed-form integral of :meth:`power_shape`.

        ``F(zeta) = integral of f from 0 to zeta``, with ``F(0) = 0`` and
        ``F(1) = 1``. Having it in closed form matters for the PINN: the
        continuous steady state -- its hard initial condition -- is
        ``T_c0(zeta) = T_in + (P_0 / (w_0 c_c)) F(zeta)``, so an exact ``F``
        makes the ansatz satisfy the inlet boundary condition identically rather
        than approximately.
        """
        k = 1.0 / (1.0 + 2.0 * self.power_extrap)
        half = 0.5 * np.pi * k
        return (np.sin(np.pi * k * (np.asarray(zeta) - 0.5)) + np.sin(half)) / (2.0 * np.sin(half))

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
        # Backend-agnostic: this is called from inside the PINN residual, where
        # `void_fraction` is a torch or JAX array that must stay differentiable.
        a = _xp(void_fraction).clip(void_fraction, 0.0, 1.0)
        return self.alpha_D_flooded + (self.alpha_D_voided - self.alpha_D_flooded) * a

    @classmethod
    def with_positive_void_worth(cls, **kwargs: Any) -> AxialParams:  # noqa: ANN401
        """Parameters under which the void feedback is actually sampled positive.

        At the shipped defaults ``zeta_sign = 0.80`` sits *below* the boiling
        onset at ``zeta ~ 0.96``, and the mixture void advects upward only, so
        every void contribution is negative and ``max rho/beta`` is identically
        zero (deviation register, D49). That is an artefact of where the channel
        boils, not a statement about stability, and it makes Objective 2
        unanswerable: scaling ``void_worth_net`` scales a term that is
        non-positive everywhere.

        Moving the sign change above the onset location samples the positive
        branch instead. **Neither set is more correct** — a real core has its
        reversal near the axial blanket, which is what the default represents.
        This one exists so the positive-feedback question can be asked at all,
        and any result from it must say which set produced it.
        """
        return cls(zeta_sign=0.995, delta_sign=0.02, **kwargs)

    def steady_decay_heat(self, power: float = 1.0) -> FloatArray:
        """Steady-state decay-heat group powers, summing to ``decay_fraction``."""
        return self.decay_fraction * self.decay_weight * power

    def steady_precursors(self, power: float = 1.0) -> FloatArray:
        """Steady-state **normalised** precursor concentrations for the given power.

        The axial model carries ``c_i = C_i / C_{i,0}`` with
        ``C_{i,0} = beta_i / (Lambda lambda_i)``, so ``dc_i/dt = lambda_i (P - c_i)``
        (Eq. 4.3-1 in normalised form) is steady exactly when ``c_i == P``. Every
        group therefore sits at the same value, and at nominal power
        ``P = sum(beta_i) / beta = 1``.

        This previously returned the *absolute* ``C_i`` — the 0D model's
        convention, where the precursors are unnormalised — which is about
        ``5e4`` times too large for this model's state vector. It was never
        called, because :func:`~pinn_sfr_transient.axial.reference.steady_state`
        writes ``np.ones(N_GROUPS)`` directly, so the two definitions are now
        pinned against each other in the tests rather than left to drift.
        """
        return np.full(self.lambda_i.shape, power, dtype=np.float64)
