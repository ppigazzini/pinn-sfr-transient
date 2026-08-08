# References

Annotated bibliography for `pinn-sfr-transient`. BibTeX entries are in
[`docs/references.bib`](references.bib).

---

## Physics-informed neural networks and operator learning

- **Raissi, Perdikaris & Karniadakis (2019).** *Physics-informed neural
  networks: a deep learning framework for solving forward and inverse problems
  involving nonlinear partial differential equations.* J. Comput. Phys. 378,
  686–707. doi:10.1016/j.jcp.2018.10.045 — the founding PINN paper.
- **Lu, Jin, Pang, Zhang & Karniadakis (2021).** *Learning nonlinear operators
  via DeepONet based on the universal approximation theorem of operators.*
  Nat. Mach. Intell. 3(3), 218–229. doi:10.1038/s42256-021-00302-5 —
  operator-learning surrogate.
- **Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar
  (2021).** *Fourier Neural Operator for parametric partial differential
  equations.* ICLR 2021. arXiv:2010.08895 — function-space surrogates.
- **Lu, Meng, Mao & Karniadakis (2021).** *DeepXDE: a deep learning library for
  solving differential equations.* SIAM Review 63(1), 208–228.
  doi:10.1137/19M1274067 — the framework used in `pinn_deepxde.py`.
  https://github.com/lululxvi/deepxde
- **Gangloff & Jouvin (2024).** *jinns: a JAX library for physics-informed neural
  networks.* arXiv:2412.14132 — a JAX/Equinox/Optax PINN library; the closest
  prior art for the JAX backend in `pinn_jax.py`.

## PINN training for stiff and multiscale problems

- **Ji, Qiu, Shi, Pan & Deng (2021).** *Stiff-PINN: physics-informed neural
  network for stiff chemical kinetics.* J. Phys. Chem. A 125(36), 8098–8106.
  doi:10.1021/acs.jpca.1c05102 — stiffness as the dominant PINN failure mode for
  kinetics.
- **Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021).** *Characterizing
  possible failure modes in physics-informed neural networks.* NeurIPS 34,
  26548–26560. arXiv:2109.01050.
- **Wang, Teng & Perdikaris (2021).** *Understanding and mitigating gradient flow
  pathologies in physics-informed neural networks.* SIAM J. Sci. Comput. 43(5),
  A3055–A3081. doi:10.1137/20M1318043 — the gradient-norm adaptive loss weights
  used in `pinn_torch.py`.
- **Wang, Yu & Perdikaris (2022).** *When and why PINNs fail to train: a neural
  tangent kernel perspective.* J. Comput. Phys. 449, 110768.
  doi:10.1016/j.jcp.2021.110768.
- **Wang, Sankaran & Perdikaris (2024).** *Respecting causality for training
  physics-informed neural networks.* Comput. Methods Appl. Mech. Engrg. 421,
  116813. doi:10.1016/j.cma.2024.116813 — the causal temporal weighting used in
  `pinn_torch.py`.
- **McClenny & Braga-Neto (2023).** *Self-adaptive physics-informed neural
  networks.* J. Comput. Phys. 474, 111722. doi:10.1016/j.jcp.2022.111722 —
  point-wise self-adaptive weights (SA-PINN).
- **Cuong, Lalić, Petrić, Binh & Roantree (2024).** *Adapting physics-informed
  neural networks to improve ODE optimization in mosquito population dynamics.*
  PLOS ONE 19(12), e0315762. doi:10.1371/journal.pone.0315762 — ODE
  normalization, gradient balancing, and causal training for multi-equation ODE
  systems.
- **Ko & Park (2024).** *VS-PINN: a fast and efficient training of
  physics-informed neural networks using variable-scaling methods for solving
  PDEs with stiff behavior.* arXiv:2406.06287; published in J. Comput. Phys. 529,
  113860 (2025), doi:10.1016/j.jcp.2025.113860.
- **Seiler, Lei & Protopapas (2025).** *Stiff transfer learning for
  physics-informed neural networks.* arXiv:2501.17281.
- **Wu, Zhu, Tan, Kartha & Lu (2023).** *A comprehensive study of non-adaptive
  and residual-based adaptive sampling for physics-informed neural networks.*
  Comput. Methods Appl. Mech. Engrg. 403, 115671. doi:10.1016/j.cma.2022.115671 —
  the residual-based adaptive sampling used in both PINN backends (`pinn_torch.py`
  grows an RAR reservoir; `pinn_jax.py` augments with a fixed-size set under `jit`).
- **Pagliardini, Ablin & Grangier (2024).** *The AdEMAMix optimizer: better,
  faster, older.* arXiv:2409.03137 — Adam with an additional slow second-moment
  EMA; the optional high-budget optimiser noted in `neural_network.md` §4.5
  (`optax.contrib.ademamix`).

## Reactor-physics PINNs

- **Schiassi, De Florio, Ganapol, Picca & Furfaro (2022).** *Physics-informed
  neural networks for the point kinetics equations for nuclear reactor
  dynamics.* Ann. Nucl. Energy 167, 108833. doi:10.1016/j.anucene.2021.108833 —
  the foundational point-kinetics PINN.
- **Prantikos, Tsoukalas & Heifetz (2022).** *Physics-informed neural network
  solution of point kinetics equations for a nuclear reactor digital twin.*
  Energies 15(20), 7697. doi:10.3390/en15207697.
- **Prantikos, Chatzidakis, Tsoukalas & Heifetz (2023).** *Physics-informed
  neural network with transfer learning (TL-PINN) based on domain similarity
  measure for prediction of nuclear reactor transients.* Sci. Rep. 13, 16840.
  doi:10.1038/s41598-023-43325-1.
- **Geng, Chen, Yu, Wang, Hu & Wang (2026).** *Physics-informed neural network
  for solving the neutron transport equation with differential order
  transformation and Fourier feature.* J. Comput. Phys. 546, 114519.
  doi:10.1016/j.jcp.2025.114519.
- **Yang, Gong, He et al. (2024).** *On the uncertainty analysis of the
  data-enabled physics-informed neural network for solving the neutron diffusion
  eigenvalue problem.* Nucl. Sci. Eng. 198(5), 1075–1096.
  doi:10.1080/00295639.2023.2236840.
- **Zhang, Zhu, Wang et al. (2026).** *Artificial intelligence in reactor
  physics: current status and future prospects.* Nucl. Sci. Tech.
  doi:10.1007/s41365-026-01928-z; arXiv:2503.02440 — recent review.

## SFR and advanced-reactor thermal-hydraulics ML

- **Abulawi, Hu, Balaprakash & Liu (2025).** *Bayesian optimized deep ensemble
  for uncertainty quantification of deep neural networks: a system-safety case
  study on sodium fast reactor thermal stratification modeling.* Reliab. Eng.
  Syst. Saf. 264, 111353. doi:10.1016/j.ress.2025.111353.
- **Lee, Oh, Song et al. (2026).** *Neural operator-based surrogate model for
  CFD: helical coil steam generator in a small modular reactor.* arXiv:2605.30277.
- **Almukhametov, Lim, Hu & Liu (2026).** *Graph neural ODE digital twins for
  control-oriented reactor thermal-hydraulic forecasting under partial
  observability.* arXiv:2604.07292 — couples a thermal-hydraulics surrogate to a
  point-kinetics module.

## Software and related code

- **neutherm-pinn** — coupled neutronics/thermal-hydraulics PINN in PyTorch.
  https://github.com/carcaraa/neutherm-pinn
- **Wen, Li, Azizzadenesheli, Anandkumar & Benson (2022).** *U-FNO—an enhanced
  Fourier neural operator-based deep-learning model for multiphase flow.* Adv.
  Water Resour. 163, 104180. doi:10.1016/j.advwatres.2022.104180. Repo:
  https://github.com/gegewen/ufno
- **Bradbury, Frostig, Hawkins et al. (2018).** *JAX: composable transformations
  of Python+NumPy programs.* https://github.com/jax-ml/jax — the array / autodiff
  / JIT (XLA) framework behind `pinn_jax.py`.
- **Kidger & Garcia (2021).** *Equinox: neural networks in JAX via callable
  PyTrees and filtered transformations.* Differentiable Programming workshop,
  NeurIPS 2021. arXiv:2111.00254. https://github.com/patrick-kidger/equinox —
  the model layer (`SFRPinn`) in `pinn_jax.py`.
- **DeepMind (2020).** *The DeepMind JAX Ecosystem* — Optax: composable gradient
  transformations and optimisers (`optax.adam`, `optax.lbfgs`).
  https://github.com/google-deepmind/optax — the optimiser layer in `pinn_jax.py`.
- **PyTorch** — https://pytorch.org · **DeepXDE** — https://github.com/lululxvi/deepxde ·
  **JAX** — https://github.com/jax-ml/jax · **uv**, **ruff**, **ty** (Astral) — project/lint/type tooling.

## Classical references (textbooks)

- **Duderstadt & Hamilton (1976).** *Nuclear Reactor Analysis.* Wiley.
  ISBN 978-0471223634 — point kinetics, reactivity.
- **Hetrick (1971).** *Dynamics of Nuclear Reactors.* University of Chicago Press —
  coupled kinetics + feedback.
- **Waltar, Todd & Tsvetkov, eds. (2012).** *Fast Spectrum Reactors.* Springer.
  doi:10.1007/978-1-4419-9572-8 — SFR physics, Doppler and sodium void
  coefficients.
- **Todreas & Kazimi (2012).** *Nuclear Systems, Vol. I: Thermal Hydraulic
  Fundamentals,* 2nd ed. CRC Press — lumped energy balances.
</content>
</invoke>
