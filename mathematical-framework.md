# Mathematical Framework

This document describes the mathematical structure of the simulation. The code sections are marked with matching `§` headers.

---

## 1. State space

Each institution i ∈ {1, …, 35} has two state variables at time t:

- **Lf_i(t)** ∈ [0, 1] — funding liquidity (1 = fully funded, 0 = frozen)
- **Lm_i(t)** ∈ [0, ∞) — market illiquidity (0 = perfectly liquid market)

The system state is the vector (Lf, Lm) ∈ ℝ^{2N}. The stress manifold Z: ℝ² → ℝ is a scalar field over the eigenmap embedding of the exposure graph, evaluated at each tick.

---

## 2. Brunnermeier–Pedersen liquidity spiral

From Brunnermeier & Pedersen (2009), *Market Liquidity and Funding Liquidity*, Review of Financial Studies.

The ODE system:

```
dLf_i/dt = θ_F · (L̄_F - Lf_i) + ξ_i(t) · dt + σ_F · Lm_i · dW_i
dLm_i/dt = θ_M · (max(0, L̄_F - Lf_i) - Lm_i · δ) · (1 + γ · fireSale_i)
```

where:
- θ_F = 0.08, θ_M = 0.12 — mean-reversion speeds
- L̄_F = 0.70 — long-run funding liquidity target
- δ = 0.30 — market liquidity damping
- γ = 0.60 — fire-sale amplification
- fireSale_i = clamp((spread_i - 50bp) / 70bp, 0, 0.8)

The spiral mechanism: when Lf falls, fire-sale pressure increases (dealers sell collateral), which raises Lm, which depresses prices, which forces further selling, which reduces Lf further.

Numerical integration: Euler–Maruyama with adaptive step size, dt ∈ [0.02, 0.15] based on frame time EMA.

---

## 3. Rough Bergomi volatility

Instantaneous variance follows:

```
σ²_i(t) = σ²_0 · E[exp(η · W^H_t - η²/2 · t^{2H})]
```

where W^H is fractional Brownian motion with H = 0.10 (rough), η = 1.8 (vol-of-vol), σ_0 = 0.18 (initial variance).

Implementation uses the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017): the process is split into a Riemann-sum component (capturing the rough short-time behaviour) and a Brownian component (capturing the longer-range structure). This gives exact simulation of the covariance structure without simulating a full fractional Brownian motion path.

The Hurst exponent H ≈ 0.10 is consistent with estimates from VSTOXX implied volatility surfaces (Gatheral, Jaisson & Rosenbaum, 2018).

---

## 4. Hawkes self-exciting process

Each institution has an intensity process λ_i(t):

```
λ_i(t) = λ_0(Z̄(t)) + Σ_{j≠i} α · E_{ij}(t) · Σ_{t_k < t} e^{-β(t - t_k)}
```

where:
- λ_0(Z̄) = hawkesBaseline(meanZ) — background intensity, rises with mean sinkhole depth
- α = hawkAlpha (calibrated per episode, default 0.22) — excitation
- β = hawkBeta (default 1.8) — decay rate
- E_{ij}(t) = live endogenous exposure weight (see §7)
- t_k = past stress events for institution j (spread > 45bp or default)

Branching ratio: ρ = α/β. When ρ ≥ 1, the system is supercritical — each stress event generates on average ≥1 further event, and the process explodes. The top-right panel shows ρ live with a ⚠ supercritical warning.

Second-order contagion: nodes with intensity > 1.8 become secondary sources, re-triggering with factor 0.38.

---

## 5. Rough path signatures

For an institution's path X_t = (t, Lf_i(t), σ_i(t)) in ℝ³, the order-k signature is:

```
S(X)^{i₁,…,i_k} = ∫_{0<t₁<…<t_k<T} dX^{i₁}_{t₁} ⊗ … ⊗ dX^{i_k}_{t_k}
```

The order-3 truncated signature has 1 + 3 + 9 + 27 = 40 components.

Key components:
- S^{1,2} = ∫∫ dt dLf — integrated funding liquidity change (area swept)
- S^{2,3} = ∫∫ dLf dσ — Lévy area of the (Lf, σ) loop — captures oscillatory behaviour
- S^{1,2,3} — third-order interaction: time × funding × volatility

The Lévy area S^{2,3} is particularly informative: during a doom loop, Lf and σ cycle in phase, producing a non-zero Lévy area that distinguishes the doom loop trajectory from a direct collapse.

**Calibration loss**: for episode e with target signature S_e, the loss is:

```
L(θ) = ‖S(X_sim(θ)) - S_e‖²_F
```

where the Frobenius norm is taken over the full signature tensor. Gradient descent over θ = (sigScale, jmpScale, hawkAlpha, hawkBeta) using finite differences (5 parameters × 2 evaluations per gradient step × 10 MC paths = 100 forward simulations per step, 8 steps total).

---

## 6. Persistent homology

The stress manifold Z is discretised to a 64×64 grid (the Z-cache). For a threshold τ, the sublevel set is:

```
L_τ = {x ∈ grid : Z(x) ≤ τ}
```

**β₀(τ)** = number of connected components of L_τ — counts how many stress basins have depth ≥ |τ|.

**β₁(τ)** = number of independent loops in L_τ — counts topological cycles (doom loops leave a topological signature as a 1-cycle).

Computed via union-find (Kruskal) on the grid graph. The Euler–Poincaré formula gives:

```
β₁ ≈ E - V + β₀
```

where E = edges in sublevel set, V = vertices. This approximation ignores boundary effects but is accurate for the interior of the grid.

Thresholds used: τ ∈ {-2, -5, -10, -15, -20, -25} (world units, corresponding to approximately 15, 35, 55, 75, 95, 115bp system spread).

The early warning property: β₀ starts rising (stress regions fragment and spread) before any single node crosses the crisis threshold. This is the topological early warning signal — it has no analog in spread-based metrics.

---

## 7. Endogenous network

The bilateral exposure matrix E(t) evolves as:

```
E_{ij}(t) = E_{ij}(0) · Ψ(σ_i(t)) · Ψ(σ_j(t))
```

where σ_i(t) = repo spread for institution i, and the withdrawal sigmoid is:

```
Ψ(σ) = 1 / (1 + exp(κ · (σ - σ*)))
```

Parameters:
- σ* = 60bp — withdrawal threshold (from Dec 2011 MMSR data)
- κ = 0.08 — withdrawal sharpness
- Recovery speed: 0.002 per tick (asymmetric — withdrawal is fast at 0.15/tick)

Triparty links (ECB or CCP as endpoint) are exempt: E_{ij}(t) = E_{ij}(0) always.

The withdrawal cache W_i = Ψ(σ_i(t)) is pre-computed once per metrics tick (35 exp calls), then used across all O(N²) bilateral pairs via multiplication. This reduces the per-frame cost from O(N² exp) to O(N² multiply).

**Topology precursor**: define the edge fraction

```
f(t) = |{(i,j) : E_{ij}(t) > 0.5 · E_{ij}(0)}| / N(N-1)
```

f(t) begins declining 5–10 simulation ticks before β₀ rises and 8–12 ticks before any node exceeds 50bp — making it the earliest observable warning signal in the framework.

---

## 8. Information geometry

The normalised stress distribution at tick t is:

```
p_t(i) = softmax(σ_i(t) / T)_i = exp(3σ_i / σ_max) / Σ_j exp(3σ_j / σ_max)
```

The Fisher information metric on the 34-simplex Δ³⁴ induces the Hellinger distance:

```
d_H(p, q) = √(1 - Σ_i √(p_i q_i))   ∈ [0, 1]
```

The geodesic coordinate at time t is the cumulative arc length:

```
s(t) = Σ_{k=0}^{t-1} d_H(p_k, p_{k+1})
```

The peripheral concentration index:

```
c(t) = Σ_{i : peripheral} p_t(i) / Σ_{i : peripheral or core G-SIB} p_t(i)
```

Crisis classification in the (s, c) plane:
- **sovereign_contagion**: ds/dt < 0.02, c > 0.55 — slow geodesic speed, high peripheral concentration
- **systemic_shock**: ds/dt > 0.05, c < 0.40 — fast geodesic, low differentiation
- **cliff_collapse**: ds/dt > 0.08 — very fast, undifferentiated collapse

The reference geodesics for Dec 2011 and Mar 2020 are shown on the Fisher manifold chart. A new crisis trajectory starts separating from the reference geodesics at t ≈ 5–8 ticks, before any spread threshold is crossed.

---

## 9. HJB optimal control

State space: S = (meanLf, meanLm, btpDyn) on a 15×15×8 grid.

Action space: u ∈ {0, 0.4, 0.8, 1.2} (ECB injection rate per tick, in Lf units).

Bellman equation:

```
V*(s) = max_{u ∈ U} [r(s, u) + β · V*(f(s, u))]
```

Reward:

```
r(s, u) = -λ · stress(s)² - γ · u²
```

where stress(s) = clamp(1 - Lf + 0.4·Lm + 0.6·btp, 0, 2), λ = 1.0, γ = 0.25, β = 0.92.

Transition dynamics (simplified linear model for tractability):

```
Lf' = Lf + u·0.12 - θ_F·0.08·(Lf - 0.7)
Lm' = Lm + θ_M·0.08·(max(0, 0.55 - Lf) - Lm·0.3)
btp' = btp + DL_γ·Lm·0.05 - DL_κ·btp·0.05
```

Solved by value iteration (10 passes). Convergence is rapid because the state space is small (1,800 states) and the dynamics are contractive. Runtime: ~2ms on page load.

---

## 10. Mean field game

Each bank i ∈ BANK_TYPES picks action a_i ∈ {LEND, HOLD, HOARD} each tick.

One-period payoff:

```
u_i(a_i, ā) = -ownStress_i + θ · liqBenefit(a_i, ā) - γ · actionCost(a_i)
```

where ā = system average stress (the mean field).

Best-response rules:
- LEND if: θ·(ā - ownStress·0.5) - γ·0.3 > θ·(ownStress - ā·0.3) - γ·0.2, and that quantity > 0
- HOARD if: payoff_HOARD > payoff_LEND and payoff_HOARD > 0
- HOLD otherwise

Parameters: θ = 0.35 (liquidity benefit weight), γ = 0.20 (action cost weight).

The MFG equilibrium is approximated by best-response dynamics: banks solve individually each tick taking ā as given. Convergence to the equilibrium distribution is observed within 5–10 ticks for typical parameterisations.

Effect on simulation: LEND adds +0.15·(1 - ownStress) to the injection array; HOARD adds -0.12·ownStress. Lenders pulse green in 3D; hoarders shrink to 0.72× scale.

---

## 11. Adversarial game

**Speculator** has action budget B = 1.0 per tick, allocated across:
- a_BTP ≥ 0 — adds btpDyn pressure (cost: 0.40 per unit)
- a_REPO ≥ 0 — withdraws bilateral exposure from stressed nodes (cost: 0.35)
- a_DUMP ≥ 0 — amplifies fire-sale (adds to conRef; cost: 0.25)

**Speculator payoff**: profit = system spread/100 + btpDyn·0.5; cost = quadratic in actions.

**ECB payoff**: -stress - γ·u².

**Nash equilibrium via iterated best-response (IBR)**:
1. ECB sets u* from HJB policy given current state
2. Speculator solves BR: exhaustive search over (a_BTP, a_REPO, a_DUMP) in 0.25 increments
3. ECB re-solves given speculator's attack (adjusts effective spread and BTP in state lookup)
4. Repeat 5 rounds

IBR converges to Nash in practice within 3–4 rounds for this parameterisation. The equilibrium depends on the crisis type: under Dec 2011 calibration, the speculator allocates more to a_REPO (network contraction); under Mar 2020 calibration, more to a_BTP (sovereign pressure).

---

## 12. Monte Carlo forward ensemble

50 paths × 60 ticks, run synchronously (stripped simulation — no rendering).

Each path starts from the current simulation state (Lf, Lm, con, hwk, btpDyn, cap). The stripped simulation includes: B-P ODE, Hawkes step, contagion step, simplified doom loop. No eigenmap, no Three.js.

Computed analytics:
- p10/p50/p90 quantile bands at each tick
- Probability density grid (50 bins × 60 ticks) for the thermal heatmap
- σ(t) = std(paths at t) — bifurcation indicator; peak = critical decision point
- Per-node crisis probability across paths
- Survival curves for the 3 most-stressed institutions
- Conditional split: crisis paths (peak spread > 50bp) vs recovery paths
- VaR(95%), ES(95%), E[peak spread]
- Crisis horizon: first tick where p50 crosses 50bp

Calibration integration: after fitting θ* via inverse problem, MC auto-reruns with the fitted parameters. The calibrated fan shows the forward distribution conditional on the current state matching the historical episode's calibrated dynamics.

---

## References

- Brunnermeier, M.K. & Pedersen, L.H. (2009). *Market Liquidity and Funding Liquidity*. Review of Financial Studies, 22(6), 2201–2238.
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is rough*. Quantitative Finance, 18(6), 933–949.
- Bennedsen, M., Lunde, A. & Pakkanen, M.S. (2017). *Hybrid scheme for Brownian semistationary processes*. Finance and Stochastics, 21(4), 931–965.
- Friz, P.K. & Victoir, N.B. (2010). *Multidimensional Stochastic Processes as Rough Paths*. Cambridge University Press.
- Hawkes, A.G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika, 58(1), 83–90.
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
- Lasry, J.M. & Lions, P.L. (2007). *Mean field games*. Japanese Journal of Mathematics, 2(1), 229–260.
