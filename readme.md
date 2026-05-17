A real time stress simulation of the euro denominated repo market, rendered as an interactive 3D manifold in the browser. Built as a research tool alongside work on rough path methods in financial risk.

The Core Concept

The European interbank repo market is the plumbing of the financial system, handling 2 to 4 trillion euros in overnight secured lending every day. When it jams (as seen in December 2011 or March 2020), credit freezes and the ECB has days, not weeks, to respond. Standard risk models struggle here because they treat each time step independently. They fail to capture the structural shape of how a crisis unfolds, whether it is a slow peripheral contagion or a sudden systemic cliff edge.
This simulation treats the market as a continuous landscape. 35 core institutions (the ECB, major European G SIBs, dealers, CCPs, money market funds, pension funds, and sovereign collateral markets) act as nodes on a surface. Systemic stress appears as sinkholes forming in real time. The geometry of those sinkholes, how they grow and connect, varies by crisis type, and that difference is mathematically meaningful.
The engine blends stochastic control, rough path theory, network topology, and information geometry to surface insights that separate models keep hidden.
Mathematical Framework
* Brunnermeier Pedersen Liquidity Spiral: The core engine. Funding illiquidity $L_f$ and market illiquidity $L_m$ are coupled. When margins rise, forced selling occurs, prices fall, and margins rise further. This system of ordinary differential equations drives the real time sinkhole formation on the manifold.
* Hawkes Self Exciting Contagion: Stress propagates across institutions via a point process where intensity depends on past events. When the branching ratio $\rho = \alpha / \beta$ crosses 1, the system enters a self sustaining cascade. This step updates using the live exposure matrix, meaning network contraction directly amplifies contagion.
* Rough Bergomi Volatility: VSTOXX dynamics use a rough Bergomi process with a Hurst exponent of $H \approx 0.10$ instead of Heston. This matches empirical options market data and makes the model non Markovian, meaning volatility today depends on the entire history of past shocks rather than just yesterday's level.
* Rough Path Signatures: An order 3 truncated signature of each institution's stress path $(t, L_f, \sigma_i)$ captures the geometric trajectory as a tensor. This distinguishes a fast exogenous shock from a slow self reinforcing contagion even if they reach the exact same endpoint. Used in the live signature panel and inverse problem calibration.
* Endogenous Network: Bilateral repo links contract via a sigmoid withdrawal function when counterparty stress crosses a threshold ($\sigma^* \approx 60$ basis points, derived from December 2011 data). Triparty links backed by the ECB or CCPs stay open regardless. Network topology becomes a leading indicator as edges disappear before nodes cross the crisis threshold, visible early via the Betti number $\beta_0$.
* HJB Optimal ECB Policy: The central bank intervention is modeled as a discounted infinite horizon control problem over the $(mean L_f, mean L_m, btpDyn)$ state space. Value iteration runs on a 15 by 15 by 8 grid in roughly 2 milliseconds to show recommended policy surfaces.
* Mean Field Game: Banks choose strategies (lend, hold, or hoard) by best responding to the system average stress. Lending reduces system stress but raises individual risk. The equilibrium is approximated via best response dynamics with live visual indicators on each node.
* Adversarial Equilibrium: A speculator plays against the ECB using attack vectors like shorting BTP futures, withdrawing bilateral repo, or dumping collateral. The ECB responds with optimal injection. Iteration finds the Nash equilibrium within 5 rounds every single tick.
* Persistent Homology: Computes topological invariants $\beta_0$ (connected stress components) and $\beta_1$ (loops or doom cycles) from sublevel sets of the data cache. A rise in $\beta_0$ before nodes hit alert thresholds serves as a geometric early warning signal.
* Information Geometry: The 35 node stress distribution acts as a point on the probability simplex $\Delta^{34}$. Hellinger distances between consecutive distributions give a geodesic coordinate in the Fisher metric, allowing crisis types to separate geometrically before major thresholds break.
* Sovereign Doom Loop: Bank stress widens BTP spreads, eroding bank capital and triggering more stress. A secondary fiscal channel models output gap drops leading to lower tax revenue and wider deficits.
Market Structure
Sovereign exposure is calibrated from ECB supervisory disclosures. Balance sheet sizes are derived from public filings.
Historical Scenarios
* Generic Baseline: Parametric baseline using interactive sliders to test custom anomalies.
* September 2019 Repo Spike: US Treasury settlement combined with quantitative tightening liquidity drains hitting quarter end simultaneously.
* March 2020 Dash for Cash: Rapid systemic COVID shock featuring low peripheral differentiation and high initial velocity.
* December 2011 LTRO 1: Slow peripheral sovereign contagion dominated by the doom loop, eventually stabilized by the Long Term Refinancing Operation.
Getting Started
Bash
git clone https://github.com/username/ecb_repo_terminal
cd ecb_repo_terminal
npm install
npm run dev
Requires Node 18 or higher. Runs entirely client side in the browser with no external server or API dependencies.
Terminal Controls
* Top Left Panel: Episode selection, playback controls, timeline scrubber, and JSON export. Toggle states include Split (layer separation), Decomp (RGB stress attribution), Synth (synthesis panel), and MC (Monte Carlo paths).
* Top Right Panel: Live metrics tracking ESTR, system spread, fragility index, Hawkes intensity, BTP premium, collateral velocity, Kyle's Lambda, output gap, and MMF portfolio NAV.
* Bottom Left Panel: Macro time series for output gaps, credit flows, and active fiscal doom loops.
* Node Hover Dossier: Displays balance sheet size, sovereign exposure, repo spread, Merton Distance to Distress, Rough Bergomi variance, haircut metrics, and rehypothecation chain depth.
* Synthesis Panel: Features ten detailed sections covering network fragmentation gauges, Fisher manifold trajectories, signature gradient descent for inverse problems, Nash equilibrium tracking, HJB policy surfaces, and live persistent homology calculations.
* Monte Carlo Card: Shows canvas rendered thermal probability density heatmaps, tail risk metrics (VaR 95% and Expected Shortfall 95%), path bifurcations, and conditional survival curves.
Calibration Notes
The inverse problem optimization fits Hawkes parameters and volatility scaling via finite difference gradient descent over a signature distance loss function. Running 8 gradient steps with 10 Monte Carlo paths per evaluation highlights stark differences: the December 2011 simulation reveals high self excitation ($\alpha \approx 0.28$), while the March 2020 run shows high jump scaling but lower self excitation ($\alpha \approx 0.18$). This confirms the mathematical distinction between an endogenous cascade and an exogenous shock.
The initial configuration utilizes public data from the Q1 2025 ECB Statistical Data Warehouse snapshot, incorporating weekly excess liquidity (€3850bn), ESTR (2.391%), and supervisor disclosed sovereign exposures.

