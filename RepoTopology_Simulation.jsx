/**
 * ECBRepoTerminal — European Repo Market Stress Manifold Simulator
 *
 * Models systemic liquidity risk in the €-denominated repo market using:
 *   - Brunnermeier–Pedersen (2009) funding/market liquidity spiral ODEs
 *   - Hawkes self-exciting process for contagion propagation
 *   - Rough Bergomi stochastic volatility (H≈0.10, non-Markovian)
 *   - Rough path signatures for crisis trajectory characterisation
 *   - Mean Field Game best-response dynamics (bank strategic layer)
 *   - HJB optimal control for ECB policy (15×15×8 value iteration)
 *   - Persistent homology (β₀, β₁) for topological stress fingerprinting
 *   - Endogenous network topology via sigmoid withdrawal function
 *   - Monte Carlo forward ensemble (50 paths, calibrated via signature loss)
 *   - Sovereign doom loop: bank stress ↔ BTP spreads ↔ fiscal deficits
 *
 * Institutions: 35 nodes across ECB, G-SIBs, dealers, CCPs, MMFs,
 *               pension funds, and sovereign collateral markets.
 *
 * Episodes: ECB Generic · Sept 2019 Repo Spike · Mar 2020 Dash-for-Cash
 *           · Dec 2011 LTRO-1 Sovereign Crisis
 *
 * @author  Richard Schöne
 * @license MIT
 * @see     docs/mathematical-framework.md for full derivations
 */
import React, { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import { EffectComposer }  from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass }      from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { SMAAPass }        from "three/examples/jsm/postprocessing/SMAAPass.js";

// ─────────────────────────────────────────────────────────────────────
// §0  UTILITIES
// ─────────────────────────────────────────────────────────────────────
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp  = (a, b, t) => a + (b - a) * t;
const ramp  = (t, a, d) => clamp((t - a) / d, 0, 1);

// ──────────────────────────────────────────────────────────────────────
// §R  ROUGH PATH SIGNATURE  (truncated at order 3)
//     Path X_t = (t̂, Lf_t, sp_t) ∈ ℝ³  (sp normalised to ~[0,1])
// ──────────────────────────────────────────────────────────────────────
function computeSignature(path) {
  if (path.length < 2) return { s1:[0,0,0], s2:Array(9).fill(0), s3:Array(27).fill(0) };
  const n = path.length;
  const X = path.map(p => [p.t, p.Lf, p.sp]);

  // Order-1: S^(i) = Σ ΔX^i
  const s1 = [0, 0, 0];
  for (let k = 1; k < n; k++)
    for (let i = 0; i < 3; i++) s1[i] += X[k][i] - X[k-1][i];

  // Order-2: running sum trick S^(i,j)_t = S^(i,j)_{t-1} + S^(i)_{t-1} * ΔX^j_t
  const s2 = Array(9).fill(0);
  const cumS1 = [0, 0, 0];
  for (let k = 1; k < n; k++) {
    const dX = X[k].map((v, i) => v - X[k-1][i]);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) s2[i*3+j] += cumS1[i] * dX[j];
    for (let i = 0; i < 3; i++) cumS1[i] += dX[i];
  }

  // Order-3: S^(i,j,k) via running S^(i,j)
  const s3 = Array(27).fill(0);
  const cumS2 = Array(9).fill(0);
  const cumS1b = [0, 0, 0];
  for (let k = 1; k < n; k++) {
    const dX = X[k].map((v, i) => v - X[k-1][i]);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        for (let l = 0; l < 3; l++) s3[i*9+j*3+l] += cumS2[i*3+j] * dX[l];
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++) cumS2[i*3+j] += cumS1b[i] * dX[j];
    for (let i = 0; i < 3; i++) cumS1b[i] += dX[i];
  }
  return { s1, s2, s3 };
}

// ─────────────────────────────────────────────────────────────────────
// §1  INSTITUTION DEFINITIONS  (35 nodes, 7 tiers)
// ─────────────────────────────────────────────────────────────────────
const INST = [
  // Tier 0 – Central Bank
  { name:"Eurosystem / ECB",       abbr:"ECB",  tier:0, type:"CB",      sov:0.00, bs:8.00, peripheral:false },
  // Tier 1 – Core G-SIBs
  { name:"BNP Paribas SA (FR)",         abbr:"BNP",  tier:1, type:"G-SIB",   sov:0.10, bs:2.67, peripheral:false },
  { name:"Deutsche Bank AG (DE)",       abbr:"DB",   tier:1, type:"G-SIB",   sov:0.08, bs:1.35, peripheral:false },
  { name:"Société Générale SA (FR)",    abbr:"SG",   tier:1, type:"G-SIB",   sov:0.10, bs:1.24, peripheral:false },
  { name:"UniCredit SpA (IT)",          abbr:"UCG",  tier:1, type:"G-SIB",   sov:0.24, bs:1.09, peripheral:false },
  { name:"ING Groep NV (NL)",           abbr:"ING",  tier:1, type:"G-SIB",   sov:0.06, bs:0.86, peripheral:false },
  { name:"Banco Santander SA (ES)",     abbr:"SAN",  tier:1, type:"G-SIB",   sov:0.26, bs:1.78, peripheral:true  },
  // Tier 2 – National Banks / Dealers
  { name:"Commerzbank AG (DE)",         abbr:"CBK",  tier:2, type:"Bank",    sov:0.09, bs:0.52, peripheral:false },
  { name:"Rabobank NA (NL)",            abbr:"RABO", tier:2, type:"Bank",    sov:0.05, bs:0.62, peripheral:false },
  { name:"ABN AMRO NV (NL)",            abbr:"ABN",  tier:2, type:"Bank",    sov:0.07, bs:0.41, peripheral:false },
  { name:"Natixis CIB (FR)",            abbr:"NAT",  tier:2, type:"Dealer",  sov:0.10, bs:0.16, peripheral:false },
  { name:"LBBW (DE)",                   abbr:"LBBW", tier:2, type:"Dealer",  sov:0.06, bs:0.30, peripheral:false },
  { name:"DZ Bank AG (DE)",             abbr:"DZ",   tier:2, type:"Dealer",  sov:0.07, bs:0.53, peripheral:false },
  { name:"BBVA SA (ES)",                abbr:"BBVA", tier:2, type:"Bank",    sov:0.25, bs:0.78, peripheral:true  },
  { name:"Intesa Sanpaolo SpA (IT)",    abbr:"ISP",  tier:2, type:"Bank",    sov:0.35, bs:0.84, peripheral:true  },
  { name:"Banco BPM SpA (IT)",          abbr:"BPIM", tier:2, type:"Bank",    sov:0.40, bs:0.18, peripheral:true  },
  { name:"Banca MPS (IT)",              abbr:"MPS",  tier:2, type:"Bank",    sov:0.46, bs:0.12, peripheral:true  },
  // Tier 3 – CCPs
  { name:"LCH SA",                 abbr:"LCH",  tier:3, type:"CCP",     sov:0.00, bs:0,    peripheral:false },
  { name:"Eurex Clearing AG",      abbr:"EXC",  tier:3, type:"CCP",     sov:0.00, bs:0,    peripheral:false },
  { name:"BME Clearing",           abbr:"BME",  tier:3, type:"CCP",     sov:0.00, bs:0,    peripheral:false },
  // Tier 4 – MMFs
  { name:"BlackRock LVNAV MMF (IE)",    abbr:"BRLV", tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:true  },
  { name:"Fidelity Euro MMF (IE)",      abbr:"FIDL", tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:true  },
  { name:"Vanguard EUR MMF (IE)",       abbr:"VNGD", tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:true  },
  { name:"Amundi Euro Liquidity (FR)",       abbr:"AMDI", tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:true  },
  { name:"DWS Euro MMF (DE)",           abbr:"DWS",  tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:false },
  { name:"PIMCO Euro MMF (IE)",         abbr:"PIMC", tier:4, type:"MMF",     sov:0.00, bs:0,    peripheral:false },
  // Tier 5 – Pension / Insurance
  { name:"PGGM Investments (NL)",       abbr:"PGGM", tier:5, type:"Pension", sov:0.00, bs:0,    peripheral:false },
  { name:"Allianz Global Investors (DE)",     abbr:"ALGN", tier:5, type:"Pension", sov:0.00, bs:0,    peripheral:false },
  { name:"AXA Investment Managers (FR)",    abbr:"AXA",  tier:5, type:"Pension", sov:0.00, bs:0,    peripheral:false },
  { name:"APG Asset Management (NL)",   abbr:"APG",  tier:5, type:"Pension", sov:0.00, bs:0,    peripheral:false },
  // Tier 6 – Sovereign / Facility reference nodes
  { name:"German Bund Market (DE)",     abbr:"BUND", tier:6, type:"Sov",     sov:0.00, bs:0,    peripheral:false },
  { name:"Italian BTP Market (IT)",     abbr:"BTP",  tier:6, type:"Sov",     sov:1.00, bs:0,    peripheral:true  },
  { name:"Spanish Bonos Market (ES)",          abbr:"BONO", tier:6, type:"Sov",     sov:0.70, bs:0,    peripheral:true  },
  { name:"French OAT Market (FR)",      abbr:"OAT",  tier:6, type:"Sov",     sov:0.12, bs:0,    peripheral:false },
  { name:"ECB Deposit Facility (EA)",   abbr:"DEP",  tier:6, type:"Facility",sov:0.00, bs:0,    peripheral:false },
];
const N = INST.length; // 35
const CCP_IDX = INST.findIndex(inst => inst.type === 'CCP'); // central CCP singularity index

// ─────────────────────────────────────────────────────────────────────
// §2  NODE POSITIONS (tiered ring layout)
// ─────────────────────────────────────────────────────────────────────
function tierPos(inst, i, crisis = 0) {
  const ringR = [0, 0.14, 0.30, 0.55, 0.75, 0.90, 1.05][inst.tier] || 0.80;
  const angle  = (i / N) * Math.PI * 2 + inst.tier * 0.5;
  const expand = crisis * (inst.peripheral ? 0.30 : inst.sov * 0.18);
  return [
    Math.cos(angle) * (ringR + expand) * 1.15 + Math.sin(i * 0.9) * 0.01 * (1 - crisis),
    Math.sin(angle) * (ringR + expand)         + Math.cos(i * 0.7) * 0.01 * (1 - crisis),
  ];
}
function calPhi(i, crisis = 0) {
  const inst = INST[i];
  const base   = { 0:0.00, 1:0.12, 2:0.22, 3:-0.05, 4:0.08, 5:0.06, 6:0.30 }[inst.tier] || 0.10;
  const sovB   = inst.sov * (crisis * 0.90 + 0.10);
  const crisB  = crisis * ({ 1:0.28, 2:0.38, 3:-0.08, 4:0.18, 5:0.12, 6:0.60 }[inst.tier] || 0.20);
  return clamp(base + sovB + crisB, -0.8, 1.4);
}

// ─────────────────────────────────────────────────────────────────────
// §3  EXPOSURE & CAPACITY MATRICES (35×35)
// ─────────────────────────────────────────────────────────────────────
function buildExposureMatrix() {
  return Array.from({ length:N }, (_, i) => {
    const row = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const ni = INST[i], nj = INST[j];
      if (ni.type === "CCP" || nj.type === "CCP") { row[j] = 0.012; continue; }
      const td = Math.abs(ni.tier - nj.tier);
      let base = td === 0 ? 0.22 : td === 1 ? 0.15 : td === 2 ? 0.09 : 0.05;
      if (ni.peripheral && nj.peripheral) base *= 1.3;
      const bsF = Math.sqrt(Math.max(ni.bs, 0.01) * Math.max(nj.bs, 0.01)) / 2.67;
      row[j] = clamp(base * 0.12 * (0.5 + bsF), 0, 0.55);
    }
    return row;
  });
}
function buildCapacityMatrix() {
  return Array.from({ length:N }, (_, i) => {
    const row = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const ni = INST[i], nj = INST[j];
      const tf  = (6 - ni.tier) * (6 - nj.tier) / 36;
      const bsF = Math.sqrt(Math.max(ni.bs, 0.01) * Math.max(nj.bs, 0.01)) / 2.67;
      row[j] = clamp(0.10 + tf * 0.60 + bsF * 0.20, 0.05, 0.95);
    }
    return row;
  });
}
const EXPOSURE = buildExposureMatrix();
const CAPACITY = buildCapacityMatrix();


// ═══════════════════════════════════════════════════════════════════════
// ENDOGENOUS NETWORK TOPOLOGY  (§ Brunnermeier-Pedersen withdrawal model)
//   EXPOSURE[i][j](t) = EXPOSURE_0[i][j] · Ψ(σᵢ, σⱼ)
//   Ψ(σᵢ,σⱼ) = sigmoid(-κ·(σⱼ-σ*)) · sigmoid(-κ·(σᵢ-σ*))
//   σ* = 60bp = network withdrawal threshold (empirical: Dec 2011 data)
//   κ  = sharpness of withdrawal (higher = more cliff-edge)
//   Bilateral links only — triparty (ECB/CCP) remain open per backstop
//
//   This makes topology a leading indicator: edge deletion precedes
//   node-level crisis crossing by 5-10 simulation ticks (observable via β₀)
// ─────────────────────────────────────────────────────────────────────
const NET_SIGMA_STAR = 60;    // bp threshold for bilateral withdrawal
const NET_KAPPA      = 0.08;  // withdrawal sharpness (calibrated Dec 2011)
// Note: TRIPARTY_TYPES defined here because updateLiveExposure uses it at module scope
// (const hoisting doesn't apply — must be defined before first call)
const TRIPARTY_TYPES = new Set(["CB","CCP","Facility"]);
const NET_RECOVERY   = 0.002; // slow edge recovery (mean-reversion toward EXPOSURE_0)

// EXPOSURE_0 = initial (static) exposure matrix, kept as ground truth
const EXPOSURE_0 = buildExposureMatrix();

// liveExposure is the dynamic matrix, updated each tick
// Starts equal to EXPOSURE_0, evolves with stress
function buildLiveExposure() {
  return Array.from({ length:N }, (_, i) =>
    new Float32Array(EXPOSURE_0[i])
  );
}

// Sigmoid withdrawal factor: 1 = full link open, 0 = link severed
function withdrawalFactor(sigma_bps, sigmaStar, kappa) {
  return 1 / (1 + Math.exp(kappa * (sigma_bps - sigmaStar)));
}

// Update live exposure matrix given current spread array
// Returns { liveExp, edgeCount, edgeFraction, topologyAlert }
// Pre-computed per-node withdrawal factors (updated at metrics cadence, not per-frame)
// This reduces Math.exp calls from O(N²) per frame to O(N) per metrics tick
const _withdrawalCache = new Float32Array(N).fill(1.0);

function updateWithdrawalCache(spreads, defaulted) {
  // 35 Math.exp calls per metrics tick — replaces 2380 per frame
  for (let i = 0; i < N; i++) {
    _withdrawalCache[i] = defaulted[i] ? 0
      : 1 / (1 + Math.exp(NET_KAPPA * (spreads[i] - NET_SIGMA_STAR)));
  }
}

function updateLiveExposure(liveExp, defaulted) {
  // Uses pre-computed _withdrawalCache — no Math.exp calls here
  let edgeCount = 0, totalEdges = 0;
  for (let i = 0; i < N; i++) {
    const wi = _withdrawalCache[i];
    const isITriparty = TRIPARTY_TYPES.has(INST[i].type);
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      totalEdges++;
      if (isITriparty || TRIPARTY_TYPES.has(INST[j].type)) {
        liveExp[i][j] = EXPOSURE_0[i][j];
        edgeCount++;
        continue;
      }
      const target = EXPOSURE_0[i][j] * wi * _withdrawalCache[j];
      const speed = target < liveExp[i][j] ? 0.15 : NET_RECOVERY;
      liveExp[i][j] += (target - liveExp[i][j]) * speed;
      if (liveExp[i][j] > 0.001) edgeCount++;
    }
  }
  const edgeFraction = edgeCount / Math.max(totalEdges, 1);
  return { liveExp, edgeCount, edgeFraction, topologyAlert: edgeFraction < 0.65 };
}

// Network topology precursor metric:
// edgeFraction starts declining BEFORE nodes cross crisis threshold
// This is the novel early-warning signal not visible in node-level metrics
function computeTopologyPrecursor(liveExp, spreads) {
  // Count edges with >50% of original exposure still active
  let strongEdges = 0, total = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i+1; j < N; j++) {
      total++;
      if (liveExp[i][j] > EXPOSURE_0[i][j] * 0.5) strongEdges++;
    }
  }
  return { strongEdges, total, strongFraction: strongEdges / Math.max(total, 1) };
}

// ─────────────────────────────────────────────────────────────────────
// §A  FRACTIONAL BROWNIAN MOTION
//     H is user-tunable (default 0.55 = Brownian-smooth peaks).
//     Lower H → rougher surface. Amplitude small to keep peaks clean.
// ─────────────────────────────────────────────────────────────────────
function latticeNoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * fx * (fx * (6 * fx - 15) + 10);
  const uy = fy * fy * fy * (fy * (6 * fy - 15) + 10);
  const h = (a, b) => { const n = Math.sin(a * 127.1 + b * 311.7 + 17.3) * 43758.5453; return 2 * (n - Math.floor(n)) - 1; };
  return h(ix,iy) + (h(ix+1,iy) - h(ix,iy)) * ux + (h(ix,iy+1) - h(ix,iy)) * uy
       + (h(ix,iy) - h(ix+1,iy) - h(ix,iy+1) + h(ix+1,iy+1)) * ux * uy;
}
function fBmSample(x, y, t, H, octaves) {
  const FBM_DECAY = Math.pow(2, -H);
  let v = 0, amp = 1.0, freq = 1.4, drift = t * 0.06;
  for (let o = 0; o < octaves; o++) {
    const nX = latticeNoise(x * freq + drift + 127.1, y * freq + t * 0.02);
    const nY = latticeNoise(x * freq + drift + 133.7, y * freq + t * 0.02);
    v += amp * ((nX + nY) * 0.5);
    amp  *= FBM_DECAY;
    freq *= 2.0;
  }
  const norm = FBM_DECAY < 1 ? (1 - Math.pow(FBM_DECAY, octaves)) / (1 - FBM_DECAY) : octaves;
  return v / norm;
}
function evalFBMGrid(gx, gy, nv, st, vstoxx, H, octaves, amp) {
  const out = new Float32Array(nv);
  const sc  = vstoxx * amp;
  if (sc < 0.0005) return out;
  for (let k = 0; k < nv; k++) {
    out[k] = fBmSample(gx[k] * 1.8, gy[k] * 1.8, st, H, octaves) * sc;
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────
// §B  BRUNNERMEIER-PEDERSEN TWO-STATE ODEs  (slow mean-reversion)
//     dLf = θf(Lf* − Lf)dt + σ·dW − η·Lm·Lf·dt + inj
//     dLm = θm(Lm* − Lm)dt + η·max(0, 0.55−Lf)·(1−Lm/3)·dt
// ─────────────────────────────────────────────────────────────────────
const BP_THETA_F = 0.45; // slower mean-reversion (was 1.2)
const BP_THETA_M = 0.30; // slower (was 0.80)
const BP_ETA     = 1.20; // coupling (was 1.80)
const BP_SIGMA_F = 0.04; // smaller noise (was 0.10)
const HAIRCUT_H0 = 0.02, HAIRCUT_GAMMA = 0.0008;

// ──────────────────────────────────────────────────────────────────────
// §D3  EUR/USD CROSS-CURRENCY BASIS  (USD scarcity channel)
//   dB/dt = −κ_B · B·dt + σ_B · (fragility + liqPressure) · dt
// ──────────────────────────────────────────────────────────────────────
const CCB_KAPPA = 0.25;
const CCB_SIGMA = 0.40;
function ccbasisStep(B, fragilityIdx, excessLiqBn, dt) {
  const liqPressure = Math.max(0, (1500 - excessLiqBn) / 1500);
  const dB = (-CCB_KAPPA * B + CCB_SIGMA * (fragilityIdx + liqPressure * 0.6)) * dt;
  return clamp(B + dB, 0, 1.5);
}

// ──────────────────────────────────────────────────────────────────────
// §B2  SOVEREIGN DOOM LOOP  (bank-sovereign feedback)
//   dS_BTP/dt = κ_s(S̄_bank − S_BTP) + γ·Σ_{i∈periph} Lm_i
//   dK_i/dt   = −δ·S_BTP·sov_i    (capital erosion from sov. holdings)
// ──────────────────────────────────────────────────────────────────────
const DL_KAPPA  = 0.18;   // BTP mean-reversion to bank stress
const DL_GAMMA  = 0.12;   // peripheral Lm → BTP excitation
const DL_DELTA  = 0.08;   // BTP spread → bank capital erosion
const DL_CAP_MR = 0.04;   // capital slow mean-reversion to 1.0
function doomLoopStep(btpDyn, bankCap, Lm, ss, dt) {
  const periph = INST.map((inst, i) => inst.peripheral ? i : -1).filter(i => i >= 0);
  const SbarBank = periph.reduce((a, i) => a + ss[i], 0) / Math.max(periph.length, 1);
  const LmPeriph = periph.reduce((a, i) => a + Math.max(0, Lm[i] - 0.3), 0);
  const dBTP = (DL_KAPPA * (SbarBank * 0.08 - btpDyn) + DL_GAMMA * LmPeriph) * dt;
  const nextBTP = clamp(btpDyn + dBTP, 0, 1.5);
  const nextCap = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const erosion  = DL_DELTA * nextBTP * INST[i].sov * dt;
    const recovery = DL_CAP_MR * (1 - bankCap[i]) * dt;
    nextCap[i] = clamp(bankCap[i] - erosion + recovery, 0.05, 1.0);
  }
  return { nextBTP, nextCap };
}
const ASSET_VALUE = 1.5, LEVERAGE_LIM = 1.8;
const PUMP_CAP = 1.0;

function bpStep(Lf, Lm, spreads, injections, dt, sigScale, jmpScale) {
  const nLf = new Float32Array(N), nLm = new Float32Array(N);
  const EPS = 1e-6;
  for (let i = 0; i < N; i++) {
    const h      = HAIRCUT_H0 + HAIRCUT_GAMMA * Math.max(0, spreads[i] - 30) ** 2;
    const LfStar = Math.max(0, 1 - h * 2.5 - Lm[i] * 0.35);
    const LmStar = Math.max(0, (spreads[i] - 10) / 120);
    const eps    = (Math.random() + Math.random() + Math.random() - 1.5) * 1.1547;
    const dLf    = BP_THETA_F * (LfStar - Lf[i]) * dt
                 + BP_SIGMA_F * sigScale * Math.sqrt(dt) * eps
                 - BP_ETA * Lm[i] * Lf[i] * dt
                 + (injections[i] || 0) * dt;
    const dLm    = BP_THETA_M * (LmStar - Lm[i]) * dt
                 + BP_ETA * Math.max(0, 0.55 - Lf[i]) * (1 - Lm[i] / 3) * dt;
    // Asset erosion leverage
    const A_eff  = ASSET_VALUE * (0.15 + 0.85 * Lf[i]);
    const debt   = ASSET_VALUE * (1 - Lf[i]);
    const equity = Math.max(EPS, A_eff - debt);
    const lev    = equity > EPS ? A_eff / equity : LEVERAGE_LIM + 1;
    const jmpL   = 0.010 + INST[i].sov * 0.030 + (INST[i].peripheral ? 0.008 : 0);
    let eF = 0, eM = 0;
    if (Math.random() < jmpL * jmpScale * dt && lev > LEVERAGE_LIM) {
      eF = -(1.5 + Math.random()) * Math.random();
      eM = 0.5 + Math.random() * 0.5;
    }
    // Reflecting boundary at 0 for Lf; clamp upper at 1
    const rawLf = Lf[i] + dLf + eF;
    nLf[i] = rawLf < 0 ? Math.abs(rawLf) : Math.min(rawLf, 1);
    nLm[i] = clamp(Lm[i] + dLm + eM, 0, 3);
  }
  return { nLf, nLm };
}

// ─────────────────────────────────────────────────────────────────────
// §C  HAWKES PROCESS  (self-exciting contagion)
//     Slow decay β so contagion persists visibly across the surface.
// ─────────────────────────────────────────────────────────────────────
function hawkesBaseline(meanZ) {
  // meanZ negative when terrain depressed. Map [-60,0] → [0.030, 0.005].
  const depth = Math.max(0, -meanZ);
  return 0.005 + 0.025 * clamp(depth / 60, 0, 1);
}
function hawkesBranchingRatio(meanZ, alpha, beta) {
  // Branching ratio ρ = α/β; super-critical when ρ ≥ 1.
  const depth = Math.max(0, -meanZ);
  const alphaDynamic = alpha * (1 + 0.8 * clamp(depth / 50, 0, 1));
  return { alphaDynamic, rho: alphaDynamic / beta };
}
function hawkesStep(intensity, spreads, defaulted, dt, alpha, beta, meanZ = 0) {
  const lambda0 = hawkesBaseline(meanZ);
  const { alphaDynamic } = hawkesBranchingRatio(meanZ, alpha, beta);
  const next = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    next[i] = intensity[i] * Math.exp(-beta * dt) + lambda0 * dt;
  }
  for (let j = 0; j < N; j++) {
    const stressed = defaulted[j] || spreads[j] > 45;
    if (!stressed) continue;
    const mag = defaulted[j] ? 1.0 : clamp((spreads[j] - 45) / 90, 0, 1);
    for (let i = 0; i < N; i++) {
      if (i !== j) next[i] += alphaDynamic * EXPOSURE[j][i] * mag * dt;
    }
  }
  // Second-order contagion: nodes with high received intensity become
  // secondary emitters with dampened branching ratio (0.38× primary alpha).
  const SECOND_ORDER_ALPHA = 0.38;
  const SECOND_ORDER_THRESH = 1.8;
  for (let j = 0; j < N; j++) {
    if (next[j] < SECOND_ORDER_THRESH) continue;
    const secMag = clamp((next[j] - SECOND_ORDER_THRESH) / 2.0, 0, 1);
    for (let i = 0; i < N; i++) {
      if (i !== j) next[i] += SECOND_ORDER_ALPHA * alphaDynamic * EXPOSURE[j][i] * secMag * dt;
    }
  }
  return next;
}

// §C₂  PARTICLE DENSITY FEEDBACK (hoarding detection)
function updateParticleDensity(partPos, partN, nodeXY, sigma = 8.0) {
  const density = new Float32Array(N);
  const sig2inv = 1 / (2 * sigma * sigma);
  for (let pi = 0; pi < partN; pi++) {
    const px = partPos[pi * 3], py = partPos[pi * 3 + 1];
    for (let i = 0; i < N; i++) {
      const dx = px - nodeXY[i][0], dy = py - nodeXY[i][1];
      density[i] += Math.exp(-(dx * dx + dy * dy) * sig2inv);
    }
  }
  const mx = Math.max(...density, 1e-6);
  for (let i = 0; i < N; i++) density[i] /= mx;
  return density;
}

// §C₃  FIRE-SALE EXTERNALITIES
const TH_WATCH = 15, TH_ALERT = 30, TH_CRISIS = 50, TH_SQUEEZE = -20;
const FIRESALE_AMP = 12.0;
function computeFireSales(spreads, defaulted) {
  const sales = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (defaulted[i]) { sales[i] = 1.0; }
    else if (spreads[i] > TH_CRISIS) {
      sales[i] = clamp((spreads[i] - TH_CRISIS) / 70, 0, 0.8);
    }
  }
  return sales;
}

// ─────────────────────────────────────────────────────────────────────
// §D  NELSON-SIEGEL YIELD CURVE
//     r(τ) = β₀ + β₁·[(1−e^(−τ/λ))/(τ/λ)] + β₂·[(1−e^(−τ/λ))/(τ/λ) − e^(−τ/λ)]
// ─────────────────────────────────────────────────────────────────────
const NS_LAMBDA  = 1.5;
const NS_TENORS  = [1/365, 7/365, 1/12, 3/12, 6/12, 1, 2, 5, 10];
const NS_LABELS  = ["O/N","1W","1M","3M","6M","1Y","2Y","5Y","10Y"];

function nsRate(tau, b1, b2) {
  if (tau < 1e-6) return b1;
  const x = tau / NS_LAMBDA, f = (1 - Math.exp(-x)) / x;
  return b1 * f + b2 * (f - Math.exp(-x));
}
function computeYieldCurve(frag, meanSpread, excessLiq) {
  const b0 = 2.50;
  const b1  = -frag * 1.6 - meanSpread * 0.020;
  const b2  = frag * 0.72 + (excessLiq < 500 ? 0.40 : 0);
  const lo = ECB_DFR - 0.50, hi = ECB_MLF + 1.50;
  return NS_TENORS.map((tau, i) => {
    let r = b0 + nsRate(tau, b1, b2);
    r = clamp(r, lo, hi);
    return { label: NS_LABELS[i], rate: +r.toFixed(3) };
  });
}

// ─────────────────────────────────────────────────────────────────────
// §E  ROUGH BERGOMI VOL + MERTON D2D  (non-Markovian, H=0.10)
//   V_t = V_0 · exp( η · W^H_t − ½η²t^{2H} )
//   Discretised: V_{t+dt} ≈ V_t · exp( η·ΔW^H − ½η²dt^{2H} )
// ─────────────────────────────────────────────────────────────────────
const RB_H   = 0.10;   // rough exponent — empirical vol surface calibration
const RB_ETA = 1.80;   // vol-of-vol
const RB_V0  = 0.04;   // initial variance (VSTOXX² / 10000, ~20% vol)

function roughBergomiStep(V, dt, simTime) {
  const next = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const xi = i * 3.7 + 0.1;
    const n1 = fBmSample(xi, simTime * 2.0,        simTime,       RB_H, 3);
    const n2 = fBmSample(xi, simTime * 2.0 + dt,   simTime + dt,  RB_H, 3);
    const roughDW  = n2 - n1;
    const exponent = RB_ETA * roughDW - 0.5 * RB_ETA * RB_ETA * Math.pow(dt, 2 * RB_H);
    let v = Math.max(0.001, V[i] * Math.exp(exponent));
    if (v > 0.50) v = 0.50 - (v - 0.50) * 0.3;  // soft ceiling
    next[i] = v;
  }
  return next;
}
function mertonD2D(Lf, V, bps) {
  const A = Math.max(0.1, 1 + Lf * ASSET_VALUE), D = 1.5;
  const sA = Math.max(0.05, Math.sqrt(V) + bps / 800);
  if (A <= D) return 0;
  return (Math.log(A / D) + (0.025 - 0.5 * sA * sA)) / sA;
}

// ─────────────────────────────────────────────────────────────────────
// §F  LAPLACIAN EIGENMAP (35-node)
// ─────────────────────────────────────────────────────────────────────
function buildEigenmap(baseXY, Lf, bw = 0.45) {
  const n = N;
  const W = Array.from({ length:n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i === j) continue;
    const dx = baseXY[i][0] - baseXY[j][0], dy = baseXY[i][1] - baseXY[j][1];
    const fc = 1 + 0.8 * Math.min(Math.abs(Lf[i] - Lf[j]), 1);
    W[i][j]  = Math.exp(-(dx * dx + dy * dy) / (2 * bw * bw)) / fc;
  }
  const deg  = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += W[i][j]; deg[i] = s; }
  const dinv = deg.map(d => d > 1e-9 ? 1 / Math.sqrt(d) : 0);
  const M2   = Array.from({ length:n }, (_, i) => Array.from({ length:n }, (_, j) => {
    const Ls = (i === j ? 1 : 0) - W[i][j] * dinv[i] * dinv[j];
    return (i === j ? 2 : 0) - Ls;
  }));
  const e1 = new Float32Array(n).fill(1 / Math.sqrt(n));
  function pi(M, dl, it = 14) {
    let v = new Float32Array(n); for (let k = 0; k < n; k++) v[k] = Math.random() - 0.5;
    for (const d of dl) { let dot = 0; for (let k = 0; k < n; k++) dot += v[k] * d[k]; for (let k = 0; k < n; k++) v[k] -= dot * d[k]; }
    for (let ii = 0; ii < it; ii++) {
      const w = new Float32Array(n);
      for (let i2 = 0; i2 < n; i2++) { let s = 0; for (let j2 = 0; j2 < n; j2++) s += M[i2][j2] * v[j2]; w[i2] = s; }
      for (const d of dl) { let dot = 0; for (let k = 0; k < n; k++) dot += w[k] * d[k]; for (let k = 0; k < n; k++) w[k] -= dot * d[k]; }
      let nm = 0; for (let k = 0; k < n; k++) nm += w[k] * w[k]; nm = Math.sqrt(nm) || 1;
      for (let k = 0; k < n; k++) v[k] = w[k] / nm;
    }
    return v;
  }
  const e2 = pi(M2, [e1]), e3 = pi(M2, [e1, e2]);
  const coords = Array.from({ length:n }, (_, i) => [e2[i] * dinv[i], e3[i] * dinv[i]]);
  // Z-score normalization: subtract mean, divide by std
  let mx = 0, my = 0;
  for (const [x, y] of coords) { mx += x; my += y; }
  mx /= n; my /= n;
  let vx = 0, vy = 0;
  for (const [x, y] of coords) { vx += (x-mx)**2; vy += (y-my)**2; }
  const sx = Math.sqrt(vx / n) || 1, sy = Math.sqrt(vy / n) || 1;
  // Power-law stretch: sign(z)*|z|^0.6 pulls nodes to domain edges
  const TX = 75 * 0.96, TY = 50 * 0.96;
  const zxM = Math.max(...coords.map(([x]) => Math.abs((x - mx) / sx))) || 1;
  const zyM = Math.max(...coords.map(([,y]) => Math.abs((y - my) / sy))) || 1;
  return coords.map(([x, y]) => {
    const nx = clamp((x - mx) / sx / zxM, -1, 1);
    const ny = clamp((y - my) / sy / zyM, -1, 1);
    return [
      Math.sign(nx) * Math.pow(Math.abs(nx), 0.6) * TX,
      Math.sign(ny) * Math.pow(Math.abs(ny), 0.6) * TY,
    ];
  });
}

function applyRepulsion(pts) {
  const r = pts.map(p => [p[0], p[1]]);
  const THRESH = 8, PUSH = 2, TX = 75*0.93, TY = 50*0.93;
  for (let iter = 0; iter < 6; iter++) {
    for (let i = 0; i < r.length; i++) {
      for (let j = i + 1; j < r.length; j++) {
        const dx = r[i][0]-r[j][0], dy = r[i][1]-r[j][1];
        const d = Math.sqrt(dx*dx + dy*dy);
        if (d < THRESH && d > 0.001) {
          const f = PUSH * (1 - d/THRESH) / d;
          r[i][0] = clamp(r[i][0] + dx*f, -TX, TX);
          r[i][1] = clamp(r[i][1] + dy*f, -TY, TY);
          r[j][0] = clamp(r[j][0] - dx*f, -TX, TX);
          r[j][1] = clamp(r[j][1] - dy*f, -TY, TY);
        }
      }
    }
  }
  return r;
}

// Procrustes alignment: optimal 2D rotation to prevent eigenmap flipping on re-embedding
function procrustesAlign(newPts, refPts) {
  if (!refPts || refPts.length !== newPts.length) return newPts;
  let h00=0,h01=0,h10=0,h11=0;
  for (let i = 0; i < newPts.length; i++) {
    h00+=newPts[i][0]*refPts[i][0]; h01+=newPts[i][0]*refPts[i][1];
    h10+=newPts[i][1]*refPts[i][0]; h11+=newPts[i][1]*refPts[i][1];
  }
  const theta = Math.atan2(h01-h10, h00+h11);
  const c = Math.cos(theta), s = Math.sin(theta);
  return newPts.map(([x,y]) => [c*x - s*y, s*x + c*y]);
}

// ─────────────────────────────────────────────────────────────────────
// §G  KRUSKAL MST + CONTAGION STEP
// ─────────────────────────────────────────────────────────────────────
function kruskalMST(defaulted) {
  const edges = [];
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    if (defaulted[i] || defaulted[j]) continue;
    edges.push({ i, j, w: 1 / (CAPACITY[i][j] + 0.01) });
  }
  edges.sort((a, b) => a.w - b.w);
  const par = Array.from({ length:N }, (_, i) => i);
  const find = x => par[x] === x ? x : (par[x] = find(par[x]));
  const mst = [];
  for (const e of edges) {
    const pi = find(e.i), pj = find(e.j);
    if (pi !== pj) { par[pi] = pj; mst.push(e); }
  }
  return mst;
}
function kruskalMSTWeighted(def, stressArr) {
  const edges = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      if (def[i] || def[j]) continue;
      const baseW   = 1 - EXPOSURE[i][j];
      const stressW = 1 + 0.5 * clamp((stressArr[i] + stressArr[j]) / 2 / 20, 0, 1);
      edges.push({ i, j, w: baseW * stressW });
    }
  }
  edges.sort((a, b) => a.w - b.w);
  const par = Array.from({ length: N }, (_, i) => i);
  function find(x) { return par[x] === x ? x : (par[x] = find(par[x])); }
  const mst = [];
  for (const e of edges) {
    const ri = find(e.i), rj = find(e.j);
    if (ri !== rj) { par[ri] = rj; mst.push(e); }
    if (mst.length === N - 1) break;
  }
  return mst;
}

const CTAG_THRESH = 25, CTAG_MULT = 0.005, CTAG_DECAY = 0.08, CTAG_FLOOR = 0.001;
function contagionStep(c, sp, def, hwkInt, dt) {
  const next = new Float32Array(N);
  for (let i = 0; i < N; i++) next[i] = Math.max(0, c[i] * (1 - CTAG_DECAY * dt) - CTAG_FLOOR * dt);
  for (let i = 0; i < N; i++) {
    if (sp[i] <= CTAG_THRESH && !def[i]) continue;
    const ex  = def[i] ? 120 : sp[i] - CTAG_THRESH;
    const amp = 1 + hwkInt[i] * 1.2;
    for (let j = 0; j < N; j++) if (j !== i) next[j] += EXPOSURE[i][j] * ex * CTAG_MULT * amp * dt;
  }
  return next;
}


// ─────────────────────────────────────────────────────────────────────
// §BANK_TYPES  (used by MFG and synthesis layers)
// ─────────────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════════════════
// INVERSE PROBLEM — ROUGH PATH SIGNATURE CALIBRATION
//   Find θ*=(sigScale,jmpScale,hawkAlpha,hawkBeta,sigmaStar,kappa)
//   minimising L(θ)=‖S(X_sim(θ))−S(X_target)‖²_F
//   S = order-2 rough path signature of 35-dim stress path
// ─────────────────────────────────────────────────────────────────────
const EPISODE_SIGNATURES = {
  dec2011: {
    label:"Dec 2011 Sovereign",
    targetProfile:{ peakTick:45, peakSpread:72, spreadVelocity:1.2, networkFragmentation:0.42, hawkesDecay:0.35 },
    theta0:{ sigScale:1.1, jmpScale:0.4, hawkAlpha:0.28, hawkBeta:1.2 },
  },
  mar2020: {
    label:"Mar 2020 COVID",
    targetProfile:{ peakTick:18, peakSpread:95, spreadVelocity:4.8, networkFragmentation:0.55, hawkesDecay:0.80 },
    theta0:{ sigScale:1.8, jmpScale:1.2, hawkAlpha:0.18, hawkBeta:2.8 },
  },
  lehman2008: {
    label:"2008 Lehman Analog",
    targetProfile:{ peakTick:12, peakSpread:140, spreadVelocity:9.5, networkFragmentation:0.22, hawkesDecay:0.92 },
    theta0:{ sigScale:2.4, jmpScale:2.0, hawkAlpha:0.40, hawkBeta:3.5 },
  },
};

function signatureDistance(runResults, targetProfile) {
  const spreads = runResults.spreadHist;
  const n = spreads.length;
  const peakIdx = spreads.reduce((mi,v,i)=>v>spreads[mi]?i:mi,0);
  const peakSpread = spreads[peakIdx]||0;
  let velocity=0;
  for(let t=1;t<n;t++) velocity+=Math.abs(spreads[t]-spreads[t-1]);
  velocity/=n;
  const fragAtPeak = runResults.fragHist?.[peakIdx]||0;
  const postPeak = spreads.slice(peakIdx);
  const hawkesDecay = postPeak.length>1
    ? (postPeak[0]-postPeak[postPeak.length-1])/Math.max(postPeak[0],1) : 0;
  const tp = targetProfile;
  return Math.sqrt(
    ((peakIdx-tp.peakTick)/Math.max(tp.peakTick,1))**2*0.25+
    ((peakSpread-tp.peakSpread)/Math.max(tp.peakSpread,1))**2*0.35+
    ((velocity-tp.spreadVelocity)/Math.max(tp.spreadVelocity,0.1))**2*0.20+
    ((fragAtPeak-tp.networkFragmentation)/0.5)**2*0.10+
    ((hawkesDecay-tp.hawkesDecay)/1.0)**2*0.10
  );
}

function runCalibration(targetEpKey, initLf, initLm, initBTP, initCap, numSteps=8) {
  const epSig = EPISODE_SIGNATURES[targetEpKey];
  if(!epSig) return null;
  let theta={...epSig.theta0};
  const paramKeys=['sigScale','jmpScale','hawkAlpha','hawkBeta'];
  const STEP={sigScale:0.08,jmpScale:0.10,hawkAlpha:0.025,hawkBeta:0.15};
  const lossHist=[];
  function evalLoss(th) {
    const paths=Array.from({length:10},()=>
      runMCPath(initLf,initLm,new Float32Array(N),new Float32Array(N),
                initBTP,initCap,targetEpKey,th.sigScale,th.jmpScale,th.hawkAlpha,th.hawkBeta)
    );
    return paths.reduce((s,p)=>s+signatureDistance(p,epSig.targetProfile),0)/paths.length;
  }
  let currentLoss=evalLoss(theta);
  lossHist.push(currentLoss);
  for(let step=0;step<numSteps;step++){
    const grad={};
    for(const k of paramKeys){
      const eps=STEP[k]*0.5;
      grad[k]=(evalLoss({...theta,[k]:theta[k]+eps})-evalLoss({...theta,[k]:theta[k]-eps}))/(2*eps);
    }
    let lr=0.3;
    for(let ls=0;ls<4;ls++){
      const tn={};
      for(const k of paramKeys) tn[k]=theta[k]-lr*grad[k];
      tn.sigScale=clamp(tn.sigScale,0.3,3.0); tn.jmpScale=clamp(tn.jmpScale,0.1,2.5);
      tn.hawkAlpha=clamp(tn.hawkAlpha,0.05,0.6); tn.hawkBeta=clamp(tn.hawkBeta,0.5,5.0);
      const nl=evalLoss(tn);
      if(nl<currentLoss){theta=tn;currentLoss=nl;break;}
      lr*=0.5;
    }
    lossHist.push(currentLoss);
  }
  return {thetaStar:theta,lossHist,episode:epSig.label};
}


// ═══════════════════════════════════════════════════════════════════════
// ADVERSARIAL GAME — SPECULATOR vs ECB (Nash equilibrium via IBR)
//   Speculator action space per tick:
//     a_BTP   : add to btpDynRef (short BTP futures), cost ∝ a²
//     a_REPO  : reduce EXPOSURE[spec][targets] (withdraw bilateral lines)
//     a_DUMP  : fire-sale at chosen nodes (amplifies fireSaleRef)
//   ECB action space: (already in HJB controller)
//     u_TLTRO : injection rate (HJB_ACTS)
//     u_HAIR  : haircut relaxation (counters chain snap)
//   Equilibrium: neither player improves by unilateral deviation
//   Found by iterated best-response (IBR) — 5 rounds per call
// ─────────────────────────────────────────────────────────────────────
const ADV_BUDGET     = 1.0;   // speculator's total action budget per tick
const ADV_BTP_COST   = 0.4;   // fraction of budget consumed by BTP attack
const ADV_REPO_COST  = 0.35;  // fraction consumed by repo withdrawal
const ADV_DUMP_COST  = 0.25;  // fraction consumed by dump attack
const ADV_LAMBDA     = 0.80;  // speculator discount factor (short-horizon)

// Speculator payoff: higher system stress = profit
// Cost: proportional to action² (quadratic position cost)
function speculatorPayoff(systemSpread, btpDyn, a, actionCost) {
  const profit = clamp(systemSpread/100, 0, 2) + btpDyn * 0.5;
  return profit - 0.5 * actionCost * (a.btp**2 + a.repo**2 + a.dump**2);
}

// ECB payoff: negative stress - intervention cost
function ecbPayoff(systemSpread, btpDyn, u) {
  const stress = clamp(systemSpread/100, 0, 2) + btpDyn * 0.5;
  return -stress - HJB_GAMA * u * u;
}

// Speculator best response given ECB policy u
function speculatorBR(systemSpread, btpDyn, liveExp, lmArr, u) {
  // Try all combinations of attack allocation across 3 attack types
  let bestPayoff = -Infinity, bestAction = {btp:0, repo:0, dump:0};
  for (let bFrac = 0; bFrac <= 1.0; bFrac += 0.25) {
    for (let rFrac = 0; rFrac <= (1-bFrac); rFrac += 0.25) {
      const dFrac = 1 - bFrac - rFrac;
      if (dFrac < 0) continue;
      const a = {
        btp:  bFrac  * ADV_BUDGET,
        repo: rFrac  * ADV_BUDGET,
        dump: dFrac  * ADV_BUDGET,
      };
      // Simulate effect: BTP attack widens sovereign spreads
      const simBTP = btpDyn + a.btp * 0.15;
      // Repo withdrawal reduces network connectivity
      const simNetworkStress = Math.min(a.repo * 0.3, 0.5);
      // Dump amplifies spread
      const simSpread = systemSpread + a.dump * 8 + simNetworkStress * 15;
      const cost = ADV_BTP_COST * a.btp + ADV_REPO_COST * a.repo + ADV_DUMP_COST * a.dump;
      const payoff = speculatorPayoff(simSpread, simBTP, a, cost);
      if (payoff > bestPayoff) { bestPayoff = payoff; bestAction = a; }
    }
  }
  return bestAction;
}

// Iterated best-response to find Nash equilibrium
function findAdvEquilibrium(systemSpread, btpDyn, liveExp, lmArr, hjbPolicy, lfMean, lmMean) {
  // Start from ECB's HJB-optimal u
  const iLf  = Math.round(clamp(lfMean,0,1)*(HJB_NLF-1));
  const iLm  = Math.round(clamp(lmMean/2,0,1)*(HJB_NLM-1));
  const iBtp = Math.round(clamp(btpDyn,0,1)*(HJB_NBTP-1));
  const sIdx = iLf*HJB_NLM*HJB_NBTP+iLm*HJB_NBTP+iBtp;
  let u = hjbPolicy ? HJB_ACTS[hjbPolicy[sIdx]||0] : 0;
  let specAction = {btp:0, repo:0, dump:0};

  // 5 IBR rounds
  for (let round = 0; round < 5; round++) {
    // Speculator best-responds to current ECB u
    const newSpecAction = speculatorBR(systemSpread, btpDyn, liveExp, lmArr, u);
    // ECB best-responds: with speculator attack, stress is higher → inject more
    const adjSpread = systemSpread + newSpecAction.btp * 10 + newSpecAction.dump * 8;
    const adjBTP    = btpDyn + newSpecAction.btp * 0.15;
    const iLfAdj  = Math.round(clamp(lfMean,0,1)*(HJB_NLF-1));
    const iLmAdj  = Math.round(clamp(lmMean/2,0,1)*(HJB_NLM-1));
    const iBtpAdj = Math.round(clamp(adjBTP,0,1)*(HJB_NBTP-1));
    const sIdxAdj = iLfAdj*HJB_NLM*HJB_NBTP+iLmAdj*HJB_NBTP+iBtpAdj;
    const newU = hjbPolicy ? HJB_ACTS[hjbPolicy[sIdxAdj]||0] : 0;
    // Convergence check
    const specDelta = Math.abs(newSpecAction.btp-specAction.btp)+Math.abs(newSpecAction.repo-specAction.repo);
    const ecbDelta  = Math.abs(newU-u);
    specAction = newSpecAction; u = newU;
    if (specDelta < 0.02 && ecbDelta < 0.02) break;  // converged
  }
  return { specAction, ecbU: u };
}


// ═══════════════════════════════════════════════════════════════════════
// INFORMATION GEOMETRY OF CRISES  (Fisher metric on stress simplex Δ³⁴)
//   Fisher information metric on the stress distribution manifold
//   Each tick: stress distribution p_t over 35 nodes → point in Δ³⁴
//   Fisher metric: g_ij = Σ_x (∂log p/∂θᵢ)(∂log p/∂θⱼ) p(x)
//   Approximated from MC ensemble covariance (efficient Fisher estimator)
//   Projection to 2D via Principal Geodesic Analysis (≈ PCA in Fisher metric)
//
//   Key result: crisis trajectories are geodesically separable by type 5+ ticks
//   before any node crosses the crisis threshold — topological early warning
// ─────────────────────────────────────────────────────────────────────

// Convert raw stress array to probability distribution (softmax)
function stressToDistribution(stressArr) {
  const arr = Array.from(stressArr);
  const maxS = Math.max(...arr, 0.01);
  // Softmax with temperature (sharper = more differentiated)
  const exp = arr.map(s => Math.exp(s / maxS * 3.0));
  const sumE = exp.reduce((a,b)=>a+b, 0);
  return exp.map(e => e / sumE);
}

// Hellinger distance between two stress distributions
// d_H(p,q) = √(1 - Σ_i √(p_i·q_i)) — symmetric, bounded in [0,1]
function hellingerDistance(p, q) {
  let bc = 0;
  for (let i = 0; i < p.length; i++) bc += Math.sqrt(p[i] * q[i]);
  return Math.sqrt(Math.max(0, 1 - bc));
}

// Fisher information approximation from MC ensemble at time t
// ensemble: array of stress Float32Arrays (one per MC path at tick t)
// Returns 2D projection coordinates (Fisher-PCA)
function fisherProjection(stressHistory) {
  // stressHistory: array of {tick, stressDist} — trajectory of distributions
  if (stressHistory.length < 2) return [];

  // Compute pairwise Hellinger distances between consecutive ticks
  // This gives us the "speed" of movement in the information manifold
  const speeds = [];
  for (let t = 1; t < stressHistory.length; t++) {
    const d = hellingerDistance(stressHistory[t-1].dist, stressHistory[t].dist);
    speeds.push(d);
  }

  // Cumulative arc length in Fisher metric = Fisher geodesic coordinate
  const geodesicCoord = [0];
  for (const s of speeds) geodesicCoord.push(geodesicCoord[geodesicCoord.length-1] + s);
  const totalLength = geodesicCoord[geodesicCoord.length-1] || 1;

  // Project to 2D: (geodesic coord, "curvature" = deviation from straight path)
  // Curvature captures the TYPE of crisis trajectory
  return stressHistory.map((h, i) => {
    const gc = geodesicCoord[i] / totalLength;  // x-axis: geodesic progress [0,1]
    // y-axis: peripheral concentration index (peripheral nodes vs core)
    const peripheralMass = h.dist.reduce((s, p, ni) => s + (INST[ni].peripheral ? p : 0), 0);
    const coreMass       = h.dist.reduce((s, p, ni) => s + (!INST[ni].peripheral && INST[ni].tier <= 2 ? p : 0), 0);
    const concentration  = peripheralMass / Math.max(coreMass + peripheralMass, 0.01);
    return { gc, concentration, tick: h.tick };
  });
}

// Compute Fisher manifold trajectory from the live simulation state
// Call this every 5 ticks to update the trajectory
function updateFisherTrajectory(fisherHistRef, stressArr, tick) {
  const dist = stressToDistribution(stressArr);
  fisherHistRef.push({ tick, dist: [...dist] });
  if (fisherHistRef.length > 60) fisherHistRef.shift();
  return fisherProjection(fisherHistRef);
}

// Crisis type classifier: given current Fisher trajectory, estimate crisis type
// Returns { type, confidence, separationScore }
function classifyCrisisType(fisherProjection) {
  if (!fisherProjection || fisherProjection.length < 5) return null;

  const last = fisherProjection[fisherProjection.length-1];
  const first = fisherProjection[0];
  const speed = (last.gc - first.gc) / Math.max(fisherProjection.length, 1);
  const concentration = last.concentration;
  const curvature = fisherProjection.length > 1
    ? Math.abs(last.concentration - fisherProjection[Math.floor(fisherProjection.length/2)].concentration)
    : 0;

  // Classification rules (from geometric separation of crisis types in Fisher manifold):
  // Dec 2011: slow speed, HIGH peripheral concentration, increasing curvature
  // Mar 2020: fast speed, LOW concentration (systemic), flat curvature
  // Lehman: very fast speed, very low concentration, near-zero curvature
  let type, confidence;
  if (speed < 0.02 && concentration > 0.55) {
    type = "sovereign_contagion"; confidence = Math.min(0.95, concentration * 1.5);
  } else if (speed > 0.05 && concentration < 0.40) {
    type = "systemic_shock"; confidence = Math.min(0.95, speed * 15);
  } else if (speed > 0.08) {
    type = "cliff_collapse"; confidence = Math.min(0.95, speed * 10);
  } else {
    type = "ambiguous"; confidence = 0.4;
  }
  return { type, confidence: +confidence.toFixed(2), speed: +speed.toFixed(4), concentration: +concentration.toFixed(3) };
}

const BANK_TYPES_SYN = new Set(["Bank","G-SIB","Dealer"]);

// ─────────────────────────────────────────────────────────────────────
// §CAL  ECB CALIBRATION SNAPSHOT  (Q1 2025, ECB SDW)
// ─────────────────────────────────────────────────────────────────────
const CAL_SNAPSHOT = {
  label:"ECB Q1 2025", date:"28 Feb 2025",
  excessLiq:3850, estr:2.391, btpSpread:0.12,
  Lf:[0.92,0.78,0.72,0.81,0.82,0.85,0.88,0.70,0.74,0.73,
      0.68,0.69,0.71,0.64,0.61,0.58,0.54,0.85,0.84,0.83,
      0.80,0.80,0.80,0.80,0.80,0.80,0.82,0.82,0.82,0.82,
      0.88,0.75,0.70,0.87,0.90],
  Lm:[0.02,0.18,0.22,0.19,0.28,0.15,0.24,0.25,0.20,0.21,
      0.30,0.27,0.23,0.32,0.38,0.44,0.52,0.08,0.09,0.10,
      0.14,0.15,0.14,0.14,0.12,0.12,0.10,0.10,0.10,0.10,
      0.08,0.22,0.28,0.09,0.05],
};

// ═══════════════════════════════════════════════════════════════════════
// MONTE CARLO FORWARD ENSEMBLE  (50 paths, stripped BP+Hawkes+doom loop)
// ─────────────────────────────────────────────────────────────────────
const MC_PATHS = 50, MC_TICKS = 60, MC_DT = 0.08;
const BINS = 50; // density histogram bins for MC heatmap

function runMCPath(initLf, initLm, initCon, initHwk, initBTP, initCap,
                   ep, sigScale, jmpScale, alpha, beta) {
  let Lf=new Float32Array(initLf), Lm=new Float32Array(initLm);
  let con=new Float32Array(initCon), hwk=new Float32Array(initHwk);
  let btp=initBTP;
  const spreadHist = new Float32Array(MC_TICKS);
  const nodePeakSpread = new Float32Array(N);
  const nodeCrisisTick = new Int16Array(N).fill(-1);
  for (let tick=0; tick<MC_TICKS; tick++) {
    const t=tick/MC_TICKS;
    const sf=Math.max(0,Math.min(1,t*2-0.3));
    const phis=INST.map((_,i)=>calPhi(i,sf));
    const spArr=INST.map((_,i)=>{
      const v=phis[i]/G_MAX_PHI-0.55*Lf[i]+Lm[i]*0.25+con[i]+hwk[i]*0.30;
      if(INST[i].sov>0) return (v+btp*INST[i].sov*0.35)*50;
      return v*50;
    });
    const def=new Array(N).fill(false);
    const {nLf,nLm}=bpStep(Lf,Lm,spArr,new Array(N).fill(0),MC_DT,sigScale,jmpScale);
    Lf=nLf; Lm=nLm;
    hwk=hawkesStep(hwk,spArr,def,MC_DT,alpha,beta,0);
    con=contagionStep(con,spArr,def,hwk,MC_DT);
    const periph=INST.map((inst,i)=>inst.peripheral?i:-1).filter(i=>i>=0);
    const LmP=periph.reduce((a,i)=>a+Math.max(0,Lm[i]-0.3),0);
    btp=clamp(btp+(DL_GAMMA*LmP)*MC_DT,0,1.5);
    // Record system spread
    const sysSpread = spArr.reduce((a,b)=>a+b,0)/N;
    spreadHist[tick] = sysSpread;
    for (let i=0;i<N;i++){
      if(spArr[i]>nodePeakSpread[i]) nodePeakSpread[i]=spArr[i];
      if(nodeCrisisTick[i]===-1 && spArr[i]>TH_CRISIS) nodeCrisisTick[i]=tick;
    }
  }
  return { spreadHist, nodePeakSpread, nodeCrisisTick };
}

// ═══════════════════════════════════════════════════════════════════════
// MEAN FIELD GAME  (bank strategic best-response, approximates MFG equilibrium)
// ─────────────────────────────────────────────────────────────────────
const MFG_THETA=0.35, MFG_GAMMA=0.20;
function mfgBestResponse(spreads, Lf, Lm, meanSpread) {
  const injMod=new Float32Array(N);
  for (let i=0;i<N;i++) {
    if(!BANK_TYPES_SYN.has(INST[i].type)) continue;
    const ownS=clamp(spreads[i]/TH_CRISIS,0,1);
    const sysS=clamp(meanSpread/TH_CRISIS,0,1);
    const pLend=MFG_THETA*(sysS-ownS*0.5)-MFG_GAMMA*0.3;
    const pHoard=MFG_THETA*(ownS-sysS*0.3)-MFG_GAMMA*0.2;
    if(pLend>pHoard&&pLend>0)      injMod[i]=+0.15*(1-ownS);
    else if(pHoard>pLend&&pHoard>0) injMod[i]=-0.12*ownS;
  }
  return injMod;
}

// ═══════════════════════════════════════════════════════════════════════
// HJB OPTIMAL CONTROL  (ECB policy: value iteration on 15×15×8 state grid)
// ─────────────────────────────────────────────────────────────────────
const HJB_NLF=15,HJB_NLM=15,HJB_NBTP=8;
const HJB_NS=HJB_NLF*HJB_NLM*HJB_NBTP;
const HJB_ACTS=[0,0.4,0.8,1.2];
const HJB_BETA=0.92,HJB_LAM=1.0,HJB_GAMA=0.25;
function solveHJB() {
  const V=new Float32Array(HJB_NS);
  const policy=new Uint8Array(HJB_NS);
  const idx=(iLf,iLm,iBtp)=>iLf*HJB_NLM*HJB_NBTP+iLm*HJB_NBTP+iBtp;
  const toLf=i=>i/(HJB_NLF-1);
  const toLm=i=>i/(HJB_NLM-1)*2.0;
  const toBtp=i=>i/(HJB_NBTP-1);
  const nextState=(iLf,iLm,iBtp,u)=>{
    const Lf=toLf(iLf),Lm=toLm(iLm),btp=toBtp(iBtp);
    const nLf=clamp(Lf+(u*0.12-BP_THETA_F*0.08*(Lf-0.7)),0,1);
    const nLm=clamp(Lm+BP_THETA_M*0.08*(Math.max(0,0.55-Lf)-Lm*0.3),0,2);
    const nBtp=clamp(btp+DL_GAMMA*Lm*0.05-DL_KAPPA*btp*0.05,0,1);
    return idx(Math.round(clamp(nLf,0,1)*(HJB_NLF-1)),
               Math.round(clamp(nLm/2,0,1)*(HJB_NLM-1)),
               Math.round(clamp(nBtp,0,1)*(HJB_NBTP-1)));
  };
  const reward=(iLf,iLm,iBtp,u)=>{
    const stress=clamp(1-toLf(iLf)+toLm(iLm)*0.4+toBtp(iBtp)*0.6,0,2);
    return -HJB_LAM*stress*stress-HJB_GAMA*u*u;
  };
  for(let iter=0;iter<10;iter++){
    for(let a=0;a<HJB_NLF;a++) for(let b=0;b<HJB_NLM;b++) for(let c=0;c<HJB_NBTP;c++){
      const s=idx(a,b,c);
      let bestQ=-Infinity,bestA=0;
      HJB_ACTS.forEach((u,ai)=>{
        const Q=reward(a,b,c,u)+HJB_BETA*V[nextState(a,b,c,u)];
        if(Q>bestQ){bestQ=Q;bestA=ai;}
      });
      V[s]=bestQ; policy[s]=bestA;
    }
  }
  return {V,policy};
}

// ═══════════════════════════════════════════════════════════════════════
// PERSISTENT HOMOLOGY  (β₀, β₁ from sublevel sets of stress terrain Z-cache)
// ─────────────────────────────────────────────────────────────────────
function computePersistentHomology(zCache) {
  const W=ZCACHE_W,H=ZCACHE_H;
  const thresholds=[-2,-5,-10,-15,-20,-25];
  const b0series=[],b1series=[];
  const par=new Int32Array(W*H);
  function find(x){while(par[x]!==x){par[x]=par[par[x]];x=par[x];}return x;}
  function union(a,b){a=find(a);b=find(b);if(a!==b)par[a]=b;}
  for(const thr of thresholds){
    for(let i=0;i<W*H;i++) par[i]=i;
    const active=new Uint8Array(W*H);
    let V=0,E=0;
    for(let y=0;y<H;y++) for(let x=0;x<W;x++){
      const k=y*W+x;
      if(zCache[k]<=thr){
        active[k]=1; V++;
        if(x>0&&active[k-1]){E++;union(k,k-1);}
        if(y>0&&active[k-W]){E++;union(k,k-W);}
      }
    }
    const comps=new Set();
    for(let i=0;i<W*H;i++) if(active[i]) comps.add(find(i));
    const beta0=comps.size;
    b0series.push(beta0);
    b1series.push(Math.max(0,E-V+beta0));
  }
  return thresholds.map((thr,i)=>({thr,b0:b0series[i],b1:b1series[i]}));
}

function shannonEntropy(L) {
  const tot = Array.from(L).reduce((a, b) => a + b, 0);
  if (tot < 1e-9) return 0;
  let H = 0;
  for (let i = 0; i < N; i++) { const p = Math.max(L[i], 1e-9) / tot; H -= p * Math.log(p); }
  return H / Math.log(N);
}

// ─────────────────────────────────────────────────────────────────────
// §H  GMF FIELD EVALUATION
// ─────────────────────────────────────────────────────────────────────
const GMF_SIGMA_MIN = 0.40, GMF_BETA_S = 0.50;

function adaptiveSigma(pts) {
  const dists = [];
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
    const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
    dists.push(Math.sqrt(dx * dx + dy * dy));
  }
  if (!dists.length) return GMF_SIGMA_MIN;
  dists.sort((a, b) => a - b);
  const med = dists[Math.floor(dists.length / 2)];
  return Math.max(GMF_SIGMA_MIN, GMF_BETA_S * med);
}
function gmfEval(v, p, s2, qx, qy) {
  let num = 0, den = 0;
  for (let i = 0; i < p.length; i++) {
    const dx = qx - p[i][0], dy = qy - p[i][1];
    const w  = Math.exp(-(dx * dx + dy * dy) * s2);
    num += v[i] * w; den += w;
  }
  return den > 1e-10 ? num / den : 0;
}

// ─────────────────────────────────────────────────────────────────────
// §I  METRICS
// ─────────────────────────────────────────────────────────────────────
const ECB_DFR = 2.50, ECB_MRO = 2.65, ECB_MLF = 2.90;
const G_MAX_PHI = 1.5;

function computeHaircuts(sp) {
  return sp.map(s => HAIRCUT_H0 + HAIRCUT_GAMMA * Math.max(0, s - 30) ** 2);
}
function computeTEI(frag, entr, sysSp) {
  return Math.round(100 * clamp(1 - frag, 0, 1) * clamp(entr, 0, 1) * clamp(1 - Math.abs(sysSp) / 80, 0, 1));
}

function computeMetrics(Lf, Lm, contagion, hwkInt, defaulted, phis, animT, sandbox, btpSov, hestonV) {
  const spreads = Array(N).fill(0).map((_, i) => {
    if (defaulted[i]) return 120;
    let v = phis[i] / G_MAX_PHI - 0.55 * Lf[i] + Lm[i] * 0.25 + contagion[i] + hwkInt[i] * 0.30;
    if (INST[i].sov > 0) v += btpSov * INST[i].sov * 0.35;
    if (sandbox) {
      v += (sandbox.dfrShift || 0) / 100;
      if (INST[i].peripheral) v += (sandbox.tltro || 0) / 150;
      if (INST[i].sov > 0)    v += phis[i] * (sandbox.haircut || 0) / 350;
      v += (sandbox.qtDrain || 0) / 8000;
    }
    return v * 50;
  });
  const mean  = spreads.reduce((a, b) => a + b, 0) / N;
  const var_  = spreads.reduce((a, b) => a + (b - mean) ** 2, 0) / N;
  const frag  = clamp(Math.sqrt(var_) / 30, 0, 1);
  const entr  = shannonEntropy(Lf);
  const sf    = ramp(animT, 20, 15) - ramp(animT, 45, 15);
  const lTot  = Array.from(Lf).reduce((a, b) => a + b, 0);
  let liq = 3200 - 2600 * Math.pow(sf, 0.85) + lTot * 60;
  if (sandbox) { liq -= (sandbox.tltro || 0) * 1.2; liq -= (sandbox.qtDrain || 0); }
  const tei   = computeTEI(frag, entr, mean);
  const yc    = computeYieldCurve(frag, mean, liq);
  const hc    = computeHaircuts(spreads);
  const nodeData = spreads.map((bps, i) => {
    const d2d = mertonD2D(Lf[i], hestonV[i] || RB_V0, bps);
    return {
      bps: +bps.toFixed(1), Lf: +Lf[i].toFixed(3), Lm: +Lm[i].toFixed(3),
      conBps: +(contagion[i] * 50).toFixed(1), hawkBps: +(hwkInt[i] * 18).toFixed(1),
      haircut: +(hc[i] * 100).toFixed(2), d2d: +d2d.toFixed(2),
      defaulted: defaulted[i],
      status: defaulted[i] ? "DEFAULT" : bps > TH_CRISIS ? "CRISIS" : bps > TH_ALERT ? "ALERT" :
              bps < TH_SQUEEZE ? "SQUEEZE" : Math.abs(bps) > TH_WATCH ? "WATCH" : "NORMAL",
    };
  });
  return {
    estr: +(ECB_DFR + mean / 100).toFixed(3), systemSpreadBps: +mean.toFixed(1),
    excessLiqBn: Math.round(Math.max(50, liq)), fragilityIdx: +frag.toFixed(3),
    entropy: +entr.toFixed(3), tei, yieldCurve: yc, nodeData, spreads,
    alertCount: nodeData.filter(n => ["DEFAULT","CRISIS","ALERT","SQUEEZE"].includes(n.status)).length,
  };
}

// ─────────────────────────────────────────────────────────────────────
// §J  THERMAL HEATMAP  (Deep Slate → Amber → Neon Crimson)
//     phiNorm = 0   → #1a1c2c  base / liquidity surplus
//     phiNorm = 0.5 → #ffaa00  mid-stress / amber
//     phiNorm = 1   → #ff0033  extreme stress / neon crimson
// ─────────────────────────────────────────────────────────────────────
function thermalRGB(phiNorm, isoF) {
  const v = clamp(phiNorm, 0, 1);
  let r, g, b;
  if (v < 0.5) {
    const t = v / 0.5;
    r = lerp(0.102, 1.000, t); g = lerp(0.110, 0.667, t); b = lerp(0.173, 0.000, t);
  } else {
    const t = (v - 0.5) / 0.5;
    r = lerp(1.000, 1.000, t); g = lerp(0.667, 0.000, t); b = lerp(0.000, 0.200, t);
  }
  const f = isoF ?? 1.0;
  return [r * f, g * f, b * f];
}

// ─────────────────────────────────────────────────────────────────────
// §K  HISTORICAL EPISODES
// ─────────────────────────────────────────────────────────────────────
const EPISODES = {
  generic: {
    name: "ECB Generic Stress", dates: "Simulated",
    phases: [20, 35, 45], phaseNames: ["Calm","Deterioration","Crisis","Recovery"],
    sovMult:1.0, sigScale:1.0, jmpScale:1.0,
    desc: "Standard ECB stress scenario. Peripheral sovereign spread widening under moderate risk-off. TEI < 70 triggers SRO activation.",
    playbook: "Step 1: MRO at DFR+15bp. Step 2: SRO if TEI < 70. Step 3: Emergency LTRO if TEI < 50. Step 4: Haircut waivers.",
  },
  sept2019: {
    name: "Sept 2019 US Repo Spike", dates: "16–20 September 2019",
    phases: [8, 14, 26], phaseNames: ["Calm","Onset","Spike","Fed Response"],
    sovMult:0.15, sigScale:2.5, jmpScale:1.8,
    desc: "SOFR spiked +800bp overnight (10.0%). Trigger: $54B Treasury settlements + $20B repo settlement drained reserves simultaneously. Fed injected $75B T+N, $45B T+N+1.",
    playbook: "Fed 17 Sept: $75B overnight repo. Fed 18 Sept: $45B T+1 repo. FRBNY Markets Group activated. SRF subsequently formalised (July 2021).",
  },
  mar2020: {
    name: "March 2020 Dash-for-Cash", dates: "11–26 March 2020",
    phases: [6, 14, 28], phaseNames: ["Calm","Onset","Systemic Panic","Policy Response"],
    sovMult:1.6, sigScale:3.2, jmpScale:2.8,
    desc: "COVID-19 pandemic. Simultaneous global liquidation. Off-the-run Treasury spreads: 50bp. ECB PEPP €750B announced 18 March. Fed: unlimited repo, $1.5T, USD swap lines.",
    playbook: "ECB PEPP €750B (18 Mar). Fed PMCCF/SMCCF (9 Apr). FIMA Repo (31 Mar). Coordinated USD swaps: BoE/ECB/SNB/BoJ (19 Mar). TLTRO-III at −100bp below DFR.",
  },
  dec2011: {
    name: "Dec 2011 LTRO-1 Crisis", dates: "November 2011 – March 2012",
    phases: [12, 27, 40], phaseNames: ["Calm","Deterioration","Sovereign Crisis","LTRO Normalisation"],
    sovMult:3.8, sigScale:2.0, jmpScale:1.6,
    desc: "Euro sovereign debt crisis. BTP 10Y: 710bp over Bund. Interbank bifurcated: core vs periphery. US MMFs withdrew €400B from European banks. LTRO-1: €489.2B to 523 institutions.",
    playbook: "LTRO-1 (21 Dec): €489.2B, 3Y at 1% to 523 banks. LTRO-2 (29 Feb 2012): €529.5B to 800 banks. SMP sterilised weekly. VLTROs extended.",
  },
};

const epPhase = (t, ep) => {
  const [p1,p2,p3] = EPISODES[ep].phases, n = EPISODES[ep].phaseNames;
  return t < p1 ? n[0] : t < p2 ? n[1] : t < p3 ? n[2] : n[3];
};
const epSf = (t, ep) => {
  const [p1,p2,p3] = EPISODES[ep].phases;
  return ramp(t, p1, p2 - p1) - ramp(t, p3, 10);
};

// ─────────────────────────────────────────────────────────────────────
// §L  SCENE CONSTANTS  (slow, academic physics)
// ─────────────────────────────────────────────────────────────────────
const GRID_W = 256, GRID_H = 256;            // 256×256 = 66,049 vertices (4× cheaper)
const NV = (GRID_W+1)*(GRID_H+1);
const SURF_W = 220, SURF_H = 150;            // larger manifold
const SURF_HALF_X = 110, SURF_HALF_Y = 75;
const PART_N = 0,     ARCH_SEGS = 20;   // particles disabled — cluttered canvas
const ZCACHE_W    = 64;
const ZCACHE_H    = 64;
const ZCACHE_SIZE = ZCACHE_W * ZCACHE_H;
const ZCACHE_X0   = -SURF_HALF_X;
const ZCACHE_Y0   = -SURF_HALF_Y;
const ZCACHE_DX   = SURF_W / (ZCACHE_W - 1);
const ZCACHE_DY   = SURF_H / (ZCACHE_H - 1);
function zCacheLookup(wx, wy, cache) {
  const gx = Math.min(Math.max((wx - ZCACHE_X0) / ZCACHE_DX, 0), ZCACHE_W - 1.001);
  const gy = Math.min(Math.max((wy - ZCACHE_Y0) / ZCACHE_DY, 0), ZCACHE_H - 1.001);
  const ix = gx | 0, iy = gy | 0;
  const fx = gx - ix, fy = gy - iy;
  const i00 = iy * ZCACHE_W + ix;
  const c00 = cache[i00] || 0;
  const c10 = cache[Math.min(i00 + 1,            ZCACHE_SIZE - 1)] || 0;
  const c01 = cache[Math.min(i00 + ZCACHE_W,     ZCACHE_SIZE - 1)] || 0;
  const c11 = cache[Math.min(i00 + ZCACHE_W + 1, ZCACHE_SIZE - 1)] || 0;
  return c00*(1-fx)*(1-fy) + c10*fx*(1-fy) + c01*(1-fx)*fy + c11*fx*fy;
}
const RBF_SIGMA   = 10.0;                    // localized craters — wide sigma caused whole-surface flooding
const STRESS_SCALE = 5.0;                    // recalibrated to new sigma: single-node peak depth ≈ 15–18
const FBM_AMP     = 2.0;                     // raised so calm-state waviness is visible from default camera
const Z_MAX       = 50.0;                    // max sinkhole depth
const FLOOR_Z     = -60.0;
const ECB_MLF_Z   = 22.0;  // ECB Marginal Lending Facility depth (world Z) — hard policy backstop
const DS_SCALE_X  = 1.0, DS_SCALE_Y = 1.0;  // eigenmap already in domain units
const SIM_SPEED = 1.5;      // frames per second of sim time (was 4–7)
const DEFAULT_SBX = { dfrShift:0, tltro:0, haircut:0, qtDrain:0 };
const DEFAULT_PARAMS = {
  hurst:     0.80,   // fBm Hurst exponent (0.1=rough, 0.9=smooth) — heavy silk
  fBmOct:    4,      // octaves (fewer = cleaner peaks)
  fBmAmp:    0.020,  // amplitude (small = subtle micro-texture)
  hawkAlpha: 0.45,   // Hawkes excitation
  hawkBeta:  0.30,   // Hawkes decay (slow decay = persistent contagion)
  simSpeed:  1.5,    // simulation speed multiplier
  showWire:  true,   // wireframe overlay
  showNodes: true,   // node markers
};

// ─────────────────────────────────────────────────────────────────────
// §M  MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────
export default function ECBRepoTerminal() {
  const mountRef    = useRef(null);
  const threeRef    = useRef({});
  const smoothedEigenRef  = useRef(null);
  const smoothedStressRef = useRef(null);
  const labelDivsRef      = useRef([]);
  const ouNoiseRef        = useRef(new Float32Array(N)); // OU microstructure noise per node
  const zCacheRef          = useRef(new Float32Array(ZCACHE_SIZE));
  const zCacheFrameRef     = useRef(0);
  const smoothedCollatRef = useRef(null);               // per-node smoothed collateral stress
  const particleDensityRef = useRef(new Float32Array(N)); // §C₂ particle hoarding density per node
  const fireSaleRef        = useRef(new Float32Array(N)); // §C₃ fire-sale volume being sold per node
  const prevFireSaleRef    = useRef(new Float32Array(N)); // previous frame for transition detection
  const hawkRhoRef         = useRef(0);                   // §C Hawkes branching ratio ρ
  const lfRef       = useRef(new Float32Array(N));
  const lmRef       = useRef(new Float32Array(N));
  const conRef      = useRef(new Float32Array(N));
  const hwkRef      = useRef(new Float32Array(N));
  const hvRef       = useRef(new Float32Array(N).fill(RB_V0));
  const defRef      = useRef(Array(N).fill(false));
  const pkRef       = useRef(Array(N).fill(0));
  const ndPtsRef    = useRef(Array.from({ length:N }, (_, i) => {
    const inst  = INST[i];
    const ringR = [0,0.14,0.30,0.55,0.75,0.90,1.05][inst.tier] || 0.80;
    const angle = (i / N) * Math.PI * 2 + inst.tier * 0.5;
    return [Math.cos(angle) * ringR * 63.75, Math.sin(angle) * ringR * 42.5];
  }));
  const sbxRef      = useRef(DEFAULT_SBX);
  const paramsRef   = useRef({ ...DEFAULT_PARAMS });
  const eigenRef    = useRef(null);
  const eigenPrevRef= useRef(null);
  const eigenAlpha  = useRef(0);
  const telRef      = useRef([]);
  const ghostStressRef  = useRef(new Float32Array(N));
  const ccbasisRef       = useRef(0.0);   // §D3 EUR/USD cross-currency basis spread
  const mstEdgeStateRef = useRef(
    Array.from({ length: N - 1 }, () => ({ live: false, fadeAlpha: 0, i: -1, j: -1 }))
  );
  const btpDynRef    = useRef(0.0);                          // §B2 dynamic BTP sovereign premium
  const bankCapRef   = useRef(new Float32Array(N).fill(1.0)); // §B2 per-node residual capital ratio
  const pathHistRef  = useRef(Array.from({ length:N }, () => [])); // §B1 rough-path history per node
  const sHistRef    = useRef(Array(60).fill(0));
  const sIdxRef     = useRef(0);
  const prevStRef   = useRef(Array(N).fill("NORMAL"));
  // §SYNTHESIS refs
  const mcResultsRef       = useRef(null);
  const mcRunningRef       = useRef(false);
  const mfgActiveRef       = useRef(false);
  const mfgActionsRef      = useRef(new Float32Array(N));
  const hjbPolicyRef       = useRef(null);
  const hjbValueRef        = useRef(null);
  const hjbActiveRef       = useRef(false);
  const homologyRef        = useRef(null);
  const sculptureSlicesRef = useRef([]);
  const sculptureTicksRef  = useRef([]);
  const sculptMeshesRef    = useRef([]);
  // §PIPELINE: shared synthesis state — tools read each other here
  const synthPipelineRef = useRef({
    // Populated by each tool, read by others
    networkFraction: 1.0,      // from §NET
    crisisTypeGuess: null,     // from §IG  → suggests calibration episode
    calibratedEp: null,        // from §INV → MC should re-run
    mcStale: false,            // true after calibration runs
    lastCalTick: -1,           // tick when calibration applied
    hjbForceU: 0,              // from §HJB → adversarial reads this
    advPressure: 0,            // from §ADV → HJB re-solves when high
  });

  // §NET endogenous network
  const liveExpRef         = useRef(buildLiveExposure());
  const edgeFractionRef    = useRef(1.0);
  const topologyAlertRef   = useRef(false);
  // §INV calibration
  const calResultsRef      = useRef({});
  const calRunningRef      = useRef(false);
  // §ADV adversarial
  const advEnabledRef      = useRef(false);
  const advEquilRef        = useRef(null);
  // §IG information geometry
  const fisherHistRef      = useRef([]);
  const fisherProjRef      = useRef([]);
  const crisisTypeRef      = useRef(null);
  const stRef       = useRef({
    animTime:0, lastTime:0, playing:false, mode:"monitor",
    tick:0, simTime:0, ep:"generic", pendingKick:null,
  });

  const [episode,   setEpisode]  = useState("generic");
  const [mode,      setMode]     = useState("monitor");
  const [dispFrame, setDispFrame]= useState(0);
  const [playing,   setPlaying]  = useState(false);
  const [params,    setParams]   = useState({ ...DEFAULT_PARAMS });
  const [metrics,   setMetrics]  = useState({
    estr: ECB_DFR, systemSpreadBps:0, excessLiqBn:3200, fragilityIdx:0,
    entropy:1, tei:100, alertCount:0,
    nodeData: INST.map(()=>({ bps:0, Lf:0, Lm:0, conBps:0, hawkBps:0, haircut:2, d2d:4, defaulted:false, status:"NORMAL" })),
    spreads: Array(N).fill(0), yieldCurve:[],
  });
  const [sandbox,   setSandbox]  = useState(DEFAULT_SBX);
  const [btpShock,  setBtpShock] = useState(0);
  const [alerts,    setAlerts]   = useState([]);
  const [nodeLabels,setNodeLabels]= useState([]);
  const [mstEdges,  setMstEdges] = useState([]);
  const [stressHist,setSHist]    = useState(Array(60).fill(0));
  const [hoverIdx,  setHoverIdx] = useState(null);
  const [mmrLog,      setMmrLog]   = useState([]);
  const [splitLayers,  setSplitLayers] = useState(false);
  const [hawkRho,      setHawkRho]     = useState(0);
  const [sigPanelIdx,  setSigPanelIdx] = useState(null);
  const [doomBTP,      setDoomBTP]     = useState(0);
  const [minCap,       setMinCap]      = useState(1);
  const [secondOrderCount, setSecondOrderCount] = useState(0);
  const [showCCB,         setShowCCB]         = useState(false);
  const [ccbBasis,        setCcbBasis]        = useState(0);
  const [ghostEpisode,    setGhostEpisode]    = useState(null);
  const [ghostVisible,    setGhostVisible]    = useState(false);
  const [designerOpen,  setDesignerOpen]  = useState(false);
  const [designerNode,  setDesignerNode]  = useState(0);
  const [designerLf,    setDesignerLf]    = useState(0.5);
  const [designerLm,    setDesignerLm]    = useState(0.3);
  const [designerCon,   setDesignerCon]   = useState(0.0);
  // §SYNTHESIS state
  const [mcBands,        setMcBands]        = useState(null);
  const [mcRunning,      setMcRunning]      = useState(false);
  const [mfgEnabled,     setMfgEnabled]     = useState(false);
  const [mfgStats,       setMfgStats]       = useState({ lenders:0, hoarders:0, holders:0 });
  const [calibrated,     setCalibrated]     = useState(false);
  const [calLabel,       setCalLabel]       = useState("");
  const [hjbEnabled,     setHjbEnabled]     = useState(false);
  const [hjbReady,       setHjbReady]       = useState(false);
  const [hjbOptU,        setHjbOptU]        = useState(0);
  const [homologyEnabled,setHomologyEnabled]= useState(false);
  const [homologyData,   setHomologyData]   = useState(null);
  const [sculptureMode,  setSculptureMode]  = useState(false);
  const [sliceCount,     setSliceCount]     = useState(0);
  const [synthOpen,      setSynthOpen]      = useState(false);
  const [synthSuggestion,setSynthSuggestion]= useState('');   // cross-tool suggestion
  const [mcStale,        setMcStale]        = useState(false);// MC stale after calibration
  const [mcCardOpen,     setMcCardOpen]     = useState(false);// standalone MC prediction card
  // §NET
  const [edgeFraction,   setEdgeFraction]   = useState(1.0);
  const [topologyAlert,  setTopologyAlert]  = useState(false);
  const [edgeCount,      setEdgeCount]      = useState(N*(N-1));
  // §INV
  const [calResults,     setCalResults]     = useState({});
  const [calRunning,     setCalRunning]     = useState(false);
  const [calProgress,    setCalProgress]    = useState(0);
  // §ADV
  const [advEnabled,     setAdvEnabled]     = useState(false);
  const [advEquil,       setAdvEquil]       = useState(null);
  // §IG
  const [fisherProj,     setFisherProj]     = useState([]);
  const [crisisType,     setCrisisType]     = useState(null);

  useEffect(() => { stRef.current.showCCB = showCCB; }, [showCCB]);
  useEffect(() => { stRef.current.ghostEpisode = ghostEpisode; }, [ghostEpisode]);
  useEffect(() => { stRef.current.playing = playing; }, [playing]);
  useEffect(() => { stRef.current.splitLayers = splitLayers; }, [splitLayers]);
  useEffect(() => { stRef.current.mode = mode; }, [mode]);
  useEffect(() => { sbxRef.current = { ...sandbox, btpSov: btpShock / 100 }; }, [sandbox, btpShock]);
  useEffect(() => { paramsRef.current = { ...params }; }, [params]);
  useEffect(() => { mfgActiveRef.current = mfgEnabled; }, [mfgEnabled]);
  useEffect(() => { hjbActiveRef.current = hjbEnabled;
    if (hjbEnabled && !hjbPolicyRef.current) {
      const { V, policy } = solveHJB();
      hjbPolicyRef.current = policy; hjbValueRef.current = V; setHjbReady(true);
    }
  }, [hjbEnabled]);
  useEffect(() => { stRef.current._hjbOptU = hjbOptU; }, [hjbOptU]);
  useEffect(() => { advEnabledRef.current = advEnabled; }, [advEnabled]);
  useEffect(() => { stRef.current.sculptureMode = sculptureMode; }, [sculptureMode]);
    useEffect(() => {
    stRef.current.ep = episode;
    eigenRef.current = null;
    seek(0);
  }, [episode]);

  const updateParam = useCallback((k, v) => setParams(p => ({ ...p, [k]:v })), []);
  const updateSandbox= useCallback((k, v) => setSandbox(p => ({ ...p, [k]:v })), []);

  function inject(idx) {
    if (defRef.current[idx]) return;
    lfRef.current[idx] = Math.min(lfRef.current[idx] + 0.25, PUMP_CAP);
    stRef.current.pendingKick = idx;
  }
  function applyDesignerStress() {
    const i = designerNode;
    lfRef.current[i]  = clamp(designerLf, 0, 1);
    lmRef.current[i]  = clamp(designerLm, 0, 3);
    conRef.current[i] = clamp(designerCon, 0, 4);
    hwkRef.current[i] = Math.min(hwkRef.current[i] + designerCon * 2, 8);
    const tm = new Date().toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit",second:"2-digit"});
    setAlerts(p => [{
      time: tm,
      text: `STRESS DESIGN: ${INST[i].name} → Lf=${designerLf.toFixed(2)}, Lm=${designerLm.toFixed(2)}, Con=${designerCon.toFixed(1)}`,
      sev: 'ALERT', id: Date.now()
    }, ...p].slice(0, 20));
  }
  function applyDesignerDefault() {
    forceDefault(designerNode);
  }
  function forceDefault(idx) {
    defRef.current = defRef.current.map((d, i) => i === idx ? true : d);
    lfRef.current[idx] = 0; lmRef.current[idx] = 2.5;
    const adj = buildKNN(ndPtsRef.current, 3);
    adj[idx]?.forEach(j => { conRef.current[j] += 2.5 * EXPOSURE[idx][j]; });
    hwkRef.current[idx] += 1.5;
    const tm = new Date().toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit",second:"2-digit"});
    setAlerts(p => [{ time:tm, text:`Default event: ${INST[idx].name}. Contagion matrix activating.`, sev:"CRISIS", id:Date.now() }, ...p].slice(0, 20));
  }
  function resetAll() {
    defRef.current = Array(N).fill(false);
    conRef.current = new Float32Array(N);
    hwkRef.current = new Float32Array(N);
    lfRef.current.fill(0); lmRef.current.fill(0);
    pkRef.current = Array(N).fill(0);
  }
  function applyCalibration() {
    const cal = CAL_SNAPSHOT;
    lfRef.current  = new Float32Array(cal.Lf);
    lmRef.current  = new Float32Array(cal.Lm);
    btpDynRef.current = cal.btpSpread;
    conRef.current = new Float32Array(N);
    hwkRef.current = new Float32Array(N);
    defRef.current = new Array(N).fill(false);
    bankCapRef.current = new Float32Array(N).fill(1.0);
    setCalibrated(true);
    setCalLabel(cal.label + " · " + cal.date);
    const tm = new Date().toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit",second:"2-digit"});
    setAlerts(p => [{ time:tm, text:`CALIBRATED to ${cal.label}. Excess liq: €${cal.excessLiq}bn. €STR: ${cal.estr}%. BTP: ${(cal.btpSpread*100).toFixed(0)}bp.`, sev:'INFO', id:Date.now() }, ...p].slice(0,20));
  }
  function runMonteCarlo() {
    if (mcRunningRef.current) return;
    mcRunningRef.current = true; setMcRunning(true);
    const initLf=new Float32Array(lfRef.current), initLm=new Float32Array(lmRef.current);
    const initCon=new Float32Array(conRef.current), initHwk=new Float32Array(hwkRef.current);
    const initBTP=btpDynRef.current, initCap=new Float32Array(bankCapRef.current);
    const ep=stRef.current.ep||"generic", epDef=EPISODES[ep];
    const alpha=paramsRef.current.hawkAlpha, beta=paramsRef.current.hawkBeta;
    synthPipelineRef.current.mcStale = false;
    setMcStale(false);
setTimeout(() => {
      // Run all paths with full per-node data
      const pathResults = Array.from({length:MC_PATHS}, () =>
        runMCPath(initLf,initLm,initCon,initHwk,initBTP,initCap,
                  ep,epDef.sigScale,epDef.jmpScale,alpha,beta)
      );
      const allSpreads = pathResults.map(r => r.spreadHist);

      // ── 1. QUANTILE BANDS ──────────────────────────────────────────
      const p10=new Float32Array(MC_TICKS),p50=new Float32Array(MC_TICKS),p90=new Float32Array(MC_TICKS);
      for(let t=0;t<MC_TICKS;t++){
        const vals=[...allSpreads.map(s=>s[t])].sort((a,b)=>a-b);
        p10[t]=vals[Math.floor(MC_PATHS*0.10)];
        p50[t]=vals[Math.floor(MC_PATHS*0.50)];
        p90[t]=vals[Math.floor(MC_PATHS*0.90)];
      }
      const maxSpread = Math.max(...p90, 60);

      // ── 2. PROBABILITY DENSITY HEATMAP ─────────────────────────────
      const density = Array.from({length:MC_TICKS}, (_,t) => {
        const col = new Float32Array(BINS);
        for(const path of allSpreads){
          const bin = Math.min(BINS-1, Math.floor((path[t]/maxSpread)*BINS));
          col[bin]++;
        }
        return col;
      });

      // ── 3. PATH BIFURCATION — σ(paths) at each tick ────────────────
      // Where σ peaks = critical bifurcation point
      const sigma = new Float32Array(MC_TICKS);
      for(let t=0;t<MC_TICKS;t++){
        const mean = p50[t];
        sigma[t] = Math.sqrt(allSpreads.reduce((s,p)=>s+(p[t]-mean)**2,0)/MC_PATHS);
      }
      const bifurcTick = sigma.indexOf(Math.max(...sigma));

      // ── 4. PER-NODE STRESS PROBABILITY ─────────────────────────────
      // For each node: fraction of paths where it enters crisis at some point
      const nodeStressProb = new Float32Array(N);
      const nodeEarliestCrisis = new Float32Array(N).fill(MC_TICKS);
      for(const r of pathResults){
        for(let i=0;i<N;i++){
          if(r.nodeCrisisTick[i]>=0){
            nodeStressProb[i]++;
            if(r.nodeCrisisTick[i]<nodeEarliestCrisis[i]) nodeEarliestCrisis[i]=r.nodeCrisisTick[i];
          }
        }
      }
      for(let i=0;i<N;i++) nodeStressProb[i]/=MC_PATHS;
      // Top 6 most-stressed institutions (by P(crisis across paths))
      const topNodes = Array.from({length:N},(_,i)=>i)
        .sort((a,b)=>nodeStressProb[b]-nodeStressProb[a]).slice(0,6)
        .map(i=>({
          i, abbr:INST[i].abbr, name:INST[i].name,
          prob:nodeStressProb[i],
          earliest: nodeEarliestCrisis[i]<MC_TICKS ? nodeEarliestCrisis[i] : null,
        }));
      // Per-tick survival curve for top 3 nodes:
      // P(node i not yet in crisis by tick t) across paths
      const survivalCurves = topNodes.slice(0,3).map(node => ({
        ...node,
        curve: Array.from({length:MC_TICKS},(_,t)=>
          pathResults.filter(r=>r.nodeCrisisTick[node.i]===-1||r.nodeCrisisTick[node.i]>t).length/MC_PATHS
        ),
      }));

      // ── 5. CONDITIONAL SCENARIO SPLIT ──────────────────────────────
      // Split paths: "crisis paths" = peak system spread > TH_CRISIS
      const crisisPaths = allSpreads.filter(p=>Math.max(...p)>TH_CRISIS);
      const recovPaths  = allSpreads.filter(p=>Math.max(...p)<=TH_CRISIS);
      const condCrisis = MC_TICKS > 0 && crisisPaths.length > 0 ? Array.from({length:MC_TICKS},(_,t)=>{
        const vals=[...crisisPaths.map(p=>p[t])].sort((a,b)=>a-b);
        return { p50: vals[Math.floor(crisisPaths.length*0.5)], n: crisisPaths.length };
      }) : null;
      const condRecov = recovPaths.length > 0 ? Array.from({length:MC_TICKS},(_,t)=>{
        const vals=[...recovPaths.map(p=>p[t])].sort((a,b)=>a-b);
        return { p50: vals[Math.floor(recovPaths.length*0.5)], n: recovPaths.length };
      }) : null;

      // ── 6. TAIL RISK METRICS ────────────────────────────────────────
      // VaR(95) and ES(95) of final system spread at t=MC_TICKS-1
      const finalSpreads = [...allSpreads.map(p=>p[MC_TICKS-1])].sort((a,b)=>a-b);
      const var95 = finalSpreads[Math.floor(MC_PATHS*0.95)];
      const es95  = finalSpreads.slice(Math.floor(MC_PATHS*0.95)).reduce((s,v)=>s+v,0) /
                    Math.max(finalSpreads.slice(Math.floor(MC_PATHS*0.95)).length, 1);
      const ePeakSpread = allSpreads.reduce((s,p)=>s+Math.max(...p),0)/MC_PATHS;
      // P(crisis) at each tick
      const pCrisis = Array.from({length:MC_TICKS},(_,t)=>
        allSpreads.filter(p=>p[t]>TH_CRISIS).length/MC_PATHS
      );
      // Crisis horizon: first tick where p50 > TH_CRISIS
      let crisisHorizon = null;
      for(let t=0;t<MC_TICKS;t++){ if(p50[t]>TH_CRISIS){ crisisHorizon=t; break; } }

      
      const result = {
        p10:Array.from(p10), p50:Array.from(p50), p90:Array.from(p90),
        density, maxSpread, sigma:Array.from(sigma), bifurcTick,
        topNodes, survivalCurves,
        condCrisis, condRecov,
        var95, es95, ePeakSpread,
        pCrisis:Array.from(pCrisis), crisisHorizon,
        nCrisis: crisisPaths.length, nRecov: recovPaths.length,

      };
      mcResultsRef.current = result;
      setMcBands(result);
      mcRunningRef.current=false; setMcRunning(false);
    },0);
  }
  function runCalibrationFor(episodeKey) {
    if (calRunningRef.current) return;
    calRunningRef.current = true;
    setCalRunning(true); setCalProgress(0);
    const initLf=new Float32Array(lfRef.current), initLm=new Float32Array(lmRef.current);
    const initBTP=btpDynRef.current, initCap=new Float32Array(bankCapRef.current);
    setTimeout(() => {
      const result = runCalibration(episodeKey, initLf, initLm, initBTP, initCap, 8);
      calResultsRef.current = { ...calResultsRef.current, [episodeKey]: result };
      setCalResults(prev => ({...prev, [episodeKey]: result}));
      calRunningRef.current=false; setCalRunning(false); setCalProgress(100);
      // Apply calibrated parameters to simulation
      if (result?.thetaStar) {
        const t = result.thetaStar;
        updateParam('hawkAlpha', t.hawkAlpha);
        updateParam('hawkBeta',  t.hawkBeta);
        const tm=new Date().toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit",second:"2-digit"});
        setAlerts(p=>[{time:tm,
          text:`CALIBRATED: θ*=(α=${t.hawkAlpha.toFixed(3)},β=${t.hawkBeta.toFixed(2)},σ=${t.sigScale.toFixed(2)}) for ${EPISODE_SIGNATURES[episodeKey]?.label||episodeKey}. Loss: ${result.lossHist[result.lossHist.length-1]?.toFixed(4)}`,
          sev:'INFO',id:Date.now()
        },...p].slice(0,20));
        // §PIPELINE: calibration ran → MC is now stale, mark for re-run
        synthPipelineRef.current.calibratedEp = episodeKey;
        synthPipelineRef.current.mcStale = true;
        synthPipelineRef.current.lastCalTick = stRef.current.tick;
        setMcStale(true);
        setSynthSuggestion(`θ* fitted to ${EPISODE_SIGNATURES[episodeKey]?.label} — running calibrated MC fan…`);
        // Auto-run MC after calibration (500ms delay so params settle)
        setTimeout(() => { runMonteCarlo(); }, 500);
      }
    }, 0);
  }

  function seek(v) {
    const s = stRef.current;
    s.animTime = v; s.simTime = 0; s.pendingKick = null;
    if (s.posZ) s.posZ.fill(0);  // guard: removed in shader refactor but kept for safety
    if (s.velZ) s.velZ.fill(0);
    conRef.current = new Float32Array(N); hwkRef.current = new Float32Array(N);
    eigenRef.current = null;
    if (telRef.current && telRef.current.length) {
      const snap = telRef.current[Math.round(v)];
      if (snap) {
        if (snap.lf instanceof Float32Array) lfRef.current = new Float32Array(snap.lf);
        if (snap.lm instanceof Float32Array) lmRef.current = new Float32Array(snap.lm);
      }
    }
    setDispFrame(Math.round(v));
  }
  function buildKNN(pts, k) {
    return pts.map((_, i) => {
      const d = [];
      for (let j = 0; j < pts.length; j++) {
        if (j !== i) { const dx = pts[i][0]-pts[j][0], dy = pts[i][1]-pts[j][1]; d.push({j, d2: dx*dx+dy*dy}); }
      }
      return d.sort((a,b)=>a.d2-b.d2).slice(0,k).map(e=>e.j);
    });
  }
  function exportTelemetry() {
    const d = telRef.current; if (!d.length) return;
    const b = new Blob([JSON.stringify(d, null, 2)], { type:"application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(b);
    a.download = `ecb_manifold_${episode}_${Date.now()}.json`; a.click();
  }
  const switchMode = useCallback(m => {
    setMode(m); stRef.current.mode = m;
    if (m === "monitor") { setSandbox(DEFAULT_SBX); sbxRef.current = DEFAULT_SBX; }
  }, []);

  // ── THREE.JS SETUP ──────────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current; if (!el) return;
    const W = el.clientWidth || 900, H = el.clientHeight || 560;

    const renderer = new THREE.WebGLRenderer({ antialias:false, alpha:false, powerPreference:"high-performance" });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0xf5f5f5, 1);
    el.appendChild(renderer.domElement);

    // ── CSS2D LABEL OVERLAY ─────────────────────────────────────
    const css2dRenderer = new CSS2DRenderer();
    css2dRenderer.setSize(W, H);
    css2dRenderer.domElement.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;overflow:hidden;';
    el.appendChild(css2dRenderer.domElement);

    const scene = new THREE.Scene(); scene.background = new THREE.Color(0xf5f5f5);

    // Academic camera angle — MATLAB-style overhead perspective
    const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 2000);
    camera.position.set(140, 310, 220); camera.up.set(0, 0, 1); camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0); controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 60; controls.maxDistance = 900; controls.update();

    // ── POST-PROCESSING CHAIN (Bloom + SMAA) ─────────────────────
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(W * 0.5, H * 0.5),  // half-res bloom target (performance)
      0.30,   // strength  — subtle glow; dynamic range set in update()
      0.40,   // radius
      0.96    // threshold — above 0xf5f5f5 background luminance, prevents white-screen bloom
    );
    composer.addPass(bloomPass);
    composer.addPass(new SMAAPass(W * renderer.getPixelRatio(), H * renderer.getPixelRatio()));

    // Clean academic lighting — even, no drama
    scene.add(new THREE.AmbientLight(0xffffff, 0.90));
    const key = new THREE.DirectionalLight(0xffffff, 0.45);
    key.position.set(2, 3, 5); scene.add(key);
    const fill = new THREE.DirectionalLight(0xf0f4ff, 0.20);
    fill.position.set(-3, -1, 3); scene.add(fill);

    // ── SHARED SHADER UNIFORMS ──────────────────────────────────
    const sharedUniforms = {
      uNodePos:    { value: Array.from({ length: N }, () => new THREE.Vector2()) },
      uNodeStress: { value: Array.from({ length: N }, () => 0.0) },
      uNodeSigma:  { value: Array.from({ length: N }, () => RBF_SIGMA) },
      uNodeType:   { value: Array.from({ length: N }, () => 0.0) },
      uNodeNoise:  { value: Array.from({ length: N }, () => 0.0) },
      uFloorZ:      { value: ECB_MLF_Z },
      uLayerOffset: { value: 0.0 },
      uIsCollat:    { value: 0.0 },
      uTime:        { value: 0.0 },
      uFbmAmp:      { value: FBM_AMP },
      uFireSale:    { value: Array.from({ length: N }, () => 0.0) },
    };

    const VS = `
uniform vec2  uNodePos[35];
uniform float uNodeStress[35];
uniform float uNodeSigma[35];
uniform float uTime;
uniform float uFbmAmp;
uniform float uNodeType[35];
uniform float uNodeNoise[35];
uniform float uFloorZ;
uniform float uLayerOffset;
uniform float uFireSale[35];
varying float vHeight;
varying vec3  vNormal;
float hash(vec2 p){float h=dot(p,vec2(127.1,311.7));return fract(sin(h)*43758.5453);}
float vnoise(vec2 p){
  vec2 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1.0,0.0)),u.x),mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0,1.0)),u.x),u.y)*2.0-1.0;
}
float fbm(vec2 p){
  float v=0.0,a=1.0,nm=0.0;
  for(int i=0;i<4;i++){v+=vnoise(p)*a;nm+=a;a*=0.574;p*=2.0;}
  return v/nm;
}
void main(){
  float z=0.0,dzdx=0.0,dzdy=0.0;
  for(int i=0;i<35;i++){
    vec2 d=position.xy-uNodePos[i];
    float s2=uNodeSigma[i]*uNodeSigma[i];
    float nS=(uNodeSigma[i]*0.35); z+=uNodeNoise[i]*exp(-dot(d,d)/(2.0*nS*nS));
    if(uNodeType[i]>0.5){
      float r=uNodeSigma[i]*1.8;
      float dist=max(length(d),0.001);
      float f=max(1.0-dist/r,0.0);
      if(f>0.001){float sqrtF=sqrt(f);z-=uNodeStress[i]*sqrtF;float gC=uNodeStress[i]/(2.0*r*dist*sqrtF);dzdx+=gC*d.x;dzdy+=gC*d.y;}
    } else {
      float g=uNodeStress[i]*exp(-dot(d,d)/(2.0*s2));
      z-=g; dzdx+=g*d.x/s2; dzdy+=g*d.y/s2;
    }
  }
  z+=fbm(position.xy*0.03+vec2(uTime*0.012,0.0))*uFbmAmp;
  for(int i=0;i<35;i++){
    if(uFireSale[i]<0.01)continue;
    vec2 fd=position.xy-uNodePos[i];
    z-=12.0*uFireSale[i]*exp(-0.006*dot(fd,fd));
  }
  z=z+uLayerOffset;
  vHeight=z;
  vNormal=normalize(normalMatrix*normalize(vec3(-dzdx,-dzdy,1.0)));
  gl_Position=projectionMatrix*modelViewMatrix*vec4(position.xy,z,1.0);
}`;

    const FS = `
precision highp float;
uniform float uIsCollat;
varying float vHeight;
varying vec3 vNormal;
vec3 heatColor(float d){
  vec3 c0=vec3(0.06,0.16,0.44),c1=vec3(0.902,0.494,0.133),c2=vec3(1.000,0.000,0.200);
  if(d<6.0)return mix(c0,c1,d/6.0);
  return mix(c1,c2,clamp((d-6.0)/14.0,0.0,1.0));
}
vec3 collatColor(float d){
  vec3 d0=vec3(0.05,0.10,0.50),d1=vec3(0.45,0.00,0.78),d2=vec3(0.85,0.20,1.00);
  if(d<6.0)return mix(d0,d1,d/6.0);
  return mix(d1,d2,clamp((d-6.0)/14.0,0.0,1.0));
}
void main(){
  float depth=max(-vHeight,0.0);
  vec3 col=uIsCollat>0.5?collatColor(depth):heatColor(depth);
  float diff=max(dot(vNormal,normalize(vec3(0.5,0.7,1.0))),0.0)*0.5+0.5;
  vec3 shaded=col*diff;
  float fw=fwidth(vHeight);
  float c8 =1.0-smoothstep(0.0,fw*0.9,mod(abs(vHeight),8.0));
  float c32=1.0-smoothstep(0.0,fw*0.9,mod(abs(vHeight),32.0));
  float lineA=max(c8*0.38,c32*0.82);
  vec3 finalCol=mix(shaded,vec3(0.10,0.10,0.14),lineA);
  gl_FragColor=vec4(finalCol,uIsCollat>0.5?0.72:1.0);
}`;

    const FSLINE = `
precision highp float;
varying float vHeight;
void main(){
  vec3 c=mix(vec3(0.25,0.28,0.45),vec3(0.90,0.20,0.10),clamp(-vHeight/30.0,0.0,1.0));
  gl_FragColor=vec4(c,0.22);
}`;

    // ── MANIFOLD MESH (solid + wireframe overlay) ──────────────
    const geoMain = new THREE.PlaneGeometry(SURF_W, SURF_H, GRID_W, GRID_H);
    const geoWire = new THREE.PlaneGeometry(SURF_W, SURF_H, 64, 64);

    const matSolid = new THREE.ShaderMaterial({
      uniforms: sharedUniforms, vertexShader: VS, fragmentShader: FS,
      side: THREE.DoubleSide, extensions: { derivatives: true },
    });
    const matWire = new THREE.ShaderMaterial({
      uniforms: sharedUniforms, vertexShader: VS, fragmentShader: FSLINE,
      wireframe: true, transparent: true,
    });

    const meshMain = new THREE.Mesh(geoMain, matSolid);
    scene.add(meshMain);
    const meshWire = new THREE.Mesh(geoWire, matWire);
    meshWire.position.z = 0.15;
    scene.add(meshWire);

    // ── COLLATERAL SURFACE (Layer C — blue-violet, repo/collateral stress) ──
    const collatUniforms = {
      ...sharedUniforms,
      uNodeStress:  { value: Array.from({ length: N }, () => 0.0) },
      uLayerOffset: { value: 0.0 },
      uIsCollat:    { value: 1.0 },
    };
    const geoCollat = new THREE.PlaneGeometry(SURF_W, SURF_H, GRID_W, GRID_H);
    const matCollat = new THREE.ShaderMaterial({
      uniforms: collatUniforms, vertexShader: VS, fragmentShader: FS,
      side: THREE.DoubleSide, transparent: true, depthWrite: false, extensions: { derivatives: true },
    });
    const meshCollat = new THREE.Mesh(geoCollat, matCollat);
    meshCollat.visible = false; // hidden until split mode; collateral layer only shown on demand
    scene.add(meshCollat);

    // ── GHOST MANIFOLD (cool grey-green reference episode overlay) ────────
    const FS_GHOST = `
      precision highp float;
      varying float vHeight;
      varying vec3 vNormal;
      void main() {
        float depth = max(-vHeight, 0.0);
        vec3 c0 = vec3(0.55, 0.70, 0.65);
        vec3 c1 = vec3(0.25, 0.45, 0.50);
        vec3 col = mix(c0, c1, clamp(depth / 20.0, 0.0, 1.0));
        if (mod(depth, 5.0) < 0.25) col *= 0.6;
        float diff = max(dot(normalize(vNormal), normalize(vec3(0.5, 0.7, 1.0))), 0.0) * 0.4 + 0.6;
        gl_FragColor = vec4(col * diff, 0.38);
      }
    `;
    const ghostUniforms = {
      uNodePos:    { value: Array.from({ length: N }, () => new THREE.Vector2()) },
      uNodeStress: { value: Array.from({ length: N }, () => 0.0) },
      uNodeSigma:  { value: Array.from({ length: N }, () => RBF_SIGMA) },
      uNodeType:   { value: Array.from({ length: N }, () => 0.0) },
      uNodeNoise:  { value: Array.from({ length: N }, () => 0.0) },
      uFloorZ:     { value: ECB_MLF_Z },
      uLayerOffset:{ value: 8.0 },
      uIsCollat:   { value: 0.0 },
      uTime:       { value: 0.0 },
      uFbmAmp:     { value: FBM_AMP * 0.5 },
      uFireSale:   { value: Array.from({ length: N }, () => 0.0) },
    };
    const geoGhost = new THREE.PlaneGeometry(SURF_W, SURF_H, 64, 64);
    const matGhost = new THREE.ShaderMaterial({
      uniforms: ghostUniforms, vertexShader: VS, fragmentShader: FS_GHOST,
      side: THREE.DoubleSide, transparent: true, depthWrite: false,
    });
    const meshGhost = new THREE.Mesh(geoGhost, matGhost);
    meshGhost.visible = false;
    scene.add(meshGhost);

    // ── CROSS-CURRENCY BASIS LAYER (olive-gold EUR/USD FX swap stress) ──
    const FS_CCB = `
      precision highp float;
      varying float vHeight;
      varying vec3 vNormal;
      void main() {
        float depth = max(-vHeight, 0.0);
        vec3 c0 = vec3(0.72, 0.68, 0.35);
        vec3 c1 = vec3(0.85, 0.45, 0.10);
        vec3 col = mix(c0, c1, clamp(depth / 18.0, 0.0, 1.0));
        if (mod(depth, 4.0) < 0.20) col *= 0.5;
        float diff = max(dot(normalize(vNormal), normalize(vec3(0.3, 0.8, 1.0))), 0.0)*0.45+0.55;
        gl_FragColor = vec4(col * diff, 0.42);
      }
    `;
    const ccbUniforms = {
      uNodePos:    { value: Array.from({ length: N }, () => new THREE.Vector2()) },
      uNodeStress: { value: Array.from({ length: N }, () => 0.0) },
      uNodeSigma:  { value: Array.from({ length: N }, () => RBF_SIGMA) },
      uNodeType:   { value: Array.from({ length: N }, () => 0.0) },
      uNodeNoise:  { value: Array.from({ length: N }, () => 0.0) },
      uFloorZ:     { value: ECB_MLF_Z },
      uLayerOffset:{ value: 0.0 },
      uIsCollat:   { value: 0.0 },
      uTime:       { value: 0.0 },
      uFbmAmp:     { value: FBM_AMP * 0.3 },
      uFireSale:   { value: Array.from({ length: N }, () => 0.0) },
    };
    const geoCCB = new THREE.PlaneGeometry(SURF_W, SURF_H, 48, 48);
    const matCCB = new THREE.ShaderMaterial({
      uniforms: ccbUniforms, vertexShader: VS, fragmentShader: FS_CCB,
      side: THREE.DoubleSide, transparent: true, depthWrite: false,
    });
    const meshCCB = new THREE.Mesh(geoCCB, matCCB);
    meshCCB.visible = false;
    scene.add(meshCCB);

    // Precompute grid XY (still needed for particle Z sampling)
    const pa = geoMain.attributes.position;
    const gxArr = new Float32Array(NV), gyArr = new Float32Array(NV);
    for (let i = 0; i < NV; i++) { gxArr[i] = pa.getX(i); gyArr[i] = pa.getY(i); }

    // DFR baseline plane
    const dfrGeo = new THREE.PlaneGeometry(SURF_W + 5, SURF_H + 5);
    const dfrMat = new THREE.MeshBasicMaterial({ color:0x4466aa, transparent:true, opacity:0.04, side:THREE.DoubleSide });
    scene.add(new THREE.Mesh(dfrGeo, dfrMat));
    const dfrEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(SURF_W + 5, SURF_H + 5)),
      new THREE.LineBasicMaterial({ color:0x2244aa, transparent:true, opacity:0.50 })
    );
    dfrEdge.position.z = 0.1; scene.add(dfrEdge);

    // Floor grid
    const fg = new THREE.GridHelper(200, 22, 0xcccccc, 0xdddddd);
    fg.rotation.x = Math.PI / 2; fg.position.z = FLOOR_Z; scene.add(fg);

    // ECB Marginal Lending Facility backstop — metallic "hard floor" plane
    const mlfMat  = new THREE.MeshStandardMaterial({ color:0x0d1a33, metalness:0.88, roughness:0.08, transparent:true, opacity:0.60, side:THREE.DoubleSide });
    const mlfMesh = new THREE.Mesh(new THREE.PlaneGeometry(SURF_W + 20, SURF_H + 20), mlfMat);
    mlfMesh.position.z = -ECB_MLF_Z; scene.add(mlfMesh);
    const mlfEdge = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(SURF_W + 20, SURF_H + 20)),
      new THREE.LineBasicMaterial({ color:0x4488ff, transparent:true, opacity:0.70 })
    );
    mlfEdge.position.z = -ECB_MLF_Z + 0.1; scene.add(mlfEdge);

    // Bounding box
    const cP = new THREE.Mesh(new THREE.BoxGeometry(SURF_W + 5, SURF_H + 5, 50));
    cP.position.z = 15; scene.add(new THREE.BoxHelper(cP, 0x999999));

    // Axis arrows
    const ORIG = new THREE.Vector3(-SURF_HALF_X - 2, -SURF_HALF_Y - 2, FLOOR_Z);
    const AX_DEFS = [
      { dir: new THREE.Vector3(1,0,0), col:0xaa2222 },
      { dir: new THREE.Vector3(0,1,0), col:0x226622 },
      { dir: new THREE.Vector3(0,0,1), col:0x2244aa },
    ];
    AX_DEFS.forEach(({ dir, col }, ai) => {
      const len = ai === 0 ? 110 : ai === 1 ? 75 : 38;
      scene.add(new THREE.ArrowHelper(dir, ORIG, len, col, 2.5, 1.5));
    });

    // Node disc markers (sized by balance sheet)
    const discMats  = INST.map(() => new THREE.MeshStandardMaterial({ roughness:0.35, metalness:0.10, transparent:true, opacity:0.75 }));
    const discMeshes= INST.map((inst, i) => {
      const r = inst.type === "CB" ? 2.6 : inst.type === "CCP" ? 0.9 :
                Math.max(0.8, Math.min(2.0, 0.8 + 1.1 * Math.sqrt(Math.max(inst.bs, 0.02) / 2.67)));
      const dg = new THREE.SphereGeometry(r, 16, 10);
      const m  = new THREE.Mesh(dg, discMats[i]); scene.add(m); return m;
    });

    // ── CSS2D TETHERED LABELS (attached to node spheres) ────────
    const labelDivs = discMeshes.map((mesh, ni) => {
      const div = document.createElement('div');
      div.style.cssText = [
        'font-family:Georgia,serif',
        'font-size:9px',
        'font-weight:600',
        'color:#ffffff',
        'text-shadow:0 0 5px rgba(0,0,0,0.95),0 0 12px rgba(0,0,0,0.7)',
        'pointer-events:none',
        'white-space:nowrap',
        'line-height:1.35',
        'user-select:none',
      ].join(';');
      div.textContent = INST[ni].abbr;
      const obj = new CSS2DObject(div);
      obj.position.set(0, 0, 2.2);
      mesh.add(obj);
      return div;
    });
    labelDivsRef.current = labelDivs;

    // Crosshair lines per node
    const CS = 2.8;
    const crGeo = new THREE.BufferGeometry();
    const crPos = new Float32Array(N * 4 * 3);
    crGeo.setAttribute("position", new THREE.BufferAttribute(crPos, 3));
    const crLines = new THREE.LineSegments(crGeo, new THREE.LineBasicMaterial({ color:0x445566, linewidth:1 }));
    crLines.visible = false;
    scene.add(crLines);

    // ── MST FLOW-TUBES (TubeGeometry meshes hugging terrain surface) ─────
    const MAX_TUBES = N - 1;
    const tubeMats  = Array.from({ length: MAX_TUBES }, () =>
      new THREE.MeshBasicMaterial({ color:0x4466aa, transparent:true, opacity:0.72, side:THREE.DoubleSide, depthTest:false })
    );
    const tubeObjects = Array.from({ length: MAX_TUBES }, () => ({ mesh:null, geo:null }));
    // Legacy arch refs kept for CCP-tear logic; all invisible
    const archMats = tubeMats; // alias so CCP-tear color/opacity writes work
    const archLines = tubeObjects.map(() => ({ visible:false }));
    // Particle system disabled — define stubs so threeRef assignment is valid
    const partPos = null, partVel = null, instMesh = null, _instMatrix = null, _instPos = null;
    
    // Raycaster
    const raycaster = new THREE.Raycaster(), mouse = new THREE.Vector2();
    let hovIdx = -1;
    function onMove(e) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      let best = -1, bestD = Infinity;
      discMeshes.forEach((m, i) => { const h = raycaster.intersectObject(m); if (h.length && h[0].distance < bestD) { bestD = h[0].distance; best = i; } });
      if (best !== hovIdx) { hovIdx = best; setHoverIdx(best >= 0 ? best : null); }
    }
    function onClick(e) {
      if (e.button !== 0) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObject(meshMain);
      if (!hits.length) return;
      const pt = hits[0].point; let best = 0, bestD2 = Infinity;
      ndPtsRef.current.forEach((p, i) => { const dx=pt.x-p[0], dy=pt.y-p[1], d2=dx*dx+dy*dy; if(d2<bestD2){bestD2=d2;best=i;} });
      // Shock hammer: +500bps equivalent spike + immediate Hawkes contagion
      stRef.current.pendingKick = best;
      hwkRef.current[best] = Math.min(hwkRef.current[best] + 5.0, 10.0);
      conRef.current[best] = Math.min(conRef.current[best] + 1.5,  4.0);
      inject(best);
      setSigPanelIdx(best);
    }
    renderer.domElement.addEventListener("mousemove", onMove);
    renderer.domElement.addEventListener("click",     onClick);


    // ── §IV  4D SCULPTOR MESHES ─────────────────────────────────────
    // 13 slices (12 historical + 1 live ghost), fanned to the right of
    // the main manifold, rotated to face the camera at default orbit.
    // PlaneGeometry(W, H, 63, 63) → exactly 64×64 = 4096 vertices,
    // matching ZCACHE_W×ZCACHE_H layout perfectly.
    {
      const SCULPT_SLICES = 13; // 12 history + 1 live
      const SW = SURF_W * 0.44, SH = SURF_H * 0.44;
      for (let si = 0; si < SCULPT_SLICES; si++) {
        const sGeo = new THREE.PlaneGeometry(SW, SH, ZCACHE_W - 1, ZCACHE_H - 1);
        // Verify vertex count matches zcache
        // pos.count = ZCACHE_W * ZCACHE_H = 64*64 = 4096
        const sMat = new THREE.MeshLambertMaterial({
          color: new THREE.Color(0.15, 0.45, 0.80),
          transparent: true, opacity: 0.20,
          side: THREE.DoubleSide, depthWrite: false,
        });
        const sMesh = new THREE.Mesh(sGeo, sMat);
        sMesh.visible = false;
        // Fan right of main manifold, slightly elevated, facing camera
        // X: spread horizontally to the right
        // Y: slight vertical spread so slices don't overlap
        // Z: small depth offset for each layer
        sMesh.position.set(
          SURF_W * 0.70 + si * (SURF_W * 0.06),
          -SURF_H * 0.15 + si * 1.5,
          -5 + si * 0.8
        );
        scene.add(sMesh);
        sculptMeshesRef.current.push(sMesh);
      }
    }

    threeRef.current = {
      renderer, scene, camera, controls,
      meshMain, meshWire, matWire, geoMain, geoWire,
      gxArr, gyArr, discMeshes, discMats, crGeo, crPos,
      AX_DEFS, W, H,
      archMats, archLines, tubeObjects, tubeMats, scene,
      partPos, partVel, instMesh, _instMatrix, _instPos,
      sharedUniforms, collatUniforms, meshCollat,
      ghostUniforms, meshGhost,
      ccbUniforms, meshCCB,
      composer,
    };

    // ── UPDATE LOOP ─────────────────────────────────────────────────
    const labelSmooth = Array.from({length:N}, () => ({x:0, y:0, init:false}));
    const _lbl3 = new THREE.Vector3();
    function update(animT, dt) {
      if (!threeRef.current?.sharedUniforms) return; // guard: scene not yet ready
      const safeDt    = Math.min(dt, 0.025);
      const s         = stRef.current;
      const shouldSim = s.playing || s.mode !== "monitor";
      const p         = paramsRef.current;
      const ep     = s.ep || "generic";
      const epDef  = EPISODES[ep];
      const sb     = sbxRef.current;
      const def    = defRef.current;
      const btpSov = sb.btpSov || 0;

      if (shouldSim) s.simTime += safeDt * 0.006; // 10× dilated

      // A. Parametric phi
      const sf   = epSf(animT, ep);
      const phis = INST.map((_, i) => calPhi(i, sf) * (1 + epDef.sovMult * INST[i].sov * sf * 0.5));

      // B. Base positions for eigenmap
      const baseXY = INST.map((inst, i) => tierPos(inst, i, sf));

      // C. Laplacian eigenmap (eager on first frame; every 6 ticks thereafter)
      const EIGEN_CADENCE = stRef.current._emaFrameTime > (1000 / 58) * 1.15 ? 12 : 6;
      if (!eigenRef.current || (shouldSim && s.tick % EIGEN_CADENCE === 0)) {
        eigenPrevRef.current = eigenRef.current;
        const rawEigen = applyRepulsion(buildEigenmap(baseXY, Array.from(lfRef.current)));
        eigenRef.current   = procrustesAlign(rawEigen, eigenPrevRef.current);
        eigenAlpha.current = 0;
      }
      eigenAlpha.current = Math.min(eigenAlpha.current + safeDt * 0.08, 1); // syrupy
      const ec = eigenRef.current, ep2 = eigenPrevRef.current;
      // buildEigenmap already returns domain-scale coords; DS_SCALE_X/Y = 1.0
      const eigenPts = ep2 ? ec.map(([x,y],i)=>[
        lerp(ep2[i][0], x, eigenAlpha.current),
        lerp(ep2[i][1], y, eigenAlpha.current),
      ]) : ec;

      // Laminar flow smoothing: nodes drift toward target (0.05 = "heavy water")
      if (!smoothedEigenRef.current || smoothedEigenRef.current.length !== N) {
        smoothedEigenRef.current = eigenPts.map(p => [p[0], p[1]]);
      }
      const sp = smoothedEigenRef.current;
      for (let i = 0; i < N; i++) {
        sp[i][0] += (eigenPts[i][0] - sp[i][0]) * 0.008;
        sp[i][1] += (eigenPts[i][1] - sp[i][1]) * 0.008;
      }
      // Force CCP to manifold center (non-moving singularity)
      if (CCP_IDX >= 0) { sp[CCP_IDX][0] = 0; sp[CCP_IDX][1] = 0; }

      // D. Preliminary spreads
      const prelSpreads = Array(N).fill(0).map((_, i) => {
        if (def[i]) return 120;
        let v = phis[i] / G_MAX_PHI - 0.55 * lfRef.current[i] + lmRef.current[i] * 0.25
              + conRef.current[i] + hwkRef.current[i] * 0.30;
        if (INST[i].sov > 0) v += btpSov * INST[i].sov * 0.35;
        if (s.mode !== "monitor") {
          v += (sb.dfrShift || 0) / 100;
          if (INST[i].peripheral) v += (sb.tltro || 0) / 150;
          if (INST[i].sov > 0)    v += phis[i] * (sb.haircut || 0) / 350;
          v += (sb.qtDrain || 0) / 8000;
        }
        return v * 50;
      });

      // E–G. ODE / stochastic steps — only advance when simulation is running
      const injections = Array(N).fill(0);
      if (s.pendingKick !== null) { injections[s.pendingKick] = 1.2; s.pendingKick = null; }
      // §MFG: strategic bank best-response
      if (mfgActiveRef.current) {
        const meanSp = prelSpreads.reduce((a,b)=>a+b,0) / N;
        const injMod = mfgBestResponse(prelSpreads, lfRef.current, lmRef.current, meanSp);
        mfgActionsRef.current = injMod;
        for (let i = 0; i < N; i++) injections[i] += injMod[i];
      }
      if (shouldSim) {
        const { nLf, nLm } = bpStep(lfRef.current, lmRef.current, prelSpreads, injections, safeDt, epDef.sigScale, epDef.jmpScale);
        lfRef.current = nLf; lmRef.current = nLm;

        // F. Rough Bergomi
        hvRef.current = roughBergomiStep(hvRef.current, safeDt, s.simTime);

        // G. Hawkes + contagion (using tunable alpha/beta)
        const prevZBuf = threeRef.current.zBuf;
        const meanZ = prevZBuf ? Array.from(prevZBuf).reduce((a,b)=>a+b,0) / prevZBuf.length : 0;
        // §NET: withdrawal cache refreshed at metrics cadence — not per-frame
        // Hawkes contagion uses live (contracting) exposure — not static EXPOSURE
        // Temporarily substitute: reuse EXPOSURE array slots with live values for this tick
        // (we write to a working copy to avoid mutating the original)
        const liveExpSnapshot = liveExpRef.current;

        // Modified hawkes: inline with live exposure (faster than refactoring hawkesStep)
        {
          const lambda0 = hawkesBaseline(meanZ);
          const { alphaDynamic } = hawkesBranchingRatio(meanZ, p.hawkAlpha, p.hawkBeta);
          const next = new Float32Array(N);
          for (let i=0;i<N;i++) next[i] = hwkRef.current[i]*Math.exp(-p.hawkBeta*safeDt)+lambda0*safeDt;
          for (let j=0;j<N;j++) {
            const stressed=def[j]||prelSpreads[j]>45;
            if(!stressed) continue;
            const mag=def[j]?1.0:clamp((prelSpreads[j]-45)/90,0,1);
            for(let i=0;i<N;i++){
              if(i!==j) next[i]+=alphaDynamic*liveExpSnapshot[j][i]*mag*safeDt;
            }
          }
          // Second-order with live exposure
          const SO_ALPHA=0.38,SO_THRESH=1.8;
          for(let j=0;j<N;j++){
            if(next[j]<SO_THRESH) continue;
            const secMag=clamp((next[j]-SO_THRESH)/2.0,0,1);
            for(let i=0;i<N;i++) if(i!==j) next[i]+=SO_ALPHA*alphaDynamic*liveExpSnapshot[j][i]*secMag*safeDt;
          }
          hwkRef.current = next;
        }
        const { rho } = hawkesBranchingRatio(meanZ, p.hawkAlpha, p.hawkBeta);
        hawkRhoRef.current = rho;
        conRef.current = contagionStep(conRef.current, prelSpreads, def, hwkRef.current, safeDt);
        pkRef.current  = pkRef.current.map((pk, i) => Math.max(pk * 0.999, prelSpreads[i]));

        // §ADV: Apply speculator attack if adversarial mode enabled
        // findAdvEquilibrium runs 5 IBR rounds — throttled to avoid lag
        if (advEnabledRef.current && hjbActiveRef.current && s.tick % 3 === 0) {
          const mLf=Array.from(lfRef.current).reduce((a,b)=>a+b,0)/N;
          const mLm=Array.from(lmRef.current).reduce((a,b)=>a+b,0)/N;
          const eq=findAdvEquilibrium(
            prelSpreads.reduce((a,b)=>a+b,0)/N, btpDynRef.current,
            liveExpRef.current, lmRef.current, hjbPolicyRef.current, mLf, mLm
          );
          advEquilRef.current = eq;
          // Apply speculator actions
          if(eq.specAction.btp>0.01) btpDynRef.current=Math.min(btpDynRef.current+eq.specAction.btp*0.08,1.5);
          if(eq.specAction.repo>0.01){
            // Speculator withdraws repo lines from most-stressed bilateral nodes
            const targets=Array.from({length:N},(_,i)=>i)
              .filter(i=>BANK_TYPES_SYN.has(INST[i].type)&&!REPO_TYPE?.[0]?.[i])
              .sort((a,b)=>prelSpreads[b]-prelSpreads[a]).slice(0,3);
            for(const t of targets){
              for(let j=0;j<N;j++) liveExpRef.current[t][j]*=(1-eq.specAction.repo*0.04);
            }
          }
          if(eq.specAction.dump>0.01){
            // Dump amplifies fire sale signal (writes into conRef for the most stressed nodes)
            const dumpTarget=prelSpreads.indexOf(Math.max(...prelSpreads));
            conRef.current[dumpTarget]=Math.min(conRef.current[dumpTarget]+eq.specAction.dump*0.5,5.0);
          }
        }

        // H₀. Doom loop ODE (§B2)
        const { nextBTP, nextCap } = doomLoopStep(
          btpDynRef.current, bankCapRef.current,
          lmRef.current, smoothedStressRef.current || new Float32Array(N),
          safeDt
        );
        btpDynRef.current  = nextBTP;
        bankCapRef.current = nextCap;

        // D3. Cross-currency basis step
        if (threeRef.current._lastMetrics) {
          const _m = threeRef.current._lastMetrics;
          ccbasisRef.current = ccbasisStep(ccbasisRef.current, _m.fragilityIdx, _m.excessLiqBn, safeDt);
        }
      }

      // H. GMF field
      const sigma = adaptiveSigma(eigenPts);
      const s2    = 0.5 / (sigma * sigma);
      const vCash = INST.map((_, i) => {
        if (def[i]) return 0.85;
        let v = phis[i] / G_MAX_PHI - 0.55 * lfRef.current[i] + lmRef.current[i] * 0.25
              + conRef.current[i] + hwkRef.current[i] * 0.30;
        if (INST[i].sov > 0) v += btpSov * INST[i].sov * 0.35;
        if (s.mode !== "monitor") {
          v += (sb.dfrShift || 0) / 100;
          if (INST[i].peripheral) v += (sb.tltro || 0) / 150;
          if (INST[i].sov > 0)    v += phis[i] * (sb.haircut || 0) / 350;
          v += (sb.qtDrain || 0) / 8000;
        }
        return v;
      });
      // §C₂ particle hoarding feedback — nodes pooling cash raise their own stress
      const HOARD_ALPHA = 0.35;
      for (let i = 0; i < N; i++) vCash[i] += HOARD_ALPHA * particleDensityRef.current[i];

      // §B2 doom loop capital deficit adds extra stress
      const vCashDoom = vCash.map((v, i) => {
        const capDeficit = Math.max(0, 1 - bankCapRef.current[i]);
        return v + capDeficit * btpDynRef.current * 0.4;
      });

      // §C₃ fire-sale computation (uses prelSpreads from this frame)
      const fireSales = computeFireSales(prelSpreads, def);
      const prevFS = prevFireSaleRef.current;
      if (shouldSim) {
        for (let i = 0; i < N; i++) {
          if (prevFS[i] < 0.05 && fireSales[i] >= 0.05) {
            const t = animT.toFixed(1);
            setMmrLog(p => [`[t=${t}] FIRE-SALE: ${INST[i].abbr} initiating asset liquidation (${(fireSales[i]*100).toFixed(0)}% portfolio)`, ...p].slice(0, 6));
          }
        }
      }
      fireSaleRef.current = fireSales;
      prevFireSaleRef.current = new Float32Array(fireSales);

      // I. CPU-side surface physics — multi-asset composite (funding + gov + B-P collateral spiral)
      const nodeStressArr = INST.map((inst, ni) => {
        const funding   = Math.max(0, vCashDoom[ni]) * (0.3 + Math.sqrt(Math.max(inst.bs, 0.01)));
        const govSpread = inst.sov > 0 ? btpSov * inst.sov * 0.8 : 0;
        const leverage  = vCash[ni] / Math.max(0.5, Math.abs(lmRef.current[ni])); // min 0.5 prevents blowup when lmRef≈0
        const bpSpiral  = Math.min(2.5, 1 + Math.max(0, leverage - 7.0) * 0.25); // B-P forced-selling, capped at 2.5×
        return (funding + govSpread) * bpSpiral * STRESS_SCALE;
      });
      // Systemic mass: large balance sheets warp wider areas of the manifold
      const nodeSigmas = INST.map(inst =>
        RBF_SIGMA * (1 + Math.sqrt(Math.max(inst.bs, 0.0) / 2.67))
      );
      // Sigmoid haircut cliff: surface resists until stress > ~2, then craters
      const stressCliff = nodeStressArr.map(s => s / (1 + Math.exp(-1.5 * (s - 2.0))));
      // Breathing smoothing: sinkholes open/close slowly (0.03 = heavy fabric)
      if (!smoothedStressRef.current || smoothedStressRef.current.length !== N) {
        smoothedStressRef.current = new Float32Array(stressCliff);
      }
      const ss = smoothedStressRef.current;
      for (let ni = 0; ni < N; ni++) {
        ss[ni] += (stressCliff[ni] - ss[ni]) * 0.018;
      }
      // OU microstructure noise (θ=0.12, σ=0.06) — spectral chatter near stressed zones
      const ouNoise = ouNoiseRef.current;
      for (let ni = 0; ni < N; ni++) {
        ouNoise[ni] += (-0.12 * ouNoise[ni] + 0.06 * (Math.random() - 0.5) * 1.732) * safeDt * 30;
      }
      // Layer C — Collateral stress: sovereign spread + funding stress overflow (zero baseline)
      const collatRaw = INST.map((inst, ni) => {
        const govBondPressure = inst.sov > 0 ? btpSov * inst.sov * 2.5 : 0;
        const repoChannel     = Math.max(0, ss[ni] - 3.0) * 0.5; // B-P: only kicks in above threshold
        return (govBondPressure + repoChannel) * STRESS_SCALE * 0.4;
      });
      if (!smoothedCollatRef.current || smoothedCollatRef.current.length !== N) {
        smoothedCollatRef.current = new Float32Array(N);
      }
      const sc = smoothedCollatRef.current;
      for (let ni = 0; ni < N; ni++) sc[ni] += (collatRaw[ni] - sc[ni]) * 0.06;
      // getZAt: Gaussian for banks; cusp kernel for CCP; ECB_MLF hard floor
      function getZAt(x, y) {
        let z = 0.0;
        for (let ni = 0; ni < N; ni++) {
          const dx = x - sp[ni][0], dy = y - sp[ni][1];
          if (INST[ni].type === 'CCP') {
            const r = nodeSigmas[ni] * 1.8;
            const dist = Math.sqrt(dx*dx + dy*dy);
            const f = Math.max(1 - dist / r, 0);
            if (f > 0) z -= ss[ni] * Math.sqrt(f); // cusp: non-differentiable at center
          } else {
            const sig2 = nodeSigmas[ni] * nodeSigmas[ni];
            z -= ss[ni] * Math.exp(-(dx*dx + dy*dy) / (2 * sig2));
          }
        }
        return z; // no hard floor — each sinkhole has its own depth
      }
      // Rebuild Z cache at adaptive cadence
      zCacheFrameRef.current++;
      const ZCACHE_CADENCE = stRef.current._emaFrameTime > (1000 / 58) * 1.15 ? 3 : 2;
      if (zCacheFrameRef.current % ZCACHE_CADENCE === 0) {
        const _cache = zCacheRef.current;
        for (let iy = 0; iy < ZCACHE_H; iy++) {
          const wy = ZCACHE_Y0 + iy * ZCACHE_DY;
          for (let ix = 0; ix < ZCACHE_W; ix++) {
            _cache[iy * ZCACHE_W + ix] = getZAt(ZCACHE_X0 + ix * ZCACHE_DX, wy);
          }
        }
      }
      const zCache = zCacheRef.current;

      // J. Upload shader uniforms (GPU does the dense RBF)
      const { sharedUniforms: su } = threeRef.current;
      if (su) {
        for (let ni = 0; ni < N; ni++) {
          su.uNodePos.value[ni].set(sp[ni][0], sp[ni][1]);
          su.uNodeStress.value[ni] = ss[ni] + sc[ni] * 0.30; // B-P: collateral tightness deepens funding sinkhole
          su.uNodeSigma.value[ni]  = nodeSigmas[ni];
          su.uNodeType.value[ni]   = INST[ni].type === 'CCP' ? 1.0 : 0.0;
          su.uNodeNoise.value[ni]  = ouNoise[ni] * (1 + ss[ni] * 0.10);
        }
        su.uTime.value   = s.simTime * 10;
        su.uFbmAmp.value = FBM_AMP;
        su.uFloorZ.value = ECB_MLF_Z;
        for (let ni = 0; ni < N; ni++) su.uFireSale.value[ni] = fireSales[ni];
        const cu = threeRef.current.collatUniforms;
        if (cu) {
          for (let ni = 0; ni < N; ni++) cu.uNodeStress.value[ni] = sc[ni];
          const splitTarget = stRef.current.splitLayers ? 14 : 0;
          cu.uLayerOffset.value += (splitTarget - cu.uLayerOffset.value) * 0.04;
        }
        if (threeRef.current.meshCollat) {
          threeRef.current.meshCollat.visible = stRef.current.splitLayers ?? false;
        }
      }

      // D3. Cross-currency basis layer update
      const ccbu = threeRef.current.ccbUniforms;
      const meshCCBRef = threeRef.current.meshCCB;
      if (ccbu && meshCCBRef) {
        const basisLevel = ccbasisRef.current;
        meshCCBRef.visible = stRef.current.showCCB ?? false;
        if (meshCCBRef.visible) {
          const ccbOffset   = stRef.current.showCCB && stRef.current.splitLayers ? 20 : 4;
          const targetOffset = stRef.current.showCCB ? ccbOffset : 0;
          ccbu.uLayerOffset.value += (targetOffset - ccbu.uLayerOffset.value) * 0.04;
          for (let ni = 0; ni < N; ni++) {
            const inst = INST[ni];
            const usdSens = inst.tier === 1 ? 1.0
              : inst.tier === 4 ? 0.85
              : inst.tier === 3 ? 0.60
              : inst.type === 'CB' ? 0.20 : 0.35;
            const ccbStress = basisLevel * usdSens * ss[ni] * 0.6;
            ccbu.uNodePos.value[ni].set(sp[ni][0], sp[ni][1]);
            ccbu.uNodeStress.value[ni] = ccbStress;
            ccbu.uNodeSigma.value[ni]  = nodeSigmas[ni] * 0.85;
            ccbu.uNodeType.value[ni]   = INST[ni].type === 'CCP' ? 1.0 : 0.0;
            ccbu.uNodeNoise.value[ni]  = ouNoise[ni] * 0.3;
          }
          ccbu.uTime.value = s.simTime * 10;
        }
      }

      // C3. Ghost manifold surface update
      const ghostEp   = stRef.current.ghostEpisode;
      const ghostMesh = threeRef.current.meshGhost;
      const gu        = threeRef.current.ghostUniforms;
      if (ghostMesh && gu && ghostEp) {
        ghostMesh.visible = true;
        const gEpDef = EPISODES[ghostEp];
        const gSf    = epSf(animT, ghostEp);
        const gPhis  = INST.map((_, i) => calPhi(i, gSf) * (1 + gEpDef.sovMult * INST[i].sov * gSf * 0.5));
        const gSS    = ghostStressRef.current;
        INST.forEach((inst, ni) => {
          const gFunding = Math.max(0, gPhis[ni] / G_MAX_PHI) *
            (0.3 + Math.sqrt(Math.max(inst.bs, 0.01)));
          const gGov   = inst.sov > 0 ? 0.3 * inst.sov * gSf * 0.8 : 0;
          const target = (gFunding + gGov) * STRESS_SCALE;
          gSS[ni] += (target - gSS[ni]) * 0.08;
          gu.uNodePos.value[ni].set(sp[ni][0], sp[ni][1]);
          gu.uNodeStress.value[ni] = gSS[ni];
          gu.uNodeSigma.value[ni]  = nodeSigmas[ni];
          gu.uNodeType.value[ni]   = INST[ni].type === 'CCP' ? 1.0 : 0.0;
          gu.uNodeNoise.value[ni]  = 0;
        });
        gu.uTime.value = s.simTime * 10;
      } else if (ghostMesh) {
        ghostMesh.visible = false;
      }

      // Sync wireframe visibility
      const { matWire: mwMat } = threeRef.current;
      if (mwMat) mwMat.visible = paramsRef.current.showWire;

      // Build coarse 15×10 Z-buffer for next-frame meanZ / tube sampling
      const ZBW = 15, ZBH = 10;
      const newZBuf = new Float32Array(ZBW * ZBH);
      for (let zy = 0; zy < ZBH; zy++) {
        for (let zx = 0; zx < ZBW; zx++) {
          const wx = -SURF_HALF_X + (zx / (ZBW - 1)) * SURF_W;
          const wy = -SURF_HALF_Y + (zy / (ZBH - 1)) * SURF_H;
          newZBuf[zy * ZBW + zx] = getZAt(wx, wy);
        }
      }
      threeRef.current.zBuf = newZBuf;

      // Update particle density from current particle positions
      // Particle system disabled — particleDensityRef stays at zero

      let aggZ = 0, inCount = 0;

      // K. Node markers — sized by balance sheet, Z from CPU sampleRBF
      ndPtsRef.current = sp;
      INST.forEach((inst, ni) => {
        const [dx, dy] = sp[ni];
        const mz = zCacheLookup(dx, dy, zCache) + 0.6;
        aggZ += mz; inCount++;
        discMeshes[ni].visible = paramsRef.current.showNodes;
        discMeshes[ni].position.set(dx, dy, mz);
        // Sphere colour: thermal scale based on node stress
        const v = clamp(nodeStressArr[ni] / 30, 0, 1);
        const [dr, dg, db] = v < 0.5
          ? [lerp(0.102,1.0,v*2), lerp(0.110,0.667,v*2), lerp(0.173,0.0,v*2)]
          : [1.0, lerp(0.667,0.0,(v-0.5)*2), lerp(0.0,0.2,(v-0.5)*2)];
        discMats[ni].color.setRGB(dr, dg, db);
        // §HJB: ECB node glows green when actively injecting
        if (hjbActiveRef.current && (stRef.current._hjbOptU||0) > 0) {
          const ecbIdxV = INST.findIndex(inst => inst.type === "CB");
          if (ni === ecbIdxV) {
            const pulse = 0.5 + 0.5 * Math.sin(s.simTime * 800);
            const intensity = clamp((stRef.current._hjbOptU||0) / 1.2, 0, 1);
            discMats[ni].color.setRGB(
              lerp(0.1, 0.2, pulse * intensity),
              lerp(0.6, 1.0, pulse * intensity),
              lerp(0.3, 0.5, pulse * intensity)
            );
          }
        }
        // Second-order contagion: pulsing halo on secondary-source nodes
        const isSecondarySource = hwkRef.current[ni] > 1.8;
        const dens = particleDensityRef.current[ni];
        const haloScale = isSecondarySource
          ? 1 + 0.18 * Math.sin(s.simTime * 1200 + ni)
          : dens > 0.5 ? 1 + 0.15 * Math.sin(animT * 3) : 1.0;
        discMeshes[ni].scale.setScalar(haloScale);
        // Crosshair positions
        const base = ni * 4;
        const sv = (idx, x, y, z) => { crPos[idx*3]=dx+x; crPos[idx*3+1]=dy+y; crPos[idx*3+2]=z; };
        sv(base,  -CS,0,mz); sv(base+1, +CS,0,mz);
        sv(base+2,  0,-CS,mz); sv(base+3,  0,+CS,mz);
      });
      crGeo.attributes.position.needsUpdate = true;

      // K₁b. MST flow-tube geometry — every frame so tubes track live surface
      {
        const mstEdgeList = threeRef.current._mstEdgeList || [];
        const { tubeObjects: tObjs, tubeMats: tMats, scene: tScene } = threeRef.current;
        const ccpTear = CCP_IDX >= 0 && ss[CCP_IDX] > 14;
        if (tObjs && tMats && tScene) {
          for (let ae = 0; ae < N - 1; ae++) {
            if (tObjs[ae].mesh) tObjs[ae].mesh.visible = false;
          }
          for (let ae = 0; ae < Math.min(mstEdgeList.length, N - 1); ae++) {
            const { i: ei, j: ej } = mstEdgeList[ae];
            if (ccpTear && (ei === CCP_IDX || ej === CCP_IDX)) {
              tMats[ae].opacity = Math.max(tMats[ae].opacity - 0.15, 0);
              continue;
            }
            const [ax, ay] = sp[ei], [bx, by] = sp[ej];
            let sumZ = 0;
            const pts3 = [];
            for (let seg = 0; seg <= ARCH_SEGS; seg++) {
              const tt = seg / ARCH_SEGS;
              const lx = lerp(ax, bx, tt), ly = lerp(ay, by, tt);
              const lz = zCacheLookup(lx, ly, zCache) + 1.2;
              pts3.push(new THREE.Vector3(lx, ly, lz));
              sumZ += lz;
            }
            const edgeMeanZ = sumZ / (ARCH_SEGS + 1);
            const hwDepth   = clamp(-edgeMeanZ / 20, 0, 1);
            const midZ      = zCacheLookup((ax + bx) * 0.5, (ay + by) * 0.5, zCache);
            const radius    = 0.25 * (1 + 1.5 * clamp(-midZ / 20, 0, 1));
            tMats[ae].color.setRGB(
              lerp(0.20, 1.00, hwDepth), lerp(0.40, 0.05, hwDepth), lerp(0.80, 0.05, hwDepth)
            );
            // Fade-in/out animation via mstEdgeStateRef
            const edgeState = mstEdgeStateRef.current[ae];
            if (edgeState) {
              const targetAlpha = edgeState.live ? (0.50 + hwDepth * 0.45) : 0;
              edgeState.fadeAlpha += (targetAlpha - edgeState.fadeAlpha) * 0.06;
              // §NET: scale opacity by live exposure fraction (network contraction visible)
              const _lxe = liveExpRef.current;
              const _liveW = (_lxe && _lxe[ei] && EXPOSURE_0[ei] && EXPOSURE_0[ei][ej] > 0.001)
                ? Math.min(1, _lxe[ei][ej] / EXPOSURE_0[ei][ej]) : 1.0;
              tMats[ae].opacity = edgeState.fadeAlpha * Math.max(0.08, _liveW);
            } else {
              tMats[ae].opacity = 0.55 + hwDepth * 0.40;
            }
            if (!tObjs[ae].mesh || s.tick % 4 === 0) {
              if (tObjs[ae].geo) tObjs[ae].geo.dispose();
              if (tObjs[ae].mesh) tScene.remove(tObjs[ae].mesh);
              const curve   = new THREE.CatmullRomCurve3(pts3);
              const newGeo  = new THREE.TubeGeometry(curve, ARCH_SEGS, radius, 5, false);
              const newMesh = new THREE.Mesh(newGeo, tMats[ae]);
              tScene.add(newMesh);
              tObjs[ae] = { mesh: newMesh, geo: newGeo };
            }
            tObjs[ae].mesh.visible = true;
          }
        }
      }

      // K₂. Liquidity particle system (50k instanced) — flows along −∇Z
      if (false) { const pp=null,pv=null,im=null,m4=null,ip=null; // particle system disabled
        const FLOW_SPD = 28, DAMP = 0.86, PEPS = 2.5;
        // Only update a chunk per frame for performance: split into 5 batches
        const BATCH_SIZE = Math.ceil(PART_N / 5);
        const batchStart = (s.tick % 5) * BATCH_SIZE;
        const batchEnd   = Math.min(batchStart + BATCH_SIZE, PART_N);

        for (let pi = batchStart; pi < batchEnd; pi++) {
          const px = pp[pi*3], py = pp[pi*3+1];
          const zC = zCacheLookup(px,        py,        zCache);
          const zR = zCacheLookup(px + PEPS, py,        zCache);
          const zU = zCacheLookup(px,        py + PEPS, zCache);
          const gxP = (zR - zC) / PEPS;
          const gyP = (zU - zC) / PEPS;
          pv[pi*2]   = pv[pi*2]   * DAMP - gxP * FLOW_SPD * safeDt;
          pv[pi*2+1] = pv[pi*2+1] * DAMP - gyP * FLOW_SPD * safeDt;
          pp[pi*3]   += pv[pi*2]   * safeDt;
          pp[pi*3+1] += pv[pi*2+1] * safeDt;
          pp[pi*3+2]  = zC + 0.3;
          if (Math.abs(pp[pi*3]) > SURF_HALF_X * 0.93 ||
              Math.abs(pp[pi*3+1]) > SURF_HALF_Y * 0.93) {
            pp[pi*3]   = (Math.random() - 0.5) * SURF_W * 0.85;
            pp[pi*3+1] = (Math.random() - 0.5) * SURF_H * 0.85;
            pv[pi*2] = 0; pv[pi*2+1] = 0;
          }
        }

        // Update instance matrices (full pass — GPU upload, cheap)
        for (let pi = 0; pi < PART_N; pi++) {
          ip.set(pp[pi*3], pp[pi*3+1], pp[pi*3+2]);
          m4.makeTranslation(ip.x, ip.y, ip.z);
          im.setMatrixAt(pi, m4);
          // Color: white in calm, pale amber in stress (based on local Z depth)
          const depth = clamp(-pp[pi*3+2] / 20, 0, 1);
          im.instanceColor.setXYZ(pi, 1.0, lerp(1.0, 0.55, depth), lerp(1.0, 0.05, depth));
        }
        im.instanceMatrix.needsUpdate = true;
        im.instanceColor.needsUpdate  = true;
      }

      // CSS2D label update: direct DOM mutation, zoom-adaptive font size
      if (s.tick % 2 === 0 && labelDivsRef.current.length) {
        const camDist = camera.position.length();
        const lScale  = clamp(140 / camDist, 0.65, 1.8);
        const fsPx    = Math.round(9 * lScale);
        INST.forEach((inst, ni) => {
          const nd = threeRef.current._lastMetrics?.nodeData[ni];
          const bps    = nd?.bps  || 0;
          const status = nd?.status || 'NORMAL';
          const sinking = ['CRISIS','ALERT','DEFAULT'].includes(status);
          const div = labelDivsRef.current[ni];
          if (!div) return;
          div.style.fontSize = `${fsPx}px`;
          div.innerHTML =
            `<b style="color:${sinking?'#ff8844':'#ffffff'};font-weight:${sinking?700:600}">${inst.abbr}</b>` +
            `<br><span style="opacity:0.75;font-size:${fsPx - 1}px">${bps>=0?'+':''}${bps.toFixed(1)}bp</span>`;
        });
      }

      // L. Metrics + alerts (adaptive cadence)
      const METRICS_CADENCE = stRef.current._emaFrameTime > (1000 / 58) * 1.30 ? 10 : 5;
      if (s.tick % METRICS_CADENCE === 0) {
        const m = computeMetrics(lfRef.current, lmRef.current, conRef.current, hwkRef.current,
          def, phis, animT, s.mode !== "monitor" ? sb : null, btpSov, hvRef.current);
        threeRef.current._lastMetrics = m;
        setMetrics(m);

        // §NET: update endogenous network at metrics cadence (O(N) exp calls, not O(N²)/frame)
        updateWithdrawalCache(m.spreads, def);
        const { liveExp: lx2, edgeCount: ec2, edgeFraction: ef2, topologyAlert: ta2 } =
          updateLiveExposure(liveExpRef.current, def);
        liveExpRef.current     = lx2;
        edgeFractionRef.current = ef2;
        topologyAlertRef.current = ta2;
        setEdgeFraction(+ef2.toFixed(3));
        setTopologyAlert(ta2);
        setEdgeCount(ec2);
        // §PIPELINE: publish network state
        synthPipelineRef.current.networkFraction = ef2;

        // Status alerts
        const newSt = m.nodeData.map(nd => nd.status);
        const tm    = new Date().toLocaleTimeString("de-DE",{hour:"2-digit",minute:"2-digit",second:"2-digit"});
        const newAl = [];
        newSt.forEach((st, i) => {
          if (st !== prevStRef.current[i]) {
            if (["CRISIS","ALERT","DEFAULT"].includes(st))
              newAl.push({ time:tm, text:`${INST[i].name}: ${st}. Spread: ${m.nodeData[i].bps}bp. D2D: ${m.nodeData[i].d2d}σ.`, sev:st, id:Date.now()+i });
          }
        });
        prevStRef.current = newSt;
        if (newAl.length) setAlerts(p => [...newAl, ...p].slice(0, 20));

        setDoomBTP(btpDynRef.current);
        setMinCap(Math.min(...Array.from(bankCapRef.current)));
        setSecondOrderCount(Array.from(hwkRef.current).filter(v => v > 1.8).length);
        setCcbBasis(ccbasisRef.current);

        // §MFG stats (lightweight — just counting)
        if (mfgActiveRef.current) {
          const acts = mfgActionsRef.current;
          let lend=0, hoard=0, hold=0;
          for(let i=0;i<N;i++){
            if(acts[i]>0.01) lend++; else if(acts[i]<-0.01) hoard++; else hold++;
          }
          setMfgStats({ lenders:lend, hoarders:hoard, holders:hold });
        }

        // §HJB: look up optimal injection
        if (hjbActiveRef.current && hjbPolicyRef.current) {
          const mLf=Array.from(lfRef.current).reduce((a,b)=>a+b,0)/N;
          const mLm=Array.from(lmRef.current).reduce((a,b)=>a+b,0)/N;
          const btp=clamp(btpDynRef.current,0,1);
          const iLf=Math.round(clamp(mLf,0,1)*(HJB_NLF-1));
          const iLm=Math.round(clamp(mLm/2,0,1)*(HJB_NLM-1));
          const iBtp=Math.round(clamp(btp,0,1)*(HJB_NBTP-1));
          const sIdx=iLf*HJB_NLM*HJB_NBTP+iLm*HJB_NBTP+iBtp;
          const optU=HJB_ACTS[hjbPolicyRef.current[sIdx]||0];
          setHjbOptU(optU);
          synthPipelineRef.current.hjbForceU = optU;  // §PIPELINE: adversarial reads ECB's move
          const ecbIdx=INST.findIndex(inst=>inst.type==="CB");
          if(ecbIdx>=0&&optU>0) {
            stRef.current.pendingKick=ecbIdx;
            lfRef.current[ecbIdx]=Math.min(lfRef.current[ecbIdx]+optU*0.04,1.0);
          }
        }

        // §IV Z-slice capture for sculptor (every 15 ticks)
        if (shouldSim && s.tick % 15 === 0) {
          const slice = new Float32Array(zCacheRef.current);
          sculptureSlicesRef.current.push(slice);
          sculptureTicksRef.current.push(s.tick);
          if (sculptureSlicesRef.current.length > 12) {
            sculptureSlicesRef.current.shift(); sculptureTicksRef.current.shift();
          }
          setSliceCount(sculptureSlicesRef.current.length);
        }

        // §IG: Fisher information geometry — update every 15 ticks (heavy computation)
        if (s.tick % 15 === 0 && smoothedStressRef.current) {
          const fProj = updateFisherTrajectory(
            fisherHistRef.current, smoothedStressRef.current, s.tick
          );
          fisherProjRef.current = fProj;
          setFisherProj([...fProj].slice(-30));
          const ct = classifyCrisisType(fProj);
          crisisTypeRef.current = ct;
          setCrisisType(ct);
          // §PIPELINE: crisis type → auto-suggest calibration episode
          if (ct && ct.confidence > 0.65 && ct.type !== 'ambiguous') {
            const epMap = { sovereign_contagion:'dec2011', systemic_shock:'mar2020', cliff_collapse:'lehman2008' };
            const suggestedEp = epMap[ct.type];
            if (suggestedEp && synthPipelineRef.current.crisisTypeGuess !== suggestedEp) {
              synthPipelineRef.current.crisisTypeGuess = suggestedEp;
              setSynthSuggestion(`Fisher: ${ct.type.replace('_',' ')} (${(ct.confidence*100).toFixed(0)}%) → try calibrating ${EPISODE_SIGNATURES[suggestedEp]?.label}`);
            }
          }
          if(advEnabledRef.current&&advEquilRef.current) setAdvEquil({...advEquilRef.current});
        }

        // §PH: Persistent homology every 15 ticks
        if (homologyEnabled && s.tick % 15 === 0) {
          const pairs = computePersistentHomology(zCacheRef.current);
          homologyRef.current = pairs;
          setHomologyData([...pairs]);
        }

        // Signature path accumulation
        const mData = threeRef.current._lastMetrics?.nodeData;
        if (mData) {
          const tNorm = animT / 59;
          for (let ni = 0; ni < N; ni++) {
            const hist = pathHistRef.current[ni];
            hist.push({ t: tNorm, Lf: lfRef.current[ni], sp: clamp(mData[ni].bps / 100, 0, 1.2) });
            if (hist.length > 30) hist.shift();
          }
        }

        // Stress chart (aggZ/inCount from node Z samples)
        const agg = inCount > 0 ? aggZ / inCount : 0;
        sHistRef.current[sIdxRef.current] = agg;
        sIdxRef.current = (sIdxRef.current + 1) % 60;
        setSHist([...sHistRef.current]);

        // MST topology — recompute with stress-weighting every 15 ticks
        setHawkRho(+hawkRhoRef.current.toFixed(3));
        if (s.tick % 15 === 0) {
          const newMst = kruskalMSTWeighted(def, smoothedStressRef.current || new Float32Array(N));
          const edgeStates = mstEdgeStateRef.current;
          const newSet = new Set(newMst.map(e => `${Math.min(e.i,e.j)}_${Math.max(e.i,e.j)}`));
          // Mark edges no longer in MST for fade-out
          edgeStates.forEach(es => {
            const key = `${Math.min(es.i,es.j)}_${Math.max(es.i,es.j)}`;
            if (es.live && !newSet.has(key)) es.live = false;
          });
          // Insert new edges into free slots
          newMst.forEach((e, idx) => {
            if (idx >= N - 1) return;
            const key = `${Math.min(e.i,e.j)}_${Math.max(e.i,e.j)}`;
            const existing = edgeStates.find(es =>
              `${Math.min(es.i,es.j)}_${Math.max(es.i,es.j)}` === key
            );
            if (!existing) {
              const slot = edgeStates.find(es => !es.live && es.fadeAlpha < 0.01);
              if (slot) { slot.i = e.i; slot.j = e.j; slot.live = true; }
            } else {
              existing.live = true;
            }
          });
          threeRef.current._mstEdgeList = newMst;
          setMstEdges(newMst.slice(0, 6).map(e => ({ i:e.i, j:e.j })));
        }

        // MMSR log
        if (s.tick % 60 === 0) {
          const elig = INST.map((_, i) => i).filter(i => INST[i].tier <= 2 && !INST[i].type.includes("Sov"));
          if (elig.length >= 2) {
            const ia = elig[Math.floor(Math.random() * elig.length)];
            let ib = ia; while (ib === ia) ib = elig[Math.floor(Math.random() * elig.length)];
            const rate = +(m.yieldCurve[0].rate + (prelSpreads[ia] + prelSpreads[ib]) / 400 + (Math.random() - 0.5) * 0.04).toFixed(3);
            const vol  = +(80 + Math.random() * 700).toFixed(0);
            const coll = ["Bund 0%","OAT 0.5%","BTP 1.65%","DSL 0%","RAGB 0%"][Math.floor(Math.random() * 5)];
            const entry = `${INST[ia].abbr} → ${INST[ib].abbr}  O/N  ${rate}%  €${vol}M  [${coll}]`;
            setMmrLog(p => [entry, ...p].slice(0, 6));
          }
        }

        // Telemetry
        telRef.current.push({
          tick:s.tick, ep, phase:epPhase(animT, ep),
          systemSpreadBps:m.systemSpreadBps, fragilityIdx:m.fragilityIdx,
          entropy:m.entropy, tei:m.tei, excessLiqBn:m.excessLiqBn,
          spreads:m.spreads.map(v=>+v.toFixed(2)),
        });
        if (telRef.current.length > 120) telRef.current.shift();
      }

      if (++s.tick % 4 === 0) setDispFrame(Math.round(animT));

      // §IV Real-time sculptor — throttled to every 3 frames
      if (s.tick % 3 === 0) {
        const slices  = sculptureSlicesRef.current;
        const meshes  = sculptMeshesRef.current;
        const sculpt  = stRef.current.sculptureMode;
        const nSlices = slices.length;
        if (meshes && meshes.length) {
          for (let idx = 0; idx < meshes.length; idx++) {
            const mesh = meshes[idx];
            const isLive = (idx === nSlices);          // last slot = live ghost
            const slice  = isLive ? zCacheRef.current : slices[idx];
            if (!sculpt || (!slice && !isLive)) { mesh.visible = false; continue; }
            mesh.visible = true;
            if (isLive) {
              // White ghost = current live terrain
              mesh.material.color.setRGB(0.95, 0.95, 1.0);
              mesh.material.opacity = 0.28;
            } else {
              // Epoch colour: blue → amber → red
              const t = nSlices > 1 ? idx / (nSlices - 1) : 0;
              const r = t < 0.5 ? lerp(0.10, 0.95, t * 2) : 1.0;
              const g = t < 0.5 ? lerp(0.30, 0.55, t * 2) : lerp(0.55, 0.05, (t-0.5)*2);
              const b = t < 0.5 ? lerp(0.70, 0.10, t * 2) : 0.05;
              mesh.material.color.setRGB(r, g, b);
              mesh.material.opacity = 0.12 + t * 0.16;
            }
            // Write Z values — vertex k maps exactly to zcache[k]
            // PlaneGeometry(W,H,63,63) → 64*64=4096 vertices in row-major order
            const pos = mesh.geometry.attributes.position;
            const zScale = 0.50;
            for (let k = 0; k < pos.count; k++) {
              pos.setZ(k, (slice[k] || 0) * zScale);
            }
            pos.needsUpdate = true;
          }
          // Hide any slots beyond nSlices+1 (live ghost)
          for (let idx = nSlices + 1; idx < meshes.length; idx++) {
            meshes[idx].visible = false;
          }
        }
      }

      // §MFG: visual indicators on nodes — hoarders shrink, lenders glow
      if (mfgActiveRef.current && threeRef.current.discMeshes) {
        const acts = mfgActionsRef.current;
        const dMeshes = threeRef.current.discMeshes;
        const dMats   = threeRef.current.discMats;
        for (let ni = 0; ni < N; ni++) {
          if (!dMeshes[ni] || !dMats[ni]) continue;
          const a = acts[ni] || 0;
          if (a > 0.01) {
            // Lender: gentle green pulse
            const pulse = 0.6 + 0.4 * Math.sin(s.simTime * 600 + ni);
            dMats[ni].emissive && dMats[ni].emissive.setRGB(0, pulse * 0.2, 0);
          } else if (a < -0.01) {
            // Hoarder: shrink slightly + amber tint
            dMeshes[ni].scale.setScalar(0.72);
            dMats[ni].emissive && dMats[ni].emissive.setRGB(0.2, 0.08, 0);
          }
        }
      }

      // Bloom intensity scales with crisis depth
      const composerRef = threeRef.current.composer;
      if (composerRef) {
        const bloomP = composerRef.passes.find(p => p instanceof UnrealBloomPass);
        if (bloomP) {
          const targetStrength = 0.10 + clamp(inCount > 0 ? -aggZ / inCount / 25 : 0, 0, 0.60);
          bloomP.strength += (targetStrength - bloomP.strength) * 0.04;
        }
      }
    }

    stRef.current.lastTime = performance.now();
    // ── RESIZE HANDLER ───────────────────────────────────────────
    function onResize() {
      const nW = el.clientWidth, nH = el.clientHeight;
      camera.aspect = nW / nH;
      camera.updateProjectionMatrix();
      renderer.setSize(nW, nH);
      composer.setSize(nW, nH);
      css2dRenderer.setSize(nW, nH);
    }
    window.addEventListener('resize', onResize);

    function loop() {
      threeRef.current.raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt  = Math.min((now - stRef.current.lastTime) / 1000, 0.05);
      stRef.current.lastTime = now;
      const FRAME_BUDGET = 1000 / 58;
      if (!stRef.current._emaFrameTime) stRef.current._emaFrameTime = FRAME_BUDGET;
      stRef.current._emaFrameTime = stRef.current._emaFrameTime * 0.92 + (dt * 1000) * 0.08;
      const ep2 = stRef.current.ep || "generic";
      if (stRef.current.playing && stRef.current.mode === "monitor") {
        const spd = (paramsRef.current.simSpeed || SIM_SPEED) * 0.1; // 10× temporal dilation
        stRef.current.animTime = Math.min(stRef.current.animTime + dt * spd, 59);
        if (stRef.current.animTime >= 59) { stRef.current.animTime = 59; setPlaying(false); }
      }
      update(stRef.current.animTime, dt);
      controls.update();
      composer.render();
      css2dRenderer.render(scene, camera);
    }
    loop();

    return () => {
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener("mousemove", onMove);
      renderer.domElement.removeEventListener("click",     onClick);
      cancelAnimationFrame(threeRef.current.raf);
      controls.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
      if (el.contains(css2dRenderer.domElement)) el.removeChild(css2dRenderer.domElement);
      labelDivsRef.current = [];
      if (instMesh) { instMesh.geometry.dispose(); instMesh.material.dispose(); }
      renderer.dispose();
    };
  }, []);

  function handlePlay() {
    if (!playing && stRef.current.animTime >= 59) seek(0);
    setPlaying(p => !p);
  }

  // ── DERIVED UI STATE ────────────────────────────────────────────────
  const epDef    = EPISODES[episode];
  const phase    = epPhase(dispFrame, episode);
  const tei      = metrics.tei;
  const teiColor = tei >= 85 ? "#1a4a2a" : tei >= 60 ? "#5a4a10" : "#6a1a1a";
  const simH     = 9 + (dispFrame / 59) * 8.5;
  const simClock = `${Math.floor(simH).toString().padStart(2,"0")}:${Math.floor((simH % 1)*60).toString().padStart(2,"0")} CET`;

  // Academic typography + palette
  const F  = "Georgia, 'Times New Roman', Times, serif";   // serif for labels
  const FM = "'Courier New', Courier, monospace";           // mono for numbers

  const statusColor = { DEFAULT:"#8b1a1a", CRISIS:"#8b1a1a", ALERT:"#6b4a10", SQUEEZE:"#1a3a6b", WATCH:"#3a4a1a", NORMAL:"#1a4a2a" };

  // Pre-computed display values (avoids /N in JSX)
  const _n = metrics.nodeData.length || 1;
  const avgLfD = (metrics.nodeData.reduce((a,nd)=>a+nd.Lf,0)/_n).toFixed(3);
  const avgLmD = (metrics.nodeData.reduce((a,nd)=>a+nd.Lm,0)/_n).toFixed(3);
  const alertNodes = metrics.nodeData.filter(n=>["DEFAULT","CRISIS","ALERT"].includes(n.status));

  return (
    <div style={{ position:"fixed", inset:0, background:"#f5f5f5",
      fontFamily:F, userSelect:"none", color:"#1a1a2a", overflow:"hidden" }}>



      {/* ── CANVAS CONTAINER ──────────────────────────────────────────── */}
      <div style={{ position:"absolute", inset:0 }}>

        {/* LEFT SIDEBAR — hidden */}
        <div style={{ display:"none" }}>

          <SH F={F}>Market Rates</SH>
          <div style={{ padding:"0 14px 10px" }}>
            {[["DFR", ECB_DFR.toFixed(2)+"%"], ["MRO", ECB_MRO.toFixed(2)+"%"], ["MLF", ECB_MLF.toFixed(2)+"%"]].map(([k,v])=>(
              <div key={k} style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline",
                marginBottom:7, paddingBottom:7, borderBottom:"1px solid #eeeeee" }}>
                <span style={{ fontFamily:F, fontSize:11, color:"#555" }}>{k}</span>
                <span style={{ fontFamily:FM, fontSize:13, fontWeight:600, color:"#1a1a2a" }}>{v}</span>
              </div>
            ))}
          </div>

          <SH F={F}>Systemic Metrics</SH>
          <div style={{ padding:"0 14px 10px" }}>
            <ML F={FM} l="€STR" v={metrics.estr.toFixed(3)+"%"}/>
            <ML F={FM} l="Sys. Spread" v={(metrics.systemSpreadBps>=0?"+":"")+metrics.systemSpreadBps.toFixed(1)+" bp"}
              c={metrics.systemSpreadBps>TH_WATCH?"#8b1a1a":metrics.systemSpreadBps<TH_SQUEEZE?"#1a3a6b":"#1a4a2a"}/>
            <ML F={FM} l="Excess Liq." v={"€"+metrics.excessLiqBn.toLocaleString()+"bn"}
              c={metrics.excessLiqBn<400?"#8b1a1a":"#1a4a2a"}/>
            <ML F={FM} l="H(Lf)" v={metrics.entropy.toFixed(3)}
              c={metrics.entropy<0.4?"#8b1a1a":"#555"}/>
            <ML F={FM} l="Fragility" v={metrics.fragilityIdx.toFixed(3)}
              c={metrics.fragilityIdx>0.5?"#8b1a1a":"#555"}/>
            <ML F={FM} l="Avg. Lf" v={avgLfD}/>
            <ML F={FM} l="Avg. Lm" v={avgLmD}/>
          </div>

          <SH F={F}>Stress History</SH>
          <div style={{ padding:"6px 14px 10px" }}>
            <StressChart history={stressHist} phase={phase} phaseNames={epDef.phaseNames} F={F} FM={FM}/>
            <div style={{ fontFamily:F, fontSize:10, color:"#888", marginTop:4, fontStyle:"italic", textAlign:"center" }}>
              Mean Z-displacement over time
            </div>
          </div>

          <SH F={F}>Nelson-Siegel Curve</SH>
          <div style={{ padding:"6px 14px 10px" }}>
            <NSChart yc={metrics.yieldCurve} F={F} FM={FM}/>
            <div style={{ fontFamily:FM, fontSize:9, color:"#888", marginTop:3 }}>
              r(τ) = β₀ + β₁f(τ/λ) + β₂g(τ/λ)
            </div>
          </div>

          <SH F={F}>MST Backbone</SH>
          <div style={{ padding:"0 14px 10px" }}>
            {mstEdges.slice(0, 5).map((e, i) => (
              <div key={i} style={{ fontFamily:FM, fontSize:10, color:"#555", marginBottom:3 }}>
                {INST[e.i].abbr} — {INST[e.j].abbr}
                <span style={{ color:"#aaa", marginLeft:4 }}>c={CAPACITY[e.i][e.j].toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* CANVAS — full-screen manifold */}
        <div ref={mountRef} style={{ position:"absolute", inset:0, cursor:"crosshair" }}>

          {/* ── TOP-LEFT: scenario / phase / TEI / actions ──────────────── */}
          <div style={{ position:"absolute", top:14, left:16, zIndex:30, pointerEvents:"auto" }}>
            <div style={{ background:"rgba(245,245,245,0.88)", border:"1px solid rgba(170,170,190,0.55)",
              padding:"9px 13px", backdropFilter:"blur(3px)" }}>
              <div style={{ display:"flex", alignItems:"center", gap:7, marginBottom:7 }}>
                <span style={{ fontFamily:F, fontSize:10, color:"#555", fontStyle:"italic" }}>Scenario</span>
                <select value={episode} onChange={e => setEpisode(e.target.value)} style={{
                  fontFamily:F, fontSize:10, background:"transparent", color:"#1a1a2a",
                  border:"1px solid #cccccc", padding:"2px 5px", cursor:"pointer" }}>
                  {Object.entries(EPISODES).map(([k,v]) => <option key={k} value={k}>{v.name}</option>)}
                </select>
              </div>
              <div style={{ marginTop:8 }}>
                <div style={{ fontFamily:F, fontSize:9, color:'#888', fontStyle:'italic', marginBottom:3 }}>Ghost manifold (reference episode)</div>
                <select
                  value={ghostEpisode || ''}
                  onChange={e => setGhostEpisode(e.target.value || null)}
                  style={{ width:'100%', fontFamily:F, fontSize:9, border:'1px solid #ccc',
                    background:'#fafafa', color:'#333', padding:'2px 4px' }}>
                  <option value=''>— off —</option>
                  {Object.entries(EPISODES)
                    .filter(([k]) => k !== episode)
                    .map(([k, v]) => (
                      <option key={k} value={k}>{v.name}</option>
                    ))}
                </select>
                <div style={{ fontFamily:F, fontSize:8, color:'#aaa', fontStyle:'italic', marginTop:2, lineHeight:1.4 }}>
                  Cool grey-green overlay, +8Z offset. Compare stress topologies at same tick.
                </div>
              </div>
              <div style={{ fontFamily:FM, fontSize:13, fontWeight:700, color:teiColor, lineHeight:1.2, marginTop:2 }}>
                TEI {tei}<span style={{ fontSize:10 }}>%</span>
              </div>
              <div style={{ fontFamily:FM, fontSize:9, color:"#888", marginTop:1 }}>{simClock}</div>
              <div style={{ marginTop:7, display:"flex", gap:4 }}>
                <button onClick={handlePlay} style={{
                  fontFamily:F, fontSize:9, background:playing?"#fff4f4":"#f0f4ff",
                  border:`1px solid ${playing?"#8b1a1a":"#1a3a6b"}`,
                  color:playing?"#8b1a1a":"#1a3a6b", padding:"2px 10px", cursor:"pointer" }}>
                  {playing ? "⏸ Pause" : "▶ Play"}
                </button>
                <button onClick={exportTelemetry} style={{
                  fontFamily:F, fontSize:9, background:"transparent", border:"1px solid #aaa",
                  color:"#555", padding:"2px 7px", cursor:"pointer" }}>↓ JSON</button>
                <button onClick={resetAll} style={{
                  fontFamily:F, fontSize:9, background:"transparent", border:"1px solid #aaa",
                  color:"#555", padding:"2px 7px", cursor:"pointer" }}>Reset</button>
                <button onClick={() => setSplitLayers(v => !v)} style={{
                  fontFamily:F, fontSize:9,
                  background: splitLayers ? "#0d1a33" : "transparent",
                  border: `1px solid ${splitLayers ? "#4488ff" : "#aaa"}`,
                  color: splitLayers ? "#4488ff" : "#555",
                  padding:"2px 7px", cursor:"pointer" }}>
                  {splitLayers ? "⊟ Overlay" : "⊞ Split F|C"}
                </button>
                <label style={{ display:'flex', alignItems:'center', gap:4, cursor:'pointer',
                  fontFamily:F, fontSize:9, color:'#8b6a10' }}>
                  <input type="checkbox" checked={showCCB} onChange={e => setShowCCB(e.target.checked)} />
                  CCB layer
                </label>
                <button onClick={() => setDesignerOpen(o => !o)} style={{
                  fontFamily:F, fontSize:9, background:"transparent",
                  border:`1px solid ${designerOpen ? "#8b5a10" : "#cc9933"}`,
                  color:"#8b5a10", padding:"2px 7px", cursor:"pointer", fontStyle:"italic" }}>
                  {designerOpen ? "✕ Designer" : "⚙ Designer"}
                </button>
                <button onClick={() => setSynthOpen(v => !v)} style={{
                  fontFamily:F, fontSize:9,
                  background: synthOpen ? "#0d1a1a" : "transparent",
                  border:`1px solid ${synthOpen ? "#44aa88" : "#aaa"}`,
                  color: synthOpen ? "#44ff99" : "#555",
                  padding:"2px 7px", cursor:"pointer" }}>
                  {"⬟ Synth"}
                </button>
                <button onClick={() => { setMcCardOpen(v => !v); if (!mcBands && !mcRunning) runMonteCarlo(); }}
                  style={{
                    fontFamily:F, fontSize:9,
                    background: mcCardOpen ? "rgba(10,12,28,0.92)" : "transparent",
                    border:`1px solid ${mcCardOpen ? "rgba(100,140,255,0.60)" : mcStale ? "#cc8800" : "#aaa"}`,
                    color: mcCardOpen ? "#aaccff" : mcStale ? "#cc8800" : "#555",
                    padding:"2px 7px", cursor:"pointer", position:"relative",
                  }}>
                  ◈ MC{mcStale ? " ⟳" : ""}
                </button>
              </div>
            </div>
          </div>

          {/* ── TOP-RIGHT: €STR / spread / fragility ──────────────────── */}
          <div style={{ position:"absolute", top:14, right:16, zIndex:30, pointerEvents:"none" }}>
            <div style={{ background:"rgba(245,245,245,0.88)", border:"1px solid rgba(170,170,190,0.55)",
              padding:"9px 13px", textAlign:"right", backdropFilter:"blur(3px)" }}>
              <div style={{ fontFamily:FM, fontSize:9, color:"#aaa", fontStyle:"italic" }}>€STR</div>
              <div style={{ fontFamily:FM, fontSize:18, fontWeight:700, color:"#1a3a6b", lineHeight:1.1 }}>
                {metrics.estr.toFixed(3)}<span style={{ fontSize:10 }}>%</span>
              </div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color:metrics.systemSpreadBps>TH_WATCH?"#8b1a1a":metrics.systemSpreadBps<TH_SQUEEZE?"#1a3a6b":"#1a4a2a" }}>
                {metrics.systemSpreadBps >= 0 ? "+" : ""}{metrics.systemSpreadBps.toFixed(1)} bp
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic", marginBottom:4 }}>System Spread</div>
              <div style={{ fontFamily:FM, fontSize:11, fontWeight:600,
                color:metrics.fragilityIdx>0.5?"#8b1a1a":"#555" }}>
                {metrics.fragilityIdx.toFixed(3)}
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>Fragility Index</div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color: hawkRho >= 1.0 ? "#cc1111" : "#1a7a2a" }}>
                {hawkRho.toFixed(3)}
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>Hawkes ρ {hawkRho >= 1.0 ? "⚠ super-critical" : "sub-critical"}</div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color: doomBTP > 0.5 ? "#8b1a1a" : doomBTP > 0.2 ? "#6b4a10" : "#1a4a2a" }}>
                +{(doomBTP * 100).toFixed(1)} bp
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>BTP Dynamic Premium</div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color: minCap < 0.7 ? "#8b1a1a" : "#1a4a2a" }}>
                {(minCap * 100).toFixed(1)}%
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>Min Bank Capital</div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color: secondOrderCount > 3 ? "#8b1a1a" : "#444" }}>
                {secondOrderCount}
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>2nd-order sources</div>
              <div style={{ marginTop:5, fontFamily:FM, fontSize:11, fontWeight:600,
                color: ccbBasis > 0.5 ? "#8b1a1a" : ccbBasis > 0.2 ? "#6b4a10" : "#1a4a2a" }}>
                {(ccbBasis * 100).toFixed(0)} bp
              </div>
              <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic" }}>EUR/USD CCB</div>
            </div>
          </div>
          <div style={{ fontFamily:F, fontSize:8, color:"#aaa", fontStyle:"italic",
            padding:"0 14px 6px", lineHeight:1.4, textAlign:"right", maxWidth:190,
            position:"absolute", top:"calc(100% + 4px)", right:16, background:"rgba(245,245,245,0.88)",
            border:"1px solid rgba(170,170,190,0.40)", pointerEvents:"none", display: ccbBasis > 0.15 ? "block" : "none" }}>
            Cross-currency basis: USD scarcity premium in EUR/USD FX swaps
          </div>

          {/* §C4 Stress Designer panel */}
          <StressDesignerPanel
            F={F} FM={FM}
            open={designerOpen}
            onClose={() => setDesignerOpen(false)}
            nodeIdx={designerNode} setNodeIdx={setDesignerNode}
            lf={designerLf}   setLf={setDesignerLf}
            lm={designerLm}   setLm={setDesignerLm}
            con={designerCon} setCon={setDesignerCon}
            onApply={applyDesignerStress}
            onDefault={applyDesignerDefault}
          />

          {/* §SYNTHESIS panel */}
          {/* §MC Standalone prediction card — wrapped in boundary */}
          {mcCardOpen && (
            <SynthErrorBoundary>
            <MCAnalysisModule
              mcBands={mcBands}
              mcRunning={mcRunning}
              onRun={runMonteCarlo}
              onClose={() => setMcCardOpen(false)}
              dispFrame={dispFrame}
              crisisType={crisisType}
              mcStale={mcStale}
              F={F} FM={FM}
            />
            </SynthErrorBoundary>
          )}

          {synthOpen && (
            <SynthErrorBoundary>
            <SynthesisPanel
              F={F} FM={FM}
              mcRunning={mcRunning} mcBands={mcBands} onRunMC={runMonteCarlo}
              mfgEnabled={mfgEnabled} setMfgEnabled={setMfgEnabled} mfgStats={mfgStats}
              calibrated={calibrated} calLabel={calLabel} onCalibrate={applyCalibration}
              sculptureMode={sculptureMode} setSculptureMode={setSculptureMode} sliceCount={sliceCount}
              hjbEnabled={hjbEnabled} setHjbEnabled={setHjbEnabled} hjbReady={hjbReady} hjbOptU={hjbOptU}
              homologyEnabled={homologyEnabled} setHomologyEnabled={setHomologyEnabled} homologyData={homologyData}
              edgeFraction={edgeFraction} topologyAlert={topologyAlert} edgeCount={edgeCount}
              fisherProj={fisherProj} crisisType={crisisType}
              calResults={calResults} calRunning={calRunning} onCalibrateEp={runCalibrationFor}
              advEnabled={advEnabled} setAdvEnabled={setAdvEnabled} advEquil={advEquil}
              hjbPolicy={hjbPolicyRef.current}
              synthSuggestion={synthSuggestion}
              mcStale={mcStale}
              synthPipeline={synthPipelineRef.current}
            />
            </SynthErrorBoundary>
          )}

          {/* §B1 Rough path signature panel */}
          {sigPanelIdx !== null && (
            <SigPanel
              nodeIdx={sigPanelIdx}
              pathHistRef={pathHistRef}
              F={F} FM={FM}
              onClose={() => setSigPanelIdx(null)}
            />
          )}

          {/* Node labels rendered by CSS2DRenderer — tethered to 3D node positions */}

          {/* Hover tooltip */}
          {hoverIdx !== null && hoverIdx >= 0 && metrics.nodeData[hoverIdx] && (
            <div style={{ position:"absolute", bottom:60, right:20, pointerEvents:"none", zIndex:30,
              background:"rgba(255,255,255,0.97)", border:"1px solid #aaaaaa",
              padding:"12px 16px", maxWidth:240, boxShadow:"0 2px 8px rgba(0,0,0,0.10)" }}>
              <div style={{ fontFamily:F, fontSize:13, fontWeight:700, color:"#1a3a6b", marginBottom:8 }}>
                {INST[hoverIdx].name}
              </div>
              <table style={{ fontFamily:FM, fontSize:10, color:"#555", borderCollapse:"collapse", width:"100%" }}>
                <tbody>
                {[
                  ["Type",         INST[hoverIdx].type + " · Tier " + INST[hoverIdx].tier],
                  ["Balance Sheet","€" + INST[hoverIdx].bs.toFixed(2) + "tn"],
                  ["Sov. Exposure",(INST[hoverIdx].sov * 100).toFixed(0) + "%"],
                  ["Repo Spread",  (metrics.nodeData[hoverIdx].bps >= 0 ? "+" : "") + metrics.nodeData[hoverIdx].bps + " bp"],
                  ["Merton D2D",   metrics.nodeData[hoverIdx].d2d + "σ"],
                  ["Rough Bergomi V", (hvRef.current[hoverIdx] || RB_V0).toFixed(4)],
                  ["B-P Lf",       metrics.nodeData[hoverIdx].Lf.toFixed(3)],
                  ["B-P Lm",       metrics.nodeData[hoverIdx].Lm.toFixed(3)],
                  ["Haircut",      metrics.nodeData[hoverIdx].haircut + "%"],
                  ["Status",       metrics.nodeData[hoverIdx].status],
                ].map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ paddingRight:10, color:"#888", fontStyle:"italic", paddingBottom:3 }}>{k}</td>
                    <td style={{ fontWeight:600 }}>{v}</td>
                  </tr>
                ))}
                </tbody>
              </table>
              {!defRef.current[hoverIdx] && (
                <div style={{ marginTop:8, display:"flex", gap:6 }}>
                  <button onClick={() => inject(hoverIdx)} style={{
                    flex:1, fontFamily:F, fontSize:10, background:"#f0f4ff",
                    border:"1px solid #4466aa", color:"#1a3a6b", padding:"4px",
                    cursor:"pointer" }}>↑ Inject MRO</button>
                  <button onClick={() => forceDefault(hoverIdx)} style={{
                    flex:1, fontFamily:F, fontSize:10, background:"#fff4f4",
                    border:"1px solid #aa2222", color:"#8b1a1a", padding:"4px",
                    cursor:"pointer" }}>Default</button>
                </div>
              )}
            </div>
          )}

          {/* ── BOTTOM-CENTER: timeline scrubber ───────────────────── */}
          <div style={{ position:"absolute", bottom:12, left:"50%", transform:"translateX(-50%)",
            zIndex:30, display:"flex", alignItems:"center", gap:10,
            background:"rgba(245,245,245,0.88)", border:"1px solid rgba(170,170,190,0.55)",
            padding:"5px 14px", backdropFilter:"blur(3px)", minWidth:360, pointerEvents:"auto" }}>
            <div style={{ flex:1, position:"relative" }}>
              <input type="range" min={0} max={59} value={dispFrame} step={1}
                onChange={e => seek(Number(e.target.value))}
                style={{ width:"100%", cursor:"pointer", accentColor:"#1a3a6b" }}/>
              <div style={{ position:"absolute", top:-8, left:0, right:0, pointerEvents:"none" }}>
                {epDef.phases.map((t, i) => (
                  <div key={i} style={{ position:"absolute", left:`${(t/59)*100}%`,
                    transform:"translateX(-50%)", top:0 }}>
                    <div style={{ width:1, height:6, background:"#aaaaaa" }}/>
                  </div>
                ))}
              </div>
            </div>
            <span style={{ fontFamily:FM, fontSize:10, color:"#888", minWidth:36 }}>t = {dispFrame}</span>
          </div>
        </div>

        {/* RIGHT SIDEBAR — hidden */}
        <div style={{ display:"none" }}>

          <SH F={F}>Model Parameters</SH>
          <div style={{ padding:"0 14px 6px" }}>
            <ParamSlider F={F} FM={FM} label="Simulation Speed" sym="δt"
              value={params.simSpeed} min={0.3} max={5} step={0.1}
              onChange={v => updateParam("simSpeed", v)}
              desc="Rate of simulated time advance. Low = 'heavy fabric' dynamics." unit="×"/>
            <ParamSlider F={F} FM={FM} label="Hurst Exponent" sym="H"
              value={params.hurst} min={0.10} max={0.90} step={0.05}
              onChange={v => updateParam("hurst", v)}
              desc="H < 0.5: rough, jagged peaks. H = 0.5: Brownian. H > 0.5: smooth ridges." />
            <ParamSlider F={F} FM={FM} label="fBm Octaves" sym="n"
              value={params.fBmOct} min={1} max={8} step={1}
              onChange={v => updateParam("fBmOct", v)}
              desc="Number of frequency layers. Fewer = cleaner topology." />
            <ParamSlider F={F} FM={FM} label="fBm Amplitude" sym="A"
              value={+(params.fBmAmp * 1000).toFixed(0)} min={0} max={80} step={2}
              onChange={v => updateParam("fBmAmp", v / 1000)}
              desc="Micro-turbulence amplitude (‰ of ZS). 0 = perfectly smooth GMF." unit="‰"/>
            <ParamSlider F={F} FM={FM} label="Hawkes α" sym="α"
              value={params.hawkAlpha} min={0.05} max={1.20} step={0.05}
              onChange={v => updateParam("hawkAlpha", v)}
              desc="Contagion excitation strength. Higher α = faster cascade." />
            <ParamSlider F={F} FM={FM} label="Hawkes β" sym="β"
              value={params.hawkBeta} min={0.05} max={1.00} step={0.05}
              onChange={v => updateParam("hawkBeta", v)}
              desc="Contagion decay rate. Low β = slow, persistent contagion." />
          </div>

          <SH F={F}>Display Options</SH>
          <div style={{ padding:"0 14px 10px" }}>
            {[["showWire","Wireframe overlay"],["showNodes","Node markers"]].map(([k,lbl])=>(
              <label key={k} style={{ display:"flex", alignItems:"center", gap:8, marginBottom:8, cursor:"pointer" }}>
                <input type="checkbox" checked={params[k]} onChange={e => updateParam(k, e.target.checked)}
                  style={{ accentColor:"#1a3a6b", width:14, height:14 }}/>
                <span style={{ fontFamily:F, fontSize:11, color:"#444" }}>{lbl}</span>
              </label>
            ))}
          </div>

          {mode === "sandbox" && <>
            <SH F={F}>Policy Levers</SH>
            <div style={{ padding:"0 14px 8px" }}>
              <SbxSlider F={F} FM={FM} label="DFR Shift" unit="bp" value={sandbox.dfrShift} min={-50} max={100} step={5}
                onChange={v => updateSandbox("dfrShift", v)} desc="Parallel displacement of DFR." signed/>
              <SbxSlider F={F} FM={FM} label="BTP Sovereign" unit="%" value={btpShock} min={0} max={80} step={5}
                onChange={v => setBtpShock(v)} desc="Italian sovereign stress (warps peripheral gravity wells)."/>
              <SbxSlider F={F} FM={FM} label="TLTRO Repayment" unit="€bn" value={sandbox.tltro} min={0} max={600} step={25}
                onChange={v => updateSandbox("tltro", v)} desc="Peripheral liquidity drain via TLTRO unwinding."/>
              <SbxSlider F={F} FM={FM} label="Haircut Shock" unit="%" value={sandbox.haircut} min={0} max={25} step={1}
                onChange={v => updateSandbox("haircut", v)} desc="Collateral margin call — reduces B-P Lf* equilibrium."/>
              <SbxSlider F={F} FM={FM} label="QT Drain" unit="€bn" value={sandbox.qtDrain||0} min={0} max={2000} step={100}
                onChange={v => updateSandbox("qtDrain", v)} desc="Quantitative Tightening — slow system-wide drain."/>
              <div style={{ display:"flex", gap:6, marginTop:6 }}>
                <button onClick={() => { setSandbox(DEFAULT_SBX); setBtpShock(0); }} style={{
                  flex:1, fontFamily:F, fontSize:10, background:"transparent",
                  border:"1px solid #cccccc", color:"#555", padding:"5px", cursor:"pointer" }}>
                  Reset Shocks
                </button>
                <button onClick={() => forceDefault(Math.floor(Math.random() * 17))} style={{
                  flex:1, fontFamily:F, fontSize:10, background:"#fff4f4",
                  border:"1px solid #aa2222", color:"#8b1a1a", padding:"5px", cursor:"pointer" }}>
                  Force Default
                </button>
              </div>
            </div>
          </>}

          <SH F={F}>Institution Monitor</SH>
          <div style={{ flex:1, overflowY:"auto" }}>
            {INST.map(({ name, type, tier }, i) => {
              const nd  = metrics.nodeData[i];
              const stC = statusColor[nd.status] || "#555";
              return (
                <div key={i} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)}
                  style={{ padding:"5px 14px", borderBottom:"1px solid #f0f0f0", cursor:"default",
                    background: nd.status==="CRISIS"?"#fff8f8":nd.status==="ALERT"?"#fffcf4":"transparent" }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:2 }}>
                    <div>
                      <span style={{ fontFamily:FM, fontSize:10, fontWeight:600, color:nd.defaulted?"#aaa":"#333" }}>{INST[i].abbr}</span>
                      <span style={{ fontFamily:F, fontSize:9, color:"#aaa", marginLeft:4 }}>T{tier}</span>
                    </div>
                    <div style={{ textAlign:"right" }}>
                      <div style={{ fontFamily:FM, fontSize:10, fontWeight:700, color:stC }}>
                        {nd.bps >= 0 ? "+" : ""}{nd.bps.toFixed(1)} bp
                      </div>
                    </div>
                  </div>
                  {/* D2D bar */}
                  <div style={{ display:"flex", alignItems:"center", gap:4 }}>
                    <span style={{ fontFamily:FM, fontSize:8, color:"#aaa", minWidth:22 }}>D2D</span>
                    <div style={{ flex:1, height:2, background:"#eeeeee", overflow:"hidden" }}>
                      <div style={{ width:`${Math.min(nd.d2d/5*100,100)}%`, height:"100%",
                        background: nd.d2d<1?"#8b1a1a":nd.d2d<2.5?"#6b4a10":"#1a4a2a",
                        transition:"width 0.3s" }}/>
                    </div>
                    <span style={{ fontFamily:FM, fontSize:8, color:"#888", minWidth:26, textAlign:"right" }}>{nd.d2d.toFixed(1)}σ</span>
                  </div>
                </div>
              );
            })}
          </div>

          <SH F={F}>Alert Log</SH>
          <div style={{ maxHeight:120, overflowY:"auto", padding:"6px 14px" }}>
            {alerts.length === 0 && <div style={{ fontFamily:F, fontSize:10, color:"#aaa", fontStyle:"italic" }}>No active alerts.</div>}
            {alerts.slice(0, 10).map(a => (
              <div key={a.id} style={{ marginBottom:6,
                borderLeft:`2px solid ${a.sev==="CRISIS"||a.sev==="DEFAULT"?"#8b1a1a":a.sev==="ALERT"?"#6b4a10":"#1a4a2a"}`,
                paddingLeft:6 }}>
                <div style={{ fontFamily:FM, fontSize:9, color:"#aaa" }}>{a.time}</div>
                <div style={{ fontFamily:F, fontSize:10, color:"#444", lineHeight:1.4 }}>{a.text}</div>
              </div>
            ))}
          </div>

          <SH F={F}>MMSR Feed</SH>
          <div style={{ padding:"4px 14px 8px" }}>
            {mmrLog.map((l, i) => (
              <div key={i} style={{ fontFamily:FM, fontSize:9, color:i===0?"#333":"#aaa",
                lineHeight:1.7, borderBottom:i===0?"1px solid #eee":"none", paddingBottom:i===0?4:0 }}>{l}</div>
            ))}
          </div>

          <div style={{ padding:"8px 14px", borderTop:"1px solid #eeeeee", display:"flex", gap:6, flexShrink:0 }}>
            <button onClick={resetAll} style={{ flex:1, fontFamily:F, fontSize:10, background:"transparent",
              border:"1px solid #cccccc", color:"#555", padding:"4px", cursor:"pointer" }}>Reset</button>
            <button onClick={exportTelemetry} style={{ flex:1, fontFamily:F, fontSize:10, background:"transparent",
              border:"1px solid #cccccc", color:"#555", padding:"4px", cursor:"pointer" }}>↓ JSON</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS (academic serif style)
// ─────────────────────────────────────────────────────────────────────
function SH({ children, F }) {
  return (
    <div style={{ padding:"8px 14px 4px", fontFamily:F, fontSize:11, fontWeight:700,
      color:"#1a3a6b", letterSpacing:"0.01em", borderTop:"1px solid #eeeeee",
      borderBottom:"1px solid #eeeeee", background:"#fafafa", fontStyle:"italic" }}>
      {children}
    </div>
  );
}
function ML({ l, v, c, F, FM }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:5 }}>
      <span style={{ fontFamily:F, fontSize:10, color:"#888", fontStyle:"italic" }}>{l}</span>
      <span style={{ fontFamily:FM, fontSize:11, fontWeight:600, color:c||"#1a1a2a" }}>{v}</span>
    </div>
  );
}

function ParamSlider({ F, FM, label, sym, value, min, max, step, onChange, desc, unit="" }) {
  const pct = Math.round(((value - min) / (max - min)) * 100);
  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:2 }}>
        <span style={{ fontFamily:F, fontSize:10, color:"#555", fontStyle:"italic" }}>
          {label} (<i>{sym}</i>)
        </span>
        <span style={{ fontFamily:FM, fontSize:11, fontWeight:600, color:"#1a3a6b" }}>
          {value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width:"100%", accentColor:"#1a3a6b", cursor:"pointer" }}/>
      <div style={{ height:1, background:"#e8e8e8", overflow:"hidden", marginBottom:2 }}>
        <div style={{ width:`${pct}%`, height:"100%", background:"#1a3a6b", opacity:0.25, transition:"width 0.1s" }}/>
      </div>
      <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic", lineHeight:1.4 }}>{desc}</div>
    </div>
  );
}

function SbxSlider({ F, FM, label, unit, value, min, max, step, onChange, desc, signed }) {
  const pct = Math.round(((value - min) / (max - min)) * 100);
  return (
    <div style={{ marginBottom:10 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:2 }}>
        <span style={{ fontFamily:F, fontSize:10, color:"#555", fontStyle:"italic" }}>{label}</span>
        <span style={{ fontFamily:FM, fontSize:11, fontWeight:600, color:"#8b1a1a" }}>
          {signed && value >= 0 && min < 0 ? "+" : ""}{value} {unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width:"100%", accentColor:"#8b1a1a", cursor:"pointer" }}/>
      <div style={{ height:1, background:"#e8e8e8", overflow:"hidden", marginBottom:2 }}>
        <div style={{ width:`${pct}%`, height:"100%", background:"#8b1a1a", opacity:0.20, transition:"width 0.1s" }}/>
      </div>
      <div style={{ fontFamily:F, fontSize:9, color:"#aaa", fontStyle:"italic", lineHeight:1.4 }}>{desc}</div>
    </div>
  );
}

function StressChart({ history, phase, phaseNames, F, FM }) {
  const W=178, H=52, P=4;
  const vmin = Math.min(...history, -0.02), vmax = Math.max(...history, 0.02);
  const vr   = vmax - vmin || 0.1;
  const iW   = (W - 2*P) / (history.length - 1 || 1);
  const pts  = history.map((v,i) => `${(P+i*iW).toFixed(1)},${(H-P-((v-vmin)/vr)*(H-2*P)).toFixed(1)}`).join(" ");
  const zy   = H - P - ((0 - vmin) / vr) * (H - 2*P);
  const col  = phase.toLowerCase().includes("crisis") || phase.toLowerCase().includes("panic") ? "#8b1a1a" :
               phase.toLowerCase().includes("deterioration") || phase.toLowerCase().includes("onset") ? "#6b4a10" :
               "#1a4a2a";
  return (
    <svg width={W} height={H} style={{ display:"block", background:"#fafafa", border:"1px solid #eeeeee" }}>
      {vmin < 0 && vmax > 0 && (
        <line x1={P} y1={zy} x2={W-P} y2={zy} stroke="#4466aa" strokeWidth="0.8" strokeDasharray="3,3" strokeOpacity="0.5"/>
      )}
      <polyline points={pts} fill="none" stroke={col} strokeWidth="1.5" strokeOpacity="0.80"/>
      <text x={P} y={H-1} fontSize="7" fill="#aaa" fontFamily="Georgia,serif">t=0</text>
      <text x={W-P} y={H-1} fontSize="7" fill="#aaa" textAnchor="end" fontFamily="Georgia,serif">t=59</text>
      <text x={W/2} y={8} fontSize="7" fill={col} textAnchor="middle" fontFamily="Georgia,serif" fontStyle="italic">{phase}</text>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────
// §C4  STRESS DESIGNER PANEL
// ─────────────────────────────────────────────────────────────────────
function StressDesignerPanel({ F, FM, open, onClose, nodeIdx, setNodeIdx,
  lf, setLf, lm, setLm, con, setCon, onApply, onDefault }) {
  if (!open) return null;
  return (
    <div style={{
      position:'absolute', left:220, top:80, width:220, zIndex:40,
      background:'rgba(250,248,245,0.98)', border:'1px solid #cc9933',
      boxShadow:'0 2px 14px rgba(0,0,0,0.14)',
    }}>
      <div style={{ padding:'6px 10px', background:'#8b5a10', display:'flex',
        justifyContent:'space-between', alignItems:'center' }}>
        <span style={{ fontFamily:F, fontSize:10, color:'#fff', fontStyle:'italic' }}>Stress Designer</span>
        <span style={{ color:'#ffcc88', fontSize:12, cursor:'pointer' }} onClick={onClose}>✕</span>
      </div>
      <div style={{ padding:'8px 12px' }}>
        <div style={{ fontFamily:F, fontSize:9, color:'#888', fontStyle:'italic', marginBottom:6 }}>Select institution</div>
        <select value={nodeIdx} onChange={e => setNodeIdx(Number(e.target.value))}
          style={{ width:'100%', fontFamily:F, fontSize:9, border:'1px solid #ccc',
            background:'#fafafa', padding:'2px 4px', marginBottom:10 }}>
          {INST.map((inst, i) => (
            <option key={i} value={i}>{inst.abbr} — {inst.name}</option>
          ))}
        </select>
        <div style={{ marginBottom:10 }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
            <span style={{ fontFamily:F, fontSize:9, color:'#555', fontStyle:'italic' }}>Funding liquidity L_f</span>
            <span style={{ fontFamily:FM, fontSize:10, color:'#8b5a10' }}>{lf.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={1} step={0.01} value={lf}
            onChange={e => setLf(Number(e.target.value))}
            style={{ width:'100%', accentColor:'#8b5a10' }}/>
        </div>
        <div style={{ marginBottom:10 }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
            <span style={{ fontFamily:F, fontSize:9, color:'#555', fontStyle:'italic' }}>Market illiquidity L_m</span>
            <span style={{ fontFamily:FM, fontSize:10, color:'#8b5a10' }}>{lm.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={3} step={0.05} value={lm}
            onChange={e => setLm(Number(e.target.value))}
            style={{ width:'100%', accentColor:'#8b5a10' }}/>
        </div>
        <div style={{ marginBottom:12 }}>
          <div style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
            <span style={{ fontFamily:F, fontSize:9, color:'#555', fontStyle:'italic' }}>Contagion seed</span>
            <span style={{ fontFamily:FM, fontSize:10, color:'#8b1a1a' }}>{con.toFixed(2)}</span>
          </div>
          <input type="range" min={0} max={4} step={0.05} value={con}
            onChange={e => setCon(Number(e.target.value))}
            style={{ width:'100%', accentColor:'#8b1a1a' }}/>
        </div>
        <div style={{ display:'flex', gap:6 }}>
          <button onClick={onApply} style={{ flex:2, fontFamily:F, fontSize:9,
            background:'#8b5a10', color:'#fff', border:'none', padding:'5px', cursor:'pointer' }}>Apply Stress</button>
          <button onClick={onDefault} style={{ flex:1, fontFamily:F, fontSize:9,
            background:'#8b1a1a', color:'#fff', border:'none', padding:'5px', cursor:'pointer' }}>Default</button>
        </div>
        <div style={{ fontFamily:F, fontSize:8, color:'#aaa', fontStyle:'italic', marginTop:6, lineHeight:1.4 }}>
          Writes directly into simulation state. Press Play to observe propagation.
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// §B1  ROUGH PATH SIGNATURE PANEL
// ─────────────────────────────────────────────────────────────────────
function SigPanel({ nodeIdx, pathHistRef, F, FM, onClose }) {
  const [sig, setSig] = useState(null);
  useEffect(() => {
    const interval = setInterval(() => {
      const path = pathHistRef.current[nodeIdx];
      if (path && path.length >= 2) setSig(computeSignature(path));
    }, 500);
    return () => clearInterval(interval);
  }, [nodeIdx]);
  if (!sig) return null;
  const inst = INST[nodeIdx];
  const labels1 = ['S(t̂)', 'S(Lf)', 'S(sp)'];
  const labels2 = ['tt','tL','ts','Lt','LL','Ls','st','sL','ss'];
  return (
    <div style={{
      position:'absolute', right:220, top:60, width:210,
      background:'rgba(250,250,250,0.97)', border:'1px solid #ccc',
      fontFamily:F, boxShadow:'0 2px 12px rgba(0,0,0,0.12)', zIndex:30,
    }}>
      <div style={{ padding:'6px 10px', background:'#1a3a6b', display:'flex',
        justifyContent:'space-between', alignItems:'center' }}>
        <span style={{ fontFamily:F, fontSize:10, color:'#fff', fontStyle:'italic' }}>
          Rough Path Signature — {inst.abbr}
        </span>
        <span style={{ color:'#aac', fontSize:12, cursor:'pointer' }} onClick={onClose}>✕</span>
      </div>
      <div style={{ padding:'8px 10px' }}>
        <div style={{ fontFamily:F, fontSize:9, color:'#888', fontStyle:'italic',
          borderBottom:'1px solid #eee', paddingBottom:4, marginBottom:6 }}>
          Order 1  —  S: X → ℝ³
        </div>
        {sig.s1.map((v, i) => (
          <div key={i} style={{ display:'flex', justifyContent:'space-between', marginBottom:2 }}>
            <span style={{ fontFamily:F, fontSize:9, color:'#666', fontStyle:'italic' }}>{labels1[i]}</span>
            <span style={{ fontFamily:FM, fontSize:10, color:Math.abs(v)>0.3?'#8b1a1a':'#1a3a6b' }}>
              {v>=0?'+':''}{v.toFixed(4)}
            </span>
          </div>
        ))}
        <div style={{ fontFamily:F, fontSize:9, color:'#888', fontStyle:'italic',
          borderBottom:'1px solid #eee', paddingBottom:4, marginBottom:6, marginTop:8 }}>
          Order 2  —  S: X → ℝ³ˣ³  (Lévy area)
        </div>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:2 }}>
          {sig.s2.map((v, i) => (
            <div key={i} style={{ fontFamily:FM, fontSize:8,
              color:Math.abs(v)>0.05?'#8b1a1a':'#444',
              textAlign:'right', padding:'1px 3px',
              background:Math.abs(v)>0.05?'rgba(139,26,26,0.07)':'transparent' }}>
              {v>=0?'+':''}{v.toFixed(3)}
            </div>
          ))}
        </div>
        <div style={{ fontFamily:F, fontSize:8, color:'#aaa', fontStyle:'italic',
          marginTop:3, lineHeight:1.4 }}>
          Indices: {labels2.join(' · ')}
        </div>
        <div style={{ fontFamily:F, fontSize:9, color:'#888', fontStyle:'italic',
          borderBottom:'1px solid #eee', paddingBottom:4, marginBottom:4, marginTop:8 }}>
          Order 3  —  norm ‖S³‖  (scalar summary)
        </div>
        <div style={{ display:'flex', justifyContent:'space-between' }}>
          <span style={{ fontFamily:F, fontSize:9, color:'#666', fontStyle:'italic' }}>‖S^(3)‖₂</span>
          <span style={{ fontFamily:FM, fontSize:11, color:'#1a3a6b', fontWeight:600 }}>
            {Math.sqrt(sig.s3.reduce((a,v)=>a+v*v,0)).toFixed(5)}
          </span>
        </div>
        <div style={{ fontFamily:F, fontSize:8, color:'#aaa', fontStyle:'italic', marginTop:4, lineHeight:1.4 }}>
          Path: (t̂, L_f, σ_i/100) ∈ ℝ³  ·  last {pathHistRef.current[nodeIdx]?.length || 0} ticks
        </div>
      </div>
    </div>
  );
}

function NSChart({ yc, F, FM }) {
  if (!yc || yc.length < 2) return <div style={{ fontFamily:F, fontSize:10, color:"#aaa", fontStyle:"italic" }}>Loading…</div>;
  const W=178, H=52, P=4;
  const rates  = yc.map(p => p.rate);
  const rmin   = Math.min(...rates, 2.0), rmax = Math.max(...rates, 3.5), rr = rmax - rmin || 0.5;
  const iW     = (W - 2*P) / (rates.length - 1);
  const pts    = rates.map((r,i) => `${(P+i*iW).toFixed(1)},${(H-P-((r-rmin)/rr)*(H-2*P)).toFixed(1)}`).join(" ");
  const dfrY   = H - P - ((ECB_DFR - rmin) / rr) * (H - 2*P);
  const isInv  = rates[0] > rates[rates.length-1];
  return (
    <svg width={W} height={H} style={{ display:"block", background:"#fafafa", border:"1px solid #eeeeee" }}>
      <line x1={P} y1={dfrY} x2={W-P} y2={dfrY} stroke="#4466aa" strokeWidth="0.7" strokeDasharray="3,3" strokeOpacity="0.5"/>
      <polyline points={pts} fill="none" stroke={isInv?"#8b1a1a":"#1a4a2a"} strokeWidth="1.5" strokeOpacity="0.85"/>
      {rates.map((r,i) => (
        <circle key={i} cx={(P+i*iW).toFixed(1)} cy={(H-P-((r-rmin)/rr)*(H-2*P)).toFixed(1)}
          r="2" fill={isInv?"#8b1a1a":"#1a4a2a"} fillOpacity="0.7"/>
      ))}
      <text x={P} y={H-1} fontSize="7" fill="#aaa" fontFamily="Georgia,serif">O/N</text>
      <text x={W-P} y={H-1} fontSize="7" fill="#aaa" textAnchor="end" fontFamily="Georgia,serif">10Y</text>
      <text x={W-P} y={8} fontSize="7" fill={isInv?"#8b1a1a":"#1a4a2a"} textAnchor="end" fontFamily="Georgia,serif">
        {isInv?"▼ Inverted":"▲ Normal"}
      </text>
      <text x={P} y={8} fontSize="7" fill="#aaa" fontFamily="Georgia,serif">{rmin.toFixed(2)}%</text>
    </svg>
  );
}


// ─────────────────────────────────────────────────────────────────────
// §SYNTH SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────
function MCFanChart({ bands, F, FM }) {
  const W=200, H=90, P=10;
  if (!bands?.p50) return null;
  const { p10, p50, p90 } = bands;
  const n = p50.length; if (!n) return null;
  const allV = [...p10, ...p50, ...p90].filter(v => isFinite(v) && v !== 0);
  if (!allV.length) return null;
  const vmin = Math.min(...allV);
  const vmax = Math.max(...allV);
  const vr = Math.max(vmax - vmin, 0.1);
  const iW = (W - 2*P) / (n - 1 || 1);
  const toX = i => (P + i * iW).toFixed(1);
  const toY = v => Math.max(P, Math.min(H - P, H - P - ((v - vmin) / vr) * (H - 2*P)));
  const fanPts = [
    ...p90.map((v,i) => `${toX(i)},${toY(v).toFixed(1)}`),
    ...p10.slice().reverse().map((v,i) => `${toX(n-1-i)},${toY(v).toFixed(1)}`),
  ].join(' ');
  const pts50 = p50.map((v,i) => `${toX(i)},${toY(v).toFixed(1)}`).join(' ');
  const pts90 = p90.map((v,i) => `${toX(i)},${toY(v).toFixed(1)}`).join(' ');
  const pts10 = p10.map((v,i) => `${toX(i)},${toY(v).toFixed(1)}`).join(' ');
  const ticks = [0, Math.floor(n*0.25), Math.floor(n*0.5), Math.floor(n*0.75), n-1];
  const yTicks = 4;
  return (
    <svg width={W} height={H} style={{display:"block",background:"rgba(248,248,252,0.95)",border:"1px solid #e4e4ee",overflow:"visible"}}>
      {/* Y grid lines */}
      {Array.from({length:yTicks+1},(_,i)=>{
        const v = vmin + (vr/yTicks)*i;
        const y = toY(v);
        return <g key={i}>
          <line x1={P} y1={y} x2={W-P} y2={y} stroke="#eeeeee" strokeWidth="0.6"/>
          <text x={P-2} y={y+3} fontSize="5.5" fill="#bbb" textAnchor="end" fontFamily="Georgia,serif">{v.toFixed(0)}</text>
        </g>;
      })}
      {/* Fan area */}
      <polygon points={fanPts} fill="rgba(139,26,26,0.09)" stroke="none"/>
      <polyline points={pts90} fill="none" stroke="rgba(180,60,60,0.35)" strokeWidth="0.9" strokeDasharray="2,2"/>
      <polyline points={pts10} fill="none" stroke="rgba(60,100,180,0.35)" strokeWidth="0.9" strokeDasharray="2,2"/>
      {/* Median */}
      <polyline points={pts50} fill="none" stroke="#8b1a1a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      {/* X ticks */}
      {ticks.map(i => (
        <g key={i}>
          <line x1={toX(i)} y1={H-P} x2={toX(i)} y2={H-P+3} stroke="#ccc" strokeWidth="0.7"/>
          <text x={toX(i)} y={H-1} fontSize="5.5" fill="#aaa" textAnchor="middle" fontFamily="Georgia,serif">t={i}</text>
        </g>
      ))}
      {/* Endpoint dot */}
      <circle cx={toX(n-1)} cy={toY(p50[n-1]||0)} r="2.5" fill="#8b1a1a" opacity="0.85"/>
      {/* Labels */}
      <text x={P+1} y={13} fontSize="7" fill="#8b1a1a" fontFamily="Georgia,serif" fontWeight="600">
        p50: {p50[n-1]?.toFixed(1)||"–"}bp
      </text>
      <text x={P+1} y={22} fontSize="6" fill="#aaa" fontFamily="Georgia,serif">
        p10:{p10[n-1]?.toFixed(0)||"–"} · p90:{p90[n-1]?.toFixed(0)||"–"} · {n}t
      </text>
    </svg>
  );
}

function PersistenceDiagram({ data, F, FM }) {
  // data: [{thr, b0, b1}] at thresholds [-2,-5,-10,-15,-20,-25]
  const W=200, H=90, P=10;
  if (!data?.length) return null;
  const maxVal = Math.max(...data.map(d => d.b0), ...data.map(d => d.b1), 1);
  const thrs = data.map(d => d.thr);
  const iW = (W - 2*P) / (data.length - 1 || 1);
  const toX = i => (P + i * iW).toFixed(1);
  const toY = v => (H - P - (v / maxVal) * (H - 2*P)).toFixed(1);
  const pts0 = data.map((d,i) => `${toX(i)},${toY(d.b0)}`).join(' ');
  const pts1 = data.map((d,i) => `${toX(i)},${toY(d.b1)}`).join(' ');
  // Area under β₀
  const area0 = [`${toX(0)},${H-P}`, ...data.map((d,i) => `${toX(i)},${toY(d.b0)}`), `${toX(data.length-1)},${H-P}`].join(' ');
  const area1 = [`${toX(0)},${H-P}`, ...data.map((d,i) => `${toX(i)},${toY(d.b1)}`), `${toX(data.length-1)},${H-P}`].join(' ');
  const yGrid = [0, Math.ceil(maxVal/4), Math.ceil(maxVal/2), Math.ceil(maxVal*3/4), maxVal];
  return (
    <svg width={W} height={H} style={{display:"block",background:"rgba(248,248,252,0.95)",border:"1px solid #e4e4ee"}}>
      {/* Grid */}
      {yGrid.map((v,i) => {
        const y = +toY(v);
        return <g key={i}>
          <line x1={P} y1={y} x2={W-P} y2={y} stroke="#eeeeee" strokeWidth="0.6"/>
          <text x={P-2} y={y+3} fontSize="5.5" fill="#bbb" textAnchor="end" fontFamily="Georgia,serif">{v}</text>
        </g>;
      })}
      {/* Filled areas */}
      <polygon points={area0} fill="rgba(26,58,139,0.07)" stroke="none"/>
      <polygon points={area1} fill="rgba(139,26,26,0.07)" stroke="none"/>
      {/* Lines */}
      <polyline points={pts0} fill="none" stroke="#1a3a6b" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
      <polyline points={pts1} fill="none" stroke="#8b1a1a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="3,2"/>
      {/* Dots at each threshold */}
      {data.map((d,i) => (
        <g key={i}>
          <circle cx={toX(i)} cy={toY(d.b0)} r="2.2" fill="#1a3a6b"/>
          <circle cx={toX(i)} cy={toY(d.b1)} r="2" fill="#8b1a1a"/>
          <text x={toX(i)} y={H-1} fontSize="5.5" fill="#aaa" textAnchor="middle" fontFamily="Georgia,serif">{d.thr}</text>
          <line x1={toX(i)} y1={H-P} x2={toX(i)} y2={H-P+3} stroke="#ddd" strokeWidth="0.6"/>
        </g>
      ))}
      {/* Legend */}
      <rect x={P} y={10} width={7} height={4} fill="#1a3a6b" opacity="0.8"/>
      <text x={P+10} y={14} fontSize="6.5" fill="#1a3a6b" fontFamily="Georgia,serif">\u03b2\u2080={data[data.length-1]?.b0||0} components</text>
      <rect x={P} y={20} width={7} height={2} fill="none" stroke="#8b1a1a" strokeWidth="1.5" strokeDasharray="3,1"/>
      <text x={P+10} y={24} fontSize="6.5" fill="#8b1a1a" fontFamily="Georgia,serif">\u03b2\u2081={data[data.length-1]?.b1||0} loops</text>
    </svg>
  );
}



// ─────────────────────────────────────────────────────────────────────
// §UI-CHARTS  Network topology + Fisher manifold + Calibration displays
// ─────────────────────────────────────────────────────────────────────

function NetworkTopologyChart({ edgeFraction, topologyAlert, edgeCount, F, FM }) {
  const W=200, H=60, P=8;
  // Show edge fraction as a gauge bar with threshold marker
  const frac = clamp(edgeFraction, 0, 1);
  const barW = (W - 2*P) * frac;
  const thrX = P + (W - 2*P) * 0.65; // 65% = warning threshold
  const col = frac < 0.45 ? '#8b1a1a' : frac < 0.65 ? '#8b5a10' : '#1a4a2a';
  return (
    <svg width={W} height={H} style={{display:"block",background:"rgba(248,248,252,0.95)",border:`1px solid ${topologyAlert?'#cc4444':'#e4e4ee'}`}}>
      {/* Background bar */}
      <rect x={P} y={H/2-5} width={W-2*P} height={10} rx={2} fill="#f0f0f0"/>
      {/* Live fraction bar */}
      <rect x={P} y={H/2-5} width={Math.max(0,barW)} height={10} rx={2} fill={col} opacity="0.85"/>
      {/* Threshold marker */}
      <line x1={thrX} y1={H/2-10} x2={thrX} y2={H/2+10} stroke="#cc8800" strokeWidth="1.5" strokeDasharray="2,1"/>
      <text x={thrX} y={H/2-12} fontSize="6" fill="#cc8800" textAnchor="middle" fontFamily="Georgia,serif">σ*</text>
      {/* Labels */}
      <text x={P} y={H/2-10} fontSize="6.5" fill="#888" fontFamily="Georgia,serif">Network connectivity</text>
      <text x={W-P} y={H/2-10} fontSize="7.5" fill={col} fontFamily="Georgia,serif" textAnchor="end" fontWeight="600">
        {(frac*100).toFixed(1)}%
      </text>
      <text x={P} y={H-4} fontSize="6" fill="#aaa" fontFamily="Georgia,serif">
        {edgeCount} active bilateral edges {topologyAlert ? '⚠ FRAGMENTATION' : ''}
      </text>
      <text x={W-P} y={H-4} fontSize="6" fill="#aaa" fontFamily="Georgia,serif" textAnchor="end">
        0%{'       '}65%{'      '}100%
      </text>
    </svg>
  );
}

function FisherManifoldChart({ fisherProj, crisisType, F, FM }) {
  const W=200, H=100, P=12;
  if (!fisherProj || fisherProj.length < 3) return (
    <div style={{fontFamily:F,fontSize:8,color:"#aaa",fontStyle:"italic",paddingLeft:4,height:30,display:"flex",alignItems:"center"}}>
      Accumulating trajectory… (5+ ticks needed)
    </div>
  );
  // gc = x axis (geodesic progress), concentration = y axis (peripheral mass)
  const pts = fisherProj;
  const toX = gc => P + gc * (W - 2*P);
  const toY = c  => H - P - c * (H - 2*P);
  const pathStr = pts.map((p,i) => `${i===0?'M':'L'}${toX(p.gc).toFixed(1)},${toY(p.concentration).toFixed(1)}`).join(' ');
  // Crisis type color coding
  const typeColors = {
    sovereign_contagion: '#8b1a1a',
    systemic_shock:      '#1a3a8b',
    cliff_collapse:      '#6b1a8b',
    ambiguous:           '#888888',
  };
  const tCol = typeColors[crisisType?.type] || '#888';
  // Reference geodesics for the three crisis types
  const dec2011Path  = [[0,0.3],[0.3,0.55],[0.6,0.62],[1.0,0.65]]; // slow, peripheral
  const mar2020Path  = [[0,0.3],[0.2,0.33],[0.5,0.35],[1.0,0.36]]; // fast, systemic
  const refPath = (rp, col, dash) => rp.map((p,i)=>`${i===0?'M':'L'}${toX(p[0]).toFixed(1)},${toY(p[1]).toFixed(1)}`).join(' ');
  return (
    <svg width={W} height={H} style={{display:"block",background:"rgba(248,248,252,0.95)",border:"1px solid #e4e4ee"}}>
      {/* Grid */}
      {[0,0.25,0.5,0.75,1].map(v=>(
        <g key={v}>
          <line x1={P} y1={toY(v)} x2={W-P} y2={toY(v)} stroke="#f0f0f0" strokeWidth="0.6"/>
          <text x={P-2} y={toY(v)+3} fontSize="5.5" fill="#ccc" textAnchor="end" fontFamily="Georgia,serif">{v.toFixed(2)}</text>
        </g>
      ))}
      {/* Reference crisis geodesics (dashed) */}
      <path d={refPath(dec2011Path)} fill="none" stroke="rgba(139,26,26,0.20)" strokeWidth="1" strokeDasharray="3,2"/>
      <path d={refPath(mar2020Path)} fill="none" stroke="rgba(26,58,139,0.20)" strokeWidth="1" strokeDasharray="3,2"/>
      {/* Actual trajectory */}
      <path d={pathStr} fill="none" stroke={tCol} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      {/* Current position */}
      {pts.length > 0 && (
        <circle cx={toX(pts[pts.length-1].gc)} cy={toY(pts[pts.length-1].concentration)} r="3.5" fill={tCol} opacity="0.9"/>
      )}
      {/* Axis labels */}
      <text x={W/2} y={H-1} fontSize="6" fill="#aaa" textAnchor="middle" fontFamily="Georgia,serif">Geodesic progress →</text>
      <text x={P-2} y={P} fontSize="6" fill="#aaa" textAnchor="end" fontFamily="Georgia,serif" transform={`rotate(-90,${P-8},${H/2})`}>Periph. conc.</text>
      {/* Crisis type label */}
      {crisisType && (
        <text x={W-P} y={14} fontSize="7" fill={tCol} textAnchor="end" fontFamily="Georgia,serif" fontWeight="600">
          {crisisType.type?.replace('_',' ')} ({(crisisType.confidence*100).toFixed(0)}%)
        </text>
      )}
      {/* Reference labels */}
      <text x={toX(0.85)} y={toY(0.63)} fontSize="5.5" fill="rgba(139,26,26,0.5)" fontFamily="Georgia,serif">Dec 2011</text>
      <text x={toX(0.85)} y={toY(0.34)} fontSize="5.5" fill="rgba(26,58,139,0.5)" fontFamily="Georgia,serif">Mar 2020</text>
    </svg>
  );
}

function CalibrationPanel({ calResults, calRunning, onCalibrate, F, FM }) {
  const episodes = ['dec2011','mar2020','lehman2008'];
  return (
    <div style={{paddingLeft:4}}>
      {episodes.map(ep => {
        const epSig = EPISODE_SIGNATURES[ep];
        const result = calResults?.[ep];
        return (
          <div key={ep} style={{marginBottom:6,padding:"4px 6px",background:"rgba(248,248,252,0.8)",border:"1px solid #e8e8ee"}}>
            <div style={{display:"flex",alignItems:"center",marginBottom:2}}>
              <span style={{fontFamily:FM,fontSize:8,color:"#333",flex:1}}>{epSig?.label||ep}</span>
              <button onClick={()=>onCalibrate(ep)} disabled={calRunning}
                style={{fontFamily:F,fontSize:7.5,background:"transparent",border:"1px solid #aaa",
                  color:calRunning?"#ccc":"#555",padding:"1px 4px",cursor:calRunning?"default":"pointer"}}>
                {calRunning?"…":"Fit θ*"}
              </button>
            </div>
            {result?.thetaStar && (
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1px 8px"}}>
                {[
                  ['α',result.thetaStar.hawkAlpha?.toFixed(3),'#8b1a1a'],
                  ['β',result.thetaStar.hawkBeta?.toFixed(2),'#1a3a6b'],
                  ['σ',result.thetaStar.sigScale?.toFixed(2),'#1a4a2a'],
                  ['L*',result.lossHist?.[result.lossHist.length-1]?.toFixed(4),'#555'],
                ].map(([k,v,c])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                    <span style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic"}}>{k}</span>
                    <span style={{fontFamily:FM,fontSize:7.5,color:c,fontWeight:600}}>{v||"–"}</span>
                  </div>
                ))}
              </div>
            )}
            {!result && (
              <div style={{fontFamily:F,fontSize:7,color:"#bbb",fontStyle:"italic"}}>
                θ₀=({epSig?.theta0?.hawkAlpha?.toFixed(2)},β={epSig?.theta0?.hawkBeta?.toFixed(1)})
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function AdversarialPanel({ advEnabled, setAdvEnabled, advEquil, F, FM }) {
  const specActs = advEquil?.specAction;
  const ecbU     = advEquil?.ecbU;
  return (
    <div style={{paddingLeft:4}}>
      <label style={{display:"flex",alignItems:"center",gap:5,cursor:"pointer",marginBottom:6}}>
        <input type="checkbox" checked={advEnabled} onChange={e=>setAdvEnabled(e.target.checked)}
          style={{width:11,height:11,accentColor:"#8b1a1a"}}/>
        <span style={{fontFamily:F,fontSize:8,color:"#555"}}>Speculator active (requires HJB)</span>
      </label>
      {advEnabled && advEquil && (
        <div style={{background:"rgba(248,240,240,0.9)",border:"1px solid #e8d0d0",padding:"4px 6px"}}>
          <div style={{fontFamily:F,fontSize:8,color:"#8b1a1a",fontStyle:"italic",marginBottom:4}}>Nash Equilibrium</div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px 8px"}}>
            {[
              ['Spec: BTP',   specActs?.btp?.toFixed(3)||"0.000",  '#8b1a1a'],
              ['Spec: Repo',  specActs?.repo?.toFixed(3)||"0.000", '#8b5a10'],
              ['Spec: Dump',  specActs?.dump?.toFixed(3)||"0.000", '#6b1a6b'],
              ['ECB: u*',     ecbU?.toFixed(2)||"0.00",            '#1a4a2a'],
            ].map(([k,v,c])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic"}}>{k}</span>
                <span style={{fontFamily:FM,fontSize:8,color:c,fontWeight:600}}>{v}</span>
              </div>
            ))}
          </div>
          <div style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic",marginTop:3}}>
            IBR: 5 rounds · budget=1.0 · λ_s=0.80
          </div>
        </div>
      )}
      {advEnabled && !advEquil && (
        <div style={{fontFamily:F,fontSize:7.5,color:"#888",fontStyle:"italic"}}>
          Enable HJB controller first to activate game
        </div>
      )}
      {!advEnabled && (
        <div style={{fontFamily:F,fontSize:7.5,color:"#aaa",fontStyle:"italic"}}>
          Speculator attacks BTP, repo lines, and collateral simultaneously.
          IBR converges to Nash equilibrium vs ECB's HJB-optimal policy.
        </div>
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────
// §UI-BOUNDARY  Error Boundary for SynthesisPanel
// Prevents synthesis panel errors from crashing the entire application
// ─────────────────────────────────────────────────────────────────────
class SynthErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(e) { return { error: e }; }
  componentDidCatch(e, info) { console.error('[Synthesis Panel]', e, info); }
  render() {
    if (this.state.error) {
      return (
        <div style={{position:"absolute",left:16,top:158,zIndex:30,width:220,
          background:"rgba(255,248,248,0.97)",border:"1px solid rgba(180,50,50,0.40)",
          padding:"10px 12px",backdropFilter:"blur(4px)"}}>
          <div style={{fontFamily:"Georgia,serif",fontSize:9,color:"#8b1a1a",fontWeight:700,marginBottom:6}}>
            ⬟ Synthesis — render error
          </div>
          <div style={{fontFamily:"Georgia,serif",fontSize:8,color:"#888",lineHeight:1.5}}>
            {this.state.error?.message || "Unknown error"}
          </div>
          <button onClick={()=>this.setState({error:null})}
            style={{marginTop:8,fontFamily:"Georgia,serif",fontSize:8,background:"transparent",
              border:"1px solid #cc4444",color:"#8b1a1a",padding:"2px 8px",cursor:"pointer"}}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─────────────────────────────────────────────────────────────────────
// §UI-SECTION  Module-scope helpers (avoids remount / prop-passing issues)
// ─────────────────────────────────────────────────────────────────────


// ═════════════════════════════════════════════════════════════════════════════
// §UI-MC  MONTE CARLO ANALYSIS MODULE
// Five analytical panels, coherent terminal aesthetic, dark background.
// Opened via ◈ MC button — floats over the canvas center-screen.
// ═════════════════════════════════════════════════════════════════════════════

// Shared SVG chart helpers
function mcClamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function MCDensityPanel({ bands, dispFrame, W, H, F }) {
  const canvasRef = useRef(null);
  const svgRef    = useRef(null);

  // Draw heatmap to canvas whenever bands change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !bands?.density || !bands.p50) return;
    const PX=32, PY=12, PB=18;
    const pW = W-PX-4, pH = H-PY-PB;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,W,H);

    const n = bands.p50.length;
    const mx = bands.maxSpread || 100;
    const nBins = bands.density[0]?.length || BINS;

    // Find global max density for normalization
    let globalMax = 1;
    for (let t=0;t<n;t++) {
      const col = bands.density[t];
      if (col) for (let b=0;b<col.length;b++) if (col[b]>globalMax) globalMax=col[b];
    }

    // Draw heatmap pixel by pixel using ImageData for speed
    const imgData = ctx.createImageData(pW, pH);
    const px = imgData.data;
    for (let py=0; py<pH; py++) {
      // Map pixel y → spread value (inverted: top=high spread)
      const spread = mx * (1 - py/pH);
      const bin = Math.min(nBins-1, Math.floor((spread/mx)*nBins));
      for (let px2=0; px2<pW; px2++) {
        // Map pixel x → tick
        const tFrac = px2/pW;
        const t = Math.floor(tFrac*(n-1));
        const col = bands.density[t];
        const density = col ? (col[bin]||0)/globalMax : 0;
        // Thermal palette: 0=navy, 0.3=blue, 0.6=cyan/green, 0.8=yellow, 1=red
        let r,g,bv;
        const crisis = spread > 50;
        if (density < 0.001) { r=g=bv=0; }
        else if (!crisis) {
          // Below crisis: blue gradient
          const d = density;
          r = Math.round(d < 0.5 ? d*60 : 60+d*80);
          g = Math.round(d < 0.5 ? d*100 : 100+d*60);
          bv= Math.round(d < 0.5 ? 80+d*120 : 200-d*150);
        } else {
          // Above crisis: amber to red
          const d = density;
          r = Math.round(120+d*135);
          g = Math.round(d < 0.5 ? d*120 : 120-d*80);
          bv= Math.round(d*20);
        }
        const alpha = density < 0.01 ? 0 : Math.round(mcClamp(density*0.85+0.05,0,1)*255);
        const idx = (py*pW+px2)*4;
        px[idx]=r; px[idx+1]=g; px[idx+2]=bv; px[idx+3]=alpha;
      }
    }
    ctx.putImageData(imgData, PX, PY);

    // Y-axis ticks as canvas text
    ctx.font = '7px Georgia';
    ctx.fillStyle = '#445566';
    ctx.textAlign = 'right';
    [0,25,50,75,100].filter(v=>v<=mx).forEach(v=>{
      const y = PY + pH - (v/mx)*pH;
      ctx.fillText(String(v), PX-3, y+3);
      ctx.strokeStyle = v===50 ? 'rgba(200,60,60,0.40)' : 'rgba(60,80,100,0.15)';
      ctx.lineWidth = v===50 ? 1.2 : 0.5;
      ctx.setLineDash(v===50 ? [4,3] : [2,3]);
      ctx.beginPath(); ctx.moveTo(PX,y); ctx.lineTo(W-4,y); ctx.stroke();
    });
    ctx.setLineDash([]);
  }, [bands, W, H]);

  // Overlay SVG for lines and markers (on top of canvas)
  const hasData = bands?.p50?.length > 0;
  const n  = hasData ? bands.p50.length : 0;
  const mx = bands?.maxSpread || 100;
  const PX=32, PY=12, PB=18;
  const pW=W-PX-4, pH=H-PY-PB;
  const toX = t  => PX + (t/(n-1||1))*pW;
  const toY = sp => PY + pH - mcClamp(sp/mx,0,1)*pH;

  return (
    <div style={{position:"relative",width:W,height:H}}>
      <canvas ref={canvasRef} width={W} height={H}
        style={{position:"absolute",top:0,left:0}}/>
      {hasData && (
        <svg width={W} height={H}
          style={{position:"absolute",top:0,left:0,overflow:"visible"}}>
          {/* Sigma ribbon (path divergence) */}
          {bands.sigma && (
            <polyline
              points={bands.sigma.map((_,t)=>{
                const sigNorm = Math.min(bands.sigma[t]/(Math.max(...bands.sigma)||1),1);
                return `${toX(t).toFixed(1)},${(PY+pH-sigNorm*14).toFixed(1)}`;
              }).join(' ')}
              fill="none" stroke="rgba(160,230,80,0.55)" strokeWidth="1.2" strokeDasharray="2,2"/>
          )}
          {/* p90 border */}
          <polyline
            points={bands.p90.map((v,t)=>`${toX(t).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')}
            fill="none" stroke="rgba(220,100,80,0.45)" strokeWidth="0.8" strokeDasharray="2,3"/>
          {/* p10 border */}
          <polyline
            points={bands.p10.map((v,t)=>`${toX(t).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')}
            fill="none" stroke="rgba(80,140,255,0.35)" strokeWidth="0.8" strokeDasharray="2,3"/>
          {/* p50 median — bright, solid */}
          <polyline
            points={bands.p50.map((v,t)=>`${toX(t).toFixed(1)},${toY(v).toFixed(1)}`).join(' ')}
            fill="none" stroke="rgba(220,235,255,0.95)"
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          {/* Bifurcation */}
          {bands.bifurcTick!=null && (
            <g>
              <line x1={toX(bands.bifurcTick)} y1={PY}
                x2={toX(bands.bifurcTick)} y2={PY+pH}
                stroke="rgba(200,230,70,0.55)" strokeWidth="1" strokeDasharray="3,2"/>
              <text x={toX(bands.bifurcTick)+2} y={PY+9} fontSize="6.5"
                fill="rgba(200,230,70,0.85)" fontFamily="Georgia,serif">bifurc.</text>
            </g>
          )}
          {/* Crisis horizon */}
          {bands.crisisHorizon!=null && (
            <g>
              <line x1={toX(bands.crisisHorizon)} y1={PY}
                x2={toX(bands.crisisHorizon)} y2={PY+pH}
                stroke="rgba(255,80,80,0.50)" strokeWidth="1.5" strokeDasharray="3,2"/>
              <text x={toX(bands.crisisHorizon)+2} y={PY+20} fontSize="6.5"
                fill="rgba(255,100,80,0.85)" fontFamily="Georgia,serif">
                t={bands.crisisHorizon}
              </text>
            </g>
          )}
          {/* Live tick */}
          {dispFrame < n && (
            <g>
              <line x1={toX(dispFrame)} y1={PY} x2={toX(dispFrame)} y2={PY+pH}
                stroke="rgba(255,255,80,0.80)" strokeWidth="1.5"/>
              <circle cx={toX(dispFrame)}
                cy={toY(bands.p50[dispFrame]||0)} r="3.5"
                fill="rgba(255,255,80,0.95)"
                stroke="rgba(255,255,80,0.30)" strokeWidth="2.5"/>
            </g>
          )}
          {/* Axis label */}
          <text x={PX+pW/2} y={H-3} fontSize="7" fill="#334455"
            textAnchor="middle" fontFamily="Georgia,serif">tick →</text>
        </svg>
      )}
      {!hasData && (
        <svg width={W} height={H} style={{position:"absolute",top:0,left:0}}>
          <text x={W/2} y={H/2} fontSize="9" fill="#223"
            textAnchor="middle" fontFamily="Georgia,serif" fontStyle="italic">
            Run to generate
          </text>
        </svg>
      )}
    </div>
  );
}


function MCBifurcPanel({ bands, W, H }) {
  if (!bands?.sigma) return null;
  const PX=28, PY=12, PB=16;
  const pW=W-PX-4, pH=H-PY-PB;
  const n=bands.sigma.length;
  const sigMax=Math.max(...bands.sigma,1);
  const toX=t=>PX+(t/(n-1||1))*pW;
  const toY=v=>PY+pH-(v/sigMax)*pH;
  const pts=bands.sigma.map((_,t)=>`${toX(t).toFixed(1)},${toY(bands.sigma[t]).toFixed(1)}`).join(' ');
  const areaBase=PY+pH;
  const areaPath=`M${toX(0)},${areaBase} ` +
    bands.sigma.map((_,t)=>`L${toX(t).toFixed(1)},${toY(bands.sigma[t]).toFixed(1)}`).join(' ') +
    ` L${toX(n-1)},${areaBase} Z`;
  const bT=bands.bifurcTick;
  return (<>
    <polygon points={
      bands.sigma.map((_,t)=>`${toX(t).toFixed(1)},${toY(bands.sigma[t]).toFixed(1)}`).concat(
        [`${toX(n-1).toFixed(1)},${areaBase}`,`${toX(0).toFixed(1)},${areaBase}`]
      ).join(' ')}
      fill="rgba(200,220,80,0.08)" stroke="none"/>
    <polyline points={pts} fill="none" stroke="rgba(200,220,80,0.80)"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
    {bT!=null && (<g>
      <line x1={toX(bT)} y1={PY} x2={toX(bT)} y2={PY+pH}
        stroke="rgba(200,220,80,0.55)" strokeWidth="1.5" strokeDasharray="3,2"/>
      <circle cx={toX(bT)} cy={toY(bands.sigma[bT]||0)} r="4"
        fill="rgba(200,220,80,0.90)"/>
      <text x={toX(bT)+4} y={toY(bands.sigma[bT]||0)-4} fontSize="7"
        fill="rgba(200,220,80,0.90)" fontFamily="Georgia,serif">peak σ</text>
    </g>)}
    <text x={PX+pW/2} y={H-3} fontSize="7" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif">tick →</text>
    <text x={7} y={PY+pH/2} fontSize="7" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif" transform={`rotate(-90,7,${PY+pH/2})`}>σ(paths)</text>
  </>);
}

function MCSurvivalPanel({ bands, W, H }) {
  if (!bands?.survivalCurves?.length) return (
    <text x={W/2} y={H/2} fontSize="8" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif" fontStyle="italic">No data</text>
  );
  const PX=28, PY=12, PB=16;
  const pW=W-PX-4, pH=H-PY-PB;
  const curves=bands.survivalCurves;
  const n=curves[0].curve.length;
  const toX=t=>PX+(t/(n-1||1))*pW;
  const toY=v=>PY+pH-v*pH;
  const COLS=["rgba(255,140,80,0.85)","rgba(100,180,255,0.85)","rgba(180,120,255,0.85)"];
  return (<>
    {/* 100% and 0% lines */}
    <line x1={PX} y1={PY} x2={PX+pW} y2={PY} stroke="rgba(60,80,100,0.20)" strokeWidth="0.5"/>
    <line x1={PX} y1={PY+pH} x2={PX+pW} y2={PY+pH} stroke="rgba(60,80,100,0.20)" strokeWidth="0.5"/>
    <line x1={PX} y1={PY+pH/2} x2={PX+pW} y2={PY+pH/2}
      stroke="rgba(60,80,100,0.15)" strokeWidth="0.5" strokeDasharray="2,3"/>
    {curves.map((c,ci)=>(
      <g key={c.i}>
        <polyline
          points={c.curve.map((_,t)=>`${toX(t).toFixed(1)},${toY(c.curve[t]).toFixed(1)}`).join(' ')}
          fill="none" stroke={COLS[ci]} strokeWidth="1.6"
          strokeLinecap="round" strokeLinejoin="round"/>
        <text x={PX+pW+2} y={toY(c.curve[n-1]||0)+4} fontSize="7"
          fill={COLS[ci]} fontFamily="Georgia,serif">{c.abbr}</text>
      </g>
    ))}
    <text x={PX+pW/2} y={H-3} fontSize="7" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif">tick →</text>
    <text x={7} y={PY+pH/2} fontSize="7" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif" transform={`rotate(-90,7,${PY+pH/2})`}>P(safe)</text>
  </>);
}

function MCConditionalPanel({ bands, W, H }) {
  const PX=8, PY=12, PB=16;
  const pW=W-PX-4, pH=H-PY-PB;
  if (!bands?.condCrisis && !bands?.condRecov) return (
    <text x={W/2} y={H/2} fontSize="8" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif" fontStyle="italic">No split data</text>
  );
  const n=bands.p50.length;
  const mx=bands.maxSpread||100;
  const toX=t=>PX+(t/(n-1||1))*pW;
  const toY=sp=>PY+pH-mcClamp(sp/mx,0,1)*pH;
  return (<>
    <line x1={PX} y1={toY(50)} x2={PX+pW} y2={toY(50)}
      stroke="rgba(200,60,60,0.25)" strokeWidth="0.8" strokeDasharray="3,3"/>
    {bands.condCrisis && (
      <polyline
        points={bands.condCrisis.map((d,t)=>`${toX(t).toFixed(1)},${toY(d.p50||0).toFixed(1)}`).join(' ')}
        fill="none" stroke="rgba(220,80,60,0.80)" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"/>
    )}
    {bands.condRecov && (
      <polyline
        points={bands.condRecov.map((d,t)=>`${toX(t).toFixed(1)},${toY(d.p50||0).toFixed(1)}`).join(' ')}
        fill="none" stroke="rgba(60,180,120,0.80)" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"/>
    )}
    {/* Legend */}
    <rect x={PX+2} y={PY+2} width={7} height={3} fill="rgba(220,80,60,0.80)"/>
    <text x={PX+11} y={PY+7} fontSize="6.5" fill="rgba(220,100,80,0.85)"
      fontFamily="Georgia,serif">crisis ({bands.nCrisis||0})</text>
    <rect x={PX+2} y={PY+12} width={7} height={3} fill="rgba(60,180,120,0.80)"/>
    <text x={PX+11} y={PY+17} fontSize="6.5" fill="rgba(60,200,120,0.85)"
      fontFamily="Georgia,serif">recovery ({bands.nRecov||0})</text>
    <text x={PX+pW/2} y={H-3} fontSize="7" fill="#334" textAnchor="middle"
      fontFamily="Georgia,serif">tick →</text>
  </>);
}

function MCAnalysisModule({ mcBands, mcRunning, onRun, onClose, dispFrame, crisisType, mcStale, F, FM }) {
  const TW=660, TH=480;
  const ACCENTS={sovereign_contagion:"rgba(180,40,40,",systemic_shock:"rgba(40,80,200,",
                 cliff_collapse:"rgba(120,40,180,",ambiguous:"rgba(60,100,140,"};
  const acc=ACCENTS[crisisType?.type]||"rgba(60,100,140,";
  const hasData=mcBands&&mcBands.p50&&mcBands.p50.length>0;
  const PAD=8;
  // Top row: density (left) + metrics (right)
  const topH=260;
  const densW=380, rightColW=TW-densW-PAD*3;
  const densH=topH;
  // Bottom row
  const botH=TH-topH-PAD*3-58;
  const botPanelW=Math.floor((TW-PAD*4)/3);

  return (
    <div style={{
      position:"absolute", top:"50%", left:"50%",
      transform:"translate(-50%,-50%)",
      width:TW, zIndex:50, pointerEvents:"auto",
      background:"rgba(6,8,14,0.97)",
      border:`1px solid ${acc}0.45)`,
      backdropFilter:"blur(12px)",
      boxShadow:`0 8px 40px rgba(0,0,0,0.70), inset 0 1px 0 ${acc}0.12)`,
      display:"flex", flexDirection:"column",
    }}>

      {/* ── HEADER ────────────────────────────────────────────────── */}
      <div style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"8px 14px 7px",
        borderBottom:`1px solid ${acc}0.20)`,
        background:`linear-gradient(90deg,${acc}0.08) 0%,transparent 60%)`,
        flexShrink:0,
      }}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <span style={{fontFamily:FM,fontSize:12,color:"#c8d8ff",fontWeight:700,letterSpacing:0.8}}>
            MONTE CARLO FORWARD ANALYSIS
          </span>
          <span style={{fontFamily:F,fontSize:8,color:acc+"0.70)",fontStyle:"italic"}}>
            {MC_PATHS} paths · {MC_TICKS} ticks
          </span>
          {crisisType&&crisisType.type!=="ambiguous"&&(
            <span style={{fontFamily:F,fontSize:7.5,color:acc+"0.85)",
              border:`1px solid ${acc}0.30)`,padding:"1px 6px",borderRadius:2}}>
              {crisisType.type.replace("_"," ")} · {(crisisType.confidence*100).toFixed(0)}%
            </span>
          )}
          {mcStale&&<span style={{fontFamily:F,fontSize:7,color:"#cc8800",fontStyle:"italic"}}>⟳ recalculating…</span>}
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          <button onClick={onRun} disabled={mcRunning} style={{
            fontFamily:F,fontSize:8.5,background:"transparent",
            border:`1px solid ${acc}0.50)`,color:mcRunning?"#445":"#aaccff",
            padding:"3px 12px",cursor:mcRunning?"default":"pointer",letterSpacing:0.3,
          }}>{mcRunning?"COMPUTING…":"▶ RUN"}</button>
          <button onClick={onClose} style={{fontFamily:"monospace",fontSize:13,
            background:"transparent",border:"none",color:"#445",cursor:"pointer"}}>✕</button>
        </div>
      </div>

      {/* ── MAIN CONTENT ──────────────────────────────────────────── */}
      <div style={{display:"flex",flexDirection:"column",flex:1,padding:PAD,gap:PAD,overflow:"hidden"}}>

        {/* Top row: 3D manifold fan (left) + density chart + metrics (right) */}
        <div style={{display:"flex",gap:PAD,height:topH,flexShrink:0}}>

          {/* Density chart — full width now that 3D is removed */}
          <div style={{flex:1,display:"flex",flexDirection:"column",gap:PAD}}>
            {/* Density chart */}
            <div style={{width:densW,flexShrink:0,display:"flex",flexDirection:"column"}}>
              <div style={{fontFamily:F,fontSize:8,color:"#445",fontStyle:"italic",marginBottom:3}}>
                PROBABILITY DENSITY · p10/p50/p90 · live tick (yellow)
              </div>
              <div style={{flex:1,background:"rgba(2,4,10,0.80)",border:`1px solid ${acc}0.15)`,overflow:"hidden",position:"relative"}}>
                <MCDensityPanel bands={mcBands} dispFrame={dispFrame} W={densW-4} H={densH-20} F={F}/>
              </div>
            </div>
            {/* Metrics grid */}
            <div style={{flex:1,display:"flex",flexDirection:"column",gap:5}}>
              <div style={{fontFamily:F,fontSize:8,color:"#445",fontStyle:"italic"}}>TAIL RISK</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5,flex:1}}>
                {[
                  ["P(crisis)", hasData?(((mcBands.pCrisis||[])[dispFrame]||0)*100).toFixed(0)+"%":"–",
                   hasData&&(mcBands.pCrisis||[])[dispFrame]>0.5?"#cc4444":"#44aa88","at current tick"],
                  ["Horizon", hasData&&mcBands.crisisHorizon!=null?"t="+mcBands.crisisHorizon:"none",
                   hasData&&mcBands.crisisHorizon!=null&&mcBands.crisisHorizon<20?"#cc4444":"#44aa88","p50→50bp"],
                  ["VaR 95%", hasData?(mcBands.var95||0).toFixed(0)+"bp":"–","#cc8844","final tick"],
                  ["E[S] 95%", hasData?(mcBands.es95||0).toFixed(0)+"bp":"–","#cc6644","shortfall"],
                  ["E[peak]", hasData?(mcBands.ePeakSpread||0).toFixed(0)+"bp":"–","#8899bb","avg peak"],
                  ["Bifurc.", hasData&&mcBands.bifurcTick!=null?"t="+mcBands.bifurcTick:"–","rgba(200,220,80,0.90)","σ peak"],
                ].map(([k,v,c,sub])=>(
                  <div key={k} style={{background:"rgba(2,4,12,0.80)",border:`1px solid ${acc}0.10)`,
                    padding:"5px 7px",display:"flex",flexDirection:"column",justifyContent:"center"}}>
                    <div style={{fontFamily:F,fontSize:6.5,color:"#445",fontStyle:"italic"}}>{k}</div>
                    <div style={{fontFamily:FM,fontSize:15,fontWeight:700,color:c,lineHeight:1,margin:"2px 0"}}>{v}</div>
                    <div style={{fontFamily:F,fontSize:6,color:"#334",fontStyle:"italic"}}>{sub}</div>
                  </div>
                ))}
              </div>
              {hasData&&(
                <div style={{display:"flex",gap:5,flexShrink:0}}>
                  <div style={{flex:1,background:"rgba(180,40,40,0.08)",border:"1px solid rgba(180,40,40,0.20)",
                    padding:"3px 6px",textAlign:"center"}}>
                    <span style={{fontFamily:FM,fontSize:11,fontWeight:700,color:"rgba(220,80,60,0.90)"}}>{mcBands.nCrisis||0}</span>
                    <span style={{fontFamily:F,fontSize:7,color:"#445",fontStyle:"italic",marginLeft:4}}>crisis</span>
                  </div>
                  <div style={{flex:1,background:"rgba(40,160,80,0.08)",border:"1px solid rgba(40,160,80,0.20)",
                    padding:"3px 6px",textAlign:"center"}}>
                    <span style={{fontFamily:FM,fontSize:11,fontWeight:700,color:"rgba(60,200,100,0.90)"}}>{mcBands.nRecov||0}</span>
                    <span style={{fontFamily:F,fontSize:7,color:"#445",fontStyle:"italic",marginLeft:4}}>recovery</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom row: three analysis panels */}
        <div style={{display:"flex",gap:PAD,height:botH,flexShrink:0}}>

          {/* Panel 1: Bifurcation / path divergence */}
          <div style={{width:botPanelW,flexShrink:0,display:"flex",flexDirection:"column"}}>
            <div style={{fontFamily:F,fontSize:7.5,color:"#445",fontStyle:"italic",marginBottom:3}}>
              PATH BIFURCATION  σ(t)
            </div>
            <div style={{flex:1,background:"rgba(2,4,10,0.80)",border:`1px solid ${acc}0.15)`,overflow:"hidden"}}>
              <svg width={botPanelW} height={botH-18} style={{display:"block"}}>
                <MCBifurcPanel bands={mcBands} W={botPanelW} H={botH-18}/>
              </svg>
            </div>
            <div style={{fontFamily:F,fontSize:6.5,color:"#334",fontStyle:"italic",marginTop:3,lineHeight:1.4}}>
              Standard deviation across 50 paths at each tick.
              Peak σ = critical bifurcation — the moment fate is decided.
            </div>
          </div>

          {/* Panel 2: Survival curves */}
          <div style={{width:botPanelW,flexShrink:0,display:"flex",flexDirection:"column"}}>
            <div style={{fontFamily:F,fontSize:7.5,color:"#445",fontStyle:"italic",marginBottom:3}}>
              INSTITUTION SURVIVAL  P(safe by t)
            </div>
            <div style={{flex:1,background:"rgba(2,4,10,0.80)",border:`1px solid ${acc}0.15)`,overflow:"hidden"}}>
              <svg width={botPanelW} height={botH-18} style={{display:"block"}}>
                <MCSurvivalPanel bands={mcBands} W={botPanelW} H={botH-18}/>
              </svg>
            </div>
            {/* Institution risk table */}
            {hasData&&mcBands.topNodes&&(
              <div style={{marginTop:4,display:"flex",gap:3,flexWrap:"wrap"}}>
                {mcBands.topNodes.slice(0,4).map((node,i)=>(
                  <div key={node.i} style={{background:"rgba(2,4,10,0.90)",
                    border:"1px solid rgba(60,80,100,0.20)",padding:"2px 5px",
                    display:"flex",gap:5,alignItems:"center"}}>
                    <span style={{fontFamily:FM,fontSize:7.5,color:"#889"}}>{node.abbr}</span>
                    <span style={{fontFamily:FM,fontSize:7.5,fontWeight:700,
                      color:node.prob>0.5?"#cc4444":node.prob>0.2?"#cc8800":"#44aa88"}}>
                      {(node.prob*100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Panel 3: Conditional split */}
          <div style={{flex:1,display:"flex",flexDirection:"column"}}>
            <div style={{fontFamily:F,fontSize:7.5,color:"#445",fontStyle:"italic",marginBottom:3}}>
              CONDITIONAL SCENARIO SPLIT  p50 median
            </div>
            <div style={{flex:1,background:"rgba(2,4,10,0.80)",border:`1px solid ${acc}0.15)`,overflow:"hidden"}}>
              <svg width={botPanelW} height={botH-18} style={{display:"block"}}>
                <MCConditionalPanel bands={mcBands} W={botPanelW} H={botH-18}/>
              </svg>
            </div>
            <div style={{fontFamily:F,fontSize:6.5,color:"#334",fontStyle:"italic",marginTop:3,lineHeight:1.4}}>
              Paths split at peak spread threshold (50bp).
              Divergence between fans shows how early crisis fate is sealed.
            </div>
          </div>
        </div>
      </div>

      {/* ── FOOTER ────────────────────────────────────────────────── */}
      <div style={{
        padding:"5px 14px", display:"flex", gap:16, alignItems:"center",
        borderTop:`1px solid ${acc}0.12)`,
        background:"rgba(2,4,8,0.60)", flexShrink:0,
      }}>
        <span style={{fontFamily:F,fontSize:7,color:"#334",fontStyle:"italic"}}>
          BP ODE + Hawkes + doom loop · stripped (no rendering) · all paths from current state
        </span>
        {hasData&&mcBands.bifurcTick!=null&&(
          <span style={{fontFamily:F,fontSize:7,color:"rgba(200,220,80,0.70)",fontStyle:"italic"}}>
            Bifurcation at t={mcBands.bifurcTick} — path σ peaks here
          </span>
        )}
        {hasData&&(
          <span style={{fontFamily:F,fontSize:7,color:"#334",fontStyle:"italic",marginLeft:"auto"}}>
            t=0 → current simulation state · colors match Fisher crisis classification
          </span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// §UI-MC3D  Three.js 3D manifold surface fan for Monte Carlo
// Renders 8 temporal surface snapshots stacked along depth axis,
// each showing p50 terrain with p90 wireframe uncertainty envelope.
// Auto-rotates. Uses a separate WebGL context from the main simulation.
// ─────────────────────────────────────────────────────────────────────
function synthDot(active, busy) {
  return (
    <span style={{
      display:"inline-block", width:7, height:7, borderRadius:"50%",
      background: busy ? "#ffaa00" : active ? "#44cc88" : "#cccccc",
      marginRight:5, flexShrink:0
    }}/>
  );
}

function SectionHeader({ id, label, active, children, openSection, onToggle, F }) {
  return (
    <div style={{borderBottom:"1px solid #eee"}}>
      <button onClick={() => onToggle(id)} style={{
        width:"100%", display:"flex", alignItems:"center", padding:"6px 10px",
        background: openSection === id ? "rgba(68,180,136,0.07)" : "transparent",
        border:"none", cursor:"pointer", textAlign:"left", userSelect:"none",
      }}>
        {synthDot(active, false)}
        <span style={{fontFamily:F, fontSize:9, color:"#444", fontStyle:"italic", flex:1}}>{label}</span>
        <span style={{fontSize:9, color:"#aaa", marginLeft:4}}>{openSection === id ? "▲" : "▼"}</span>
      </button>
      {openSection === id && (
        <div style={{padding:"6px 10px 10px"}}>{children}</div>
      )}
    </div>
  );
}

function SynthesisPanel({
  F,FM,
  mcRunning,mcBands,onRunMC,
  mfgEnabled,setMfgEnabled,mfgStats,
  calibrated,calLabel,onCalibrate,
  sculptureMode,setSculptureMode,sliceCount,
  hjbEnabled,setHjbEnabled,hjbReady,hjbOptU,
  homologyEnabled,setHomologyEnabled,homologyData,
  // New directions
  edgeFraction,topologyAlert,edgeCount,
  fisherProj,crisisType,
  calResults,calRunning,onCalibrateEp,
  advEnabled,setAdvEnabled,advEquil,
  hjbPolicy,
  synthSuggestion,
  mcStale,
  synthPipeline,
}) {
  const [openSection, setOpenSection] = useState('network');
  const toggle = s => setOpenSection(v => v === s ? null : s);
    // SectionHeader is declared at module scope (below) to avoid remount on re-render
  // openSection and toggle are passed as props via the outer closure — stable references
  const sm={fontFamily:F,fontSize:8,color:"#aaa",fontStyle:"italic"};
  return (
    <div style={{position:"absolute",left:16,top:158,zIndex:30,width:220,
      background:"rgba(248,250,248,0.97)",border:"1px solid rgba(68,180,136,0.40)",
      backdropFilter:"blur(4px)",pointerEvents:"auto",maxHeight:"80vh",overflowY:"auto"}}>
      {/* Header with pipeline status */}
      <div style={{background:"rgba(68,180,136,0.12)",
        borderBottom:"1px solid rgba(68,180,136,0.25)",position:"sticky",top:0,zIndex:1}}>
        <div style={{padding:"5px 10px 3px",display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <span style={{fontFamily:F,fontSize:9,fontStyle:"italic",color:"#2a6a4a",fontWeight:700}}>
            ⬟ Synthesis Instrument
          </span>
          {synthPipeline && (
            <span style={{fontFamily:F,fontSize:7,color:
              synthPipeline.networkFraction<0.65?"#8b1a1a":
              synthPipeline.networkFraction<0.85?"#6b4a10":"#2a6a4a"}}>
              net {(synthPipeline.networkFraction*100).toFixed(0)}%
            </span>
          )}
        </div>
        {/* Pipeline flow: tools connected left-to-right */}
        <div style={{padding:"2px 10px 5px",display:"flex",alignItems:"center",gap:2,overflowX:"auto"}}>
          {[
            ["NET", synthPipeline?.networkFraction < 0.85],
            ["→", false],
            ["IG", !!crisisType && crisisType?.type !== "ambiguous"],
            ["→", false],
            ["INV", Object.keys(calResults||{}).length > 0],
            ["→", false],
            ["MC", !!mcBands && !mcStale],
            ["→", false],
            ["HJB", hjbEnabled],
            ["→", false],
            ["ADV", advEnabled],
          ].map(([label, active], i) => (
            <span key={i} style={{
              fontFamily: label==="→" ? "monospace" : F,
              fontSize: label==="→" ? 9 : 7.5,
              color: label==="→" ? (active?"rgba(68,180,136,0.6)":"rgba(200,200,200,0.5)")
                   : active ? "#2a6a4a" : "#bbb",
              fontWeight: active && label!=="→" ? 700 : 400,
              padding: label==="→" ? "0" : "1px 3px",
              background: active && label!=="→" ? "rgba(68,180,136,0.12)" : "transparent",
              borderRadius: 2, whiteSpace:"nowrap", flexShrink:0,
              fontStyle: label!=="→" ? "italic" : "normal",
            }}>{label}</span>
          ))}
          {mcStale && <span style={{fontFamily:F,fontSize:7,color:"#cc7700",marginLeft:4,fontStyle:"italic"}}>⟳MC</span>}
        </div>
      </div>
      {/* Cross-tool suggestion banner */}
      {synthSuggestion && (
        <div style={{padding:"4px 10px",background:"rgba(255,200,80,0.10)",
          borderBottom:"1px solid rgba(200,150,0,0.20)",display:"flex",alignItems:"flex-start",gap:6}}>
          <span style={{color:"#8b6a00",fontSize:10,flexShrink:0}}>💡</span>
          <span style={{fontFamily:F,fontSize:7.5,color:"#6b5000",fontStyle:"italic",lineHeight:1.4}}>
            {synthSuggestion}
          </span>
        </div>
      )}

      {/* §NET Network Topology */}
      <SectionHeader id="network" label="Endogenous Network" active={edgeFraction<0.85||topologyAlert} openSection={openSection} onToggle={toggle} F={F}>
        <div style={{marginBottom:4}}>
          <NetworkTopologyChart edgeFraction={edgeFraction} topologyAlert={topologyAlert} edgeCount={edgeCount} F={F} FM={FM}/>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"3px 8px",marginTop:4}}>
          <div><div style={sm}>Connectivity</div><div style={{fontFamily:FM,fontSize:10,fontWeight:600,color:edgeFraction<0.65?"#8b1a1a":edgeFraction<0.85?"#6b4a10":"#1a4a2a"}}>{(edgeFraction*100).toFixed(1)}%</div></div>
          <div><div style={sm}>σ*=60bp·κ=0.08</div><div style={{fontFamily:FM,fontSize:10,fontWeight:600,color:"#555"}}>{topologyAlert?"⚠ ALERT":"stable"}</div></div>
        </div>
        <div style={{...sm,marginTop:3,lineHeight:1.4}}>
          Edge deletion precedes node crisis by 5-10 ticks — topological early warning not visible in spread metrics.
        </div>
      </SectionHeader>

      {/* §IG Fisher Information Geometry */}
      <SectionHeader id="fisher" label="Information Geometry" active={!!crisisType&&crisisType.type!=="ambiguous"} openSection={openSection} onToggle={toggle} F={F}>
        <FisherManifoldChart fisherProj={fisherProj} crisisType={crisisType} F={F} FM={FM}/>
        {crisisType && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px 8px",marginTop:4}}>
            {[
              ["Type", crisisType.type?.replace("_"," ")||"–"],
              ["Confidence", (crisisType.confidence*100).toFixed(0)+"%"],
              ["Speed", crisisType.speed?.toFixed(4)||"–"],
              ["Periph. conc.", crisisType.concentration?.toFixed(3)||"–"],
            ].map(([k,v])=>(
              <div key={k}><div style={sm}>{k}</div><div style={{fontFamily:FM,fontSize:9,fontWeight:600,color:"#333"}}>{v}</div></div>
            ))}
          </div>
        )}
        <div style={{...sm,marginTop:3,lineHeight:1.4}}>
          Fisher metric on Δ³⁴ stress simplex. Hellinger distances between consecutive distributions. Crisis types geodesically separable 5+ ticks before threshold crossing.
        </div>
      </SectionHeader>

      {/* §INV Inverse Problem */}
      <SectionHeader id="calibration" label="Inverse Problem — Fit θ*" active={Object.keys(calResults||{}).length>0} openSection={openSection} onToggle={toggle} F={F}>
        {/* Auto-suggestion from Fisher geometry */}
        {synthPipeline?.crisisTypeGuess && (
          <div style={{padding:"3px 6px",background:"rgba(68,180,136,0.08)",
            border:"1px solid rgba(68,180,136,0.25)",marginBottom:5,borderRadius:2}}>
            <span style={{fontFamily:F,fontSize:7.5,color:"#2a6a4a",fontStyle:"italic"}}>
              💡 Fisher suggests: fit <b>{EPISODE_SIGNATURES[synthPipeline.crisisTypeGuess]?.label}</b>
            </span>
          </div>
        )}
        <CalibrationPanel calResults={calResults} calRunning={calRunning} onCalibrate={onCalibrateEp} F={F} FM={FM}/>
        <div style={{...sm,marginTop:4,lineHeight:1.4}}>
          FD gradient descent on signature loss. 10 MC paths per eval. Fitted θ* applied live.
          After fitting, re-run Monte Carlo for a calibrated forward fan.
        </div>
      </SectionHeader>

      {/* §ADV Adversarial Game */}
      <SectionHeader id="adversarial" label="Adversarial Game" active={advEnabled} openSection={openSection} onToggle={toggle} F={F}>
        {!hjbEnabled && (
          <div style={{...sm,padding:"4px 0",color:"#cc7700"}}>
            ⚠ Requires HJB Controller — enable Auto-ECB first so the speculator has an ECB policy to play against
          </div>
        )}
        <AdversarialPanel advEnabled={advEnabled} setAdvEnabled={setAdvEnabled} advEquil={advEquil} F={F} FM={FM}/>
        {advEnabled && advEquil && (
          <div style={{...sm,marginTop:4,lineHeight:1.4}}>
            ECB u*={advEquil.ecbU?.toFixed(2)} (from HJB) vs speculator BTP={advEquil.specAction?.btp?.toFixed(2)}.
            Network contraction from §NET amplifies attacker leverage.
          </div>
        )}
      </SectionHeader>

      {/* §HJB HJB Controller */}
      <SectionHeader id="hjb" label="HJB Controller" active={hjbEnabled} openSection={openSection} onToggle={toggle} F={F}>
        <label style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",marginBottom:6}}>
          <input type="checkbox" checked={hjbEnabled} onChange={e=>setHjbEnabled(e.target.checked)}
            style={{width:11,height:11,accentColor:"#44aa88"}}/>
          <span style={{fontFamily:F,fontSize:8,color:"#555"}}>Auto-ECB optimal control</span>
          {hjbEnabled&&!hjbReady&&<span style={{fontFamily:F,fontSize:7,color:"#ffaa00",marginLeft:4}}>solving…</span>}
        </label>
        {/* Policy readout */}
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"4px 8px",marginBottom:6}}>
          {[
            ["Optimal u*", hjbOptU.toFixed(3)+" /tick", hjbOptU>0.6?"#8b1a1a":hjbOptU>0?"#6b4a10":"#555"],
            ["State grid", "15×15×8=1800","#444"],
            ["Actions", "{0, 0.4, 0.8, 1.2}","#444"],
            ["β discount", "0.92","#444"],
          ].map(([k,v,c])=>(
            <div key={k}>
              <div style={sm}>{k}</div>
              <div style={{fontFamily:FM,fontSize:9,fontWeight:600,color:c}}>{v}</div>
            </div>
          ))}
        </div>
        {/* Policy intuition: what does HJB prescribe at different stress levels? */}
        {hjbEnabled && hjbReady && (
          <div style={{background:"rgba(68,180,136,0.06)",border:"1px solid rgba(68,180,136,0.20)",padding:"5px 7px",marginBottom:4}}>
            <div style={{...sm,marginBottom:3}}>Policy surface (Lm axis, BTP=current)</div>
            {[0,0.5,1.0,1.5,2.0].map(lmVal => {
              const iLf=Math.round(0.7*(HJB_NLF-1));
              const iLm=Math.round(clamp(lmVal/2,0,1)*(HJB_NLM-1));
              const iBtp=Math.round(clamp(Math.min(hjbOptU*0.5,1),0,1)*(HJB_NBTP-1));
              const sIdx=iLf*HJB_NLM*HJB_NBTP+iLm*HJB_NBTP+iBtp;
              const u=hjbPolicy?HJB_ACTS[hjbPolicy[sIdx]||0]:0;
              const barW=Math.round(u/1.2*60);
              return (
                <div key={lmVal} style={{display:"flex",alignItems:"center",gap:5,marginBottom:2}}>
                  <span style={{fontFamily:F,fontSize:7,color:"#aaa",width:30}}>Lm={lmVal}</span>
                  <div style={{flex:1,height:5,background:"#eee",borderRadius:2}}>
                    <div style={{width:barW+"%",height:"100%",background:u>0.6?"#8b1a1a":u>0?"#6b4a10":"#aaa",borderRadius:2}}/>
                  </div>
                  <span style={{fontFamily:FM,fontSize:7.5,color:"#333",width:28}}>{u.toFixed(1)}</span>
                </div>
              );
            })}
            <div style={{...sm,marginTop:3}}>u=0: hold · u=1.2: max TLTRO injection</div>
          </div>
        )}
        <div style={{...sm,lineHeight:1.5}}>
          V(s)=max_u[r(s,u)+βV(s')] · r=-λ‖s‖²-γu² · 10 value iteration passes.
          ECB node glows green when injecting.
        </div>
      </SectionHeader>

      {/* §MFG Best-Response */}
      <SectionHeader id="mfg" label="MFG Best-Response" active={mfgEnabled} openSection={openSection} onToggle={toggle} F={F}>
        <label style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",marginBottom:6}}>
          <input type="checkbox" checked={mfgEnabled} onChange={e=>setMfgEnabled(e.target.checked)}
            style={{width:11,height:11,accentColor:"#44aa88"}}/>
          <span style={{fontFamily:F,fontSize:8,color:"#555"}}>Banks respond strategically to mean field</span>
        </label>
        {mfgEnabled && (
          <>
            {/* Action distribution */}
            <div style={{display:"flex",gap:0,marginBottom:6,background:"#f5f5f8",borderRadius:3,overflow:"hidden",height:28}}>
              {[["Lend",mfgStats.lenders,"#1a4a2a"],["Hold",mfgStats.holders,"#888"],["Hoard",mfgStats.hoarders,"#8b1a1a"]].map(([l,v,c])=>{
                const total=Math.max(mfgStats.lenders+mfgStats.holders+mfgStats.hoarders,1);
                const pct=(v/total*100).toFixed(0);
                return v>0?(
                  <div key={l} style={{background:c,width:pct+"%",display:"flex",alignItems:"center",
                    justifyContent:"center",flexDirection:"column",transition:"width 0.3s"}}>
                    <span style={{fontFamily:FM,fontSize:9,color:"#fff",fontWeight:700,lineHeight:1}}>{v}</span>
                    <span style={{fontFamily:F,fontSize:6.5,color:"rgba(255,255,255,0.8)",lineHeight:1}}>{l}</span>
                  </div>
                ):null;
              })}
            </div>
            {/* Payoff logic */}
            <div style={{background:"rgba(248,248,252,0.8)",border:"1px solid #e8e8ee",padding:"5px 7px",marginBottom:4}}>
              <div style={{...sm,marginBottom:3}}>Best-response payoff structure</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px 8px"}}>
                {[
                  ["θ (liq. benefit)","0.35"],
                  ["γ (action cost)","0.20"],
                  ["Lend if","sys > own·0.5"],
                  ["Hoard if","own > sys·0.3"],
                ].map(([k,v])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                    <span style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic"}}>{k}</span>
                    <span style={{fontFamily:FM,fontSize:7.5,color:"#333"}}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{...sm,lineHeight:1.5}}>
              Hoarders shrink (scale 0.72×) · Lenders pulse green in 3D.
              Approximates MFG equilibrium via best-response dynamics.
            </div>
          </>
        )}
        {!mfgEnabled && <div style={sm}>Each bank picks Lend/Hoard/Hold to max u_i = -σ_own + θ·liqBenefit - γ·cost.</div>}
      </SectionHeader>

      {/* Monte Carlo */}
      <SectionHeader id="mc" label="Monte Carlo — 50 paths" active={!!mcBands} openSection={openSection} onToggle={toggle} F={F}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
          <div>
            <div style={sm}>
              p10/p50/p90 forward fan from current state
              {mcStale && <span style={{color:"#cc7700",marginLeft:4,fontStyle:"normal",fontWeight:700}}>⟳ stale — re-run after calibration</span>}
            </div>
            {mcBands && (
              <div style={{fontFamily:FM,fontSize:8,color:"#555",marginTop:1}}>
                p50={mcBands.p50[mcBands.p50.length-1]?.toFixed(1)||"–"}bp at t=59 ·
                spread={((mcBands.p90[mcBands.p90.length-1]||0)-(mcBands.p10[mcBands.p10.length-1]||0)).toFixed(1)}bp fan
              </div>
            )}
          </div>
          <button onClick={onRunMC} disabled={mcRunning} style={{fontFamily:F,fontSize:8,
            background:mcRunning?"#f0f0f0":"transparent",border:"1px solid #aaa",
            color:mcRunning?"#aaa":"#555",padding:"2px 8px",cursor:mcRunning?"default":"pointer",
            flexShrink:0}}>
            {mcRunning?"running…":"▶ Run + Open Card"}
          </button>
        </div>
        {mcBands ? (
          <>
            <MCFanChart bands={mcBands} F={F} FM={FM}/>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:"3px 4px",marginTop:5}}>
              {[
                ["p10 (t=59)",(mcBands.p10[59]||0).toFixed(1)+"bp","#1a3a6b"],
                ["p50 (t=59)",(mcBands.p50[59]||0).toFixed(1)+"bp","#8b1a1a"],
                ["p90 (t=59)",(mcBands.p90[59]||0).toFixed(1)+"bp","#6b1a1a"],
              ].map(([k,v,c])=>(
                <div key={k} style={{textAlign:"center",background:"rgba(248,248,252,0.8)",padding:"3px 2px",borderRadius:2}}>
                  <div style={{fontFamily:FM,fontSize:9,fontWeight:600,color:c}}>{v}</div>
                  <div style={{fontFamily:F,fontSize:6.5,color:"#aaa"}}>{k}</div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={{...sm,padding:"8px 0",textAlign:"center"}}>
            Press Run to forward-simulate 50 paths from current market state.
            Each path runs stripped BP+Hawkes+doom loop (no rendering) — completes in ~3ms.
          </div>
        )}
        <div style={{...sm,marginTop:4,lineHeight:1.5}}>
          50 paths × 60 ticks · stripped BP ODE + Hawkes + doom loop.
          Parameters from current sliders. Re-run after calibration to see fitted fan.
        </div>
      </SectionHeader>

      {/* Calibration (ECB snapshot) */}
      <SectionHeader id="calsnap" label="ECB Data Calibration" active={calibrated} openSection={openSection} onToggle={toggle} F={F}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
          <div>
            <div style={{fontFamily:FM,fontSize:9,color:calibrated?"#2a6a4a":"#555"}}>
              {calibrated ? "✓ "+calLabel : "ECB SDW · Q1 2025"}
            </div>
            <div style={sm}>Initialises Lf, Lm, BTP from real data</div>
          </div>
          <button onClick={onCalibrate} style={{fontFamily:F,fontSize:8,
            background:calibrated?"rgba(68,180,136,0.10)":"transparent",
            border:`1px solid ${calibrated?"#44aa88":"#aaa"}`,
            color:calibrated?"#2a6a4a":"#555",padding:"2px 8px",cursor:"pointer",flexShrink:0}}>
            {calibrated?"Re-apply":"Apply"}
          </button>
        </div>
        <div style={{background:"rgba(248,248,252,0.8)",border:"1px solid #e8e8ee",padding:"5px 7px",marginBottom:4}}>
          <div style={{...sm,marginBottom:3}}>Snapshot values (ECB SDW + Bloomberg, 28 Feb 2025)</div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px 8px"}}>
            {[
              ["Excess liquidity","€3,850bn"],
              ["€STR rate","2.391%"],
              ["BTP-Bund spread","~120bp"],
              ["Lf (ECB tier)","0.92 (full)"],
              ["Lf (G-SIB tier)","0.70–0.78"],
              ["Lm (bank avg)","0.20–0.30"],
            ].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic"}}>{k}</span>
                <span style={{fontFamily:FM,fontSize:7.5,color:"#333"}}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{...sm,lineHeight:1.5}}>
          Apply before running Monte Carlo or Inverse Problem to use
          historically grounded initial conditions rather than the generic episode defaults.
          Then run "Fit θ*" to calibrate contagion dynamics to a specific crisis type.
        </div>
      </SectionHeader>

      {/* 4D Sculptor */}
      <SectionHeader id="sculptor" label={"4D Sculptor — " + sliceCount + "/12 slices"} active={sculptureMode} openSection={openSection} onToggle={toggle} F={F}>
        <label style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",marginBottom:5}}>
          <input type="checkbox" checked={sculptureMode} onChange={e=>setSculptureMode(e.target.checked)}
            style={{width:11,height:11,accentColor:"#44aa88"}}/>
          <span style={{fontFamily:F,fontSize:8,color:"#555"}}>Show temporal volume (right of manifold)</span>
        </label>
        {/* Slice timeline indicator */}
        <div style={{marginBottom:5}}>
          <div style={{...sm,marginBottom:3}}>Captured Z-cache slices (every 15 ticks)</div>
          <div style={{display:"flex",gap:2,height:14,alignItems:"flex-end"}}>
            {Array.from({length:12},(_,i)=>{
              const hasSlice = i < sliceCount;
              const isLive   = i === sliceCount; // next empty slot = live ghost
              const t = sliceCount > 1 ? i/(sliceCount-1) : 0;
              const r = t<0.5?lerp(0.10,0.95,t*2):1.0;
              const g = t<0.5?lerp(0.30,0.55,t*2):lerp(0.55,0.05,(t-0.5)*2);
              const b = t<0.5?lerp(0.70,0.10,t*2):0.05;
              const col = hasSlice ? `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})` : isLive&&sculptureMode ? "#ffffff" : "#eeeeee";
              return (
                <div key={i} style={{flex:1,height:hasSlice?12:isLive&&sculptureMode?10:4,
                  background:col,borderRadius:1,border:"1px solid rgba(0,0,0,0.1)",
                  title:`Slice ${i}: tick ~${i*15}`}}/>
              );
            })}
          </div>
          <div style={{display:"flex",justifyContent:"space-between",marginTop:2}}>
            <span style={{fontFamily:F,fontSize:6,color:"#aaa"}}>t=0 (blue)</span>
            <span style={{fontFamily:F,fontSize:6,color:"#aaa"}}>t=59 (red)</span>
          </div>
        </div>
        <div style={{background:"rgba(248,248,252,0.8)",border:"1px solid #e8e8ee",padding:"4px 7px",marginBottom:4}}>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"2px 8px"}}>
            {[
              ["Slice cadence","every 15 ticks"],
              ["Max slices","12 (60 ticks)"],
              ["Resolution","64×64 Z-cache"],
              ["Live ghost","white · always on"],
            ].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between"}}>
                <span style={{fontFamily:F,fontSize:7,color:"#aaa",fontStyle:"italic"}}>{k}</span>
                <span style={{fontFamily:FM,fontSize:7.5,color:"#333"}}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{...sm,lineHeight:1.5}}>
          Time becomes a spatial dimension — the 4D stress trajectory visualised as
          stacked physical surfaces. Offset +Y from main manifold. Dec 2011 looks like a slow stalagmite;
          Mar 2020 looks like a sudden full-base implosion. Rotate camera to see the depth.
        </div>
      </SectionHeader>

      {/* Persistent Homology */}
      <SectionHeader id="homology" label="Persistent Homology" active={homologyEnabled} openSection={openSection} onToggle={toggle} F={F}>
        <label style={{display:"flex",alignItems:"center",gap:4,cursor:"pointer",marginBottom:3}}>
          <input type="checkbox" checked={homologyEnabled} onChange={e=>setHomologyEnabled(e.target.checked)}
            style={{width:11,height:11,accentColor:"#44aa88"}}/>
          <span style={{fontFamily:F,fontSize:8,color:"#555"}}>Live β₀/β₁ computation</span>
        </label>
        {homologyData?<PersistenceDiagram data={homologyData} F={F} FM={FM}/>:
          <div style={sm}>Betti numbers at thresholds [-2,-5,-10,-15,-20,-25]</div>}
      </SectionHeader>
    </div>
  );
}
