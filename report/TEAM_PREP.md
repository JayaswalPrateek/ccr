# CCR Engine — Complete Team Prep Guide

> **Who this is for:** Team members who haven't been in the weeds of this project. Read this top-to-bottom and you will know the system well enough to present it, demo it, and field technical questions confidently.

---

## Table of Contents

1. [What We Built and Why](#1-what-we-built-and-why)
2. [Finance Foundations — The Vocabulary](#2-finance-foundations--the-vocabulary)
3. [The Three Core Risk Numbers](#3-the-three-core-risk-numbers)
4. [The Regulatory Context](#4-the-regulatory-context)
5. [The Monte Carlo Engine — How Risk is Computed](#5-the-monte-carlo-engine--how-risk-is-computed)
6. [The Four-Layer Architecture](#6-the-four-layer-architecture)
7. [Tech Stack Top to Bottom](#7-tech-stack-top-to-bottom)
8. [The Six Demo Counterparties](#8-the-six-demo-counterparties)
9. [The Web Interface — Every Page](#9-the-web-interface--every-page)
10. [User Roles and Access Control](#10-user-roles-and-access-control)
11. [Data Flow End-to-End](#11-data-flow-end-to-end)
12. [Database Schema](#12-database-schema)
13. [The Research Gaps We Identified](#13-the-research-gaps-we-identified)
14. [Common Questions and Sharp Answers](#14-common-questions-and-sharp-answers)

---

## 1. What We Built and Why

### The Problem

Before 2008, banks measured counterparty risk by looking at the face value of contracts — a blunt, notional-based number that told you nothing about what exposure would look like in the future. The 2008 financial crisis exposed how catastrophic this was: two-thirds of counterparty credit risk losses came not from defaults themselves but from mark-to-market losses that accumulated faster than any bank could detect.

The regulatory response was Basel III and FRTB (Fundamental Review of the Trading Book), which mandated that banks compute *dynamic, stochastic* exposure measures — Potential Future Exposure (PFE), Expected Positive Exposure (EPE), and Credit Valuation Adjustment (CVA) — in near real-time. Static overnight batch processes became structurally non-compliant.

A real production engine for a G-SIB (globally systemically important bank) must run O(N × M × T) computations — N instruments, M simulation paths, T timesteps — across thousands of paths and potentially hundreds of timesteps per instrument. A single L3 cache miss costs 200-300 CPU cycles. This is not a software problem, it is a hardware-aware systems problem.

### What We Built

A complete, three-tier counterparty credit risk and margin engine:

- **C++20 Monte Carlo engine** — the number-cruncher. Runs GBM path simulation with SIMD vectorisation (AVX-512/AVX2/NEON), xoroshiro128aox PRNG, Cholesky-correlated hazard rates, Kahan-summation CVA, and jump-at-default wrong-way risk.
- **Python/FastAPI server** — the brain. Wraps the C++ engine via pybind11, handles authentication, CRUD, scheduling, market data, PDF/CSV export, email alerts, and WebSocket streaming.
- **SvelteKit TypeScript dashboard** — the face. A dark financial terminal UI with real-time charts, stress testing controls, margin call management, and regulatory report generation.
- **PostgreSQL + TimescaleDB** — the memory. Stores all counterparty data, simulation results, audit events, and price history as time-series.

The specific engineering angle of this project: we identify and address four layers of optimisation that prior literature attacks in isolation (modelling, algorithmic, numerical, hardware) — and we argue that their interactions are multiplicative, not additive.

---

## 2. Finance Foundations — The Vocabulary

Before anything else, these terms need to be second nature.

### Derivative

A financial contract whose value *derives* from an underlying asset. Examples in our system:
- **IRS (Interest Rate Swap)**: one party pays a fixed rate, the other pays floating (e.g., SOFR). Used to hedge interest rate exposure.
- **CDS (Credit Default Swap)**: protection against default. Buyer pays a spread; seller pays face value if the reference entity defaults.
- **FX Forward**: agreement to exchange currencies at a fixed rate on a future date.
- **Equity Swap / TRS (Total Return Swap)**: exchange of total return of an equity index for a fixed/floating rate.
- **Commodity Swap**: exchange of fixed price for floating commodity price (e.g., oil at $75/bbl vs. spot).

These are OTC (Over-The-Counter) — bilaterally negotiated, not exchange-traded. This means credit risk matters: if your counterparty defaults mid-trade, you're exposed.

### Counterparty

The other party in a bilateral derivative contract. If you enter a 10-year IRS with Alpha Bank, Alpha Bank is your counterparty.

### Exposure

How much money you'd lose if the counterparty defaulted *right now*. Specifically: if the contract is in-the-money for you (you would receive money), and your counterparty defaults, you lose that in-the-money amount. If the contract is out-of-the-money (you owe them money), you lose nothing from credit risk — you still have to pay, but there's no credit loss. Hence:

```
Exposure = max(MtM, 0)
```

where MtM is the mark-to-market value of the contract from your perspective.

### Hazard Rate (λ)

The instantaneous probability of default per unit time. If λ = 0.02 (2% per year), the survival probability to time T is exp(−λT). A counterparty with λ = 0.085 (like Zeta Corp in our demo) has an 8.5% annual default probability — very distressed.

### Recovery Rate (R)

The fraction of the exposure you recover after a default. If exposure is $1M and R = 0.40, you recover $400K and lose $600K. Recovery rates for investment-grade firms typically 40-45%; distressed corporates 15-20%.

### Collateral / CSA

The Credit Support Annex (CSA) governs collateral posting between counterparties. When exposure exceeds the threshold, the exposed party can demand collateral — a margin call. Collateral reduces net exposure.

### MPoR — Margin Period of Risk

The most important operational risk concept in the system. When a counterparty defaults, you cannot immediately close out the position — it takes 10-30 days of legal and operational process. During that window, **no collateral is exchanged** even though exposure is accumulating. The MPoR (typically 10-20 days) is the window of uncollateralised exposure after a default event. This generates exposure *spikes* that collateral agreements cannot mitigate.

### Wrong-Way Risk (WWR)

The dangerous scenario where exposure magnitude and default probability are positively correlated — you are most exposed precisely when the counterparty is most likely to default. Example: you sold CDS protection to a bank on its own bonds. As the bank deteriorates, the CDS protection becomes more valuable (your exposure increases) while the bank's default probability increases simultaneously. That's wrong-way risk.

Two channels in our engine:
1. **Linear correlation (ρ)**: the Brownian driver of the credit asset and the exposure path are correlated via Cholesky decomposition.
2. **Jump-at-Default**: an instantaneous multiplicative shock applied to the exposure when default is simulated. At jump size J=1%, CVA doubles. At J=5%, CVA increases 9x. This is the dominant WWR channel for collateralised FX derivatives.

### Netting

Within a single counterparty portfolio, derivative gains and losses offset each other. If you have a IRS with PV +$1M and a CDS with PV −$400K, your net exposure is $600K, not $1M. Netting is applied at the portfolio level in our engine.

---

## 3. The Three Core Risk Numbers

### PFE — Potential Future Exposure

**Definition:** The α-quantile (99th percentile in our system) of the exposure distribution at each future date.

**Intuition:** "In the worst 1% of market scenarios, how large is our credit exposure at each future date?"

**Formula:** At each timestep t, sort the exposures across all M simulation paths. PFE(t) = the exposure at the 99th percentile of that cross-sectional distribution.

**Use:** Setting credit limits. A bank will say: "We will not permit exposure to counterparty X to exceed $50M PFE." If a new trade would push PFE over the limit, the trade is rejected.

**Chart:** PFE over time typically has a hump shape — it rises as the contract accumulates value, then falls as it approaches maturity (shorter time to cash-flow, less uncertainty). An IRS PFE peaks around the 3-5 year mark for a 10-year trade.

### EPE — Expected Positive Exposure

**Definition:** The mean of positive exposures across all simulation paths at each time point.

**Formula:** `EPE(t) = E[max(V(t), 0)] = (1/M) × Σ max(V_i(t), 0)` over all M paths.

**Intuition:** "On average, across all market scenarios, what is our positive exposure?" It's lower than PFE because it's an average, not a 99th percentile.

**Use:** EPE is the input to CVA calculation.

### CVA — Credit Valuation Adjustment

**Definition:** The fair-value cost of counterparty credit risk. The present value of expected losses from counterparty default.

**Formula:**
```
CVA = (1 − R) × Σᵢ [ EPE(tᵢ) × (PD(tᵢ₋₁) − PD(tᵢ)) × df(tᵢ) ]
```
where:
- R = recovery rate
- PD(t) = survival probability = exp(−λt) using the counterparty's hazard rate
- PD(tᵢ₋₁) − PD(tᵢ) = marginal default probability in the interval [tᵢ₋₁, tᵢ]
- df(tᵢ) = risk-free discount factor at time tᵢ
- The sum runs over all T timesteps

**Intuition:** CVA is what you should charge a counterparty upfront (or mark down from the trade's value) to compensate for the possibility they default. If CVA = $85K on a trade, that trade is worth $85K less than its risk-free value.

**Implementation detail:** We use **Kahan compensated summation** for this integral. Over 252 timesteps, naive floating-point accumulation introduces rounding error. Kahan summation maintains a running compensation term so the accumulated error stays bounded at machine epsilon — required for regulatory-grade reproducibility.

---

## 4. The Regulatory Context

### Basel III and SA-CCR

The post-2008 Basel III framework replaced the old Current Exposure Method (CEM) with the **Standardised Approach for Counterparty Credit Risk (SA-CCR)**.

SA-CCR computes the regulatory capital charge via:
```
EAD = 1.4 × (RC + AddOn_aggregate)
```
where:
- **EAD** = Exposure At Default (the capital denominator)
- **RC** = Replacement Cost — the current MtM value of the portfolio
- **AddOn** = a forward-looking add-on per asset class, computed using supervisory factors that encode historical volatility by asset class and maturity bucket
- **1.4** = the alpha multiplier — a regulatory conservatism buffer

**Supervisory Factors (SF)** are fixed by the regulator per asset class:
- Interest rate derivatives: 0.5% (low vol)
- FX derivatives: 4% (medium vol)
- Equity derivatives: 32% (high vol)
- Commodity derivatives: 18% (medium-high)
- Credit derivatives: 5% (medium)

**Maturity Factor (MF)** adjusts for time horizon and the MPoR:
- For collateralised trades: MF = 1.5 × √(MPoR/252)
- For uncollateralised trades: MF = √(min(M, 1))

Our SA-CCR endpoint (`/api/v1/simulate/{run_id}/sa-ccr`) computes this for any simulation run and returns EAD, RC, AddOn, and a per-derivative breakdown.

**Critical limitation of SA-CCR:** Supervisory factors are standardised, not model-specific. In the Archegos Capital Management collapse (March 2021), SA-CCR-mandated initial margins of 7.5% proved catastrophically insufficient against actual unwinding losses of 27-50% of notional. This is because SA-CCR cannot capture concentration risk, illiquid market impact, or wrong-way risk. SA-CCR is a regulatory floor. Our internal Monte Carlo model exceeds that floor by computing actual path-dependent WWR.

### FRTB — Fundamental Review of the Trading Book

FRTB introduced the **Profit-and-Loss Attribution (PLA) test**: a bank's internal risk model P&L must correlate at ρ > 0.8 with its front-office pricing system across 250 trading days. If it fails, the bank is forced from the Internal Model Approach (IMA) to the standardised approach — which carries higher capital requirements.

This creates a constraint on any approximation scheme: you can speed up computation but you cannot introduce bias that decorrelates P&L attribution.

---

## 5. The Monte Carlo Engine — How Risk is Computed

This is the heart of the system. Understanding this is the most important thing.

### The Big Picture

We simulate 10,000 possible futures (paths) for asset prices over a 1-year horizon with daily timesteps (252 steps). At each timestep, on each path, we compute the portfolio's mark-to-market value. The cross-sectional distribution of positive MTMs at each timestep gives us PFE (the 99th percentile) and EPE (the mean). We then integrate EPE against the default probability curve to get CVA.

### Step 1 — Path Simulation via GBM

Geometric Brownian Motion is the standard model for equity/FX prices. The exact discrete update is:

```
S(t+dt) = S(t) × exp( (μ − σ²/2)×dt + σ×√dt×Z )
```

where:
- μ = drift (risk-neutral, typically close to the risk-free rate)
- σ = volatility (annualised, e.g., 0.20 = 20%)
- dt = timestep = 1/252 (one trading day)
- Z ~ N(0,1) = standard normal random draw

This ensures prices remain positive (log-normal) and the log-returns are normally distributed — consistent with the Black-Scholes-Merton framework.

### Step 2 — Random Number Generation (xoroshiro128aox)

The normal variate Z is generated using:
1. **xoroshiro128aox** — a pseudorandom number generator (PRNG) based on an F₂-linear recurrence with an AND-OR-XOR (AOX) output scrambler. It produces 64 bits/cycle, passes the BigCrush and PractRand statistical test suites across all bit permutations, and supports 2⁶⁴ non-overlapping parallel streams via a pre-computed jump function.
   - Why not plain xoroshiro128+? The `+` variant fails the MatrixRank and LinearComp tests due to lower-bit weaknesses. The AOX scrambler fixes this while using only 684 logic cells.
   - Each simulation path gets its own independent PRNG stream — genuinely decorrelated paths.

2. **Inverse CDF transform** — the uniform output is converted to N(0,1) using Wichura's AS241 rational approximation (maximum absolute error < 10⁻¹⁵). This is the most accurate closed-form approximation to the normal quantile function.

### Step 3 — Correlated Assets (Cholesky)

When a portfolio has multiple correlated assets (e.g., EUR/USD and crude oil), we need correlated normal shocks. We use Banachiewicz Cholesky decomposition of the correlation matrix. If L is the Cholesky factor (L × Lᵀ = Σ), then correlated shocks are:

```
Z_correlated = L × Z_independent
```

This is also how WWR is implemented: the credit asset (hazard rate driver) and the market asset (exposure driver) are correlated at coefficient ρ.

### Step 4 — Exposure Calculation

At each timestep t on each path i:
```
Exposure_i(t) = max( Portfolio_MtM_i(t), 0 )
```

The floor at zero is critical: if the portfolio is out-of-the-money for you (you owe them), your credit exposure is zero. You still have to pay them, but there's no credit *loss* — they won't default on a contract they're winning.

### Step 5 — PFE and EPE Extraction

After all paths complete:
- **PFE(t):** Use `std::nth_element` (O(N) partial sort) to find the 99th percentile of the exposure distribution at time t. This is much faster than a full sort.
- **EPE(t):** Mean of all exposure values at time t (including zeros).

### Step 6 — CVA Integration (Kahan)

Sum EPE(tᵢ) × marginal_PD(tᵢ) × (1−R) × discount_factor(tᵢ) over all timesteps, using Kahan summation to prevent floating-point drift.

### Step 7 — Jump-at-Default (Wrong-Way Risk)

When jump-at-default is enabled, we simulate a default time for the counterparty using the hazard rate. On each path where default occurs within the horizon:
- The exposure at the default time is multiplied by (1 + J) where J is the jump amplitude
- This models the gap risk: at the moment of default, a distressed counterparty's FX position may spike (they were selling their home currency to survive)

The math from Salonen [2021]: at J=1%, CVA doubles. At J=5%, CVA is 9× higher. At J=10%, CVA is 18× higher.

### Step 8 — Margin Calculation

```
Margin = max(PFE_peak − Collateral_posted, 0) × (1 + MPOR_days / 360)
```

The MPOR adjustment accounts for the uncollateralised period after default.

### Stress Scenarios

The stress test runs the *same* pipeline twice: once with base parameters, once with shocked parameters (e.g., σ × 1.5, λ × 2, equity spot × 0.7). The delta between base and stressed CVA/PFE/margin is the stress impact.

---

## 6. The Four-Layer Architecture

This is the core academic contribution. We taxonomise the CCR optimisation problem into four layers whose interactions are **multiplicative**, not additive. You can't just optimise one layer — you need all four, and they interact.

### Layer 1 — Modelling

**What it addresses:** Scenario realism and wrong-way risk.

- GBM + Jump Diffusion for asset prices
- Cholesky-correlated hazard rate factor model for WWR: `ε_hazard = ρ × ε_asset + √(1−ρ²) × ε_random`
- Term-structure of hazard rates (hz_1y, hz_3y, hz_5y, hz_10y) for non-flat credit curves
- MPoR-aware daily time grid

**Speedup:** 26× from parsimonious time grids (Silotto et al. [2024]).

### Layer 2 — Algorithmic

**What it addresses:** O(N×M×T) valuation bottleneck.

- **Chebyshev polynomial surrogates** replace full derivative revaluation. Instead of re-pricing a complex derivative at every path×timestep, fit a polynomial approximation once and evaluate it (O(N) in polynomial degree, typically 8-16 terms). Average 87× speedup (Demeterfi et al.).
- Domain splitting at jump-at-default discontinuities: separate Chebyshev approximants for pre-jump and post-jump domains (otherwise Chebyshev convergence degrades from exponential to algebraic at discontinuities).

### Layer 3 — Numerical

**What it addresses:** Gaussian variate generation throughput.

- xoroshiro128aox: 64 bits/cycle, BigCrush-validated
- Approximate inverse CDF (Giles & Sheridan-Methven): 7× faster than Intel MKL for Gaussian variates
- Kahan summation: eliminates floating-point cancellation in CVA integration

### Layer 4 — Hardware

**What it addresses:** CPU instruction throughput and memory bandwidth.

- **Policy-based SIMD dispatch**: `SimdOps<Arch>` template parameterised at compile time. AVX-512 (8 doubles/cycle), AVX2 (4 doubles/cycle), ARM NEON (2 doubles/cycle), scalar. Zero runtime branching — the compiler generates a separate binary for each SIMD width.
- **Structure-of-Arrays (SoA) memory layout**: all spot prices across paths are stored contiguously (one column per asset, one row per path). This enables SIMD loads of 4/8 consecutive prices with a single instruction.
- **Thread-local RNG state**: each thread maintains its own PRNG state — no locks, no false sharing.
- **Single contiguous arena**: all simulation state is allocated in one aligned memory block before the hot loop. Zero heap allocations inside the Monte Carlo path computation.

**The critical insight:** Chebyshev approximation (Layer 2) shifts the bottleneck from compute-bound to memory-bound. Memory-bound workloads benefit most from cache-oblivious data structures (PMA, SoA layout). So hardware optimisation only fully pays off when algorithmic optimisation is applied first.

---

## 7. Tech Stack Top to Bottom

### C++20 Engine

| Component | File | What it does |
|-----------|------|--------------|
| `CcrEngine` | `ccr_engine.hpp` | Single entry point `run(config, callback) → RiskMetrics` |
| `TimeGrid` | `time_grid.hpp` | Constructs daily/parsimonious/uniform time grids |
| `RngEngine` | `rng_engine.hpp` | xoroshiro128aox PRNG with jump states for parallel streams |
| `PathSimulator` | `path_simulator.hpp` | GBM hot loop — SoA, branch-free, SIMD-templated |
| `CorrelationEngine` | `correlation_engine.hpp` | Cholesky decomposition for WWR correlation |
| `ExposureEngine` | `exposure_engine.hpp` | Computes max(V,0) per derivative per path×timestep |
| `QuantileExtractor` | `quantile_extractor.hpp` | std::nth_element-based PFE at any α-quantile |
| `CvaIntegrator` | `cva_integrator.hpp` | Kahan summation CVA with hazard term structure |
| `JumpDiffusion` | `jump_diffusion.hpp` | Jump-at-default shock application |
| `SimdAbstraction` | `simd_abstraction.hpp` | `SimdOps<Arch>` policy class — add/mul/load/store/sqrt |
| `types.hpp` | — | `SimParams`, `RiskMetrics`, `CcrResult`, `CounterpartyConfig`, `StressScenario` |

**Build system:** CMake 3.20+ with auto-detected SIMD target via CPU flags. `./scripts/build_engine.sh --bindings` produces a `_ccr_engine.so` Python extension module.

### Python/FastAPI Server

| Component | What it does |
|-----------|--------------|
| `server/main.py` | ASGI app entry, router registration, startup/shutdown hooks |
| `server/api/` | REST + WebSocket route handlers |
| `server/auth/` | JWT HS256 issuing/validation, RBAC decorators (`require_role`) |
| `server/bindings/` | `engine_runner.py` — converts Python dicts to C++ structs, calls `_ccr_engine.run()` with GIL released |
| `server/core/database.py` | asyncpg connection pool via SQLAlchemy async |
| `server/core/scheduler.py` | APScheduler: market refresh every 15 min, auto-sim every hour |
| `server/market_data/` | yfinance fetcher (equity/FX/commodity), FRED client (SOFR/Treasury) |
| `server/models/` | SQLAlchemy ORM models + Pydantic request/response schemas |
| `server/notifications/` | aiosmtplib email alerts, audit log writer |
| `server/reports/exporter.py` | ReportLab PDF generation, CSV serialisation |

**Key design decision:** The C++ engine runs in a `ThreadPoolExecutor` — the GIL is released during `_ccr_engine.run()`, so the Python event loop is free to handle other requests during simulation. Progress updates stream via WebSocket.

**Authentication:** JWT HS256, 8-hour expiry. Every endpoint checks role via a `Depends(require_role(Role.X))` FastAPI dependency. Role hierarchy: ADMIN > RISK_MANAGER > AUDITOR.

### SvelteKit Frontend

| Page | Route | Key Components |
|------|-------|---------------|
| Dashboard | `/dashboard` | PFEChart, EPEChart, MetricCard, concentration table |
| Counterparties | `/counterparties` | CP list, inline create form, sparkline SVGs |
| Counterparty Detail | `/counterparties/[id]` | SurvivalCurveChart, BacktestChart, portfolio accordion |
| Simulate | `/simulate` | SimParamsForm, PFEChart, EPEChart, AttributionChart, SA-CCR card |
| Stress Test | `/stress` | StressScenarioForm (7 sliders), side-by-side comparison |
| Margin Calls | `/margin-calls` | Status tabs, bulk acknowledge, email notify |
| Query Builder | `/query` | 5 query templates, line/bar/scatter charts |
| Reports | `/reports` | Run history, run comparison, PDF/CSV export |
| Presets | `/presets` | Saved parameter sets, import/export |
| Admin | `/admin` | User management, audit log table |

**State management:** Svelte stores (`$lib/state.ts`) — `authToken`, `latestMetrics`, `simRunning`, `simProgress`. WebSocket client in `$lib/ws-client.ts`.

**Charting:** Chart.js with `maintainAspectRatio: false` in bounded wrapper divs. Chart types: line (PFE/EPE), bar (CVA attribution, concentration), scatter (Vol vs CVA).

### Database (PostgreSQL 16 + TimescaleDB)

TimescaleDB adds three things:
1. **Hypertables** — automatically partition time-series tables by time chunk (7-day chunks). Queries with `WHERE time > ...` only scan relevant chunks.
2. **Continuous aggregates** — materialised views that update incrementally.
3. **Compression** — columnar compression for old data.

Three of our tables are hypertables: `risk_metrics`, `audit_log`, `price_history`.

---

## 8. The Six Demo Counterparties

The demo book is seeded with realistic diversity across credit quality, asset class, and collateralisation.

| Name | Rating | λ (hazard) | Recovery | Collateral | MPOR | Key trades |
|------|--------|-----------|---------|-----------|------|------------|
| **Alpha Bank S.A.** | AA | 0.4% | 45% | $2M | 10d | 10Y EUR IRS ($25M) + 5Y CDS ($15M) |
| **Beta Capital LLC** | BB | 1.8% | 35% | $500K | 5d | 3Y EUR/USD FX fwd ($8M) + 2Y SPY equity TRS ($5M) |
| **Gamma Hedge Fund** | B | 3.5% | 20% | $0 (uncollateralised) | 20d | 5Y IRS ($30M) + equity variance swap + CDS HY |
| **Delta Energy Corp** | BBB | 0.8% | 40% | $1M | 7d | 2Y WTI commodity swap ($12M) + 5Y IRS ($20M) |
| **Epsilon Insurance** | AAA | 0.2% | 50% | $5M | 10d | 20Y GBP IRS ($50M) + 10Y EUR IRS ($30M) |
| **Zeta Corp** | CCC | 8.5% | 15% | $0 | 30d | 1Y CDS protection ($5M) + 6M FX fwd ($3M) |

**Notable for the demo:**
- **Gamma and Zeta** are uncollateralised — they will generate margin calls when simulations run.
- **Epsilon** has the highest notional ($80M combined) with the lowest hazard rate — a long-dated rates book typical of an insurer liability-matching portfolio.
- **Zeta** is distressed CCC with 8.5% hazard rate and 30-day MPoR — the worst case. Running a stress scenario on Zeta with hazard rate shock shows dramatic CVA spikes.
- Alpha Bank has a hazard term structure configured (hz_1y, hz_3y, hz_5y, hz_10y) — you can see the survival curve chart with two lines on its detail page.

---

## 9. The Web Interface — Every Page

### Login (`/login`)

Standard username/password form. Backend returns a JWT stored in the Svelte `authToken` store (localStorage). All subsequent API calls include `Authorization: Bearer <token>`. Expired or missing tokens redirect back to login. **Demo credentials:** admin/admin123, risk/risk123, auditor/auditor123.

### Dashboard (`/dashboard`)

The hub. Three columns:
- **Left:** KPI metric cards (CVA, WWR-CVA, Margin Required, Compute Time) + PFE chart + EPE chart. These show the most recent simulation run from the global history.
- **Centre:** CVA bar chart by counterparty (concentration view) + Top Risk card.
- **Right:** Recent simulations table with note column + recent margin calls with status badges + recent audit activity feed.

Actions available: "Auto-Run" button (re-runs most recent simulation with last known parameters), theme toggle (◑), settings (⚙, opens Alert Thresholds modal).

### Counterparties (`/counterparties`)

Master list of all six counterparties. Per row: name, credit rating badge (AA green, BB amber, B/CCC red), hazard rate, collateral, CVA sparkline (mini SVG line chart of CVA history), and "Simulate" button (navigates to `/simulate?cp_id=...` pre-filled).

"+ New Counterparty" expands an inline create form. Auditors cannot see this or the delete button.

### Counterparty Detail (`/counterparties/[id]`)

The richest page in the system. Sections:
1. **Info cards** — hazard rate, recovery rate, collateral, MPOR
2. **Summary stats** — total runs, avg CVA, latest CVA, total margin called, pending calls
3. **Edit form** — expandable, includes the hz_1y/hz_3y/hz_5y/hz_10y term structure fields
4. **Survival Curve chart** — appears when hz fields are set. Two lines: term structure (blue) + flat baseline from the single hazard_rate (amber dashed)
5. **Portfolio accordion** — each portfolio expands to show its derivatives table
6. **Add Portfolio form** — inline, auto-generates external ID as `{CP_ID}-PORT-N`
7. **Backtest chart** — plots the PFE profile from the most recent simulation against 90 days of realised exposures from price_history. Shows breach count and coverage %.
8. **Simulation History** — all past runs for this counterparty
9. **Margin Call History** — all margin calls for this counterparty

### Simulate (`/simulate`)

The main workspace. Left panel: SimParamsForm with sections for:
- Counterparty selection (or pre-filled from `?cp_id=`)
- Portfolio + derivative builder
- Simulation parameters: num_paths (default 10K), num_timesteps (default 50), mu, sigma, rho_wwr, horizon
- Simulation mode: REGULATORY, STANDARD, APPROX_FAST
- Options: enable WWR, enable Jump Diffusion, enable Collateral
- Stress scenario toggle

"Load µ/σ from Market" button fetches current volatility from the market data cache and fills the form.

Right panel (post-run):
- Metric cards: CVA, WWR-CVA, Margin Required, Compute Time
- PFE chart (with spike annotation if Jump Diffusion was used)
- EPE chart
- CVA Attribution chart (bar chart per derivative, proportional to contribution)
- SA-CCR card (loads async after run): EAD, RC, AddOn. "Show breakdown" reveals per-derivative SF × MF × notional table
- Netting Benefit section: Gross CVA (sum of per-derivative CVAs), Net CVA (portfolio CVA after netting), benefit % and dollar amount
- Suggested Collateral card: margin_required × 1.10 buffer
- Export PDF / Export CSV buttons
- "Save as Preset" opens a modal

**Keyboard shortcut:** Ctrl+Enter triggers form submission.

URL parameters: `?cp_id=UUID` pre-fills counterparty, `?preset_id=UUID` loads a saved preset, `?rerun_id=UUID` shows "Re-run ·" label and replicates previous params.

### Stress Test (`/stress`)

Two-column layout. Left: StressScenarioForm (7 sliders: vol shock, FX shock, equity shock, rate shock, credit spread shock, hazard rate shock, jump amplitude) + SimParamsForm. Right: results after run.

Results panel shows six metric cards (CVA base, CVA stressed, CVA Δ%, Margin base, Margin stressed, WWR-CVA base). PFE and EPE charts overlay base (solid) vs stressed (dashed) lines. Profile Comparison table shows T / PFE Base / PFE Stressed / PFE Δ% / EPE Base / EPE Stressed / EPE Δ% rows.

"Apply Stress" stores the scenario; "Reset" zeroes all sliders. Running without applying a scenario runs a base (unstressed) simulation.

### Margin Calls (`/margin-calls`)

Shows all margin breach events. Status summary cards at top (PENDING count, ACKNOWLEDGED count, SETTLED count). CVA Exposure Trend chart. Filter tabs: All / PENDING / ACKNOWLEDGED / SETTLED.

Per row: counterparty, amount, status badge, created date, reason (truncated with hover tooltip for full text). Actions: Acknowledge (PENDING → ACKNOWLEDGED), Settle (ACKNOWLEDGED → SETTLED), Notify (sends email + returns `{sent: true, to: [...]}`). Checkbox for bulk operations. Export CSV.

Auditors see no action buttons.

### Query Builder (`/query`)

Five pre-built analytics queries over the `risk_metrics` hypertable:
1. **Risk Timeline** — CVA over time, line chart
2. **Exposure Ranking** — counterparties by peak CVA, bar chart
3. **PFE Peaks** — peak PFE per run, bar chart
4. **Margin Activity** — margin calls status breakdown, includes summary stats
5. **Vol vs CVA** — scatter plot of sigma vs CVA to show model sensitivity

Each template has date range and limit filters. Results show in a table with Download CSV. Bookmark button saves the query state.

### Reports (`/reports`)

Paginated list of all simulation runs. Select a run → Selected Run card shows CVA/WWR-CVA/Margin + Download PDF / Download CSV / Re-run. Compare mode: select two runs via checkbox → Compare button → delta table with CVA change row.

### Presets (`/presets`)

Saved parameter sets. My Presets section + Recently Used section. Each preset card shows name, description, shared indicator (blue left border + ⇄ Shared badge), use count, last used. Actions: Run (→ `/simulate?preset_id=...`), Edit, Export (.ccr-preset.json), Delete. Import button for loading exported files.

When creating/editing: `is_shared` checkbox with explanation text: "All users (including Auditors) can view and run this preset. Only you (or ADMIN) can edit or delete it."

Invalid JSON in params_json → Save blocked with error.

### Admin (`/admin`)

ADMIN role only (others are redirected to dashboard). Two tabs:

**Users tab:** Table of all users (username, role, active, created date, last login). Role select dropdown to change role in-place. Active toggle. "+ New User" expands form. "Send Test Email" button → spinner → success toast → email arrives in Mailtrap.

**Audit Log tab:** Full filterable event log. Action badges (create_counterparty, run_simulation, login, etc.). "Refresh" reloads.

---

## 10. User Roles and Access Control

Three roles, enforced at both API and UI layers.

| Capability | ADMIN | RISK_MANAGER | AUDITOR |
|-----------|-------|--------------|---------|
| Login + read all data | ✓ | ✓ | ✓ |
| Run simulations | ✓ | ✓ | ✗ |
| Create/edit counterparties | ✓ | ✓ | ✗ |
| Acknowledge/settle margin calls | ✓ | ✓ | ✗ |
| Send email notifications | ✓ | ✓ | ✗ |
| Create/edit presets | ✓ | ✓ | ✗ |
| Refresh market data | ✓ | ✓ | ✗ |
| View audit log | ✓ | ✗ | ✓ |
| Manage users | ✓ | ✗ | ✗ |
| Create users | ✓ | ✗ | ✗ |
| Send test emails | ✓ | ✗ | ✗ |

**Implementation:** FastAPI `Depends(require_role(Role.X))` decorators check the JWT claim on every request. The frontend additionally hides/disables buttons based on the decoded role stored in the auth store.

**Preset ownership:** `is_shared=true` makes a preset visible to all users. However, only the owner or ADMIN can edit or delete it — enforced by `_get_owned_or_404()` in the presets API.

**Demo credentials:**
```
admin    / admin123   → ADMIN
risk     / risk123    → RISK_MANAGER
auditor  / auditor123 → AUDITOR
```

---

## 11. Data Flow End-to-End

### Simulation Request Flow (WebSocket path)

```
1. User clicks "Run Simulation" in the browser
   ↓
2. Browser sends auth token + simulation params over WebSocket /ws/simulate
   ↓
3. FastAPI validates JWT, checks role (RISK_MANAGER+)
   ↓
4. SimulationRequest is deserialised from JSON
   ↓
5. Python converts request to C++ SimParams struct + CounterpartyConfig + PortfolioConfig
   ↓
6. ThreadPoolExecutor.submit(engine_runner.run, config, progress_callback)
   → GIL released, FastAPI event loop stays live
   ↓
7. C++ CcrEngine::run() executes:
   a. Build time grid (T steps)
   b. Allocate SoA memory arena (M × T doubles)
   c. For each path i in parallel:
      - Seed thread-local xoroshiro128aox with path-specific jump state
      - For each timestep t:
        - Generate Z ~ N(0,1) via AS241
        - Apply Cholesky correlation (if WWR enabled)
        - Update spot: S(t+dt) = S(t) × exp(...)
        - Compute derivative MTM (Chebyshev surrogate or analytic)
        - Apply jump-at-default if τ ≤ t (if jump diffusion enabled)
      - Store exposure = max(portfolio_MtM, 0) at each t
   d. progress_callback fires after each batch of paths (→ WebSocket progress message)
   e. PFE: nth_element at 99th percentile for each t
   f. EPE: mean of positive exposures for each t
   g. CVA: Kahan summation of EPE × marginal_PD × (1−R) × df
   h. Margin: max(PFE_peak − collateral, 0) × MPOR_factor
   → Returns RiskMetrics
   ↓
8. If stress scenario set: run again with shocked parameters → returns stressed RiskMetrics
   ↓
9. Python persists to risk_metrics (TimescaleDB hypertable)
   ↓
10. Check: margin_required > 0? → insert into margin_calls, send email if SMTP configured
    ↓
11. Write audit log entry (user_id, action="run_simulation", resource_id=run_id, details=JSON)
    ↓
12. Send final result JSON over WebSocket → Browser updates charts + metric cards
```

### Market Data Flow

```
APScheduler (every 15 min)
→ yfinance.download(SPY, AAPL, MSFT, GS, JPM, EURUSD=X, ..., period=60d)
→ compute 30-day rolling volatility from log-returns
→ FRED API: fetch SOFR, DGS1, DGS5, DGS10
→ insert into price_history (TimescaleDB hypertable)
→ update in-process market cache (TTL 60s)

GET /api/v1/market/prices → returns from in-process cache (fastest path)
WS /ws/prices → GBM walk seeded from latest cached prices (synthetic tick stream)
```

---

## 12. Database Schema

### Tables Overview

```
users              — credentials, roles, active flags
counterparties     — credit master (name, rating, λ, R, collateral, MPOR, hz_1y..hz_10y)
portfolios         — trade book containers (one or more per counterparty)
derivatives        — individual trades (type, notional, maturity, underlying_price, strike)
simulation_runs    — metadata for each sim run (who ran it, when, what params)
risk_metrics       — hypertable: PFE/EPE/CVA arrays per run, indexed by time
margin_calls       — breach events (run_id, amount, status, acknowledged/settled timestamps)
sim_presets        — saved parameter sets (owner, shared flag, params_json)
audit_log          — hypertable: every action with user, IP, resource, timestamp (append-only)
price_history      — hypertable: market tick data (symbol, price, volatility, timestamp)
```

### Key Relationships

```
Counterparty 1 ──── * Portfolio
Portfolio    1 ──── * Derivative
Counterparty 1 ──── * SimulationRun
SimulationRun 1 ──── 1 RiskMetrics (via run_id)
RiskMetrics   0..1 ── * MarginCall
User          1 ──── * SimulationRun (triggered_by)
User          1 ──── * AuditLog (actor)
User          1 ──── * SimPreset (owner)
```

### Cascade Delete

Deleting a counterparty with `?cascade=true` deletes all portfolios → derivatives → simulation_runs → risk_metrics → margin_calls in a single transaction. Without the cascade flag, the delete is blocked if child records exist.

### TimescaleDB Hypertables

`risk_metrics`, `audit_log`, and `price_history` are partitioned by `time` column into 7-day chunks. Queries like `WHERE time > NOW() - INTERVAL '90 days'` only scan the relevant chunks — critical for the backtest query which reads 90 days of price_history.

### Migration History

Managed by Alembic. Current head: `005`. Run `uv run alembic current` to confirm.

---

## 13. The Research Gaps We Identified

This is the academic contribution — 7 gaps identified through cross-paper synthesis across 13 papers. Be ready to discuss these.

**Gap 1 — End-to-End Error Composition [Critical]**
Individual components have rigorous error bounds (Chebyshev approximation, inverse CDF Lᵖ error, Monte Carlo O(M⁻½) convergence) but no framework composes them into a guarantee on final PFE or CVA. This blocks EU prudent valuation AVA quantification. The triangle inequality gives a conservative bound but the tightness depends on whether the error sources interfere constructively or destructively.

**Gap 2 — Jump-at-Default in Surrogate Models [Critical]**
Chebyshev approximations require analytically smooth pricing functions. Jump-at-default introduces payoff discontinuities at the default time, degrading Chebyshev convergence from exponential to algebraic. Optimal node placement near the discontinuity is uncharacterised. We address this in our implementation via domain splitting (separate pre-jump and post-jump approximants).

**Gap 3 — PRNG Artefacts and Tail Risk [High]**
F₂-linear generators (including xoroshiro128aox) exhibit Hamming Weight Dependency artefacts after 1.8-11.4 TB of output. At 99.9th percentile PFE, even tiny lower-bit biases in the uniform output, when transformed through the inverse Gaussian CDF, could shift the tail distribution systematically. This interaction with Chebyshev approximation error at the same tail is unquantified.

**Gap 4 — PMA Density Bounds for Skewed Exposures [High]**
Packed Memory Array (PMA) variants have only been validated on graph workloads with power-law degree distributions. Counterparty exposure distributions are right-skewed and heavy-tailed due to optionality and collateral dynamics. Whether PMA density bounds [1/4, 3/4] remain efficient under financial exposure profiles is unvalidated.

**Gap 5 — AVX-512 and SMT Interaction [High]**
SMT provides 30% throughput gain for compute-bound workloads. AVX-512 theoretically enables 8× throughput per cycle. But on Intel Skylake-SP and Ice Lake-SP, SMT sibling cores share AVX-512 execution units — concurrent dispatch contends for the same physical port. Whether to maximise physical cores or logical cores is a 30-50% performance decision that the literature does not address.

**Gap 6 — PLA Test Compliance for Chebyshev [Medium]**
FRTB requires ρ > 0.8 correlation between front-office P&L and risk model P&L. Whether a low interpolation degree Chebyshev approximant can maintain this correlation across 250 trading days — while achieving 87× speedup — depends on instrument mix. This is unvalidated.

**Gap 7 — Collateral-Aware Approximation on Parsimonious Grids [Medium]**
Parsimonious time grids concentrate simulation mass around cash flow and collateral call dates. Near MPoR spike windows, the exposure profile has high gradients. Whether Chebyshev smoothness assumptions hold at these grid-induced high-gradient regions is uncharacterised — monthly primary grids may violate the analyticity conditions needed for exponential convergence.

---

## 14. Common Questions and Sharp Answers

**Q: Why C++ for the engine and not just Python with NumPy?**

NumPy is excellent but has two structural problems for this use case. First, it operates on the CPython interpreter — the GIL limits true parallelism (you can use multiprocessing but with IPC overhead). Second, NumPy's SIMD is implicit — you cannot explicitly dispatch to AVX-512 vs NEON vs scalar based on the host CPU at compile time with zero runtime overhead. Our C++ engine uses policy-based SIMD dispatch via a template parameter: the compiler generates separate machine code for each instruction set width. There is zero runtime branching in the hot loop.

**Q: Why xoroshiro128aox instead of the standard Mersenne Twister?**

Mersenne Twister (MT19937) has a 624-integer state and is not designed for parallel use — generating independent streams requires manual seed management or expensive splicing. xoroshiro128aox has 128-bit state, generates 64 bits/cycle (much faster than MT), supports 2⁶⁴ non-overlapping parallel streams via a precomputed jump function, and passes all BigCrush and PractRand tests. MT19937 is slower, statistically weaker on parallelism, and produces lower statistical quality in the lower bits.

**Q: What's the difference between PFE and VaR?**

Value at Risk (VaR) answers: "What is the maximum loss over a single horizon (e.g., 1 day or 10 days) at a given confidence level?" It's a single number for the portfolio today. PFE answers: "At each future date, across simulation paths, what is the worst-case exposure to *this specific counterparty* at the 99th percentile?" PFE is a time-series of credit exposure profiles, not a single P&L number. VaR is about market risk; PFE is about credit risk.

**Q: Why does the engine run GBM but also support CDS and IRS derivatives?**

GBM models the *underlying* risk factor (equity price, FX rate, interest rate level). Each derivative type uses that simulated risk factor to compute its mark-to-market:
- IRS: the simulated rate vs. the fixed rate determines the PV of the floating leg
- CDS: the simulated credit spread determines the present value of protection
- FX forward: the simulated FX rate vs. the strike determines the payoff
- Equity: the simulated equity price directly determines the swap payoff

GBM is the engine; each derivative type is an analytic formula applied to the GBM output.

**Q: How does wrong-way risk amplify CVA so dramatically?**

Without WWR, exposure and default probability are independent. CVA = (1−R) × ∫ EPE(t) × f_τ(t) dt where f_τ is the default time density. With jump-at-default, exposure at the moment of default is shocked: if the counterparty defaults at time t, the MtM jumps by factor (1+J). Since the CVA integral weights EPE by the default density, and exposure is highest precisely at the moment of default, the two peaks multiply. At J=10%, the exposure spike is 10× normal — and it occurs at exactly the moment when the integration weight is highest (the actual default event). That's why CVA increases 18×.

**Q: How does SA-CCR relate to the Monte Carlo simulation?**

They're parallel computations with the same goal but different methods. Monte Carlo gives you the *actual* exposure distribution based on simulated paths — this is the internal model. SA-CCR is a regulatory formula that gives you a standardised, conservative capital charge without running any simulation — it uses fixed supervisory factors (SF) per asset class and maturity, multiplied by notional and a maturity factor. The SA-CCR endpoint in our system computes it analytically from the trade book data after a simulation run. They will generally differ, and SA-CCR is typically more conservative — it's the *floor* the regulator requires you to hold capital against.

**Q: What is Kahan summation and why do you need it for CVA?**

Floating-point addition is not associative. When summing 252 terms in the CVA integral, the accumulated rounding error can be on the order of 252 × machine_epsilon ≈ 10⁻¹³. Kahan summation maintains a running "compensation" term that captures the lost precision at each step. The sum error stays at machine_epsilon (~10⁻¹⁶) regardless of how many terms. For regulatory compliance, two runs with the same inputs must produce bit-identical results. Without Kahan, path order or parallel reduction order would produce different floating-point results.

**Q: What is the Margin Period of Risk and why does it matter?**

When a counterparty defaults, you don't immediately close out the position. Legal and operational process takes 10-30 days. During that window, the CSA (Credit Support Annex) collateral agreement is effectively suspended — no collateral is exchanged even though exposure is accumulating and potentially spiking. The MPoR is the window of uncollateralised exposure. This is why collateral doesn't fully eliminate credit risk. Our margin calculation includes an MPoR adjustment factor: `max(PFE_peak − collateral, 0) × (1 + MPOR_days/360)`. The Salonen paper (one of our 13 references) shows that for collateralised FX derivatives, jump-at-default during the MPoR window is the *dominant* CVA amplification mechanism — much more than gradual correlation.

**Q: Why TimescaleDB instead of plain PostgreSQL for the time-series data?**

TimescaleDB partitions hypertables into time-ordered chunks (7-day chunks in our setup). A query like `WHERE time > NOW() - INTERVAL '90 days'` on `price_history` only scans the 13 most recent chunks rather than the entire table. For our backtest query — which reads 90 days of daily prices for multiple symbols — this is the difference between scanning 450 rows efficiently from hot cache vs. scanning a full unbounded heap. TimescaleDB also adds columnar compression for old chunks and continuous aggregates for precomputed rollups.

**Q: What happens to the engine when there are multiple assets / derivatives in a portfolio?**

Each derivative has its own underlying risk factor. Multiple derivatives in a portfolio are simulated with their own GBM paths (separate PRNG streams, independent by default). Their individual MTMs are summed to get the portfolio MTM at each timestep. Netting means the sum can be positive even if one derivative is negative — you only lose the net positive amount. If the Cholesky correlation matrix is provided, the underlying risk factors are correlated across assets (not independent streams).

**Q: Is the market data real?**

Partially. yfinance provides delayed (15-minute lag) prices for equity, FX, and commodity symbols. FRED provides daily risk-free rate data (SOFR, Treasury yields). Hazard rates are entered manually — there's no live CDS spread feed. The tick stream on the WebSocket (`/ws/prices`) is synthetic — it's a GBM walk seeded from the latest cached real prices, explicitly labelled "Demo Ticks — GBM simulation, not real market data" in the UI.

**Q: How does the system handle a counterparty with a term structure of hazard rates?**

Each counterparty can have hz_1y, hz_3y, hz_5y, hz_10y set in addition to a flat `hazard_rate`. If these are set, the CVA integrator uses piecewise linear interpolation between the term structure points to get λ(t) at each simulation timestep. The survival probability becomes exp(−∫₀ᵀ λ(t)dt) rather than exp(−λT). The Counterparty Detail page shows a Survival Curve chart with two lines: the term structure curve (blue) and the flat baseline (amber dashed) — so you can visually compare them.

---

## Quick Reference — Numbers to Know

| Parameter | Default | Meaning |
|-----------|---------|---------|
| num_paths | 10,000 | Monte Carlo paths |
| num_timesteps | 50 | Time steps (daily = 252 for 1Y) |
| confidence_level | 99% | PFE quantile |
| alpha (SA-CCR) | 1.4 | Regulatory multiplier on EAD |
| MPoR default | 10 days | Basel III minimum for OTC derivatives |
| SOFR (fallback) | 5% | Risk-free discount rate |
| CVA amplification (J=10%) | 18× | Jump-at-default impact |
| Chebyshev speedup | avg 87× | Over full derivative revaluation |
| PRNG throughput | 64 bits/cycle | xoroshiro128aox |
| SIMD width (NEON) | 2 doubles/cycle | ARM (demo machine) |
| SIMD width (AVX2) | 4 doubles/cycle | Intel/AMD mainstream |
| SIMD width (AVX-512) | 8 doubles/cycle | Intel Ice Lake+ |
| JWT expiry | 8 hours | Token lifetime |
| Market refresh | 15 min | yfinance + FRED |
| Auto-sim interval | 1 hour | Background scheduler |
| DB migration | 005 (head) | Alembic current |
| API base | localhost:8000 | FastAPI + uvicorn |
| Frontend | localhost:5173 | Vite dev server |

---

## Quick Reference — Key Formulas

```
GBM:      S(t+dt) = S(t) × exp( (μ − σ²/2)×dt + σ×√dt×Z )

Exposure: E_i(t) = max( ΣV_k(t), 0 )   [net portfolio MtM across K derivatives]

PFE(t):   99th percentile of { E_i(t) : i = 1..M }   [via nth_element]

EPE(t):   (1/M) × Σᵢ E_i(t)

Survival: P(τ > t) = exp(−λt)   [flat] or exp(−∫₀ᵗ λ(s)ds)   [term structure]

CVA:      (1−R) × Σᵢ [ EPE(tᵢ) × (e^{-λtᵢ₋₁} − e^{-λtᵢ}) × e^{-r×tᵢ} ]

Margin:   max( max_t(PFE(t)) − Collateral, 0 ) × (1 + MPoR/360)

SA-CCR:   EAD = 1.4 × (RC + AddOn)
          AddOn = SF × MF × Notional × δ   (per derivative)
          MF_collateralised = 1.5 × √(MPoR/252)
```

---

*Last updated: 2026-04-24. Server at localhost:8000, frontend at localhost:5173. Run `./scripts/run_dev.sh --skip-build` + `cd web && npm run dev`.*
