# Realtime CCR & Margin Engine — Complete Reference

> Merged document combining the Theoretical Framework & Idea Notes with the UML Diagrams & System Design.

---

## Table of Contents

1. [The Shift: Why We Simulate the Future](#1-the-shift-why-we-simulate-the-future)
2. [The Engine of Uncertainty: Geometric Brownian Motion (GBM)](#2-the-engine-of-uncertainty-geometric-brownian-motion-gbm)
3. [The Power of Many: Monte Carlo Simulations](#3-the-power-of-many-monte-carlo-simulations)
4. [Defining the "Worst Case": Potential Future Exposure (PFE)](#4-defining-the-worst-case-potential-future-exposure-pfe)
5. [Advanced Shadows: CVA and Wrong Way Risk (WWR)](#5-advanced-shadows-cva-and-wrong-way-risk-wwr)
6. [Theoretical Financial Framework — Detailed Notes](#6-theoretical-financial-framework--detailed-notes)
7. [Advanced Risk Concepts](#7-advanced-risk-concepts)
8. [Regulatory Context](#8-regulatory-context)
9. [System Architecture & Execution Flow](#9-system-architecture--execution-flow)
10. [User Dashboard Features](#10-user-dashboard-features)
11. [UML Diagrams & System Design](#11-uml-diagrams--system-design)
    - [11.1 System Architecture Diagram](#111-system-architecture-diagram)
    - [11.2 Class Diagram](#112-class-diagram)
    - [11.3 Data Flow Diagrams (DFD)](#113-data-flow-diagrams-dfd)
    - [11.4 Use Case Diagrams](#114-use-case-diagrams)
    - [11.5 Sequence Diagrams](#115-sequence-diagrams)
    - [11.6 Activity Diagrams](#116-activity-diagrams)
12. [Notes on Pending Design Decisions](#12-notes-on-pending-design-decisions)

---

# Part I — Theoretical Framework & Project Idea

## 1. The Shift: Why We Simulate the Future

For decades, the financial industry leaned on **notional value** — the static face value of a contract — to gauge risk. However, the 2008 crisis revealed that notional value is a blind metric. Modern risk management has shifted toward **risk-sensitive measures** that capture the probabilistic nature of market movements.

Under the **Basel III** and **Fundamental Review of the Trading Book (FRTB)** frameworks, the "overnight batch" approach is no longer sufficient. Regulators now demand intra-day monitoring and sophisticated capital requirements like the **Standardized Approach for Counterparty Credit Risk (SA-CCR)**. For the modern bank, real-time risk calculation is a regulatory and operational imperative.

### The Evolution of Risk Monitoring

| Feature | Legacy Methods (Notional/Static) | Modern Real-Time Risk (Stochastic/Dynamic) |
|---|---|---|
| Core Metric | Notional Value (Face value) | Statistical Distributions (Scenarios) |
| Frequency | Overnight Batch (24-hour lag) | Intra-day / Real-time (Continuous) |
| Data Type | Static / Historical | Stochastic (Probabilistic) |
| Regulatory Logic | Crude Capital Reserves | SA-CCR / FRTB Risk-Sensitive Capital |

> **Key Insight:** Real-time calculation allows banks to issue margin calls instantly when exposure thresholds are breached. If you cannot see your exposure as it happens, you cannot manage the liquidity required to back it — a realization that is now codified in the SA-CCR standards.

---

## 2. The Engine of Uncertainty: Geometric Brownian Motion (GBM)

To predict risk, we model the evolution of market factors using **Geometric Brownian Motion (GBM)**. This is the foundational "math of movement" for equity prices, treating price changes as a combination of predictable **Trend** and unpredictable **Noise**.

In a simulation engine, we don't just solve an equation; we implement a time-stepping algorithm. We discretize the continuous Stochastic Differential Equation (SDE) into small time steps (Δt), transforming abstract math into executable code.

### Stochastic Differential Equation (SDE)

- **Deterministic baseline:** `dx/dt = f(x)`
- **GBM SDE:** `dS_t = μ·S_t·dt + σ·S_t·dW_t`
  - `μ` = drift (average growth)
  - `σ·dW_t` = Brownian motion shock (noise)

### The Discretized GBM Formula

```
S_{t+Δt} = S_t · exp( (μ - σ²/2)·Δt + σ·√Δt·Z )
```

where `Z ~ N(0,1)` (standard normal random variable).

### The Three Core Inputs

| Parameter | Symbol | Meaning |
|---|---|---|
| Drift | μ | Expected return / average growth |
| Volatility | σ | Magnitude of the noise / price swings |
| Time Step | Δt | Discretization interval (e.g., daily, hourly) |

**GBM guarantees:** price stays positive, log distribution of growth factors is normal, smooth to simulate.

> **Key Insight:** The variable `Z` (standard normal random variable) is the "spark of randomness." In high-performance systems, generating these Z values efficiently is the primary bottleneck. By updating thousands of `S_t` paths simultaneously, we move from a single guess to a map of all possible futures.

---

## 3. The Power of Many: Monte Carlo Simulations

Monte Carlo simulation is a "strength in numbers" approach. By simulating thousands of potential paths, we "vote" on the future, allowing us to see the full range of potential outcomes — especially the **tail risks** that destroy banks.

### The 4-Step Simulation Workflow

1. **Scenario Generation:** Use discretized GBM to create thousands of paths (e.g., 5,000 simulations) over a 1-year horizon.
2. **Portfolio Valuation:** At every time step along every path, re-value every derivative in the portfolio.
3. **Exposure Calculation:** Apply the "logic of loss" to determine potential credit hits: `E = max(V, 0)`.
4. **Aggregation:** Combine the results into a distribution to find specific quantiles of risk.

Example paths (starting at S₀ = 100):
```
Path 1:    100 → 103 → 97  → 110 ...
Path 2:    100 → 95  → 92  → 98  ...
Path 5000: ...
```

> **Key Insight:** In a real-time environment, speed is limited by the **Processor-Memory Gap**. Traditional Object-Oriented Programming (OOP) leads to "pointer chasing" and cache misses. Architects use **Data-Oriented Design** (Structure of Arrays — SoA) to ensure cache locality. Using **AVX-512 SIMD Vectorization**, we can process multiple simulation paths in a single clock cycle — updating four or eight paths simultaneously at the hardware level.

---

## 4. Defining the "Worst Case": Potential Future Exposure (PFE)

The ultimate output of our simulation engine is **Potential Future Exposure (PFE)**. PFE is not an average; it is a **quantile** that defines the upper bound of potential loss.

**PFE is a 99th percentile quantile.** For example, a bank might run 5,000 paths over a 1-year horizon to find the level of loss that is only exceeded in 1% of scenarios. It tells the risk manager: *"We are 99% confident our loss will not exceed this number."*

### The Logic of Loss

```
E = max(V, 0)
```

- If portfolio value `V > 0`: the counterparty owes us → we face credit risk if they default.
- If `V < 0`: we owe them → our exposure is floored at zero.

### PFE Calculation Process

1. Simulate portfolio value under 5,000 scenarios at time `t` (Monte Carlo).
2. Take only positive values (we lose money only if the counterparty owes us).
3. Take the 99th percentile.

**Example exposure distribution at 1 year:** `0, 0, 0, 1M, 2M, 3M, 10M`
→ 99% PFE = **10M** → with 99% confidence, exposure won't exceed 10 million.

| Simulation No. | Portfolio Value |
|---|---|
| 1 | −2 million |
| 2 | 0 million |
| N | +8 million |

---

## 5. Advanced Shadows: CVA and Wrong Way Risk (WWR)

### Credit Valuation Adjustment (CVA)

CVA is the **"market price" of counterparty risk**. It represents the "haircut" — the difference between a risk-free contract and its true value after accounting for the probability of the other party failing.

```
CVA ≈ (1 − R) · Σ EPE(tᵢ) · PD(tᵢ₋₁, tᵢ)
```

Where:
- `R` = Recovery Rate
- `EPE(t)` = Expected Positive Exposure at time `t`
- `PD(tᵢ₋₁, tᵢ)` = Probability of default in that time slice

**Example:**
- Risk-free portfolio value: 10M
- Probability of default next year: 5%
- Recovery Rate: 40%
- Loss if default = 10M × (1 − 0.4) = 6M
- CVA = 0.05 × 6M = **300K**
- Adjusted portfolio value: 10M → **9.7M**

### Risk Nuances

| Metric | Focus | Description |
|---|---|---|
| Expected Positive Exposure (EPE) | The Average | Mean of the exposure distribution; basis for CVA |
| Wrong Way Risk (WWR) | The Correlation | Danger that exposure rises exactly as default probability increases |

> **Key Insight:** Wrong Way Risk (WWR) is the danger of "the house burning down at the same time the fire department goes on strike." Linear correlation alone does **not** generate substantial WWR in collateralized portfolios. The true driver is the **"Jump-at-Default"** — an instantaneous spike in exposure at the exact moment of a counterparty's collapse, which lagged collateral cannot mitigate.

---

## 6. Theoretical Financial Framework — Detailed Notes

### 6.1 Market Simulation

*How do prices move if markets move randomly?*

#### (i) Stochastic Differential Equations

- Deterministic: `dx/dt = f(x)`
- SDE: `dS_t = μ·S_t·dt + σ·S_t·dW_t`
  - Drift (average growth) + Brownian Motion Shock (volatility/noise)

#### (ii) Geometric Brownian Motion (GBM)

Solution to the SDE, discretized:
```
S_{t+Δt} = S_t · exp( (μ − σ²/2)·Δt + σ·√Δt·Z )    where Z ~ N(0,1)
```
*Next Price = Current Price × Random Growth Factor*

#### (iii) Monte Carlo Path Generation

- Simulate thousands of paths (each = a sequence of random shocks).
- All outcomes collected as a distribution — prerequisite for computing risk.

### 6.2 Risk Metrics

#### (i) Potential Future Exposure (PFE)

- The X% worst exposure in the future.
- Exposure = how much money would be lost if the counterparty immediately defaults.
- Flipped case: if contract is worth −3M to us, we owe them; if they default, we lose nothing → `max(V, 0)`.

**PFE Calculation:**
1. Simulate portfolio value under 5,000 scenarios at time `t`.
2. Take only positive values.
3. Take the 99th percentile → **PFE**.

#### (ii) Expected Positive Exposure (EPE)

Instead of worst-case (PFE), take the **average** of positive exposures:
```
EPE(t) = E[max(V, 0)]
```
Smoother than PFE. Used in CVA computation.

#### (iii) Credit Valuation Adjustment (CVA)

- `Actual Loss = Exposure × (1 − Recovery Rate)`
- CVA = Exposure × Probability of Default × Severity of Loss, summed across time.
- PFE = market risk only. CVA adds credit risk (exposure + PD + recovery).

---

## 7. Advanced Risk Concepts

### 7.1 Wrong Way Risk (WWR)

- **Naive assumption:** Exposure and default probability are independent.
- **WWR:** Exposure increases **exactly when** the counterparty is more likely to default.
- **Example:** Trading oil swaps with an oil company. Oil prices crash → company's credit quality deteriorates → they default when most needed.

**Credit Risk Correlation formula:**
```
ε_hazard = ρ·ε_asset + √(1 − ρ²)·ε_r
```
When random shocks to asset prices drop, counterparty default probability increases (Bad WWR).

**ρ interpretation:**
- `ρ = 0` → independent
- `ρ > 0` → wrong way risk
- `ρ < 0` → right way risk

### 7.2 Jump-at-Default Modelling

- Brownian Motion assumes smooth, continuous price changes.
- In reality, a default causes an **instant market shock** — a "jump."
- **Jump Model:** `S → S·(1 + y)` (sudden discontinuity)
- Captures spikes in FX, interest rates, and credit spreads.
- **Credit Spread** = premium borrower pays above risk-free rate; spikes if the market fears default.
- GBM cannot model these abrupt spikes; Jump Diffusion can.
- Jump effects usually **dominate WWR** as they are more violent than gradual smooth trends.

### 7.3 Correlation Parameters Summary

| ρ | Interpretation |
|---|---|
| 0 | Independent (no correlation) |
| > 0 | Wrong Way Risk |
| < 0 | Right Way Risk |

---

## 8. Regulatory Context

### 8.1 Basel III

After 2008, banks were undercapitalized against derivatives and counterparty risk. Basel III forces banks to:
- Hold more capital.
- Measure counterparty risk rigorously.
- Conduct stress testing — capital at the bank must reflect PFE under stress.
- Shift from overnight batch risk calculation to real-time.

### 8.2 FRTB (Fundamental Review of the Trading Book)

A part of Basel III with:
- Stricter models, backtesting, P&L tests, and a fallback approach.
- Banks must prove models match real P&L, capture tail risk, and work under stress.

### 8.3 SA-CCR (Standardized Approach for CCR)

Replaced the older Current Exposure Method (CEM). SA-CCR is:
- Risk-sensitive, asset-class specific, collateral-aware, netting-aware.
- During jumps, earlier margin calls would fail. Now: real-time margin call triggered if `Exposure > Counterparty Collateral` → bank is unsecured until topped up.

---

## 9. System Architecture & Execution Flow

### Components

```
Browser → Backend (API Layer) → Risk Engine (C++) → Database
```

### Execution Steps

1. **Generate normal shocks** `N(0,1)` representing random market factors.
2. **Apply correlation:** Multiply independent shocks by Cholesky matrix.
3. **Update price via GBM:** Move asset/portfolio price forward in time.
4. **Compute exposure:** `max(Value, 0)`.
5. **Aggregate exposures** at every time strip into a distribution.
6. **Compute EPE, PFE, CVA.**
7. **Trigger margin call alert** if exposure exceeds collateral.
8. **Send computed risk metrics** to user dashboard.

### Computationally Costly Operations

- Normal random number generation (RNG)
- Exponentials, multiplications

### Optimizations

| Optimization | Description |
|---|---|
| SIMD / AVX-512 | Vectorized Monte Carlo — process 4–8 paths per clock cycle |
| xoroshiro128+ RNG | Hardware-level fast RNG for Monte Carlo |
| Inverse CDF Approximation | Cheap approximation to generate normal random variables |
| Thread-local RNG | Lock-free parallelism — each thread has its own RNG state |
| Structure of Arrays (SoA) | Cache-optimized data layouts for processor-memory gap |
| Antithetic Variates | Variance reduction technique |
| Quickselect / nth_element | Find quantile without full sort |
| Discard paths, keep values | Reduced memory footprint |

### Risk Engine Internals

- **Thread-local RNG:** Each thread has its own RNG — no lock contention across 10,000-path simulations.
- **Cholesky Decomposition:** Computed once per simulation run; shared across all Monte Carlo paths. Turns independent standard shocks into correlated ones.
- Each simulation runs 10,000 paths representing 10,000 possible future worlds, evolving over a 1-year horizon.

### Key Algorithms

| Algorithm | Purpose |
|---|---|
| Xoshiro / xoroshiro128+ | Random Number Generation |
| Inverse CDF | Convert uniform randoms to normal distribution |
| Variance Reduction | Reduce Monte Carlo noise (antithetic variates, etc.) |
| Quickselect / nth_element | Quantile selection for PFE |
| Cholesky Decomposition | Multi-asset correlation / WWR |

---

## 10. User Dashboard Features

### Stress Testing Controls

Risk managers can adjust:
- Volatility shock (e.g., +20%)
- FX rate shock (e.g., −10%)
- Equity index shock (e.g., −15%)
- RBI interest rate changes
- Widening credit spreads
- Recovery rate and hazard rate multiplier
- Correlation increase (WWR amplification)
- Jump spike/dip amplitude
- Reduced collateral scenarios
- Macro shocks (all prices)
- Margin Period of Risk (MPR)

### Dashboard Capabilities

| Feature | Description |
|---|---|
| Real-Time Mode | Tick-by-tick market data, 24/7 live metrics |
| Historical Data | TimescaleDB-backed time-series for decision queries |
| Latency Reporting | Engine and API latency metrics |
| Margin Call Button | Notify counterparty directly from dashboard |
| WebGL / Canvas Charts | High-performance charting |
| Export | PDF / CSV report export |
| Counterparty Management | Add counterparties, select confidence level |
| Streaming Progress | Live simulation progress indicator |
| Audit Log | Usage and access logs with role-based access control |
| Email Alerts | Real-time alerts for risk managers |
| Historical Dashboards | "Wayback machine" style time comparison |
| Explain the Spike | Jump logging and explanation |
| Quick Approximation Mode | Reduced-path fast approximate calculations |
| Dark Mode | Green/red delta highlights |
| Freeze Market Data | Capture current market state |
| Change Tracker | Changes since last login / parameter diff |
| Optimal Collateral Suggestion | Recommend optimal transfer amount |
| Risk Concentration | Exposure by counterparty, asset, or sector |
| Contributor Analysis | Breakdown of what's driving counterparty risk |
| Unstable Parameter Warning | Alert when paths are too few for reliable output |

---

# Part II — UML Diagrams & System Design

> Total diagrams: **12** (1 Architecture, 1 Class, 2 DFD, 3 Use Case, 3 Sequence, 2 Activity)

---

## 11. UML Diagrams & System Design

### 11.1 System Architecture Diagram

High-level logical component view across the full application stack.

```
SYSTEM ARCHITECTURE
Realtime CCR & Margin Engine

┌──────────────────────────────────────────────────────────┐
│                    CLIENT LAYER (Browser)                │
│                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────────┐   │
│  │  Dashboard │   │ Stress Test│   │  Margin /      │   │
│  │  Overview  │   │  Controls  │   │ Regulatory View│   │
│  └────────────┘   └────────────┘   └────────────────┘   │
│              WebSocket / REST API (JSON)                 │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│             BACKEND / API LAYER (Node / Python)          │
│                                                          │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────┐   │
│  │   Auth &     │  │    Input       │  │   Cache    │   │
│  │ Role Manager │  │   Validator    │  │   Layer    │   │
│  └──────────────┘  └────────────────┘  └────────────┘   │
│                                                          │
└────────────┬──────────────────────────────────┬──────────┘
             │                                  │
┌────────────▼────────────┐       ┌─────────────▼──────────┐
│    RISK ENGINE (C++)    │       │     DATABASE LAYER      │
│                         │       │                         │
│  ┌───────────────────┐  │       │  ┌─────────────────┐   │
│  │  Monte Carlo      │  │       │  │ Counterparty &  │   │
│  │  (GBM + Jump      │  │       │  │ Portfolio Data  │   │
│  │   Diffusion)      │  │       │  └─────────────────┘   │
│  └───────────────────┘  │       │  ┌─────────────────┐   │
│  ┌───────────────────┐  │       │  │ Historical      │   │
│  │ Cholesky / WWR    │  │       │  │ Metrics &       │   │
│  │ Correlation Module│  │       │  │ Audit Log       │   │
│  └───────────────────┘  │       │  └─────────────────┘   │
│  ┌───────────────────┐  │       │  ┌─────────────────┐   │
│  │ Exposure / PFE /  │  │       │  │ Simulation      │   │
│  │ EPE / CVA Calc    │  │       │  │ Results         │   │
│  └───────────────────┘  │       │  │ (TimescaleDB)   │   │
│  ┌───────────────────┐  │       │  └─────────────────┘   │
│  │ Margin Call Alert │  │       └────────────────────────┘
│  │ Trigger Module    │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

> *Note: Detailed internal architecture (server topology, deployment nodes, DB schema) is to be finalized. This diagram captures the logical component view.*

---

### 11.2 Class Diagram

Core entities, attributes, methods, and relationships.

```
CLASS DIAGRAM
Realtime CCR & Margin Engine — Core Entities

┌─────────────────────┐          ┌──────────────────────────┐
│    Counterparty     │ 1      * │       Portfolio           │
├─────────────────────┤──────────├──────────────────────────┤
│ -id: String         │          │ -id: String               │
│ -name: String       │          │ -counterpartyId: String   │
│ -creditRating: Enum │          │ -derivatives: List<Deriv> │
│ -hazardRate: Float  │          │ -collateral: Float        │
│ -recoveryRate: Float│          │ -netValue: Float          │
├─────────────────────┤          ├──────────────────────────┤
│ +getDefaultProb()   │          │ +getValue(): Float        │
│ +updateHazardRate() │          │ +getExposure(): Float     │
└─────────────────────┘          └──────────┬───────────────┘
                                             │ 1
                                             │ has-many
                                             │ *
                                  ┌──────────▼───────────────┐
                                  │       Derivative          │
                                  ├──────────────────────────┤
                                  │ -id: String               │
                                  │ -type: Enum (IRS/CDS/FX)  │
                                  │ -notional: Float          │
                                  │ -maturity: Date           │
                                  │ -underlyingPrice: Float   │
                                  ├──────────────────────────┤
                                  │ +markToMarket(): Float    │
                                  └──────────────────────────┘

┌─────────────────────┐  1    1  ┌──────────────────────────┐
│  SimulationEngine   │──────────│       RiskMetrics         │
├─────────────────────┤          ├──────────────────────────┤
│ -numPaths: Int      │          │ -pfe99: Float             │
│ -horizon: Float     │          │ -epe: Float               │
│ -dt: Float          │          │ -cva: Float               │
│ -mu: Float          │          │ -wwr: Float               │
│ -sigma: Float       │          │ -marginRequired: Float    │
│ -rho: Float (WWR)   │          │ -computedAt: Timestamp    │
├─────────────────────┤          ├──────────────────────────┤
│ +generateNormals()  │          │ +toJSON(): String         │
│ +applyCholesky()    │          │ +export(): File           │
│ +runGBM(): Paths    │          └──────────────────────────┘
│ +calcExposure()     │
│ +calcPFE(): Float   │
│ +calcCVA(): Float   │
└─────────┬───────────┘
          │ uses
          ▼
┌──────────────────────┐        ┌──────────────────────────┐
│    StressScenario    │        │       MarginCall          │
├──────────────────────┤        ├──────────────────────────┤
│ -volShock: Float     │        │ -id: String               │
│ -fxShock: Float      │        │ -counterpartyId: String   │
│ -equityShock: Float  │        │ -amount: Float            │
│ -interestRateShock   │        │ -status: Enum             │
│ -creditSpreadShock   │        │ -triggeredAt: Timestamp   │
│ -jumpAmplitude: Float│        │ -dueBy: Timestamp         │
├──────────────────────┤        ├──────────────────────────┤
│ +apply(): Scenario   │        │ +send()                   │
└──────────────────────┘        │ +acknowledge()            │
                                └──────────────────────────┘

Relationships:
  Counterparty    1──*   Portfolio
  Portfolio       1──*   Derivative
  SimulationEngine 1──1  RiskMetrics  (produces)
  StressScenario  ──uses──> SimulationEngine
  RiskMetrics     ──triggers──> MarginCall
```

---

### 11.3 Data Flow Diagrams (DFD)

#### 11.3.1 DFD Level 0 — Context Diagram

```
DFD LEVEL 0 — CONTEXT DIAGRAM
Realtime CCR & Margin Engine

┌──────────────┐  Market Data          ┌──────────────┐
│ Market Data  │──────────────────────►│              │
│   Provider   │  Prices/Rates/Spreads │              │
└──────────────┘                       │              │
                                       │              │
┌──────────────┐  Counterparty Config  │              │
│ Risk Manager │◄─────────────────────►│   REALTIME   │
│   (User)     │  PFE / CVA / Margin   │     CCR      │
└──────────────┘  Stress Params        │    ENGINE    │
                                       │    SYSTEM    │
┌──────────────┐  Margin Call Alert    │              │
│ Counterparty │◄─────────────────────►│              │
│  (External)  │  Collateral Top-up    │              │
└──────────────┘                       │              │
                                       │              │
┌──────────────┐  Regulatory Reports   │              │
│  Regulator   │◄──────────────────────│              │
│(Basel/FRTB)  │  SA-CCR Capital Data  │              │
└──────────────┘                       └──────────────┘

External Entities: Market Data Provider | Risk Manager | Counterparty | Regulator
```

#### 11.3.2 DFD Level 1 — Detailed Process View

```
DFD LEVEL 1 — DETAILED

[External]           Market Prices
Provider  ─────────────────────────► P1: MARKET DATA INGESTION
                                             │
                                             │ Normalised Price Ticks
                                             ▼
                                 ┌──── D1: Price Cache ─────┐
                                 │    (In-Memory / Redis)    │
                                 └─────────────┬─────────────┘
                                               │
Stress Params              Spot Prices         │
[Risk Manager] ────────────────────────► P2: SIMULATION ENGINE
                                              (Monte Carlo / GBM)
                                               │
                    10k Paths                  │
                    ◄──────────────────────────│
                             ┌─────────────────▼──────────────┐
                             │   D2: Simulation Results Store   │
                             └──────────────┬─────────────────┘
                                            │ Exposure Distribution
                                            ▼
[Risk Manager] ◄─────────── P3: RISK METRIC CALCULATION
                               (PFE / CVA / WWR)      (PFE, EPE, CVA, WWR)
                                            │
                                            │ Risk Metrics
                             ┌─────────────▼──────────────────┐
                             │    D3: Risk Metrics Database     │
                             │  (TimescaleDB — time-series)     │
                             └──────────────┬─────────────────┘
                                            │
                                            ▼
[Counterparty] ◄───────────── P4: MARGIN CALL ENGINE
(Alert/Request)           (Compare Exposure vs Collateral)
                                            │
                                            │ Audit Trail
                             ┌─────────────▼──────────────────┐
                             │       D4: Audit & Event Log      │
                             └──────────────────────────────────┘
                                            │
[Regulator] ◄────────────── P5: REGULATORY REPORT GEN
(SA-CCR / FRTB Reports)     (Basel III / FRTB Compliance)

Data Stores: D1 Price Cache | D2 Simulation Results | D3 Risk Metrics DB | D4 Audit Log
```

---

### 11.4 Use Case Diagrams

#### 11.4.1 Use Case: Simulation & Exposure Module

```
USE CASE DIAGRAM — SIMULATION & EXPOSURE MODULE

                         ┌─────────────────────────────────────┐
                         │           <<System>>                │
                         │    CCR Simulation & Exposure        │
                         │                                     │
                         │  ○ Configure Simulation Params      │
                         │    (paths, horizon, dt, σ, μ)       │
                         │                                     │
                         │  ○ Run Monte Carlo Simulation       │
                         │    <<include>>                      │
                         │    └─ Generate GBM Paths            │
                         │    └─ Apply Cholesky Correlation    │
                         │                                     │
┌──────────────┐         │  ○ View PFE Profile (over time)    │
│              │         │                                     │
│    Risk      │─────────│  ○ View Exposure Distribution       │
│   Manager    │         │                                     │
│              │         │  ○ Freeze / Capture Market Data     │
└──────────────┘         │                                     │
                         │  ○ Export Results (PDF / CSV)       │
                         │                                     │
                         │  ○ Enable Quick Approx Mode         │
                         │    <<extend>>                       │
                         │    └─ Use Reduced Paths             │
                         │                                     │
                         └─────────────────────────────────────┘

<<include>> = mandatory sub-flow     <<extend>> = optional enhancement
```

#### 11.4.2 Use Case: Margin Management Module

```
USE CASE DIAGRAM — MARGIN MANAGEMENT MODULE

                     ┌────────────────────────────────────┐
                     │   <<System>>  Margin Management    │
                     │                                    │
                     │  ○ View Active Margin Calls        │
                     │    (status, amount, counterparty)  │
                     │                                    │
┌───────────────┐    │  ○ Trigger Manual Margin Call      │
│     Risk      │    │    <<include>>                     │
│    Manager    │────│    └─ Compute Excess Exposure      │
└───────────────┘    │    └─ Notify Counterparty          │
                     │                                    │
                     │  ○ Acknowledge Margin Call         │
                     │                                    │
                     │  ○ Suggest Optimal Collateral Amt  │
┌───────────────┐    │    <<extend>>                      │
│ Counterparty  │────│    └─ Receive Collateral Top-up    │
│  (External)   │    │                                    │
└───────────────┘    │  ○ View Collateral vs Exposure     │
                     │    (Current / Historical)          │
                     │                                    │
                     │  ○ Set Margin Alert Thresholds     │
                     │                                    │
                     │  ○ Export Margin Report            │
                     └────────────────────────────────────┘

Actors: Risk Manager, Counterparty (external)
```

#### 11.4.3 Use Case: Stress Testing & Regulatory Module

```
USE CASE DIAGRAM — STRESS TESTING & REGULATORY MODULE

                      ┌────────────────────────────────────────┐
                      │           <<System>>                   │
                      │   Stress Testing & Regulatory          │
                      │                                        │
                      │  ○ Configure Stress Parameters         │
                      │    (vol +20%, FX -10%, equity -15%,   │
                      │     credit spread, hazard rate, ρ)     │
┌───────────────┐     │                                        │
│     Risk      │     │  ○ Run Stress Scenario                 │
│    Manager    │─────│    <<include>>                         │
└───────────────┘     │    └─ Re-run Simulation w/ Shocks      │
                      │    └─ Recompute PFE / CVA              │
                      │                                        │
                      │  ○ Compare Stressed vs Base Metrics    │
                      │    (Delta highlights: green / red)     │
                      │                                        │
                      │  ○ Explain Exposure Spike              │
                      │    (Jump-at-Default logging)           │
                      │                                        │
                      │  ○ View Regulatory Capital (SA-CCR)    │
                      │                                        │
┌───────────────┐     │  ○ View FRTB Compliance Status         │
│  Regulator    │─────│                                        │
└───────────────┘     │  ○ Generate Regulatory Report          │
                      │    <<include>>                         │
                      │    └─ Compile SA-CCR / Basel III       │
                      │    └─ Export as PDF / CSV              │
                      │                                        │
                      └────────────────────────────────────────┘

Actors: Risk Manager, Regulator
```

---

### 11.5 Sequence Diagrams

#### 11.5.1 Sequence: Run Monte Carlo Simulation

End-to-end flow from the Risk Manager clicking 'Run' to receiving computed PFE, CVA, and WWR metrics.

```
SEQUENCE DIAGRAM — Run Monte Carlo Simulation

RiskManager  Browser/UI      Backend      RiskEngine(C++)    Database
     │             │             │               │               │
     │──Run Sim───►│             │               │               │
     │             │──POST /run─►│               │               │
     │             │             │──Validate─────►               │
     │             │             │   inputs       │               │
     │             │             │──Fetch Static──────────────────►
     │             │             │   Data         │               │
     │             │             │◄────────────────── Portfolio   │
     │             │             │   Data         │               │
     │             │             │──Dispatch─────►│               │
     │             │             │   Sim Job      │               │
     │             │             │               │──Init Thread   │
     │             │             │               │  Local RNG     │
     │             │             │               │──Cholesky      │
     │             │             │               │  Decomp        │
     │             │             │               │               │
     │             │             │    [For each path 1..10k]      │
     │             │             │               │──Generate Z    │
     │             │             │               │  N(0,1) shocks │
     │             │             │               │──Apply corr.   │
     │             │             │               │  (Cholesky ×)  │
     │             │             │               │──Update price  │
     │             │             │               │  via GBM       │
     │             │             │               │──calc max(V,0) │
     │             │             │    [End loop]  │               │
     │             │             │               │               │
     │             │             │               │──Aggregate     │
     │             │             │               │  Distributions │
     │             │             │               │──Compute PFE   │
     │             │             │               │  (99th pctile) │
     │             │             │               │──Compute EPE   │
     │             │             │               │──Compute CVA   │
     │             │             │               │──Compute WWR   │
     │             │             │◄──RiskMetrics─│               │
     │             │             │──Store────────────────────────►│
     │             │             │  Metrics       │               │
     │             │◄──JSON──────│               │               │
     │             │  Response   │               │               │
     │◄──Update────│             │               │               │
     │  Dashboard  │             │               │               │
```

#### 11.5.2 Sequence: Margin Call Trigger

Automated and manual margin call workflows — from exposure threshold breach to counterparty notification.

```
SEQUENCE DIAGRAM — Margin Call Trigger

RiskEngine    Backend     RiskManager    Counterparty    Database
     │             │             │               │           │
  [After each simulation tick]   │               │           │
     │──Exposure───►             │               │           │
     │  exceeds     │             │               │           │
     │  Collateral  │             │               │           │
     │             │──Alert────►│               │           │
     │             │  Margin     │               │           │
     │             │  Breach     │               │           │
     │             │            │──Review────────│           │
     │             │            │  Exposure      │           │
     │             │            │  Details       │           │
     │             │            │               │           │
  alt [Manual Override]         │               │           │
     │             │            │──Adjust────────│           │
     │             │            │  Threshold     │           │
     │             │            │               │           │
  alt [Confirm Margin Call]     │               │           │
     │             │◄──Confirm─│               │           │
     │             │            │               │           │
     │             │──Send Margin Call──────────►           │
     │             │  Notification  │           │           │
     │             │──Log Event─────────────────────────────►
     │             │  (Audit Trail) │           │           │
     │             │            │◄──Collateral──│           │
     │             │            │  Top-up       │           │
     │             │            │  Received     │           │
     │             │──Update───►│               │           │
     │             │  Collateral│               │           │
     │             │  Balance   │               │           │
     │──Re-run─────►            │               │           │
     │  Exposure    │            │               │           │
     │  Check       │            │               │           │
     │             │──Status───►│               │           │
     │             │  Resolved  │               │           │
```

#### 11.5.3 Sequence: User Login & Role-Based Access

```
SEQUENCE DIAGRAM — Login & Role-Based Access

User/Browser     Backend/Auth        Database      RiskEngine
     │                │                  │               │
     │──POST /login──►│                  │               │
     │  {email, pwd}  │                  │               │
     │                │──Query User──────►               │
     │                │                  │               │
  alt [User Not Found]│                  │               │
     │                │◄──404 Not Found──│               │
     │◄──401──────────│                  │               │
  alt [Valid Credentials]               │               │
     │                │◄──User + Role────│               │
     │                │  (RISK_MANAGER   │               │
     │                │   /AUDITOR)      │               │
     │                │──Generate JWT────►               │
     │◄──JWT Token────│                  │               │
     │                │                  │               │
  [Subsequent requests]                 │               │
     │──GET /api/pfe─►│                  │               │
     │  Authorization:│                  │               │
     │  Bearer <token>│                  │               │
     │                │──Validate JWT     │               │
     │                │──Check Role       │               │
  alt [RISK_MANAGER]  │                  │               │
     │                │──Dispatch──────────────────────►│
     │                │  simulation job   │               │
     │◄──Metrics──────│                  │               │
  alt [AUDITOR — Read Only]             │               │
     │                │──Query────────────►              │
     │                │  Metrics Only     │               │
     │◄──Read-Only────│                  │               │
  alt [Unauthorized]  │                  │               │
     │◄──403──────────│                  │               │
```

---

### 11.6 Activity Diagrams

#### 11.6.1 Activity: Monte Carlo Simulation Pipeline

Step-by-step process flow inside the C++ Risk Engine for one full simulation run.

```
ACTIVITY DIAGRAM — Monte Carlo Simulation Pipeline

● (Start)
│
▼
┌─────────────────────────────┐
│  Receive Simulation Request │
│  (params: paths, horizon,   │
│   mu, sigma, rho, portfolio)│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐                ┌──────────────────────┐
│  Validate Input Parameters  │── [Invalid] ──►│ Return Error to UI   │
└──────────────┬──────────────┘                └──────────────────────┘
               │ [Valid]
               ▼
┌─────────────────────────────┐
│   Init Thread-Local RNGs    │
│      (xoroshiro128+)        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Compute Cholesky          │
│   Decomposition (for WWR)   │
│   (done once; shared memory)│
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│  [PARALLEL — each path i]   │ ◄────────────────────────┐
└──────────────┬──────────────┘                          │
               │                                         │
               ▼                                         │
┌─────────────────────────────┐                         │
│   Generate Z ~ N(0,1)       │                         │
│   via inverse CDF approx    │                         │
└──────────────┬──────────────┘                         │
               │                                         │
               ▼                                         │
┌─────────────────────────────┐                         │
│   Apply Cholesky: correlated│                         │
│   shocks for WWR modelling  │                         │
└──────────────┬──────────────┘                         │
               │                                         │
               ▼                                         │
┌─────────────────────────────┐                         │
│   Step time-forward via GBM │                         │
│   S_{t+dt} = S_t * exp(...) │                         │
└──────────────┬──────────────┘                         │
               │                                         │
         [Jump event?]                                   │
         Yes │      │ No                                 │
             │      └─────────────────────────────────►│
             ▼                                           │
┌─────────────────────────────┐                         │
│   Apply Jump-at-Default:    │                         │
│   S → S * (1 + y)           │──────────────────────►│
└─────────────────────────────┘  (loop over time steps 1..T)
               │
      (all paths complete)
               │
               ▼
┌─────────────────────────────┐
│  Compute Exposure per path: │
│  E_i(t) = max(V_i(t), 0)    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Aggregate into Distribution│
│  (exposure array at each t) │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Compute Risk Metrics:                        │
│  PFE = 99th percentile (quickselect)          │
│  EPE = mean(positive exposures)               │
│  CVA = Σ EPE(t) * PD(t) * (1 − Recovery)     │
│  WWR adjustment (if ρ > 0)                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Check: Exposure > Collateral│
└──────┬──────────────────────┘
       │ Yes                   │ No
       ▼                       ▼
┌──────────────┐      ┌────────────────────┐
│ Trigger      │      │ Return Metrics     │
│ Margin Call  │      │ to Backend / UI    │
│ Alert        │      └────────────────────┘
└──────────────┘               │
       │                       │
       └──────────┬────────────┘
                  ▼
               ⊙ (End)
```

#### 11.6.2 Activity: Margin Call Decision & Resolution

```
ACTIVITY DIAGRAM — Margin Call Decision & Resolution

● (Start — triggered by exposure breach)
│
▼
┌───────────────────────────┐
│  Compute Excess Exposure  │
│  = Current PFE - Collateral│
└─────────────┬─────────────┘
              │
       [Excess > Threshold?]
        Yes │        │ No
            │        ▼
            │  ┌───────────────────┐
            │  │  Log & Monitor;   │
            │  │  No Action        │──► ⊙ End
            │  └───────────────────┘
            ▼
┌───────────────────────────┐
│  Generate Margin Call     │
│  (amount, due date, reason)│
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Notify Risk Manager      │
│  (Dashboard Alert + Email)│
└─────────────┬─────────────┘
              │
       [Risk Manager reviews]
              │
     ┌────────┴──────────────┐
     │ Approve               │ Override / Dismiss
     ▼                       ▼
┌──────────────────┐  ┌──────────────────────────┐
│  Send Margin Call│  │ Log Override with Reason  │
│  to Counterparty │  │ (Audit trail entry)       │──► ⊙ End
└────────┬─────────┘  └──────────────────────────┘
         │
  [Within MPoR window?]
   Yes │        │ No
       ▼        ▼
┌──────────────────┐   ┌──────────────────────┐
│ Await Collateral │   │ Escalate: Mark as     │
│ Top-up from      │   │ Uncollateralised      │
│ Counterparty     │   │ Exposure; Report      │
└──────────┬───────┘   └──────────────────────┘
           │
    [Collateral received?]
     Yes │        │ No (timeout)
         ▼        ▼
┌──────────────┐  ┌──────────────────────────────┐
│ Update       │  │ Trigger Escalation:            │
│ Collateral   │  │ Senior Risk / Regulator Alert  │
│ Balance      │  └──────────────────────────────┘
└──────┬───────┘
       │
┌──────▼──────────────┐
│ Re-run Exposure Check│
│ (close if resolved)  │
└──────────────────────┘
       │
       ▼
    ⊙ (End)
```

---

## 12. Notes on Pending Design Decisions

The following areas require additional detail before the relevant diagrams can be fully finalized:

| Area | What's Needed | Affects Diagram |
|---|---|---|
| System Architecture | Server topology, deployment nodes, tech stack specifics (message queue, load balancer) | System Architecture |
| Database Schema | Table names, relationships, time-series schema, primary/foreign keys | Class Diagram, DFD Level 1 |
| Authentication | Auth provider (JWT/OAuth), session management, refresh token flow | Sequence: Login |
| API Spec | Exact REST/WebSocket endpoint names, request/response schemas | Sequence Diagrams |
| Deployment | Cloud provider, containerisation (Docker/K8s), server specs | Deployment Diagram (future) |

---

*End of Document — Realtime CCR & Margin Engine Complete Reference*
