# CCR Engine

**Counterparty Credit Risk & XVA Computation Platform for OTC Derivatives**

---

## What It Does

CCR Engine computes the three quantities that sit at the heart of modern counterparty credit risk management:

- **PFE — Potential Future Exposure**: the worst-case exposure to a counterparty at each future date, at a chosen confidence level (default 95%). Regulators and credit officers use PFE to set credit limits and determine how much collateral a counterparty must post.

- **EPE — Expected Positive Exposure**: the time-averaged mean of positive exposures across all simulated paths. Used as the input to CVA calculation and as a basis for economic capital.

- **CVA — Credit Valuation Adjustment**: the fair-value cost of counterparty default risk. CVA is a P&L charge that represents how much cheaper a derivative is worth because the counterparty might default before maturity.

All three are computed simultaneously via a single Monte Carlo simulation run. The engine then derives:

- **WWR-CVA**: CVA adjusted for Wrong-Way Risk — the scenario where a counterparty is most likely to default precisely when its exposure to you is largest (e.g., a bank that wrote you CDS protection on itself).
- **Margin Requirement**: the collateral posting required to cover the simulated exposure.
- **Stress scenario**: how PFE/EPE/CVA change under simultaneous shocks to volatility, equity, rates, and hazard rates.

The platform covers five derivative classes: **Interest Rate Swaps (IRS)**, **Credit Default Swaps (CDS)**, **FX forwards**, **Equity options/swaps**, and **Commodity swaps**. These can be mixed in any combination within a single counterparty portfolio.

---

## The Financial Models

### Geometric Brownian Motion — the path engine

Every asset's price evolves under GBM:

```
dS = μ·S·dt + σ·S·dW
```

where `dW` is a Wiener increment. The discretised version used in the engine is the exact log-normal update:

```
S(t+dt) = S(t) · exp( (μ − σ²/2)·dt + σ·√dt·Z )
```

`Z` is a standard normal drawn from a xoroshiro128/AOX PRNG, inverse-transformed via Wichura's AS241 rational approximation (maximum absolute error < 10⁻¹⁵). This PRNG produces 2⁶⁴ statistically independent streams so each simulation path is genuinely decorrelated.

### Correlated assets — Cholesky decomposition

When a portfolio contains multiple assets (e.g., EUR/USD and crude oil in the same trade book), the engine draws correlated standard normals using a Banachiewicz Cholesky decomposition of the user-supplied correlation matrix. This is essential for portfolios where netting benefits matter — ignoring correlation overstates exposure.

### PFE extraction

At each time step, across all simulation paths, the engine computes the portfolio mark-to-market. Exposure is `max(MtM, 0)` — you only lose money if the portfolio is in-the-money when the counterparty defaults. The 95th percentile of the cross-sectional distribution of exposures at each time step is extracted using `std::nth_element` (O(N) partial sort) to form the PFE profile.

### EPE and CVA

EPE at time `t` is simply the mean of positive exposures across all paths. CVA integrates EPE against the counterparty's default probability curve:

```
CVA = (1 − R) · Σ [ EPE(tᵢ) · (PD(tᵢ₋₁) − PD(tᵢ)) · df(tᵢ) ]
```

where `R` is the recovery rate, `PD(t)` is the survival probability from the hazard rate `λ` (PD(t) = exp(−λ·t)), and `df(t)` is the risk-free discount factor. The summation uses **Kahan compensated summation** to prevent floating-point cancellation error — required for regulatory-grade reproducibility.

### Wrong-Way Risk

The engine introduces correlation `ρ` between the Brownian driver of the counterparty's "credit asset" and the exposure path. When `ρ < 0` (the counterparty deteriorates as your exposure grows), WWR-CVA > base CVA. The Cholesky block handles the joint simulation.

### Jump-at-Default

An optional multiplicative shock is applied at the simulated default time. When the credit asset hits a threshold (drawn from the hazard rate), remaining exposure paths are shocked by a user-specified factor. This captures gap risk — the discontinuous jump in MtM when a counterparty defaults without prior notice.

### Margin requirement

```
Margin = max(PFE_peak − Collateral_posted, 0) × (1 + MPOR_days / 360)
```

The Margin Period of Risk (MPOR) adjustment accounts for the time it takes to close out a position after a default, during which you remain exposed.

---

## How Data Flows Through the System

```
┌─────────────────────────┐
│   yfinance (15-min)     │   equity, FX, commodity spot prices
│   FRED API (daily)      │   SOFR, 1Y/5Y/10Y Treasury yields
└───────────┬─────────────┘
            │  every 15 min (background scheduler)
            ▼
┌─────────────────────────┐
│  market_data store      │   cached in-process + price_history table
│  (Python server layer)  │
└───────────┬─────────────┘
            │  user submits sim request (HTTP or WebSocket)
            ▼
┌─────────────────────────┐
│   C++ Monte Carlo       │   xoroshiro128 PRNG → GBM paths → exposure
│   Engine (.so module)   │   → PFE/EPE/CVA/WWR-CVA → RiskMetrics
└───────────┬─────────────┘
            │  results returned in milliseconds
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL / TimescaleDB                  │
│                                                             │
│  risk_metrics   (hypertable)  — one row per simulation run  │
│  audit_log      (hypertable)  — every user action           │
│  price_history  (hypertable)  — tick prices                 │
│  counterparties               — CP master record            │
│  portfolios / derivatives     — trade book                  │
│  margin_calls                 — open / acknowledged / settled│
│  simulation_presets           — saved param sets            │
│  users                        — credentials + roles         │
└───────────┬─────────────────────────────────────────────────┘
            │  after each simulation run
            ▼
┌─────────────────────────┐
│  Margin Call detector   │   flags breach if margin > 0
│  (server post-process)  │   inserts into margin_calls table
│                         │   sends email alert (if SMTP set)
└───────────┬─────────────┘
            │  REST + WebSocket
            ▼
┌─────────────────────────┐
│   SvelteKit dashboard   │   charts, tables, forms, real-time progress
└─────────────────────────┘
```

Every state change — login, simulation, margin call acknowledgement, user creation — is written to the **audit log** hypertable with a timestamp, actor, action, and IP address. The audit log is append-only: no row is ever deleted or modified.

---

## The Web Interface

There are nine pages, each accessible from the left-side navigation rail.

### 1 · Dashboard

The landing page after login. Shows:

- **PFE chart**: the current peak PFE value and the full PFE profile of the most recent simulation run
- **EPE chart**: the expected positive exposure profile for the same run
- **Metric cards**: CVA, WWR-CVA, margin required, compute time, and SIMD architecture used
- **Concentration table**: all counterparties ranked by CVA, with colour-coded credit ratings
- **Market data**: current SOFR, equity and commodity spot prices, API latency badge

### 2 · Counterparties

The counterparty master book. Each record stores:

- Legal name and credit rating (AAA → C)
- Hazard rate `λ` (annualised default probability)
- Recovery rate (fraction recovered if the CP defaults)
- Collateral posted (in USD)
- MPOR — Margin Period of Risk in days

From the counterparty detail page you can click **Simulate Now** to run a simulation pre-filled with that counterparty's ID, credit parameters, and full trade book. You can add, edit, and delete counterparties (RISK_MANAGER and above).

### 3 · Simulate

The main simulation workspace:

- **Left panel**: parameter form covering the counterparty, portfolio (any number of derivatives), simulation settings (path count, time steps, confidence level), stress scenario toggle, and Wrong-Way Risk / Jump-at-Default options
- **Right panel**: live results after the run — PFE chart with spike annotation, EPE chart, metric cards (CVA, WWR-CVA, margin, compute time), CVA attribution by derivative, suggested collateral (margin × 1.10 buffer), and overflow warning if paths diverged

Results stream in real time over a WebSocket — you see the progress bar advance as the C++ engine completes batches of paths.

You can export results as **PDF** (full report) or **CSV** (raw PFE/EPE profile), or save the parameter set as a **Preset** for reuse.

### 4 · Stress Test

Runs the same Monte Carlo simulation but with simultaneous shocks applied:

- **Volatility shock**: multiply σ by a factor (default ×2)
- **Equity shock**: multiply equity spot by a factor (default ×0.7 — a 30% crash)
- **Rate shock**: shift the risk-free rate up by basis points (default +200 bps)
- **Hazard rate shock**: multiply λ by a factor (default ×3)

The page shows base vs. stressed PFE side by side on the same chart and reports the delta in CVA and margin. Stress scenarios must be run from the simulate page with the stress toggle enabled; the Stress Test route provides a dedicated workspace.

### 5 · Margin Calls

Lists every breach event detected across all simulation runs. Each margin call shows:

- Counterparty, run date, breach amount (margin required above zero)
- Status: **PENDING** → **ACKNOWLEDGED** → **SETTLED**
- A **Notify** button that sends an email to the counterparty (requires SMTP configuration)

RISK_MANAGERs can acknowledge and settle calls. Bulk export to CSV is available.

### 6 · Query Builder

An ad-hoc analytics layer over the `risk_metrics` hypertable. You can filter by:

- Date range (from / to)
- Counterparty
- Minimum CVA threshold
- Simulation status

Results appear in a table with CVA, EPE peak, PFE peak, margin required, and compute time. This page is for pattern detection — spotting which counterparties generated the highest CVA over a rolling window, or which runs had the longest compute times.

### 7 · Reports

A paginated history of all past simulation runs. Each row shows the run timestamp, counterparty, CVA, margin, and whether a stress scenario was included. You can:

- Click any run to see its full PFE/EPE charts and metric cards
- Compare two runs side by side (select via checkbox, click Compare)
- Export a run as PDF or CSV
- Re-run a simulation with the same parameters

### 8 · Presets

Saved parameter sets. A preset captures everything needed to reproduce a simulation: all counterparty fields, all derivative specs, simulation settings. Presets can be:

- Private (visible only to the creator) or shared with the team
- Loaded from the Simulate page (URL parameter `?preset_id=...`)
- Renamed, described, and deleted

Use presets to capture approved stress scenarios, regulatory test cases, or common counterparty configurations so they can be reproduced with one click.

### 9 · Admin

Visible to ADMIN role only. Two sub-sections:

- **User Management**: create new users, change roles, activate/deactivate accounts
- **Audit Log**: a full, filterable, read-only view of every recorded event in the system — login attempts, simulation runs, entity changes, margin call state transitions

---

## User Roles

| Role | What they can do |
|---|---|
| `ADMIN` | Everything: user management, all CRUD, all reads, all simulation runs |
| `RISK_MANAGER` | Run simulations, manage counterparties/portfolios/derivatives, acknowledge and settle margin calls, trigger market data refresh |
| `AUDITOR` | Read-only access to all data — no writes, no simulation — plus the audit log |

Role enforcement is applied at the API layer. The frontend additionally hides or disables controls that the current user cannot use.

**Demo credentials:**

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | ADMIN |
| `risk` | `risk123` | RISK_MANAGER |
| `auditor` | `auditor123` | AUDITOR |

---

## Demo Environment — Pre-loaded Book

After running the seed script (`scripts/seed_demo_data.py`), the system contains six counterparties representing different sectors of the OTC derivatives market.

### Alpha Bank S.A. — AA / Rates + Credit

A European investment-grade bank. Hazard rate 0.4% pa, recovery 45%, collateral $2M, MPOR 10 days.

Trade book:
- 10-year EUR interest rate swap, $25M notional, fixed rate 4.5%, semi-annual settlement
- 5-year CDS on European IG index, $15M notional, spread 120 bps, quarterly premium

### Beta Capital LLC — BB / FX + Equity

A mid-market US asset manager. Hazard rate 1.8% pa, recovery 35%, collateral $500k, MPOR 5 days.

Trade book:
- 3-year EUR/USD FX forward, $8M notional, forward rate 1.08
- 2-year equity total return swap on SPY, $5M notional, quarterly reset

### Gamma Hedge Fund Ltd — B / Multi-strategy

An offshore hedge fund. Hazard rate 3.5% pa, recovery 20%, collateral $0 (uncollateralised), MPOR 20 days.

Trade book:
- 5-year IRS, $30M notional (floating leg)
- 3-year equity variance swap on AAPL, $10M notional
- 2-year CDS on US HY index, $20M notional

### Delta Energy Corp — BBB / Commodity + Rates

An energy company with commodity hedging needs. Hazard rate 0.8% pa, recovery 40%, collateral $1M, MPOR 7 days.

Trade book:
- 2-year crude oil commodity swap, $12M notional, fixed $75/bbl
- 5-year IRS, $20M notional (hedging floating-rate debt)

### Epsilon Insurance Group — AAA / Long-dated Rates

A life insurer with long-duration liability matching. Hazard rate 0.2% pa, recovery 50%, collateral $5M, MPOR 10 days.

Trade book:
- 20-year GBP IRS, $50M notional, fixed 4.2%
- 10-year EUR IRS, $30M notional, fixed 3.8%

### Zeta Corp — CCC / Distressed

A distressed corporate. Hazard rate 8.5% pa, recovery 15%, collateral $0, MPOR 30 days.

Trade book:
- 1-year CDS protection bought, $5M notional
- 6-month FX forward, $3M notional

---

## Market Data Sources

### Equity, FX, and Commodity (yfinance)

| Symbol | Asset | Used for |
|---|---|---|
| `SPY` | S&P 500 ETF | US equity benchmark underlying |
| `AAPL` | Apple | Single-name equity underlying |
| `MSFT` | Microsoft | Single-name equity underlying |
| `GS` | Goldman Sachs | Financials underlying |
| `JPM` | J.P. Morgan | Financials underlying |
| `EURUSD=X` | EUR/USD spot | FX underlying |
| `GBPUSD=X` | GBP/USD spot | FX underlying |
| `USDJPY=X` | USD/JPY spot | FX underlying |
| `GC=F` | Gold futures | Commodity underlying |
| `CL=F` | Crude oil futures (WTI) | Commodity underlying |
| `NG=F` | Natural gas futures | Commodity underlying |

Prices are fetched via yfinance with a 15-minute delay (free tier). 30-day rolling historical volatility is computed from log-returns of closing prices. Volatility is used as the σ parameter in GBM.

### Risk-Free Rates (FRED API)

| Series | Description | Used for |
|---|---|---|
| `SOFR` | Secured Overnight Financing Rate | Short-end discount rate, CVA discounting |
| `DGS1` | 1-year Treasury yield | Short-term rate input |
| `DGS5` | 5-year Treasury yield | Mid-curve rate input |
| `DGS10` | 10-year Treasury yield | Long-end rate input |

FRED rates require a free API key from `fred.stlouisfed.org`. Without a key, the server falls back to hard-coded defaults (SOFR 5%, DGS10 4%). With a key, rates are refreshed every 15 minutes alongside equity data.

### What is mocked

- **Credit spreads / CDS hazard rates**: these are entered manually per counterparty (the `hazard_rate` field). There is no automated source for single-name CDS spreads in the free tier.
- **Live price tick stream** (WebSocket `/ws/prices`): this is a GBM walk seeded from the most recent yfinance prices. It is explicitly labelled "Demo Ticks — GBM simulation, not real market data" in both the payload and the UI.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (SvelteKit SPA — TypeScript)                            │
│  Dark financial terminal UI · Chart.js PFE/EPE charts            │
│  Live price ticks (WebSocket) · Role-gated views                 │
└──────────────┬──────────────────────────────┬────────────────────┘
               │ REST /api/v1/*               │ WebSocket /ws/*
┌──────────────▼──────────────────────────────▼────────────────────┐
│  FastAPI Server (Python 3.9 + uvicorn)                           │
│  JWT HS256 auth (ADMIN / RISK_MANAGER / AUDITOR)                 │
│  Counterparty / Portfolio / Derivative CRUD                      │
│  Simulation orchestration → C++ engine via pybind11 (.so)        │
│  Market data pipeline (yfinance + FRED API)                      │
│  APScheduler: market refresh every 15 min, auto-rerun hourly     │
│  PDF / CSV export (ReportLab)                                     │
│  Margin call detection + email alerts (aiosmtplib)               │
│  Audit log (append-only)                                         │
└──────────────┬───────────────────────────────────────────────────┘
               │ pybind11 (.so)
┌──────────────▼───────────────────────────────────────────────────┐
│  C++20 Monte Carlo Engine                                        │
│  xoroshiro128/AOX PRNG  (2^64 independent streams)              │
│  Wichura AS241 normal quantile                                   │
│  Banachiewicz Cholesky for correlated asset simulation           │
│  GBM path simulation  (SoA memory layout, branch-free hot loop) │
│  PFE via nth_element  ·  EPE average  ·  CVA Kahan summation    │
│  Wrong-way risk  ·  Jump-at-default  ·  Stress scenarios        │
│  SIMD dispatch: AVX-512 / AVX2 / ARM NEON / scalar              │
└──────────────────────────────────────────────────────────────────┘
               │ asyncpg
┌──────────────▼───────────────────────────────────────────────────┐
│  PostgreSQL 16 + TimescaleDB 2.x                                 │
│  risk_metrics   (hypertable — PFE/EPE/CVA per run)              │
│  audit_log      (hypertable — append-only event stream)         │
│  price_history  (hypertable — tick prices)                      │
│  users · counterparties · portfolios · derivatives              │
│  margin_calls · simulation_presets                              │
└──────────────────────────────────────────────────────────────────┘
```

The C++ engine is compiled once to a shared library (`.so`). The Python server loads it via pybind11 at startup. Simulation calls cross the Python–C++ boundary in microseconds; the engine does not allocate heap memory inside the Monte Carlo hot loop and uses a single contiguous aligned arena for all simulation state.

SIMD dispatch is resolved at **compile time** via a template `Arch` parameter — AVX-512, AVX2, NEON, or scalar — with zero runtime branching. The build script auto-detects the host CPU and selects the best available target.

---

## Getting Started

### Automated setup (recommended for macOS and Ubuntu)

```bash
git clone <repo-url> ccr
cd ccr
chmod +x scripts/setup_demo.sh
./scripts/setup_demo.sh
```

The script:
1. Checks for `uv`, `node`, `npm`
2. Installs PostgreSQL 16 + TimescaleDB via Homebrew (macOS) or apt (Ubuntu)
3. Creates the `ccr` database and user
4. Copies `.env.example` → `.env` and generates a random JWT secret
5. Runs `uv sync` (Python dependencies) and `alembic upgrade head` (schema migrations)
6. Seeds demo data (runs actual Monte Carlo simulations — takes ~30 seconds)
7. Installs Node.js dependencies

After the script completes:

```bash
# Terminal 1 — API server (port 8000)
./scripts/run_dev.sh --skip-build

# Terminal 2 — Web dashboard (port 5173)
cd web && npm run dev
```

Open **http://localhost:5173** and log in with `admin / admin123`.

### Manual setup (step by step)

**Prerequisites:**
- Python 3.9 (exact version — the pre-built `.so` is version-locked)
- `uv` package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Node.js 18+
- PostgreSQL 16 with TimescaleDB extension

**Step 1 — Database**

```bash
psql -U postgres -c "CREATE USER ccr WITH PASSWORD 'ccr';"
psql -U postgres -c "CREATE DATABASE ccr OWNER ccr;"
psql -U postgres -d ccr -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

**Step 2 — Environment**

```bash
cp .env.example .env
# Edit .env:
#   DATABASE_URL=postgresql+asyncpg://ccr:ccr@localhost:5432/ccr
#   JWT_SECRET=<any long random string>
#   FRED_API_KEY=<your key from fred.stlouisfed.org — free>
```

**Step 3 — Python dependencies**

```bash
uv sync
```

**Step 4 — Database migrations**

```bash
uv run alembic -c server/alembic.ini upgrade head
```

This creates all tables, constraints, indices, and TimescaleDB hypertables.

**Step 5 — Seed demo data**

```bash
uv run python scripts/seed_demo_data.py
```

This creates three users (admin, risk, auditor), six counterparties with portfolios and derivatives, ten simulation runs with real Monte Carlo outputs, margin calls in all three lifecycle states, and 37 audit log entries representing a curated six-day operating history.

**Step 6 — Frontend**

```bash
cd web && npm install
```

### Running

```bash
# API server (stays running — Ctrl-C to stop)
./scripts/run_dev.sh --skip-build

# In a second terminal
cd web && npm run dev
```

---

## Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | **Yes** | — | `postgresql+asyncpg://ccr:ccr@localhost:5432/ccr` |
| `JWT_SECRET` | **Yes** | — | Any long random string for signing JWTs |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `480` | Token lifetime (8 hours) |
| `CORS_ORIGINS` | No | `http://localhost:5173` | Comma-separated allowed frontend origins |
| `FRED_API_KEY` | No | — | Free key from `fred.stlouisfed.org` — enables live SOFR and Treasury yields |
| `SMTP_HOST` | No | — | SMTP server hostname — enables margin call email alerts |
| `SMTP_PORT` | No | `587` | SMTP port |
| `SMTP_USER` | No | — | SMTP username |
| `SMTP_PASSWORD` | No | — | SMTP password |
| `SMTP_FROM` | No | `ccr-alerts@example.com` | From address for margin call notifications |
| `DEBUG_SQL` | No | — | Set to `true` to echo all SQL statements |

---

## API Endpoints

All endpoints require `Authorization: Bearer <token>` except `/api/v1/auth/login`.
Interactive documentation is available at **http://localhost:8000/docs** while the server is running.

| Method | Path | Min role | Description |
|---|---|---|---|
| `POST` | `/api/v1/auth/login` | — | Username + password → JWT |
| `GET` | `/api/v1/auth/me` | Any | Current user profile |
| `GET` | `/api/v1/auth/users` | ADMIN | List all users |
| `POST` | `/api/v1/auth/register` | ADMIN | Create user |
| `PUT` | `/api/v1/auth/users/{id}` | ADMIN | Update role / active status |
| `GET` | `/api/v1/health` | — | Liveness probe + engine build info |
| `POST` | `/api/v1/simulate` | RISK_MGR | Run simulation, persist result, detect margin breach |
| `GET` | `/api/v1/simulate/history` | Any | Past simulation runs (paginated) |
| `POST` | `/api/v1/simulate/compare` | Any | Side-by-side run comparison |
| `GET` | `/api/v1/simulate/{id}/export/pdf` | Any | Download PDF risk report |
| `GET` | `/api/v1/simulate/{id}/export/csv` | Any | Download PFE/EPE profile as CSV |
| `GET` | `/api/v1/simulate/{id}/attribution` | Any | CVA attribution by derivative |
| `GET` | `/api/v1/analytics/concentration` | Any | Counterparty CVA concentration ranking |
| `GET` | `/api/v1/counterparties` | Any | List all counterparties |
| `POST` | `/api/v1/counterparties` | RISK_MGR | Create counterparty |
| `GET` | `/api/v1/counterparties/{id}` | Any | Detail view with portfolios and derivatives |
| `PUT` | `/api/v1/counterparties/{id}` | RISK_MGR | Update |
| `DELETE` | `/api/v1/counterparties/{id}` | RISK_MGR | Delete |
| `POST` | `/api/v1/portfolios` | RISK_MGR | Create portfolio for a counterparty |
| `POST` | `/api/v1/portfolios/{id}/derivatives` | RISK_MGR | Add derivative to portfolio |
| `DELETE` | `/api/v1/portfolios/{pid}/derivatives/{did}` | RISK_MGR | Remove derivative |
| `GET` | `/api/v1/margin-calls` | Any | List margin calls (filterable) |
| `PUT` | `/api/v1/margin-calls/{id}/acknowledge` | RISK_MGR | Acknowledge a breach |
| `PUT` | `/api/v1/margin-calls/{id}/settle` | RISK_MGR | Mark as settled |
| `POST` | `/api/v1/margin-calls/{id}/notify` | RISK_MGR | Send email notification |
| `GET` | `/api/v1/margin-calls/export/csv` | Any | Bulk CSV export |
| `GET` | `/api/v1/market/prices` | Any | Current market parameters (60-second cache) |
| `POST` | `/api/v1/market/refresh` | RISK_MGR | Force immediate market data refresh |
| `GET` | `/api/v1/presets` | Any | List simulation presets |
| `POST` | `/api/v1/presets` | RISK_MGR | Save a new preset |
| `PUT` | `/api/v1/presets/{id}` | RISK_MGR | Update preset |
| `DELETE` | `/api/v1/presets/{id}` | RISK_MGR | Delete preset |
| `GET` | `/api/v1/audit-log` | ADMIN / AUDITOR | Query the audit trail |
| `WS` | `/ws/simulate` | Any (token in first msg) | Streaming simulation progress |
| `WS` | `/ws/prices` | Any (token in first msg) | Live demo price tick stream |

---

## Known Limitations and Caveats

### Model caveats

- **GBM is not a term-structure model.** The engine simulates assets as individual GBM processes. It does not implement a full yield curve model (e.g., Hull-White, LMM). IRS valuation is an approximation based on a fixed notional and a single representative rate.

- **Flat hazard rate.** Each counterparty has a single constant hazard rate `λ`. The engine does not model a term-structure of CDS spreads or stochastic credit.

- **No netting across counterparties.** Netting is applied within a single portfolio (all derivatives with the same counterparty net to a single exposure). There is no cross-counterparty netting or portfolio-level CVA.

- **Correlation input is user-supplied.** The WWR correlation `ρ` between exposure and credit must be provided by the user. The system does not estimate it from historical data.

- **Independent GBM paths per derivative.** When a portfolio contains multiple derivatives, each is simulated independently (separate PRNG streams). The portfolio MtM is the sum of individual derivative MtMs. This captures netting at the portfolio level but not cross-asset dependence (unless the Cholesky correlation matrix is set).

### Market data caveats

- **15-minute delayed prices.** yfinance provides data with a 15-minute lag. Spot prices are not real-time.

- **No CDS spread data.** Hazard rates must be entered manually. The engine does not pull live CDS spreads.

- **FRED rate refresh.** Rates are refreshed every 15 minutes. The server falls back to hard-coded defaults (SOFR 5%) if FRED is unreachable or no key is configured.

- **Demo tick stream is synthetic.** The WebSocket price feed is a GBM walk, not a real market data connection. It is explicitly labelled in the UI.

### Operational caveats

- **Email notifications require SMTP.** The "Notify counterparty" button on the Margin Calls page sends an email only if `SMTP_HOST` and credentials are configured in `.env`. Without these, the button completes without actually sending.

- **Pre-built `.so` is macOS arm64 + Python 3.9.** On other platforms (Linux x86_64, Windows, different Python versions), the C++ engine must be rebuilt using `./scripts/build_engine.sh --bindings`. See the CLAUDE.md build instructions.

- **Auto-rerun scheduler.** The background scheduler re-runs the most recent simulation every hour using the last known parameters. In a production deployment this would be gated on a change event; here it runs unconditionally.

- **No database connection pooling tuning.** The asyncpg connection pool uses framework defaults. For high-concurrency production use, tune `pool_size` and `max_overflow` in `server/core/database.py`.

---

## Building the C++ Engine (if needed)

The pre-built `.so` targets macOS arm64 + Python 3.9. To rebuild for a different platform:

```bash
# Auto-detect SIMD (recommended)
./scripts/build_engine.sh --bindings

# Force a SIMD target
./scripts/build_engine.sh --bindings --arch avx2    # avx512 | avx2 | neon | scalar

# Debug build (adds sanitizers, disables optimisation)
./scripts/build_engine.sh --bindings --debug

# Clean rebuild
./scripts/build_engine.sh --bindings --clean
```

Requires GCC 12+ or Clang 15+, CMake 3.20+. The output `.so` lands in `server/bindings/`.

---

## Project Structure

```
ccr/
├── engine/                  C++20 Monte Carlo engine (static library + pybind11 bindings)
│   ├── include/ccr/         Public headers
│   │   ├── types.hpp        SimParams, RiskMetrics — all data types crossing the boundary
│   │   ├── ccr_engine.hpp   CcrEngine::run() — single public entry point
│   │   └── ...              path_simulator, exposure_engine, cva_integrator, etc.
│   └── src/                 Implementations
│
├── server/                  FastAPI application
│   ├── api/                 REST routes and WebSocket endpoints
│   ├── auth/                JWT security, RBAC decorators
│   ├── bindings/            pybind11 .so + Python glue (SimParams → C++ struct)
│   ├── core/                DB session, config loader, TTL cache, scheduler, engine runner
│   ├── market_data/         yfinance fetcher, FRED client, mock tick generator
│   ├── models/              SQLAlchemy ORM models + Pydantic request/response schemas
│   ├── notifications/       Email alerts (aiosmtplib) + audit log writer
│   ├── reports/             PDF/CSV export (ReportLab)
│   ├── alembic/             Database migration history (4 versions)
│   └── logs/                Rotating log files (git-ignored)
│
├── web/                     SvelteKit TypeScript dashboard
│   └── src/
│       ├── lib/             api.ts · ws-client.ts · state.ts · types.ts · auth.ts
│       ├── components/      charts/ · forms/ · ui/
│       └── routes/          dashboard · simulate · stress · margin-calls
│                            counterparties · reports · query · presets · admin · login
│
├── scripts/
│   ├── setup_demo.sh        One-shot automated setup (macOS + Ubuntu)
│   ├── run_dev.sh           Build + start API server with hot-reload
│   ├── build_engine.sh      CMake wrapper for C++ engine build
│   └── seed_demo_data.py    Demo data seeder (runs real simulations)
│
├── docs/                    Screenshots, literature review (LR.md)
├── config/                  defaults.toml, CMake toolchain modules
├── docker-compose.yml       PostgreSQL + TimescaleDB container
└── .env.example             All supported environment variables with descriptions
```

---

## Database Schema (Appendix)

The schema is managed by Alembic. Run `uv run alembic -c server/alembic.ini upgrade head` to apply all migrations.

**`users`** — authentication and roles
- `id` UUID PK, `username` unique, `hashed_password`, `role` (ADMIN/RISK_MANAGER/AUDITOR), `is_active`, `created_at`, `last_login`

**`counterparties`** — credit master book
- `id` UUID PK, `name`, `rating` (AAA/AA/A/BBB/BB/B/CCC/D), `hazard_rate` float, `recovery_rate` float, `collateral` float, `mpor_days` int, `created_by` FK→users

**`portfolios`** — trade book containers (one or many per counterparty)
- `id` UUID PK, `counterparty_id` FK→counterparties (CASCADE DELETE), `net_value` float, `collateral` float

**`derivatives`** — individual trade records
- `id` UUID PK, `portfolio_id` FK→portfolios (CASCADE DELETE), `deriv_type` (IRS/CDS/FX/EQUITY/COMMODITY), `notional` float, `maturity_years` float, `underlying_price` float, `strike` float, `cash_flow_freq` int

**`risk_metrics`** (TimescaleDB hypertable, partitioned by `time`)
- `run_id` UUID, `time` timestamptz, `counterparty_id`, `cva`, `wwr_cva`, `epe_profile` float[], `pfe_profile` float[], `time_grid` float[], `margin_required`, `compute_time_us`, `arch_used`, `is_stressed`, `note`

**`margin_calls`** — breach events
- `id` UUID PK, `run_id` FK→risk_metrics.run_id, `counterparty_id`, `breach_amount`, `status` (PENDING/ACKNOWLEDGED/SETTLED), `created_at`, `acknowledged_at`, `settled_at`, `acknowledged_by`, `settled_by`

**`audit_log`** (TimescaleDB hypertable, partitioned by `time`)
- `id` UUID, `time` timestamptz, `user_id`, `username`, `action`, `resource_type`, `resource_id`, `details` JSONB, `ip_address`

**`simulation_presets`** — saved parameter sets
- `id` UUID PK, `name`, `description`, `params_json` JSONB, `is_shared`, `created_by` FK→users, `created_at`, `last_used_at`, `use_count`

**`price_history`** (TimescaleDB hypertable, partitioned by `time`)
- `time` timestamptz, `symbol`, `price` float, `volatility` float

---

## Literature

`LR.md` contains the 13 foundational papers behind the algorithmic and numerical choices — Andersen-Pykhtin-Sokol PFE methodology, Kahan CVA integration, xoroshiro128 PRNG, Wichura AS241 rational approximation, Banachiewicz Cholesky, SIMD dispatch patterns, and Basel III / SA-CCR regulatory framework.
