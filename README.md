# CCR Engine

**Counterparty Credit Risk & XVA Computation Platform**

Monte Carlo simulation engine for OTC derivatives, computing Potential Future Exposure (PFE), Credit Valuation Adjustment (CVA), and Expected Positive Exposure (EPE). Three-tier stack: C++20 engine → Python FastAPI server → SvelteKit dashboard.

---

## Quick Start (5 commands)

```bash
# 1. Start the database
docker compose up db -d

# 2. Copy and configure secrets
cp .env.example .env        # edit JWT_SECRET at minimum

# 3. Sync Python deps + run migrations + start server
./scripts/run_dev.sh --skip-build

# 4. (separate terminal) Start the frontend dev server
cd web && npm install && npm run dev
# → http://localhost:5173

# 5. (optional) Seed demo counterparties + simulations
python scripts/seed_demo_data.py
```

Default login: **admin / admin123** — change on first use.

---

## Full Docker Stack (production-like)

```bash
# Build the frontend first
cd web && npm install && npm run build && cd ..

# Build + start all services
docker compose up --build

# Seed demo data (run once)
python scripts/seed_demo_data.py
```

→ Dashboard at **http://localhost:8000**

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker + Compose | v2.x | For PostgreSQL / TimescaleDB |
| Python | 3.9 | Must match the pre-built `.so` |
| uv | latest | `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | ≥ 18 | Frontend build only |
| C++ toolchain | GCC 12+ / Clang 15+ | Only needed to rebuild the engine |

> **Note:** The pre-built `_ccr_engine.cpython-39-darwin.so` targets **macOS arm64 (Apple Silicon)**. On other platforms, rebuild with `./scripts/build_engine.sh --bindings`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser (SvelteKit SPA)                                         │
│  • Dark financial terminal UI  • Chart.js PFE/EPE charts         │
│  • Live price ticks (WebSocket)  • Role-gated views              │
└──────────────┬──────────────────────────────┬────────────────────┘
               │ REST /api/v1/*               │ WebSocket /ws/*
┌──────────────▼──────────────────────────────▼────────────────────┐
│  FastAPI Server (Python 3.9 + uvicorn)                           │
│  • JWT auth (ADMIN / RISK_MANAGER / AUDITOR)                     │
│  • Counterparty / Portfolio / Derivative CRUD                    │
│  • Simulation orchestration → calls C++ engine via pybind11      │
│  • Market data pipeline (yfinance + FRED API)                    │
│  • APScheduler: market refresh every 15 min, auto-rerun hourly   │
│  • PDF/CSV export (ReportLab)                                     │
│  • Margin call detection + email alerts (aiosmtplib)             │
│  • Audit log (append-only hypertable)                            │
└──────────────┬───────────────────────────────────────────────────┘
               │ pybind11 (.so)
┌──────────────▼───────────────────────────────────────────────────┐
│  C++20 Monte Carlo Engine                                        │
│  • xoroshiro128/AOX PRNG  (2^64 independent streams)            │
│  • Wichura AS241 normal quantile                                 │
│  • Banachiewicz Cholesky for correlation                         │
│  • GBM path simulation  (SoA layout, branch-free hot loop)      │
│  • PFE via nth_element  •  EPE average  •  CVA Kahan summation  │
│  • Wrong-way risk  •  Jump-at-default  •  SIMD dispatch         │
│    (AVX-512 / AVX2 / NEON / scalar — zero runtime overhead)     │
└──────────────────────────────────────────────────────────────────┘
               │ asyncpg
┌──────────────▼───────────────────────────────────────────────────┐
│  PostgreSQL 16 + TimescaleDB 2.x                                 │
│  • risk_metrics   (hypertable — PFE/EPE/CVA per run)            │
│  • audit_log      (hypertable — append-only event stream)       │
│  • price_history  (hypertable — tick prices)                    │
│  • users / counterparties / portfolios / derivatives / ...      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Market Data

| Input | Source | Status |
|---|---|---|
| Equity spot prices (SPY, AAPL, MSFT, GS, JPM) | yfinance (15-min delayed) | **Real** |
| FX rates (EURUSD, GBPUSD, USDJPY) | yfinance | **Real** |
| Commodity futures (GC=F, CL=F, NG=F) | yfinance | **Real** |
| 30-day rolling historical volatility | Computed from yfinance log-returns | **Real (derived)** |
| Risk-free rate / SOFR | FRED API (free key required) | **Real** |
| Credit spreads / CDS hazard rates | No free source available | **Mocked** |
| Live tick stream (WebSocket `/ws/prices`) | GBM walk seeded from yfinance price | **Mocked — labeled "Demo Ticks"** |

The mock tick generator labels all streamed prices as *"Demo Ticks — GBM simulation, not real market data"* in both the WebSocket payload and the UI.

---

## Roles

| Role | Permissions |
|---|---|
| `ADMIN` | Full access: user management, all CRUD, all reads |
| `RISK_MANAGER` | Simulate, CRUD for entities, acknowledge/settle margin calls, market refresh |
| `AUDITOR` | All GET endpoints only — no simulation, no writes |

---

## API Reference

All endpoints require `Authorization: Bearer <token>` except `/api/v1/auth/login`.

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/auth/login` | — | OAuth2 login → JWT |
| `GET` | `/api/v1/auth/me` | Any | Current user profile |
| `GET` | `/api/v1/auth/users` | ADMIN | List all users |
| `POST` | `/api/v1/auth/register` | ADMIN | Create user |
| `PUT` | `/api/v1/auth/users/{id}` | ADMIN | Update role / active |
| `GET` | `/api/v1/health` | — | Liveness probe + engine info |
| `POST` | `/api/v1/simulate` | RM / ADMIN | Run simulation, persist, check margin |
| `GET` | `/api/v1/simulate/history` | Any | Past simulation results |
| `POST` | `/api/v1/simulate/compare` | Any | Side-by-side run comparison |
| `GET` | `/api/v1/simulate/{id}/export/pdf` | Any | Download PDF report |
| `GET` | `/api/v1/simulate/{id}/export/csv` | Any | Download PFE/EPE CSV |
| `GET` | `/api/v1/counterparties` | Any | List counterparties |
| `POST` | `/api/v1/counterparties` | RM / ADMIN | Create counterparty |
| `GET` | `/api/v1/counterparties/{id}` | Any | Get with portfolios |
| `PUT` | `/api/v1/counterparties/{id}` | RM / ADMIN | Update |
| `DELETE` | `/api/v1/counterparties/{id}` | RM / ADMIN | Delete |
| `GET` | `/api/v1/portfolios` | Any | List portfolios |
| `POST` | `/api/v1/portfolios` | RM / ADMIN | Create portfolio |
| `POST` | `/api/v1/portfolios/{id}/derivatives` | RM / ADMIN | Add derivative |
| `DELETE` | `/api/v1/portfolios/{pid}/derivatives/{did}` | RM / ADMIN | Remove derivative |
| `GET` | `/api/v1/margin-calls` | Any | List margin calls |
| `PUT` | `/api/v1/margin-calls/{id}/acknowledge` | RM / ADMIN | Acknowledge |
| `PUT` | `/api/v1/margin-calls/{id}/settle` | RM / ADMIN | Settle |
| `GET` | `/api/v1/margin-calls/export/csv` | Any | Bulk CSV export |
| `GET` | `/api/v1/market/prices` | Any | Current market params (60 s cache) |
| `GET` | `/api/v1/market/prices/{symbol}/history` | Any | Price history |
| `POST` | `/api/v1/market/refresh` | RM / ADMIN | Trigger immediate refresh |
| `GET` | `/api/v1/audit-log` | ADMIN / AUDITOR | Query audit trail |
| `WS` | `/ws/simulate` | Token (first msg) | Streaming simulation progress |
| `WS` | `/ws/prices` | Token (first msg) | Live price tick stream |

Interactive docs: **http://localhost:8000/docs**

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | **Yes** | — | `postgresql+asyncpg://user:pw@host:5432/db` |
| `JWT_SECRET` | **Yes** | — | Long random string for signing JWTs |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | `480` | Token lifetime (8 hours) |
| `CORS_ORIGINS` | No | `http://localhost:5173` | Comma-separated allowed origins |
| `FRED_API_KEY` | No | — | Free key from fred.stlouisfed.org — enables real rate data |
| `SMTP_HOST` | No | — | SMTP server — enables margin call email alerts |
| `SMTP_PORT` | No | `587` | SMTP port |
| `SMTP_USER` | No | — | SMTP username |
| `SMTP_PASSWORD` | No | — | SMTP password |
| `SMTP_FROM` | No | `ccr-alerts@example.com` | From address for alerts |
| `DEBUG_SQL` | No | — | Set to `true` to log all SQL statements |

---

## Building the C++ Engine

The pre-built `.so` is for macOS arm64 + Python 3.9. To rebuild:

```bash
# Auto-detect SIMD target
./scripts/build_engine.sh --bindings

# Force a specific SIMD target
./scripts/build_engine.sh --bindings --arch avx2    # avx512 | avx2 | neon | scalar

# Debug build
./scripts/build_engine.sh --bindings --debug

# Clean rebuild
./scripts/build_engine.sh --bindings --clean
```

Output `.so` lands in `server/bindings/`.

---

## Development Scripts

```bash
# Build C++ + start server (with hot-reload)
./scripts/run_dev.sh

# Skip C++ rebuild (bindings already built)
./scripts/run_dev.sh --skip-build

# Also build the SvelteKit frontend and serve it via FastAPI
./scripts/run_dev.sh --skip-build --build-web

# Custom port
./scripts/run_dev.sh --skip-build --port 8080
```

---

## Project Structure

```
ccr/
├── engine/                  C++20 Monte Carlo engine
│   ├── include/ccr/         Public headers (types.hpp, ccr_engine.hpp, …)
│   └── src/                 Implementations
├── server/                  FastAPI application
│   ├── api/                 REST routes + WebSocket endpoints
│   ├── auth/                JWT security + RBAC
│   ├── bindings/            pybind11 .so + Python glue layer
│   ├── core/                DB session, config, TTL cache, scheduler, engine runner
│   ├── market_data/         yfinance, FRED, mock tick generator
│   ├── models/              SQLAlchemy ORM models + Pydantic schemas
│   ├── notifications/       Email alerts (aiosmtplib) + audit log
│   ├── reports/             PDF/CSV export (ReportLab)
│   ├── alembic/             Database migrations
│   └── logs/                Rotating log files (git-ignored)
├── web/                     SvelteKit TypeScript dashboard
│   └── src/
│       ├── lib/             api.ts, ws-client.ts, state.ts, types.ts, auth.ts
│       ├── components/      charts/ · forms/ · ui/
│       └── routes/          dashboard · simulate · stress · margin-calls
│                            counterparties · reports · admin · login
├── config/                  defaults.toml, CMake modules
├── scripts/                 build_engine.sh, run_dev.sh, seed_demo_data.py
├── docker-compose.yml
└── .env.example
```

---

## Literature

See `LR.md` for the 13 foundational papers behind the algorithmic choices — Andersen-Pykhtin-Sokol PFE, Kahan CVA integration, xoroshiro128 PRNG, Wichura AS241, Banachiewicz Cholesky, SIMD dispatch patterns, and regulatory Basel III/SA-CCR methodology.
