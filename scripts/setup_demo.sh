#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# CCR Engine — One-shot demo setup
#
# Installs PostgreSQL + TimescaleDB, creates the database, applies schema
# migrations, seeds demo data, and starts the API server.
#
# Prerequisites (install these first):
#   macOS : brew (https://brew.sh)  +  uv (https://docs.astral.sh/uv/)
#   Linux : apt / dnf  +  uv
#   Both  : Node.js 18+ (https://nodejs.org)
#
# Usage:
#   chmod +x scripts/setup_demo.sh
#   ./scripts/setup_demo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}→${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
section() { echo -e "\n${BOLD}$*${NC}"; }
die()     { echo -e "${RED}✗${NC} $*"; exit 1; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

section "CCR Engine — Demo Setup"

# ── 1. Check prerequisites ────────────────────────────────────────────────────
section "[1/7] Checking prerequisites"

command -v uv   >/dev/null 2>&1 || die "uv not found. Install from https://docs.astral.sh/uv/"
command -v node >/dev/null 2>&1 || die "Node.js not found. Install from https://nodejs.org"
command -v npm  >/dev/null 2>&1 || die "npm not found (comes with Node.js)"
info "uv, node, npm — OK"

# ── 2. PostgreSQL + TimescaleDB ───────────────────────────────────────────────
section "[2/7] PostgreSQL + TimescaleDB"

if ! command -v psql >/dev/null 2>&1; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    info "Installing PostgreSQL via Homebrew..."
    brew install postgresql@16 timescaledb/tap/timescaledb
    brew services start postgresql@16
  elif command -v apt-get >/dev/null 2>&1; then
    info "Installing PostgreSQL + TimescaleDB via apt..."
    sudo apt-get install -y postgresql postgresql-client
    # TimescaleDB apt setup
    sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -cs) main' > /etc/apt/sources.list.d/timescaledb.list"
    curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y timescaledb-2-postgresql-16
    sudo timescaledb-tune --quiet --yes
    sudo systemctl restart postgresql
  else
    die "Unsupported OS. Install PostgreSQL 16 + TimescaleDB manually, then re-run."
  fi
  info "PostgreSQL installed"
else
  info "PostgreSQL already installed ($(psql --version | head -1))"
fi

# ── 3. Create database + user ─────────────────────────────────────────────────
section "[3/7] Creating database"

DB_EXISTS=$(psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='ccr'" 2>/dev/null || true)
if [[ "$DB_EXISTS" != "1" ]]; then
  psql -U postgres -c "CREATE USER ccr WITH PASSWORD 'ccr';" 2>/dev/null || true
  psql -U postgres -c "CREATE DATABASE ccr OWNER ccr;" 2>/dev/null || true
  psql -U postgres -d ccr -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" 2>/dev/null || true
  info "Database 'ccr' created"
else
  info "Database 'ccr' already exists"
fi

# ── 4. Create .env ────────────────────────────────────────────────────────────
section "[4/7] Environment configuration"

if [[ ! -f .env ]]; then
  cp .env.example .env
  # Generate a random JWT secret
  JWT_SECRET="ccr-$(openssl rand -hex 12 2>/dev/null || echo 'demo-secret-change-in-production')"
  sed -i.bak "s/JWT_SECRET=.*/JWT_SECRET=${JWT_SECRET}/" .env && rm -f .env.bak
  info ".env created from .env.example"
  warn "Set FRED_API_KEY in .env for live interest rates (free at fred.stlouisfed.org)"
else
  info ".env already exists — skipping"
fi

# ── 5. Python dependencies + schema migrations ────────────────────────────────
section "[5/7] Python dependencies + schema migrations"

info "Installing Python dependencies..."
uv sync --quiet

info "Applying database schema..."
uv run alembic -c server/alembic.ini upgrade head

info "Schema applied"

# ── 6. Seed demo data ─────────────────────────────────────────────────────────
section "[6/7] Seeding demo data"

USER_COUNT=$(psql postgresql://ccr:ccr@localhost:5432/ccr -tAc "SELECT COUNT(*) FROM users;" 2>/dev/null || echo "0")
if [[ "$USER_COUNT" -gt "0" ]]; then
  warn "Database already has data — skipping seed (run 'python scripts/seed_demo_data.py' manually to re-seed)"
else
  info "Running seed script (this takes ~30 seconds — runs Monte Carlo simulations)..."
  uv run python scripts/seed_demo_data.py
  info "Demo data seeded"
fi

# ── 7. Frontend dependencies ──────────────────────────────────────────────────
section "[7/7] Frontend dependencies"

info "Installing Node.js dependencies..."
cd web && npm install --silent && cd ..
info "Frontend ready"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Setup complete. Start the application:${NC}"
echo ""
echo -e "  ${BOLD}API server${NC} (terminal 1):"
echo -e "    ./scripts/run_dev.sh --skip-build"
echo ""
echo -e "  ${BOLD}Web dashboard${NC} (terminal 2):"
echo -e "    cd web && npm run dev"
echo ""
echo -e "  ${BOLD}Then open:${NC}  http://localhost:5173"
echo ""
echo -e "  ${BOLD}Login credentials:${NC}"
echo -e "    admin   / admin123    (full access + user management)"
echo -e "    risk    / risk123     (trading + simulation)"
echo -e "    auditor / auditor123  (read-only + audit log)"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
