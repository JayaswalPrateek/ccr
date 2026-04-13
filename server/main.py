"""
CCR Engine — FastAPI application entry point.

Run with:
    uvicorn server.main:app --reload --port 8000

Or via the dev script:
    ./scripts/run_dev.sh
"""

from __future__ import annotations

import logging
import logging.config
import os
import subprocess
import sys
import traceback
import warnings

# Suppress urllib3 LibreSSL warning on macOS — LibreSSL is functionally
# compatible for our use; the warning is cosmetic noise.
warnings.filterwarnings("ignore", message=".*urllib3.*LibreSSL.*", category=Warning)
warnings.filterwarnings("ignore", message=".*OpenSSL.*", module="urllib3")
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Ensure the pybind11 shared library is on the path.
_bindings_dir = os.path.join(os.path.dirname(__file__), "bindings")
if _bindings_dir not in sys.path:
    sys.path.insert(0, _bindings_dir)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.api.auth_routes import auth_router
from server.api.entities_routes import entities_router
from server.api.market_routes import market_router
from server.api.preset_routes import preset_router
from server.api.query_routes import query_router
from server.api.routes import router
from server.api.websocket import ws_router
from server.core.config import settings
from server.core.engine_runner import engine_info
from server.core.scheduler import setup_scheduler


# ── Logging ───────────────────────────────────────────────────────────────────

_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d  %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)-8s %(name)s  %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "verbose",
            "filename": str(_log_dir / "ccr.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
    "loggers": {
        "uvicorn.access": {"level": "WARNING"},
        "sqlalchemy.engine": {
            "level": "DEBUG" if os.getenv("DEBUG_SQL") else "WARNING",
        },
        "apscheduler": {"level": "WARNING"},
    },
})

logger = logging.getLogger(__name__)


# ── Startup helpers ───────────────────────────────────────────────────────────

async def _run_migrations() -> None:
    """Run `alembic upgrade head` so the DB schema is always current."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # repo root (where alembic.ini lives)
    )
    if result.returncode != 0:
        logger.error("Alembic migration failed:\n%s", result.stderr)
        raise RuntimeError("Database migration failed — see logs above")
    logger.info("Alembic migrations applied")


async def _seed_admin_if_empty() -> None:
    """Create the default admin user if the users table is empty."""
    from sqlalchemy import func, select

    from server.auth.security import hash_password
    from server.core.database import AsyncSessionLocal
    from server.models.db_models import User

    async with AsyncSessionLocal() as db:
        count_result = await db.execute(select(func.count()).select_from(User))
        if count_result.scalar() == 0:
            admin = User(
                username  = "admin",
                email     = "admin@ccr.local",
                hashed_pw = hash_password("admin123"),
                role      = "ADMIN",
            )
            db.add(admin)
            await db.commit()
            logger.warning(
                "Seeded default admin user (admin / admin123). "
                "CHANGE THIS PASSWORD BEFORE EXPOSING TO A NETWORK."
            )


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Migrations run synchronously in a subprocess — alembic uses its own
    # connection management and does not support asyncpg natively in CLI.
    await _run_migrations()
    await _seed_admin_if_empty()

    # Initial market data fetch on startup (non-fatal if network unavailable).
    try:
        from server.core.database import AsyncSessionLocal
        from server.market_data.fetcher import refresh_market_params
        async with AsyncSessionLocal() as db:
            await refresh_market_params(db)
    except Exception as exc:
        logger.warning("Initial market data fetch failed (non-fatal): %s", exc)

    setup_scheduler()
    logger.info("CCR Engine started — engine arch: %s", engine_info())
    yield

    # Shutdown.
    from server.core.scheduler import scheduler
    scheduler.shutdown(wait=False)
    logger.info("CCR Engine stopped")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CCR Engine API",
    description=(
        "Counterparty Credit Risk & XVA computation engine for OTC derivatives. "
        "Computes PFE, CVA, and EPE via Monte Carlo simulation (C++20 + SIMD backend)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception on %s %s\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.cors_origins_list,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(entities_router)
app.include_router(market_router)
app.include_router(preset_router)
app.include_router(query_router)
app.include_router(router)
app.include_router(ws_router)

# ── Static frontend ───────────────────────────────────────────────────────────
# Serve the SvelteKit static build when it exists.  The mount is added AFTER
# all API routers so that explicit API routes always take priority over the
# catch-all.  The "/" GET below is only reachable in dev (no build present).
_static_web = Path(__file__).parent / "static_web"
if _static_web.exists():
    app.mount("/", StaticFiles(directory=str(_static_web), html=True), name="static")


@app.get("/")
async def root():
    """Dev-mode landing page.  Shadowed by the StaticFiles mount in production."""
    return {
        "service": "CCR Engine API",
        "docs":    "/docs",
        "engine":  engine_info(),
    }
