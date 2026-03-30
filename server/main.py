"""
CCR Engine — FastAPI application entry point.

Run with:
    uvicorn server.main:app --reload --port 8000

Or via the dev script:
    ./scripts/run_dev.sh
"""

from __future__ import annotations

import sys
import os

# Ensure the pybind11 shared library is on the path.
_bindings_dir = os.path.join(os.path.dirname(__file__), "bindings")
if _bindings_dir not in sys.path:
    sys.path.insert(0, _bindings_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.routes import router
from server.api.websocket import ws_router
from server.core.engine_runner import engine_info

app = FastAPI(
    title="CCR Engine API",
    description=(
        "Counterparty Credit Risk & XVA computation engine for OTC derivatives. "
        "Computes PFE, CVA, and EPE via Monte Carlo simulation (C++20 + SIMD backend)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "service": "CCR Engine API",
        "docs":    "/docs",
        "engine":  engine_info(),
    }
