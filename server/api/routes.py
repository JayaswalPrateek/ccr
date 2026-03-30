"""REST API routes for the CCR engine."""

from __future__ import annotations

import _ccr_engine as _ccr
from fastapi import APIRouter, HTTPException

from server.bindings.engine_client import build_engine_config
from server.core.engine_runner import run_simulation, engine_info
from server.models.schemas import SimulationRequest, SimulationResponse

router = APIRouter(prefix="/api/v1", tags=["simulation"])


@router.get("/health")
async def health():
    """Liveness probe — also returns engine architecture info."""
    return {"status": "ok", "engine": engine_info()}


@router.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest) -> SimulationResponse:
    """
    Run a CCR Monte Carlo simulation.

    Returns CVA, EPE and PFE profiles, and optionally stressed results.
    """
    err = _ccr.CcrEngine.validate_config(build_engine_config(request))
    if err:
        raise HTTPException(status_code=422, detail=err)

    result = await run_simulation(request)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error_msg)

    return result
