"""
WebSocket endpoints.

/ws/simulate  — streams Monte Carlo progress then delivers the final result.

Requires a JWT token sent as the first message:
    {"token": "<bearer-token>"}
The connection is closed with code 4001 if auth fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from server.auth.security import decode_token
from server.core.database import AsyncSessionLocal
from server.core.engine_runner import run_simulation
from server.models.db_models import User
from server.models.schemas import SimulationRequest

logger    = logging.getLogger(__name__)
ws_router = APIRouter(tags=["websocket"])


# ── Auth helper ───────────────────────────────────────────────────────────────

async def _authenticate_ws(ws: WebSocket) -> Optional[User]:
    """Read the first message, validate the JWT, return the User or None."""
    try:
        raw  = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        data = json.loads(raw)
        token = data.get("token", "").removeprefix("Bearer ").strip()
        if not token:
            return None
        payload = decode_token(token)
        user_id = payload.get("sub", "")
        if not user_id:
            return None
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            return user if user and user.is_active else None
    except Exception:
        return None


# ── /ws/simulate ──────────────────────────────────────────────────────────────

@ws_router.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    await ws.accept()
    try:
        # ── Auth ─────────────────────────────────────────────────────────────
        user = await _authenticate_ws(ws)
        if user is None:
            await ws.close(code=4001, reason="Unauthorized")
            return

        raw     = await ws.receive_text()
        request = SimulationRequest.model_validate_json(raw)
        loop    = asyncio.get_running_loop()

        def progress_cb(timestep: int, total: int, pfe_so_far: float):
            msg = json.dumps({
                "type":       "progress",
                "timestep":   timestep,
                "total":      total,
                "pfe_so_far": pfe_so_far,
                "pct":        round(100.0 * (timestep + 1) / max(total, 1), 1),
            })
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)

        result = await run_simulation(request, progress_cb)

        await ws.send_text(json.dumps({
            "type":   "result",
            "result": result.model_dump(),
        }))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            pass


