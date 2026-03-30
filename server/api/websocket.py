"""
WebSocket endpoint for streaming Monte Carlo progress updates.

Client sends a SimulationRequest as JSON.
Server streams progress messages as the C++ engine completes each timestep,
then sends the final SimulationResponse.
"""

from __future__ import annotations

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.core.engine_runner import run_simulation
from server.models.schemas import SimulationRequest

ws_router = APIRouter(tags=["websocket"])


@ws_router.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        request = SimulationRequest.model_validate_json(raw)

        total_steps = request.sim_params.num_timesteps

        # Capture the running event loop now (in the async handler) so the
        # progress callback — which fires from a worker thread — can schedule
        # sends without relying on asyncio.get_event_loop() from a non-async
        # context (deprecated / unreliable in Python ≥ 3.10).
        import asyncio
        loop = asyncio.get_running_loop()

        def progress_cb(timestep: int, total: int, pfe_so_far: float):
            # Called from a ThreadPoolExecutor worker — must not await directly.
            # run_coroutine_threadsafe is thread-safe and returns a Future.
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
