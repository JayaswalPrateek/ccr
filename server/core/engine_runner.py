"""
Async engine runner — bridges the FastAPI async event loop and the C++ engine.

The pybind11 binding releases the GIL during CcrEngine.run(), so calling it
from a thread pool executor is safe and non-blocking for the event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import _ccr_engine as _ccr

from server.bindings.engine_client import build_engine_config, result_to_response
from server.models.schemas import SimulationRequest, SimulationResponse

# Single shared executor; C++ is GIL-free so threads run concurrently.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ccr-worker")


def _run_sync(
    request: SimulationRequest,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> SimulationResponse:
    """Blocking call — runs in a worker thread, not the event loop thread."""
    engine = _ccr.CcrEngine()
    config = build_engine_config(request)

    result = engine.run(config, progress_cb)
    return result_to_response(result)


async def run_simulation(
    request: SimulationRequest,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> SimulationResponse:
    """Non-blocking coroutine: runs the C++ engine on the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor,
        _run_sync,
        request,
        progress_cb,
    )


def engine_info() -> dict:
    """Return static engine metadata (architecture, SIMD width)."""
    return {
        "arch":       _ccr.active_arch(),
        "simd_width": _ccr.simd_width(),
    }
