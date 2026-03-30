# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CCR Engine is a Counterparty Credit Risk (CCR) & XVA computation engine for OTC derivatives. It computes Potential Future Exposure (PFE), Credit Valuation Adjustment (CVA), and Expected Positive Exposure (EPE) via Monte Carlo simulation. The stack is three-tier: C++20 engine → Python FastAPI server → TypeScript web frontend.

## Commands

### C++ Engine

```bash
# Release build (auto-detects SIMD: AVX-512, AVX2, NEON)
./scripts/build_engine.sh

# Debug build
./scripts/build_engine.sh --debug

# With Python pybind11 bindings
./scripts/build_engine.sh --bindings

# Force a specific SIMD target
./scripts/build_engine.sh --arch avx2   # or avx512, neon, scalar

# Clean rebuild
./scripts/build_engine.sh --clean

# Control parallelism
./scripts/build_engine.sh --jobs 8
```

### Full Dev Loop (C++ + Python server)

```bash
# Build engine + bindings, install Python deps, start FastAPI server
./scripts/run_dev.sh

# Skip C++ rebuild (faster iteration on Python/server)
./scripts/run_dev.sh --skip-build

# Custom port (default 8000)
./scripts/run_dev.sh --port 8080

# Debug build of engine
./scripts/run_dev.sh --debug
```

### Manual CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release [-DCCR_BUILD_BINDINGS=ON]
cmake --build build --parallel
```

### Tests & Benchmarks

Test and benchmark subdirectories are currently commented out in `engine/CMakeLists.txt`. Re-enable by uncommenting `add_subdirectory(tests)` and `add_subdirectory(benchmarks)`.

## Architecture

### C++ Engine (`engine/`)

Implements a four-stage Monte Carlo pipeline:

1. **Scenario Generation** — time-grid construction (`time_grid`), PRNG seeding (`rng_engine`: xoroshiro128aox, 2^64 independent streams), Cholesky decomposition (`correlation_engine`)
2. **Path Simulation** — GBM evolution (`path_simulator`): branch-free hot loop, SoA memory layout
3. **Portfolio Valuation** — exposure = max(V, 0) (`exposure_engine`), PFE quantile extraction via `std::nth_element` (`quantile_extractor`), EPE averaging
4. **XVA Integration** — CVA with Kahan summation (`cva_integrator`), wrong-way risk jump-at-default (`jump_diffusion`)

**Orchestration**: `CcrEngine` in `engine/include/ccr/ccr_engine.hpp` — single entry point `CcrEngine::run(EngineConfig, callback) → RiskMetrics`.

**Key design principles**:
- **Policy-based SIMD dispatch**: `Arch` template parameter selects AVX-512/AVX2/NEON/scalar at compile time — zero runtime overhead (`simd_abstraction.hpp`)
- **Structure-of-Arrays**: all hot-path data is SoA for cache locality
- **Single memory arena**: one contiguous aligned allocation; no heap allocations inside the hot loop
- **Deterministic reproducibility**: Kahan summation, fixed-position quantile extraction, IEEE 754 semantics — required for regulatory compliance

**Shared types** (`engine/include/ccr/types.hpp`): `SimParams` (paths, timesteps, assets, mode, grid type, WWR correlation) and `RiskMetrics` (PFE/EPE profiles, CVA, margin).

### Python Server (`server/`)

FastAPI + Uvicorn wrapping the C++ extension module (`_ccr_engine` pybind11 binding).

- `server/main.py` — ASGI app entry point
- `server/api/` — REST endpoint definitions
- `server/core/` — engine runner, scheduler, caching
- `server/bindings/` — Python ↔ C++ glue
- `server/models/` — Pydantic schemas
- API docs at `http://localhost:8000/docs` when running

### Web Frontend (`web/`)

TypeScript dashboard (framework TBD — check `web/` for `package.json`).

- `web/src/lib/` — API client, WebSocket client, state management
- `web/src/components/` — UI components
- `web/src/workers/` — Web Workers for off-main-thread work
- WebSocket endpoint used for real-time Monte Carlo progress updates

## Development Status

The project is built in incremental chunks:
- **Chunks 1–2** (complete): C++ engine core and static library
- **Chunk 3** (in progress): Python bindings, FastAPI server, web dashboard

Many server-side modules are scaffolded stubs awaiting implementation. The compiled library `build/engine/libccr_engine.a` is present in the repo.

## Key References

`LR.md` contains an extensive literature review covering the 13 foundational papers behind the algorithmic and numerical choices. Consult it before modifying simulation modes, quantile methods, or SIMD paths.
