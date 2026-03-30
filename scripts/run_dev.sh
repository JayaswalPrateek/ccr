#!/usr/bin/env bash
# =============================================================================
# scripts/run_dev.sh
#
# Full dev-loop: build engine → start Python server → open dashboard.
# Usage:
#   ./scripts/run_dev.sh              # Release build, native SIMD
#   ./scripts/run_dev.sh --skip-build # Skip C++ build (bindings already exist)
#   ./scripts/run_dev.sh --debug      # Debug build
#   ./scripts/run_dev.sh --port 8080  # Custom port (default: 8000)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
SKIP_BUILD=false
BUILD_TYPE="Release"
PORT=8000
ARCH="native"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
    --skip-build)
        SKIP_BUILD=true
        shift
        ;;
    --debug)
        BUILD_TYPE="Debug"
        shift
        ;;
    --port)
        PORT="$2"
        shift 2
        ;;
    --arch)
        ARCH="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

# ── Step 1: Build engine + bindings ──────────────────────────────────────────
if ! $SKIP_BUILD; then
    echo "→ Building C++ engine and Python bindings…"
    "${SCRIPT_DIR}/build_engine.sh" \
        --bindings \
        --arch "${ARCH}" \
        $([ "${BUILD_TYPE}" = "Debug" ] && echo "--debug" || echo "--release")
else
    echo "→ Skipping build (--skip-build)"
fi

# ── Step 2: Verify bindings are present ──────────────────────────────────────
BINDINGS_DIR="${PROJECT_ROOT}/server/bindings"
SO=$(find "${BINDINGS_DIR}" -name "_ccr_engine*.so" 2>/dev/null | head -1)

if [[ -z "${SO}" ]]; then
    echo ""
    echo "✗ No _ccr_engine*.so found in ${BINDINGS_DIR}"
    echo "  Run without --skip-build, or check pybind11 installation."
    exit 1
fi
echo "→ Using bindings: ${SO}"

# ── Step 3: Install Python deps (if requirements.txt present) ─────────────────
REQUIREMENTS="${PROJECT_ROOT}/server/requirements.txt"
if [[ -f "${REQUIREMENTS}" ]]; then
    echo "→ Checking Python dependencies…"
    pip install -q -r "${REQUIREMENTS}"
fi

# ── Step 4: Start FastAPI server ──────────────────────────────────────────────
echo ""
echo "→ Starting server on http://localhost:${PORT}"
echo "  Dashboard: http://localhost:${PORT}/"
echo "  API docs:  http://localhost:${PORT}/docs"
echo "  Press Ctrl-C to stop."
echo ""

cd "${PROJECT_ROOT}/server"
PYTHONPATH="${BINDINGS_DIR}:${PYTHONPATH:-}" \
    uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --reload \
    --reload-dir "${PROJECT_ROOT}/server"
