#!/usr/bin/env bash
# =============================================================================
# scripts/build_engine.sh
#
# Configures and builds the C++ CCR engine.
# Usage:
#   ./scripts/build_engine.sh             # Release, native SIMD, header-only (Chunk 1)
#   ./scripts/build_engine.sh --debug     # Debug build
#   ./scripts/build_engine.sh --arch avx2 # Force AVX2 target
#   ./scripts/build_engine.sh --bindings  # Include pybind11 Python bindings (Chunk 3+)
#   ./scripts/build_engine.sh --clean     # Wipe build directory first
#   ./scripts/build_engine.sh --jobs 8    # Parallel jobs (default: nproc)
# =============================================================================

set -euo pipefail

# ── Locate project root (works regardless of where the script is called from) ─
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_TYPE="Release"
CCR_ARCH="native"
BUILD_BINDINGS="OFF"
CLEAN=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
VERBOSE=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)        BUILD_TYPE="Debug"; shift ;;
        --release)      BUILD_TYPE="Release"; shift ;;
        --arch)         CCR_ARCH="$2"; shift 2 ;;
        --bindings)     BUILD_BINDINGS="ON"; shift ;;
        --clean)        CLEAN=true; shift ;;
        --jobs|-j)      JOBS="$2"; shift 2 ;;
        --verbose|-v)   VERBOSE=true; shift ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Print banner ──────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════╗"
echo "║          CCR Engine — Build Script                   ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Build type : ${BUILD_TYPE}"
echo "║  SIMD arch  : ${CCR_ARCH}"
echo "║  Bindings   : ${BUILD_BINDINGS}"
echo "║  Jobs       : ${JOBS}"
echo "║  Build dir  : ${BUILD_DIR}"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Clean ─────────────────────────────────────────────────────────────────────
if $CLEAN && [[ -d "${BUILD_DIR}" ]]; then
    echo "→ Cleaning build directory…"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# ── Configure ─────────────────────────────────────────────────────────────────
echo "→ Configuring…"
cmake \
    -S "${PROJECT_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCCR_ARCH="${CCR_ARCH}" \
    -DCCR_BUILD_BINDINGS="${BUILD_BINDINGS}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "→ Building…"
VERBOSE_FLAG=""
if $VERBOSE; then VERBOSE_FLAG="--verbose"; fi

cmake --build "${BUILD_DIR}" \
    --config "${BUILD_TYPE}" \
    --parallel "${JOBS}" \
    ${VERBOSE_FLAG}

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "✓ Build complete."
echo ""

# Chunk 2+: report library artifact location
LIB="${BUILD_DIR}/engine/libccr_engine.a"
if [[ -f "${LIB}" ]]; then
    echo "  Library  : ${LIB}"
fi

# Chunk 3+: report Python bindings location
SO=$(find "${BUILD_DIR}" -name "_ccr_engine*.so" 2>/dev/null | head -1)
if [[ -n "${SO}" ]]; then
    echo "  Bindings : ${SO}"
    echo ""
    echo "  To use in Python:"
    echo "    export PYTHONPATH=\"$(dirname "${SO}")":\$PYTHONPATH\""
fi
echo ""
