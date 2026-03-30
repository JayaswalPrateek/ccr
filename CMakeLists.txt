cmake_minimum_required(VERSION 3.20)

project(CcrEngine
    VERSION 0.1.0
    DESCRIPTION "Realtime Counterparty Credit Risk & Margin Engine"
    LANGUAGES CXX
)

# ─── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)        # C++20 for std::span; downgrade to 17 if needed
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)     # No GNU extensions — keeps code portable

# ─── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─── Module path for custom Find*.cmake ───────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/config/cmake")

# ─── Global compiler options ─────────────────────────────────────────────────
# Applied via an INTERFACE target so subdirectories can opt in selectively.

add_library(ccr_compile_options INTERFACE)

target_compile_options(ccr_compile_options INTERFACE
    # Release optimisations
    $<$<CONFIG:Release>:
        -O3
        -fno-math-errno          # Allow compiler to treat errno-setting math as no-op
        -ffp-contract=fast       # Enable FMA fusion
        # NOTE: do NOT add -fno-trapping-math for regulatory builds
        #       (IEEE 754 trap semantics must be preserved to detect NaN/Inf)
    >

    # Debug: no optimisation, full symbols, sanitisers
    $<$<CONFIG:Debug>:
        -O0
        -g3
        -fno-omit-frame-pointer
    >

    # Warning flags (all configs)
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter    # Too noisy in stub phase
    >
)

# ─── SIMD options (selectable at configure time) ─────────────────────────────
# Default: -march=native (auto-detect host ISA).
# Override with -DCCR_ARCH=avx2 / avx512 / neon / scalar.

set(CCR_ARCH "native" CACHE STRING "SIMD target: native | avx2 | avx512 | neon | scalar")
set_property(CACHE CCR_ARCH PROPERTY STRINGS native avx2 avx512 neon scalar)

if(CCR_ARCH STREQUAL "native")
    target_compile_options(ccr_compile_options INTERFACE -march=native)
elseif(CCR_ARCH STREQUAL "avx2")
    target_compile_options(ccr_compile_options INTERFACE -mavx2 -mfma)
elseif(CCR_ARCH STREQUAL "avx512")
    target_compile_options(ccr_compile_options INTERFACE -mavx512f -mavx512dq -mfma)
elseif(CCR_ARCH STREQUAL "neon")
    target_compile_options(ccr_compile_options INTERFACE -march=armv8-a+simd)
elseif(CCR_ARCH STREQUAL "scalar")
    # No -march flag — plain scalar, portable across x86 and ARM.
    # Clang on Apple Silicon does not accept -march=x86-64.
else()
    message(FATAL_ERROR "Unknown CCR_ARCH='${CCR_ARCH}'. Use: native avx2 avx512 neon scalar")
endif()
message(STATUS "SIMD target: ${CCR_ARCH}")

# ─── LTO (Link-Time Optimisation) ────────────────────────────────────────────
# Critical for cross-module inlining of SimdTraits specialisations.
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "LTO: enabled")
else()
    message(STATUS "LTO: not supported — ${ipo_msg}")
endif()

# ─── OpenMP ──────────────────────────────────────────────────────────────────
# AppleClang does not ship OpenMP. Homebrew provides it via libomp.
# Install with: brew install libomp
# On Linux / GCC this branch is never taken.

if(APPLE)
    # Try to locate Homebrew libomp before calling find_package
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX AND EXISTS "${HOMEBREW_LIBOMP_PREFIX}")
        message(STATUS "Homebrew libomp found at: ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS
            "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include"
            CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY
            "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib"
            CACHE FILEPATH "" FORCE)
    else()
        message(WARNING
            "Homebrew libomp not found. OpenMP disabled.\n"
            "Install with:  brew install libomp\n"
            "Then re-run cmake."
        )
    endif()
endif()

find_package(OpenMP)   # REQUIRED removed — graceful degradation if missing

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found — simulation will run single-threaded.")
    # Provide a dummy INTERFACE target so engine/CMakeLists.txt links cleanly
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED GLOBAL)
endif()

# ─── pybind11 (optional — required for Python bindings in Chunk 3+) ──────────
option(CCR_BUILD_BINDINGS "Build Python pybind11 bindings" OFF)
if(CCR_BUILD_BINDINGS)
    find_package(pybind11 REQUIRED)
    message(STATUS "pybind11: ${pybind11_VERSION}")
endif()

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(engine)
