# ============================================================================
# config/cmake/Findpybind11.cmake
#
# Attempts to locate pybind11 in this order:
#   1. Standard CMake find_package (works if pybind11 is installed via pip/conda)
#   2. Python3 sysconfig path (pip install pybind11 --user)
#   3. Vendored copy at engine/third_party/pybind11 (fallback)
#
# Usage in CMakeLists.txt:
#   find_package(pybind11 REQUIRED)
#   pybind11_add_module(ccr_bindings bindings/bindings.cpp)
# ============================================================================

cmake_minimum_required(VERSION 3.15)

# ── Attempt 1: standard CMake config ─────────────────────────────────────────
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11 found via CMake config: ${pybind11_VERSION}")
    return()
endif()

# ── Attempt 2: derive path from Python3 ──────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmake_dir AND EXISTS "${_pybind11_cmake_dir}")
        find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_cmake_dir}")
        message(STATUS "pybind11 found via Python3 sysconfig: ${pybind11_VERSION}")
        return()
    endif()
endif()

# ── Attempt 3: vendored copy ──────────────────────────────────────────────────
set(_vendored "${CMAKE_SOURCE_DIR}/engine/third_party/pybind11")
if(EXISTS "${_vendored}/CMakeLists.txt")
    add_subdirectory("${_vendored}" pybind11_vendored EXCLUDE_FROM_ALL)
    message(STATUS "pybind11 found (vendored): ${_vendored}")
    return()
endif()

# ── Not found ─────────────────────────────────────────────────────────────────
if(pybind11_FIND_REQUIRED)
    message(FATAL_ERROR
        "pybind11 not found. Install it with:\n"
        "  pip install pybind11\n"
        "or place a copy at engine/third_party/pybind11\n"
    )
else()
    message(WARNING "pybind11 not found; Python bindings will not be built.")
endif()
