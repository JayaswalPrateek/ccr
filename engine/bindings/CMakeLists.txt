# ============================================================================
# engine/bindings/CMakeLists.txt
#
# Python extension module: _ccr_engine
# Import in Python as:  import _ccr_engine as ccr
# The server-side engine_client.py wraps this with a friendlier API.
#
# Naming: leading underscore follows C extension convention for private C
# backing modules.  Avoids CMake target-name collision with ccr_engine (static).
# ============================================================================

pybind11_add_module(_ccr_engine bindings.cpp)

target_link_libraries(_ccr_engine
    PRIVATE
        ccr_engine          # the static library defined in engine/CMakeLists.txt
)

target_include_directories(_ccr_engine
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include   # engine/include (→ ccr/*.hpp)
        ${CMAKE_CURRENT_SOURCE_DIR}/../src       # engine/src    (private headers)
)

target_compile_features(_ccr_engine PRIVATE cxx_std_20)

# Place the .so next to the Python server for easy PYTHONPATH resolution.
set_target_properties(_ccr_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/server/bindings"
)
