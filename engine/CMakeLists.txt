# ============================================================================
# engine/CMakeLists.txt
#
# Chunk 2: ccr_engine is a STATIC library.
#          All .cpp stubs exist; libccr_engine.a links cleanly.
# ============================================================================

set(CCR_ENGINE_SOURCES
    src/simd_abstraction.cpp
    src/rng_engine.cpp
    src/normal_variate.cpp
    src/correlation_engine.cpp
    src/path_simulator.cpp
    src/exposure_engine.cpp
    src/quantile_extractor.cpp
    src/cva_integrator.cpp
    src/jump_diffusion.cpp
    src/time_grid.cpp
    src/ccr_engine.cpp
)

# ─── STATIC library ───────────────────────────────────────────────────────────
add_library(ccr_engine STATIC ${CCR_ENGINE_SOURCES})

target_include_directories(ccr_engine
    PUBLIC
        # Headers live at engine/include/ccr/*.hpp
        # so #include "ccr/types.hpp" resolves correctly.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # Private headers (memory_arena.hpp) in src/
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(ccr_engine
    PUBLIC
        ccr_compile_options
        OpenMP::OpenMP_CXX
)

target_compile_features(ccr_engine PUBLIC cxx_std_20)

# ─── Tests ───────────────────────────────────────────────────────────────────
# add_subdirectory(tests)

# ─── Benchmarks ──────────────────────────────────────────────────────────────
# add_subdirectory(benchmarks)

# ─── Python bindings (Chunk 3) ───────────────────────────────────────────────
if(CCR_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

install(DIRECTORY include/ccr DESTINATION include)
