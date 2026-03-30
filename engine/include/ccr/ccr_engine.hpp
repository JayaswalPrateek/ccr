#pragma once

// ============================================================================
// ccr/ccr_engine.hpp  —  Top-level orchestrator (CcrEngine)
//
// CcrEngine owns the memory arena and coordinates the full pipeline:
//
//   1. Validate config
//   2. Build TimeGrid
//   3. Compute Cholesky decomposition (once)
//   4. Allocate aligned arena (single contiguous block)
//   5. Initialise thread-local RNGs via jump()
//   6. For each timestep t:
//        a. Generate bulk normals (thread-local, SIMD)
//        b. Apply Cholesky correlation
//        c. Evolve GBM paths (SIMD)
//        d. Compute E(t) = max(V(t), 0)
//        e. Store exposure column
//   7. Extract PFE(t) and EPE(t) profiles
//   8. Integrate CVA
//   9. (Optional) apply stress scenario and re-run
//  10. Return CcrResult
//
// Thread safety: one CcrEngine per run.  Not safe to call run() concurrently
// on the same instance.  Create separate instances per thread if needed.
//
// Dependencies: all engine modules.
// ============================================================================

#include "ccr/types.hpp"
#include "ccr/simd_abstraction.hpp"
#include "ccr/time_grid.hpp"
#include "ccr/correlation_engine.hpp"
#include "ccr/path_simulator.hpp"
#include "ccr/exposure_engine.hpp"
#include "ccr/quantile_extractor.hpp"
#include "ccr/cva_integrator.hpp"
#include "ccr/jump_diffusion.hpp"
#include <memory>
#include <functional>
#include <optional>

namespace ccr {

// ─── Progress callback ───────────────────────────────────────────────────────

/// Called by CcrEngine::run() after each completed timestep.
/// @param timestep   0-based index of the completed step
/// @param total      Total number of steps
/// @param pfe_so_far PFE at the just-completed step (0 while still computing)
using ProgressCallback = std::function<void(int timestep, int total, double pfe_so_far)>;

// ─── CcrEngine ───────────────────────────────────────────────────────────────

class CcrEngine {
public:
    explicit CcrEngine() = default;
    ~CcrEngine();  // defined in ccr_engine.cpp where Arena is complete

    // Non-copyable, movable
    CcrEngine(const CcrEngine&)            = delete;
    CcrEngine& operator=(const CcrEngine&) = delete;
    CcrEngine(CcrEngine&&)                 = default;
    CcrEngine& operator=(CcrEngine&&)      = default;

    // ── Main entry point ────────────────────────────────────────────────────

    /// Run the full CCR pipeline for the given config.
    ///
    /// If config.stress is set, a second pass is run with shocked parameters
    /// and stored in CcrResult::stressed.
    ///
    /// @param callback  Optional: called after each timestep for streaming progress.
    CcrResult run(
        const EngineConfig&            config,
        std::optional<ProgressCallback> callback = std::nullopt);

    // ── Validation ──────────────────────────────────────────────────────────

    /// Validate config before run(). Returns empty string on success,
    /// or a human-readable error message.
    static std::string validate_config(const EngineConfig& config);

    // ── Introspection ────────────────────────────────────────────────────────

    /// Return the compile-time SIMD architecture string, e.g. "AVX2".
    static const char* active_arch() noexcept { return ActiveArch::NAME; }

    /// Return the SIMD register width in doubles (1, 2, 4, or 8).
    static constexpr std::size_t simd_width() noexcept { return ActiveArch::WIDTH; }

    /// Estimated arena size in bytes for a given config (for pre-allocation / UI display).
    static std::size_t estimate_arena_bytes(const EngineConfig& config) noexcept;

    // ── Margin call evaluation ───────────────────────────────────────────────

    /// Given a completed run result, produce a MarginCallInfo if the exposure
    /// exceeds the counterparty's margin threshold.
    static std::optional<MarginCallInfo> evaluate_margin_call(
        const CcrResult&           result,
        const CounterpartyConfig&  counterparty);

private:
    // Run a single pass (base or stressed)
    RiskMetrics run_single(
        const EngineConfig&            config,
        const SimParams&               params,
        std::optional<ProgressCallback> callback);

    // Arena memory owned by this engine instance — reset on each run().
    // Defined in memory_arena.hpp (private header, not part of public API).
    // Custom deleter avoids requiring Arena to be complete in this header
    // (pimpl idiom: std::default_delete would static_assert sizeof > 0).
    struct Arena;
    struct ArenaDeleter { void operator()(Arena*) noexcept; };
    std::unique_ptr<Arena, ArenaDeleter> arena_;
};

} // namespace ccr
