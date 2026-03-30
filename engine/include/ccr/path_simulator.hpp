#pragma once

// ============================================================================
// ccr/path_simulator.hpp  —  GBM path evolution engine (Module 5)
//
// Implements the discretised GBM SDE:
//   S_{t+Δt} = S_t × exp( (μ − σ²/2)Δt  +  σ√(Δt) · Z )
//
// Memory layout: SoA.  For K assets, M paths, T timesteps:
//   spot_prices[k * M_padded + m]  — current price of asset k, path m
//
// The inner loop is written once as a template on Arch.  All SIMD divergence
// is resolved by SimdOps<Arch> — no #ifdef inside the loop body.
//
// Jump-at-default is applied via JumpDiffusionHook after the full diffusion
// pass, keeping the hot loop branch-free.
//
// Dependencies: simd_abstraction, normal_variate, correlation_engine,
//               jump_diffusion, time_grid, types.
// ============================================================================

#include "ccr/simd_abstraction.hpp"
#include "ccr/normal_variate.hpp"
#include "ccr/correlation_engine.hpp"
#include "ccr/jump_diffusion.hpp"
#include "ccr/time_grid.hpp"
#include "ccr/types.hpp"
#include <span>
#include <memory>

namespace ccr {

// ─── PathState — SoA working memory ─────────────────────────────────────────

/// View into the shared memory arena owned by CcrEngine.
/// All spans point into the arena — PathState does not own memory.
struct PathState {
    std::span<double> spot_prices;      ///< [K][M_padded]  — current spot prices
    std::span<double> normals_buf;      ///< [K][M_padded]  — raw N(0,1) shocks
    std::span<double> correlated_buf;   ///< [K][M_padded]  — post-Cholesky shocks
    std::span<double> portfolio_values; ///< [M_padded]     — V(t) per path
    std::span<double> exposures;        ///< [T][M_padded]  — column-major
    std::span<double> pfe_profile;      ///< [T]
    std::span<double> epe_profile;      ///< [T]
    int K;          ///< Number of assets
    int M;          ///< Requested path count
    int M_padded;   ///< M rounded up to ActiveArch::WIDTH
    int T;          ///< Number of timesteps
};

// ─── PathSimulator ───────────────────────────────────────────────────────────

class PathSimulator {
public:
    PathSimulator(
        const SimParams&      params,
        const CholeskyMatrix& cholesky,
        const TimeGrid&       time_grid,
        JumpDiffusionHook*    jump_hook = nullptr);  ///< nullable — no-op if nullptr

    // ── Core simulation step ─────────────────────────────────────────────────

    /// Evolve all M paths by ONE timestep dt using pre-generated correlated normals.
    ///
    /// Called in sequence by CcrEngine::run() for t = 0..T-1.
    /// `correlated_normals` is a [K][M_padded] SoA buffer.
    ///
    /// Template Arch is fixed at compile time for the entire run.
    template <typename Arch = ActiveArch>
    void evolve_step(
        PathState&              state,
        std::span<const double> correlated_normals,
        double                  drift,    ///< (μ − σ²/2)Δt — pre-computed
        double                  vol_dt,   ///< σ√Δt — pre-computed
        int                     timestep) noexcept;

    // ── Full run helper ──────────────────────────────────────────────────────

    /// Evolve all T steps in sequence.  Normals are generated internally from
    /// `rng` using the active SIMD path. Writes exposures into state.exposures.
    template <typename Arch = ActiveArch>
    void run_all_steps(
        PathState&       state,
        Xoroshiro128aox& rng) noexcept;

private:
    SimParams             params_;
    const CholeskyMatrix& cholesky_;
    const TimeGrid&       time_grid_;
    JumpDiffusionHook*    jump_hook_;
};

} // namespace ccr
