#pragma once

// ============================================================================
// ccr/exposure_engine.hpp  —  Exposure computation (Module 6)
//
// Computes E(t) = max(V(t), 0) for each path at each timestep.
// Branch-free via _mm256_max_pd / vmaxq_f64 / std::max.
//
// Memory regime: MEMORY-BOUND (writing the T×M exposure matrix).
// SIMD delivers high ROI only on the max itself; the bottleneck is cache
// bandwidth writing exposures[t][0..M].
//
// The exposure matrix is stored COLUMN-MAJOR: exposures[t * M_padded .. +M_padded]
// so that quantile extraction at timestep t is a single contiguous sweep.
//
// Dependencies: simd_abstraction.
// ============================================================================

#include "ccr/simd_abstraction.hpp"
#include <span>

namespace ccr {

// ─── Single-timestep exposure ────────────────────────────────────────────────

/// Compute E_i = max(V_i, 0) for all M_padded paths at one timestep.
///
/// `portfolio_values`  input  — [M_padded] portfolio MTM values
/// `exposures_col`     output — [M_padded] positive exposures for this timestep
///
/// Both spans must be ARENA_ALIGNMENT-byte aligned.
template <typename Arch = ActiveArch>
void compute_exposures_step(
    std::span<const double> portfolio_values,  // [M_padded]
    std::span<double>       exposures_col      // [M_padded]
) noexcept;

// ─── Full exposure matrix ────────────────────────────────────────────────────

/// Compute the full T × M_padded exposure matrix in one call.
///
/// `portfolio_values`  — [T][M_padded] — V(t,m) for all timesteps and paths
/// `exposures`         — [T][M_padded] — output, column-major
///
/// Equivalent to calling compute_exposures_step() T times.
template <typename Arch = ActiveArch>
void compute_exposures_full(
    std::span<const double> portfolio_values,  // [T * M_padded]
    std::span<double>       exposures,         // [T * M_padded]
    int                     T,
    int                     M_padded) noexcept;

// ─── Collateral-adjusted exposure (Phase 2+) ─────────────────────────────────

/// E(t) = max( V(t) − CollateralAdjusted(t − MPoR), 0 )
///
/// Placeholder signature.  Collateral logic is reserved for Phase 2.
/// Currently equivalent to compute_exposures_step with zero collateral.
template <typename Arch = ActiveArch>
void compute_exposures_collateralised(
    std::span<const double> portfolio_values,  // [M_padded]
    std::span<const double> collateral_values, // [M_padded] — zero in Phase 1
    std::span<double>       exposures_col,     // [M_padded]
    int                     M_padded) noexcept;

} // namespace ccr
