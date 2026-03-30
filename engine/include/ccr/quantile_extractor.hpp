#pragma once

// ============================================================================
// ccr/quantile_extractor.hpp  —  Quantile extraction (Module 7)
//
// Extracts PFE (α-quantile, typically α=0.99) and EPE (mean of positive
// exposures) from the exposure distribution at each timestep.
//
// Two quantile implementations:
//   STANDARD      — std::nth_element (introselect, O(N) avg).
//                   NOT bitwise reproducible across platforms.
//   DETERMINISTIC — Custom introselect with median-of-three from fixed
//                   positions (first/middle/last).  ~10% slower but bitwise
//                   identical across GCC / Clang / MSVC for the same input.
//                   Set via EngineConfig::deterministic_quantile.
//                   Required for SimMode::REGULATORY.
//
// NOTE: nth_element modifies the exposure column IN-PLACE.
//       The caller must not rely on column order after this call.
//
// Memory regime: MEMORY-BOUND (scanning M values with random-access partitioning).
// PMA variants (Wheatman 2024) are theoretically optimal for streaming inserts
// but not advantageous for bulk static arrays — see Phase 1 design doc §3.4.
//
// Dependencies: none (pure algorithm on contiguous double spans).
// ============================================================================

#include <cstddef>
#include <span>
#include <vector>

namespace ccr {

// ─── Single-timestep extraction ─────────────────────────────────────────────

/// Extract the α-quantile of the exposure distribution at one timestep.
///
/// `exposures_col`  — [M] positive exposures (MODIFIED in-place by nth_element)
/// `alpha`          — quantile level, e.g. 0.99 for PFE-99
/// `deterministic`  — use custom introselect for regulatory reproducibility
///
/// Returns the α-quantile value (e.g. the 100th-largest for M=10000, α=0.99).
double extract_pfe(
	std::span<double> exposures_col,
	double alpha = 0.99,
	bool deterministic = true) noexcept;

/// Extract EPE = mean of strictly positive exposures at one timestep.
/// Does NOT modify the input (read-only scan).
double extract_epe(std::span<const double> exposures_col) noexcept;

// ─── Full-profile extraction ─────────────────────────────────────────────────

/// Extract PFE and EPE profiles across all T timesteps.
///
/// `exposures`   — [T * M_padded] column-major exposure matrix (MODIFIED)
/// `pfe_profile` — [T] output — PFE(t) for each timestep
/// `epe_profile` — [T] output — EPE(t) for each timestep
/// `T`           — number of timesteps
/// `M`           — number of paths (non-padded)
/// `M_padded`    — padded path count
/// `alpha`       — PFE quantile level
/// `deterministic` — use bitwise-reproducible introselect
void extract_profiles(
	std::span<double> exposures,	// [T * M_padded] — modified in-place
	std::span<double> pfe_profile,	// [T] output
	std::span<double> epe_profile,	// [T] output
	int T,
	int M,
	int M_padded,
	double alpha = 0.99,
	bool deterministic = true) noexcept;

// ─── Diagnostic helpers ──────────────────────────────────────────────────────

/// Compute the full exposure histogram for visualisation (does NOT modify input).
/// Returns `num_bins` bin counts for the range [0, max_exposure].
std::vector<std::size_t> exposure_histogram(
	std::span<const double> exposures_col,
	int num_bins = 50,
	double max_exposure = 0.0  ///< 0 = auto-detect from data
	) noexcept;

}  // namespace ccr
