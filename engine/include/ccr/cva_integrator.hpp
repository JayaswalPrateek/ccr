#pragma once

// ============================================================================
// ccr/cva_integrator.hpp  —  CVA / XVA integration (Module 8)
//
// Implements:
//   CVA ≈ (1 − R) · Σ_i  EPE(t_i) · PD(t_{i-1}, t_i)
//
// WWR-adjusted CVA: when ρ > 0, hazard shocks are coupled to asset shocks
// (via CholeskyMatrix), so the EPE(t) already embeds the wrong-way effect.
// The integrator just sums; the coupling happens in PathSimulator.
//
// Numerical note: summation uses Kahan compensated addition regardless of
// parallelism to ensure bitwise reproducibility of CVA across runs.
// CVA integration is SEQUENTIAL (T is small: 12–250 timesteps).
//
// Dependencies: quantile_extractor (for EPE), types.
// ============================================================================

#include "ccr/types.hpp"
#include <span>

namespace ccr {

// ─── CVA integration ────────────────────────────────────────────────────────

/// Compute the baseline CVA scalar.
///
///   CVA = (1 − recovery) · Σ_i  EPE(t_i) · marginal_pd[i]
///
/// Uses Kahan compensated summation for reproducibility.
/// Both spans must have the same length T.
double compute_cva(
    std::span<const double> epe_profile,    ///< EPE(t_i), length T
    std::span<const double> marginal_pd,    ///< PD(t_{i-1}, t_i), length T
    double                  recovery_rate) noexcept;

/// Compute WWR-adjusted CVA.
/// When WWR is enabled, EPE already reflects the ρ-coupled hazard process.
/// This function is identical to compute_cva — provided for semantic clarity
/// and to enable separate reporting of WWR delta.
double compute_wwr_cva(
    std::span<const double> epe_profile,
    std::span<const double> marginal_pd,
    double                  recovery_rate) noexcept;

// ─── PD term structure utilities ────────────────────────────────────────────

/// Derive marginal default probabilities from a flat (constant) hazard rate.
///   PD(t_{i-1}, t_i) = exp(−λ · t_{i-1}) − exp(−λ · t_i)
std::vector<double> marginal_pd_from_flat_hazard(
    std::span<const double> time_grid,   ///< t_0=0, t_1, …, t_T in years
    double                  hazard_rate);

/// Derive marginal PD from a flat hazard rate + stress shock.
std::vector<double> marginal_pd_stressed(
    std::span<const double> time_grid,
    double                  hazard_rate,
    double                  hazard_rate_shock);  ///< additive shock

// ─── Required margin ─────────────────────────────────────────────────────────

/// Suggest optimal VM transfer: max PFE that isn't covered by current collateral.
///   margin_required = max(0, max_t PFE(t) − current_collateral)
double compute_required_margin(
    std::span<const double> pfe_profile,
    double                  current_collateral) noexcept;

} // namespace ccr
