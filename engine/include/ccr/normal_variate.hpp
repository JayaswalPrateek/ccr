#pragma once

// ============================================================================
// ccr/normal_variate.hpp  —  N(0,1) variate generation (Module 3)
//
// Two code paths:
//   EXACT   — Wichura's rational approximation, < 1e-15 relative error.
//             Default for SimMode::REGULATORY and SimMode::STANDARD.
//
//   APPROX  — Piecewise-linear on geometrically-decaying intervals.
//             Branch-free via bit-manipulation lookup (Giles & Sheridan-Methven).
//             7× faster than Intel MKL; may introduce bias at 99.9th pctile.
//             Only for SimMode::APPROX_FAST.
//
// Reference: Giles & Sheridan-Methven, ACM Trans. Mathematical Software, 2023.
//
// Dependencies: rng_engine, simd_abstraction.
// ============================================================================

#include "ccr/rng_engine.hpp"
#include "ccr/simd_abstraction.hpp"
#include "ccr/types.hpp"
#include <span>

namespace ccr {

// ─── Scalar transforms ───────────────────────────────────────────────────────

/// Exact inverse CDF: u → N(0,1).  Uses Wichura rational approximation.
/// Valid domain: u ∈ (0, 1).  Returns ±8.3 for u near 0/1.
double inv_cdf_exact(double u) noexcept;

/// Approximate inverse CDF: piecewise-linear, branch-free.
/// Faster but may introduce systematic bias for |z| > 5.
/// The engine logs a warning when this path is active.
double inv_cdf_approx(double u) noexcept;

// ─── Bulk vectorised fill ────────────────────────────────────────────────────

/// Fill `out` with N(0,1) variates using the path selected by `mode`.
/// `rng` must be a thread-local instance (not shared).
///
/// Template parameter Arch controls SIMD width.
/// The exact path uses Wichura scalar per-element (no SIMD benefit for rational approx).
/// The approx path uses SIMD lookup tables for WIDTH elements in parallel.
template <typename Arch = ActiveArch>
void fill_normal(
    std::span<double>  out,
    Xoroshiro128aox&   rng,
    SimMode            mode = SimMode::REGULATORY) noexcept;

// ─── Pre-computed per-step volatility factor ─────────────────────────────────

/// σ√(Δt) — computed once outside the hot loop, reused per timestep.
/// Avoids redundant sqrt in the inner loop.
inline double vol_factor(double sigma, double dt) noexcept {
    return sigma * std::sqrt(dt);
}

/// (μ - σ²/2) · Δt — the deterministic GBM drift per step.
inline double drift_factor(double mu, double sigma, double dt) noexcept {
    return (mu - 0.5 * sigma * sigma) * dt;
}

} // namespace ccr
