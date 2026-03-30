#pragma once

// ============================================================================
// ccr/correlation_engine.hpp  —  Cholesky correlation module (Module 4)
//
// Computes the Cholesky decomposition of the K×K cross-asset correlation
// matrix once per simulation run, then applies it to transform K independent
// N(0,1) shocks into correlated shocks.
//
// For WWR: the hazard-rate shock is coupled to asset shocks via the factor
// model:  ε_hazard = ρ · ε_asset + √(1 − ρ²) · ε_r
// This is encoded as an augmented (K+1)×(K+1) correlation matrix where the
// last row/column represents the hazard-rate factor.
//
// The Cholesky matrix is stored as a dense lower-triangular matrix with
// SIMD-aligned rows (padding to ARENA_ALIGNMENT bytes per row).
//
// Dependencies: simd_abstraction (for aligned allocation + vectorised multiply).
// ============================================================================

#include "ccr/simd_abstraction.hpp"
#include <span>
#include <vector>

namespace ccr {

// ─── Cholesky matrix ─────────────────────────────────────────────────────────

/// Dense lower-triangular Cholesky factor L such that Σ = L · Lᵀ.
/// Rows are ARENA_ALIGNMENT-byte aligned for SIMD loads.
/// After construction, the matrix is treated as const (read by all threads).
class CholeskyMatrix {
public:
    /// Construct from a K×K symmetric positive-definite correlation matrix
    /// provided in row-major order (K² elements).
    /// Throws std::invalid_argument if the matrix is not positive-definite.
    CholeskyMatrix(const std::vector<double>& corr_matrix, int K);

    /// Convenience: identity matrix for K factors (no correlation).
    static CholeskyMatrix identity(int K);

    /// Convenience: 2×2 correlation matrix for single-asset + hazard-rate WWR.
    ///   [  1   ρ ]
    ///   [  ρ   1 ]
    static CholeskyMatrix wwr_2x2(double rho);

    // Non-copyable (owns aligned memory); movable.
    CholeskyMatrix(const CholeskyMatrix&)            = delete;
    CholeskyMatrix& operator=(const CholeskyMatrix&) = delete;
    CholeskyMatrix(CholeskyMatrix&&)                 = default;
    CholeskyMatrix& operator=(CholeskyMatrix&&)      = default;

    // ── Application ──────────────────────────────────────────────────────────

    /// Transform M×K independent shocks → M×K correlated shocks.
    /// `independent` and `correlated` are both SoA layout: factor k occupies
    /// rows [k*M_padded, (k+1)*M_padded).
    ///
    /// Implementation: matrix-vector multiply per path, vectorised over paths.
    template <typename Arch = ActiveArch>
    void apply(
        std::span<const double> independent,  // [K][M_padded]
        std::span<double>       correlated,   // [K][M_padded]
        std::size_t             M_padded,
        int                     K) const noexcept;

    // ── Accessors ────────────────────────────────────────────────────────────

    int    dim()         const noexcept { return K_; }
    double at(int i, int j) const noexcept;  ///< L[i][j], lower-triangular

private:
    int                  K_;       ///< Number of factors
    std::size_t          row_stride_; ///< Padded row width in doubles
    std::vector<double>  data_;    ///< Aligned storage (row_stride_ × K_ doubles)
};

} // namespace ccr
