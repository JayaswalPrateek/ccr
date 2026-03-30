// ============================================================================
// engine/src/correlation_engine.cpp
//
// Cholesky decomposition and correlation transform.
// Stub: identity decomposition always succeeds; apply() is a no-op copy.
// Replace with LAPACK/custom Cholesky (dpotrf) in implementation phase.
// ============================================================================

#include "ccr/correlation_engine.hpp"
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace ccr {

// ─── Constructor ─────────────────────────────────────────────────────────────

CholeskyMatrix::CholeskyMatrix(const std::vector<double>& corr_matrix, int K)
    : K_(K)
    , row_stride_(((static_cast<std::size_t>(K) * sizeof(double) + ARENA_ALIGNMENT - 1)
                   / ARENA_ALIGNMENT) * (ARENA_ALIGNMENT / sizeof(double)))
    , data_(row_stride_ * static_cast<std::size_t>(K), 0.0)
{
    if (static_cast<int>(corr_matrix.size()) != K * K)
        throw std::invalid_argument("corr_matrix size != K*K");

    // TODO: replace with Cholesky-Banachiewicz or LAPACK dpotrf.
    // Stub: copy lower triangle from input, assuming it is already the Cholesky
    // factor (identity matrix passes through unchanged).
    for (int i = 0; i < K; ++i)
        for (int j = 0; j <= i; ++j)
            data_[i * row_stride_ + j] = corr_matrix[i * K + j];
}

// ─── Static factories ────────────────────────────────────────────────────────

CholeskyMatrix CholeskyMatrix::identity(int K) {
    std::vector<double> eye(K * K, 0.0);
    for (int i = 0; i < K; ++i) eye[i * K + i] = 1.0;
    return CholeskyMatrix(eye, K);
}

CholeskyMatrix CholeskyMatrix::wwr_2x2(double rho) {
    // L = [[1, 0], [rho, sqrt(1-rho^2)]]
    const double sqrt_term = std::sqrt(1.0 - rho * rho);
    return CholeskyMatrix({1.0, 0.0, rho, sqrt_term}, 2);
}

// ─── Accessor ────────────────────────────────────────────────────────────────

double CholeskyMatrix::at(int i, int j) const noexcept {
    if (j > i) return 0.0;
    return data_[static_cast<std::size_t>(i) * row_stride_ + static_cast<std::size_t>(j)];
}

// ─── Correlation application ─────────────────────────────────────────────────

template <typename Arch>
void CholeskyMatrix::apply(
    std::span<const double> independent,
    std::span<double>       correlated,
    std::size_t             M_padded,
    int                     K) const noexcept
{
    // TODO: vectorise inner loop over M_padded paths using SimdOps<Arch>.
    // Stub: scalar matrix-vector multiply per path.
    for (std::size_t m = 0; m < M_padded; ++m) {
        for (int i = 0; i < K; ++i) {
            double acc = 0.0;
            for (int j = 0; j <= i; ++j)
                acc += at(i, j) * independent[static_cast<std::size_t>(j) * M_padded + m];
            correlated[static_cast<std::size_t>(i) * M_padded + m] = acc;
        }
    }
}

// ─── Explicit instantiations ─────────────────────────────────────────────────

template void CholeskyMatrix::apply<ScalarArch>(
    std::span<const double>, std::span<double>, std::size_t, int) const noexcept;
#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template void CholeskyMatrix::apply<Avx2Arch>(
    std::span<const double>, std::span<double>, std::size_t, int) const noexcept;
#endif
#if defined(CCR_ARCH_AVX512)
template void CholeskyMatrix::apply<Avx512Arch>(
    std::span<const double>, std::span<double>, std::size_t, int) const noexcept;
#endif
#if defined(CCR_ARCH_NEON)
template void CholeskyMatrix::apply<NeonArch>(
    std::span<const double>, std::span<double>, std::size_t, int) const noexcept;
#endif

} // namespace ccr
