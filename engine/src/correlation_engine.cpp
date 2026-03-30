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

    // Cholesky-Banachiewicz decomposition: A = L * L^T.
    // Reads from the symmetric input corr_matrix and writes into data_ (lower triangle).
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = corr_matrix[static_cast<std::size_t>(i) * K + j];
            for (int k = 0; k < j; ++k)
                sum -= data_[i * row_stride_ + k] * data_[j * row_stride_ + k];
            if (i == j) {
                if (sum <= 0.0)
                    throw std::invalid_argument(
                        "correlation matrix is not positive-definite");
                data_[i * row_stride_ + j] = std::sqrt(sum);
            } else {
                data_[i * row_stride_ + j] = sum / data_[j * row_stride_ + j];
            }
        }
    }
}

// ─── Static factories ────────────────────────────────────────────────────────

CholeskyMatrix CholeskyMatrix::identity(int K) {
    std::vector<double> eye(K * K, 0.0);
    for (int i = 0; i < K; ++i) eye[i * K + i] = 1.0;
    return CholeskyMatrix(eye, K);
}

CholeskyMatrix CholeskyMatrix::wwr_2x2(double rho) {
    // Correlation matrix [[1, rho], [rho, 1]] — constructor factorises it.
    // Resulting Cholesky factor: L = [[1, 0], [rho, sqrt(1-rho²)]].
    return CholeskyMatrix({1.0, rho, rho, 1.0}, 2);
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
    // SIMD over the M_padded (path) dimension.
    // For each output row i, accumulate: correlated[i][m] = Σ_j L[i][j] * x[j][m].
    // Scalar Cholesky coefficients L[i][j] are broadcast via set1.
    // M_padded is guaranteed a multiple of Arch::WIDTH.

    const std::size_t step = Arch::WIDTH;

    for (int i = 0; i < K; ++i) {
        for (std::size_t m = 0; m < M_padded; m += step) {
            auto acc = SimdOps<Arch>::zero();
            for (int j = 0; j <= i; ++j) {
                auto L_ij = SimdOps<Arch>::set1(at(i, j));
                auto x_j  = SimdOps<Arch>::load(
                    independent.data() + static_cast<std::size_t>(j) * M_padded + m);
                acc = SimdOps<Arch>::fmadd(L_ij, x_j, acc);
            }
            SimdOps<Arch>::store(
                correlated.data() + static_cast<std::size_t>(i) * M_padded + m, acc);
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
