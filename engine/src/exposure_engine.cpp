// ============================================================================
// engine/src/exposure_engine.cpp
//
// E(t) = max(V(t), 0) for each path.
// Stub: scalar loop with std::max.
// Replace inner loop with SimdOps<Arch>::max(v, zero) in implementation phase.
// ============================================================================

#include "ccr/exposure_engine.hpp"
#include <algorithm>

namespace ccr {

// ─── Single timestep ─────────────────────────────────────────────────────────

template <typename Arch>
void compute_exposures_step(
    std::span<const double> portfolio_values,
    std::span<double>       exposures_col) noexcept
{
    // TODO: replace with SimdOps<Arch>::max(load(pv), zero()) store.
    const std::size_t n = portfolio_values.size();
    for (std::size_t m = 0; m < n; ++m)
        exposures_col[m] = portfolio_values[m] > 0.0 ? portfolio_values[m] : 0.0;
}

// ─── Full exposure matrix ─────────────────────────────────────────────────────

template <typename Arch>
void compute_exposures_full(
    std::span<const double> portfolio_values,
    std::span<double>       exposures,
    int T, int M_padded) noexcept
{
    for (int t = 0; t < T; ++t) {
        std::span<const double> pv_col{
            portfolio_values.data() + static_cast<std::size_t>(t) * M_padded,
            static_cast<std::size_t>(M_padded)};
        std::span<double> exp_col{
            exposures.data() + static_cast<std::size_t>(t) * M_padded,
            static_cast<std::size_t>(M_padded)};
        compute_exposures_step<Arch>(pv_col, exp_col);
    }
}

// ─── Collateral-adjusted (Phase 2 stub) ──────────────────────────────────────

template <typename Arch>
void compute_exposures_collateralised(
    std::span<const double> portfolio_values,
    std::span<const double> collateral_values,
    std::span<double>       exposures_col,
    int M_padded) noexcept
{
    // TODO: Phase 2 — subtract collateral with MPoR look-back.
    for (int m = 0; m < M_padded; ++m) {
        double net = portfolio_values[m] - collateral_values[m];
        exposures_col[m] = net > 0.0 ? net : 0.0;
    }
}

// ─── Explicit instantiations ─────────────────────────────────────────────────

template void compute_exposures_step<ScalarArch>(
    std::span<const double>, std::span<double>) noexcept;
template void compute_exposures_full<ScalarArch>(
    std::span<const double>, std::span<double>, int, int) noexcept;
template void compute_exposures_collateralised<ScalarArch>(
    std::span<const double>, std::span<const double>, std::span<double>, int) noexcept;

#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template void compute_exposures_step<Avx2Arch>(
    std::span<const double>, std::span<double>) noexcept;
template void compute_exposures_full<Avx2Arch>(
    std::span<const double>, std::span<double>, int, int) noexcept;
template void compute_exposures_collateralised<Avx2Arch>(
    std::span<const double>, std::span<const double>, std::span<double>, int) noexcept;
#endif

#if defined(CCR_ARCH_AVX512)
template void compute_exposures_step<Avx512Arch>(
    std::span<const double>, std::span<double>) noexcept;
template void compute_exposures_full<Avx512Arch>(
    std::span<const double>, std::span<double>, int, int) noexcept;
template void compute_exposures_collateralised<Avx512Arch>(
    std::span<const double>, std::span<const double>, std::span<double>, int) noexcept;
#endif

#if defined(CCR_ARCH_NEON)
template void compute_exposures_step<NeonArch>(
    std::span<const double>, std::span<double>) noexcept;
template void compute_exposures_full<NeonArch>(
    std::span<const double>, std::span<double>, int, int) noexcept;
template void compute_exposures_collateralised<NeonArch>(
    std::span<const double>, std::span<const double>, std::span<double>, int) noexcept;
#endif

} // namespace ccr
