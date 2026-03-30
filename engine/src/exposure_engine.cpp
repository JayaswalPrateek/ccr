// ============================================================================
// engine/src/exposure_engine.cpp
//
// E(t) = max(V(t), 0) for each path — branch-free via SIMD max.
//
// compute_exposures_step<Arch>     — single timestep column (M_padded elements)
// compute_exposures_full<Arch>     — full T × M_padded exposure matrix
// compute_exposures_collateralised — net-of-collateral exposure (Phase 2 stub)
//
// M_padded is guaranteed a multiple of Arch::WIDTH by PathSimulator::pad_to_width,
// so the SIMD loop requires no tail handling.
// ============================================================================

#include "ccr/exposure_engine.hpp"

namespace ccr {

// ─── Single timestep ─────────────────────────────────────────────────────────

template <typename Arch>
void compute_exposures_step(
    std::span<const double> portfolio_values,
    std::span<double>       exposures_col) noexcept
{
    // E(t) = max(V(t), 0) — branch-free via SimdOps<Arch>::max.
    // M_padded is guaranteed to be a multiple of Arch::WIDTH.
    const std::size_t n    = portfolio_values.size();
    const std::size_t step = Arch::WIDTH;
    const auto        zero = SimdOps<Arch>::zero();

    for (std::size_t m = 0; m < n; m += step) {
        auto v = SimdOps<Arch>::load(portfolio_values.data() + m);
        SimdOps<Arch>::store(exposures_col.data() + m, SimdOps<Arch>::max(v, zero));
    }
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
