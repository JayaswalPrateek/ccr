// ============================================================================
// engine/src/normal_variate.cpp
//
// N(0,1) variate generation via inverse CDF.
// Stubs: inv_cdf_exact delegates to erfc-based approximation (< 1e-7 error,
// sufficient for validation). inv_cdf_approx delegates to exact for now.
// Replace both with Wichura rational approximation / Giles piecewise-linear
// respectively during implementation phase.
// ============================================================================

#include "ccr/normal_variate.hpp"
#include "ccr/rng_engine.hpp"
#include "ccr/simd_abstraction.hpp"
#include "ccr/types.hpp"
#include <cmath>
#include <stdexcept>

namespace ccr {

// ─── Scalar inverse CDF ──────────────────────────────────────────────────────

double inv_cdf_exact(double u) noexcept {
    // TODO: replace with Wichura AS241 rational approximation (< 1e-15 error).
    // Stub: adequate Beasley-Springer-Moro approximation.
    if (u <= 0.0) return -8.3;
    if (u >= 1.0) return  8.3;

    static const double a[] = {
        2.50662823884, -18.61500062529,  41.39119773534, -25.44106049637};
    static const double b[] = {
        -8.47351093090, 23.08336743743, -21.06224101826,   3.13082909833};
    static const double c[] = {
        0.3374754822726147,  0.9761690190917186,  0.1607979714918209,
        0.0276438810333863,  0.0038405729373609,  0.0003951896511349,
        0.0000321767881768,  0.0000002888167364,  0.0000003960315187};

    double y = u - 0.5;
    if (std::abs(y) < 0.42) {
        double r = y * y;
        return y * (((a[3]*r+a[2])*r+a[1])*r+a[0]) /
               ((((b[3]*r+b[2])*r+b[1])*r+b[0])*r+1.0);
    }
    double r = (y < 0.0) ? u : 1.0 - u;
    r = std::log(-std::log(r));
    double x = c[0] + r*(c[1]+r*(c[2]+r*(c[3]+r*(c[4]+r*(c[5]+r*(c[6]+r*(c[7]+r*c[8])))))));
    return (y < 0.0) ? -x : x;
}

double inv_cdf_approx(double u) noexcept {
    // TODO: replace with piecewise-linear branch-free lookup (Giles & Sheridan-Methven 2023).
    // Stub delegates to exact for correctness during scaffolding.
    return inv_cdf_exact(u);
}

// ─── Bulk fill ───────────────────────────────────────────────────────────────

template <typename Arch>
void fill_normal(std::span<double> out, Xoroshiro128aox& rng, SimMode mode) noexcept {
    // Generate uniforms into the same buffer, then transform in-place.
    // TODO: SIMD-vectorise the approx path (Arch::WIDTH elements in parallel).
    const bool approx = (mode == SimMode::APPROX_FAST);
    for (double& z : out) {
        const double u = rng.next_double();
        z = approx ? inv_cdf_approx(u) : inv_cdf_exact(u);
    }
}

// ─── Explicit instantiations ─────────────────────────────────────────────────

template void fill_normal<ScalarArch>(std::span<double>, Xoroshiro128aox&, SimMode) noexcept;
#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template void fill_normal<Avx2Arch>(std::span<double>, Xoroshiro128aox&, SimMode) noexcept;
#endif
#if defined(CCR_ARCH_AVX512)
template void fill_normal<Avx512Arch>(std::span<double>, Xoroshiro128aox&, SimMode) noexcept;
#endif
#if defined(CCR_ARCH_NEON)
template void fill_normal<NeonArch>(std::span<double>, Xoroshiro128aox&, SimMode) noexcept;
#endif

} // namespace ccr
