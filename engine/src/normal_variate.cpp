// ============================================================================
// engine/src/normal_variate.cpp
//
// N(0,1) variate generation via inverse CDF (Acklam / Wichura AS241).
//
// inv_cdf_exact — three-region piecewise rational polynomial.
//   Max absolute error < 1.15e-9 (regulatory-grade).
//   Central region (|u−0.5| ≤ 0.0775): degree-5/5 rational in q² and q.
//   Tail regions (u < 0.02425, u > 0.97575): degree-5/4 rational in sqrt(−2 ln u).
//
// inv_cdf_approx — currently delegates to exact.
//   TODO: replace with Giles & Sheridan-Methven (2023) piecewise-linear
//   SIMD path for APPROX_FAST mode.
//
// fill_normal<Arch> — two-phase: SIMD fill_uniform then scalar CDF inversion.
// ============================================================================

#include "ccr/normal_variate.hpp"
#include "ccr/rng_engine.hpp"
#include "ccr/simd_abstraction.hpp"
#include "ccr/types.hpp"
#include <cmath>

namespace ccr {

// ─── Scalar inverse CDF ──────────────────────────────────────────────────────

double inv_cdf_exact(double u) noexcept {
    // Rational approximation (Acklam 2002 / Wichura AS241 variant).
    // Three-region piecewise rational polynomial: max absolute error < 1.15e-9.
    // Regulatory-grade accuracy for Gaussian quantile inversion.

    if (u <= 0.0) return -8.3;
    if (u >= 1.0) return  8.3;

    // Breakpoint separating central region from tails.
    static constexpr double P_LOW  = 0.02425;
    static constexpr double P_HIGH = 1.0 - P_LOW;

    // Coefficients for central region |p - 0.5| <= (0.5 - P_LOW).
    static constexpr double A1 = -3.969683028665376e+01;
    static constexpr double A2 =  2.209460984245205e+02;
    static constexpr double A3 = -2.759285104469687e+02;
    static constexpr double A4 =  1.383577518672690e+02;
    static constexpr double A5 = -3.066479806614716e+01;
    static constexpr double A6 =  2.506628277459239e+00;

    static constexpr double B1 = -5.447609879822406e+01;
    static constexpr double B2 =  1.615858368580409e+02;
    static constexpr double B3 = -1.556989798598866e+02;
    static constexpr double B4 =  6.680131188771972e+01;
    static constexpr double B5 = -1.328068155288572e+01;

    // Coefficients for tail regions p < P_LOW or p > P_HIGH.
    static constexpr double C1 = -7.784894002430293e-03;
    static constexpr double C2 = -3.223964580411365e-01;
    static constexpr double C3 = -2.400758277161838e+00;
    static constexpr double C4 = -2.549732539343734e+00;
    static constexpr double C5 =  4.374664141464968e+00;
    static constexpr double C6 =  2.938163982698783e+00;

    static constexpr double D1 =  7.784695709041462e-03;
    static constexpr double D2 =  3.224671290700398e-01;
    static constexpr double D3 =  2.445134137142996e+00;
    static constexpr double D4 =  3.754408661907416e+00;

    if (u < P_LOW) {
        // Lower tail: r = sqrt(-2 * ln(p)).
        const double r = std::sqrt(-2.0 * std::log(u));
        return (((((C1*r+C2)*r+C3)*r+C4)*r+C5)*r+C6) /
               ((((D1*r+D2)*r+D3)*r+D4)*r+1.0);
    }

    if (u <= P_HIGH) {
        // Central region.
        const double q = u - 0.5;
        const double r = q * q;
        return (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6) * q /
               (((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1.0);
    }

    // Upper tail: symmetric to lower tail.
    const double r = std::sqrt(-2.0 * std::log(1.0 - u));
    return -((((( C1*r+C2)*r+C3)*r+C4)*r+C5)*r+C6) /
             ((((D1*r+D2)*r+D3)*r+D4)*r+1.0);
}

double inv_cdf_approx(double u) noexcept {
    // TODO: replace with piecewise-linear branch-free lookup (Giles & Sheridan-Methven 2023).
    // Stub delegates to exact for correctness during scaffolding.
    return inv_cdf_exact(u);
}

// ─── Bulk fill ───────────────────────────────────────────────────────────────

template <typename Arch>
void fill_normal(std::span<double> out, Xoroshiro128aox& rng, SimMode mode) noexcept {
    // Step 1: fill buffer with uniforms using the SIMD RNG path.
    // For SIMD archs this uses WIDTH parallel generators; for Scalar it is a
    // plain loop.  The two-phase design decouples SIMD RNG from the scalar CDF.
    rng.fill_uniform<Arch>(out);

    // Step 2: transform uniforms to N(0,1) in-place (scalar CDF inversion).
    // The REGULATORY/STANDARD path uses Wichura AS241 (< 1.15e-9 error).
    // APPROX_FAST delegates to inv_cdf_approx (same impl for now; a SIMD
    // piecewise-linear CDF is the natural Chunk 3 extension if needed).
    const bool approx = (mode == SimMode::APPROX_FAST);
    for (double& z : out)
        z = approx ? inv_cdf_approx(z) : inv_cdf_exact(z);
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
