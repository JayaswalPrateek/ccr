// ============================================================================
// engine/src/simd_abstraction.cpp
//
// Non-inline SIMD helpers: exp_approx for each SIMD specialisation.
// Stubs delegate to std::exp via lane-by-lane scalar loop.
// Replace with vectorised 6th-order minimax polynomial in implementation phase.
// ============================================================================

#include "ccr/simd_abstraction.hpp"
#include <cmath>

namespace ccr {

// ─── AVX2 / AVX-512 base ─────────────────────────────────────────────────────

#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
auto SimdOps<Avx2Arch>::exp_approx(__m256d v) -> __m256d {
    // TODO: replace with 6th-order minimax polynomial (branch-free, < 1 ULP).
    alignas(32) double buf[4];
    _mm256_store_pd(buf, v);
    for (int i = 0; i < 4; ++i) buf[i] = std::exp(buf[i]);
    return _mm256_load_pd(buf);
}
#endif

// ─── AVX-512 ─────────────────────────────────────────────────────────────────

#if defined(CCR_ARCH_AVX512)
auto SimdOps<Avx512Arch>::exp_approx(__m512d v) -> __m512d {
    alignas(64) double buf[8];
    _mm512_store_pd(buf, v);
    for (int i = 0; i < 8; ++i) buf[i] = std::exp(buf[i]);
    return _mm512_load_pd(buf);
}
#endif

// ─── ARM NEON ────────────────────────────────────────────────────────────────

#if defined(CCR_ARCH_NEON)
auto SimdOps<NeonArch>::exp_approx(float64x2_t v) -> float64x2_t {
    alignas(16) double buf[2];
    vst1q_f64(buf, v);
    buf[0] = std::exp(buf[0]);
    buf[1] = std::exp(buf[1]);
    return vld1q_f64(buf);
}
#endif

} // namespace ccr
