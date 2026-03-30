#pragma once

// ============================================================================
// ccr/simd_abstraction.hpp  —  Compile-time SIMD policy layer
//
// Design rule: the hot simulation loop is a single template parameterised on
// Arch. NO #ifdef blocks inside the loop body. All platform divergence is
// resolved here at the SimdOps<Arch> specialisation level.
//
// Dependencies: nothing (this is the foundation module).
// ============================================================================

#include <cstddef>
#include <cmath>
#include <algorithm>

// ─── Architecture detection macros ──────────────────────────────────────────
// Detected via compiler predefined macros; never set manually in user code.

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define CCR_ARCH_AVX512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define CCR_ARCH_AVX2 1
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define CCR_ARCH_NEON 1
#else
#  define CCR_ARCH_SCALAR 1
#endif

namespace ccr {

// ─── Architecture policy structs ─────────────────────────────────────────────
// Each describes the register width and native vector type.
// These are zero-size tag types — all cost is at compile time.

struct ScalarArch {
    static constexpr std::size_t WIDTH = 1;
    using reg_t                        = double;
    static constexpr const char* NAME  = "Scalar";
};

#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
struct Avx2Arch {
    static constexpr std::size_t WIDTH = 4;
    using reg_t                        = __m256d;
    static constexpr const char* NAME  = "AVX2";
};
#endif

#if defined(CCR_ARCH_AVX512)
struct Avx512Arch {
    static constexpr std::size_t WIDTH = 8;
    using reg_t                        = __m512d;
    static constexpr const char* NAME  = "AVX-512";
};
#endif

#if defined(CCR_ARCH_NEON)
struct NeonArch {
    static constexpr std::size_t WIDTH = 2;
    using reg_t                        = float64x2_t;
    static constexpr const char* NAME  = "ARM NEON";
};
#endif

// ─── Active architecture (compile-time selection) ───────────────────────────

#if defined(CCR_ARCH_AVX512)
using ActiveArch = Avx512Arch;
#elif defined(CCR_ARCH_AVX2)
using ActiveArch = Avx2Arch;
#elif defined(CCR_ARCH_NEON)
using ActiveArch = NeonArch;
#else
using ActiveArch = ScalarArch;
#endif

// ─── SimdOps primary template — scalar fallback ───────────────────────────
// Implementations are in simd_abstraction.cpp (for non-templated helpers)
// or inlined here for templated code.
//
// Specialisations for Avx2Arch, Avx512Arch, NeonArch follow below.

template <typename Arch>
struct SimdOps {
    using reg_t = typename Arch::reg_t;

    // Load WIDTH doubles from a WIDTH-aligned address
    static inline reg_t load(const double* ptr);

    // Store WIDTH doubles to a WIDTH-aligned address
    static inline void store(double* ptr, reg_t v);

    // Broadcast scalar to all lanes
    static inline reg_t set1(double v);

    // All-zero register
    static inline reg_t zero();

    // Element-wise arithmetic
    static inline reg_t add(reg_t a, reg_t b);
    static inline reg_t sub(reg_t a, reg_t b);
    static inline reg_t mul(reg_t a, reg_t b);
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c);  // a*b + c

    // Element-wise max (branch-free — maps to _mm256_max_pd / vmaxq_f64)
    static inline reg_t max(reg_t a, reg_t b);

    // exp(x) via 6th-order minimax polynomial, branch-free, < 1 ULP on [-20, 20]
    // See: Cephes-style range reduction → Horner + IEEE 754 exponent bit trick
    static inline reg_t exp_approx(reg_t v);

    // sqrt — used for σ√(Δt) pre-computation outside the hot loop
    static inline reg_t sqrt(reg_t v);

    // Emit VZEROUPPER (AVX) or no-op (NEON/scalar).
    // MUST be called before any non-SIMD function call from the hot loop.
    static inline void fence();
};

// ─── Scalar specialisation (always available, reference impl) ────────────────

template <>
struct SimdOps<ScalarArch> {
    using reg_t = double;

    static inline reg_t load(const double* p)           { return *p; }
    static inline void  store(double* p, reg_t v)       { *p = v; }
    static inline reg_t set1(double v)                  { return v; }
    static inline reg_t zero()                          { return 0.0; }
    static inline reg_t add(reg_t a, reg_t b)           { return a + b; }
    static inline reg_t sub(reg_t a, reg_t b)           { return a - b; }
    static inline reg_t mul(reg_t a, reg_t b)           { return a * b; }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c){ return a * b + c; }
    static inline reg_t max(reg_t a, reg_t b)           { return a > b ? a : b; }
    static inline reg_t sqrt(reg_t v)                   { return std::sqrt(v); }
    static inline void  fence()                         {}

    // Scalar exp_approx delegates to std::exp (exact, slow) — acceptable for
    // the scalar fallback path.  SIMD specialisations use the polynomial.
    static inline reg_t exp_approx(reg_t v) { return std::exp(v); }
};

// ─── AVX2 specialisation ─────────────────────────────────────────────────────
// Declared here; implementations in simd_abstraction.cpp.

#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template <>
struct SimdOps<Avx2Arch> {
    using reg_t = __m256d;

    static inline reg_t load(const double* p)            { return _mm256_load_pd(p); }
    static inline void  store(double* p, reg_t v)        { _mm256_store_pd(p, v); }
    static inline reg_t set1(double v)                   { return _mm256_set1_pd(v); }
    static inline reg_t zero()                           { return _mm256_setzero_pd(); }
    static inline reg_t add(reg_t a, reg_t b)            { return _mm256_add_pd(a, b); }
    static inline reg_t sub(reg_t a, reg_t b)            { return _mm256_sub_pd(a, b); }
    static inline reg_t mul(reg_t a, reg_t b)            { return _mm256_mul_pd(a, b); }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c) { return _mm256_fmadd_pd(a, b, c); }
    static inline reg_t max(reg_t a, reg_t b)            { return _mm256_max_pd(a, b); }
    static inline reg_t sqrt(reg_t v)                    { return _mm256_sqrt_pd(v); }
    static inline void  fence()                          { _mm256_zeroupper(); }

    // Declared here, defined in simd_abstraction.cpp
    static reg_t exp_approx(reg_t v);
};
#endif

// ─── AVX-512 specialisation ──────────────────────────────────────────────────

#if defined(CCR_ARCH_AVX512)
template <>
struct SimdOps<Avx512Arch> {
    using reg_t = __m512d;

    static inline reg_t load(const double* p)            { return _mm512_load_pd(p); }
    static inline void  store(double* p, reg_t v)        { _mm512_store_pd(p, v); }
    static inline reg_t set1(double v)                   { return _mm512_set1_pd(v); }
    static inline reg_t zero()                           { return _mm512_setzero_pd(); }
    static inline reg_t add(reg_t a, reg_t b)            { return _mm512_add_pd(a, b); }
    static inline reg_t sub(reg_t a, reg_t b)            { return _mm512_sub_pd(a, b); }
    static inline reg_t mul(reg_t a, reg_t b)            { return _mm512_mul_pd(a, b); }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c) { return _mm512_fmadd_pd(a, b, c); }
    static inline reg_t max(reg_t a, reg_t b)            { return _mm512_max_pd(a, b); }
    static inline reg_t sqrt(reg_t v)                    { return _mm512_sqrt_pd(v); }
    static inline void  fence()                          { _mm256_zeroupper(); } // still needed

    static reg_t exp_approx(reg_t v); // defined in simd_abstraction.cpp
};
#endif

// ─── ARM NEON specialisation ─────────────────────────────────────────────────

#if defined(CCR_ARCH_NEON)
template <>
struct SimdOps<NeonArch> {
    using reg_t = float64x2_t;

    static inline reg_t load(const double* p)            { return vld1q_f64(p); }
    static inline void  store(double* p, reg_t v)        { vst1q_f64(p, v); }
    static inline reg_t set1(double v)                   { return vdupq_n_f64(v); }
    static inline reg_t zero()                           { return vdupq_n_f64(0.0); }
    static inline reg_t add(reg_t a, reg_t b)            { return vaddq_f64(a, b); }
    static inline reg_t sub(reg_t a, reg_t b)            { return vsubq_f64(a, b); }
    static inline reg_t mul(reg_t a, reg_t b)            { return vmulq_f64(a, b); }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c) { return vfmaq_f64(c, a, b); }
    static inline reg_t max(reg_t a, reg_t b)            { return vmaxq_f64(a, b); }
    static inline reg_t sqrt(reg_t v)                    { return vsqrtq_f64(v); }
    static inline void  fence()                          {} // no-op on ARM

    static reg_t exp_approx(reg_t v); // defined in simd_abstraction.cpp
};
#endif

// ─── Utility: pad path count to next SIMD multiple ───────────────────────────

/// Round M up to the next multiple of Arch::WIDTH.
/// Padded elements are zero-initialised and masked out during reductions.
template <typename Arch = ActiveArch>
constexpr std::size_t pad_to_width(std::size_t m) {
    constexpr std::size_t W = Arch::WIDTH;
    return ((m + W - 1) / W) * W;
}

// ─── Alignment constants ─────────────────────────────────────────────────────

/// Baseline arena alignment.  Satisfies AVX-512 (64 bytes) and x86 cache lines.
/// On Apple Silicon the arena is over-aligned to 128 bytes (see memory_arena.hpp).
#if defined(__APPLE__) && defined(__arm64__)
inline constexpr std::size_t ARENA_ALIGNMENT = 128;
#else
inline constexpr std::size_t ARENA_ALIGNMENT = 64;
#endif

/// Per-thread chunk size must be a multiple of this many doubles to avoid false sharing.
/// 8 doubles × 8 bytes = 64 bytes = one x86 cache line.
inline constexpr std::size_t CACHE_LINE_DOUBLES = 8;

} // namespace ccr
