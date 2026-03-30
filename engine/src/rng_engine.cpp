// ============================================================================
// engine/src/rng_engine.cpp
//
// xoroshiro128aox pseudorandom number generator (Hanlon & Felix 2022).
//
// Output function: (s0 & s1) | rotl(s0 ^ s1, 7)  — AND-OR-XOR scrambler.
// State update: standard xoroshiro128 twist (a=24, b=16, c=37).
// Period: 2^128 − 1.  Jump polynomial advances 2^64 steps (disjoint streams).
//
// fill_uniform<Arch> specialisations:
//   ScalarArch  — plain scalar loop (reference)
//   NeonArch    — 2 parallel generators in uint64x2_t (WIDTH = 2)
//   Avx2Arch    — 4 parallel generators in __m256i    (WIDTH = 4)
//   Avx512Arch  — 8 parallel generators in __m512i    (WIDTH = 8)
// All SIMD paths maintain independent sub-streams separated by jump().
// ============================================================================

#include "ccr/rng_engine.hpp"
#include <cstdint>

namespace ccr {

// ─── Internal helpers ────────────────────────────────────────────────────────

/// splitmix64 — used to expand a single seed into two independent 64-bit words.
static uint64_t splitmix64(uint64_t& z) noexcept {
    z += 0x9e3779b97f4a7c15ULL;
    uint64_t r = z;
    r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
    r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
    return r ^ (r >> 31);
}

/// Rotate left helper.
static inline uint64_t rotl(uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
}

// ─── Constructors ────────────────────────────────────────────────────────────

Xoroshiro128aox::Xoroshiro128aox(uint64_t s0, uint64_t s1) noexcept
    : s_{s0, s1}
{}

Xoroshiro128aox::Xoroshiro128aox(uint64_t seed) noexcept {
    // Expand single seed via splitmix64 to avoid the zero-state trap.
    s_[0] = splitmix64(seed);
    s_[1] = splitmix64(seed);
}

// ─── Core generator ──────────────────────────────────────────────────────────

uint64_t Xoroshiro128aox::next_u64() noexcept {
    const uint64_t s0 = s_[0];
    uint64_t       s1 = s_[1];

    // AOX output function from Hanlon & Felix (2022): AND-OR-XOR scrambler.
    // Provides better statistical quality than the + (xoroshiro128+) scrambler.
    const uint64_t result = (s0 & s1) | rotl(s0 ^ s1, 7);

    s1 ^= s0;
    s_[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s_[1] = rotl(s1, 37);                    // c

    return result;
}

double Xoroshiro128aox::next_double() noexcept {
    // Extract upper 53 bits → uniform [0, 1) with 1 ULP resolution.
    return static_cast<double>(next_u64() >> 11) * (1.0 / (UINT64_C(1) << 53));
}

// ─── Jump functions ──────────────────────────────────────────────────────────

void Xoroshiro128aox::jump() noexcept {
    uint64_t s0 = 0, s1 = 0;
    for (int b = 0; b < 2; ++b) {
        uint64_t poly = JUMP_POLY[b];
        for (int bit = 0; bit < 64; ++bit) {
            if (poly & (UINT64_C(1) << bit)) {
                s0 ^= s_[0];
                s1 ^= s_[1];
            }
            next_u64();
        }
    }
    s_[0] = s0;
    s_[1] = s1;
}

void Xoroshiro128aox::long_jump() noexcept {
    uint64_t s0 = 0, s1 = 0;
    for (int b = 0; b < 2; ++b) {
        uint64_t poly = LONG_JUMP_POLY[b];
        for (int bit = 0; bit < 64; ++bit) {
            if (poly & (UINT64_C(1) << bit)) {
                s0 ^= s_[0];
                s1 ^= s_[1];
            }
            next_u64();
        }
    }
    s_[0] = s0;
    s_[1] = s1;
}

// ─── Bulk generation ─────────────────────────────────────────────────────────

// Scalar fallback — used by ScalarArch and as tail handler.
template <typename Arch>
void Xoroshiro128aox::fill_uniform(std::span<double> out) noexcept {
    for (double& d : out) d = next_double();
}

void Xoroshiro128aox::fill_raw(std::span<uint64_t> out) noexcept {
    for (uint64_t& u : out) u = next_u64();
}

// ── NEON specialisation (WIDTH = 2) ──────────────────────────────────────────
// Two independent generators (separated by 2^64 jump) run in parallel.
// Lane 0 tracks the main stream; its state is saved back to s_ after the fill.
// Lane 1 runs a non-overlapping stream 2^64 steps ahead — statistically disjoint.
// Output order: [gen0[0], gen1[0], gen0[1], gen1[1], ...] (interleaved pairs).
#if defined(CCR_ARCH_NEON)
template <>
void Xoroshiro128aox::fill_uniform<NeonArch>(std::span<double> out) noexcept {
    const std::size_t n  = out.size();
    const std::size_t n2 = n & ~std::size_t{1};  // round down to even

    if (n2 >= 2) {
        // Second generator: start from same state, then jump 2^64.
        Xoroshiro128aox rng2(s_[0], s_[1]);
        rng2.jump();

        // Pack both states into uint64x2_t: lane 0 = main, lane 1 = jumped.
        uint64_t vs0_init[2] = {s_[0],          rng2.state0()};
        uint64_t vs1_init[2] = {s_[1],          rng2.state1()};
        uint64x2_t vs0 = vld1q_u64(vs0_init);
        uint64x2_t vs1 = vld1q_u64(vs1_init);

        // Constant: IEEE 754 exponent for 1.0 (for mantissa injection trick).
        const uint64x2_t one_bits = vreinterpretq_u64_f64(vdupq_n_f64(1.0));
        const float64x2_t one_f   = vdupq_n_f64(1.0);

        for (std::size_t i = 0; i < n2; i += 2) {
            // AOX output: (s0 & s1) | rotl(s0 ^ s1, 7)
            uint64x2_t xor01 = veorq_u64(vs0, vs1);
            uint64x2_t and01 = vandq_u64(vs0, vs1);
            uint64x2_t rot7  = vorrq_u64(vshlq_n_u64(xor01, 7),
                                          vshrq_n_u64(xor01, 57));
            uint64x2_t raw   = vorrq_u64(and01, rot7);

            // State update (xoroshiro128 twist, applied to both lanes in parallel).
            vs1              = veorq_u64(vs1, vs0);
            uint64x2_t rot24 = vorrq_u64(vshlq_n_u64(vs0, 24),
                                          vshrq_n_u64(vs0, 40));
            vs0 = veorq_u64(veorq_u64(rot24, vs1), vshlq_n_u64(vs1, 16));
            vs1 = vorrq_u64(vshlq_n_u64(vs1, 37), vshrq_n_u64(vs1, 27));

            // Convert raw uint64 to double in [0, 1):
            // Use upper 53 bits as mantissa of 1.x, then subtract 1.0.
            uint64x2_t mant = vorrq_u64(vshrq_n_u64(raw, 11), one_bits);
            float64x2_t d   = vsubq_f64(vreinterpretq_f64_u64(mant), one_f);

            vst1q_f64(&out[i], d);
        }

        // Save lane 0 state back (lane 1 is an independent stream, discarded).
        uint64_t s0_out[2], s1_out[2];
        vst1q_u64(s0_out, vs0);
        vst1q_u64(s1_out, vs1);
        s_[0] = s0_out[0];
        s_[1] = s1_out[0];
    }

    // Scalar tail for odd-length spans.
    for (std::size_t i = n2; i < n; ++i) out[i] = next_double();
}
#endif

// ── AVX2 specialisation (WIDTH = 4) ──────────────────────────────────────────
// Four independent generators in __m256i (4 × 64-bit lanes).
#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template <>
void Xoroshiro128aox::fill_uniform<Avx2Arch>(std::span<double> out) noexcept {
    const std::size_t n  = out.size();
    const std::size_t n4 = (n / 4) * 4;

    if (n4 >= 4) {
        // Build 3 additional generators via successive jumps.
        Xoroshiro128aox rng2(s_[0], s_[1]); rng2.jump();
        Xoroshiro128aox rng3(rng2.state0(), rng2.state1()); rng3.jump();
        Xoroshiro128aox rng4(rng3.state0(), rng3.state1()); rng4.jump();

        __m256i vs0 = _mm256_set_epi64x(
            (int64_t)rng4.state0(), (int64_t)rng3.state0(),
            (int64_t)rng2.state0(), (int64_t)s_[0]);
        __m256i vs1 = _mm256_set_epi64x(
            (int64_t)rng4.state1(), (int64_t)rng3.state1(),
            (int64_t)rng2.state1(), (int64_t)s_[1]);

        const __m256i one_bits = _mm256_castpd_si256(_mm256_set1_pd(1.0));
        const __m256d one_f    = _mm256_set1_pd(1.0);

        // Helper lambdas for emulated 64-bit rotate left.
        auto rotl_n = [](__m256i x, int n) -> __m256i {
            return _mm256_or_si256(_mm256_slli_epi64(x, n),
                                   _mm256_srli_epi64(x, 64 - n));
        };

        for (std::size_t i = 0; i < n4; i += 4) {
            __m256i xor01 = _mm256_xor_si256(vs0, vs1);
            __m256i and01 = _mm256_and_si256(vs0, vs1);
            __m256i raw   = _mm256_or_si256(and01, rotl_n(xor01, 7));

            vs1 = _mm256_xor_si256(vs1, vs0);
            vs0 = _mm256_xor_si256(
                    _mm256_xor_si256(rotl_n(vs0, 24), vs1),
                    _mm256_slli_epi64(vs1, 16));
            vs1 = rotl_n(vs1, 37);

            __m256i mant = _mm256_or_si256(_mm256_srli_epi64(raw, 11), one_bits);
            __m256d d    = _mm256_sub_pd(_mm256_castsi256_pd(mant), one_f);

            _mm256_store_pd(&out[i], d);
        }

        // Extract lane 0 state.
        alignas(32) int64_t s0_out[4], s1_out[4];
        _mm256_store_si256((__m256i*)s0_out, vs0);
        _mm256_store_si256((__m256i*)s1_out, vs1);
        s_[0] = (uint64_t)s0_out[0];
        s_[1] = (uint64_t)s1_out[0];
    }

    for (std::size_t i = n4; i < n; ++i) out[i] = next_double();
}
#endif

// ── AVX-512 specialisation (WIDTH = 8) ───────────────────────────────────────
#if defined(CCR_ARCH_AVX512)
template <>
void Xoroshiro128aox::fill_uniform<Avx512Arch>(std::span<double> out) noexcept {
    const std::size_t n  = out.size();
    const std::size_t n8 = (n / 8) * 8;

    if (n8 >= 8) {
        // 8 independent generators via successive jumps.
        // Cannot use an array (no default constructor) — use named locals.
        Xoroshiro128aox g0(s_[0], s_[1]);
        Xoroshiro128aox g1(g0.state0(), g0.state1()); g1.jump();
        Xoroshiro128aox g2(g1.state0(), g1.state1()); g2.jump();
        Xoroshiro128aox g3(g2.state0(), g2.state1()); g3.jump();
        Xoroshiro128aox g4(g3.state0(), g3.state1()); g4.jump();
        Xoroshiro128aox g5(g4.state0(), g4.state1()); g5.jump();
        Xoroshiro128aox g6(g5.state0(), g5.state1()); g6.jump();
        Xoroshiro128aox g7(g6.state0(), g6.state1()); g7.jump();

        // _mm512_set_epi64 is big-endian (last arg → lane 0).
        __m512i vs0 = _mm512_set_epi64(
            (int64_t)g7.state0(), (int64_t)g6.state0(),
            (int64_t)g5.state0(), (int64_t)g4.state0(),
            (int64_t)g3.state0(), (int64_t)g2.state0(),
            (int64_t)g1.state0(), (int64_t)g0.state0());
        __m512i vs1 = _mm512_set_epi64(
            (int64_t)g7.state1(), (int64_t)g6.state1(),
            (int64_t)g5.state1(), (int64_t)g4.state1(),
            (int64_t)g3.state1(), (int64_t)g2.state1(),
            (int64_t)g1.state1(), (int64_t)g0.state1());

        const __m512i one_bits = _mm512_castpd_si512(_mm512_set1_pd(1.0));
        const __m512d one_f    = _mm512_set1_pd(1.0);

        auto rotl_n = [](__m512i x, int n) -> __m512i {
            return _mm512_or_si512(_mm512_slli_epi64(x, n),
                                   _mm512_srli_epi64(x, 64 - n));
        };

        for (std::size_t i = 0; i < n8; i += 8) {
            __m512i xor01 = _mm512_xor_si512(vs0, vs1);
            __m512i and01 = _mm512_and_si512(vs0, vs1);
            __m512i raw   = _mm512_or_si512(and01, rotl_n(xor01, 7));

            vs1 = _mm512_xor_si512(vs1, vs0);
            vs0 = _mm512_xor_si512(
                    _mm512_xor_si512(rotl_n(vs0, 24), vs1),
                    _mm512_slli_epi64(vs1, 16));
            vs1 = rotl_n(vs1, 37);

            __m512i mant = _mm512_or_si512(_mm512_srli_epi64(raw, 11), one_bits);
            __m512d d    = _mm512_sub_pd(_mm512_castsi512_pd(mant), one_f);

            _mm512_store_pd(&out[i], d);
        }

        alignas(64) int64_t s0_out[8], s1_out[8];
        _mm512_store_si512(s0_out, vs0);
        _mm512_store_si512(s1_out, vs1);
        s_[0] = (uint64_t)s0_out[0];
        s_[1] = (uint64_t)s1_out[0];
    }

    for (std::size_t i = n8; i < n; ++i) out[i] = next_double();
}
#endif

// ─── Explicit instantiations ─────────────────────────────────────────────────
// Only ScalarArch uses the generic template; all SIMD archs have explicit
// specializations above and do not need (or allow) separate instantiation.

template void Xoroshiro128aox::fill_uniform<ScalarArch>(std::span<double>) noexcept;

// ─── Factory ─────────────────────────────────────────────────────────────────

std::vector<Xoroshiro128aox> make_thread_rngs(uint64_t base_seed, int num_threads) {
    std::vector<Xoroshiro128aox> rngs;
    rngs.reserve(num_threads);
    rngs.emplace_back(base_seed);          // first stream
    for (int i = 1; i < num_threads; ++i) {
        // Construct from current tail state, then advance 2^64 steps.
        const auto& prev = rngs.back();
        Xoroshiro128aox next(prev.state0(), prev.state1());
        next.jump();
        rngs.push_back(std::move(next));
    }
    return rngs;
}

} // namespace ccr
