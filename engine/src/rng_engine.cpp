// ============================================================================
// engine/src/rng_engine.cpp
//
// xoroshiro128aox pseudorandom number generator.
// Stub: state machine is functionally correct (uses xoroshiro128+ output as
// placeholder). fill_uniform/fill_raw are scalar loops.
// Replace output function with AND-OR-XOR from Hanlon & Felix (2022) in
// implementation phase.
// ============================================================================

#include "ccr/rng_engine.hpp"
#include <cstdint>
#include <stdexcept>

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

    // TODO: replace with AOX output function from Hanlon & Felix (2022):
    //   result = (s0 & s1) | rotl(s0 ^ s1, 7)   [exact polynomial TBD]
    // Stub uses xoroshiro128+ addition output (statistically adequate for
    // scaffolding; not production-grade).
    const uint64_t result = s0 + s1;

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

template <typename Arch>
void Xoroshiro128aox::fill_uniform(std::span<double> out) noexcept {
    // Stub: scalar loop over all lanes.
    // TODO: vectorise by packing Arch::WIDTH independent generators.
    for (double& d : out) d = next_double();
}

void Xoroshiro128aox::fill_raw(std::span<uint64_t> out) noexcept {
    for (uint64_t& u : out) u = next_u64();
}

// ─── Explicit instantiations ─────────────────────────────────────────────────

template void Xoroshiro128aox::fill_uniform<ScalarArch>(std::span<double>) noexcept;
#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template void Xoroshiro128aox::fill_uniform<Avx2Arch>(std::span<double>) noexcept;
#endif
#if defined(CCR_ARCH_AVX512)
template void Xoroshiro128aox::fill_uniform<Avx512Arch>(std::span<double>) noexcept;
#endif
#if defined(CCR_ARCH_NEON)
template void Xoroshiro128aox::fill_uniform<NeonArch>(std::span<double>) noexcept;
#endif

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
