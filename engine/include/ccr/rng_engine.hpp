#pragma once

// ============================================================================
// ccr/rng_engine.hpp  —  Pseudorandom number generation (Module 2)
//
// Implements xoroshiro128aox: 128-bit state, AND-OR-XOR output function.
// Passes BigCrush and PractRand; no lower-bit weaknesses of xoroshiro128+.
// Supports 2^64 non-overlapping parallel streams via pre-computed jump.
//
// Reference: Hanlon & Felix, ACM Trans. Reconfigurable Technology, 2022.
//
// Dependencies: simd_abstraction (for optional bulk SIMD fill).
// ============================================================================

#include <cstdint>
#include <span>	 // C++20; replace with {T*, size_t} if targeting C++17 only
#include <vector>

#include "ccr/simd_abstraction.hpp"

namespace ccr {

// ─── xoroshiro128aox state machine ──────────────────────────────────────────

class Xoroshiro128aox {
public:
	/// Construct with explicit 128-bit seed.
	/// NOTE: seeding multiple threads with sequential integers produces
	/// correlated streams — use jump() instead (see factory below).
	explicit Xoroshiro128aox(uint64_t s0, uint64_t s1) noexcept;

	/// Construct from a single 64-bit seed (splitmix64 initialises state).
	explicit Xoroshiro128aox(uint64_t seed) noexcept;

	// Xoroshiro128aox is movable but not copyable (prevents accidental
	// stream aliasing between threads).
	Xoroshiro128aox(const Xoroshiro128aox &) = delete;
	Xoroshiro128aox &operator=(const Xoroshiro128aox &) = delete;
	Xoroshiro128aox(Xoroshiro128aox &&) = default;
	Xoroshiro128aox &operator=(Xoroshiro128aox &&) = default;

	/// Raw 64-bit output.
	uint64_t next_u64() noexcept;

	/// Uniform double in [0, 1).  Uses the upper 53 bits.
	double next_double() noexcept;

	/// Apply the 2^64-step jump operator.
	/// After calling jump(), this instance generates the *next* non-overlapping
	/// stream.  Used by the thread-local factory below.
	void jump() noexcept;

	/// Apply the 2^96-step long-jump operator (for coarse stream partitioning).
	void long_jump() noexcept;

	// ── Bulk generation ──────────────────────────────────────────────────────

	/// Fill `out` with uniform doubles in [0, 1).
	/// Uses SIMD if Arch::WIDTH > 1 (packs WIDTH independent generators).
	template <typename Arch = ActiveArch>
	void fill_uniform(std::span<double> out) noexcept;

	/// Fill `out` with uint64_t raw output.
	void fill_raw(std::span<uint64_t> out) noexcept;

	// ── Accessors ────────────────────────────────────────────────────────────

	uint64_t state0() const noexcept { return s_[0]; }
	uint64_t state1() const noexcept { return s_[1]; }

private:
	uint64_t s_[2];

	// Jump polynomial coefficients (pre-computed constants from the paper)
	static constexpr uint64_t JUMP_POLY[2] = {0xdf900294d8f554a5ULL, 0x170865df4b3201fcULL};
	static constexpr uint64_t LONG_JUMP_POLY[2] = {0xd2a98b26625eee7bULL, 0xdddf9b1090aa7ac1ULL};
};

// ─── Thread-local RNG factory ────────────────────────────────────────────────

/// Creates `num_threads` Xoroshiro128aox instances, each guaranteed to produce
/// a non-overlapping stream of length 2^64.
///
/// Thread i is initialised to base.jump()^i — O(num_threads) jumps.
///
/// Usage:
///   auto rngs = make_thread_rngs(seed, omp_get_max_threads());
///   #pragma omp parallel for
///   for (int t = 0; t < T; ++t)
///       rngs[omp_get_thread_num()].fill_uniform(buffer);
///
std::vector<Xoroshiro128aox> make_thread_rngs(uint64_t base_seed, int num_threads);

}  // namespace ccr
