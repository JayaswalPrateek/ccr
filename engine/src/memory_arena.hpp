#pragma once

// ============================================================================
// memory_arena.hpp  —  Private single-contiguous-allocation arena
//
// NOT part of the public ccr/ API.  Only CcrEngine and PathSimulator use this.
//
// Design:
//   - One aligned_alloc() call at CcrEngine::run_single() entry.
//   - Sub-arrays carved by pointer arithmetic with 64/128-byte aligned offsets.
//   - Zero dynamic allocation in the simulation core.
//   - One free() at Arena destruction.
//   - Lifetime: scoped to a single run_single() invocation.
//
// Sub-array layout (see §2.1 of Phase 1 design doc):
//
//   [ spot_prices      K×M_padded  ]
//   [ normals_buf      K×M_padded  ]
//   [ correlated_buf   K×M_padded  ]
//   [ portfolio_values   M_padded  ]
//   [ exposures        T×M_padded  ]
//   [ pfe_profile            T     ]
//   [ epe_profile            T     ]
//   [ default_times          M     ]  (only when jump diffusion enabled)
//   [ uniform_scratch        M     ]  (scratch for default-time sampling)
//
// ============================================================================

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "ccr/path_simulator.hpp"  // PathState
#include "ccr/simd_abstraction.hpp"

namespace ccr {

/// Rounds `n` doubles up to the next multiple of `alignment` bytes.
inline std::size_t align_doubles(std::size_t n, std::size_t alignment = ARENA_ALIGNMENT) {
	const std::size_t bytes = n * sizeof(double);
	const std::size_t padded = (bytes + alignment - 1) & ~(alignment - 1);
	return padded / sizeof(double);
}

/// Single-allocation arena for one simulation run.
struct CcrEngine::Arena {
	void *raw_ptr = nullptr;
	std::size_t total_bytes = 0;

	double *spot_prices = nullptr;
	double *normals_buf = nullptr;
	double *correlated_buf = nullptr;
	double *portfolio_values = nullptr;
	double *exposures = nullptr;
	double *pfe_profile = nullptr;
	double *epe_profile = nullptr;
	double *default_times = nullptr;
	double *uniform_scratch = nullptr;

	Arena() = default;	///< Zero-initialises all pointers; allocate() must be called.

	/// Allocate and lay out sub-arrays for the given dimensions.
	void allocate(int K, int M_padded, int T, bool need_jump) {
		// Compute sub-array sizes (aligned)
		const std::size_t sz_spot = align_doubles((std::size_t)K * M_padded);
		const std::size_t sz_norms = sz_spot;
		const std::size_t sz_corr = sz_spot;
		const std::size_t sz_portval = align_doubles(M_padded);
		const std::size_t sz_expo = align_doubles((std::size_t)T * M_padded);
		const std::size_t sz_pfe = align_doubles(T);
		const std::size_t sz_epe = align_doubles(T);
		const std::size_t sz_jump = need_jump ? align_doubles(M_padded) * 2 : 0;

		total_bytes = (sz_spot + sz_norms + sz_corr + sz_portval +
					   sz_expo + sz_pfe + sz_epe + sz_jump) *
					  sizeof(double);

		raw_ptr = std::aligned_alloc(ARENA_ALIGNMENT, total_bytes);
		if (!raw_ptr) throw std::bad_alloc{};

		// Zero-initialise (padded elements must be 0 for reductions)
		std::memset(raw_ptr, 0, total_bytes);

		// Lay out sub-arrays by pointer arithmetic
		double *cursor = static_cast<double *>(raw_ptr);
		spot_prices = cursor;
		cursor += sz_spot;
		normals_buf = cursor;
		cursor += sz_norms;
		correlated_buf = cursor;
		cursor += sz_corr;
		portfolio_values = cursor;
		cursor += sz_portval;
		exposures = cursor;
		cursor += sz_expo;
		pfe_profile = cursor;
		cursor += sz_pfe;
		epe_profile = cursor;
		cursor += sz_epe;
		if (need_jump) {
			default_times = cursor;
			cursor += align_doubles(M_padded);
			uniform_scratch = cursor;
		}
	}

	/// Build a PathState view over this arena's memory.
	PathState make_path_state(int K, int M, int M_padded, int T) {
		return PathState{
			.spot_prices = {spot_prices, (std::size_t)K * M_padded},
			.normals_buf = {normals_buf, (std::size_t)K * M_padded},
			.correlated_buf = {correlated_buf, (std::size_t)K * M_padded},
			.portfolio_values = {portfolio_values, (std::size_t)M_padded},
			.exposures = {exposures, (std::size_t)T * M_padded},
			.pfe_profile = {pfe_profile, (std::size_t)T},
			.epe_profile = {epe_profile, (std::size_t)T},
			.K = K,
			.M = M,
			.M_padded = M_padded,
			.T = T};
	}

	~Arena() { std::free(raw_ptr); }

	// Non-copyable
	Arena(const Arena &) = delete;
	Arena &operator=(const Arena &) = delete;
};

}  // namespace ccr
