// ============================================================================
// engine/src/path_simulator.cpp
//
// GBM path evolution: S_{t+Δt} = S_t × exp((μ−σ²/2)Δt + σ√(Δt)·Z)
// Stub: evolve_step applies the GBM formula correctly in scalar.
//       run_all_steps wires the full pipeline (generate → correlate → evolve).
// SIMD vectorisation of the inner loop left as TODO.
// ============================================================================

#include "ccr/path_simulator.hpp"
#include "ccr/normal_variate.hpp"
#include "ccr/exposure_engine.hpp"
#include <cmath>

namespace ccr {

// ─── Constructor ─────────────────────────────────────────────────────────────

PathSimulator::PathSimulator(
    const SimParams&      params,
    const CholeskyMatrix& cholesky,
    const TimeGrid&       time_grid,
    JumpDiffusionHook*    jump_hook)
    : params_(params)
    , cholesky_(cholesky)
    , time_grid_(time_grid)
    , jump_hook_(jump_hook)
{}

// ─── Single timestep evolution ───────────────────────────────────────────────

template <typename Arch>
void PathSimulator::evolve_step(
    PathState&              state,
    std::span<const double> correlated_normals,
    double                  drift,
    double                  vol_dt,
    int                     /*timestep*/) noexcept
{
    // TODO: vectorise inner loop with SimdOps<Arch>.
    // Stub: scalar loop over M paths, first asset (k=0) drives portfolio value.
    const std::size_t M_padded = static_cast<std::size_t>(state.M_padded);

    for (std::size_t m = 0; m < M_padded; ++m) {
        // Evolve each asset k independently (post-Cholesky shocks already applied).
        double log_sum = 0.0;
        for (int k = 0; k < state.K; ++k) {
            const double z = correlated_normals[k * M_padded + m];
            const double exponent = drift + vol_dt * z;
            const double new_spot = state.spot_prices[k * M_padded + m] * std::exp(exponent);
            state.spot_prices[k * M_padded + m] = new_spot;
            log_sum += new_spot; // simple sum across assets for MTM
        }
        // Portfolio MTM: weighted sum of spot prices minus notional.
        // TODO: replace with proper derivative valuation per DerivativeSpec.
        state.portfolio_values[m] = log_sum - static_cast<double>(state.K);
    }
}

// ─── Full simulation run ─────────────────────────────────────────────────────

template <typename Arch>
void PathSimulator::run_all_steps(PathState& state, Xoroshiro128aox& rng) noexcept {
    const int T        = state.T;
    const int K        = state.K;
    const int M_padded = state.M_padded;

    const double sigma  = params_.sigma;
    const double mu     = params_.mu;

    for (int t = 0; t < T; ++t) {
        const double dt      = time_grid_.dt()[t];
        const double vol_dt  = vol_factor(sigma, dt);
        const double drift   = drift_factor(mu, sigma, dt);

        // 1. Generate K × M_padded independent N(0,1) shocks.
        std::span<double> norms{state.normals_buf.data(),
                                static_cast<std::size_t>(K * M_padded)};
        fill_normal<Arch>(norms, rng, params_.mode);

        // 2. Apply Cholesky to correlate shocks.
        cholesky_.apply<Arch>(state.normals_buf, state.correlated_buf,
                              static_cast<std::size_t>(M_padded), K);

        // 3. Evolve GBM.
        evolve_step<Arch>(state, state.correlated_buf, drift, vol_dt, t);

        // 4. Optional jump hook (post-diffusion, cold path).
        if (jump_hook_) {
            jump_hook_->on_paths_complete(
                state.spot_prices, {}, time_grid_.times(), K, state.M, M_padded);
        }

        // 5. Compute exposures E(t) = max(V(t), 0) and store column t.
        std::span<double> exp_col{
            state.exposures.data() + static_cast<std::size_t>(t) * M_padded,
            static_cast<std::size_t>(M_padded)};
        compute_exposures_step<Arch>(state.portfolio_values, exp_col);
    }
}

// ─── Explicit instantiations ─────────────────────────────────────────────────

template void PathSimulator::evolve_step<ScalarArch>(
    PathState&, std::span<const double>, double, double, int) noexcept;
template void PathSimulator::run_all_steps<ScalarArch>(
    PathState&, Xoroshiro128aox&) noexcept;

#if defined(CCR_ARCH_AVX2) || defined(CCR_ARCH_AVX512)
template void PathSimulator::evolve_step<Avx2Arch>(
    PathState&, std::span<const double>, double, double, int) noexcept;
template void PathSimulator::run_all_steps<Avx2Arch>(
    PathState&, Xoroshiro128aox&) noexcept;
#endif

#if defined(CCR_ARCH_AVX512)
template void PathSimulator::evolve_step<Avx512Arch>(
    PathState&, std::span<const double>, double, double, int) noexcept;
template void PathSimulator::run_all_steps<Avx512Arch>(
    PathState&, Xoroshiro128aox&) noexcept;
#endif

#if defined(CCR_ARCH_NEON)
template void PathSimulator::evolve_step<NeonArch>(
    PathState&, std::span<const double>, double, double, int) noexcept;
template void PathSimulator::run_all_steps<NeonArch>(
    PathState&, Xoroshiro128aox&) noexcept;
#endif

} // namespace ccr
