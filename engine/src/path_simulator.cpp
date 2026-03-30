// ============================================================================
// engine/src/path_simulator.cpp
//
// GBM path evolution: S_{t+Δt} = S_t × exp((μ−σ²/2)Δt + σ√(Δt)·Z)
//
// evolve_step<Arch> — two-phase hot loop:
//   Phase 1: GBM update for all K assets × M_padded paths (SIMD over m).
//   Phase 2: Portfolio MTM per derivative type (SIMD over m):
//     EQUITY/FX/COMMODITY: V = notional × (S_t − K × df)
//     IRS:                 V = notional × DV01 × (S_t − K)
//                              DV01 = t_rem × exp(−μ × t_rem / 2)
//     CDS:                 V = notional × (1 − R) × (S_t − K) × t_rem × df
//
// run_all_steps<Arch> — outer loop over T timesteps:
//   fill_normal → Cholesky apply → evolve_step → jump hook → compute_exposures
// ============================================================================

#include "ccr/path_simulator.hpp"
#include "ccr/normal_variate.hpp"
#include "ccr/exposure_engine.hpp"
#include <cmath>
#include <vector>

namespace ccr {

// ─── Constructor ─────────────────────────────────────────────────────────────

PathSimulator::PathSimulator(
    const SimParams&       params,
    const CholeskyMatrix&  cholesky,
    const TimeGrid&        time_grid,
    const PortfolioConfig& portfolio,
    double                 recovery_rate,
    JumpDiffusionHook*     jump_hook)
    : params_(params)
    , cholesky_(cholesky)
    , time_grid_(time_grid)
    , portfolio_(portfolio)
    , recovery_rate_(recovery_rate)
    , jump_hook_(jump_hook)
{}

// ─── Single timestep evolution ───────────────────────────────────────────────

template <typename Arch>
void PathSimulator::evolve_step(
    PathState&              state,
    std::span<const double> correlated_normals,
    double                  drift,
    double                  vol_dt,
    int                     timestep) noexcept
{
    const std::size_t M_padded = static_cast<std::size_t>(state.M_padded);
    const int         K        = state.K;
    const double      mu       = params_.mu;
    const std::size_t step     = Arch::WIDTH;

    // ── Pre-compute per-derivative scalar coefficients ────────────────────────
    const auto& times  = time_grid_.times();
    const double t_now = (static_cast<std::size_t>(timestep) + 1 < times.size())
                         ? times[static_cast<std::size_t>(timestep) + 1]
                         : times.back();

    const auto& derivs = portfolio_.derivatives;
    const int   D      = static_cast<int>(derivs.size());

    struct DerivCoeff {
        double rate0;
        double t_rem;
        double df;
        double dv01;
        double notional;
        double strike;
        double cds_scale;  // notional × (1−R) × t_rem × df  (CDS only)
        DerivativeType type;
        int    asset_k;
    };
    std::vector<DerivCoeff> coeffs(static_cast<std::size_t>(D));
    for (int di = 0; di < D; ++di) {
        const auto& d    = derivs[static_cast<std::size_t>(di)];
        const double rem = (d.maturity_years > t_now) ? (d.maturity_years - t_now) : 0.0;
        const double df  = std::exp(-mu * rem);
        coeffs[static_cast<std::size_t>(di)] = {
            d.underlying_price,
            rem,
            df,
            rem * std::exp(-mu * rem * 0.5),
            d.notional,
            d.strike,
            d.notional * (1.0 - recovery_rate_) * rem * df,
            d.type,
            std::min(di, K - 1)
        };
    }

    // ── Phase 1: GBM evolution — SIMD over m for each asset k ─────────────────
    const auto drift_v  = SimdOps<Arch>::set1(drift);
    const auto vol_dt_v = SimdOps<Arch>::set1(vol_dt);

    for (int k = 0; k < K; ++k) {
        double*       spot_k = state.spot_prices.data()
                               + static_cast<std::size_t>(k) * M_padded;
        const double* z_k    = correlated_normals.data()
                               + static_cast<std::size_t>(k) * M_padded;
        for (std::size_t m = 0; m < M_padded; m += step) {
            auto z_v    = SimdOps<Arch>::load(z_k + m);
            auto s_v    = SimdOps<Arch>::load(spot_k + m);
            auto arg_v  = SimdOps<Arch>::fmadd(z_v, vol_dt_v, drift_v);
            auto s_new  = SimdOps<Arch>::mul(s_v, SimdOps<Arch>::exp_approx(arg_v));
            SimdOps<Arch>::store(spot_k + m, s_new);
        }
    }

    // ── Phase 2: Portfolio MTM — zero pv buffer, then add each derivative ──────
    double* pv = state.portfolio_values.data();
    for (std::size_t m = 0; m < M_padded; m += step)
        SimdOps<Arch>::store(pv + m, SimdOps<Arch>::zero());

    for (int di = 0; di < D; ++di) {
        const auto& c       = coeffs[static_cast<std::size_t>(di)];
        const double* spot_k = state.spot_prices.data()
                               + static_cast<std::size_t>(c.asset_k) * M_padded;
        const auto rate0_v   = SimdOps<Arch>::set1(c.rate0);
        const auto notl_v    = SimdOps<Arch>::set1(c.notional);

        switch (c.type) {
        case DerivativeType::EQUITY:
        case DerivativeType::FX:
        case DerivativeType::COMMODITY: {
            // V = notional × (S_t − strike × df)
            const auto k_df_v = SimdOps<Arch>::set1(c.strike * c.df);
            for (std::size_t m = 0; m < M_padded; m += step) {
                auto S_v  = SimdOps<Arch>::mul(SimdOps<Arch>::load(spot_k + m), rate0_v);
                auto v_v  = SimdOps<Arch>::mul(notl_v,
                               SimdOps<Arch>::sub(S_v, k_df_v));
                auto acc  = SimdOps<Arch>::load(pv + m);
                SimdOps<Arch>::store(pv + m, SimdOps<Arch>::add(acc, v_v));
            }
            break;
        }
        case DerivativeType::IRS: {
            // V = notional × DV01 × (S_t − strike)
            const auto dv01_v   = SimdOps<Arch>::set1(c.dv01);
            const auto strike_v = SimdOps<Arch>::set1(c.strike);
            for (std::size_t m = 0; m < M_padded; m += step) {
                auto S_v  = SimdOps<Arch>::mul(SimdOps<Arch>::load(spot_k + m), rate0_v);
                auto v_v  = SimdOps<Arch>::mul(notl_v,
                               SimdOps<Arch>::mul(dv01_v,
                                   SimdOps<Arch>::sub(S_v, strike_v)));
                auto acc  = SimdOps<Arch>::load(pv + m);
                SimdOps<Arch>::store(pv + m, SimdOps<Arch>::add(acc, v_v));
            }
            break;
        }
        case DerivativeType::CDS: {
            // V = notional × (1−R) × (S_t − strike) × t_rem × df
            const auto scale_v  = SimdOps<Arch>::set1(c.cds_scale);
            const auto strike_v = SimdOps<Arch>::set1(c.strike);
            for (std::size_t m = 0; m < M_padded; m += step) {
                auto S_v  = SimdOps<Arch>::mul(SimdOps<Arch>::load(spot_k + m), rate0_v);
                auto v_v  = SimdOps<Arch>::mul(scale_v,
                               SimdOps<Arch>::sub(S_v, strike_v));
                auto acc  = SimdOps<Arch>::load(pv + m);
                SimdOps<Arch>::store(pv + m, SimdOps<Arch>::add(acc, v_v));
            }
            break;
        }
        }
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
