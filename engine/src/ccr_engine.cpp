// ============================================================================
// engine/src/ccr_engine.cpp
//
// Top-level orchestrator — CcrEngine::run().
// Stub: allocates arena, wires modules together, returns zeroed RiskMetrics.
// Full pipeline implementation fills in the TODO sections.
// ============================================================================

#include "ccr/ccr_engine.hpp"
#include "memory_arena.hpp"        // private header in engine/src/

#include "ccr/time_grid.hpp"
#include "ccr/correlation_engine.hpp"
#include "ccr/path_simulator.hpp"
#include "ccr/quantile_extractor.hpp"
#include "ccr/cva_integrator.hpp"
#include "ccr/jump_diffusion.hpp"
#include "ccr/simd_abstraction.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace ccr {

// ─── Validation ──────────────────────────────────────────────────────────────

std::string CcrEngine::validate_config(const EngineConfig& cfg) {
    const auto& p = cfg.sim_params;
    if (p.num_paths <= 0)     return "num_paths must be > 0";
    if (p.num_timesteps <= 0) return "num_timesteps must be > 0";
    if (p.num_assets <= 0)    return "num_assets must be > 0";
    if (p.sigma <= 0.0)       return "sigma must be > 0";
    if (p.horizon_years <= 0) return "horizon_years must be > 0";
    if (p.recovery_rate < 0.0 || p.recovery_rate > 1.0)
        return "recovery_rate must be in [0, 1]";
    if (p.rho_wwr < -1.0 || p.rho_wwr > 1.0)
        return "rho_wwr must be in [-1, 1]";
    return {};
}

// ─── Arena size estimate ─────────────────────────────────────────────────────

std::size_t CcrEngine::estimate_arena_bytes(const EngineConfig& cfg) noexcept {
    const auto& p = cfg.sim_params;
    const std::size_t M_padded = pad_to_width(static_cast<std::size_t>(p.num_paths));
    const std::size_t K = static_cast<std::size_t>(p.num_assets);
    const std::size_t T = static_cast<std::size_t>(p.num_timesteps);
    // Rough: 3 × K×M (spot, normals, correlated) + M (portfolio) + T×M (exposures)
    return (3 * K * M_padded + M_padded + T * M_padded + 2 * T) * sizeof(double)
           + 256; // arena metadata padding
}

// ─── Main entry point ────────────────────────────────────────────────────────

CcrResult CcrEngine::run(
    const EngineConfig&            config,
    std::optional<ProgressCallback> callback)
{
    CcrResult result;

    // Validate.
    const std::string err = validate_config(config);
    if (!err.empty()) {
        result.error_msg = err;
        return result;
    }

    try {
        // Base scenario.
        result.base = run_single(config, config.sim_params, callback);

        // Stress scenario (second pass with shocked params).
        if (config.stress.has_value()) {
            SimParams stressed = config.sim_params;
            stressed.sigma          += config.stress->vol_shock;
            stressed.sigma          *= (1.0 + config.stress->equity_shock);
            // TODO: apply remaining shock fields (fx, rates, hazard, jump).
            result.stressed = run_single(config, stressed, std::nullopt);
        }

        result.success = true;
    } catch (const std::exception& ex) {
        result.error_msg = ex.what();
    }

    return result;
}

// ─── Single-pass run ─────────────────────────────────────────────────────────

RiskMetrics CcrEngine::run_single(
    const EngineConfig&            config,
    const SimParams&               params,
    std::optional<ProgressCallback> callback)
{
    const auto t_start = std::chrono::steady_clock::now();

    const int K        = params.num_assets;
    const int M        = params.num_paths;
    const int M_padded = static_cast<int>(pad_to_width(static_cast<std::size_t>(M)));

    // 1. Build time grid.
    TimeGrid tg(params.horizon_years, params.grid_type,
                config.counterparty.mpor_days);
    const int T = tg.num_steps();

    // 2. Cholesky (identity for K=1; WWR 2×2 when enabled).
    CholeskyMatrix chol = (config.enable_wwr && K == 1)
        ? CholeskyMatrix::wwr_2x2(params.rho_wwr)
        : CholeskyMatrix::identity(K);

    // 3. Allocate arena.
    if (!arena_) arena_ = std::make_unique<Arena>();
    arena_->allocate(K, M_padded, T, config.enable_jump_diffusion);

    // 4. Initialise arena spot prices to 1.0 (normalised).
    std::fill(arena_->spot_prices,
              arena_->spot_prices + K * M_padded, 1.0);

    // Build PathState view.
    PathState state = arena_->make_path_state(K, M, M_padded, T);

    // 5. Thread-local RNG — single-threaded stub.
    //    TODO: fan out to omp_get_max_threads() via make_thread_rngs().
    Xoroshiro128aox rng(config.rng_seed);

    // 6. Jump hook (nullable).
    std::unique_ptr<JumpDiffusionHook> hook;
    if (config.enable_jump_diffusion && config.stress.has_value()) {
        JumpParams jp;
        jp.enabled      = true;
        jp.jump_size    = config.stress->jump_amplitude;
        jp.hazard_rate  = config.counterparty.hazard_rate;
        hook = std::make_unique<MultiplicativeJumpHook>(jp);
    }

    // 7. Simulate.
    PathSimulator sim(params, chol, tg, hook.get());
    sim.run_all_steps<ActiveArch>(state, rng);

    // Report progress (every 10% of timesteps).
    if (callback) {
        for (int t = 0; t < T; ++t) {
            double pfe_t = (t < static_cast<int>(arena_->pfe_profile[0]))
                         ? arena_->pfe_profile[t] : 0.0;
            (*callback)(t, T, pfe_t);
        }
    }

    // 8. Extract PFE & EPE profiles.
    extract_profiles(
        state.exposures, state.pfe_profile, state.epe_profile,
        T, M, M_padded,
        /*alpha=*/0.99,
        config.deterministic_quantile);

    // 9. Marginal PD term structure.
    auto marginal_pd = marginal_pd_from_flat_hazard(
        tg.times(), config.counterparty.hazard_rate);

    // 10. CVA.
    std::span<const double> epe_span{arena_->epe_profile,
                                     static_cast<std::size_t>(T)};
    const double cva = compute_cva(
        epe_span,
        std::span<const double>{marginal_pd},
        config.counterparty.recovery_rate);

    const double wwr_cva = config.enable_wwr
        ? compute_wwr_cva(epe_span, std::span<const double>{marginal_pd},
                          config.counterparty.recovery_rate)
        : cva;

    // 11. Required margin.
    std::span<const double> pfe_span{arena_->pfe_profile,
                                     static_cast<std::size_t>(T)};
    const double margin = compute_required_margin(
        pfe_span, config.portfolio.collateral);

    // 12. Assemble result.
    RiskMetrics metrics;
    metrics.cva             = cva;
    metrics.wwr_cva         = wwr_cva;
    metrics.margin_required = margin;
    metrics.pfe_profile.assign(arena_->pfe_profile,
                               arena_->pfe_profile + T);
    metrics.epe_profile.assign(arena_->epe_profile,
                               arena_->epe_profile + T);
    metrics.time_grid_years = tg.times();
    metrics.time_grid_years.resize(static_cast<std::size_t>(T)); // drop t=0
    metrics.arch_used  = ActiveArch::NAME;
    metrics.paths_used = M;

    const auto t_end = std::chrono::steady_clock::now();
    metrics.compute_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t_end - t_start);

    return metrics;
}

// ─── Margin call evaluation ──────────────────────────────────────────────────

std::optional<MarginCallInfo> CcrEngine::evaluate_margin_call(
    const CcrResult&          result,
    const CounterpartyConfig& counterparty)
{
    if (!result.success) return std::nullopt;

    const auto& pfe = result.base.pfe_profile;
    if (pfe.empty()) return std::nullopt;

    double max_pfe = *std::max_element(pfe.begin(), pfe.end());
    double excess  = max_pfe - counterparty.collateral;
    if (excess <= counterparty.margin_threshold) return std::nullopt;

    MarginCallInfo mc;
    mc.id               = "MC-" + counterparty.id + "-AUTO";
    mc.counterparty_id  = counterparty.id;
    mc.amount           = excess;
    mc.excess_exposure  = excess;
    mc.status           = MarginCallStatus::PENDING;
    mc.reason           = "PFE exceeds collateral + threshold";
    return mc;
}

} // namespace ccr
