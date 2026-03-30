// ============================================================================
// engine/src/ccr_engine.cpp
//
// Top-level orchestrator — CcrEngine::run().
//
// Pipeline (single pass):
//   1. Validate config
//   2. Build time grid + Cholesky matrix
//   3. Allocate arena (single contiguous allocation)
//   4. Run PathSimulator (GBM + MTM)
//   5. Extract PFE / EPE profiles
//   6. Integrate CVA (Kahan summation)
//   7. Fire optional progress callback
//
// A second pass with shocked SimParams / hazard rate produces stressed metrics.
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

#include <algorithm>
#include <chrono>
#include <cstring>

namespace ccr {

// ─── Pimpl support (Arena is complete here via memory_arena.hpp) ─────────────
CcrEngine::~CcrEngine() = default;
void CcrEngine::ArenaDeleter::operator()(Arena* p) noexcept { delete p; }

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
            const auto& shock = *config.stress;

            SimParams stressed  = config.sim_params;
            stressed.sigma     += shock.vol_shock;
            stressed.sigma     *= (1.0 + shock.equity_shock);
            stressed.mu        += shock.interest_rate_shock;

            // Build a stressed config so run_single picks up the hazard shock too.
            EngineConfig stressed_cfg            = config;
            stressed_cfg.sim_params              = stressed;
            stressed_cfg.counterparty.hazard_rate =
                std::max(0.0, config.counterparty.hazard_rate + shock.hazard_rate_shock);

            result.stressed = run_single(stressed_cfg, stressed_cfg.sim_params, std::nullopt);
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
    if (!arena_) arena_.reset(new Arena());
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
    PathSimulator sim(params, chol, tg,
                      config.portfolio, config.counterparty.recovery_rate,
                      hook.get());
    sim.run_all_steps<ActiveArch>(state, rng);

    // 8. Extract PFE & EPE profiles.
    extract_profiles(
        state.exposures, state.pfe_profile, state.epe_profile,
        T, M, M_padded,
        /*alpha=*/0.99,
        config.deterministic_quantile);

    // 9. Report per-timestep results to caller.
    // Fired here — after extract_profiles — so pfe_profile values are valid.
    // The WebSocket handler uses these callbacks to stream progress to the client.
    if (callback) {
        for (int t = 0; t < T; ++t)
            (*callback)(t, T, static_cast<float>(arena_->pfe_profile[t]));
    }

    // 10. Marginal PD term structure.
    auto marginal_pd = marginal_pd_from_flat_hazard(
        tg.times(), config.counterparty.hazard_rate);

    // 11. CVA.
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

    // 12. Required margin.
    std::span<const double> pfe_span{arena_->pfe_profile,
                                     static_cast<std::size_t>(T)};
    const double margin = compute_required_margin(
        pfe_span, config.portfolio.collateral);

    // 13. Assemble result.
    RiskMetrics metrics;
    metrics.cva             = cva;
    metrics.wwr_cva         = wwr_cva;
    metrics.margin_required = margin;
    metrics.pfe_profile.assign(arena_->pfe_profile,
                               arena_->pfe_profile + T);
    metrics.epe_profile.assign(arena_->epe_profile,
                               arena_->epe_profile + T);
    // Profile element [t] corresponds to the END of interval t → times[t+1].
    // Assign times[1..T] (skip the initial t=0 anchor point).
    {
        const auto& all = tg.times();
        metrics.time_grid_years.assign(all.begin() + 1, all.end());
    }
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
