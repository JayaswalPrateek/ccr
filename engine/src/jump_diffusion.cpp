// ============================================================================
// engine/src/jump_diffusion.cpp
//
// Jump-at-default post-processing pass.
// Stub: sample_default_times computes τ = -ln(u)/λ correctly.
//       apply_jump_at_default is a no-op (J=0 returns quickly).
//       MultiplicativeJumpHook::on_paths_complete delegates to apply.
// ============================================================================

#include "ccr/jump_diffusion.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace ccr {

// ─── Default time sampling ───────────────────────────────────────────────────

std::vector<double> sample_default_times(
    std::span<const double> rng_uniforms,
    double                  lambda,
    double                  horizon)
{
    const double INF = horizon + 1.0; // sentinel for "no default in window"
    std::vector<double> tau;
    tau.reserve(rng_uniforms.size());

    for (double u : rng_uniforms) {
        if (u <= 0.0 || lambda <= 0.0) {
            tau.push_back(INF);
        } else {
            double t = -std::log(u) / lambda;
            tau.push_back(t > horizon ? INF : t);
        }
    }
    return tau;
}

// ─── Jump application ────────────────────────────────────────────────────────

void apply_jump_at_default(
    std::span<double>          spot_prices,
    std::span<const double>    default_times,
    const std::vector<double>& time_grid,
    const JumpParams&          jump_params,
    int K, int M, int M_padded)
{
    if (!jump_params.enabled || jump_params.jump_size == 0.0) return;

    // TODO: vectorise; for now iterate over defaulted paths only.
    for (int m = 0; m < M; ++m) {
        const double tau = default_times[m];
        // Find first grid step at or after tau.
        auto it = std::lower_bound(time_grid.begin(), time_grid.end(), tau);
        if (it == time_grid.end()) continue; // no default in window

        // Apply S → S * (1 + J) for all assets on this path from that step.
        // In the full simulation, spot_prices holds the CURRENT timestep's
        // prices; this function is called once per timestep by the hook.
        // The per-path, per-asset layout is spot_prices[k * M_padded + m].
        const double factor = 1.0 + jump_params.jump_size;
        for (int k = 0; k < K; ++k) {
            spot_prices[static_cast<std::size_t>(k) * M_padded + m] *= factor;
        }
    }
}

// ─── Hook implementations ────────────────────────────────────────────────────

void MultiplicativeJumpHook::on_paths_complete(
    std::span<double>          spot_prices,
    std::span<const double>    default_times,
    const std::vector<double>& time_grid,
    int K, int M, int M_padded)
{
    apply_jump_at_default(
        spot_prices, default_times, time_grid, params_, K, M, M_padded);
}

} // namespace ccr
