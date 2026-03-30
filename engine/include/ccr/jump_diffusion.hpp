#pragma once

// ============================================================================
// ccr/jump_diffusion.hpp  —  Jump-at-default extension point (Module 1 / Modelling)
//
// Architecture note (from Phase 1 design doc):
//   The GBM hot loop runs branch-free on all paths.  Jump-at-default is a
//   POST-PROCESSING pass applied to a *small subset* of paths (those where
//   default occurs in the simulation window).  This avoids contaminating the
//   hot loop with a conditional branch.
//
// Salonen (2023) calibration reference:
//   J = 1%  → CVA × 2.15
//   J = 5%  → CVA × 9
//   J = 10% → CVA × 18
//
// Dependencies: types.
// ============================================================================

#include "ccr/types.hpp"
#include <span>
#include <vector>

namespace ccr {

// ─── Jump parameters ─────────────────────────────────────────────────────────

struct JumpParams {
    double jump_size        = 0.0;   ///< J — multiplicative spike (e.g. 0.05 = 5%)
    double hazard_rate      = 0.02;  ///< λ used to sample default time τ
    bool   enabled          = false;
};

// ─── Default time sampling ───────────────────────────────────────────────────

/// Sample the default time τ for each path under a constant hazard rate λ.
///
/// τ_i = − ln(U_i) / λ  where U_i ~ Uniform(0,1)
///
/// Returns a vector of length M.  Paths where τ > horizon are non-defaulted
/// (encoded as +Inf or horizon + 1).
///
/// @param rng_uniforms  Pre-generated uniform variates, length M
/// @param lambda        Hazard rate (annualised)
/// @param horizon       Simulation horizon in years
std::vector<double> sample_default_times(
    std::span<const double> rng_uniforms,
    double                  lambda,
    double                  horizon);

// ─── Jump application pass ──────────────────────────────────────────────────

/// Apply S → S × (1 + J) at the timestep corresponding to default time τ.
///
/// Called AFTER the full GBM diffusion pass (not inside the hot loop).
/// Modifies `spot_prices` in-place for affected paths only.
///
/// @param spot_prices    SoA layout: asset k, path m → spot_prices[k*M + m]
///                       Shape: [K][M_padded]
/// @param default_times  Default time for each path (years), length M
/// @param time_grid      Simulation time points (years), length T+1
/// @param jump_params    Jump size J and enablement flag
/// @param K              Number of assets
/// @param M              Number of paths (non-padded)
/// @param M_padded       Padded path count (M rounded up to SIMD width)
void apply_jump_at_default(
    std::span<double>          spot_prices,
    std::span<const double>    default_times,
    const std::vector<double>& time_grid,
    const JumpParams&          jump_params,
    int                        K,
    int                        M,
    int                        M_padded);

// ─── Extension hook (for future Chebyshev domain split) ─────────────────────

/// Abstract hook invoked by path_simulator after the GBM diffusion completes
/// but before exposure is computed.  The default no-op implementation is used
/// unless jump_diffusion is enabled.
///
/// The virtual method lives on the COLD path only — never called per timestep
/// inside the SIMD hot loop.
class JumpDiffusionHook {
public:
    virtual ~JumpDiffusionHook() = default;

    /// Called once after all paths have been evolved to time t.
    /// Override to apply the jump overlay or any post-diffusion transformation.
    virtual void on_paths_complete(
        std::span<double>          spot_prices,
        std::span<const double>    default_times,
        const std::vector<double>& time_grid,
        int K, int M, int M_padded) {}
};

/// Concrete implementation that applies the multiplicative jump.
class MultiplicativeJumpHook final : public JumpDiffusionHook {
public:
    explicit MultiplicativeJumpHook(JumpParams params) : params_(params) {}

    void on_paths_complete(
        std::span<double>          spot_prices,
        std::span<const double>    default_times,
        const std::vector<double>& time_grid,
        int K, int M, int M_padded) override;

private:
    JumpParams params_;
};

} // namespace ccr
