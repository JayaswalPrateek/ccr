// ============================================================================
// engine/src/cva_integrator.cpp
//
// CVA = (1-R) · Σ EPE(t_i) · PD(t_{i-1}, t_i)
// Kahan compensated summation for bitwise reproducibility.
// Also implements PdTermStructure::from_flat_hazard (declared in types.hpp).
// ============================================================================

#include "ccr/cva_integrator.hpp"
#include "ccr/types.hpp"
#include <cmath>
#include <stdexcept>

namespace ccr {

// ─── Kahan sum helper ────────────────────────────────────────────────────────

static double kahan_dot(
    std::span<const double> a,
    std::span<const double> b) noexcept
{
    double sum  = 0.0;
    double comp = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double y = a[i] * b[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum  = t;
    }
    return sum;
}

// ─── CVA integration ────────────────────────────────────────────────────────

double compute_cva(
    std::span<const double> epe_profile,
    std::span<const double> marginal_pd,
    double                  recovery_rate) noexcept
{
    if (epe_profile.size() != marginal_pd.size()) return 0.0;
    return (1.0 - recovery_rate) * kahan_dot(epe_profile, marginal_pd);
}

double compute_wwr_cva(
    std::span<const double> epe_profile,
    std::span<const double> marginal_pd,
    double                  recovery_rate) noexcept
{
    // Semantically identical to baseline CVA when WWR coupling is handled
    // upstream in PathSimulator (EPE already embeds ρ-coupled hazard shocks).
    return compute_cva(epe_profile, marginal_pd, recovery_rate);
}

// ─── PD term structure ──────────────────────────────────────────────────────

std::vector<double> marginal_pd_from_flat_hazard(
    std::span<const double> time_grid,
    double                  hazard_rate)
{
    // PD(t_{i-1}, t_i) = exp(-λ·t_{i-1}) - exp(-λ·t_i)
    std::vector<double> pd;
    const std::size_t T = time_grid.size();
    if (T == 0) return pd;
    pd.reserve(T - 1);
    for (std::size_t i = 1; i < T; ++i) {
        double p = std::exp(-hazard_rate * time_grid[i-1])
                 - std::exp(-hazard_rate * time_grid[i]);
        pd.push_back(p > 0.0 ? p : 0.0);
    }
    return pd;
}

std::vector<double> marginal_pd_stressed(
    std::span<const double> time_grid,
    double                  hazard_rate,
    double                  hazard_rate_shock)
{
    return marginal_pd_from_flat_hazard(time_grid, hazard_rate + hazard_rate_shock);
}

// ─── Required margin ─────────────────────────────────────────────────────────

double compute_required_margin(
    std::span<const double> pfe_profile,
    double                  current_collateral) noexcept
{
    double max_pfe = 0.0;
    for (double p : pfe_profile) if (p > max_pfe) max_pfe = p;
    double margin = max_pfe - current_collateral;
    return margin > 0.0 ? margin : 0.0;
}

// ─── PdTermStructure static method (declared in types.hpp) ──────────────────

PdTermStructure PdTermStructure::from_flat_hazard(
    double                     hazard_rate,
    double                     recovery,
    const std::vector<double>& time_points)
{
    PdTermStructure ts;
    ts.times         = time_points;
    ts.recovery_rate = recovery;
    ts.marginal_pd   = marginal_pd_from_flat_hazard(time_points, hazard_rate);
    return ts;
}

} // namespace ccr
