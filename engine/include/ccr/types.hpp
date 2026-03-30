#pragma once

// ============================================================================
// ccr/types.hpp  —  Shared data types for the CCR/XVA engine
//
// All public-facing structs, enums, and aliases live here.
// Rule: no #include of other ccr/ headers inside this file.
// Every other module may include this header.
// ============================================================================

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ccr {

// ─── Enumerations ────────────────────────────────────────────────────────────

enum class DerivativeType : uint8_t {
    IRS,       ///< Interest Rate Swap
    CDS,       ///< Credit Default Swap
    FX,        ///< FX Forward / Option
    EQUITY,    ///< Equity TRS / Option
    COMMODITY
};

enum class CreditRating : uint8_t { AAA, AA, A, BBB, BB, B, CCC, D };

enum class MarginCallStatus : uint8_t {
    PENDING,
    SENT,
    ACKNOWLEDGED,
    RECEIVED,
    OVERDUE,
    ESCALATED,
    DISMISSED
};

/// Controls the trade-off between speed and tail accuracy
enum class SimMode : uint8_t {
    REGULATORY,   ///< Wichura exact CDF + deterministic quantile + IEEE traps (default)
    STANDARD,     ///< Exact CDF, std::nth_element
    APPROX_FAST   ///< Piecewise-linear approx CDF — fastest, may bias 99.9th pctile
};

/// How the simulation time-grid is constructed
enum class GridType : uint8_t {
    MONTHLY,        ///< Uniform monthly steps
    WEEKLY,
    DAILY,
    PARSIMONIOUS    ///< Monthly + cash-flow + collateral-call augmentation (Silotto et al.)
};

// ─── Scalar type aliases ─────────────────────────────────────────────────────

using HazardRate     = double;   ///< Annualised instantaneous default intensity λ
using DiscountFactor = double;   ///< P(0, t)
using YearFraction   = double;   ///< Time measured in years

// ─── Simulation parameters ──────────────────────────────────────────────────

/// Per-run simulation knobs.  All values must be validated before passing to
/// CcrEngine::run() — see CcrEngine::validate_config().
struct SimParams {
    int    num_paths      = 10'000; ///< M — Monte Carlo path count
    int    num_timesteps  = 50;     ///< T — number of simulation timesteps
    int    num_assets     = 1;      ///< K — number of correlated risk factors
    double mu             = 0.05;   ///< Risk-neutral drift (annualised)
    double sigma          = 0.20;   ///< Volatility (annualised)
    double rho_wwr        = 0.0;    ///< WWR asset–hazard correlation ρ ∈ [-1, 1]
    double recovery_rate  = 0.40;   ///< R; LGD = 1 - R
    double horizon_years  = 5.0;    ///< Simulation horizon T_max
    SimMode  mode       = SimMode::REGULATORY;
    GridType grid_type  = GridType::PARSIMONIOUS;
};

/// Shock parameters applied additively/multiplicatively on top of SimParams
struct StressScenario {
    double vol_shock            = 0.0;  ///< Additive: +0.20 → vol += 20%
    double fx_shock             = 0.0;  ///< Multiplicative: -0.10 → FX × 0.90
    double equity_shock         = 0.0;
    double interest_rate_shock  = 0.0;
    double credit_spread_shock  = 0.0;
    double hazard_rate_shock    = 0.0;
    double jump_amplitude       = 0.0;  ///< J — jump-at-default size (e.g. 0.05 = 5%)
    std::string label;                  ///< Human-readable name shown in UI
};

// ─── Counterparty & portfolio ────────────────────────────────────────────────

struct CounterpartyConfig {
    std::string  id;
    std::string  name;
    CreditRating credit_rating    = CreditRating::BBB;
    HazardRate   hazard_rate      = 0.02;   ///< Annualised λ
    double       recovery_rate    = 0.40;
    double       collateral       = 0.0;    ///< Current posted collateral ($)
    double       margin_threshold = 1e6;    ///< Breach level triggers margin call
    int          mpor_days        = 10;     ///< Margin Period of Risk
};

struct DerivativeSpec {
    std::string    id;
    DerivativeType type;
    double         notional;
    double         maturity_years;
    double         underlying_price;
    double         strike          = 0.0;
    YearFraction   cash_flow_freq  = 0.25;  ///< Quarterly = 0.25 yrs
};

struct PortfolioConfig {
    std::string                 id;
    std::string                 counterparty_id;
    std::vector<DerivativeSpec> derivatives;
    double                      collateral = 0.0;
    double                      net_value  = 0.0;
};

// ─── Top-level engine configuration ─────────────────────────────────────────

struct EngineConfig {
    SimParams          sim_params;
    CounterpartyConfig counterparty;
    PortfolioConfig    portfolio;

    std::optional<StressScenario> stress;       ///< nullopt → base scenario only

    bool enable_wwr             = false;  ///< Couple hazard shocks to asset shocks
    bool enable_jump_diffusion  = false;  ///< Jump-at-default overlay
    bool enable_collateral      = false;  ///< VM/IM dynamics (reserved, Phase 2+)
    bool deterministic_quantile = true;   ///< Custom introselect for reproducibility
    bool log_overflow_warnings  = true;   ///< Warn if exp() overflows

    uint64_t rng_seed = 0xDEAD'BEEF'CAFE'1234ULL;
};

// ─── Risk metrics & results ──────────────────────────────────────────────────

struct RiskMetrics {
    double cva             = 0.0;  ///< Baseline CVA = (1-R)·Σ EPE(t)·PD(t)
    double wwr_cva         = 0.0;  ///< WWR-adjusted CVA (ρ-coupled hazard)
    double margin_required = 0.0;  ///< Suggested collateral transfer

    std::vector<double> pfe_profile;       ///< PFE(t) at each timestep (99th pctile)
    std::vector<double> epe_profile;       ///< EPE(t) = mean of positive exposures
    std::vector<double> time_grid_years;   ///< Corresponding t_i values

    std::chrono::microseconds compute_time_us{0};
    bool        overflow_detected = false;
    std::string arch_used;                 ///< e.g. "AVX2", "NEON", "Scalar"
    int         paths_used        = 0;
};

/// Returned by CcrEngine::run()
struct CcrResult {
    RiskMetrics                 base;
    std::optional<RiskMetrics>  stressed;  ///< Present when stress scenario was run
    bool        success   = false;
    std::string error_msg;
};

// ─── Margin call ─────────────────────────────────────────────────────────────

struct MarginCallInfo {
    std::string      id;
    std::string      counterparty_id;
    double           amount;
    double           excess_exposure;
    MarginCallStatus status          = MarginCallStatus::PENDING;
    int64_t          triggered_at_ms = 0;  ///< Unix epoch milliseconds
    int64_t          due_by_ms       = 0;  ///< Deadline
    std::string      reason;
};

// ─── Default probability term structure ─────────────────────────────────────

/// Marginal default probability at each simulation timestep.
/// CVA = (1 - R) · Σ EPE(t_i) · marginal_pd[i]
struct PdTermStructure {
    std::vector<double> times;        ///< t_i in years, length T
    std::vector<double> marginal_pd;  ///< PD(t_{i-1}, t_i), length T
    double              recovery_rate = 0.40;

    /// Convenience: derive from a flat (constant) hazard rate.
    static PdTermStructure from_flat_hazard(
        double                     hazard_rate,
        double                     recovery,
        const std::vector<double>& time_points);
};

} // namespace ccr
