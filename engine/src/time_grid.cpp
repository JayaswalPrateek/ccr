// ============================================================================
// engine/src/time_grid.cpp
//
// Simulation time-grid construction.
// Stub: PARSIMONIOUS mode falls back to MONTHLY for now.
// Replace with union merge of cash-flow and collateral-call dates per
// Silotto et al. (2023) in implementation phase.
// ============================================================================

#include "ccr/time_grid.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ccr {

// ─── Constructor ─────────────────────────────────────────────────────────────

TimeGrid::TimeGrid(
    YearFraction               horizon_years,
    GridType                   grid_type,
    int                        mpor_days,
    const std::vector<double>& cash_flow_dates)
{
    switch (grid_type) {
    case GridType::MONTHLY:
        *this = monthly(horizon_years);
        break;
    case GridType::WEEKLY:
        *this = weekly(horizon_years);
        break;
    case GridType::DAILY:
        *this = daily(horizon_years);
        break;
    case GridType::PARSIMONIOUS:
        *this = parsimonious(horizon_years, mpor_days, cash_flow_dates);
        break;
    }
}

// ─── Static factories ────────────────────────────────────────────────────────

TimeGrid TimeGrid::monthly(YearFraction horizon) {
    TimeGrid g;
    int steps = static_cast<int>(std::ceil(horizon * 12.0));
    g.times_.reserve(steps + 1);
    g.times_.push_back(0.0);
    for (int i = 1; i <= steps; ++i) {
        double t = static_cast<double>(i) / 12.0;
        g.times_.push_back(std::min(t, horizon));
    }
    g.deduplicate_and_sort();
    g.compute_dt();
    g.is_cash_flow_.assign(g.times_.size(), false);
    g.is_collateral_call_.assign(g.times_.size(), false);
    return g;
}

TimeGrid TimeGrid::weekly(YearFraction horizon) {
    TimeGrid g;
    int steps = static_cast<int>(std::ceil(horizon * 52.0));
    g.times_.reserve(steps + 1);
    g.times_.push_back(0.0);
    for (int i = 1; i <= steps; ++i) {
        double t = static_cast<double>(i) / 52.0;
        g.times_.push_back(std::min(t, horizon));
    }
    g.deduplicate_and_sort();
    g.compute_dt();
    g.is_cash_flow_.assign(g.times_.size(), false);
    g.is_collateral_call_.assign(g.times_.size(), false);
    return g;
}

TimeGrid TimeGrid::daily(YearFraction horizon) {
    TimeGrid g;
    int steps = static_cast<int>(std::ceil(horizon * 252.0));
    g.times_.reserve(steps + 1);
    g.times_.push_back(0.0);
    for (int i = 1; i <= steps; ++i) {
        double t = static_cast<double>(i) / 252.0;
        g.times_.push_back(std::min(t, horizon));
    }
    g.deduplicate_and_sort();
    g.compute_dt();
    g.is_cash_flow_.assign(g.times_.size(), false);
    g.is_collateral_call_.assign(g.times_.size(), false);
    return g;
}

TimeGrid TimeGrid::parsimonious(
    YearFraction               horizon,
    int                        mpor_days,
    const std::vector<double>& cash_flow_dates)
{
    // TODO: implement full Silotto et al. joint grid.
    // Stub: monthly primary + cash flow dates union.
    TimeGrid g = monthly(horizon);

    const double mpor_yrs = static_cast<double>(mpor_days) / 365.0;

    // Merge cash flow dates and their MPoR-shifted collateral-call dates.
    for (double cf : cash_flow_dates) {
        if (cf > 0.0 && cf <= horizon) {
            g.times_.push_back(cf);
            double cc = cf - mpor_yrs;
            if (cc > 0.0) g.times_.push_back(cc);
        }
    }

    g.deduplicate_and_sort();
    g.compute_dt();

    // Mark cash-flow and collateral-call steps.
    std::size_t n = g.times_.size();
    g.is_cash_flow_.assign(n, false);
    g.is_collateral_call_.assign(n, false);

    for (double cf : cash_flow_dates) {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::abs(g.times_[i] - cf) < 1e-9)
                g.is_cash_flow_[i] = true;
            double cc = cf - mpor_yrs;
            if (std::abs(g.times_[i] - cc) < 1e-9)
                g.is_collateral_call_[i] = true;
        }
    }

    return g;
}

// ─── Accessors ───────────────────────────────────────────────────────────────

bool TimeGrid::is_cash_flow_date(int step) const noexcept {
    if (step < 0 || step >= static_cast<int>(is_cash_flow_.size())) return false;
    return is_cash_flow_[step];
}

bool TimeGrid::is_collateral_call_date(int step) const noexcept {
    if (step < 0 || step >= static_cast<int>(is_collateral_call_.size())) return false;
    return is_collateral_call_[step];
}

// ─── Internal helpers ────────────────────────────────────────────────────────

void TimeGrid::compute_dt() {
    std::size_t n = times_.size();
    dt_.resize(n > 0 ? n - 1 : 0);
    for (std::size_t i = 0; i + 1 < n; ++i)
        dt_[i] = times_[i + 1] - times_[i];
}

void TimeGrid::deduplicate_and_sort() {
    std::sort(times_.begin(), times_.end());
    times_.erase(std::unique(times_.begin(), times_.end(),
        [](double a, double b){ return std::abs(a - b) < 1e-9; }),
        times_.end());
}

} // namespace ccr
