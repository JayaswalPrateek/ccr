#pragma once

// ============================================================================
// ccr/time_grid.hpp  —  Simulation time-grid construction
//
// Implements the parsimonious joint grid (Silotto, Scaringi & Bianchetti, 2023):
//   primary grid (monthly) ∪ cash-flow dates ∪ collateral call dates
//
// This captures MPoR exposure spikes at <1% relative PFE error vs. daily,
// while achieving a 26× reduction in simulation overhead.
//
// Also supports uniform MONTHLY / WEEKLY / DAILY grids for comparison.
//
// Dependencies: types (for GridType, YearFraction).
// ============================================================================

#include <vector>

#include "ccr/types.hpp"

namespace ccr {

// ─── TimeGrid ────────────────────────────────────────────────────────────────

class TimeGrid {
public:
	/// Build the time grid from parameters.
	///
	/// @param horizon_years    Simulation horizon T_max
	/// @param grid_type        MONTHLY / WEEKLY / DAILY / PARSIMONIOUS
	/// @param mpor_days        Margin Period of Risk (shifts collateral-call dates)
	/// @param cash_flow_dates  Additional dates from contract schedule (years)
	TimeGrid(
		YearFraction horizon_years,
		GridType grid_type,
		int mpor_days = 10,
		const std::vector<double> &cash_flow_dates = {});

	// ── Accessors ────────────────────────────────────────────────────────────

	/// Sorted vector of timestep boundaries t_0=0, t_1, …, t_T in years.
	const std::vector<double> &times() const noexcept { return times_; }

	/// Number of simulation steps T (= times().size() - 1).
	int num_steps() const noexcept { return static_cast<int>(times_.size()) - 1; }

	/// Δt_i = t_{i+1} − t_i for step i ∈ [0, T-1].
	const std::vector<double> &dt() const noexcept { return dt_; }

	/// True if step i corresponds to a cash-flow date (MPoR spike candidate).
	bool is_cash_flow_date(int step) const noexcept;

	/// True if step i corresponds to a collateral-call date (t − l).
	bool is_collateral_call_date(int step) const noexcept;

	// ── Static constructors ──────────────────────────────────────────────────

	static TimeGrid monthly(YearFraction horizon);
	static TimeGrid weekly(YearFraction horizon);
	static TimeGrid daily(YearFraction horizon);

	/// Parsimonious joint grid: primary + cash-flow + collateral-call dates.
	static TimeGrid parsimonious(
		YearFraction horizon,
		int mpor_days,
		const std::vector<double> &cash_flow_dates);

private:
	std::vector<double> times_;
	std::vector<double> dt_;
	std::vector<bool> is_cash_flow_;
	std::vector<bool> is_collateral_call_;

	TimeGrid() = default;  ///< Used by static factory methods only.
	void compute_dt();
	void deduplicate_and_sort();
};

}  // namespace ccr
