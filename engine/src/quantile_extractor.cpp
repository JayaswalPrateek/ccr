// ============================================================================
// engine/src/quantile_extractor.cpp
//
// PFE (α-quantile) and EPE (mean positive exposure) extraction.
// Stub: uses std::nth_element with deterministic pivot fallback.
// The custom introselect for REGULATORY mode is a TODO.
// ============================================================================

#include "ccr/quantile_extractor.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstddef>

namespace ccr {

// ─── Single step ─────────────────────────────────────────────────────────────

double extract_pfe(
    std::span<double> exposures_col,
    double            alpha,
    bool              /*deterministic*/) noexcept
{
    if (exposures_col.empty()) return 0.0;

    // TODO: deterministic == true → use custom introselect with
    //       median-of-three pivot (first/middle/last) for bitwise reproducibility.
    // Stub: std::nth_element (average O(N), not bitwise reproducible).
    const std::size_t k = static_cast<std::size_t>(
        static_cast<double>(exposures_col.size()) * alpha);
    const std::size_t idx = k < exposures_col.size() ? k : exposures_col.size() - 1;

    std::nth_element(exposures_col.begin(),
                     exposures_col.begin() + static_cast<std::ptrdiff_t>(idx),
                     exposures_col.end());
    return exposures_col[idx];
}

double extract_epe(std::span<const double> exposures_col) noexcept {
    if (exposures_col.empty()) return 0.0;
    double sum   = 0.0;
    std::size_t n = 0;
    for (double e : exposures_col) {
        if (e > 0.0) { sum += e; ++n; }
    }
    return n > 0 ? sum / static_cast<double>(exposures_col.size()) : 0.0;
    // EPE = mean of max(V,0) across ALL paths (not just positive), so divide
    // by total M, not by count of positive paths.
}

// ─── Full profile ────────────────────────────────────────────────────────────

void extract_profiles(
    std::span<double>  exposures,
    std::span<double>  pfe_profile,
    std::span<double>  epe_profile,
    int T, int M, int M_padded,
    double alpha,
    bool   deterministic) noexcept
{
    for (int t = 0; t < T; ++t) {
        std::span<double> col{
            exposures.data() + static_cast<std::size_t>(t) * M_padded,
            static_cast<std::size_t>(M)};
        // EPE must be computed BEFORE nth_element modifies order.
        epe_profile[t] = extract_epe(col);
        pfe_profile[t] = extract_pfe(col, alpha, deterministic);
    }
}

// ─── Histogram ───────────────────────────────────────────────────────────────

std::vector<std::size_t> exposure_histogram(
    std::span<const double> exposures_col,
    int    num_bins,
    double max_exposure) noexcept
{
    std::vector<std::size_t> bins(num_bins, 0);
    if (exposures_col.empty() || num_bins <= 0) return bins;

    double hi = max_exposure;
    if (hi <= 0.0) {
        for (double e : exposures_col) if (e > hi) hi = e;
    }
    if (hi <= 0.0) return bins;

    for (double e : exposures_col) {
        if (e <= 0.0) continue;
        int b = static_cast<int>(e / hi * num_bins);
        if (b >= num_bins) b = num_bins - 1;
        ++bins[b];
    }
    return bins;
}

} // namespace ccr
