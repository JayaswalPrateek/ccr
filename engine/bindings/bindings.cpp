// ============================================================================
// engine/bindings/bindings.cpp
//
// pybind11 Python bindings for the CCR engine.
//
// Python usage:
//   import _ccr_engine as ccr
//
//   cfg = ccr.EngineConfig()
//   cfg.sim_params.num_paths = 10_000
//   cfg.counterparty.hazard_rate = 0.03
//
//   engine = ccr.CcrEngine()
//   result = engine.run(cfg)
//
//   print(result.base.cva)
//   print(result.base.pfe_profile)
//
// All structs are exposed with read-write attributes mirroring types.hpp.
// Enums are exposed as Python IntEnum-compatible values.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "ccr/types.hpp"
#include "ccr/ccr_engine.hpp"

namespace py = pybind11;
using namespace ccr;

PYBIND11_MODULE(_ccr_engine, m) {
    m.doc() = "Real-Time Counterparty Credit Risk & Margin Engine (C++ backend)";

    // ── Enumerations ─────────────────────────────────────────────────────────

    py::enum_<DerivativeType>(m, "DerivativeType")
        .value("IRS",       DerivativeType::IRS)
        .value("CDS",       DerivativeType::CDS)
        .value("FX",        DerivativeType::FX)
        .value("EQUITY",    DerivativeType::EQUITY)
        .value("COMMODITY", DerivativeType::COMMODITY)
        .export_values();

    py::enum_<CreditRating>(m, "CreditRating")
        .value("AAA", CreditRating::AAA)
        .value("AA",  CreditRating::AA)
        .value("A",   CreditRating::A)
        .value("BBB", CreditRating::BBB)
        .value("BB",  CreditRating::BB)
        .value("B",   CreditRating::B)
        .value("CCC", CreditRating::CCC)
        .value("D",   CreditRating::D)
        .export_values();

    py::enum_<MarginCallStatus>(m, "MarginCallStatus")
        .value("PENDING",      MarginCallStatus::PENDING)
        .value("SENT",         MarginCallStatus::SENT)
        .value("ACKNOWLEDGED", MarginCallStatus::ACKNOWLEDGED)
        .value("RECEIVED",     MarginCallStatus::RECEIVED)
        .value("OVERDUE",      MarginCallStatus::OVERDUE)
        .value("ESCALATED",    MarginCallStatus::ESCALATED)
        .value("DISMISSED",    MarginCallStatus::DISMISSED)
        .export_values();

    py::enum_<SimMode>(m, "SimMode")
        .value("REGULATORY",   SimMode::REGULATORY)
        .value("STANDARD",     SimMode::STANDARD)
        .value("APPROX_FAST",  SimMode::APPROX_FAST)
        .export_values();

    py::enum_<GridType>(m, "GridType")
        .value("MONTHLY",      GridType::MONTHLY)
        .value("WEEKLY",       GridType::WEEKLY)
        .value("DAILY",        GridType::DAILY)
        .value("PARSIMONIOUS", GridType::PARSIMONIOUS)
        .export_values();

    // ── SimParams ─────────────────────────────────────────────────────────────

    py::class_<SimParams>(m, "SimParams")
        .def(py::init<>())
        .def_readwrite("num_paths",     &SimParams::num_paths)
        .def_readwrite("num_timesteps", &SimParams::num_timesteps)
        .def_readwrite("num_assets",    &SimParams::num_assets)
        .def_readwrite("mu",            &SimParams::mu)
        .def_readwrite("sigma",         &SimParams::sigma)
        .def_readwrite("rho_wwr",       &SimParams::rho_wwr)
        .def_readwrite("recovery_rate", &SimParams::recovery_rate)
        .def_readwrite("horizon_years", &SimParams::horizon_years)
        .def_readwrite("mode",          &SimParams::mode)
        .def_readwrite("grid_type",     &SimParams::grid_type)
        .def("__repr__", [](const SimParams& p) {
            return "<SimParams paths=" + std::to_string(p.num_paths)
                 + " sigma=" + std::to_string(p.sigma)
                 + " horizon=" + std::to_string(p.horizon_years) + "y>";
        });

    // ── StressScenario ───────────────────────────────────────────────────────

    py::class_<StressScenario>(m, "StressScenario")
        .def(py::init<>())
        .def_readwrite("vol_shock",            &StressScenario::vol_shock)
        .def_readwrite("fx_shock",             &StressScenario::fx_shock)
        .def_readwrite("equity_shock",         &StressScenario::equity_shock)
        .def_readwrite("interest_rate_shock",  &StressScenario::interest_rate_shock)
        .def_readwrite("credit_spread_shock",  &StressScenario::credit_spread_shock)
        .def_readwrite("hazard_rate_shock",    &StressScenario::hazard_rate_shock)
        .def_readwrite("jump_amplitude",       &StressScenario::jump_amplitude)
        .def_readwrite("label",                &StressScenario::label);

    // ── CounterpartyConfig ───────────────────────────────────────────────────

    py::class_<CounterpartyConfig>(m, "CounterpartyConfig")
        .def(py::init<>())
        .def_readwrite("id",               &CounterpartyConfig::id)
        .def_readwrite("name",             &CounterpartyConfig::name)
        .def_readwrite("credit_rating",    &CounterpartyConfig::credit_rating)
        .def_readwrite("hazard_rate",      &CounterpartyConfig::hazard_rate)
        .def_readwrite("recovery_rate",    &CounterpartyConfig::recovery_rate)
        .def_readwrite("collateral",       &CounterpartyConfig::collateral)
        .def_readwrite("margin_threshold", &CounterpartyConfig::margin_threshold)
        .def_readwrite("mpor_days",        &CounterpartyConfig::mpor_days);

    // ── DerivativeSpec ───────────────────────────────────────────────────────

    py::class_<DerivativeSpec>(m, "DerivativeSpec")
        .def(py::init<>())
        .def_readwrite("id",               &DerivativeSpec::id)
        .def_readwrite("type",             &DerivativeSpec::type)
        .def_readwrite("notional",         &DerivativeSpec::notional)
        .def_readwrite("maturity_years",   &DerivativeSpec::maturity_years)
        .def_readwrite("underlying_price", &DerivativeSpec::underlying_price)
        .def_readwrite("strike",           &DerivativeSpec::strike)
        .def_readwrite("cash_flow_freq",   &DerivativeSpec::cash_flow_freq);

    // ── PortfolioConfig ──────────────────────────────────────────────────────

    py::class_<PortfolioConfig>(m, "PortfolioConfig")
        .def(py::init<>())
        .def_readwrite("id",               &PortfolioConfig::id)
        .def_readwrite("counterparty_id",  &PortfolioConfig::counterparty_id)
        .def_readwrite("derivatives",      &PortfolioConfig::derivatives)
        .def_readwrite("collateral",       &PortfolioConfig::collateral)
        .def_readwrite("net_value",        &PortfolioConfig::net_value);

    // ── EngineConfig ─────────────────────────────────────────────────────────

    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("sim_params",             &EngineConfig::sim_params)
        .def_readwrite("counterparty",           &EngineConfig::counterparty)
        .def_readwrite("portfolio",              &EngineConfig::portfolio)
        .def_readwrite("stress",                 &EngineConfig::stress)
        .def_readwrite("enable_wwr",             &EngineConfig::enable_wwr)
        .def_readwrite("enable_jump_diffusion",  &EngineConfig::enable_jump_diffusion)
        .def_readwrite("enable_collateral",      &EngineConfig::enable_collateral)
        .def_readwrite("deterministic_quantile", &EngineConfig::deterministic_quantile)
        .def_readwrite("log_overflow_warnings",  &EngineConfig::log_overflow_warnings)
        .def_readwrite("rng_seed",               &EngineConfig::rng_seed);

    // ── RiskMetrics ──────────────────────────────────────────────────────────

    py::class_<RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readonly("cva",              &RiskMetrics::cva)
        .def_readonly("wwr_cva",          &RiskMetrics::wwr_cva)
        .def_readonly("margin_required",  &RiskMetrics::margin_required)
        .def_readonly("pfe_profile",      &RiskMetrics::pfe_profile)
        .def_readonly("epe_profile",      &RiskMetrics::epe_profile)
        .def_readonly("time_grid_years",  &RiskMetrics::time_grid_years)
        .def_readonly("compute_time_us",  &RiskMetrics::compute_time_us)
        .def_readonly("overflow_detected",&RiskMetrics::overflow_detected)
        .def_readonly("arch_used",        &RiskMetrics::arch_used)
        .def_readonly("paths_used",       &RiskMetrics::paths_used)
        .def("__repr__", [](const RiskMetrics& r) {
            return "<RiskMetrics cva=" + std::to_string(r.cva)
                 + " arch=" + r.arch_used + ">";
        });

    // ── CcrResult ────────────────────────────────────────────────────────────

    py::class_<CcrResult>(m, "CcrResult")
        .def(py::init<>())
        .def_readonly("base",      &CcrResult::base)
        .def_readonly("stressed",  &CcrResult::stressed)
        .def_readonly("success",   &CcrResult::success)
        .def_readonly("error_msg", &CcrResult::error_msg)
        .def("__repr__", [](const CcrResult& r) {
            return std::string("<CcrResult success=")
                 + (r.success ? "True" : "False")
                 + (r.error_msg.empty() ? "" : " error=" + r.error_msg)
                 + ">";
        });

    // ── MarginCallInfo ───────────────────────────────────────────────────────

    py::class_<MarginCallInfo>(m, "MarginCallInfo")
        .def(py::init<>())
        .def_readonly("id",               &MarginCallInfo::id)
        .def_readonly("counterparty_id",  &MarginCallInfo::counterparty_id)
        .def_readonly("amount",           &MarginCallInfo::amount)
        .def_readonly("excess_exposure",  &MarginCallInfo::excess_exposure)
        .def_readonly("status",           &MarginCallInfo::status)
        .def_readonly("triggered_at_ms",  &MarginCallInfo::triggered_at_ms)
        .def_readonly("due_by_ms",        &MarginCallInfo::due_by_ms)
        .def_readonly("reason",           &MarginCallInfo::reason);

    // ── CcrEngine ────────────────────────────────────────────────────────────

    py::class_<CcrEngine>(m, "CcrEngine")
        .def(py::init<>())

        // Main entry point.
        // Optional callback: callable(timestep: int, total: int, pfe_so_far: float)
        .def("run",
            [](CcrEngine& self,
               const EngineConfig& config,
               std::optional<py::function> callback_fn)
            {
                std::optional<ProgressCallback> cb;
                if (callback_fn.has_value()) {
                    cb = [fn = *callback_fn](int t, int total, double pfe) {
                        fn(t, total, pfe);
                    };
                }
                // Release the GIL during the compute-intensive C++ run.
                CcrResult result;
                {
                    py::gil_scoped_release release;
                    result = self.run(config, cb);
                }
                return result;
            },
            py::arg("config"),
            py::arg("callback") = py::none(),
            "Run the full CCR pipeline. Returns CcrResult.\n"
            "callback(timestep, total, pfe_so_far) is called after each timestep.")

        // Validation helper.
        .def_static("validate_config",
            &CcrEngine::validate_config,
            py::arg("config"),
            "Returns empty string on success, error message on failure.")

        // Introspection.
        .def_static("active_arch",
            []() { return std::string(CcrEngine::active_arch()); },
            "SIMD architecture the engine was compiled for (e.g. 'AVX2').")

        .def_static("simd_width",
            []() { return static_cast<int>(CcrEngine::simd_width()); },
            "Number of doubles processed per SIMD register.")

        .def_static("estimate_arena_bytes",
            &CcrEngine::estimate_arena_bytes,
            py::arg("config"),
            "Estimated arena allocation in bytes for the given config.")

        // Margin call evaluation.
        .def_static("evaluate_margin_call",
            &CcrEngine::evaluate_margin_call,
            py::arg("result"),
            py::arg("counterparty"),
            "Returns MarginCallInfo if PFE exceeds threshold, else None.");

    // ── Module-level helpers ─────────────────────────────────────────────────

    m.def("active_arch",
          []() { return std::string(ActiveArch::NAME); },
          "Compiled SIMD target (e.g. 'AVX2', 'ARM NEON', 'Scalar').");

    m.def("simd_width",
          []() { return static_cast<int>(ActiveArch::WIDTH); },
          "Doubles per SIMD register for the active architecture.");
}
