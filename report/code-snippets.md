# CCR Engine — Annotated Code Snippets

> This document walks through the most important source code in the CCR Engine, ordered so that a reader going top-down follows the full pipeline: types → SIMD layer → random numbers → Monte Carlo paths → CVA → Python bindings → REST/WebSocket API → SA-CCR → database models.
>
> Every snippet is taken verbatim from the codebase. Line references are approximate; see the actual files for full context.

---

## 1. Pipeline Overview

The CCR Engine is a three-tier system:

```
  Browser (SvelteKit / TypeScript)
       │  REST (JSON) + WebSocket
  FastAPI server (Python 3.12)
       │  pybind11 in-process call — GIL released during compute
  C++20 Monte Carlo engine
       │
  PostgreSQL 15 + TimescaleDB
```

The C++ engine performs four stages for each simulation run:

1. **Scenario generation** — time grid, PRNG initialisation, Cholesky decomposition
2. **Path simulation** — GBM evolution over every path × timestep (SIMD hot loop)
3. **Portfolio valuation** — MTM per derivative type, exposure = max(V, 0)
4. **XVA integration** — PFE/EPE extraction, CVA with Kahan summation, margin evaluation

---

## 2. Core Data Types

**File:** `engine/include/ccr/types.hpp`

These structs flow from the Python layer down into C++ and back. Knowing them makes all other code readable.

```cpp
// ── Simulation parameters sent by the user ───────────────────────────────────
struct SimParams {
    int    num_paths      = 10'000; // Monte Carlo paths (rows in the SoA arrays)
    int    num_timesteps  = 50;     // time steps T (columns)
    int    num_assets     = 1;      // number of correlated assets K
    double mu             = 0.05;   // risk-neutral drift (annualised)
    double sigma          = 0.20;   // volatility (annualised)
    double rho_wwr        = 0.0;    // correlation ρ for Wrong-Way Risk
    double recovery_rate  = 0.40;   // R — fraction recovered on default
    double horizon_years  = 5.0;    // simulation horizon T
    SimMode  mode         = SimMode::REGULATORY; // REGULATORY / STANDARD / APPROX_FAST
    GridType grid_type    = GridType::PARSIMONIOUS; // time-grid spacing
};

// ── Results returned after each run ──────────────────────────────────────────
struct RiskMetrics {
    double cva             = 0.0;  // Credit Valuation Adjustment ($)
    double wwr_cva         = 0.0;  // CVA adjusted for Wrong-Way Risk
    double margin_required = 0.0;  // Variation Margin = max(0, max_PFE − collateral)

    std::vector<double> pfe_profile;      // PFE(t) at 99th percentile, length T
    std::vector<double> epe_profile;      // EPE(t) = mean(max(V,0)), length T
    std::vector<double> time_grid_years;  // time points t_1..t_T

    std::chrono::microseconds compute_time_us{0};
    bool        overflow_detected = false;
    std::string arch_used;   // "AVX2", "ARM NEON", "Scalar" …
    int         paths_used  = 0;
};

// ── Top-level result (base + optional stressed pass) ─────────────────────────
struct CcrResult {
    RiskMetrics             base;
    std::optional<RiskMetrics> stressed; // populated when stress scenario is set
    bool        success   = false;
    std::string error_msg;
};

// ── Counterparty credit parameters ───────────────────────────────────────────
struct CounterpartyConfig {
    std::string   id;
    std::string   name;
    CreditRating  credit_rating    = CreditRating::BBB;
    double        hazard_rate      = 0.02;  // λ — instantaneous default intensity
    double        recovery_rate    = 0.40;
    double        collateral       = 0.0;
    double        margin_threshold = 0.0;   // dead-band before a margin call fires
    int           mpor_days        = 10;    // Margin Period of Risk (regulatory minimum)
};

// ── Stress scenario overlays applied in a second engine pass ─────────────────
struct StressScenario {
    double vol_shock           = 0.0;  // additive Δσ
    double fx_shock            = 0.0;
    double equity_shock        = 0.0;
    double interest_rate_shock = 0.0;
    double credit_spread_shock = 0.0;
    double hazard_rate_shock   = 0.0;  // additive Δλ
    double jump_amplitude      = 0.0;  // J for jump-at-default
    std::string label;
};
```

---

## 3. Policy-Based SIMD Abstraction

**File:** `engine/include/ccr/simd_abstraction.hpp`

The hot loop is templated on an `Arch` policy type. All platform-specific intrinsics are confined to `SimdOps<Arch>` specialisations. The loop body contains **zero `#ifdef` blocks**.

```cpp
// ── Zero-size tag types — resolved 100% at compile time ──────────────────────
struct ScalarArch {
    static constexpr std::size_t WIDTH = 1;
    using reg_t                        = double;
    static constexpr const char* NAME  = "Scalar";
};

struct Avx2Arch {
    static constexpr std::size_t WIDTH = 4;   // 256-bit / 64-bit = 4 doubles
    using reg_t                        = __m256d;
    static constexpr const char* NAME  = "AVX2";
};

struct Avx512Arch {
    static constexpr std::size_t WIDTH = 8;   // 512-bit / 64-bit = 8 doubles
    using reg_t                        = __m512d;
    static constexpr const char* NAME  = "AVX-512";
};

struct NeonArch {
    static constexpr std::size_t WIDTH = 2;   // 128-bit / 64-bit = 2 doubles
    using reg_t                        = float64x2_t;
    static constexpr const char* NAME  = "ARM NEON";
};

// ── Compile-time arch selection ───────────────────────────────────────────────
#if defined(__AVX512F__)
using ActiveArch = Avx512Arch;
#elif defined(__AVX2__)
using ActiveArch = Avx2Arch;
#elif defined(__ARM_NEON)
using ActiveArch = NeonArch;
#else
using ActiveArch = ScalarArch;
#endif

// ── Primary template (scalar) ─────────────────────────────────────────────────
// Every SIMD platform specialises these operations; the loop body calls them
// identically regardless of architecture.
template <>
struct SimdOps<ScalarArch> {
    using reg_t = double;
    static inline reg_t load(const double* p)            { return *p; }
    static inline void  store(double* p, reg_t v)        { *p = v; }
    static inline reg_t set1(double v)                   { return v; }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c){ return a * b + c; } // a*b+c
    static inline reg_t max(reg_t a, reg_t b)            { return a > b ? a : b; }
    static inline reg_t exp_approx(reg_t v)              { return std::exp(v); }
    static inline void  fence()                          {}  // no-op on scalar
};

// ── AVX2 specialisation ───────────────────────────────────────────────────────
// Four doubles per register. _mm256_fmadd_pd is a single fused multiply-add
// instruction (one cycle on Haswell+), twice as fast as separate mul + add.
template <>
struct SimdOps<Avx2Arch> {
    using reg_t = __m256d;
    static inline reg_t load(const double* p)             { return _mm256_load_pd(p); }
    static inline void  store(double* p, reg_t v)         { _mm256_store_pd(p, v); }
    static inline reg_t set1(double v)                    { return _mm256_set1_pd(v); }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c)  { return _mm256_fmadd_pd(a, b, c); }
    static inline reg_t max(reg_t a, reg_t b)             { return _mm256_max_pd(a, b); }
    static inline void  fence()                           { _mm256_zeroupper(); } // required before non-AVX calls
    static reg_t exp_approx(reg_t v); // 6th-order minimax polynomial — defined in .cpp
};

// ── ARM NEON specialisation ───────────────────────────────────────────────────
// Used on the demo machine (Apple Silicon M-series).
template <>
struct SimdOps<NeonArch> {
    using reg_t = float64x2_t;
    static inline reg_t load(const double* p)             { return vld1q_f64(p); }
    static inline void  store(double* p, reg_t v)         { vst1q_f64(p, v); }
    static inline reg_t fmadd(reg_t a, reg_t b, reg_t c)  { return vfmaq_f64(c, a, b); }
    static inline reg_t max(reg_t a, reg_t b)             { return vmaxq_f64(a, b); }
    static inline void  fence()                           {} // no-op on ARM
    static reg_t exp_approx(reg_t v);
};

// ── Helper: round path count up to SIMD width ─────────────────────────────────
// Ensures no partial register loads at the end of a path array.
template <typename Arch = ActiveArch>
constexpr std::size_t pad_to_width(std::size_t m) {
    constexpr std::size_t W = Arch::WIDTH;
    return ((m + W - 1) / W) * W;
}
```

---

## 4. Pseudo-Random Number Generator

**File:** `engine/src/rng_engine.cpp`

The engine uses **xoroshiro128aox** — a 128-bit state, 64-bit output generator that passes BigCrush. The AOX (AND-OR-XOR) scrambler fixes the lower-bit weakness of the simpler xoroshiro128+.

```cpp
// ── Core state transition and output ─────────────────────────────────────────
uint64_t Xoroshiro128aox::next_u64() noexcept {
    const uint64_t s0 = s_[0];
    uint64_t       s1 = s_[1];

    // AOX output function: bitwise AND-OR-XOR scrambler + rotation
    // Fixes the weakness of xoroshiro128+ (low bits fail linear-complexity tests)
    const uint64_t result = (s0 & s1) | rotl(s0 ^ s1, 7);

    // State update (shift register constants a=24, b=16, c=37)
    s1     ^= s0;
    s_[0]  = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    s_[1]  = rotl(s1, 37);

    return result;
}

// ── Convert 64-bit integer to double in [0, 1) ───────────────────────────────
// Drop 11 bits (exponent + sign), multiply by 2^-53.
// This is the standard IEEE 754 trick — guaranteed uniform on 2^53 dyadic rationals.
double Xoroshiro128aox::next_double() noexcept {
    return static_cast<double>(next_u64() >> 11) * (1.0 / (UINT64_C(1) << 53));
}

// ── Parallel streams via jump polynomial ─────────────────────────────────────
// Each call to jump() advances the state by 2^64 steps in the sequence.
// This creates non-overlapping streams for multi-threaded simulation:
// thread 0 uses positions [0, 2^64), thread 1 uses [2^64, 2×2^64), etc.
std::vector<Xoroshiro128aox> make_thread_rngs(uint64_t base_seed, int num_threads) {
    std::vector<Xoroshiro128aox> rngs;
    rngs.emplace_back(base_seed);
    for (int i = 1; i < num_threads; ++i) {
        const auto& prev = rngs.back();
        Xoroshiro128aox next(prev.state0(), prev.state1());
        next.jump();   // advance 2^64 steps — guaranteed non-overlapping
        rngs.push_back(std::move(next));
    }
    return rngs;
}
```

---

## 5. GBM Path Simulation — the SIMD Hot Loop

**File:** `engine/src/path_simulator.cpp`

Paths are stored in **Structure-of-Arrays (SoA)** layout: all spot prices for all M paths of asset k are contiguous in memory. This allows SIMD to load `WIDTH` paths per instruction.

The GBM equation evolved at each timestep is:

> **S(t+Δt) = S(t) · exp((μ − σ²/2)·Δt + σ·√Δt · Z)**

where Z ~ N(0,1). The drift term is pre-computed once; the hot loop only does: load, fmadd, exp, store.

```cpp
// ── One GBM timestep for all paths × assets ───────────────────────────────────
// Arch template resolves at compile time — no runtime branch or vtable.
template <typename Arch>
void PathSimulator::evolve_step(PathState& state,
    std::span<const double> correlated_normals,
    double drift, double vol_dt, int timestep) noexcept
{
    constexpr std::size_t step = Arch::WIDTH;  // e.g. 4 for AVX2

    const auto drift_v  = SimdOps<Arch>::set1(drift);   // broadcast scalar to all lanes
    const auto vol_dt_v = SimdOps<Arch>::set1(vol_dt);  // σ√Δt pre-computed once

    for (int k = 0; k < K; ++k) {
        const double* z_k    = correlated_normals.data() + k * M_padded_;
        double*       spot_k = state.spot_prices        + k * M_padded_;

        // Process WIDTH paths per iteration — innermost loop is SIMD-vectorised
        for (std::size_t m = 0; m < M_padded_; m += step) {
            auto z_v   = SimdOps<Arch>::load(z_k    + m);   // load WIDTH random normals
            auto s_v   = SimdOps<Arch>::load(spot_k  + m);   // load WIDTH spot prices

            // GBM increment: drift + vol × Z
            auto arg_v = SimdOps<Arch>::fmadd(z_v, vol_dt_v, drift_v);  // z*vol_dt + drift
            // S_new = S_old × exp(drift + vol*Z)
            auto s_new = SimdOps<Arch>::mul(s_v, SimdOps<Arch>::exp_approx(arg_v));
            SimdOps<Arch>::store(spot_k + m, s_new);
        }
    }

    // After updating spot prices, compute MTM per derivative type:
    // IRS:    V = notional × DV01 × (S_t − strike)
    // CDS:    V = notional × (1−R) × (S_t − strike) × t_rem × df
    // Equity/FX: V = notional × (S_t − strike × df)
    // Exposure E(t) = max(V_net, 0)  — stored into state.exposures[t * M_padded + m]
}

// ── Outer loop: all timesteps ─────────────────────────────────────────────────
template <typename Arch>
void PathSimulator::run_all_steps(PathState& state, Xoroshiro128aox& rng) noexcept {
    for (int t = 0; t < T_; ++t) {
        fill_normal<Arch>(norms_, rng, params_.mode);   // 1. Generate K×M N(0,1) variates
        cholesky_.apply<Arch>(...);                     // 2. Apply Cholesky for WWR coupling
        evolve_step<Arch>(state, ...);                  // 3. GBM step
        if (jump_hook_)                                 // 4. Post-diffusion jump overlay
            jump_hook_->on_paths_complete(...);
        compute_exposures_step<Arch>(...);              // 5. E(t) = max(V_net, 0)
    }
}
```

---

## 6. CVA Integration — Kahan Compensated Summation

**File:** `engine/src/cva_integrator.cpp`

CVA is the present value of expected credit losses:

> **CVA = (1 − R) · Σᵢ EPE(tᵢ) · PD(tᵢ₋₁, tᵢ)**

where PD(tᵢ₋₁, tᵢ) is the marginal default probability in interval i.

Summation uses **Kahan compensated addition** to maintain full double precision regardless of path count. This is a regulatory requirement: CVA must be bitwise reproducible across runs.

```cpp
// ── Kahan dot product: Σ a[i] × b[i], compensated ────────────────────────────
// Standard floating-point addition accumulates O(n·ε) error; Kahan keeps it O(ε).
static double kahan_dot(
    std::span<const double> a,
    std::span<const double> b) noexcept
{
    double sum  = 0.0;
    double comp = 0.0;     // running compensation for lost low-order bits

    for (std::size_t i = 0; i < a.size(); ++i) {
        double y = a[i] * b[i] - comp;   // subtract the running error
        double t = sum + y;
        comp     = (t - sum) - y;        // capture the bits lost in (sum + y)
        sum      = t;
    }
    return sum;
}

// ── CVA scalar ────────────────────────────────────────────────────────────────
double compute_cva(
    std::span<const double> epe_profile,   // EPE(t_i) for i=1..T
    std::span<const double> marginal_pd,   // PD(t_{i-1}, t_i) for i=1..T
    double                  recovery_rate) noexcept
{
    if (epe_profile.size() != marginal_pd.size()) return 0.0;
    return (1.0 - recovery_rate) * kahan_dot(epe_profile, marginal_pd);
}

// ── Marginal PD from constant hazard rate λ ───────────────────────────────────
// Under a constant-intensity model: PD(t_{i-1}, t_i) = e^{-λ·t_{i-1}} − e^{-λ·t_i}
std::vector<double> marginal_pd_from_flat_hazard(
    std::span<const double> time_grid,
    double                  hazard_rate)
{
    std::vector<double> pd;
    const std::size_t T = time_grid.size();
    pd.reserve(T - 1);
    for (std::size_t i = 1; i < T; ++i) {
        double p = std::exp(-hazard_rate * time_grid[i-1])
                 - std::exp(-hazard_rate * time_grid[i]);
        pd.push_back(p > 0.0 ? p : 0.0);
    }
    return pd;
}

// ── Required variation margin ─────────────────────────────────────────────────
// Basel III: the bank must hold enough VM to cover peak exposure minus posted collateral.
double compute_required_margin(
    std::span<const double> pfe_profile,
    double                  current_collateral) noexcept
{
    double max_pfe = 0.0;
    for (double p : pfe_profile) if (p > max_pfe) max_pfe = p;
    double margin = max_pfe - current_collateral;
    return margin > 0.0 ? margin : 0.0;
}
```

---

## 7. Jump-at-Default Extension

**File:** `engine/include/ccr/jump_diffusion.hpp` and `engine/src/jump_diffusion.cpp`

Wrong-Way Risk at default: when a counterparty defaults, their reference assets spike downward, amplifying the bank's exposure. This is modelled as a multiplicative jump `S → S × (1 + J)` applied at the first timestep after the simulated default time τ.

**Design**: the GBM hot loop is kept branch-free. The jump is a **post-processing pass** run after all paths are evolved.

```cpp
// ── Sample default times τ_m ~ Exp(λ) for each path ─────────────────────────
// τ = −ln(U) / λ  where U ~ Uniform(0,1)
// Paths where τ > horizon are non-defaulting (stored as +inf).
std::vector<double> sample_default_times(
    std::span<const double> rng_uniforms,
    double                  lambda,
    double                  horizon);

// ── Hook invoked after GBM diffusion, before exposure computation ─────────────
// The base class is a no-op (zero-cost when jump is disabled).
class JumpDiffusionHook {
public:
    virtual void on_paths_complete(
        std::span<double>          spot_prices,
        std::span<const double>    default_times,
        const std::vector<double>& time_grid,
        int K, int M, int M_padded) {}
};

// ── Concrete implementation: S → S × (1 + J) at the default timestep ─────────
// "jump_size" J = 0.05 → 5% spike.
// Salonen (2023) calibration: J=1% → CVA×2.15; J=5% → CVA×9; J=10% → CVA×18.
class MultiplicativeJumpHook final : public JumpDiffusionHook {
public:
    explicit MultiplicativeJumpHook(JumpParams params) : params_(params) {}

    void on_paths_complete(
        std::span<double>       spot_prices,
        std::span<const double> default_times,
        const std::vector<double>& time_grid,
        int K, int M, int M_padded) override;

private:
    JumpParams params_;
};
```

---

## 8. Top-Level Orchestrator

**File:** `engine/src/ccr_engine.cpp`

`CcrEngine::run()` sequences the full pipeline. The stress scenario is a second invocation of `run_single()` with shocked parameters.

```cpp
CcrResult CcrEngine::run(
    const EngineConfig&             config,
    std::optional<ProgressCallback> callback)
{
    CcrResult result;

    // 1. Validate parameters (returns human-readable error if invalid)
    const std::string err = validate_config(config);
    if (!err.empty()) { result.error_msg = err; return result; }

    try {
        // 2. Base pass
        result.base = run_single(config, config.sim_params, callback);

        // 3. Optional stress pass — apply shocks and re-run
        if (config.stress.has_value()) {
            const auto& shock = *config.stress;

            SimParams stressed = config.sim_params;
            stressed.sigma    += shock.vol_shock;          // additive vol shock
            stressed.sigma    *= (1.0 + shock.equity_shock);
            stressed.mu       += shock.interest_rate_shock;

            EngineConfig stressed_cfg                  = config;
            stressed_cfg.sim_params                    = stressed;
            stressed_cfg.counterparty.hazard_rate      =
                std::max(0.0, config.counterparty.hazard_rate + shock.hazard_rate_shock);

            result.stressed = run_single(stressed_cfg, stressed_cfg.sim_params, std::nullopt);
        }
        result.success = true;
    } catch (const std::exception& ex) {
        result.error_msg = ex.what();
    }
    return result;
}

// ── Single-pass internals ─────────────────────────────────────────────────────
RiskMetrics CcrEngine::run_single(
    const EngineConfig&             config,
    const SimParams&                params,
    std::optional<ProgressCallback> callback)
{
    const auto t_start = std::chrono::steady_clock::now();

    // Time grid (PARSIMONIOUS: dense near MPoR, sparse for long horizon)
    TimeGrid tg(params.horizon_years, params.grid_type, config.counterparty.mpor_days);
    const int T = tg.num_steps();

    // Cholesky for WWR: 2×2 matrix [1, ρ; ρ, 1] decomposed when ρ ≠ 0
    CholeskyMatrix chol = (config.enable_wwr && K == 1)
        ? CholeskyMatrix::wwr_2x2(params.rho_wwr)
        : CholeskyMatrix::identity(K);

    // Single contiguous arena allocation (no heap allocations inside hot loop)
    if (!arena_) arena_.reset(new Arena());
    arena_->allocate(K, M_padded, T, config.enable_jump_diffusion);

    // RNG — seeded deterministically; reproducibility is a regulatory requirement
    Xoroshiro128aox rng(config.rng_seed);

    // PathSimulator runs the SIMD GBM loop
    PathSimulator sim(params, chol, tg,
                      config.portfolio, config.counterparty.recovery_rate, hook.get());
    sim.run_all_steps<ActiveArch>(state, rng);

    // Extract PFE (99th percentile via nth_element) and EPE (mean) profiles
    extract_profiles(state.exposures, state.pfe_profile, state.epe_profile,
                     T, M, M_padded, /*alpha=*/0.99, config.deterministic_quantile);

    // Fire progress callbacks — used by WebSocket to stream updates to browser
    if (callback) {
        for (int t = 0; t < T; ++t)
            (*callback)(t, T, static_cast<float>(arena_->pfe_profile[t]));
    }

    // CVA = (1−R) × Σ EPE(t) × PD(t) via Kahan summation
    auto marginal_pd = marginal_pd_from_flat_hazard(tg.times(), config.counterparty.hazard_rate);
    const double cva     = compute_cva(epe_span, marginal_pd, config.counterparty.recovery_rate);
    const double wwr_cva = config.enable_wwr ? compute_wwr_cva(...) : cva;
    const double margin  = compute_required_margin(pfe_span, config.portfolio.collateral);

    // Assemble result struct and return
    RiskMetrics metrics;
    metrics.cva             = cva;
    metrics.wwr_cva         = wwr_cva;
    metrics.margin_required = margin;
    metrics.arch_used       = ActiveArch::NAME;   // e.g. "ARM NEON"
    metrics.paths_used      = M;
    metrics.compute_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - t_start);
    return metrics;
}
```

---

## 9. pybind11 Bindings — C++ to Python Bridge

**File:** `engine/bindings/bindings.cpp`

pybind11 compiles `_ccr_engine.cpython-*.so` — a shared library imported directly into the FastAPI process. No IPC, no serialisation overhead. The GIL is released during the compute-intensive `CcrEngine::run()` call so other Python threads/requests can proceed concurrently.

```cpp
PYBIND11_MODULE(_ccr_engine, m) {
    m.doc() = "Real-Time Counterparty Credit Risk & Margin Engine (C++ backend)";

    // ── Expose enums as Python-compatible integers ────────────────────────────
    py::enum_<DerivativeType>(m, "DerivativeType")
        .value("IRS", DerivativeType::IRS)
        .value("CDS", DerivativeType::CDS)
        .value("FX",  DerivativeType::FX)
        .value("EQUITY",    DerivativeType::EQUITY)
        .value("COMMODITY", DerivativeType::COMMODITY)
        .export_values();

    py::enum_<SimMode>(m, "SimMode")
        .value("REGULATORY",  SimMode::REGULATORY)   // Basel III-grade quantile grid
        .value("STANDARD",    SimMode::STANDARD)
        .value("APPROX_FAST", SimMode::APPROX_FAST)
        .export_values();

    // ── Expose CcrEngine with GIL release on run() ───────────────────────────
    py::class_<CcrEngine>(m, "CcrEngine")
        .def(py::init<>())
        .def("run",
            [](CcrEngine& self,
               const EngineConfig& config,
               std::optional<py::function> callback_fn)
            {
                std::optional<ProgressCallback> cb;
                if (callback_fn.has_value()) {
                    cb = [fn = *callback_fn](int t, int total, double pfe) {
                        py::gil_scoped_acquire acquire;  // re-acquire GIL for Python callback
                        fn(t, total, pfe);
                    };
                }
                CcrResult result;
                {
                    py::gil_scoped_release release;   // release GIL — C++ runs freely
                    result = self.run(config, cb);
                }
                return result;
            },
            py::arg("config"),
            py::arg("callback") = py::none())

        // Static introspection: reports "AVX2", "ARM NEON", etc.
        .def_static("active_arch",
            []() { return std::string(CcrEngine::active_arch()); })
        .def_static("simd_width",
            []() { return static_cast<int>(CcrEngine::simd_width()); })
        .def_static("validate_config", &CcrEngine::validate_config)
        .def_static("estimate_arena_bytes", &CcrEngine::estimate_arena_bytes)
        .def_static("evaluate_margin_call", &CcrEngine::evaluate_margin_call);
}
```

Python usage after building the `.so`:

```python
import _ccr_engine as ccr

cfg = ccr.EngineConfig()
cfg.sim_params.num_paths     = 10_000
cfg.sim_params.sigma         = 0.20
cfg.counterparty.hazard_rate = 0.03
cfg.counterparty.recovery_rate = 0.40

engine = ccr.CcrEngine()
result = engine.run(cfg)

print(f"CVA:    {result.base.cva:.2f}")
print(f"Arch:   {result.base.arch_used}")   # "ARM NEON" on Apple Silicon
print(f"PFE[0]: {result.base.pfe_profile[0]:.2f}")
```

---

## 10. Python Glue Layer

**File:** `server/bindings/engine_client.py`

Converts Pydantic request models to `_ccr_engine` C++ types. Handles the hazard rate term structure: if the counterparty has hz_1y/hz_3y/hz_5y/hz_10y columns populated, these are integrated via the trapezoid rule to produce an effective flat hazard rate for the engine.

```python
def _effective_hazard_rate(
    hz_1y: float | None, hz_3y: float | None,
    hz_5y: float | None, hz_10y: float | None,
    horizon: float, flat_rate: float,
) -> float:
    """
    Piecewise-linear interpolation of the CDS term structure, integrated over
    the simulation horizon to produce a single effective flat hazard rate.

    Falls back to the scalar flat_rate when fewer than two term-structure
    tenors are provided.
    """
    tenors = [(t, h) for t, h in [(1.0, hz_1y), (3.0, hz_3y),
                                   (5.0, hz_5y), (10.0, hz_10y)] if h is not None]
    if len(tenors) < 2:
        return flat_rate
    ts  = [p[0] for p in tenors]
    hzs = [p[1] for p in tenors]
    t_grid  = np.linspace(0.0, horizon, 200)
    hz_grid = np.interp(t_grid, ts, hzs, left=hzs[0], right=hzs[-1])
    eff = float(np.trapezoid(hz_grid, t_grid) / max(horizon, 1e-9))
    return max(eff, 0.0)


def build_engine_config(req: SimulationRequest) -> _ccr.EngineConfig:
    """Map a Pydantic SimulationRequest to a _ccr_engine.EngineConfig (C++ struct)."""
    cfg = _ccr.EngineConfig()

    sp = _ccr.SimParams()
    sp.num_paths     = req.sim_params.num_paths
    sp.sigma         = req.sim_params.sigma
    sp.rho_wwr       = req.sim_params.rho_wwr
    sp.horizon_years = req.sim_params.horizon_years
    sp.mode          = _ccr.SimMode(int(req.sim_params.mode))
    cfg.sim_params   = sp

    cp = _ccr.CounterpartyConfig()
    cp.hazard_rate   = req.counterparty.hazard_rate
    cp.recovery_rate = req.counterparty.recovery_rate
    cp.collateral    = req.counterparty.collateral
    cp.mpor_days     = req.counterparty.mpor_days
    cfg.counterparty = cp

    cfg.enable_wwr            = req.enable_wwr
    cfg.enable_jump_diffusion = req.enable_jump_diffusion
    cfg.rng_seed              = req.rng_seed
    return cfg
```

**File:** `server/core/engine_runner.py`

FastAPI's event loop is async; the C++ engine is synchronous. This layer offloads the blocking call to a `ThreadPoolExecutor` (GIL-free thanks to pybind11's `py::gil_scoped_release`).

```python
# Four worker threads — C++ is GIL-free so they run truly in parallel
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ccr-worker")

def _run_sync(request, progress_cb=None) -> SimulationResponse:
    """Blocking: creates a fresh CcrEngine and runs the pipeline."""
    engine = _ccr.CcrEngine()
    config = build_engine_config(request)
    result = engine.run(config, progress_cb)  # GIL released here
    return result_to_response(result)

async def run_simulation(request, progress_cb=None) -> SimulationResponse:
    """Non-blocking coroutine: schedules _run_sync on the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_sync, request, progress_cb)
```

---

## 11. FastAPI Simulation Endpoint

**File:** `server/api/routes.py`

The `/api/v1/simulate` endpoint validates the request, runs the engine, persists results, and checks for margin calls — all within a single database transaction.

```python
@router.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request:      SimulationRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> SimulationResponse:
    """Run a CCR Monte Carlo simulation, persist results, and check margin."""

    # 1. Validate C++ engine config before queuing work
    err = _ccr.CcrEngine.validate_config(build_engine_config(request))
    if err:
        raise HTTPException(status_code=422, detail=err)

    # 2. Create the simulation run record (status=RUNNING)
    sim_run = SimulationRun(
        triggered_by    = current_user.id,
        trigger_type    = TriggerType.MANUAL,
        counterparty_id = resolved_cp_id,
        sim_params_json = request.sim_params.model_dump(),
        status          = SimStatus.RUNNING,
    )
    db.add(sim_run)
    await db.flush()   # get sim_run.id without committing

    # 3. Run engine (offloaded to thread pool, event loop is free)
    result = await run_simulation(request)

    # 4. Persist risk metrics (base + optional stressed)
    await _persist_metrics(db, sim_run.id, resolved_cp_id, result.base, is_stressed=False)
    if result.stressed:
        await _persist_metrics(db, sim_run.id, resolved_cp_id, result.stressed, is_stressed=True)

    sim_run.status       = SimStatus.DONE
    sim_run.completed_at = datetime.now(timezone.utc)

    # 5. Auto-generate margin call if exposure exceeds threshold
    if resolved_cp_id:
        await check_and_alert_margin_calls(
            db,
            counterparty_id   = resolved_cp_id,
            margin_required   = result.base.margin_required,
            collateral        = request.counterparty.collateral,
            simulation_run_id = sim_run.id,
        )

    # 6. Audit trail
    await log_event(db, action="simulate", user_id=current_user.id,
                    resource_type="simulation_run", resource_id=sim_run.id,
                    detail={"cva": result.base.cva, "margin_required": result.base.margin_required})

    await db.commit()
    return result
```

---

## 12. WebSocket Streaming

**File:** `server/api/websocket.py`

The WebSocket endpoint at `/ws/simulate` streams 12–50 progress events as the engine works through timesteps, then delivers the final result JSON. The client uses these events to animate a progress bar.

```python
@ws_router.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    await ws.accept()

    # Auth: client sends {"token": "Bearer <jwt>"} as first message
    user = await _authenticate_ws(ws)
    if user is None:
        await ws.close(code=4001, reason="Unauthorized")
        return

    raw     = await ws.receive_text()
    request = SimulationRequest.model_validate_json(raw)
    loop    = asyncio.get_running_loop()

    # Progress callback — called by C++ engine after each completed timestep.
    # asyncio.run_coroutine_threadsafe bridges the thread pool back to the event loop.
    def progress_cb(timestep: int, total: int, pfe_so_far: float):
        msg = json.dumps({
            "type":       "progress",
            "timestep":   timestep,
            "total":      total,
            "pfe_so_far": pfe_so_far,
            "pct":        round(100.0 * (timestep + 1) / max(total, 1), 1),
        })
        asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)

    result = await run_simulation(request, progress_cb)

    # Final message — full SimulationResponse as JSON
    await ws.send_text(json.dumps({
        "type":   "result",
        "result": result.model_dump(),
    }))
```

Message sequence observed by the browser:

```json
{"type":"progress","timestep":0,"total":12,"pfe_so_far":0.0,"pct":8.3}
{"type":"progress","timestep":1,"total":12,"pfe_so_far":12450.3,"pct":16.7}
...
{"type":"result","result":{"success":true,"base":{"cva":85234.5,"pfe_profile":[...],...}}}
```

---

## 13. SA-CCR Regulatory Capital

**File:** `server/reports/sa_ccr.py`

SA-CCR (BCBS 279 / Basel III CRE52) is the standardised approach for computing Exposure-at-Default for derivatives. The formula is:

> **EAD = α × (RC + AddOn_aggregate)**
> **AddOnᵢ = |notionalᵢ| × SFᵢ × MFᵢ**

where α = 1.4, SF is the supervisory factor (asset-class and maturity-dependent), and MF = √(min(M, 1)) is the maturity factor.

```python
ALPHA = 1.4  # BCBS 279 alpha multiplier (constant, non-negotiable)

# Supervisory factors from CRE52.72 (rate derivatives):
# IRS <1yr: 0.5%, 1–5yr: 0.5%, >5yr: 1.5%
_IRS_SF  = {(0.0, 1.0): 0.005, (1.0, 5.0): 0.005, (5.0, 999.0): 0.015}
_CDS_SF  = {(0.0, 1.0): 0.0038, (1.0, 5.0): 0.0042, (5.0, 999.0): 0.0045}
_SF_FIXED = {"FX": 0.04, "EQ": 0.32, "CMDTY": 0.40}


def compute_sa_ccr(
    derivatives: list[dict],   # {id, deriv_type, notional, maturity_years}
    collateral:  float,        # C — posted collateral
    margin_required: float,    # simulation-derived VM requirement (used to approximate V)
    mpor_days:   int = 10,     # Margin Period of Risk
) -> SACCRResult:

    # Replacement cost: RC = max(V − C, 0)
    # Approximate portfolio MtM from the simulation's margin_required output
    v_approx = margin_required / (1 + mpor_days / 360)
    rc       = max(v_approx - collateral, 0.0)

    add_on_total = 0.0
    breakdown    = []

    for d in derivatives:
        sf  = _supervisory_factor(d["deriv_type"], d["maturity_years"])
        mf  = math.sqrt(min(d["maturity_years"], 1.0))  # MF = √(min(M,1))
        ao  = abs(d["notional"]) * sf * mf
        add_on_total += ao
        breakdown.append(AddOnBreakdown(
            deriv_id=d["id"], sf=sf, mf=mf, add_on=ao, ...))

    ead = ALPHA * (rc + add_on_total)   # EAD = 1.4 × (RC + ΣAddOn)
    return SACCRResult(ead=ead, rc=rc, add_on_aggregate=add_on_total, breakdown=breakdown)
```

---

## 14. Database Models

**File:** `server/models/db_models.py`

SQLAlchemy 2.0 declarative models. The `price_history` table is a TimescaleDB hypertable (partitioned by `ts`); all other tables are regular PostgreSQL.

```python
class User(Base):
    __tablename__ = "users"
    id:         Mapped[str]      = mapped_column(String, primary_key=True,
                                      server_default=text("gen_random_uuid()::text"))
    username:   Mapped[str]      = mapped_column(String(64), unique=True, nullable=False)
    email:      Mapped[str]      = mapped_column(String(256), unique=True, nullable=False)
    hashed_pw:  Mapped[str]      = mapped_column(String(256), nullable=False)
    role:       Mapped[str]      = mapped_column(String(32), nullable=False,
                                      default=UserRole.AUDITOR)  # ADMIN|RISK_MANAGER|AUDITOR
    is_active:  Mapped[bool]     = mapped_column(Boolean, nullable=False, default=True)


class Counterparty(Base):
    __tablename__ = "counterparties"
    id:               Mapped[str]   = mapped_column(String, primary_key=True, ...)
    external_id:      Mapped[str]   = mapped_column(String(64), unique=True, nullable=False)
    name:             Mapped[str]   = mapped_column(String(256), nullable=False)
    credit_rating:    Mapped[str]   = mapped_column(String(8), nullable=False, default="BBB")
    hazard_rate:      Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    recovery_rate:    Mapped[float] = mapped_column(Float, nullable=False, default=0.40)
    collateral:       Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    margin_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mpor_days:        Mapped[int]   = mapped_column(Integer, nullable=False, default=10)
    # Hazard rate term structure (CDS-implied; optional — overrides flat hazard_rate)
    hz_1y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_3y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_5y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_10y:           Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    portfolios:       Mapped[List["Portfolio"]] = relationship("Portfolio",
                          back_populates="counterparty", lazy="selectin")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"
    id:              Mapped[str]      = mapped_column(String, primary_key=True, ...)
    triggered_by:    Mapped[str]      = mapped_column(String, ForeignKey("users.id"))
    counterparty_id: Mapped[Optional[str]] = mapped_column(String,
                         ForeignKey("counterparties.id"), nullable=True)
    trigger_type:    Mapped[str]      = mapped_column(String(32), default=TriggerType.MANUAL)
    status:          Mapped[str]      = mapped_column(String(16), default=SimStatus.RUNNING)
    sim_params_json: Mapped[dict]     = mapped_column(JSONB, nullable=False)
    stress_json:     Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    note:            Mapped[Optional[str]]  = mapped_column(Text, nullable=True)
    created_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), ...)
    completed_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), ...)


class RiskMetric(Base):
    """One row per simulation pass (base or stressed)."""
    __tablename__ = "risk_metrics"
    simulation_run_id: Mapped[str]   = mapped_column(String, ForeignKey("simulation_runs.id"))
    counterparty_id:   Mapped[Optional[str]] = mapped_column(...)
    cva:               Mapped[float] = mapped_column(Float, nullable=False)
    wwr_cva:           Mapped[float] = mapped_column(Float, nullable=False)
    pfe_profile:       Mapped[str]   = mapped_column(Text, nullable=False)  # JSON array
    epe_profile:       Mapped[str]   = mapped_column(Text, nullable=False)
    time_grid_years:   Mapped[str]   = mapped_column(Text, nullable=False)
    margin_required:   Mapped[float] = mapped_column(Float, nullable=False)
    is_stressed:       Mapped[bool]  = mapped_column(Boolean, default=False)


class MarginCall(Base):
    __tablename__ = "margin_calls"
    id:               Mapped[str]   = mapped_column(String, primary_key=True, ...)
    counterparty_id:  Mapped[str]   = mapped_column(String, ForeignKey("counterparties.id"))
    simulation_run_id: Mapped[Optional[str]] = mapped_column(...)
    amount:           Mapped[float] = mapped_column(Float, nullable=False)
    status:           Mapped[str]   = mapped_column(String(32), default=MarginCallStatus.PENDING)
    acknowledged_at:  Mapped[Optional[datetime]] = mapped_column(...)
    settled_at:       Mapped[Optional[datetime]] = mapped_column(...)
```

---

## 15. Frontend API Client

**File:** `web/src/lib/api.ts`

Typed singleton that wraps every REST endpoint. All requests include the JWT from `localStorage`; 401 responses redirect to `/login`.

```typescript
class ApiClient {
  private token: string | null = null;

  setToken(t: string | null) { this.token = t; }

  private get headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await fetch(path, {
      method,
      headers: this.headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const json = await res.json().catch(() => ({}));
      throw new ApiError(res.status, json.detail ?? res.statusText);
    }
    return res.json();
  }

  // ── Simulation ───────────────────────────────────────────────────────────────
  async simulate(req: SimulationRequest): Promise<SimulationResponse> {
    return this.request<SimulationResponse>('POST', '/api/v1/simulate', req);
  }

  // ── SA-CCR capital (async, loads after simulation result is shown) ───────────
  async getSACCR(runId: string): Promise<SACCRResult> {
    return this.request<SACCRResult>('GET', `/api/v1/simulate/${runId}/sa-ccr`);
  }

  // ── Counterparties ───────────────────────────────────────────────────────────
  async listCounterparties(): Promise<Counterparty[]> {
    return this.request<Counterparty[]>('GET', '/api/v1/counterparties');
  }
  async createCounterparty(data: Partial<Counterparty>): Promise<Counterparty> {
    return this.request<Counterparty>('POST', '/api/v1/counterparties', data);
  }
  async deleteCounterparty(id: string): Promise<void> {
    await this.request('DELETE', `/api/v1/counterparties/${id}?cascade=true`);
  }

  // ── Margin calls ─────────────────────────────────────────────────────────────
  async acknowledgeMarginCall(id: string): Promise<MarginCall> {
    return this.request<MarginCall>('PUT', `/api/v1/margin-calls/${id}/acknowledge`);
  }
  async notifyMarginCall(id: string): Promise<{ status: string }> {
    return this.request('POST', `/api/v1/margin-calls/${id}/notify`);
  }
}

export const api = new ApiClient();
```

---

## Appendix: Key Numerical Parameters (Demo Defaults)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `num_paths` | 10,000 | Monte Carlo scenarios (regulatory minimum ≈ 5,000) |
| `num_timesteps` | 12 (parsimonious) | Time steps over the horizon |
| `sigma` | 0.20 | 20% annualised volatility |
| `hazard_rate` | 0.008–0.05 | Annual default intensity λ |
| `recovery_rate` | 0.40 | 40% LGD complement (standard CDS convention) |
| `mpor_days` | 10 | Margin Period of Risk (Basel III minimum for cleared) |
| `rng_seed` | 42 | Fixed seed → reproducible results across runs |
| `alpha (PFE)` | 99% | Regulatory PFE quantile (Basel III IMM) |
| `α (SA-CCR)` | 1.4 | BCBS 279 exposure multiplier (non-negotiable constant) |

---

## Appendix: Role-Based Access Control

Three JWT roles enforced at every API endpoint via `require_role()`:

| Role | Can Do |
|------|--------|
| `ADMIN` | Everything: manage users, run sims, view audit log |
| `RISK_MANAGER` | Run simulations, manage counterparties/portfolios, acknowledge/settle margin calls |
| `AUDITOR` | Read-only: view results, export CSV, view audit log |

Demo credentials: `admin/admin123` · `risk/risk123` · `auditor/auditor123`
