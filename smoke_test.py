"""
Smoke test for the CCR engine Python binding.
Run from the project root:
    PYTHONPATH=server/bindings python3 smoke_test.py
"""
import sys
sys.path.insert(0, "server/bindings")

import _ccr_engine as ccr

print(f"Engine arch : {ccr.active_arch()}")
print(f"SIMD width  : {ccr.simd_width()}")

# ── Build config: single IRS, 1000 paths, 12 monthly steps ──────────────────

cfg = ccr.EngineConfig()

sp = ccr.SimParams()
sp.num_paths     = 1000
sp.num_timesteps = 12
sp.num_assets    = 1
sp.mu            = 0.02
sp.sigma         = 0.20
sp.rho_wwr       = 0.0
sp.recovery_rate = 0.40
sp.horizon_years = 1.0
sp.mode          = ccr.SimMode.STANDARD
sp.grid_type     = ccr.GridType.MONTHLY
cfg.sim_params   = sp

cp = ccr.CounterpartyConfig()
cp.id            = "CP-001"
cp.hazard_rate   = 0.03
cp.recovery_rate = 0.40
cp.collateral    = 0.0
cfg.counterparty = cp

# IRS: 5Y, 1M notional, 5% fixed rate
d = ccr.DerivativeSpec()
d.id               = "IRS-001"
d.type             = ccr.DerivativeType.IRS
d.notional         = 1_000_000.0
d.maturity_years   = 5.0
d.underlying_price = 0.05   # initial swap rate
d.strike           = 0.05   # fixed rate (at-the-money)
d.cash_flow_freq   = 2.0

port = ccr.PortfolioConfig()
port.derivatives = [d]
port.collateral  = 0.0
cfg.portfolio    = port

cfg.deterministic_quantile = True
cfg.rng_seed               = 42

# ── Run ─────────────────────────────────────────────────────────────────────

engine = ccr.CcrEngine()
result = engine.run(cfg)

# ── Assertions ───────────────────────────────────────────────────────────────

assert result.success, f"Engine failed: {result.error_msg}"
print(f"\nBase metrics:")
print(f"  CVA              : {result.base.cva:.6f}")
print(f"  WWR CVA          : {result.base.wwr_cva:.6f}")
print(f"  Margin required  : {result.base.margin_required:.4f}")
print(f"  PFE profile len  : {len(result.base.pfe_profile)}")
print(f"  EPE profile len  : {len(result.base.epe_profile)}")
print(f"  Arch used        : {result.base.arch_used}")
print(f"  Paths used       : {result.base.paths_used}")
print(f"  Compute time     : {result.base.compute_time_us.microseconds} µs")

assert result.base.cva > 0, "CVA should be positive for non-zero hazard rate"
assert len(result.base.pfe_profile) == 12, \
    f"Expected 12 PFE steps, got {len(result.base.pfe_profile)}"
assert len(result.base.epe_profile) == 12, \
    f"Expected 12 EPE steps, got {len(result.base.epe_profile)}"
assert result.base.paths_used == 1000

print(f"\n  PFE profile: {[f'{v:.4f}' for v in result.base.pfe_profile]}")
print(f"  EPE profile: {[f'{v:.4f}' for v in result.base.epe_profile]}")

# ── Stress scenario test ─────────────────────────────────────────────────────

sc = ccr.StressScenario()
sc.vol_shock         = 0.10
sc.hazard_rate_shock = 0.02
sc.label             = "market-stress"
cfg.stress           = sc

result_stressed = engine.run(cfg)
assert result_stressed.success
assert result_stressed.stressed is not None
assert result_stressed.stressed.cva >= result_stressed.base.cva * 0.5, \
    "Stressed CVA should be in a reasonable range"

print(f"\nStressed metrics:")
print(f"  CVA (stressed) : {result_stressed.stressed.cva:.6f}")
print(f"  CVA (base)     : {result_stressed.base.cva:.6f}")

print("\n✓ All assertions passed.")
