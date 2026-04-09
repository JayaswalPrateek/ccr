<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { api } from '$lib/api';
  import type {
    CounterpartyRequest,
    DerivativeSpecRequest,
    PortfolioRequest,
    SimParamsRequest,
    SimulationRequest,
  } from '$lib/types';
  import { DerivativeType, GridType, SimMode } from '$lib/types';

  const dispatch = createEventDispatcher<{ submit: SimulationRequest }>();

  // Increment this prop to programmatically trigger submission from a parent.
  export let trigger = 0;
  $: if (trigger > 0) submit();

  // Optional initial sim params (e.g. loaded from a preset)
  export let initialSimParams: Record<string, unknown> | null = null;
  export let initialNote: string = '';
  let runNote = '';
  $: runNote = initialNote;

  onMount(() => {
    if (initialSimParams) {
      simParams = { ...simParams, ...(initialSimParams as Partial<SimParamsRequest>) };
      // Apply counterparty fields if passed via cp_id URL param
      if (initialSimParams.counterparty_id)          counterparty.id            = initialSimParams.counterparty_id as string;
      if (initialSimParams.counterparty_name)         counterparty.name          = initialSimParams.counterparty_name as string;
      if (initialSimParams.counterparty_hazard_rate != null) counterparty.hazard_rate   = initialSimParams.counterparty_hazard_rate as number;
      if (initialSimParams.counterparty_recovery_rate != null) counterparty.recovery_rate = initialSimParams.counterparty_recovery_rate as number;
      if (initialSimParams.counterparty_collateral != null) counterparty.collateral    = initialSimParams.counterparty_collateral as number;
      if (initialSimParams.counterparty_mpor_days != null) counterparty.mpor_days     = initialSimParams.counterparty_mpor_days as number;
      counterparty = { ...counterparty };
    }
    function onKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  // ── Default form state ────────────────────────────────────────────────────
  let simParams: SimParamsRequest = {
    num_paths: 10000, num_timesteps: 12, num_assets: 1,
    mu: 0.02, sigma: 0.20, rho_wwr: 0.0, recovery_rate: 0.40,
    horizon_years: 1.0, mode: SimMode.STANDARD, grid_type: GridType.MONTHLY,
  };

  let counterparty: CounterpartyRequest = {
    id: 'CP-001', name: 'Counterparty A', credit_rating: 3,
    hazard_rate: 0.02, recovery_rate: 0.40, collateral: 0,
    margin_threshold: 0, mpor_days: 10,
  };

  let portfolio: PortfolioRequest = {
    id: 'PORT-001', counterparty_id: 'CP-001', derivatives: [], collateral: 0, net_value: 0,
  };

  let enableWwr       = false;
  let enableJump      = false;
  let enableCollateral= false;
  let rngSeed         = 42;
  let loadingMarket   = false;
  let marketError     = '';
  let marketFrozen    = false;
  let frozenAt: Date | null = null;

  function toggleFreeze() {
    if (!marketFrozen) { marketFrozen = true;  frozenAt = new Date(); }
    else               { marketFrozen = false; frozenAt = null; }
  }

  $: paramWarnings = (() => {
    const w: string[] = [];
    if (simParams.num_paths < 1000) w.push('Low path count may reduce accuracy');
    if (simParams.sigma > 1.0)      w.push('Extreme volatility — results may be unreliable');
    if (simParams.sigma < 0.01)     w.push('Near-zero volatility');
    return w;
  })();

  // ── Derivative helpers ────────────────────────────────────────────────────
  function addDerivative() {
    portfolio.derivatives = [
      ...portfolio.derivatives,
      {
        id: `DERIV-${portfolio.derivatives.length + 1}`,
        type: DerivativeType.IRS,
        notional: 1_000_000,
        maturity_years: 5,
        underlying_price: 0.05,
        strike: 0.05,
        cash_flow_freq: 2,
      },
    ];
  }

  function removeDerivative(i: number) {
    portfolio.derivatives = portfolio.derivatives.filter((_, idx) => idx !== i);
  }

  // ── Load from market ──────────────────────────────────────────────────────
  async function loadFromMarket() {
    loadingMarket = true;
    marketError   = '';
    try {
      const prices = await api.getMarketPrices();
      const sofr   = prices.find((p) => p.symbol === 'SOFR' && p.param_type === 'RATE');
      const vol    = prices.find((p) => p.param_type === 'VOL');
      if (sofr) simParams.mu    = sofr.value;
      if (vol)  simParams.sigma = vol.value;
      simParams = { ...simParams };
    } catch (e) {
      marketError = e instanceof Error ? e.message : 'Could not load market data';
    } finally {
      loadingMarket = false;
    }
  }

  function submit() {
    dispatch('submit', {
      sim_params:             simParams,
      counterparty,
      portfolio:              { ...portfolio, counterparty_id: counterparty.id },
      enable_wwr:             enableWwr,
      enable_jump_diffusion:  enableJump,
      enable_collateral:      enableCollateral,
      deterministic_quantile: true,
      log_overflow_warnings:  false,
      rng_seed:               rngSeed,
      note:                   runNote || undefined,
    });
  }

  const derivTypeNames: Record<number, string> = {
    0: 'IRS', 1: 'CDS', 2: 'FX', 3: 'Equity', 4: 'Commodity',
  };
  const simModeNames: Record<number, string> = {
    0: 'Regulatory', 1: 'Standard', 2: 'Approx Fast',
  };
  const gridTypeNames: Record<number, string> = {
    0: 'Monthly', 1: 'Weekly', 2: 'Daily', 3: 'Parsimonious',
  };
</script>

<div class="sim-form">

  <!-- ── Simulation Parameters ──────────────────────────────────────── -->
  <section>
    <div class="section-title">Simulation Parameters</div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Paths</label>
        <input class="form-input" type="number" bind:value={simParams.num_paths} min="100" max="100000" step="1000" />
      </div>
      <div class="form-group">
        <label class="form-label">Timesteps</label>
        <input class="form-input" type="number" bind:value={simParams.num_timesteps} min="1" max="60" />
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Horizon (years)</label>
        <input class="form-input" type="number" bind:value={simParams.horizon_years} min="0.1" max="30" step="0.5" />
      </div>
      <div class="form-group">
        <label class="form-label">RNG Seed</label>
        <input class="form-input" type="number" bind:value={rngSeed} min="0" />
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Drift (µ)</label>
        <input class="form-input" type="number" bind:value={simParams.mu} step="0.001" />
      </div>
      <div class="form-group">
        <label class="form-label">Volatility (σ)</label>
        <input class="form-input" type="number" bind:value={simParams.sigma} min="0.001" step="0.01" />
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Mode</label>
        <select class="form-select" bind:value={simParams.mode}>
          {#each Object.entries(simModeNames) as [k, v]}
            <option value={parseInt(k)}>{v}</option>
          {/each}
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Grid Type</label>
        <select class="form-select" bind:value={simParams.grid_type}>
          {#each Object.entries(gridTypeNames) as [k, v]}
            <option value={parseInt(k)}>{v}</option>
          {/each}
        </select>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:.4rem;margin-top:.25rem">
      <div style="display:flex;gap:.4rem">
        <button class="btn btn-ghost btn-sm" on:click={loadFromMarket} disabled={loadingMarket || marketFrozen}>
          {#if loadingMarket}<span class="spinner" style="width:12px;height:12px"></span>{/if}
          Load µ/σ from Market
        </button>
        <button
          class="btn btn-sm"
          style={marketFrozen ? 'background:rgba(245,158,11,.15);border-color:rgba(245,158,11,.4);color:var(--amber)' : ''}
          class:btn-ghost={!marketFrozen}
          on:click={toggleFreeze}
          title={marketFrozen ? 'Click to unfreeze market values' : 'Freeze current µ/σ values'}
        >
          {#if marketFrozen}
            <svg viewBox="0 0 16 16" fill="currentColor" style="width:12px;height:12px;display:inline;margin-right:3px"><path d="M8 1a2 2 0 0 1 2 2v4H6V3a2 2 0 0 1 2-2m3 6V3a3 3 0 0 0-6 0v4a2 2 0 0 0-2 2v5a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2"/></svg>Frozen
          {:else}
            <svg viewBox="0 0 16 16" fill="currentColor" style="width:12px;height:12px;display:inline;margin-right:3px"><path d="M11 1a2 2 0 0 0-2 2v4a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h5V3a3 3 0 0 1 6 0v4a.5.5 0 0 1-1 0V3a2 2 0 0 0-2-2"/></svg>Freeze
          {/if}
        </button>
      </div>
      {#if marketFrozen && frozenAt}
        <div style="font-size:.72rem;color:var(--amber)">Market values frozen at {frozenAt.toLocaleTimeString()}</div>
      {/if}
      {#if marketError}
        <div style="font-size:.75rem;color:var(--red)">{marketError}</div>
      {/if}
    </div>
    {#each paramWarnings as w}
      <div class="alert alert-warn" style="font-size:.76rem;padding:.35rem .65rem;margin-top:.35rem">{w}</div>
    {/each}
    <div style="display:flex;gap:1.25rem;margin-top:.75rem;flex-wrap:wrap">
      <label style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;cursor:pointer">
        <input type="checkbox" bind:checked={enableWwr} /> Enable WWR
      </label>
      <label style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;cursor:pointer">
        <input type="checkbox" bind:checked={enableJump} /> Jump Diffusion
      </label>
      <label style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;cursor:pointer">
        <input type="checkbox" bind:checked={enableCollateral} /> Collateral
      </label>
    </div>
  </section>

  <hr class="divider" />

  <!-- ── Counterparty ───────────────────────────────────────────────── -->
  <section>
    <div class="section-title">Counterparty</div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">ID</label>
        <input class="form-input" bind:value={counterparty.id} />
      </div>
      <div class="form-group">
        <label class="form-label">Name</label>
        <input class="form-input" bind:value={counterparty.name} />
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Hazard Rate</label>
        <input class="form-input" type="number" bind:value={counterparty.hazard_rate} min="0" step="0.001" />
      </div>
      <div class="form-group">
        <label class="form-label">Recovery Rate</label>
        <input class="form-input" type="number" bind:value={counterparty.recovery_rate} min="0" max="1" step="0.05" />
      </div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Collateral</label>
        <input class="form-input" type="number" bind:value={counterparty.collateral} min="0" step="1000" />
      </div>
      <div class="form-group">
        <label class="form-label">MPOR Days</label>
        <input class="form-input" type="number" bind:value={counterparty.mpor_days} min="1" />
      </div>
    </div>
  </section>

  <hr class="divider" />

  <!-- ── Derivatives ───────────────────────────────────────────────── -->
  <section>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem">
      <div class="section-title" style="margin-bottom:0">Derivatives</div>
      <button class="btn btn-ghost btn-sm" on:click={addDerivative}>+ Add</button>
    </div>

    {#if portfolio.derivatives.length === 0}
      <div style="font-size:.8rem;color:var(--muted);padding:.5rem 0">
        No derivatives — click "Add" to create one.
      </div>
    {/if}

    {#each portfolio.derivatives as d, i}
      <div class="deriv-row">
        <div class="form-row-3">
          <div class="form-group">
            <label class="form-label">Type</label>
            <select class="form-select" bind:value={d.type}>
              {#each Object.entries(derivTypeNames) as [k, v]}
                <option value={parseInt(k)}>{v}</option>
              {/each}
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Notional</label>
            <input class="form-input" type="number" bind:value={d.notional} min="0" step="100000" />
          </div>
          <div class="form-group">
            <label class="form-label">Maturity (yrs)</label>
            <input class="form-input" type="number" bind:value={d.maturity_years} min="0.1" step="0.5" />
          </div>
        </div>
        <div class="form-row-3">
          <div class="form-group">
            <label class="form-label">Underlying</label>
            <input class="form-input" type="number" bind:value={d.underlying_price} min="0" step="0.01" />
          </div>
          <div class="form-group">
            <label class="form-label">Strike</label>
            <input class="form-input" type="number" bind:value={d.strike} step="0.001" />
          </div>
          <div class="form-group">
            <label class="form-label">Cash-Flow Freq</label>
            <input class="form-input" type="number" bind:value={d.cash_flow_freq} min="0.5" step="0.5" />
          </div>
        </div>
        <button class="btn btn-danger btn-sm" style="align-self:flex-end" on:click={() => removeDerivative(i)}>Remove</button>
      </div>
    {/each}
  </section>

  <hr class="divider" />

  <!-- ── Run Annotation ───────────────────────────────────────────── -->
  <section>
    <div class="section-title">Run Annotation</div>
    <div class="form-group">
      <label class="form-label">Note (optional)</label>
      <input class="form-input" type="text" bind:value={runNote} placeholder="e.g. Pre-FOMC stress check" maxlength="200" />
    </div>
    <div style="font-size:.68rem;color:var(--muted)">Tip: Press Ctrl+Enter to run</div>
  </section>

</div>

<style>
  .sim-form      { display: flex; flex-direction: column; gap: .25rem; }
  .section-title { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: .75rem; }
  .deriv-row     { border: 1px solid var(--border); border-radius: var(--radius-sm); padding: .75rem; margin-bottom: .75rem; background: var(--surface2); }
</style>
