<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { StressScenarioRequest } from '$lib/types';

  const dispatch = createEventDispatcher<{
    apply: StressScenarioRequest;
    clear: void;
  }>();

  export let value: StressScenarioRequest | null = null;

  let vol_shock:           number = 0;
  let fx_shock:            number = 0;
  let equity_shock:        number = 0;
  let interest_rate_shock: number = 0;
  let credit_spread_shock: number = 0;
  let hazard_rate_shock:   number = 0;
  let jump_amplitude:      number = 0;
  let label:               string = 'stress';

  // Sync local state from the value prop whenever the parent updates it
  // (e.g. after "Apply Stress" is confirmed, or when navigating back to this page)
  $: if (value) {
    vol_shock           = value.vol_shock           ?? 0;
    fx_shock            = value.fx_shock            ?? 0;
    equity_shock        = value.equity_shock        ?? 0;
    interest_rate_shock = value.interest_rate_shock ?? 0;
    credit_spread_shock = value.credit_spread_shock ?? 0;
    hazard_rate_shock   = value.hazard_rate_shock   ?? 0;
    jump_amplitude      = value.jump_amplitude      ?? 0;
    label               = value.label               ?? 'stress';
  }

  function fmt(v: number, decimals = 2) {
    return v >= 0 ? `+${v.toFixed(decimals)}` : v.toFixed(decimals);
  }

  function apply() {
    dispatch('apply', {
      vol_shock, fx_shock, equity_shock,
      interest_rate_shock, credit_spread_shock,
      hazard_rate_shock, jump_amplitude, label,
    });
  }

  function clear() {
    vol_shock = fx_shock = equity_shock = interest_rate_shock
              = credit_spread_shock = hazard_rate_shock = jump_amplitude = 0;
    dispatch('clear');
  }

  // Static slider metadata — no closures, no getters
  const sliders = [
    { key: 'vol_shock',           label: 'Vol Shock',           min: -0.5,  max: 2.0,  step: 0.05  },
    { key: 'fx_shock',            label: 'FX Shock',            min: -0.3,  max: 0.3,  step: 0.01  },
    { key: 'equity_shock',        label: 'Equity Shock',        min: -0.5,  max: 0.5,  step: 0.01  },
    { key: 'interest_rate_shock', label: 'Rate Shock',          min: -0.05, max: 0.05, step: 0.001 },
    { key: 'credit_spread_shock', label: 'Credit Spread Shock', min: -0.02, max: 0.10, step: 0.001 },
    { key: 'hazard_rate_shock',   label: 'Hazard Rate Shock',   min: -0.05, max: 0.30, step: 0.005 },
    { key: 'jump_amplitude',      label: 'Jump Amplitude',      min:  0,    max: 0.50, step: 0.01  },
  ] as const;

  // Reactive snapshot — Svelte CAN track these direct variable reads
  $: vals = [vol_shock, fx_shock, equity_shock, interest_rate_shock,
             credit_spread_shock, hazard_rate_shock, jump_amplitude];

  const setters = [
    (v: number) => { vol_shock           = v; },
    (v: number) => { fx_shock            = v; },
    (v: number) => { equity_shock        = v; },
    (v: number) => { interest_rate_shock = v; },
    (v: number) => { credit_spread_shock = v; },
    (v: number) => { hazard_rate_shock   = v; },
    (v: number) => { jump_amplitude      = v; },
  ];
</script>

<div class="stress-form">
  <div class="form-group" style="margin-bottom:1rem">
    <label class="form-label" for="stress-label">Scenario Label</label>
    <input id="stress-label" class="form-input" bind:value={label} placeholder="stress" />
  </div>

  {#each sliders as s, i}
    <div class="slider-row">
      <div class="slider-header">
        <span class="form-label">{s.label}</span>
        <span class="slider-val" class:nonzero={vals[i] !== 0}>{fmt(vals[i])}</span>
      </div>
      <input
        type="range"
        min={s.min} max={s.max} step={s.step}
        value={vals[i]}
        on:input={(e) => setters[i](parseFloat((e.target as HTMLInputElement).value))}
      />
      <div class="slider-range">
        <span>{s.min}</span><span>{s.max}</span>
      </div>
    </div>
  {/each}

  <div style="display:flex;gap:.5rem;margin-top:1rem">
    <button class="btn btn-primary" on:click={apply}>Apply Stress</button>
    <button class="btn btn-ghost"   on:click={clear}>Reset</button>
  </div>
</div>

<style>
  .stress-form { display: flex; flex-direction: column; gap: .5rem; }
  .slider-row  { padding: .4rem 0; }
  .slider-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: .2rem; }
  .slider-val { font-size: .8rem; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; }
  .slider-val.nonzero { color: var(--amber); }
  .slider-range { display: flex; justify-content: space-between; font-size: .65rem; color: var(--muted); margin-top: .1rem; }
</style>
