<script lang="ts">
  export let label:     string  = '';
  export let value:     string  = '—';
  export let unit:      string  = '';
  export let delta:     number  = 0;    // positive = up (green), negative = down (red)
  export let breached:  boolean = false;
  export let subtitle:  string  = '';
</script>

<div class="card metric-card" class:breached>
  <div class="metric-label">{label}</div>
  <div class="metric-value">
    {value}<span class="metric-unit">{unit}</span>
  </div>
  {#if delta !== 0}
    <div class="metric-delta" class:up={delta > 0} class:down={delta < 0}>
      {delta > 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(2)}%
    </div>
  {/if}
  {#if breached}
    <div class="badge badge-red" style="margin-top:.4rem">Breach</div>
  {/if}
  {#if subtitle}
    <div class="metric-sub">{subtitle}</div>
  {/if}
</div>

<style>
  .metric-card { padding: 1rem 1.25rem; }
  .metric-card.breached { border-color: rgba(255,77,106,.4); background: rgba(255,77,106,.05); }
  .metric-label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: .4rem; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: var(--text); line-height: 1; }
  .metric-unit  { font-size: .75rem; color: var(--muted); margin-left: .15rem; font-weight: 400; }
  .metric-delta { font-size: .78rem; margin-top: .3rem; }
  .metric-delta.up   { color: var(--red); }
  .metric-delta.down { color: var(--green); }
  .metric-sub   { font-size: .72rem; color: var(--muted); margin-top: .3rem; }
</style>
