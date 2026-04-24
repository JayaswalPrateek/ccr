<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/api';
  import RoleGuard from '$components/ui/RoleGuard.svelte';
  import CVABarChart from '$components/charts/CVABarChart.svelte';
  import SurvivalCurveChart from '$components/charts/SurvivalCurveChart.svelte';
  import BacktestChart from '$components/charts/BacktestChart.svelte';
  import type { BacktestResult, Counterparty, MarginCall, Portfolio, SimulationHistoryItem } from '$lib/types';
  import { fmtNum } from '$lib/fmt';

  interface CpSummary { total_runs: number; avg_cva: number; latest_cva: number | null; total_margin_called: number; pending_calls: number; settled_calls: number; total_derivatives: number; }

  let cp:       Counterparty | null = null;
  let history:  SimulationHistoryItem[] = [];
  let mcs:      MarginCall[]            = [];
  let loading   = true;
  let error     = '';
  let editing   = false;
  let editForm: Partial<Counterparty> = {};
  let expandedPortfolio: string | null = null;
  let summary: CpSummary | null = null;
  let backtest: BacktestResult | null = null;

  // Inline add-portfolio form
  let addingPortfolio = false;
  let portForm = { collateral: 0, auto_run: false };

  // Inline add-derivative form: keyed by portfolio id, null = closed
  let addingDerivFor: string | null = null;
  let derivForm = { deriv_type: 'IRS', notional: 1_000_000, maturity_years: 5,
                    underlying_price: 0.05, strike: 0.05, cash_flow_freq: 2 };

  $: id = $page.params.id!;

  onMount(async () => {
    try {
      [cp, history, mcs] = await Promise.all([
        api.getCounterparty(id),
        api.getSimHistory({ counterparty_id: id, limit: 20 }),
        api.listMarginCalls({ counterparty_id: id, limit: 20 }),
      ]);
      editForm = { ...cp };
      api.getCounterpartySummary(id).then(s => { summary = s; }).catch(() => {});
      api.getBacktest(id).then(b => { backtest = b; }).catch(() => {});
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function saveEdit() {
    if (!cp) return;
    try {
      cp = await api.updateCounterparty(cp.id, editForm);
      editing = false;
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function submitPortfolio() {
    if (!cp) return;
    const n = (cp.portfolios ?? []).length + 1;
    try {
      const p = await api.createPortfolio({
        external_id:     `${cp.external_id}-PORT-${n}`,
        counterparty_id: cp.id,
        collateral:      portForm.collateral,
        net_value:       0,
        auto_run:        portForm.auto_run,
      });
      cp = { ...cp, portfolios: [...(cp.portfolios ?? []), p] };
      addingPortfolio = false;
      portForm = { collateral: 0, auto_run: false };
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function submitDerivative(portfolioId: string) {
    try {
      const d = await api.addDerivative(portfolioId, {
        external_id:     `DERIV-${Date.now()}`,
        deriv_type:      derivForm.deriv_type,
        notional:        derivForm.notional,
        maturity_years:  derivForm.maturity_years,
        underlying_price: derivForm.underlying_price,
        strike:          derivForm.strike,
        cash_flow_freq:  derivForm.cash_flow_freq,
      });
      if (!cp) return;
      cp = {
        ...cp,
        portfolios: (cp.portfolios ?? []).map((p) =>
          p.id === portfolioId
            ? { ...p, derivatives: [...(p.derivatives ?? []), d] }
            : p
        ),
      };
      addingDerivFor = null;
      derivForm = { deriv_type: 'IRS', notional: 1_000_000, maturity_years: 5,
                    underlying_price: 0.05, strike: 0.05, cash_flow_freq: 2 };
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function removeDerivative(portfolioId: string, derivId: string) {
    if (!confirm('Remove derivative?')) return;
    try {
      await api.deleteDerivative(portfolioId, derivId);
      if (!cp) return;
      cp = {
        ...cp,
        portfolios: (cp.portfolios ?? []).map((p) =>
          p.id === portfolioId
            ? { ...p, derivatives: (p.derivatives ?? []).filter((d) => d.id !== derivId) }
            : p
        ),
      };
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  const statusBadge: Record<string, string> = {
    PENDING: 'badge-amber', ACKNOWLEDGED: 'badge-blue',
    SETTLED: 'badge-green', DISPUTED: 'badge-red',
  };
</script>

<svelte:head><title>{cp?.name ?? 'Counterparty'} — CCR Engine</title></svelte:head>

{#if loading}
  <div style="padding:2rem;text-align:center"><div class="spinner"></div></div>
{:else if error}
  <div class="alert alert-error">{error}</div>
{:else if cp}
  <div class="page-header">
    <div>
      <div class="page-title">{cp.name}</div>
      <div class="page-sub">{cp.external_id} · {cp.credit_rating}</div>
    </div>
    <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
      <a href="/simulate?cp_id={cp.id}" class="btn btn-primary btn-sm">Run Simulation</a>
      <button class="btn btn-ghost btn-sm" on:click={() => { editing = !editing; editForm = { ...cp }; }}>
        {editing ? 'Cancel' : 'Edit'}
      </button>
    </RoleGuard>
  </div>

  {#if editing}
    <div class="card" style="margin-bottom:1rem">
      <div class="card-header"><span class="card-title">Edit Counterparty</span></div>
      <div class="form-row">
        <div class="form-group"><label class="form-label">Name</label><input class="form-input" bind:value={editForm.name} /></div>
        <div class="form-group">
          <label class="form-label">Credit Rating</label>
          <select class="form-select" bind:value={editForm.credit_rating}>
            {#each ['AAA','AA','A','BBB','BB','B','CCC','D'] as r}<option>{r}</option>{/each}
          </select>
        </div>
      </div>
      <div class="form-row">
        <div class="form-group"><label class="form-label">Hazard Rate</label><input class="form-input" type="number" bind:value={editForm.hazard_rate} step="0.001" /></div>
        <div class="form-group"><label class="form-label">Recovery Rate</label><input class="form-input" type="number" bind:value={editForm.recovery_rate} min="0" max="1" step="0.05" /></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label class="form-label">Collateral</label><input class="form-input" type="number" bind:value={editForm.collateral} /></div>
        <div class="form-group"><label class="form-label">MPOR Days</label><input class="form-input" type="number" bind:value={editForm.mpor_days} /></div>
      </div>
      <div style="font-size:.78rem;color:var(--muted);margin:.5rem 0 .25rem">CDS Term Structure (optional — overrides flat hazard rate for CVA)</div>
      <div class="form-row">
        <div class="form-group"><label class="form-label">λ 1Y</label><input class="form-input" type="number" bind:value={editForm.hz_1y} step="0.001" placeholder="e.g. 0.01" /></div>
        <div class="form-group"><label class="form-label">λ 3Y</label><input class="form-input" type="number" bind:value={editForm.hz_3y} step="0.001" placeholder="e.g. 0.02" /></div>
        <div class="form-group"><label class="form-label">λ 5Y</label><input class="form-input" type="number" bind:value={editForm.hz_5y} step="0.001" placeholder="e.g. 0.03" /></div>
        <div class="form-group"><label class="form-label">λ 10Y</label><input class="form-input" type="number" bind:value={editForm.hz_10y} step="0.001" placeholder="e.g. 0.05" /></div>
      </div>
      <button class="btn btn-success" on:click={saveEdit}>Save</button>
    </div>
  {/if}

  <!-- Info cards -->
  <div class="grid-4" style="margin-bottom:1rem">
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Hazard Rate</div><div style="font-size:1.25rem;font-weight:700">{cp.hazard_rate.toFixed(4)}</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Recovery Rate</div><div style="font-size:1.25rem;font-weight:700">{(cp.recovery_rate*100).toFixed(0)}%</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Collateral</div><div style="font-size:1.25rem;font-weight:700">{fmtNum(cp.collateral, 0)}</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">MPOR</div><div style="font-size:1.25rem;font-weight:700">{cp.mpor_days}d</div></div>
  </div>

  {#if summary}
    <div class="grid-4" style="margin-bottom:1rem">
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Total Runs</div><div style="font-size:1.25rem;font-weight:700">{summary.total_runs}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Avg CVA</div><div style="font-size:1.25rem;font-weight:700">{fmtNum(summary.avg_cva)}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Total Margin Called</div><div style="font-size:1.25rem;font-weight:700">{fmtNum(summary.total_margin_called, 0)}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Pending Calls</div><div style="font-size:1.25rem;font-weight:700;color:{summary.pending_calls>0?'var(--amber)':'var(--text)'}">{summary.pending_calls}</div></div>
    </div>
  {/if}

  <!-- Portfolios accordion -->
  <div class="card" style="margin-bottom:1rem">
    <div class="card-header">
      <span class="card-title">Portfolios ({(cp.portfolios ?? []).length})</span>
      <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
        <button class="btn btn-ghost btn-sm" on:click={() => { addingPortfolio = !addingPortfolio; }}>
          {addingPortfolio ? 'Cancel' : '+ Add Portfolio'}
        </button>
      </RoleGuard>
    </div>

    {#if addingPortfolio}
      <div style="padding:.75rem 0;border-top:1px solid var(--border)">
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">Collateral</label>
            <input class="form-input" type="number" bind:value={portForm.collateral} min="0" step="10000" placeholder="0" />
          </div>
          <div class="form-group" style="display:flex;align-items:center;gap:.5rem;padding-top:1.4rem">
            <input type="checkbox" id="auto_run_chk" bind:checked={portForm.auto_run} />
            <label for="auto_run_chk" class="form-label" style="margin:0">Auto-run on market refresh</label>
          </div>
        </div>
        <button class="btn btn-success btn-sm" on:click={submitPortfolio}>Create Portfolio</button>
      </div>
    {/if}

    {#each cp.portfolios ?? [] as port}
      <div class="portfolio-item">
        <div
          class="portfolio-header"
          on:click={() => expandedPortfolio = expandedPortfolio === port.id ? null : port.id}
          role="button" tabindex="0"
        >
          <span>{port.external_id}</span>
          <span class="text-muted text-sm">{(port.derivatives ?? []).length} derivatives · {port.auto_run ? '⟳ auto' : 'manual'}</span>
          <span style="color:var(--muted);font-size:.75rem">{expandedPortfolio === port.id ? '▲' : '▼'}</span>
        </div>
        {#if expandedPortfolio === port.id}
          <div style="padding:.75rem 0 .25rem">
            <div class="table-wrap">
              <table>
                <thead><tr><th>Type</th><th>Notional</th><th>Maturity</th><th>Underlying</th><th>Strike</th><th></th></tr></thead>
                <tbody>
                  {#each port.derivatives ?? [] as d}
                    <tr>
                      <td><span class="badge badge-blue">{d.deriv_type}</span></td>
                      <td>{fmtNum(d.notional, 0)}</td>
                      <td>{d.maturity_years}y</td>
                      <td>{d.underlying_price.toFixed(4)}</td>
                      <td>{d.strike.toFixed(4)}</td>
                      <td>
                        <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
                          <button class="btn btn-danger btn-sm" on:click={() => removeDerivative(port.id, d.id)}>✕</button>
                        </RoleGuard>
                      </td>
                    </tr>
                  {/each}
                  {#if (port.derivatives ?? []).length === 0}
                    <tr><td colspan="6" style="color:var(--muted);text-align:center">No derivatives yet.</td></tr>
                  {/if}
                </tbody>
              </table>
            </div>

            <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
              {#if addingDerivFor === port.id}
                <div style="margin-top:.75rem;padding:.75rem;background:var(--surface2);border-radius:6px">
                  <div class="form-row">
                    <div class="form-group">
                      <label class="form-label">Type</label>
                      <select class="form-select" bind:value={derivForm.deriv_type}>
                        {#each ['IRS','CDS','FX','EQUITY','COMMODITY'] as t}<option>{t}</option>{/each}
                      </select>
                    </div>
                    <div class="form-group">
                      <label class="form-label">Notional</label>
                      <input class="form-input" type="number" bind:value={derivForm.notional} min="0" step="100000" />
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label class="form-label">Maturity (years)</label>
                      <input class="form-input" type="number" bind:value={derivForm.maturity_years} min="0.1" step="0.5" />
                    </div>
                    <div class="form-group">
                      <label class="form-label">Cash Flow Freq / yr</label>
                      <input class="form-input" type="number" bind:value={derivForm.cash_flow_freq} min="1" step="1" />
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label class="form-label">Underlying Price</label>
                      <input class="form-input" type="number" bind:value={derivForm.underlying_price} step="0.001" />
                    </div>
                    <div class="form-group">
                      <label class="form-label">Strike</label>
                      <input class="form-input" type="number" bind:value={derivForm.strike} step="0.001" />
                    </div>
                  </div>
                  <div style="display:flex;gap:.5rem">
                    <button class="btn btn-success btn-sm" on:click={() => submitDerivative(port.id)}>Add</button>
                    <button class="btn btn-ghost btn-sm" on:click={() => addingDerivFor = null}>Cancel</button>
                  </div>
                </div>
              {:else}
                <button class="btn btn-ghost btn-sm" style="margin-top:.5rem"
                  on:click={() => { addingDerivFor = port.id; derivForm = { deriv_type: 'IRS', notional: 1_000_000, maturity_years: 5, underlying_price: 0.05, strike: 0.05, cash_flow_freq: 2 }; }}>
                  + Add Derivative
                </button>
              {/if}
            </RoleGuard>
          </div>
        {/if}
      </div>
    {/each}
    {#if (cp.portfolios ?? []).length === 0}
      <div style="color:var(--muted);font-size:.8rem;padding:.75rem 0">No portfolios.</div>
    {/if}
  </div>

  <!-- Simulation history + CVA chart -->
  <div class="grid-2" style="margin-bottom:1rem">
    <div class="card">
      <div class="card-header"><span class="card-title">Simulation History</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>CVA</th><th>Margin</th><th>Type</th></tr></thead>
          <tbody>
            {#each history.slice(0,8) as h}
              <tr>
                <td class="text-muted">{new Date(h.time).toLocaleDateString()}</td>
                <td>{fmtNum(h.cva)}</td>
                <td>{fmtNum(h.margin_required, 0)}</td>
                <td><span class="badge {h.is_stressed ? 'badge-amber' : 'badge-blue'}">{h.is_stressed ? 'Stress' : 'Base'}</span></td>
              </tr>
            {/each}
            {#if history.length === 0}
              <tr><td colspan="4" style="color:var(--muted);text-align:center">No runs yet</td></tr>
            {/if}
          </tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">CVA Trend</span></div>
      <CVABarChart {history} height={220} />
    </div>
  </div>

  <!-- Historical Backtesting -->
  {#if backtest && (backtest.realised.length > 0 || backtest.pfe_profile.length > 0)}
    <div class="card" style="margin-bottom:1rem">
      <div class="card-header">
        <span class="card-title">Historical Backtesting</span>
        <div style="display:flex;gap:.5rem;align-items:center">
          {#if backtest.breach_count > 0}
            <span class="badge badge-red">{backtest.breach_count} breach{backtest.breach_count !== 1 ? 'es' : ''}</span>
          {/if}
          <span class="badge {backtest.coverage_pct >= 95 ? 'badge-green' : backtest.coverage_pct >= 85 ? 'badge-amber' : 'badge-red'}">
            {backtest.coverage_pct.toFixed(1)}% coverage
          </span>
        </div>
      </div>
      {#if backtest.realised.length > 0}
        <BacktestChart
          pfeProfile={backtest.pfe_profile}
          realised={backtest.realised}
          coveragePct={backtest.coverage_pct}
          height={240}
        />
        <div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">
          Realised exposures are indicative mark-to-model estimates using GBM log-return pricing.
          Coverage = % of historical dates where realised exposure stayed within PFE band.
        </div>
      {:else}
        <div style="color:var(--muted);font-size:.8rem;padding:.75rem 0">
          No price history data available. Run market data refresh and simulate first.
        </div>
      {/if}
    </div>
  {/if}

  <!-- Survival Curve — shown when term structure is configured -->
  {#if cp.hz_1y || cp.hz_3y || cp.hz_5y || cp.hz_10y}
    <div class="card" style="margin-bottom:1rem">
      <div class="card-header">
        <span class="card-title">Credit Survival Curve</span>
        <span class="badge badge-blue">CDS Term Structure</span>
      </div>
      <SurvivalCurveChart
        hz_1y={cp.hz_1y}
        hz_3y={cp.hz_3y}
        hz_5y={cp.hz_5y}
        hz_10y={cp.hz_10y}
        flatRate={cp.hazard_rate}
        height={200}
      />
      <div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">
        S(t) = exp(−λ(t)·t) · Blue: term-structure curve · Amber dashed: flat hazard rate baseline
      </div>
    </div>
  {/if}

  <!-- Margin calls -->
  <div class="card">
    <div class="card-header"><span class="card-title">Margin Call History</span></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Status</th><th>Amount</th><th>Excess</th><th>Date</th><th>Reason</th></tr></thead>
        <tbody>
          {#each mcs as mc}
            <tr>
              <td><span class="badge {statusBadge[mc.status] ?? 'badge-muted'}">{mc.status}</span></td>
              <td>{fmtNum(mc.amount, 0)}</td>
              <td style="color:var(--red)">{fmtNum(mc.excess_exposure, 0)}</td>
              <td class="text-muted">{new Date(mc.issued_at).toLocaleDateString()}</td>
              <td class="reason-cell" style="font-size:.78rem;color:var(--muted)">
                <span class="reason-text">{mc.reason}</span>
                <span class="reason-tip">{mc.reason}</span>
              </td>
            </tr>
          {/each}
          {#if mcs.length === 0}
            <tr><td colspan="5" style="color:var(--muted);text-align:center">No margin calls</td></tr>
          {/if}
        </tbody>
      </table>
    </div>
  </div>
{/if}

<style>
  .portfolio-item   { border-top: 1px solid var(--border); }
  .portfolio-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: .65rem 0; cursor: pointer; font-size: .85rem; font-weight: 500;
  }
  .portfolio-header:hover { color: var(--text); }

  .reason-cell { position: relative; max-width: 220px; }
  .reason-text {
    display: block; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; cursor: default;
  }
  .reason-tip {
    display: none;
    position: absolute; left: 0; top: calc(100% + 4px); z-index: 200;
    background: var(--surface2, #1e293b); color: var(--text, #e2e8f0);
    border: 1px solid var(--border, #334155); border-radius: 5px;
    padding: .4rem .6rem; font-size: .78rem; line-height: 1.4;
    white-space: normal; min-width: 200px; max-width: 360px;
    box-shadow: 0 4px 16px rgba(0,0,0,.4); pointer-events: none;
  }
  .reason-cell:hover .reason-tip { display: block; }
</style>
