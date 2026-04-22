<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { api } from '$lib/api';
  import RoleGuard from '$components/ui/RoleGuard.svelte';
  import CVABarChart from '$components/charts/CVABarChart.svelte';
  import type { Counterparty, MarginCall, Portfolio, SimulationHistoryItem } from '$lib/types';

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

  async function addPortfolio() {
    if (!cp) return;
    const extId = prompt('Portfolio external ID:');
    if (!extId) return;
    try {
      const p = await api.createPortfolio({
        external_id: extId,
        counterparty_id: cp.id,
        collateral: 0,
        net_value: 0,
        auto_run: false,
      });
      cp = { ...cp, portfolios: [...(cp.portfolios ?? []), p] };
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function addDerivative(portfolioId: string) {
    try {
      const d = await api.addDerivative(portfolioId, {
        external_id: `DERIV-${Date.now()}`,
        deriv_type: 'IRS',
        notional: 1_000_000,
        maturity_years: 5,
        underlying_price: 0.05,
        strike: 0.05,
        cash_flow_freq: 2,
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
      <button class="btn btn-success" on:click={saveEdit}>Save</button>
    </div>
  {/if}

  <!-- Info cards -->
  <div class="grid-4" style="margin-bottom:1rem">
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Hazard Rate</div><div style="font-size:1.25rem;font-weight:700">{cp.hazard_rate.toFixed(4)}</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Recovery Rate</div><div style="font-size:1.25rem;font-weight:700">{(cp.recovery_rate*100).toFixed(0)}%</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">Collateral</div><div style="font-size:1.25rem;font-weight:700">{cp.collateral.toLocaleString(undefined,{maximumFractionDigits:0})}</div></div>
    <div class="card"><div style="font-size:.72rem;color:var(--muted)">MPOR</div><div style="font-size:1.25rem;font-weight:700">{cp.mpor_days}d</div></div>
  </div>

  {#if summary}
    <div class="grid-4" style="margin-bottom:1rem">
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Total Runs</div><div style="font-size:1.25rem;font-weight:700">{summary.total_runs}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Avg CVA</div><div style="font-size:1.25rem;font-weight:700">{summary.avg_cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Total Margin Called</div><div style="font-size:1.25rem;font-weight:700">{summary.total_margin_called.toLocaleString(undefined,{maximumFractionDigits:0})}</div></div>
      <div class="card"><div style="font-size:.7rem;color:var(--muted)">Pending Calls</div><div style="font-size:1.25rem;font-weight:700;color:{summary.pending_calls>0?'var(--amber)':'var(--text)'}">{summary.pending_calls}</div></div>
    </div>
  {/if}

  <!-- Portfolios accordion -->
  <div class="card" style="margin-bottom:1rem">
    <div class="card-header">
      <span class="card-title">Portfolios ({(cp.portfolios ?? []).length})</span>
      <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
        <button class="btn btn-ghost btn-sm" on:click={addPortfolio}>+ Add Portfolio</button>
      </RoleGuard>
    </div>
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
                      <td>{d.notional.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
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
                    <tr><td colspan="6" style="color:var(--muted);text-align:center">No derivatives</td></tr>
                  {/if}
                </tbody>
              </table>
            </div>
            <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
              <button class="btn btn-ghost btn-sm" style="margin-top:.5rem" on:click={() => addDerivative(port.id)}>+ Add Derivative</button>
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
                <td>{h.cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                <td>{h.margin_required.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
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
              <td>{mc.amount.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
              <td style="color:var(--red)">{mc.excess_exposure.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
              <td class="text-muted">{new Date(mc.issued_at).toLocaleDateString()}</td>
              <td style="font-size:.78rem;color:var(--muted)">{mc.reason.slice(0,60)}</td>
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
</style>
