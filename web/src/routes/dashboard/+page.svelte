<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { latestMetrics, marginCalls, pendingMarginCallCount } from '$lib/state';
  import MetricCard from '$components/ui/MetricCard.svelte';
  import PFEChart from '$components/charts/PFEChart.svelte';
  import EPEChart from '$components/charts/EPEChart.svelte';
  import CVABarChart from '$components/charts/CVABarChart.svelte';
  import { get } from 'svelte/store';
  import type { AuditLogItem, ConcentrationItem, SimulationHistoryItem } from '$lib/types';
  import { fmtNum } from '$lib/fmt';

  let history:       SimulationHistoryItem[] = [];
  let activityFeed:  AuditLogItem[]          = [];
  let concentration: ConcentrationItem[]     = [];
  let cpNameMap:     Record<string, string>  = {};
  let errorMsg       = '';
  let loading        = true;
  let autoRunning = false;
  let autoRunResults: any[] = [];

  onMount(async () => {
    try {
      [history] = await Promise.all([
        api.getSimHistory({ limit: 20 }),
        api.listMarginCalls({ limit: 50 }).then((mc) => marginCalls.set(mc)),
      ]);
      // Non-blocking: activity feed, concentration, and counterparty name map
      api.getMyActivity({ limit: 10 }).then((a) => { activityFeed = a; }).catch(() => {});
      api.getConcentration(10).then((c) => { concentration = c; }).catch(() => {});
      api.listCounterparties().then((cps) => {
        cpNameMap = Object.fromEntries(cps.map((cp) => [cp.id, cp.name]));
      }).catch(() => {});

      // Set latestMetrics from most recent base run if no in-session result.
      if (!get(latestMetrics) && history.length > 0) {
        const base    = history.find((h) => !h.is_stressed);
        const stressed= history.find((h) => h.is_stressed);
        if (base) {
          latestMetrics.set({
            base: {
              cva: base.cva, wwr_cva: base.wwr_cva,
              margin_required: base.margin_required,
              pfe_profile: base.pfe_profile, epe_profile: base.epe_profile,
              time_grid_years: base.time_grid_years,
              compute_time_us: base.compute_time_us,
              overflow_detected: false, arch_used: '', paths_used: 0,
            },
            stressed: stressed ? {
              cva: stressed.cva, wwr_cva: stressed.wwr_cva,
              margin_required: stressed.margin_required,
              pfe_profile: stressed.pfe_profile, epe_profile: stressed.epe_profile,
              time_grid_years: stressed.time_grid_years,
              compute_time_us: stressed.compute_time_us,
              overflow_detected: false, arch_used: '', paths_used: 0,
            } : undefined,
            success: true, error_msg: '',
          });
        }
      }
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : 'Failed to load dashboard';
    } finally {
      loading = false;
    }
  });

  async function runAutoSims() {
    autoRunning = true;
    try {
      autoRunResults = await api.triggerAutoRun();
      [history] = await Promise.all([api.getSimHistory({ limit: 20 })]);
      api.getConcentration(10).then(c => { concentration = c; }).catch(() => {});
    } catch (e) { errorMsg = e instanceof Error ? e.message : 'Auto-run failed'; }
    finally { autoRunning = false; }
  }

  $: base    = $latestMetrics?.base;
  $: stressed= $latestMetrics?.stressed;
  $: pendingMC = $pendingMarginCallCount;
</script>

<svelte:head><title>Dashboard — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Dashboard</div>
    <div class="page-sub">Latest risk metrics &amp; live market data</div>
  </div>
  <div style="display:flex;gap:.5rem;align-items:center">
    {#if pendingMC > 0}
      <a href="/margin-calls" class="btn btn-danger">
        {pendingMC} Pending Margin Call{pendingMC > 1 ? 's' : ''}
      </a>
    {/if}
    <button class="btn btn-ghost btn-sm" on:click={runAutoSims} disabled={autoRunning}>
      {#if autoRunning}<span class="spinner" style="width:12px;height:12px"></span>{/if}
      ⟳ Auto-Run
    </button>
  </div>
</div>

{#if autoRunResults.length > 0}
  <div class="alert alert-success" style="margin-bottom:.75rem">
    Auto-run complete: {autoRunResults.filter(r=>r.success).length}/{autoRunResults.length} succeeded.
    <button style="margin-left:auto;background:none;border:none;cursor:pointer;color:inherit" on:click={() => autoRunResults = []}>×</button>
  </div>
{/if}

{#if errorMsg}
  <div class="alert alert-error">{errorMsg}</div>
{/if}

{#if loading}
  <div style="display:flex;align-items:center;gap:.75rem;padding:2rem 0">
    <div class="spinner"></div> Loading…
  </div>
{:else}
  {#if history.filter(h => !h.is_stressed).length === 0}
    <div class="card" style="margin-bottom:1rem;border:1px solid rgba(59,130,246,.3);background:rgba(59,130,246,.04)">
      <div class="card-header">
        <span class="card-title">Getting Started</span>
        <span class="badge badge-blue">Setup Guide</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
        {#each [
          { n:'1', label:'Add Counterparty', desc:'Create your first counterparty with credit rating and risk parameters.', href:'/counterparties', cta:'Go to Counterparties' },
          { n:'2', label:'Build a Portfolio', desc:'Add portfolios and derivatives (IRS, CDS, FX swaps) to your counterparty.', href:'/counterparties', cta:'Manage Portfolios' },
          { n:'3', label:'Run a Simulation', desc:'Execute a Monte Carlo simulation to compute CVA, PFE and margin requirements.', href:'/simulate', cta:'Start Simulation' },
        ] as step}
          <div style="background:var(--surface2);border-radius:var(--radius-sm);padding:1rem;border:1px solid var(--border)">
            <div style="font-size:1.5rem;font-weight:700;color:var(--blue);margin-bottom:.4rem">{step.n}</div>
            <div style="font-weight:600;margin-bottom:.3rem">{step.label}</div>
            <div style="font-size:.78rem;color:var(--muted);margin-bottom:.75rem;line-height:1.5">{step.desc}</div>
            <a href={step.href} class="btn btn-primary btn-sm">{step.cta}</a>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- ── Row 1: KPI cards ─────────────────────────────────────────── -->
  <div class="grid-4" style="margin-bottom:1rem">
    <MetricCard
      label="CVA"
      value={base ? fmtNum(base.cva) : '—'}
      breached={!!base && base.cva > 0.05}
    />
    <MetricCard
      label="WWR-CVA"
      value={base ? fmtNum(base.wwr_cva) : '—'}
      subtitle="Wrong-way risk adjusted"
    />
    <MetricCard
      label="Margin Required"
      value={base ? fmtNum(base.margin_required, 0) : '—'}
      breached={!!base && base.margin_required > 0}
    />
    <MetricCard
      label="Compute Time"
      value={base ? (base.compute_time_us / 1000).toFixed(1) : '—'}
      unit="ms"
    />
  </div>

  <!-- ── Row 2: Charts ───────────────────────────────────────────────── -->
  <div class="grid-3" style="margin-bottom:1rem">
    <div class="card col-span-2">
      <div class="card-header">
        <span class="card-title">PFE Profile</span>
        {#if stressed}<span class="badge badge-amber">Stress overlay</span>{/if}
      </div>
      <PFEChart
        timeGrid={base?.time_grid_years ?? []}
        pfeBase={base?.pfe_profile ?? []}
        pfeStressed={stressed?.pfe_profile ?? []}
      />
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Top Risk</span>
        <span class="badge badge-muted">by CVA · latest run</span>
      </div>
      {#if concentration.length === 0}
        <div style="color:var(--muted);font-size:.8rem;padding:.5rem 0">No simulation data yet.</div>
      {:else}
        <div style="display:flex;flex-direction:column;gap:.5rem">
          {#each concentration.slice(0, 5) as item, i}
            <div style="display:flex;align-items:center;gap:.5rem">
              <span style="color:var(--muted);font-size:.72rem;width:1rem;flex-shrink:0">{i + 1}</span>
              <div style="flex:1;min-width:0">
                <div style="font-size:.82rem;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                  {item.counterparty_name ?? item.counterparty_id.slice(0, 8)}
                </div>
                <div style="font-size:.7rem;color:var(--muted)">{new Date(item.last_run_time).toLocaleDateString()}</div>
              </div>
              <div style="text-align:right;flex-shrink:0">
                <div style="font-size:.82rem;font-weight:600;color:{item.cva > 0.05 ? 'var(--red)' : 'var(--green)'}">
                  {fmtNum(item.cva)}
                </div>
                <div style="font-size:.68rem;color:var(--muted)">CVA</div>
              </div>
            </div>
          {/each}
        </div>
        <a href="/counterparties" class="btn btn-ghost btn-sm w-full" style="margin-top:.75rem;font-size:.74rem">View all counterparties</a>
      {/if}
    </div>
  </div>

  <!-- ── Row 3: EPE + history ─────────────────────────────────────────── -->
  <div class="grid-3" style="margin-bottom:1rem">
    <div class="card col-span-2">
      <div class="card-header">
        <span class="card-title">EPE Profile</span>
      </div>
      <EPEChart
        timeGrid={base?.time_grid_years ?? []}
        epeBase={base?.epe_profile ?? []}
        epeStressed={stressed?.epe_profile ?? []}
        cva={base?.cva ?? 0}
      />
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">CVA History</span>
      </div>
      <CVABarChart {history} height={200} />
    </div>
  </div>

  <!-- ── Row 4: Margin calls + recent runs + activity ───────────────── -->
  <div class="grid-3" style="margin-bottom:1rem">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Recent Changes</span>
      </div>
      {#if activityFeed.length === 0}
        <div style="color:var(--muted);font-size:.8rem;padding:.25rem 0">No recent activity.</div>
      {:else}
        <div style="display:flex;flex-direction:column;gap:.4rem">
          {#each activityFeed as item}
            <div style="display:flex;gap:.4rem;align-items:flex-start;font-size:.76rem">
              <span class="badge badge-blue" style="flex-shrink:0;white-space:nowrap">{item.action.replace(/_/g,' ')}</span>
              <span style="color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{item.resource_type} {item.resource_id?.slice(0,8) ?? ''}</span>
              <span style="margin-left:auto;color:var(--muted);white-space:nowrap">{new Date(item.time).toLocaleDateString()}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <!-- ── Old Row 4: Margin calls + recent runs ──────────────────────── -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Recent Margin Calls</span>
        <a href="/margin-calls" class="btn btn-ghost btn-sm">View all</a>
      </div>
      {#if $marginCalls.length === 0}
        <div style="color:var(--muted);font-size:.8rem;padding:.5rem 0">No margin calls.</div>
      {:else}
        <div class="table-wrap">
          <table>
            <thead><tr><th>Status</th><th>Amount</th><th>Counterparty</th><th>Date</th></tr></thead>
            <tbody>
              {#each $marginCalls.slice(0, 5) as mc}
                <tr>
                  <td>
                    <span class="badge {mc.status === 'PENDING' ? 'badge-amber' : mc.status === 'SETTLED' ? 'badge-green' : 'badge-blue'}">
                      {mc.status}
                    </span>
                  </td>
                  <td class="text-right">{fmtNum(mc.amount, 0)}</td>
                  <td class="text-muted">{cpNameMap[mc.counterparty_id] ?? mc.counterparty_id.slice(0,8) + '…'}</td>
                  <td class="text-muted">{new Date(mc.issued_at).toLocaleDateString()}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>

    <div class="card">
      <div class="card-header">
        <span class="card-title">Recent Simulations</span>
        <a href="/simulate" class="btn btn-primary btn-sm">New Run</a>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Date</th><th>CVA</th><th>Margin</th><th>Type</th></tr></thead>
          <tbody>
            {#each history.filter((h) => !h.is_stressed).slice(0, 6) as item}
              <tr>
                <td class="text-muted">{new Date(item.time).toLocaleDateString()}</td>
                <td>{fmtNum(item.cva)}</td>
                <td>{fmtNum(item.margin_required, 0)}</td>
                <td><span class="badge badge-blue">Base</span></td>
              </tr>
            {/each}
            {#if history.length === 0}
              <tr><td colspan="4" style="color:var(--muted);text-align:center;padding:1rem">No simulations yet</td></tr>
            {/if}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ── Row 5: Risk Concentration ─────────────────────────────────── -->
  <div class="card" style="margin-top:1rem">
    <div class="card-header">
      <span class="card-title">Risk Concentration</span>
      <span class="badge badge-muted">by counterparty · latest run</span>
    </div>
    {#if concentration.length === 0}
      <div style="color:var(--muted);font-size:.8rem">No simulation data yet.</div>
    {:else}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Counterparty</th>
              <th class="text-right">CVA</th>
              <th class="text-right">Margin Required</th>
              <th>Last Run</th>
            </tr>
          </thead>
          <tbody>
            {#each concentration as item, i}
              <tr>
                <td style="color:var(--muted)">{i + 1}</td>
                <td style="font-weight:500">{item.counterparty_name ?? item.counterparty_id.slice(0,8)}</td>
                <td class="text-right" style="color:{item.cva > 0.05 ? 'var(--red)' : 'var(--text)'}">
                  {fmtNum(item.cva)}
                </td>
                <td class="text-right">{fmtNum(item.margin_required, 0)}</td>
                <td class="text-muted">{new Date(item.last_run_time).toLocaleDateString()}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>
{/if}
