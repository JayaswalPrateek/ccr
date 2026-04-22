<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import type { SimulationHistoryItem } from '$lib/types';

  let history:  SimulationHistoryItem[] = [];
  let loading   = true;
  let error     = '';
  let selected: string | null = null;
  let downloading = false;

  onMount(async () => {
    try {
      history = await api.getSimHistory({ limit: 100 });
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function downloadPDF() {
    if (!selected) return;
    downloading = true;
    try {
      const blob = await api.downloadBlob(api.exportPDFUrl(selected));
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `ccr-report-${selected.slice(0, 8)}.pdf`;
      a.click();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Download failed';
    } finally {
      downloading = false;
    }
  }

  async function downloadCSV() {
    if (!selected) return;
    downloading = true;
    try {
      const blob = await api.downloadBlob(api.exportCSVUrl(selected));
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `ccr-profile-${selected.slice(0, 8)}.csv`;
      a.click();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Download failed';
    } finally {
      downloading = false;
    }
  }

  async function downloadMCCsv() {
    downloading = true;
    try {
      const blob = await api.downloadBlob('/api/v1/margin-calls/export/csv');
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'margin-calls.csv';
      a.click();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Download failed';
    } finally {
      downloading = false;
    }
  }

  $: baseRuns    = history.filter((h) => !h.is_stressed);
  $: selectedRun = baseRuns.find((h) => h.run_id === selected) ?? null;

  // ── Historical comparison ─────────────────────────────────────────────────
  let compareMode     = false;
  let compareIds:     string[] = [];
  let compareResults: SimulationHistoryItem[] = [];
  let comparing       = false;
  let compareError    = '';

  function toggleCompare(runId: string) {
    compareIds = compareIds.includes(runId)
      ? compareIds.filter((id) => id !== runId)
      : [...compareIds, runId].slice(-5);
    compareResults = [];
    compareError   = '';
  }

  async function runComparison() {
    if (compareIds.length < 2) return;
    comparing    = true;
    compareError = '';
    try {
      compareResults = await api.compareSimulations(compareIds);
    } catch (e) {
      compareError = e instanceof Error ? e.message : 'Comparison failed';
    } finally {
      comparing = false;
    }
  }
</script>

<svelte:head><title>Reports — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Reports &amp; Exports</div>
    <div class="page-sub">Download PDF reports and CSV data exports</div>
  </div>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

<div class="grid-2">
  <!-- Run selector -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Select Simulation Run</span>
      <button
        class="btn btn-sm"
        class:btn-primary={compareMode}
        class:btn-ghost={!compareMode}
        on:click={() => { compareMode = !compareMode; compareIds = []; compareResults = []; }}
      >Compare mode</button>
    </div>
    {#if loading}
      <div style="padding:1rem;text-align:center"><div class="spinner"></div></div>
    {:else if baseRuns.length === 0}
      <div style="color:var(--muted);font-size:.8rem">No simulation runs yet.</div>
    {:else}
      <div style="display:flex;flex-direction:column;gap:.3rem;max-height:420px;overflow-y:auto">
        {#each baseRuns as run}
          <div
            class="run-item"
            class:selected={compareMode ? (run.run_id !== null && compareIds.includes(run.run_id)) : selected === run.run_id}
            on:click={() => compareMode ? (run.run_id && toggleCompare(run.run_id)) : selected = run.run_id}
            role="option"
            aria-selected={compareMode ? (run.run_id !== null && compareIds.includes(run.run_id)) : selected === run.run_id}
            tabindex="0"
          >
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span style="font-size:.82rem;font-weight:500">{new Date(run.time).toLocaleString()}</span>
              <span class="badge badge-blue" style="font-size:.65rem">{run.run_id?.slice(0,8) ?? '—'}</span>
            </div>
            <div style="display:flex;gap:1rem;margin-top:.2rem;font-size:.75rem;color:var(--muted)">
              <span>CVA: {run.cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
              <span>Margin: {run.margin_required.toLocaleString(undefined,{maximumFractionDigits:0})}</span>
            </div>
            <div style="margin-top:.3rem">
              <a href="/simulate?rerun_id={run.run_id}" style="font-size:.68rem;color:var(--muted);text-decoration:underline">↩ re-run</a>
              {#if run.note}<span style="font-size:.68rem;color:var(--muted);margin-left:.5rem">· {run.note}</span>{/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Actions -->
  <div class="flex flex-col gap-3">
    <!-- Selected run export -->
    <div class="card">
      <div class="card-header"><span class="card-title">Simulation Report</span></div>
      {#if selectedRun}
        <div style="margin-bottom:1rem">
          <div style="font-size:.78rem;color:var(--muted);margin-bottom:.5rem">Selected run:</div>
          <div style="font-size:.85rem">{new Date(selectedRun.time).toLocaleString()}</div>
          <div class="grid-3" style="margin-top:.75rem">
            <div>
              <div style="font-size:.7rem;color:var(--muted)">CVA</div>
              <div style="font-weight:700">{selectedRun.cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            </div>
            <div>
              <div style="font-size:.7rem;color:var(--muted)">WWR-CVA</div>
              <div style="font-weight:700">{selectedRun.wwr_cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
            </div>
            <div>
              <div style="font-size:.7rem;color:var(--muted)">Margin</div>
              <div style="font-weight:700">{selectedRun.margin_required.toLocaleString(undefined,{maximumFractionDigits:0})}</div>
            </div>
          </div>
        </div>
        <div style="display:flex;gap:.5rem">
          <button class="btn btn-primary" on:click={downloadPDF} disabled={downloading}>
            {#if downloading}<span class="spinner" style="width:12px;height:12px"></span>{/if}
            Download PDF
          </button>
          <button class="btn btn-ghost" on:click={downloadCSV} disabled={downloading}>
            Download CSV
          </button>
        </div>
      {:else}
        <div style="color:var(--muted);font-size:.82rem">Select a run from the list.</div>
      {/if}
    </div>

    <!-- Bulk exports -->
    <div class="card">
      <div class="card-header"><span class="card-title">Bulk Exports</span></div>
      <div style="display:flex;flex-direction:column;gap:.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;border-bottom:1px solid var(--border)">
          <div>
            <div style="font-size:.85rem;font-weight:500">Margin Calls</div>
            <div style="font-size:.73rem;color:var(--muted)">All margin calls in CSV format</div>
          </div>
          <button class="btn btn-ghost btn-sm" on:click={downloadMCCsv}>Export CSV</button>
        </div>
      </div>
    </div>
  </div>
</div>

{#if compareMode}
  <div class="card" style="margin-top:1rem">
    <div class="card-header">
      <span class="card-title">Run Comparison ({compareIds.length} selected)</span>
      <button class="btn btn-primary btn-sm" on:click={runComparison} disabled={compareIds.length < 2 || comparing}>
        {#if comparing}<span class="spinner" style="width:12px;height:12px"></span>{/if}
        Compare
      </button>
    </div>
    {#if compareError}<div class="alert alert-error" style="margin-bottom:.75rem">{compareError}</div>{/if}
    {#if compareIds.length < 2}
      <div style="color:var(--muted);font-size:.82rem">Select at least 2 runs above to compare.</div>
    {:else if compareResults.length > 0}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run ID</th><th>Date</th><th class="text-right">CVA</th>
              <th class="text-right">WWR-CVA</th><th class="text-right">Margin</th><th>Type</th>
            </tr>
          </thead>
          <tbody>
            {#each compareResults as r}
              <tr>
                <td><span class="badge badge-blue">{r.run_id?.slice(0,8) ?? '—'}</span></td>
                <td class="text-muted">{new Date(r.time).toLocaleString()}</td>
                <td class="text-right">{r.cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                <td class="text-right">{r.wwr_cva.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                <td class="text-right">{r.margin_required.toLocaleString(undefined,{maximumFractionDigits:0})}</td>
                <td><span class="badge {r.is_stressed ? 'badge-amber' : 'badge-blue'}">{r.is_stressed ? 'Stressed' : 'Base'}</span></td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      {#if compareResults.length >= 2}
        <div style="margin-top:.5rem;font-size:.75rem;color:var(--muted)">
          CVA delta (last vs first):
          <span style="color:{compareResults[compareResults.length-1].cva > compareResults[0].cva ? 'var(--red)' : 'var(--green)'}">
            {(compareResults[compareResults.length-1].cva - compareResults[0].cva > 0 ? '+' : '')}{(compareResults[compareResults.length-1].cva - compareResults[0].cva).toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
        </div>
      {/if}
    {/if}
  </div>
{/if}

<style>
  .run-item {
    padding: .6rem .75rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: var(--transition);
  }
  .run-item:hover   { background: var(--surface2); border-color: var(--border2); }
  .run-item.selected{ background: rgba(59,130,246,.08); border-color: var(--blue); }
</style>
