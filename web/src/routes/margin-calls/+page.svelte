<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { marginCalls } from '$lib/state';
  import MarginCallRow from '$components/ui/MarginCallRow.svelte';
  import CVABarChart from '$components/charts/CVABarChart.svelte';
  import type { MarginCall, SimulationHistoryItem } from '$lib/types';

  let loading  = true;
  let error    = '';
  let filter: string = 'ALL';
  let history: SimulationHistoryItem[] = [];
  let selectedIds = new Set<string>();

  const FILTERS = ['ALL', 'PENDING', 'ACKNOWLEDGED', 'SETTLED', 'DISPUTED'];

  function toggleSelect(id: string) {
    const s = new Set(selectedIds);
    s.has(id) ? s.delete(id) : s.add(id);
    selectedIds = s;
  }

  function selectAll() { selectedIds = new Set(filtered.map(m => m.id)); }
  function clearAll()  { selectedIds = new Set(); }

  async function bulkAcknowledge() {
    try {
      const updates = await Promise.all([...selectedIds].map(id => api.acknowledgeMarginCall(id)));
      marginCalls.update((list) => list.map(m => updates.find(u => u.id === m.id) ?? m));
      selectedIds = new Set();
    } catch (err) { error = err instanceof Error ? err.message : 'Error'; }
  }

  async function bulkSettle() {
    try {
      const updates = await Promise.all([...selectedIds].map(id => api.settleMarginCall(id)));
      marginCalls.update((list) => list.map(m => updates.find(u => u.id === m.id) ?? m));
      selectedIds = new Set();
    } catch (err) { error = err instanceof Error ? err.message : 'Error'; }
  }

  onMount(async () => {
    try {
      const [mc, hist] = await Promise.all([
        api.listMarginCalls({ limit: 200 }),
        api.getSimHistory({ limit: 20 }),
      ]);
      marginCalls.set(mc);
      history = hist;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function acknowledge(e: CustomEvent<string>) {
    try {
      const updated = await api.acknowledgeMarginCall(e.detail);
      marginCalls.update((list) =>
        list.map((m) => (m.id === updated.id ? updated : m))
      );
    } catch (err) { error = err instanceof Error ? err.message : 'Error'; }
  }

  async function settle(e: CustomEvent<string>) {
    try {
      const updated = await api.settleMarginCall(e.detail);
      marginCalls.update((list) =>
        list.map((m) => (m.id === updated.id ? updated : m))
      );
    } catch (err) { error = err instanceof Error ? err.message : 'Error'; }
  }

  async function notify(e: CustomEvent<string>) {
    try {
      await api.notifyCounterparty(e.detail);
    } catch (err) { error = err instanceof Error ? err.message : 'Notification failed'; }
  }

  function downloadCSV() {
    api.downloadBlob('/api/v1/margin-calls/export/csv').then((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'margin-calls.csv';
      a.click();
    });
  }

  $: filtered = filter === 'ALL'
    ? $marginCalls
    : $marginCalls.filter((m) => m.status === filter);

  $: pending   = $marginCalls.filter((m) => m.status === 'PENDING').length;
  $: total     = $marginCalls.length;
</script>

<svelte:head><title>Margin Calls — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Margin Calls</div>
    <div class="page-sub">{total} total · {pending} pending</div>
  </div>
  <button class="btn btn-ghost btn-sm" on:click={downloadCSV}>Export CSV</button>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

<!-- Summary cards -->
<div class="grid-4" style="margin-bottom:1rem">
  {#each [['PENDING','badge-amber'],['ACKNOWLEDGED','badge-blue'],['SETTLED','badge-green'],['DISPUTED','badge-red']] as [s, cls]}
    <div class="card" style="padding:.75rem 1rem;cursor:pointer" on:click={() => filter = s} role="button" tabindex="0">
      <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em">{s}</div>
      <div style="font-size:1.6rem;font-weight:700;margin-top:.2rem">
        {$marginCalls.filter((m) => m.status === s).length}
      </div>
    </div>
  {/each}
</div>

<div class="grid-3" style="margin-bottom:1rem">
  <div class="card col-span-2">
    <!-- Filter bar -->
    <div style="display:flex;gap:.4rem;margin-bottom:.75rem;flex-wrap:wrap">
      {#each FILTERS as f}
        <button
          class="btn btn-sm"
          class:btn-primary={filter === f}
          class:btn-ghost={filter !== f}
          on:click={() => filter = f}
        >{f}</button>
      {/each}
    </div>

    {#if selectedIds.size > 0}
      <div style="display:flex;align-items:center;gap:.5rem;padding:.5rem .75rem;background:rgba(59,130,246,.08);border:1px solid rgba(59,130,246,.3);border-radius:var(--radius-sm);margin-bottom:.5rem">
        <span style="font-size:.8rem;color:var(--text-2)">{selectedIds.size} selected</span>
        <button class="btn btn-ghost btn-sm" on:click={bulkAcknowledge}>Acknowledge All</button>
        <button class="btn btn-success btn-sm" on:click={bulkSettle}>Settle All</button>
        <button class="btn btn-ghost btn-sm" style="margin-left:auto" on:click={clearAll}>Clear</button>
      </div>
    {/if}

    {#if loading}
      <div style="padding:2rem;text-align:center"><div class="spinner"></div></div>
    {:else if filtered.length === 0}
      <div style="color:var(--muted);padding:1.5rem;text-align:center">No margin calls for this filter.</div>
    {:else}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width:32px"><input type="checkbox" on:change={(e) => (e.target as HTMLInputElement).checked ? selectAll() : clearAll()} /></th>
              <th>Status</th>
              <th>Issued At</th>
              <th>Age</th>
              <th class="text-right">Amount</th>
              <th class="text-right">Excess</th>
              <th>Reason</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each filtered as mc}
              <MarginCallRow {mc} selected={selectedIds.has(mc.id)} on:acknowledge={acknowledge} on:settle={settle} on:notify={notify} on:select={e => toggleSelect(e.detail)} />
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

  <div class="card">
    <div class="card-header"><span class="card-title">CVA Exposure Trend</span></div>
    <CVABarChart {history} height={220} />
  </div>
</div>
