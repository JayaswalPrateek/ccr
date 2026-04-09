<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '$lib/api';
  import { counterparties } from '$lib/state';
  import RoleGuard from '$components/ui/RoleGuard.svelte';
  import type { Counterparty } from '$lib/types';

  let loading  = true;
  let error    = '';
  let creating = false;
  let newForm  = emptyForm();
  let sparklines: Record<string, number[]> = {};

  function emptyForm() {
    return {
      external_id: '', name: '', credit_rating: 'BBB',
      hazard_rate: 0.02, recovery_rate: 0.40, collateral: 0,
      margin_threshold: 0, mpor_days: 10,
    };
  }

  const fetchSparklines = async (cps: Counterparty[]) => {
    const results = await Promise.allSettled(
      cps.map(cp => api.getSimHistory({ counterparty_id: cp.id, limit: 8 }))
    );
    const map: Record<string, number[]> = {};
    cps.forEach((cp, i) => {
      const r = results[i];
      if (r.status === 'fulfilled') {
        map[cp.id] = r.value.filter(h => !h.is_stressed).map(h => h.cva).reverse();
      }
    });
    sparklines = map;
  };

  onMount(async () => {
    try {
      const list = await api.listCounterparties();
      counterparties.set(list);
      fetchSparklines(list).catch(() => {});
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load';
    } finally {
      loading = false;
    }
  });

  async function create() {
    try {
      const cp = await api.createCounterparty(newForm);
      counterparties.update((list) => [...list, cp]);
      creating = false;
      newForm  = emptyForm();
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  async function remove(id: string) {
    if (!confirm('Delete this counterparty?')) return;
    try {
      await api.deleteCounterparty(id);
      counterparties.update((list) => list.filter((c) => c.id !== id));
    } catch (e) { error = e instanceof Error ? e.message : 'Error'; }
  }

  const ratingColors: Record<string, string> = {
    AAA: 'badge-green', AA: 'badge-green', A: 'badge-green',
    BBB: 'badge-blue', BB: 'badge-amber', B: 'badge-amber',
    CCC: 'badge-red', D: 'badge-red',
  };
</script>

<svelte:head><title>Counterparties — CCR Engine</title></svelte:head>

<div class="page-header">
  <div>
    <div class="page-title">Counterparties</div>
    <div class="page-sub">{$counterparties.length} counterparties</div>
  </div>
  <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
    <button class="btn btn-primary" on:click={() => creating = !creating}>
      {creating ? 'Cancel' : '+ New Counterparty'}
    </button>
  </RoleGuard>
</div>

{#if error}<div class="alert alert-error">{error}</div>{/if}

{#if creating}
  <div class="card" style="margin-bottom:1rem">
    <div class="card-header"><span class="card-title">New Counterparty</span></div>
    <div class="form-row">
      <div class="form-group"><label class="form-label">External ID</label><input class="form-input" bind:value={newForm.external_id} placeholder="CP-001" /></div>
      <div class="form-group"><label class="form-label">Name</label><input class="form-input" bind:value={newForm.name} placeholder="Acme Corp" /></div>
    </div>
    <div class="form-row">
      <div class="form-group">
        <label class="form-label">Credit Rating</label>
        <select class="form-select" bind:value={newForm.credit_rating}>
          {#each ['AAA','AA','A','BBB','BB','B','CCC','D'] as r}<option>{r}</option>{/each}
        </select>
      </div>
      <div class="form-group"><label class="form-label">Hazard Rate</label><input class="form-input" type="number" bind:value={newForm.hazard_rate} step="0.001" /></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label class="form-label">Recovery Rate</label><input class="form-input" type="number" bind:value={newForm.recovery_rate} min="0" max="1" step="0.05" /></div>
      <div class="form-group"><label class="form-label">Collateral</label><input class="form-input" type="number" bind:value={newForm.collateral} min="0" step="10000" /></div>
    </div>
    <div class="form-row">
      <div class="form-group"><label class="form-label">Margin Threshold</label><input class="form-input" type="number" bind:value={newForm.margin_threshold} min="0" /></div>
      <div class="form-group"><label class="form-label">MPOR Days</label><input class="form-input" type="number" bind:value={newForm.mpor_days} min="1" /></div>
    </div>
    <button class="btn btn-success" on:click={create}>Create</button>
  </div>
{/if}

{#if loading}
  <div style="padding:2rem;text-align:center"><div class="spinner"></div></div>
{:else if $counterparties.length === 0}
  <div class="card" style="padding:2rem;text-align:center;color:var(--muted)">
    No counterparties yet. Create one above.
  </div>
{:else}
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Name</th><th>Ext ID</th><th>Rating</th>
            <th>Hazard</th><th>Recovery</th><th>Collateral</th><th>MPOR</th>
            <th>CVA Trend</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each $counterparties as cp}
            <tr>
              <td>
                <a href="/counterparties/{cp.id}" style="color:var(--blue);font-weight:500">{cp.name}</a>
              </td>
              <td class="text-muted">{cp.external_id}</td>
              <td><span class="badge {ratingColors[cp.credit_rating] ?? 'badge-muted'}">{cp.credit_rating}</span></td>
              <td>{cp.hazard_rate.toFixed(4)}</td>
              <td>{(cp.recovery_rate * 100).toFixed(0)}%</td>
              <td>{cp.collateral.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td>{cp.mpor_days}d</td>
              <td>
                {#if sparklines[cp.id]?.length > 1}
                  {@const vals = sparklines[cp.id]}
                  {@const max = Math.max(...vals)}
                  {@const min = Math.min(...vals)}
                  {@const range = max - min || 0.001}
                  {@const pts = vals.map((v,i) => `${(i/(vals.length-1))*60},${12-((v-min)/range)*10}`).join(' ')}
                  <svg width="60" height="14" viewBox="0 0 60 14">
                    <polyline points={pts} fill="none" stroke={vals[vals.length-1] > vals[0] ? '#ff4d6a' : '#00d4aa'} stroke-width="1.5" stroke-linejoin="round"/>
                  </svg>
                {:else}
                  <span style="color:var(--muted);font-size:.7rem">—</span>
                {/if}
              </td>
              <td>
                <div style="display:flex;gap:.4rem">
                  <a href="/counterparties/{cp.id}" class="btn btn-ghost btn-sm">View</a>
                  <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
                    <button class="btn btn-danger btn-sm" on:click={() => remove(cp.id)}>Delete</button>
                  </RoleGuard>
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
{/if}
