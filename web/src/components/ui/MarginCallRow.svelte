<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import RoleGuard from './RoleGuard.svelte';
  import type { MarginCall } from '$lib/types';

  export let mc: MarginCall;
  export let selected: boolean = false;

  const dispatch = createEventDispatcher<{ acknowledge: string; settle: string; notify: string; select: string }>();

  const statusClass: Record<string, string> = {
    PENDING:      'badge-amber',
    ACKNOWLEDGED: 'badge-blue',
    SETTLED:      'badge-green',
    DISPUTED:     'badge-red',
  };

  $: ageD = Math.floor((Date.now() - new Date(mc.issued_at).getTime()) / 86400000);
</script>

<tr>
  <td style="width:32px"><input type="checkbox" checked={selected} on:change={() => dispatch('select', mc.id)} /></td>
  <td>
    <span class="badge {statusClass[mc.status] ?? 'badge-muted'}">{mc.status}</span>
  </td>
  <td>{new Date(mc.issued_at).toLocaleString()}</td>
  <td style="color:{mc.status==='PENDING'&&ageD>5?'var(--amber)':'var(--muted)'}">
    {ageD}d{mc.status==='PENDING'&&ageD>5?' ⚠':''}</td>
  <td class="text-right"><strong>{mc.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong></td>
  <td class="text-right" style="color:var(--red)">{mc.excess_exposure.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
  <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title={mc.reason}>{mc.reason}</td>
  <td>
    <RoleGuard roles={['ADMIN', 'RISK_MANAGER']}>
      <div style="display:flex;gap:.4rem">
        {#if mc.status === 'PENDING'}
          <button class="btn btn-ghost btn-sm" on:click={() => dispatch('acknowledge', mc.id)}>
            Acknowledge
          </button>
        {/if}
        {#if mc.status === 'ACKNOWLEDGED'}
          <button class="btn btn-success btn-sm" on:click={() => dispatch('settle', mc.id)}>
            Settle
          </button>
        {/if}
        {#if mc.status === 'PENDING' || mc.status === 'ACKNOWLEDGED'}
          <button class="btn btn-ghost btn-sm" on:click={() => dispatch('notify', mc.id)}>
            Notify
          </button>
        {/if}
      </div>
    </RoleGuard>
  </td>
</tr>
