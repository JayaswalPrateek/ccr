<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import RoleGuard from './RoleGuard.svelte';
  import type { MarginCall } from '$lib/types';
  import { fmtNum } from '$lib/fmt';

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

<tr class="mc-row">
  <td style="width:32px"><input type="checkbox" checked={selected} on:change={() => dispatch('select', mc.id)} /></td>
  <td>
    <span class="badge {statusClass[mc.status] ?? 'badge-muted'}">{mc.status}</span>
  </td>
  <td>{new Date(mc.issued_at).toLocaleString()}</td>
  <td style="color:{mc.status==='PENDING'&&ageD>5?'var(--amber)':'var(--muted)'}">
    {ageD}d{mc.status==='PENDING'&&ageD>5?' ⚠':''}</td>
  <td class="text-right"><strong>{fmtNum(mc.amount)}</strong></td>
  <td class="text-right" style="color:var(--red)">{fmtNum(mc.excess_exposure)}</td>
  <td class="reason-cell">
    <span class="reason-text">{mc.reason}</span>
    <span class="reason-tip">{mc.reason}</span>
  </td>
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

<style>
  .reason-cell {
    position: relative;
    max-width: 240px;
  }
  .reason-text {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: default;
  }
  .reason-tip {
    display: none;
    position: absolute;
    left: 0;
    top: calc(100% + 4px);
    z-index: 200;
    background: var(--surface2, #1e293b);
    color: var(--text, #e2e8f0);
    border: 1px solid var(--border, #334155);
    border-radius: 5px;
    padding: .4rem .6rem;
    font-size: .78rem;
    line-height: 1.4;
    white-space: normal;
    min-width: 200px;
    max-width: 360px;
    box-shadow: 0 4px 16px rgba(0,0,0,.4);
    pointer-events: none;
  }
  .reason-cell:hover .reason-tip {
    display: block;
  }
</style>
