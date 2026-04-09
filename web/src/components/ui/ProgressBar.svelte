<script lang="ts">
  export let value:   number  = 0;    // 0–100
  export let visible: boolean = false;
  export let label:   string  = '';
</script>

{#if visible}
  <div class="progress-wrap">
    <div class="progress-header">
      <span>{label || 'Running simulation…'}</span>
      <span>{value.toFixed(1)}%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" style="width:{value}%"></div>
    </div>
  </div>
{/if}

<style>
  .progress-wrap   { margin: .75rem 0; }
  .progress-header { display: flex; justify-content: space-between; font-size: .78rem; color: var(--text-2); margin-bottom: .35rem; }
  .progress-track  { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .progress-fill   {
    height: 100%;
    background: linear-gradient(90deg, var(--blue), var(--green));
    border-radius: 3px;
    transition: width .15s ease;
  }
</style>
