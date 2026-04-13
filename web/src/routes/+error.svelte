<script lang="ts">
  import { page } from '$app/stores';
</script>

<svelte:head><title>{$page.status} — CCR Engine</title></svelte:head>

<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;gap:1rem;text-align:center;">
  <div style="font-size:4rem;font-weight:800;color:var(--green);line-height:1">{$page.status}</div>
  <div style="font-size:1.1rem;font-weight:600;color:var(--text)">{$page.error?.message ?? 'Page not found'}</div>
  <div style="color:var(--muted);font-size:.85rem;max-width:360px">
    The page you're looking for doesn't exist or you don't have permission to view it.
  </div>
  <a href="/dashboard" class="btn btn-primary" style="margin-top:.5rem">← Back to Dashboard</a>
</div>
