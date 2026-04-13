<script lang="ts">
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import { currentUser, pendingMarginCallCount, authToken, lastApiLatencyMs, alertThresholds } from '$lib/state';
  import { initAuth, logout } from '$lib/auth';
  import { api } from '$lib/api';

  const PUBLIC_ROUTES = ['/login'];

  let loading = true;
  let lightMode = false;
  let showSettings = false;
  let thresholds = { cva: 0.05, margin: 0 };

  onMount(async () => {
    const path = $page.url.pathname;
    if (PUBLIC_ROUTES.includes(path)) { loading = false; return; }

    const ok = await initAuth();
    loading = false;  // always clear spinner regardless of auth result
    if (!ok) { goto('/login'); return; }

    // Initialise the API token on every page load.
    const token = $authToken;
    if (token) api.setToken(token);

    lightMode = localStorage.getItem('ccr_light_mode') === 'true';
    if (lightMode) document.body.classList.add('light');
    thresholds = { ...$alertThresholds };
  });

  // Handle post-login SPA navigation: when the user logs in from /login and
  // goto('/dashboard') fires, the layout onMount already ran (with loading=false
  // and no auth check). Re-initialize theme/thresholds if $currentUser is now set.
  $: if ($currentUser && !loading) {
    if (typeof localStorage !== 'undefined') {
      lightMode = localStorage.getItem('ccr_light_mode') === 'true';
      document.body.classList.toggle('light', lightMode);
    }
  }

  function toggleTheme() {
    lightMode = !lightMode;
    document.body.classList.toggle('light', lightMode);
    localStorage.setItem('ccr_light_mode', String(lightMode));
  }

  function saveThresholds() {
    alertThresholds.set({ ...thresholds });
    showSettings = false;
  }

  $: isPublic  = PUBLIC_ROUTES.includes($page.url.pathname);
  $: path      = $page.url.pathname;

  $: navClass = (href: string) =>
    path === href || path.startsWith(href + '/') ? 'nav-item active' : 'nav-item';
</script>

{#if loading && !isPublic}
  <div style="display:flex;align-items:center;justify-content:center;height:100vh;gap:.75rem;">
    <div class="spinner"></div>
    <span class="text-muted">Loading…</span>
  </div>
{:else if isPublic}
  <slot />
{:else}
  <div class="app-shell">
    <!-- ── Sidebar ─────────────────────────────────────────────────── -->
    <aside class="sidebar">
      <div class="sidebar-brand">
        <div>
          <div class="logo">CCR</div>
          <div class="sub">Credit Risk Engine</div>
        </div>
      </div>

      <nav class="nav-section">
        <div class="nav-label">Overview</div>
        <a href="/dashboard" class={navClass('/dashboard')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M2 2h5v5H2zm7 0h5v5H9zm-7 7h5v5H2zm7 0h5v5H9z"/>
          </svg>
          Dashboard
        </a>
      </nav>

      <nav class="nav-section">
        <div class="nav-label">Trading</div>
        <a href="/counterparties" class={navClass('/counterparties')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0M0 8.5a.5.5 0 0 1 .5-.5h15a.5.5 0 0 1 0 1H.5a.5.5 0 0 1-.5-.5"/>
          </svg>
          Counterparties
        </a>
        <a href="/simulate" class={navClass('/simulate')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>
          </svg>
          Simulate
        </a>
        <a href="/stress" class={navClass('/stress')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1a2.5 2.5 0 0 1 2.5 2.5V4h-5v-.5A2.5 2.5 0 0 1 8 1m3.5 3v-.5a3.5 3.5 0 1 0-7 0V4H1v10h14V4z"/>
          </svg>
          Stress Test
        </a>
      </nav>

      <nav class="nav-section">
        <div class="nav-label">Analytics</div>
        <a href="/query" class={navClass('/query')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M6 10.5a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3a.5.5 0 0 1-.5-.5m-2-3a.5.5 0 0 1 .5-.5h7a.5.5 0 0 1 0 1h-7a.5.5 0 0 1-.5-.5m-2-3a.5.5 0 0 1 .5-.5h11a.5.5 0 0 1 0 1h-11a.5.5 0 0 1-.5-.5"/>
          </svg>
          Query Builder
        </a>
        <a href="/presets" class={navClass('/presets')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M2 2a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2zm4.5 0h-1v1h1zm2 0h-1v1h1zm2 0h-1v1h1zM2 6v1h12V6zm0 2v1h12V8zm0 2v1h12v-1zm0 2v2a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1v-2z"/>
          </svg>
          Presets
        </a>
      </nav>

      <nav class="nav-section">
        <div class="nav-label">Risk</div>
        <a href="/margin-calls" class={navClass('/margin-calls')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5m.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2"/>
          </svg>
          Margin Calls
          {#if $pendingMarginCallCount > 0}
            <span class="badge badge-red" style="margin-left:auto">{$pendingMarginCallCount}</span>
          {/if}
        </a>
        <a href="/reports" class={navClass('/reports')}>
          <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
            <path d="M14 4.5V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2h5.5zm-3 0A1.5 1.5 0 0 1 9.5 3V1L14 4.5zM4.5 8a.5.5 0 0 0 0 1h7a.5.5 0 0 0 0-1zm0 2.5a.5.5 0 0 0 0 1h7a.5.5 0 0 0 0-1zm0 2.5a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1z"/>
          </svg>
          Reports
        </a>
      </nav>

      {#if $currentUser?.role === 'ADMIN' || $currentUser?.role === 'AUDITOR'}
        <nav class="nav-section">
          <div class="nav-label">System</div>
          {#if $currentUser?.role === 'ADMIN'}
            <a href="/admin" class={navClass('/admin')}>
              <svg class="nav-icon" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6m2-3a2 2 0 1 1-4 0 2 2 0 0 1 4 0m4 8c0 1-1 1-1 1H3s-1 0-1-1 1-4 6-4 6 3 6 4"/>
              </svg>
              Admin
            </a>
          {/if}
        </nav>
      {/if}

      <!-- Spacer + user footer -->
      <div style="flex:1"></div>
      <div style="padding:.75rem 1.25rem;border-top:1px solid var(--border);">
        <div style="font-size:.78rem;color:var(--text-2);margin-bottom:.4rem;">
          {$currentUser?.username ?? '—'}
          <span class="badge badge-blue" style="margin-left:.4rem">{$currentUser?.role ?? ''}</span>
        </div>
        <button class="btn btn-ghost btn-sm w-full" on:click={logout}>Sign out</button>
      </div>
    </aside>

    <!-- ── Header ──────────────────────────────────────────────────── -->
    <header class="header">
      <span style="font-size:.78rem;color:var(--muted);flex:1">
        Counterparty Credit Risk &amp; XVA Platform
      </span>
      {#if $lastApiLatencyMs !== null}
        <span class="badge badge-muted" style="font-variant-numeric:tabular-nums;margin-right:.5rem">API {$lastApiLatencyMs}ms</span>
      {/if}
      <span style="font-size:.75rem;color:var(--muted)">
        Demo Ticks — not real market data
      </span>
      <button class="btn btn-ghost btn-sm" style="font-size:.75rem" on:click={toggleTheme} title="Toggle light/dark mode">
        {lightMode ? '◑' : '●'}
      </button>
      <button class="btn btn-ghost btn-sm" style="font-size:.75rem" on:click={() => showSettings = true} title="Alert thresholds">
        ⚙
      </button>
    </header>

    <!-- ── Main ────────────────────────────────────────────────────── -->
    <main class="main">
      <slot />
    </main>
  </div>

  {#if showSettings}
    <div style="position:fixed;inset:0;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;z-index:2000">
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;width:320px">
        <div style="font-weight:600;margin-bottom:.75rem">Alert Thresholds</div>
        <div class="form-group" style="margin-bottom:.5rem">
          <label class="form-label" style="font-size:.75rem">CVA Breach Level</label>
          <input class="form-input" type="number" bind:value={thresholds.cva} step="0.01" min="0" />
        </div>
        <div class="form-group" style="margin-bottom:.75rem">
          <label class="form-label" style="font-size:.75rem">Margin Breach Level</label>
          <input class="form-input" type="number" bind:value={thresholds.margin} step="1000" min="0" />
        </div>
        <div style="display:flex;gap:.5rem;justify-content:flex-end">
          <button class="btn btn-ghost" on:click={() => showSettings = false}>Cancel</button>
          <button class="btn btn-primary" on:click={saveThresholds}>Save</button>
        </div>
      </div>
    </div>
  {/if}
{/if}
